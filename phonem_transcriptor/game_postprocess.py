"""PitchNet-parity post-processing after GAME note detection.

Ported from SessionLoops/PitchNet (AGPL-3.0), ``EditorController``.

Mirrors ``EditorController`` after ``GAMEDetector::detectNotes``:

1. Drop rests / keep them only as attach guides
2. Drop notes shorter than 3 frames
3. Clip to F0/audio frame bounds
4. VAD + GAME-rest boundary refine (expand head/tail into consonants)

Pitch labels stay as GAME ``scores`` floats (no RMVPE reclassify).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# PitchNet frame grid: 44.1 kHz, hop 512.
SR = 44_100
HOP = 512

# PitchNet EditorController constants
_MIN_NOTE_FRAMES = 3
_VAD_RMS_THRESHOLD = 0.008
_MAX_HEAD_FRAMES = 14  # ~162 ms
_MAX_TAIL_FRAMES = 10  # ~116 ms
_REST_BRIDGE_GAP = 4
_MAX_REST_ATTACH = 18
_VAD_ATTACH_RATIO = 0.30


@dataclass(frozen=True)
class GameEvent:
    """One GAME event (voiced note or rest), frame-indexed @ hop 512 / 44.1 kHz."""

    start_frame: int
    end_frame: int
    midi_note: float
    is_rest: bool = False


def frames_to_seconds(frame: int, *, hop: int = HOP, sr: int = SR) -> float:
    return float(frame) * float(hop) / float(sr)


def seconds_to_frames(sec: float, *, hop: int = HOP, sr: int = SR) -> int:
    """PitchNet ``secondsToFrames`` — truncate toward zero."""
    return int(float(sec) * float(sr) / float(hop))


def compute_vad_mask(
    audio: np.ndarray,
    n_frames: int,
    *,
    hop: int = HOP,
    threshold: float = _VAD_RMS_THRESHOLD,
) -> np.ndarray:
    """Energy VAD on hop windows (PitchNet ``kVadThreshold = 0.008``)."""
    wav = np.asarray(audio, dtype=np.float32).reshape(-1)
    n = int(n_frames)
    out = np.zeros(n, dtype=bool)
    if n <= 0 or len(wav) == 0:
        return out
    thr = float(threshold)
    for i in range(n):
        s = i * hop
        if s >= len(wav):
            break
        e = min(s + hop, len(wav))
        seg = wav[s:e]
        rms = float(np.sqrt(np.mean(seg * seg))) if e > s else 0.0
        out[i] = rms > thr
    return out


def events_from_game_outputs(
    durations_sec: np.ndarray,
    presence: np.ndarray,
    scores: np.ndarray,
    *,
    hop: int = HOP,
    sr: int = SR,
    frame_offset: int = 0,
) -> list[GameEvent]:
    """Build frame events from GAME ``durations/presence/scores`` (batch item)."""
    d = np.asarray(durations_sec, dtype=np.float64).reshape(-1)
    p = np.asarray(presence, dtype=bool).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = min(len(d), len(p), len(s))
    events: list[GameEvent] = []
    t = 0.0
    spf = float(hop) / float(sr)
    for i in range(n):
        length = float(d[i])
        if length <= 0:
            continue
        t0, t1 = t, t + length
        t = t1
        i0 = frame_offset + int(round(t0 / spf))
        i1 = frame_offset + int(round(t1 / spf))
        if i1 <= i0:
            i1 = i0 + 1
        events.append(
            GameEvent(
                start_frame=i0,
                end_frame=i1,
                midi_note=float(s[i]),
                is_rest=not bool(p[i]),
            )
        )
    return events


def _vad_ratio(vad: np.ndarray, s: int, e: int) -> float:
    if e <= s or len(vad) == 0:
        return 0.0
    s = max(0, s)
    e = min(e, len(vad))
    if e <= s:
        return 0.0
    return float(np.count_nonzero(vad[s:e])) / float(e - s)


def refine_game_boundaries(
    events: list[GameEvent],
    vad_mask: np.ndarray | None,
    *,
    n_frames: int | None = None,
    min_frames: int = _MIN_NOTE_FRAMES,
    max_head: int = _MAX_HEAD_FRAMES,
    max_tail: int = _MAX_TAIL_FRAMES,
    rest_bridge: int = _REST_BRIDGE_GAP,
    max_rest_attach: int = _MAX_REST_ATTACH,
    vad_attach_ratio: float = _VAD_ATTACH_RATIO,
) -> list[GameEvent]:
    """PitchNet post-GAME: filter + optional VAD/rest boundary refine.

    Returns **voiced** notes only (rests consumed as attach guides).
    """
    if not events:
        return []

    f0_size = int(n_frames) if n_frames is not None else max(e.end_frame for e in events)
    f0_size = max(f0_size, 1)
    vad = (
        np.asarray(vad_mask, dtype=bool).reshape(-1)
        if vad_mask is not None
        else np.zeros(0, dtype=bool)
    )

    # Collect rests (guides) and seed voiced notes (min length + clip).
    rests: list[tuple[int, int]] = []
    notes: list[GameEvent] = []
    for ev in events:
        i0 = max(0, min(int(ev.start_frame), f0_size - 1))
        i1 = max(i0 + 1, min(int(ev.end_frame), f0_size))
        if ev.is_rest:
            if i1 > i0:
                rests.append((i0, i1))
            continue
        if i1 - i0 < int(min_frames):
            continue
        notes.append(
            GameEvent(
                start_frame=i0,
                end_frame=i1,
                midi_note=float(ev.midi_note),
                is_rest=False,
            )
        )

    if not notes or len(vad) == 0:
        return notes

    refined: list[GameEvent] = []
    for ni, note in enumerate(notes):
        start = note.start_frame
        end = note.end_frame
        prev_end = refined[-1].end_frame if refined else 0
        next_start = notes[ni + 1].start_frame if ni + 1 < len(notes) else len(vad)

        # 1) Plain VAD backward/forward expansion
        new_start = start
        lo = max(prev_end, start - max_head)
        for i in range(start - 1, lo - 1, -1):
            if 0 <= i < len(vad) and vad[i]:
                new_start = i
            else:
                break

        new_end = end
        hi = min(next_start, end + max_tail)
        for i in range(end, hi):
            if 0 <= i < len(vad) and vad[i]:
                new_end = i + 1
            else:
                break

        # 2) Rest-guided attach gated by VAD ratio
        for rs, re in rests:
            rest_len = re - rs
            if rest_len <= 0 or rest_len > max_rest_attach:
                continue
            if re <= start and start - re <= rest_bridge:
                cand = max(prev_end, rs)
                if cand < new_start and _vad_ratio(vad, cand, start) >= vad_attach_ratio:
                    new_start = max(cand, start - max_head)
            if rs >= end and rs - end <= rest_bridge:
                cand = min(next_start, re)
                if cand > new_end and _vad_ratio(vad, end, cand) >= vad_attach_ratio:
                    new_end = min(cand, end + max_tail)

        new_start = max(prev_end, new_start)
        new_end = max(new_start + 1, min(next_start, new_end))
        new_start = max(0, min(new_start, f0_size - 1))
        new_end = max(new_start + 1, min(new_end, f0_size))
        if new_end - new_start < int(min_frames):
            continue
        refined.append(
            GameEvent(
                start_frame=new_start,
                end_frame=new_end,
                midi_note=float(note.midi_note),
                is_rest=False,
            )
        )
    return refined


def postprocess_game_events(
    events: list[GameEvent],
    audio: np.ndarray | None = None,
    *,
    hop: int = HOP,
    sr: int = SR,
    n_frames: int | None = None,
) -> list[GameEvent]:
    """Full PitchNet post-GAME on raw events (+ optional audio for VAD)."""
    if n_frames is None:
        if events:
            n_frames = max(e.end_frame for e in events)
        elif audio is not None:
            n_frames = (len(np.asarray(audio).reshape(-1)) + hop - 1) // hop
        else:
            n_frames = 0
    vad = None
    if audio is not None and int(n_frames) > 0:
        vad = compute_vad_mask(audio, int(n_frames), hop=hop)
    return refine_game_boundaries(events, vad, n_frames=int(n_frames))


def events_to_note_tuples(
    events: list[GameEvent],
    *,
    hop: int = HOP,
    sr: int = SR,
) -> list[tuple[float, float, float]]:
    """``(onset_sec, offset_sec, midi_float)`` for voiced events."""
    out: list[tuple[float, float, float]] = []
    for ev in events:
        if ev.is_rest:
            continue
        o = frames_to_seconds(ev.start_frame, hop=hop, sr=sr)
        f = frames_to_seconds(ev.end_frame, hop=hop, sr=sr)
        if f <= o:
            continue
        out.append((o, f, float(ev.midi_note)))
    return out
