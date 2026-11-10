"""GAME notes for a vocal, with the aligned phonemes written as MIDI lyrics."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import mido

from phonem_transcriptor.game_lang import game_language_id, load_lang_map
from phonem_transcriptor.paths import game_model_dir, game_python

_TICKS_PER_BEAT = 480
_TEMPO = 500_000  # microseconds per quarter, 120 bpm
_WORKER = Path(__file__).with_name("game_worker.py")


def render_game(
    audio_path: str | Path,
    language: str | None,
    phonemes: list[dict],
    device: str,
    model_dir: str | Path | None = None,
    midi_path: str | Path | None = None,
) -> dict:
    model = Path(model_dir) if model_dir is not None else game_model_dir()
    _require_model(model)
    lang_map = load_lang_map(model)
    code = (language or "").strip().lower()
    # Only codes present in the checkpoint map are forwarded. Anything else stays
    # on the universal embedding inside transcribe_wav.
    language_arg = code if code in lang_map else ""
    notes = _transcribe(Path(audio_path), model, device, language_arg)
    attached = attach_phonemes(notes, phonemes)
    result = {
        "language_id": game_language_id(language, lang_map),
        "model": model.name,
        "notes": attached,
    }
    if midi_path is not None:
        destination = Path(midi_path)
        write_game_midi(destination, attached, phonemes)
        result["midi"] = str(destination)
    return result


def attach_phonemes(notes: list[dict], phonemes: list[dict]) -> list[dict]:
    buckets: list[list[dict]] = [[] for _ in notes]
    for phoneme in phonemes:
        best_index = None
        best_overlap = 0.0
        start = float(phoneme["start"])
        end = float(phoneme["end"])
        for index, note in enumerate(notes):
            overlap = _overlap(start, end, float(note["onset"]), float(note["offset"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = index
        if best_index is not None:
            buckets[best_index].append(phoneme)
    attached = []
    for note, group in zip(notes, buckets):
        words = list(dict.fromkeys(item["word"] for item in group if item.get("word")))
        attached.append(
            {
                "pitch": int(note["pitch"]),
                "onset": float(note["onset"]),
                "offset": float(note["offset"]),
                "velocity": int(note.get("velocity", 80)),
                "phonemes": [item["phoneme"] for item in group],
                "words": words,
            }
        )
    return attached


def write_game_midi(path: str | Path, notes: list[dict], phonemes: list[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    midi = mido.MidiFile(ticks_per_beat=_TICKS_PER_BEAT, charset="utf-8")
    timing = mido.MidiTrack()
    timing.append(mido.MetaMessage("set_tempo", tempo=_TEMPO, time=0))
    timing.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    lyrics = [
        (
            _ticks(float(phoneme["start"])),
            mido.MetaMessage("lyrics", text=str(phoneme["phoneme"])),
        )
        for phoneme in phonemes
        if phoneme.get("phoneme")
    ]
    timing.extend(_with_deltas(lyrics))
    timing.append(mido.MetaMessage("end_of_track", time=0 if not lyrics else 1))
    midi.tracks.append(timing)

    vocal = mido.MidiTrack()
    vocal.append(mido.MetaMessage("track_name", name="GAME vocal", time=0))
    vocal.append(mido.Message("program_change", program=0, channel=0, time=0))
    note_events = []
    for note in notes:
        start = _ticks(float(note["onset"]))
        end = max(_ticks(float(note["offset"])), start + 1)
        pitch = int(note["pitch"])
        velocity = int(note.get("velocity", 80))
        note_events.append((start, 1, mido.Message("note_on", note=pitch, velocity=velocity, channel=0)))
        note_events.append((end, 0, mido.Message("note_off", note=pitch, velocity=0, channel=0)))
    vocal.extend(_with_deltas(note_events))
    vocal.append(mido.MetaMessage("end_of_track", time=0 if not note_events else 1))
    midi.tracks.append(vocal)
    midi.save(str(destination))


def _transcribe(audio_path: Path, model: Path, device: str, language: str) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "notes.json"
        completed = subprocess.run(
            [
                str(game_python()),
                str(_WORKER),
                "--wav",
                str(audio_path),
                "--model",
                str(model),
                "--out",
                str(out),
                "--device",
                device,
                "--language",
                language,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(detail[-4000:] or f"GAME exited {completed.returncode}")
        return json.loads(out.read_text(encoding="utf-8"))


def _require_model(model: Path) -> None:
    missing = [name for name in ("model.pt", "config.yaml", "lang_map.json") if not (model / name).is_file()]
    if missing:
        raise FileNotFoundError(f"GAME model at {model} is missing {', '.join(missing)}")


def _overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _ticks(seconds: float) -> int:
    return int(round(max(0.0, seconds) * _TICKS_PER_BEAT * 1_000_000 / _TEMPO))


def _with_deltas(events: list[tuple]) -> list[mido.Message]:
    ordered = sorted(events, key=lambda item: item[:-1])
    messages = []
    last = 0
    for item in ordered:
        tick, message = item[0], item[-1]
        message.time = tick - last
        last = tick
        messages.append(message)
    return messages
