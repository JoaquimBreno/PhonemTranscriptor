import numpy as np
from pathlib import Path
from utils import read_wav, read_sr, resample, preprocess_midi, hz_to_midi, duration_secs_to_frames, extract_f0, expand_seq, align_by_dtw, correct_octave_error
import rle
import librosa
import matplotlib.pyplot as plt
from numba import njit
from joblib import Parallel, delayed

INF_COST = 1e9

@njit(cache=True, fastmath=True)
def _find_rightmost_silence(
    vocal: np.ndarray,
    left: int,
    right: int,
    back_limit: int = 4000
) -> tuple:
    """
    Find the right-most silence segment (start, end) ending at or before `right`.
    If the window starts with zeros, search leftward for the true segment start.
    Returns (-1, -1) if not found.
    """
    a = vocal[left:right+1]
    if (a == 0).sum() == 0:
        return -1, -1

    z = (a == 0).astype(np.int8)
    diff = np.diff(np.concatenate((np.zeros(1, np.int8), z, np.zeros(1, np.int8))))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    seg_start, seg_end = starts[-1], ends[-1]
    seg_start_abs_pos = left + seg_start
    seg_end_abs_pos = left + seg_end

    # Expand to true leftmost silence if window begins with silence
    if seg_start == 0:
        target_left = max(0, left - back_limit)
        while seg_start_abs_pos > target_left and vocal[seg_start_abs_pos - 1] == 0:
            seg_start_abs_pos -= 1
        if seg_start_abs_pos == 0 and vocal[0] == 0:
            return -1, -1

    if seg_start_abs_pos > 0 and vocal[seg_start_abs_pos-1] != 0:
        return seg_start_abs_pos, seg_end_abs_pos
    return -1, -1

@njit(cache=True, fastmath=True)
def _build_band_edges(dur, k, m, margin_frames=1000):
    """
    For each note (duration = dur[i] frames) build Sakoe-Chiba band:
    j ∈ [cumsum(i) - k·dur[i],  cumsum(i+1) + k·dur[i]], clipped to [0, m-1].
    Returns (N, 2) array of [j_min, j_max] per note.
    """
    cs = np.empty(dur.size + 1, np.int32)
    cs[0] = 0
    cs[1:] = np.cumsum(dur)
    half = np.where(k * dur < margin_frames, margin_frames, k * dur).astype(np.int32)
    j_min = np.maximum(0, cs[:-1] - half)
    j_max = np.minimum(m - 1, cs[1:] + half)
    out = np.empty((dur.size, 2), np.int32)
    out[:, 0] = j_min
    out[:, 1] = j_max
    return out

@njit(cache=True, fastmath=True)
def _pitch_seg_cost(note_val, seg):
    """
    Mean(|note_val - seg|), clipped at 2 semitones (6 MIDI cents).
    """
    raw_diff = np.abs(seg - note_val)
    diff = np.minimum(raw_diff, 6)
    return diff.sum()

@njit(cache=True, fastmath=True)
def _dtw_core(vocal, note, dur, lam_pitch, lam_dur, lam_trans, band):
    """
    Band-limited DTW kernel.
    """
    n, m = note.shape[0], vocal.shape[0]
    INF = 1e9
    cost = np.full((n, m), INF, np.float32)
    ptr = np.zeros((n, m, 2), np.int16)

    cost[0, 0] = abs(vocal[0] - note[0])

    for i in range(n):
        j_start, j_end = band[i, 0], band[i, 1]
        for j in range(j_start, j_end + 1):
            if i == 0 and j == 0:
                continue
            max_v = int(dur[i] * 3)
            max_v = min(max_v, j)
            if i == 0 and j - max_v < 0:
                max_v = j
            elif i > 0 and j - max_v <= 0:
                max_v = j - 1
            if max_v <= 0:
                continue

            best = INF
            bu = bv = 0
            u_max = 0 if i == 0 else 1
            win_left = j - max_v
            win_right = j - 1

            if vocal[j] > 0:
                s0, s1 = _find_rightmost_silence(vocal, win_left, win_right)
                unvoiced_cost = 0
            else:
                s0, s1 = -1, -1
                unvoiced_cost = 20

            if s0 != -1:
                max_v = j - s0 + 1

            for u in range(u_max + 1):
                for v in range(1, max_v + 1):
                    prev = cost[i - u, j - v]
                    if prev >= INF:
                        continue
                    if s0 != -1 and u == 0 and note[i-u] != 0 and j - v <= s1:
                        continue
                    dur_cost = abs(dur[i] - np.sum(vocal[j - v + 1 : j + 1] != 0))
                    seg = vocal[j - v + 1 : j + 1]
                    p_cost = _pitch_seg_cost(note[i], seg)
                    tot = prev + lam_pitch * p_cost + lam_dur * dur_cost + unvoiced_cost
                    if u == 1:
                        tot += lam_trans
                    if tot < best:
                        best = tot
                        bu = u
                        bv = v
            cost[i, j] = best
            ptr[i, j, 0] = bu
            ptr[i, j, 1] = bv
    return cost, ptr

class VocalNoteAlign:
    """
    Class for aligning vocal pitch sequences to note pitch using band-limited DTW.
    """
    def __init__(self, lambda_pitch=2, lambda_duration=0.1, lambda_unvoiced=20, lambda_transition=1e-3,
                 sampling_rate=24000, hop_length=240):
        """
        Args:
            lambda_pitch (float): Weight for pitch difference.
            lambda_duration (float): Weight for duration difference.
            lambda_unvoiced (float): Penalty for mismatched silence.
        """
        self.lambda_pitch = lambda_pitch
        self.lambda_duration = lambda_duration
        self.lambda_unvoiced = lambda_unvoiced
        self.lambda_transition = lambda_transition
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.frame_per_sec = sampling_rate / hop_length
        self._log_context: str | None = None

    def _log(self, level: str, message: str, *, segment: int | None = None) -> None:
        label = self._log_context or "align"
        segment_label = f" seg={segment}" if segment is not None else ""
        print(f"[{level}] {label}{segment_label} | {message}")

    def __call__(self, wav_path, midi_path):
        return self.multi_process_run(wav_path, midi_path)

    def run(self, vocal_pitch, note_pitch, note_duration, *, segment_idx: int | None = None):
        """
        Main alignment pipeline: handles silence, DTW, and returns aligned note sequence.
        """
        cost_matrix, pointer_matrix = self.compute_dtw_band(vocal_pitch, note_pitch, note_duration)
        aligned_note_pitch, _note_pitch, _note_duration = self.align(
            vocal_pitch, note_pitch, cost_matrix, pointer_matrix, segment_idx=segment_idx
        )
        aligned_note_pitch[vocal_pitch == 0] = 0
        unfilled = int(np.sum(aligned_note_pitch == -1))
        if unfilled > 0:
            self._log(
                "WARN",
                f"DTW backtrack incomplete ({unfilled}/{len(vocal_pitch)} frames unfilled); "
                "falling back to vanilla DTW",
                segment=segment_idx,
            )
            aligned_note_pitch = align_by_dtw(
                vocal_pitch, expand_seq(note_pitch, note_duration)
            )
            aligned_note_pitch[vocal_pitch == 0] = 0
            self._log("INFO", "vanilla DTW fallback applied", segment=segment_idx)
        return aligned_note_pitch, _note_pitch, _note_duration, pointer_matrix, cost_matrix

    def _align_segments_parallel(self, segments, align_segment, n_jobs: int, parallel_backend: str):
        if len(segments) < 2 or n_jobs <= 1 or parallel_backend == "sequential":
            return [align_segment(seg) for seg in segments]
        return Parallel(n_jobs=min(n_jobs, len(segments)), backend=parallel_backend)(
            delayed(align_segment)(seg) for seg in segments
        )

    def multi_process_run(
        self,
        wav_path,
        midi_path,
        n_jobs=8,
        log_context: str | None = None,
        parallel_backend: str = "sequential",
        f0: np.ndarray | None = None,
    ):
        """
        Main entry for vocal-MIDI alignment. Handles IO, segmentation, and parallel alignment.
        """
        def extract_note_segment(notes, start_sec, end_sec):
            """
            Extract pitches and durations for notes within a given segment (in seconds).
            Trim to start_sec and end_sec, remove unvoiced notes at ends.
            """
            pitch_segment, duration_segment = [], []
            first_note_handled = False
            for note in notes:
                if note.end < start_sec or note.start > end_sec:
                    continue
                if not first_note_handled:
                    note.start = start_sec
                    first_note_handled = True
                pitch_segment.append(note.pitch)
                duration_segment.append(note.end - note.start)
                last_start, last_end = note.start, end_sec
            if duration_segment:
                duration_segment[-1] = last_end - last_start
            pitch_segment = np.array(pitch_segment)
            duration_segment = np.array(duration_segment)
            if pitch_segment.size > 1:
                if pitch_segment[0] == 0:
                    pitch_segment = pitch_segment[1:]
                    duration_segment = duration_segment[1:]
                if pitch_segment[-1] == 0:
                    pitch_segment = pitch_segment[:-1]
                    duration_segment = duration_segment[:-1]
            mask = (pitch_segment != 0)
            return pitch_segment[mask], duration_segment[mask]

        def split_segments(unvoiced_intervals, f0, notes):
            """
            Split the audio into segments based on unvoiced intervals.
            Returns a list of (idx, vocal_pitch, note_pitch, note_duration, start_frame, end_frame).
            """
            segments = []
            for idx in range(len(unvoiced_intervals) - 1):
                start_frame = unvoiced_intervals[idx][1]
                end_frame = unvoiced_intervals[idx + 1][0]
                start_sec = start_frame / self.frame_per_sec
                end_sec = end_frame / self.frame_per_sec
                note_pitch, note_duration = extract_note_segment(notes, start_sec, end_sec)
                note_duration = duration_secs_to_frames(note_duration, self.sampling_rate, self.hop_length)
                vocal_pitch = hz_to_midi(f0[start_frame:end_frame])
                segments.append((idx, vocal_pitch, note_pitch, note_duration, start_frame, end_frame))
            return segments

        def align_segment(segment):
            """
            Align a single segment using DTW and postprocess the aligned notes.
            """
            idx, vocal_pitch, note_pitch, note_duration, start_frame, end_frame = segment
            start_sec = start_frame / self.frame_per_sec
            end_sec = end_frame / self.frame_per_sec
            if len(note_pitch) == 0:
                self._log(
                    "WARN",
                    f"segment skipped: no pitched notes "
                    f"(frames {start_frame}-{end_frame}, {start_sec:.2f}-{end_sec:.2f}s)",
                    segment=idx,
                )
                empty = np.array([], dtype=np.float32)
                return empty, empty, empty, vocal_pitch, None, None
            dur_sum = int(np.sum(note_duration))
            self._log(
                "INFO",
                f"{len(note_pitch)} notes, {len(vocal_pitch)} vocal frames "
                f"(note dur sum={dur_sum}, delta={dur_sum - len(vocal_pitch):+d}) "
                f"[{start_sec:.2f}-{end_sec:.2f}s]",
                segment=idx,
            )
            frame_pitch, src_pitch, src_duration, pointer_matrix, cost_matrix = self.run(
                vocal_pitch, note_pitch, note_duration, segment_idx=idx
            )
            src_duration = np.asarray(src_duration)
            # Keep each GAME note, including same-pitch splits that mark syllables.
            # RLE would collapse those into one note. Use it only if backtrack failed.
            backtrack_ok = (
                src_duration.size == len(note_pitch)
                and int(src_duration.sum()) == len(vocal_pitch)
                and not np.any(np.asarray(frame_pitch) < 0)
            )
            if backtrack_ok:
                keep = src_duration > 0
                aligned_note_pitch = np.asarray(src_pitch, dtype=np.float64)[keep]
                aligned_note_duration = src_duration[keep].astype(np.int32, copy=False)
            else:
                aligned_note_pitch, aligned_note_duration = [
                    np.array(x) for x in rle.encode(frame_pitch)
                ]
            aligned_note_pitch, aligned_note_duration = self.insert_unvoiced_segments(
                aligned_note_pitch, aligned_note_duration, vocal_pitch
            )
            return aligned_note_pitch, aligned_note_duration, frame_pitch, vocal_pitch, pointer_matrix, cost_matrix

        def merge_segments(segments, results, total_frames):
            """
            Merge aligned note sequences for each segment into a global note sequence.
            """
            note_pitch, note_duration = [], []
            curr_seg_end = 0
            for i in range(len(segments)):
                curr_seg_start, curr_seg_end = segments[i][4], segments[i][5]
                if i == 0:
                    prev_seg_end = 0
                else:
                    prev_seg_end = segments[i-1][5]
                # Fill unvoiced gap if any
                gap = curr_seg_start - prev_seg_end
                if gap > 0:
                    note_pitch.append(0)
                    note_duration.append(gap)
                note_pitch.extend(results[i][0])
                note_duration.extend(results[i][1])
            # Tail silence
            if curr_seg_end < total_frames:
                note_pitch.append(0)
                note_duration.append(total_frames - curr_seg_end)
            return note_pitch, note_duration

        # ---- I/O and preprocessing ----
        self._log_context = log_context or f"{Path(wav_path).stem} <- {Path(midi_path).name}"
        self._log("INFO", f"start wav={wav_path} midi={midi_path}")

        if f0 is None:
            wav, _ = librosa.load(wav_path, sr=self.sampling_rate)
            f0 = extract_f0(wav, self.sampling_rate, self.hop_length, medfilt_kernel_size=5)
        else:
            f0 = np.asarray(f0, dtype=np.float64).reshape(-1)
            if f0.size == 0:
                raise ValueError(f"empty precomputed f0 for {wav_path}")

        vs, ds = rle.encode(f0)
        cum = np.pad(np.cumsum(ds), (1, 0))
        intervals = [(cum[i], cum[i + 1]) for i in range(len(cum) - 1)]
        unvoiced_duration_threshold = 1 # sec
        unvoiced_intervals = [
            (start, end) for v, (start, end) in zip(vs, intervals)
            if v == 0 and (end - start) > unvoiced_duration_threshold * self.frame_per_sec
        ]
        T = len(f0)
        # Pad for edge cases (start/end)
        if not unvoiced_intervals:
            unvoiced_intervals = [(0, 0), (T, T)]
        else:
            if unvoiced_intervals[0][0] != 0:
                unvoiced_intervals = [(0, 0)] + unvoiced_intervals
            if unvoiced_intervals[-1][1] != T:
                unvoiced_intervals = unvoiced_intervals + [(T, T)]
        notes = preprocess_midi(
            midi_path,
            start_sec=unvoiced_intervals[0][1] / self.frame_per_sec,
            end_sec=unvoiced_intervals[-1][0] / self.frame_per_sec
        )
        if not notes:
            self._log("WARN", "GAME MIDI has no pitched notes; writing all-unvoiced alignment")
            return np.array([0], dtype=np.float32), np.array([f0.shape[0]], dtype=np.int32), []
        trimmed_f0 = f0[unvoiced_intervals[0][1]:unvoiced_intervals[-1][0]]
        octave_log_prefix = f"{self._log_context} | " if self._log_context else ""
        notes, best_shift = correct_octave_error(
            notes,
            trimmed_f0,
            sampling_rate=self.sampling_rate,
            hop_length=self.hop_length,
            log_prefix=octave_log_prefix,
        )
        self._log("INFO", f"octave correction: global shift={best_shift:+d} semitones")
        segments = split_segments(unvoiced_intervals, f0, notes)
        self._log("INFO", f"{len(segments)} voiced segment(s) to align")

        results = self._align_segments_parallel(segments, align_segment, n_jobs, parallel_backend)
        aligned_note_pitch, aligned_note_duration = merge_segments(segments, results, total_frames=f0.shape[0])
        return np.array(aligned_note_pitch), np.array(aligned_note_duration), results

    def compute_dtw_band(self, vocal_pitch, note_pitch, note_duration, band_width_multiply=40):
        """
        Run band-limited DTW using Numba JIT kernels.
        """
        vocal = np.asarray(vocal_pitch, dtype=np.float32)
        note = np.asarray(note_pitch, dtype=np.float32)
        dur = np.asarray(note_duration, dtype=np.int32).ravel()
        n, m = note.shape[0], vocal.shape[0]
        band = _build_band_edges(dur, band_width_multiply, m)
        cost, ptr = _dtw_core(
            vocal, note, dur,
            self.lambda_pitch,
            self.lambda_duration,
            self.lambda_transition,
            band
        )
        return cost, ptr

    def align(self, vocal_pitch, note_pitch, cost_matrix, pointer_matrix, *, segment_idx: int | None = None):
        """
        Backtrack through pointer_matrix to produce frame-level aligned note sequence.
        Returns aligned_note_pitch, note_pitch, note_duration.
        """
        n, m = len(note_pitch), len(vocal_pitch)
        end_cost = float(cost_matrix[n - 1, m - 1]) if n > 0 and m > 0 else INF_COST
        if end_cost >= INF_COST:
            self._log(
                "WARN",
                f"DTW has no valid path to (note={n - 1}, frame={m - 1}); "
                f"end_cost=inf, notes={n}, frames={m}",
                segment=segment_idx,
            )

        aligned_note_pitch = np.full(m, -1.0)
        note_duration = np.zeros([n], dtype=np.int32)
        curr_i, curr_j = n - 1, m - 1
        steps = 0
        max_steps = n * m  # Safety bound for infinite loop
        stuck_at: tuple[int, int] | None = None
        while steps < max_steps:
            u, v = int(pointer_matrix[curr_i][curr_j][0]), int(pointer_matrix[curr_i][curr_j][1])
            if u == 0 and v == 0 and (curr_i, curr_j) != (0, 0):
                stuck_at = (curr_i, curr_j)
                break
            aligned_note_pitch[curr_j - v + 1: curr_j + 1] = note_pitch[curr_i]
            note_duration[curr_i] += v
            curr_i -= u
            curr_j -= v
            steps += 1
            if curr_i == 0 and curr_j == 0:
                aligned_note_pitch[0] = note_pitch[0]
                note_duration[0] += 1
                break
        else:
            stuck_at = (curr_i, curr_j)

        if stuck_at is not None:
            unfilled = int(np.sum(aligned_note_pitch == -1))
            self._log(
                "WARN",
                f"DTW backtrack failed at note_idx={stuck_at[0]} frame_idx={stuck_at[1]} "
                f"(steps={steps}/{max_steps}, notes={n}, frames={m}, unfilled={unfilled})",
                segment=segment_idx,
            )
        return aligned_note_pitch, note_pitch, note_duration

    def insert_unvoiced_segments(
    self,
    note_pitch,
    note_duration,
    vocal_pitch,
    ):
        """
        Align note_pitch/note_duration sequence with frame-level vocal_pitch
        by inserting note=0 (unvoiced) for unvoiced regions.
    
        Parameters
        ----------
        note_pitch : Sequence[int or float]
            Note pitch sequence (0 = unvoiced).
        note_duration : Sequence[int]
            Note durations (in frames), same length as note_pitch.
        vocal_pitch : Sequence[int or float]
            Frame-level vocal pitch. Should have length sum(note_duration).
    
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (New note_pitch, new note_duration) with unvoiced notes correctly inserted and short notes merged.
        """
        new_pitch, new_dur = [], []
        frame_ptr = 0  # current frame position in vocal_pitch
    
        for pitch, dur in zip(note_pitch, note_duration):
            segment = vocal_pitch[frame_ptr : frame_ptr + dur]
            frame_ptr += dur
    
            if pitch == 0:
                # Already unvoiced: preserve as is
                new_pitch.append(0)
                new_dur.append(dur)
                continue
    
            # Find change-points between voiced/unvoiced using numpy diff (vectorized)
            mask = (segment != 0)
            # Get indices where voiced/unvoiced changes, plus [0] and [len]
            if len(mask) == 0:
                continue
            change_idxs = np.flatnonzero(np.diff(mask.astype(int), prepend=mask[0]))  # change points
            # Build block boundaries
            block_starts = np.r_[0, change_idxs]
            block_ends = np.r_[change_idxs, len(mask)]
            for start, end in zip(block_starts, block_ends):
                is_voiced = mask[start]
                length = end - start
                if length <= 0:
                    continue
                new_pitch.append(pitch if is_voiced else 0)
                new_dur.append(length)
    
        # Step 2: Remove/merge very short voiced notes (<=2 frames) into previous note, unless at sequence start
        final_pitch, final_dur = [], []
        duration_buffer = 0
        for p, d in zip(new_pitch, new_dur):
            if duration_buffer > 0:
                d += duration_buffer
                duration_buffer = 0
            if d <= 2 and p != 0:
                # If this is the very first note, keep as is
                if not final_dur:
                    final_pitch.append(p)
                    final_dur.append(d)
                else:
                    if final_pitch[-1] != 0:
                        final_dur[-1] += d  # Merge into previous note
                    else:
                        duration_buffer += d  # Merge into next note (if previous was unvoiced)
            else:
                final_pitch.append(p)
                final_dur.append(d)
        # Handle remaining duration_buffer at the end (rare)
        if duration_buffer > 0 and final_dur:
            final_dur[-1] += duration_buffer
    
        return np.array(final_pitch), np.array(final_dur)