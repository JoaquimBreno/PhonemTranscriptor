import librosa
import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
from aligner import VocalNoteAlign, extract_f0, hz_to_midi, read_wav, expand_seq, resample, read_sr

wav_path = "/home/vocod/GAME/0006.wav"
midi_path = "/home/vocod/GAME/0006.mid"

# Create the aligner object
_aligner = VocalNoteAlign(
    lambda_pitch=2,       # Weight for pitch difference
    lambda_duration=0.1,  # Weight for duration mismatch
    lambda_unvoiced=20,   # Penalty for unvoiced/silence mismatch
    sampling_rate=24000,
    hop_length=240
)

# Perform alignment
note_pitch, note_duration, *_ = _aligner(wav_path, midi_path)
aligned_note_pitch = expand_seq(note_pitch, note_duration)

# Visualize
wav, _ = librosa.load(wav_path, sr=24000)
vocal_pitch = hz_to_midi(extract_f0(wav, 24000, 240))

midi_data = pretty_midi.PrettyMIDI(midi_path)
unaligned_note_pitch = np.zeros(len(vocal_pitch))

for instrument in midi_data.instruments:
    if not instrument.is_drum:
        for note in instrument.notes:
            start_idx = int(note.start * 24000 / 240)
            end_idx = int(note.end * 24000 / 240)
            
            start_idx = max(0, min(start_idx, len(vocal_pitch) - 1))
            end_idx = max(0, min(end_idx, len(vocal_pitch)))
            
            unaligned_note_pitch[start_idx:end_idx] = note.pitch

unaligned_note_pitch[unaligned_note_pitch == 0] = np.nan

plt.figure(figsize=(14, 6))
plt.title("Vocal Pitch vs Aligned & Unaligned Note Pitch")

plt.plot(vocal_pitch, label="Vocal Pitch (Extracted F0)", color='blue', alpha=0.5)
plt.plot(unaligned_note_pitch, label="Unaligned Note Pitch (Original MIDI)", color='gray', linestyle='--', linewidth=2, alpha=0.8)
plt.plot(aligned_note_pitch, label="Aligned Note Pitch", color='red', linewidth=2, alpha=0.8)

plt.xlabel("Frames")
plt.ylabel("MIDI Pitch")
plt.legend()

save_path = "/home/vocod/GAME/0006_alignment.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
