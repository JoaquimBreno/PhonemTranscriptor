import tempfile
import unittest
from pathlib import Path

import mido

from phonem_transcriptor.game_midi import attach_phonemes, write_game_midi


class GameMidiTest(unittest.TestCase):
    def test_a_phoneme_lands_on_the_note_it_overlaps_most(self):
        notes = [
            {"pitch": 60, "onset": 0.0, "offset": 1.0, "velocity": 80},
            {"pitch": 62, "onset": 1.0, "offset": 2.0, "velocity": 80},
        ]
        phonemes = [
            {"phoneme": "a", "start": 0.2, "end": 0.8, "word": "amor"},
            {"phoneme": "m", "start": 0.9, "end": 1.4, "word": "amor"},
            {"phoneme": "o", "start": 3.0, "end": 3.2, "word": "amor"},
        ]
        attached = attach_phonemes(notes, phonemes)
        self.assertEqual(attached[0]["phonemes"], ["a"])
        self.assertEqual(attached[1]["phonemes"], ["m"])
        self.assertEqual(attached[1]["words"], ["amor"])
        self.assertEqual(attached[0]["pitch"], 60)

    def test_midi_roundtrip_keeps_the_ipa_lyric(self):
        notes = [{"pitch": 57, "onset": 1.0, "offset": 1.5, "velocity": 80, "phonemes": ["ɐ"], "words": ["uma"]}]
        phonemes = [{"phoneme": "ɐ", "start": 1.1, "end": 1.4, "word": "uma"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vocal.mid"
            write_game_midi(path, notes, phonemes)
            midi = mido.MidiFile(path, charset="utf-8")
        lyrics = [message.text for track in midi.tracks for message in track if message.type == "lyrics"]
        played = [
            message.note
            for track in midi.tracks
            for message in track
            if message.type == "note_on" and message.velocity > 0
        ]
        self.assertEqual(lyrics, ["ɐ"])
        self.assertEqual(played, [57])


if __name__ == "__main__":
    unittest.main()
