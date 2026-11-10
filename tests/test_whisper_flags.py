import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from phonem_transcriptor.whisper_asr import (
    CONDITION_ON_PREVIOUS_TEXT,
    VAD_FILTER,
    WhisperTranscriber,
)


class WhisperFlagTest(unittest.TestCase):
    def test_constants(self):
        self.assertIs(CONDITION_ON_PREVIOUS_TEXT, False)
        self.assertIs(VAD_FILTER, True)

    def test_transcribe_passes_the_flags(self):
        word = SimpleNamespace(word=" ola ", start=0.1, end=0.4)
        segment = SimpleNamespace(start=0.0, end=0.5, words=[word])
        info = SimpleNamespace(language="pt")
        with patch("phonem_transcriptor.whisper_asr.WhisperModel") as model:
            model.return_value.transcribe.return_value = ([segment], info)
            transcript = WhisperTranscriber(Path("models/whisper"), device="cpu").transcribe(
                "audio.wav",
                "pt",
            )
        kwargs = model.return_value.transcribe.call_args.kwargs
        self.assertIs(kwargs["condition_on_previous_text"], False)
        self.assertIs(kwargs["vad_filter"], True)
        self.assertIs(kwargs["word_timestamps"], True)
        self.assertEqual(transcript.language, "pt")
        self.assertEqual(transcript.segments[0].words[0].text, "ola")


if __name__ == "__main__":
    unittest.main()
