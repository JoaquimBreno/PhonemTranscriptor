"""faster-whisper with the two flags that stop lyric loops on instrumental bridges."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from faster_whisper import WhisperModel

# LyricWhiz-style decoding: do not feed the previous line back in, and drop
# non-speech so a bridge is not filled with the last verse.
CONDITION_ON_PREVIOUS_TEXT = False
VAD_FILTER = True
WHISPER_MODEL = "large-v3-turbo"


@dataclass
class WordSpan:
    text: str
    start: float
    end: float


@dataclass
class SpeechSegment:
    start: float
    end: float
    words: list[WordSpan] = field(default_factory=list)


@dataclass
class Transcript:
    language: str
    segments: list[SpeechSegment]


class WhisperTranscriber:
    def __init__(self, model_dir: Path, device: str = "cuda"):
        compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(
            WHISPER_MODEL,
            device=device,
            compute_type=compute_type,
            download_root=str(model_dir),
        )

    def transcribe(self, audio_path: str | Path, language: str | None) -> Transcript:
        segments, info = self.model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
            vad_filter=VAD_FILTER,
        )
        collected: list[SpeechSegment] = []
        for segment in segments:
            words = []
            for word in segment.words or []:
                text = (word.word or "").strip()
                if not text or word.start is None or word.end is None:
                    continue
                if word.end <= word.start:
                    continue
                words.append(WordSpan(text=text, start=float(word.start), end=float(word.end)))
            if not words:
                continue
            collected.append(
                SpeechSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    words=words,
                )
            )
        detected = language or info.language or "en"
        return Transcript(language=detected, segments=collected)
