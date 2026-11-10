"""Torchaudio MMS forced alignment, then dictionary phonemes inherit letter times."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from phonem_transcriptor.paths import MODELS as _MODELS

os.environ.setdefault("TORCH_HOME", str(_MODELS / "torch"))

from phonem_transcriptor.g2p import clean_word, load_tab_dictionary, lookup_word, romanize_word
from phonem_transcriptor.paths import MFA_LANGUAGES, mfa_dictionary
from phonem_transcriptor.phones import assign_phone_times
from phonem_transcriptor.whisper_asr import SpeechSegment

_BUNDLE = None
_MODEL = None
_TOKENIZER = None
_ALIGNER = None
_DICTIONARIES: dict[str, dict[str, list[str]]] = {}


def align(
    language: str,
    segments: list[SpeechSegment],
    waveform: torch.Tensor,
    sample_rate: int,
    models_dir: Path,
    device: str,
) -> list[dict]:
    _load(device)
    dictionary = _language_dictionary(language, models_dir)
    import torchaudio

    if sample_rate != _BUNDLE.sample_rate:
        audio = waveform
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = torchaudio.functional.resample(audio.cpu(), sample_rate, _BUNDLE.sample_rate)
    else:
        audio = waveform if waveform.dim() == 2 else waveform.unsqueeze(0)
    audio = audio.cpu()
    rate = _BUNDLE.sample_rate
    phonemes = []
    for segment in segments:
        phonemes.extend(_align_segment(segment, audio, rate, dictionary, device))
    return phonemes


def _load(device: str) -> None:
    global _BUNDLE, _MODEL, _TOKENIZER, _ALIGNER
    if _MODEL is not None:
        return
    from torchaudio.pipelines import MMS_FA

    _BUNDLE = MMS_FA
    _MODEL = MMS_FA.get_model().to(device)
    _MODEL.eval()
    _TOKENIZER = MMS_FA.get_tokenizer()
    _ALIGNER = MMS_FA.get_aligner()


def _language_dictionary(language: str, models_dir: Path) -> dict[str, list[str]]:
    if language not in MFA_LANGUAGES:
        return {}
    cached = _DICTIONARIES.get(language)
    if cached is not None:
        return cached
    path = mfa_dictionary(models_dir, language)
    dictionary = load_tab_dictionary(path) if path.is_file() else {}
    _DICTIONARIES[language] = dictionary
    return dictionary


def _align_segment(segment, audio, rate, dictionary, device) -> list[dict]:
    words = []
    romans = []
    for word in segment.words:
        roman = romanize_word(word.text)
        if not roman:
            continue
        words.append(word)
        romans.append(roman)
    if not romans:
        return []
    try:
        return _align_span(words, romans, segment.start, segment.end, audio, rate, dictionary, device)
    except RuntimeError:
        # The whole Whisper segment has more letters than the audio has CTC frames
        # (common when a sung segment is transcribed with squeezed word times).
        # Align word by word inside each word's own time span instead.
        rows = []
        for word, roman in zip(words, romans):
            try:
                rows.extend(
                    _align_span([word], [roman], word.start, word.end, audio, rate, dictionary, device)
                )
            except RuntimeError:
                rows.append(_romanized_fallback(word, roman))
        return rows


def _align_span(words, romans, span_start, span_end, audio, rate, dictionary, device) -> list[dict]:
    start = max(0, int(span_start * rate))
    end = min(audio.shape[-1], int(round(span_end * rate)))
    if end - start < int(0.05 * rate):
        return [
            _romanized_fallback(word, roman)
            for word, roman in zip(words, romans)
        ]
    piece = audio[:, start:end].to(device)
    with torch.inference_mode():
        emission, _ = _MODEL(piece)
        token_spans = _ALIGNER(emission[0], _TOKENIZER(romans))
    ratio = piece.shape[-1] / emission.shape[1] / rate
    rows = []
    for word, roman, spans in zip(words, romans, token_spans):
        if not spans:
            rows.append(_romanized_fallback(word, roman))
            continue
        letter_spans = [
            (
                round(span_start + span.start * ratio, 3),
                round(span_start + span.end * ratio, 3),
            )
            for span in spans
        ]
        if len(letter_spans) != len(roman):
            letter_spans = _fit_spans(letter_spans, len(roman), span_start, word)
        phones = _phones_for(word.text, roman, dictionary)
        if phones is None:
            for letter, (letter_start, letter_end) in zip(roman, letter_spans):
                if letter_end <= letter_start:
                    continue
                rows.append(
                    {
                        "phoneme": letter,
                        "start": letter_start,
                        "end": letter_end,
                        "word": word.text,
                        "phoneme_source": "romanized",
                    }
                )
            continue
        for phone, phone_start, phone_end in assign_phone_times(roman, letter_spans, phones):
            if phone_end <= phone_start:
                continue
            rows.append(
                {
                    "phoneme": phone,
                    "start": round(phone_start, 3),
                    "end": round(phone_end, 3),
                    "word": word.text,
                }
            )
    return rows


def _phones_for(text: str, roman: str, dictionary: dict[str, list[str]]) -> list[str] | None:
    if not dictionary:
        return None
    phones = lookup_word(dictionary, text)
    if phones is None and clean_word(text) != roman:
        phones = lookup_word(dictionary, roman)
    return phones


def _fit_spans(spans, count, offset, word):
    if not spans:
        width = max(word.end - word.start, 0.01) / count
        return [
            (round(word.start + index * width, 3), round(word.start + (index + 1) * width, 3))
            for index in range(count)
        ]
    if len(spans) > count:
        return spans[:count]
    start = spans[0][0]
    end = spans[-1][1]
    width = max(end - start, 0.01) / count
    return [(round(start + index * width, 3), round(start + (index + 1) * width, 3)) for index in range(count)]


def _romanized_fallback(word, roman: str) -> dict:
    return {
        "phoneme": roman or word.text,
        "start": round(word.start, 3),
        "end": round(word.end, 3),
        "word": word.text,
        "phoneme_source": "romanized",
    }
