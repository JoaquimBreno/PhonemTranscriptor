"""In-process SOFA alignment. One language model stays on GPU after the first use."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from phonem_transcriptor.g2p import (
    KoreanG2P,
    chinese_syllables,
    load_tab_dictionary,
    lookup_word,
    match_vocab,
)
from phonem_transcriptor.paths import SOFA_ROOT
from phonem_transcriptor.whisper_asr import SpeechSegment, WordSpan

_MODELS: dict[str, "SofaAligner"] = {}


def align(language: str, segments: list[SpeechSegment], waveform, sample_rate: int, models_dir: Path, device: str) -> list[dict]:
    aligner = _MODELS.get(language)
    if aligner is None:
        from phonem_transcriptor.paths import sofa_checkpoint, sofa_dictionary

        aligner = SofaAligner(
            language,
            sofa_checkpoint(models_dir, language),
            sofa_dictionary(models_dir, language),
            device,
        )
        _MODELS[language] = aligner
    phonemes = []
    for segment in segments:
        phonemes.extend(aligner.align_segment(segment, waveform, sample_rate))
    return phonemes


class SofaAligner:
    def __init__(self, language: str, checkpoint: Path, dictionary: Path, device: str):
        self.language = language
        self.device = device
        self.dictionary = load_tab_dictionary(dictionary)
        if str(SOFA_ROOT) not in sys.path:
            sys.path.insert(0, str(SOFA_ROOT))
        from modules.rmvpe import MelSpectrogram
        from modules.task.forced_alignment import LitForcedAlignmentTask

        self.model = LitForcedAlignmentTask.load_from_checkpoint(
            str(checkpoint),
            map_location=device,
        )
        self.model.eval()
        self.model.to(device)
        self.model.set_inference_mode("force")
        config = dict(self.model.melspec_config)
        self.sample_rate = int(config["sample_rate"])
        self.scale_factor = int(config.get("scale_factor") or 1)
        self.mel = MelSpectrogram(
            n_mel_channels=config["n_mels"],
            sampling_rate=self.sample_rate,
            win_length=config["win_length"],
            hop_length=config["hop_length"],
            n_fft=config["n_fft"],
            mel_fmin=config["fmin"],
            mel_fmax=config["fmax"],
            clamp=config["clamp"],
        ).to(device)
        self.korean = None
        if language == "ko":
            self.korean = KoreanG2P(dictionary, self.model.vocab)

    def align_segment(self, segment: SpeechSegment, waveform: torch.Tensor, sample_rate: int) -> list[dict]:
        start = max(0, int(segment.start * sample_rate))
        end = min(waveform.shape[-1], int(segment.end * sample_rate))
        if end - start < int(0.05 * sample_rate):
            return _unmapped_words(segment.words, "too_short")
        piece = waveform[..., start:end]
        if sample_rate != self.sample_rate:
            piece = torchaudio_resample(piece, sample_rate, self.sample_rate)
        piece = piece.to(self.device)
        forced, pending = self._sequence(segment.words)
        if forced is None:
            return pending
        ph_seq, word_seq, ph_idx_to_word_idx = forced
        try:
            predicted = self._infer(piece, ph_seq, word_seq, ph_idx_to_word_idx)
        except Exception:
            return pending + _unmapped_words(segment.words, "align_failed")
        predicted = _shift(predicted, segment.start)
        return pending + predicted

    def _sequence(self, words: list[WordSpan]):
        entries: list[tuple[str, list[str]]] = []
        pending: list[dict] = []
        for word in words:
            phones, leftover = self._phones_for(word.text)
            if phones:
                entries.append((word.text, phones))
            elif leftover or word.text.strip():
                pending.append(_unmapped(word, "unmapped"))
        if not entries:
            return None, pending
        ph_seq = ["SP"]
        ph_idx_to_word_idx = [-1]
        word_seq = []
        for word, phones in entries:
            index = len(word_seq)
            word_seq.append(word)
            for phone in phones:
                ph_seq.append(phone)
                ph_idx_to_word_idx.append(index)
            ph_seq.append("SP")
            ph_idx_to_word_idx.append(-1)
        non_silence = [phone for phone in ph_seq if phone != "SP"]
        if not non_silence:
            return None, pending
        return (ph_seq, word_seq, ph_idx_to_word_idx), pending

    def _phones_for(self, text: str) -> tuple[list[str], list[str]]:
        if self.language == "zh":
            return self._chinese(text)
        if self.language == "ko":
            return self.korean.phones_for_word(text)
        return self._dictionary_word(text)

    def _dictionary_word(self, text: str) -> tuple[list[str], list[str]]:
        phones = lookup_word(self.dictionary, text)
        if not phones:
            return [], [text]
        matched = []
        missing = []
        for phone in phones:
            hit = match_vocab(phone, self.model.vocab)
            if hit is None:
                missing.append(phone)
            else:
                matched.append(hit)
        if missing and not matched:
            return [], missing
        return matched, missing

    def _chinese(self, text: str) -> tuple[list[str], list[str]]:
        pairs = chinese_syllables(text)
        if not pairs and lookup_word(self.dictionary, text):
            return self._dictionary_word(text)
        phones = []
        missing = []
        for _char, syllable in pairs:
            syllable_phones = self.dictionary.get(syllable)
            if not syllable_phones:
                missing.append(syllable)
                continue
            for phone in syllable_phones:
                hit = match_vocab(phone, self.model.vocab)
                if hit is None:
                    missing.append(phone)
                else:
                    phones.append(hit)
        return phones, missing

    def _infer(self, waveform, ph_seq, word_seq, ph_idx_to_word_idx):
        from einops import repeat

        wav_length = waveform.shape[-1] / self.sample_rate
        with torch.no_grad():
            mel = self.mel(waveform.unsqueeze(0), 0).squeeze(0).detach().unsqueeze(0)
            std = mel.std().clamp_min(1e-5)
            mel = (mel - mel.mean()) / std
            mel = repeat(mel, "B C T -> B C (T N)", N=self.scale_factor)
            ph_seq_pred, ph_intervals, _word_seq, _word_intervals, _confidence, _ctc, _fig = (
                self.model._infer_once(
                    mel,
                    wav_length,
                    ph_seq,
                    word_seq,
                    ph_idx_to_word_idx,
                    False,
                    False,
                )
            )
        owned = []
        for phone, word_index in zip(ph_seq, ph_idx_to_word_idx):
            if phone == "SP":
                continue
            owned.append(word_seq[word_index])
        rows = []
        intervals = np.asarray(ph_intervals)
        for index, phone in enumerate(ph_seq_pred):
            if index >= len(intervals) or index >= len(owned):
                break
            start = float(intervals[index, 0])
            end = float(intervals[index, 1])
            if end <= start:
                continue
            rows.append(
                {
                    "phoneme": str(phone),
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "word": owned[index],
                }
            )
        return rows


def _shift(rows: list[dict], offset: float) -> list[dict]:
    shifted = []
    for row in rows:
        item = dict(row)
        item["start"] = round(item["start"] + offset, 3)
        item["end"] = round(item["end"] + offset, 3)
        shifted.append(item)
    return shifted


def _unmapped(word: WordSpan, source: str) -> dict:
    return {
        "phoneme": word.text,
        "start": round(word.start, 3),
        "end": round(word.end, 3),
        "word": word.text,
        "phoneme_source": source,
    }


def _unmapped_words(words: list[WordSpan], source: str) -> list[dict]:
    return [_unmapped(word, source) for word in words]


def torchaudio_resample(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    import torchaudio

    squeezed = waveform
    if squeezed.dim() == 1:
        squeezed = squeezed.unsqueeze(0)
    return torchaudio.functional.resample(squeezed.cpu(), source_rate, target_rate).squeeze(0)
