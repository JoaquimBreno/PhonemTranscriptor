from __future__ import annotations

from pathlib import Path

import torch

from phonem_transcriptor.game_midi import render_game
from phonem_transcriptor.mms_align import align as align_mms
from phonem_transcriptor.paths import MODELS
from phonem_transcriptor.routing import aligner_for
from phonem_transcriptor.sofa_align import align as align_sofa
from phonem_transcriptor.whisper_asr import WhisperTranscriber

_WHISPER: dict[tuple[str, str], WhisperTranscriber] = {}


def transcribe(
    path: str | Path,
    language: str = "auto",
    models_dir: Path | None = None,
    device: str | None = None,
    midi_path: str | Path | None = None,
    game_model: str | Path | None = None,
) -> dict:
    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    models = Path(models_dir) if models_dir is not None else MODELS
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    requested = None if language in (None, "", "auto") else language
    transcript = _transcriber(models, device).transcribe(audio_path, requested)
    kind = aligner_for(transcript.language, models)
    if not transcript.segments:
        phonemes = []
    else:
        waveform, sample_rate = _load_audio(audio_path)
        if kind == "sofa":
            phonemes = align_sofa(
                transcript.language, transcript.segments, waveform, sample_rate, models, device
            )
        else:
            phonemes = align_mms(
                transcript.language, transcript.segments, waveform, sample_rate, models, device
            )
    game = render_game(
        audio_path,
        transcript.language,
        phonemes,
        device=device,
        model_dir=game_model,
        midi_path=midi_path,
    )
    return {
        "language": transcript.language,
        "aligner": kind,
        "phonemes": phonemes,
        "game": game,
    }


def _transcriber(models: Path, device: str) -> WhisperTranscriber:
    key = (str(models), device)
    model = _WHISPER.get(key)
    if model is None:
        model = WhisperTranscriber(models / "whisper", device=device)
        _WHISPER[key] = model
    return model


def _load_audio(path: Path):
    import torchaudio

    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0), int(sample_rate)
