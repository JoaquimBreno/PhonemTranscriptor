"""GAME Medium transcription of an unaccompanied vocal into notes.

Post-GAME matches PitchNet ``EditorController``: drop rests/short notes, then
VAD + rest-guided boundary refine. Pitch is rounded to the nearest semitone.

GAME's top-level packages (``lib``, ``modules``, ``inference``) clash with
SOFA's, so this module is only imported from ``game_worker`` in its own process.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from phonem_transcriptor.game_postprocess import (
    HOP,
    SR,
    events_from_game_outputs,
    events_to_note_tuples,
    postprocess_game_events,
)
from phonem_transcriptor.paths import GAME_ROOT, game_model_dir


def _ensure_game_path() -> None:
    if str(GAME_ROOT) not in sys.path:
        sys.path.insert(0, str(GAME_ROOT))


def load_game(model_dir: str | Path | None = None, device: str | torch.device = "cpu"):
    """Load GAME Medium inference model (encoder + segmenter + estimator)."""
    _ensure_game_path()
    from inference.me_infer import SegmentationEstimationInferenceModel
    from lib.config.schema import InferenceConfig, ModelConfig

    model_dir = Path(model_dir).expanduser() if model_dir else game_model_dir()
    raw_cfg = yaml.safe_load((model_dir / "config.yaml").read_text(encoding="utf-8"))
    model_cfg = ModelConfig.model_validate(raw_cfg["model"])
    infer_cfg = InferenceConfig.model_validate(raw_cfg["inference"])
    model = SegmentationEstimationInferenceModel(model_cfg, infer_cfg)
    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)
    state_dict = dict(checkpoint.get("state_dict", {}))
    if checkpoint.get("ema_state_dict"):
        state_dict.update(checkpoint["ema_state_dict"])
    model.load_state_dict(state_dict, strict=True)
    return model.eval().to(device), infer_cfg


@torch.inference_mode()
def transcribe_wav(
    wav_path: str | Path,
    *,
    model_dir: str | Path | None = None,
    device: str | torch.device = "cuda",
    presence_threshold: float = 0.2,
    language: str | None = None,
    pitchnet_postprocess: bool = True,
    seed: int | None = 0,
) -> list[dict]:
    """Transcribe a singing WAV with GAME into ``{pitch, onset, offset, velocity}`` notes.

    D3PM boundary sampling is random; ``seed=None`` keeps it unseeded.
    """
    import librosa

    if seed is not None:
        torch.manual_seed(seed)
    model_dir = Path(model_dir).expanduser() if model_dir else game_model_dir()
    model, infer_cfg = load_game(model_dir, device=device)

    game_sr = int(
        getattr(
            infer_cfg.features,
            "audio_sample_rate",
            getattr(infer_cfg.features, "sample_rate", 44100),
        )
    )
    waveform_np, _ = librosa.load(str(wav_path), sr=game_sr, mono=True)
    if waveform_np.size == 0:
        return []

    device_t = torch.device(device)
    wav = torch.from_numpy(waveform_np).float().unsqueeze(0).to(device_t)
    dur_sec = float(len(waveform_np) / game_sr)
    # A single known region spanning the whole clip (unsupervised segmentation).
    known = torch.tensor([[dur_sec]], dtype=torch.float32, device=device_t)
    seg = getattr(infer_cfg, "segmentation", None)
    b_thr = torch.tensor(
        float(getattr(seg, "threshold", 0.2) if seg is not None else 0.2),
        device=device_t,
    )
    b_rad = torch.tensor(
        int(getattr(seg, "radius", 2) if seg is not None else 2),
        device=device_t,
        dtype=torch.long,
    )
    s_thr = torch.tensor(float(presence_threshold), device=device_t)

    # Language id 0 = unset/universal (required by language_embedding).
    lang_id = 0
    if language:
        lang_map_path = model_dir / "lang_map.json"
        if lang_map_path.is_file():
            lang_map = json.loads(lang_map_path.read_text(encoding="utf-8"))
            if language in lang_map:
                lang_id = int(lang_map[language])
    lang_t = torch.tensor([lang_id], dtype=torch.long, device=device_t)

    # D3PM sampling schedule (ValidationConfig defaults: t0=0, steps=8).
    nsteps = 8
    t0 = 0.0
    step = (1.0 - t0) / float(nsteps)
    t_sched = torch.tensor(
        [t0 + i * step for i in range(nsteps)],
        device=device_t,
        dtype=torch.float32,
    )
    durations, presence, scores = model.forward(
        waveform=wav,
        known_durations=known,
        boundary_threshold=b_thr,
        boundary_radius=b_rad,
        score_threshold=s_thr,
        language=lang_t,
        t=t_sched,
    )
    d = durations[0].detach().cpu().numpy()
    pres = presence[0].detach().cpu().numpy().astype(bool)
    pit = scores[0].detach().cpu().numpy()

    events = events_from_game_outputs(d, pres, pit, hop=HOP, sr=SR)

    if pitchnet_postprocess:
        # VAD refine runs on the PitchNet grid (44.1 kHz / hop 512).
        if game_sr != SR:
            audio_44 = librosa.resample(waveform_np, orig_sr=game_sr, target_sr=SR)
        else:
            audio_44 = waveform_np
        n_frames = max(
            (len(audio_44) + HOP - 1) // HOP,
            max((e.end_frame for e in events), default=0),
        )
        events = postprocess_game_events(events, audio_44, hop=HOP, sr=SR, n_frames=n_frames)
        tuples = events_to_note_tuples(events, hop=HOP, sr=SR)
    else:
        tuples = events_to_note_tuples(
            [e for e in events if not e.is_rest and (e.end_frame - e.start_frame) >= 3],
            hop=HOP,
            sr=SR,
        )

    return [
        {
            "pitch": int(np.clip(np.round(float(midi)), 0, 127)),
            "onset": float(onset),
            "offset": float(offset),
            "velocity": 80,
        }
        for onset, offset, midi in tuples
    ]
