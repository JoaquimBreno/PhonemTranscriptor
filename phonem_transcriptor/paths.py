import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
SOFA_ROOT = ROOT / "third_party" / "SOFA"
GAME_ROOT = ROOT / "third_party" / "GAME"
CACHE = MODELS / "cache"

SOFA_LANGUAGES = ("en", "zh", "ko")
MFA_LANGUAGES = {"pt": "portuguese_mfa.dict", "es": "spanish_mfa.dict"}


def sofa_checkpoint(models_dir: Path, language: str) -> Path:
    return models_dir / "sofa" / language / "model.ckpt"


def sofa_dictionary(models_dir: Path, language: str) -> Path:
    return models_dir / "sofa" / language / "dictionary.txt"


def mfa_dictionary(models_dir: Path, language: str) -> Path:
    name = MFA_LANGUAGES[language]
    return models_dir / "dictionaries" / name


def game_model_dir() -> Path:
    env = os.environ.get("PHONEM_GAME_MODEL")
    if env:
        return Path(env).expanduser()
    local = MODELS / "GAME-1.0-medium"
    if (local / "model.pt").is_file():
        return local
    raise FileNotFoundError(f"GAME checkpoint missing: {local / 'model.pt'}")


def game_python() -> Path:
    """Interpreter for the GAME worker. ``PHONEM_GAME_PYTHON`` overrides it."""
    env = os.environ.get("PHONEM_GAME_PYTHON")
    if env:
        return Path(env).expanduser()
    return Path(sys.executable)
