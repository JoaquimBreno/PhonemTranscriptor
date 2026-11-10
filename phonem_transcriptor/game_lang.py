"""Map a Whisper language code onto a GAME language embedding id."""

from __future__ import annotations

import json
from pathlib import Path

# Embedding size is num_languages + 1 (padding at 0). Ids outside this crash inference.
_MAX_LANGUAGE_ID = 127


def load_lang_map(model_dir: Path) -> dict[str, int]:
    path = Path(model_dir) / "lang_map.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    mapping = {str(key).lower(): int(value) for key, value in raw.items()}
    for code, value in mapping.items():
        if not 0 <= value <= _MAX_LANGUAGE_ID:
            raise ValueError(f"lang id {value} for {code!r} is outside 0..{_MAX_LANGUAGE_ID}")
    return mapping


def game_language_id(language: str | None, lang_map: dict[str, int]) -> int:
    """Return the trained id, or 0 when the checkpoint has no row for this language.

    Id 0 is the universal embedding. Portuguese, Spanish and Korean are not in the
    GAME Medium map, so they stay on 0. English, Japanese, Cantonese and Chinese use
    the ids shipped in lang_map.json.
    """
    if not language:
        return 0
    return int(lang_map.get(language.strip().lower(), 0))
