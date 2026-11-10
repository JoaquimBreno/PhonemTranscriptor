"""Transcribe a wav with GAME. Run as a script so SOFA never shares this process.

GAME and SOFA both ship top-level ``lib``/``modules`` packages, so they cannot
be imported side by side.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--language", default="")
    args = parser.parse_args()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from phonem_transcriptor.game_transcribe import transcribe_wav

    notes = transcribe_wav(
        args.wav,
        model_dir=args.model,
        device=args.device,
        language=args.language.strip() or None,
    )
    args.out.write_text(json.dumps(notes), encoding="utf-8")


if __name__ == "__main__":
    main()
