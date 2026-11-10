import argparse
import json
from pathlib import Path

from phonem_transcriptor.pipeline import transcribe

LANGUAGES = ("auto", "en", "pt", "es", "zh", "ko")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recebe áudio e devolve fonemas com tempo.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--lang", default="auto", choices=LANGUAGES)
    args = parser.parse_args()
    midi_path = _midi_path(args.output)
    result = transcribe(args.audio, language=args.lang, midi_path=midi_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    game = result["game"]
    print(
        f"{result['language']} via {result['aligner']}: {len(result['phonemes'])} fonemas, "
        f"GAME lang {game['language_id']}, {len(game['notes'])} notas -> {args.output} | {midi_path}"
    )


def _midi_path(output: Path) -> Path:
    if output.suffix.lower() == ".mid":
        return output.with_name(output.stem + ".notes.mid")
    return output.with_suffix(".mid")


if __name__ == "__main__":
    main()
