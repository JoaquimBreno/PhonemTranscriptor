"""Download Whisper, SOFA, GAME, the dictionaries, and the MMS aligner into models/."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from urllib.request import urlopen

from phonem_transcriptor.paths import (
    CACHE,
    MFA_LANGUAGES,
    MODELS,
    SOFA_LANGUAGES,
    SOFA_ROOT,
    mfa_dictionary,
    sofa_checkpoint,
    sofa_dictionary,
)

SOFA_REPO = "https://github.com/qiuqiao/SOFA.git"
SOFA_TAG = "v1.0.3"

ASSETS = {
    "en_ckpt": "https://github.com/spicytigermeat/SOFA-Models/releases/download/v1.0.0_en/tgm_en_v100.ckpt",
    "en_dict": "https://raw.githubusercontent.com/spicytigermeat/SOFA-Models/main/tgm_sofa_dict.txt",
    "zh_zip": "https://github.com/qiuqiao/SOFA/releases/download/v1.0.1/pretrained_mandarin_singing.zip",
    "ko_zip": "https://github.com/colstone/SOFA_Models/releases/download/KOR-V0.01b/SOFA_model_KOR_Ver0.0.1_Beta.zip",
    "pt_dict": "https://raw.githubusercontent.com/MontrealCorpusTools/mfa-models/main/dictionary/portuguese/mfa/portuguese_mfa.dict",
    "es_dict": "https://raw.githubusercontent.com/MontrealCorpusTools/mfa-models/main/dictionary/spanish/mfa/spanish_mfa.dict",
    "game_zip": "https://github.com/openvpi/GAME/releases/download/v1.0.0/GAME-1.0-medium.zip",
}


def main() -> None:
    os.environ.setdefault("HF_HOME", str(MODELS / "hf"))
    os.environ.setdefault("TORCH_HOME", str(MODELS / "torch"))
    CACHE.mkdir(parents=True, exist_ok=True)
    _clone_sofa()
    _fetch_assets()
    _place_game()
    _place_english()
    _place_chinese()
    _place_korean()
    _place_mfa()
    _download_whisper()
    _download_vad()
    _download_mms()
    manifest = _write_manifest()
    missing = [path for path, minimum in required_files() if not _large_enough(Path(path), minimum)]
    if missing:
        raise SystemExit("Missing files:\n" + "\n".join(missing))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def required_files() -> list[tuple[str, int]]:
    files = []
    for language in SOFA_LANGUAGES:
        files.append((str(sofa_checkpoint(MODELS, language)), 10_000_000))
        files.append((str(sofa_dictionary(MODELS, language)), 100))
    for language in MFA_LANGUAGES:
        files.append((str(mfa_dictionary(MODELS, language)), 100_000))
    whisper_bins = [path for path in (MODELS / "whisper").rglob("model.bin") if path.is_file()]
    files.append((str(whisper_bins[0]) if whisper_bins else str(MODELS / "whisper" / "model.bin"), 100_000_000))
    mms_weights = [path for path in (MODELS / "torch").rglob("*.pt") if path.is_file()]
    files.append((str(mms_weights[0]) if mms_weights else str(MODELS / "torch" / "model.pt"), 100_000_000))
    files.append((str(MODELS / "GAME-1.0-medium" / "model.pt"), 100_000_000))
    return files


def _large_enough(path: Path, minimum: int) -> bool:
    return path.is_file() and path.stat().st_size >= minimum


def _clone_sofa() -> None:
    if (SOFA_ROOT / "infer.py").is_file():
        return
    SOFA_ROOT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", SOFA_TAG, SOFA_REPO, str(SOFA_ROOT)],
        check=True,
    )


def _fetch_assets() -> None:
    targets = {
        "en_ckpt": CACHE / "tgm_en_v100.ckpt",
        "en_dict": CACHE / "tgm_sofa_dict.txt",
        "zh_zip": CACHE / "pretrained_mandarin_singing.zip",
        "ko_zip": CACHE / "SOFA_model_KOR_Ver0.0.1_Beta.zip",
        "pt_dict": CACHE / "portuguese_mfa.dict",
        "es_dict": CACHE / "spanish_mfa.dict",
    }
    for key, destination in targets.items():
        _download(ASSETS[key], destination)


def _download(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    subprocess.run(
        ["curl", "-fL", "--retry", "3", "-C", "-", "-o", str(partial), url],
        check=True,
    )
    partial.replace(destination)


def _place_game() -> None:
    if (MODELS / "GAME-1.0-medium" / "model.pt").is_file():
        return
    archive = CACHE / "GAME-1.0-medium.zip"
    _download(ASSETS["game_zip"], archive)
    # The archive already holds a GAME-1.0-medium/ folder.
    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            relative = Path(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                continue
            zipped.extract(info, MODELS)


def _place_english() -> None:
    _link(CACHE / "tgm_en_v100.ckpt", sofa_checkpoint(MODELS, "en"))
    _link(CACHE / "tgm_sofa_dict.txt", sofa_dictionary(MODELS, "en"))


def _place_chinese() -> None:
    extracted = CACHE / "zh"
    _unzip(CACHE / "pretrained_mandarin_singing.zip", extracted)
    checkpoint = _largest(extracted, ".ckpt")
    _link(checkpoint, sofa_checkpoint(MODELS, "zh"))
    opencpop = SOFA_ROOT / "dictionary" / "opencpop-extension.txt"
    _link(opencpop, sofa_dictionary(MODELS, "zh"))


def _place_korean() -> None:
    extracted = CACHE / "ko"
    _unzip(CACHE / "SOFA_model_KOR_Ver0.0.1_Beta.zip", extracted)
    checkpoint = _largest(extracted, ".ckpt")
    _link(checkpoint, sofa_checkpoint(MODELS, "ko"))
    table = CACHE / "hangul.table"
    if not table.is_file():
        url = _hangul_table_url()
        _download(url, table)
    _link(table, sofa_dictionary(MODELS, "ko"))


def _place_mfa() -> None:
    _link(CACHE / "portuguese_mfa.dict", mfa_dictionary(MODELS, "pt"))
    _link(CACHE / "spanish_mfa.dict", mfa_dictionary(MODELS, "es"))


def _hangul_table_url() -> str:
    with urlopen(
        "https://api.github.com/repos/Coda-SVS/nnsvs-korean-support/git/trees/main?recursive=1",
        timeout=60,
    ) as response:
        tree = json.load(response)["tree"]
    matches = [
        item["path"]
        for item in tree
        if item["path"].endswith("hangul.table") and "ext" not in item["path"]
    ]
    if not matches:
        raise FileNotFoundError("hangul.table was not found in nnsvs-korean-support")
    matches.sort(key=len)
    return "https://raw.githubusercontent.com/Coda-SVS/nnsvs-korean-support/main/" + matches[0]


def _download_whisper() -> None:
    from faster_whisper import WhisperModel

    from phonem_transcriptor.whisper_asr import WHISPER_MODEL

    WhisperModel(
        WHISPER_MODEL,
        device="cpu",
        compute_type="int8",
        download_root=str(MODELS / "whisper"),
    )


def _download_vad() -> None:
    from faster_whisper.vad import get_vad_model

    get_vad_model()


def _download_mms() -> None:
    from torchaudio.pipelines import MMS_FA

    MMS_FA.get_model()


def _write_manifest() -> dict:
    languages = {}
    for language in SOFA_LANGUAGES:
        languages[language] = {
            "aligner": "sofa",
            "checkpoint": str(sofa_checkpoint(MODELS, language)),
            "dictionary": str(sofa_dictionary(MODELS, language)),
        }
    for language in MFA_LANGUAGES:
        languages[language] = {
            "aligner": "mms",
            "dictionary": str(mfa_dictionary(MODELS, language)),
        }
    manifest = {
        "whisper": str(MODELS / "whisper"),
        "mms": str(MODELS / "torch"),
        "languages": languages,
    }
    destination = MODELS / "manifest.json"
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.is_file():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return
        if destination.is_file() and not destination.is_symlink() and destination.stat().st_size > 0:
            return
        destination.unlink()
    destination.symlink_to(source.resolve())


def _unzip(archive: Path, destination: Path) -> None:
    marker = destination / ".extracted"
    if marker.is_file():
        return
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            relative = Path(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                continue
            zipped.extract(info, destination)
    marker.write_text("ok\n", encoding="utf-8")


def _largest(root: Path, suffix: str) -> Path:
    found = [path for path in root.rglob(f"*{suffix}") if path.is_file()]
    if not found:
        raise FileNotFoundError(f"No {suffix} file under {root}")
    found.sort(key=lambda path: path.stat().st_size, reverse=True)
    return found[0]


if __name__ == "__main__":
    main()
