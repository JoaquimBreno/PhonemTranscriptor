from pathlib import Path

from phonem_transcriptor.paths import SOFA_LANGUAGES, sofa_checkpoint, sofa_dictionary


def aligner_for(language: str, models_dir: Path) -> str:
    """SOFA when that language has both a checkpoint and a dictionary on disk.

    English, Chinese and Korean are required to use SOFA. A missing asset is
    an error, not a silent fall back to MMS. Every other language uses MMS.
    """
    language = language.lower()
    if language in SOFA_LANGUAGES:
        checkpoint = sofa_checkpoint(models_dir, language)
        dictionary = sofa_dictionary(models_dir, language)
        missing = [path for path in (checkpoint, dictionary) if not path.is_file()]
        if missing:
            names = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                f"SOFA assets for '{language}' are missing: {names}"
            )
        return "sofa"
    return "mms"
