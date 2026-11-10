"""Turn Whisper words into the phoneme string the aligner expects."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

_PUNCT = re.compile(r"^[^\w']+|[^\w']+$", re.UNICODE)


def clean_word(word: str) -> str:
    text = word.strip().lower().replace("’", "'").replace("`", "'")
    return _PUNCT.sub("", text)


def load_tab_dictionary(path: Path) -> dict[str, list[str]]:
    dictionary: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if "\t" in line:
            word, phones = line.split("\t", 1)
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            word, phones = parts[0], " ".join(parts[1:])
        word = word.strip().lower()
        phone_list = [phone for phone in phones.strip().split() if phone and phone != "SP"]
        phone_list = [phone for phone in phone_list if not _is_probability(phone)]
        if word and phone_list and word not in dictionary:
            dictionary[word] = phone_list
    return dictionary


def _is_probability(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def lookup_word(dictionary: dict[str, list[str]], word: str) -> list[str] | None:
    cleaned = clean_word(word)
    if not cleaned:
        return None
    for candidate in (cleaned, cleaned.replace("'", "")):
        phones = dictionary.get(candidate)
        if phones:
            return list(phones)
    return None


def chinese_syllables(text: str) -> list[tuple[str, str]]:
    """Return (character, pinyin syllable) for each Han character."""
    from pypinyin import Style, lazy_pinyin

    pairs = []
    for char in text:
        if not _is_han(char):
            continue
        syllable = lazy_pinyin(char, style=Style.NORMAL, errors="ignore")
        if not syllable:
            continue
        normalized = syllable[0].lower().replace("ü", "v").replace("u:", "v")
        normalized = re.sub(r"[^a-z]", "", normalized)
        if normalized:
            pairs.append((char, normalized))
    return pairs


def _is_han(char: str) -> bool:
    if len(char) != 1:
        return False
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF
    )


# Onset, nucleus and coda of the Korean SOFA checkpoint.
# Compound vowels try the single phone first (ya, wa, wi) and otherwise the glide plus vowel.
_KO_ONSET = {
    "ㄱ": "g", "ㄲ": "kk", "ㄴ": "n", "ㄷ": "d", "ㄸ": "tt", "ㄹ": "r",
    "ㅁ": "m", "ㅂ": "b", "ㅃ": "pp", "ㅅ": "s", "ㅆ": "ss", "ㅇ": "",
    "ㅈ": "j", "ㅉ": "jj", "ㅊ": "ch", "ㅋ": "k", "ㅌ": "t", "ㅍ": "p", "ㅎ": "h",
}
_KO_NUCLEUS = {
    "ㅏ": (("a",),),
    "ㅐ": (("e",),),
    "ㅑ": (("ya",), ("y", "a")),
    "ㅒ": (("y", "e"),),
    "ㅓ": (("eo",),),
    "ㅔ": (("e",),),
    "ㅕ": (("yeo",), ("y", "eo")),
    "ㅖ": (("y", "e"),),
    "ㅗ": (("o",),),
    "ㅘ": (("wa",), ("w", "a")),
    "ㅙ": (("we",), ("w", "e")),
    "ㅚ": (("we",), ("w", "e")),
    "ㅛ": (("yo",), ("y", "o")),
    "ㅜ": (("u",),),
    "ㅝ": (("weo",), ("w", "eo")),
    "ㅞ": (("we",), ("w", "e")),
    "ㅟ": (("wi",), ("w", "i")),
    "ㅠ": (("y", "u"),),
    "ㅡ": (("eu",),),
    "ㅢ": (("eu", "i"),),
    "ㅣ": (("i",),),
}
_KO_CODA = {
    "ㄱ": "K", "ㄲ": "K", "ㄳ": "K", "ㄴ": "N", "ㄵ": "N", "ㄶ": "N", "ㄷ": "T",
    "ㄹ": "L", "ㄺ": "K", "ㄻ": "M", "ㄼ": "L", "ㄽ": "L", "ㄾ": "L", "ㄿ": "P",
    "ㅀ": "L", "ㅁ": "M", "ㅂ": "P", "ㅄ": "P", "ㅅ": "T", "ㅆ": "T", "ㅇ": "NG",
    "ㅈ": "T", "ㅊ": "T", "ㅋ": "K", "ㅌ": "T", "ㅍ": "P", "ㅎ": "T",
}
_CHOSEONG = list(_KO_ONSET)
_JUNGSEONG = list(_KO_NUCLEUS)
_JONGSEONG = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ",
    "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]


class KoreanG2P:
    """g2pk pronounces the hangul, then jamo are mapped onto the checkpoint inventory."""

    def __init__(self, table_path: Path, vocab: dict):
        if not table_path.is_file():
            raise FileNotFoundError(table_path)
        self.vocab = vocab
        from g2pk import G2p

        self._g2p = G2p()

    def phones_for_word(self, word: str) -> tuple[list[str], list[str]]:
        pronounced = self._g2p(clean_word(word) or word)
        known: list[str] = []
        missing: list[str] = []
        for char in pronounced:
            if char.isspace() or not _is_hangul_syllable(char):
                continue
            onset, nucleus, coda = _split_syllable(char)
            onset_phone = _KO_ONSET.get(onset)
            if onset_phone:
                matched = match_vocab(onset_phone, self.vocab)
                if matched is None:
                    missing.append(onset)
                else:
                    known.append(matched)
            nucleus_phones = _realize(_KO_NUCLEUS.get(nucleus, ()), self.vocab)
            if nucleus_phones is None:
                missing.append(nucleus)
            else:
                known.extend(nucleus_phones)
            if coda:
                coda_phone = _KO_CODA.get(coda)
                matched = match_vocab(coda_phone, self.vocab) if coda_phone else None
                if matched is None:
                    missing.append(coda)
                else:
                    known.append(matched)
        return known, missing


def match_vocab(phone: str, vocab: dict) -> str | None:
    for candidate in (phone, phone.upper(), phone.lower()):
        if candidate in vocab and isinstance(candidate, str) and not candidate.startswith("<"):
            return candidate
    return None


def _realize(options: tuple, vocab: dict) -> list[str] | None:
    for option in options:
        matched = [match_vocab(phone, vocab) for phone in option]
        if matched and all(matched):
            return matched
    return None


def _split_syllable(char: str) -> tuple[str, str, str]:
    code = ord(char) - 0xAC00
    return (
        _CHOSEONG[code // 588],
        _JUNGSEONG[(code % 588) // 28],
        _JONGSEONG[code % 28],
    )


def _is_hangul_syllable(char: str) -> bool:
    return len(char) == 1 and 0xAC00 <= ord(char) <= 0xD7A3


def romanize_word(word: str) -> str:
    from uroman import Uroman

    romanizer = _romanizer()
    roman = romanizer.romanize_string(word)
    roman = unicodedata.normalize("NFKD", roman)
    roman = "".join(ch for ch in roman if not unicodedata.combining(ch))
    roman = roman.lower()
    return re.sub(r"[^a-z]", "", roman)


_UROMAN = None


def _romanizer():
    global _UROMAN
    if _UROMAN is None:
        from uroman import Uroman

        _UROMAN = Uroman()
    return _UROMAN
