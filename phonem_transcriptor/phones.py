"""Place dictionary phonemes onto the letters MMS already timed."""

from __future__ import annotations

import unicodedata

VOWEL_LETTERS = set("aeiouy")
VOWEL_PHONE_START = set("aeiouyɑɐəɛɜɪɔʊɨʉɯɤøœæɶʌ")


def is_vowel_letter(letter: str) -> bool:
    return letter.lower() in VOWEL_LETTERS


def is_vowel_phone(phone: str) -> bool:
    base = "".join(ch for ch in phone.lower() if not unicodedata.combining(ch))
    if not base:
        return False
    return base[0] in VOWEL_PHONE_START


def _same_class(letter: str, phone: str) -> bool:
    return is_vowel_letter(letter) == is_vowel_phone(phone)


def _align_indices(letters: list[str], phones: list[str]) -> list[list[int]]:
    """Return, for each phone, the letter indexes it covers. Indexes may be shared."""
    count_letters = len(letters)
    count_phones = len(phones)
    inf = 10**9
    cost = [[inf] * (count_phones + 1) for _ in range(count_letters + 1)]
    previous: list[list[tuple[int, int, str] | None]] = [
        [None] * (count_phones + 1) for _ in range(count_letters + 1)
    ]
    cost[0][0] = 0

    for letter_index in range(count_letters + 1):
        for phone_index in range(count_phones + 1):
            here = cost[letter_index][phone_index]
            if here >= inf:
                continue
            if letter_index < count_letters and phone_index < count_phones:
                step = 0 if _same_class(letters[letter_index], phones[phone_index]) else 2
                _relax(
                    cost,
                    previous,
                    letter_index + 1,
                    phone_index + 1,
                    here + step,
                    (letter_index, phone_index, "both"),
                )
            if letter_index < count_letters and phone_index > 0:
                step = (
                    0.25
                    if _same_class(letters[letter_index], phones[phone_index - 1])
                    else 1.5
                )
                _relax(
                    cost,
                    previous,
                    letter_index + 1,
                    phone_index,
                    here + step,
                    (letter_index, phone_index, "letter"),
                )
            if phone_index < count_phones and letter_index > 0:
                step = (
                    0.5
                    if _same_class(letters[letter_index - 1], phones[phone_index])
                    else 1.75
                )
                _relax(
                    cost,
                    previous,
                    letter_index,
                    phone_index + 1,
                    here + step,
                    (letter_index, phone_index, "phone"),
                )

    if cost[count_letters][count_phones] >= inf:
        return [[min(index, count_letters - 1)] for index in range(count_phones)]

    operations: list[str] = []
    letter_index = count_letters
    phone_index = count_phones
    while letter_index or phone_index:
        step = previous[letter_index][phone_index]
        if step is None:
            break
        previous_letter, previous_phone, operation = step
        operations.append(operation)
        letter_index = previous_letter
        phone_index = previous_phone
    operations.reverse()

    covered: list[list[int]] = [[] for _ in phones]
    last_letter = 0
    last_phone = -1
    letter_cursor = 0
    phone_cursor = 0
    for operation in operations:
        if operation == "both":
            covered[phone_cursor].append(letter_cursor)
            last_letter = letter_cursor
            last_phone = phone_cursor
            letter_cursor += 1
            phone_cursor += 1
        elif operation == "letter":
            if last_phone >= 0:
                covered[last_phone].append(letter_cursor)
            last_letter = letter_cursor
            letter_cursor += 1
        else:
            covered[phone_cursor].append(last_letter)
            last_phone = phone_cursor
            phone_cursor += 1
    return covered


def _relax(cost, previous, letter_index, phone_index, candidate, pointer) -> None:
    if candidate < cost[letter_index][phone_index]:
        cost[letter_index][phone_index] = candidate
        previous[letter_index][phone_index] = pointer


def assign_phone_times(
    letters: str,
    spans: list[tuple[float, float]],
    phones: list[str],
) -> list[tuple[str, float, float]]:
    """Each phoneme inherits the interval of the letter or letters that realize it.

    A vowel written with two letters keeps the whole span. Two phonemes forced
    onto one letter split that letter only because there is no other boundary.
    """
    if not phones or not letters or len(spans) != len(letters):
        return []
    if len(letters) == len(phones):
        return [
            (phone, spans[index][0], spans[index][1])
            for index, phone in enumerate(phones)
        ]

    covered = _align_indices(list(letters), phones)
    raw: list[list] = []
    for indexes in covered:
        if not indexes:
            raw.append([])
            continue
        start = spans[indexes[0]][0]
        end = spans[indexes[-1]][1]
        raw.append([start, end, tuple(indexes)])

    cursor = 0
    while cursor < len(raw):
        if len(raw[cursor]) != 3:
            cursor += 1
            continue
        end_group = cursor
        while (
            end_group + 1 < len(raw)
            and len(raw[end_group + 1]) == 3
            and raw[end_group + 1][2] == raw[cursor][2]
            and len(raw[cursor][2]) == 1
        ):
            end_group += 1
        if end_group > cursor:
            start = raw[cursor][0]
            end = raw[cursor][1]
            width = (end - start) / (end_group - cursor + 1)
            for offset, index in enumerate(range(cursor, end_group + 1)):
                raw[index][0] = start + offset * width
                raw[index][1] = start + (offset + 1) * width
        cursor = end_group + 1

    timed = []
    for phone, item in zip(phones, raw):
        if len(item) != 3:
            continue
        timed.append((phone, float(item[0]), float(item[1])))
    return timed
