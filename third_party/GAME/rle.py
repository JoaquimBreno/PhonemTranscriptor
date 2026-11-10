from __future__ import annotations

from typing import Iterable, TypeVar

T = TypeVar("T")


def encode(sequence: Iterable[T]) -> tuple[list[T], list[int]]:
    values: list[T] = []
    lengths: list[int] = []

    iterator = iter(sequence)
    try:
        current = next(iterator)
    except StopIteration:
        return values, lengths

    run_length = 1
    for item in iterator:
        if item == current:
            run_length += 1
        else:
            values.append(current)
            lengths.append(run_length)
            current = item
            run_length = 1

    values.append(current)
    lengths.append(run_length)
    return values, lengths
