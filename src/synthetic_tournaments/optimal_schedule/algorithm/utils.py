from __future__ import annotations

from abc import abstractmethod
from typing import Generic, Sequence, TypeVar

T = TypeVar("T")

Round = tuple[tuple[int, int], ...]


class Summable(Generic[T]):
    @abstractmethod
    def __add__(self, other: Summable) -> Summable:
        ...


def sum_sequence_values(sequences: Sequence[Summable[T]]) -> Summable[T]:
    first, *others = sequences
    for other in others:
        first += other
    return first


def split_in_middle(sequence: Sequence[T]) -> tuple[Sequence[T], Sequence[T]]:
    """
    Split a sequence in the middle: len(sequence) // 2.
    """
    midpoint = len(sequence) // 2
    return sequence[:midpoint], sequence[midpoint:]
