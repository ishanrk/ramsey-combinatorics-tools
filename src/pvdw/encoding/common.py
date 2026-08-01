"""Clause sinks, statistics, and stable CNF variable maps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol


class ClauseSink(Protocol):
    def add_clause(self, clause: Sequence[int]) -> None:
        """Consume one DIMACS clause without retaining it necessarily."""


@dataclass
class ListClauseSink:
    clauses: list[list[int]] = field(default_factory=list)

    def add_clause(self, clause: Sequence[int]) -> None:
        self.clauses.append(list(clause))


@dataclass(frozen=True)
class CnfStatistics:
    variables: int
    clauses: int
    literals: int
    encoding: str


@dataclass(frozen=True)
class OneHotVariableMap:
    n: int
    colors: int

    def __post_init__(self) -> None:
        if self.n < 1 or self.colors < 2:
            raise ValueError("one-hot maps require n >= 1 and colors >= 2")

    @property
    def primary_variables(self) -> int:
        return self.n * self.colors

    @property
    def first_auxiliary(self) -> int:
        return self.primary_variables + 1

    def var(self, vertex: int, color: int) -> int:
        if not 0 <= vertex < self.n:
            raise IndexError("vertex is outside the variable map")
        if not 0 <= color < self.colors:
            raise IndexError("color is outside the variable map")
        return vertex * self.colors + color + 1


@dataclass(frozen=True)
class BinaryVariableMap:
    n: int
    bits: int

    def __post_init__(self) -> None:
        if self.n < 1 or self.bits < 1:
            raise ValueError("binary maps require n >= 1 and bits >= 1")

    @property
    def variables(self) -> int:
        return self.n * self.bits

    def var(self, vertex: int, bit: int) -> int:
        if not 0 <= vertex < self.n:
            raise IndexError("vertex is outside the variable map")
        if not 0 <= bit < self.bits:
            raise IndexError("bit is outside the variable map")
        return vertex * self.bits + bit + 1


class _CountingSink:
    """Internal forwarding sink used for exact accounting."""

    def __init__(self, target: ClauseSink) -> None:
        self.target = target
        self.clauses = 0
        self.literals = 0

    def add_clause(self, clause: Sequence[int]) -> None:
        materialized = list(clause)
        if any(type(literal) is not int or literal == 0 for literal in materialized):
            raise ValueError("clauses require nonzero ordinary integer literals")
        self.target.add_clause(materialized)
        self.clauses += 1
        self.literals += len(materialized)


def counting_sink(target: ClauseSink) -> _CountingSink:
    return _CountingSink(target)
