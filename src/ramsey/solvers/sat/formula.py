from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CNFFormula:
    """an immutable boolean formula written as dimacs-style clauses."""

    variable_count: int
    clauses: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if type(self.variable_count) is not int:
            raise TypeError("variable_count must be an ordinary integer")
        if self.variable_count < 0:
            raise ValueError("variable_count must be nonnegative")

        normalized: list[tuple[int, ...]] = []
        for clause in self.clauses:
            saved_clause: list[int] = []
            seen: set[int] = set()
            tautology = False
            for literal in clause:
                if type(literal) is not int:
                    raise TypeError("cnf literals must be ordinary integers")
                if literal == 0:
                    raise ValueError("zero is not a cnf literal")
                if abs(literal) > self.variable_count:
                    raise ValueError("a literal refers to an unknown variable")
                if -literal in seen:
                    tautology = True
                    break
                if literal not in seen:
                    seen.add(literal)
                    saved_clause.append(literal)
            if not tautology:
                normalized.append(tuple(saved_clause))
        object.__setattr__(self, "clauses", tuple(normalized))

    @classmethod
    def from_clauses(
        cls,
        variable_count: int,
        clauses: Iterable[Sequence[int]],
    ) -> CNFFormula:
        return cls(
            variable_count=variable_count,
            clauses=tuple(tuple(clause) for clause in clauses),
        )

    @property
    def clause_count(self) -> int:
        return len(self.clauses)

    @property
    def literal_count(self) -> int:
        return sum(len(clause) for clause in self.clauses)

    def is_satisfied_by(self, model: Iterable[int]) -> bool:
        """check a dimacs-style model against every clause."""
        values: dict[int, bool] = {}
        for literal in model:
            if type(literal) is not int:
                raise TypeError("model literals must be ordinary integers")
            variable = abs(literal)
            if literal == 0 or variable > self.variable_count:
                raise ValueError("the model contains an invalid literal")
            value = literal > 0
            if variable in values and values[variable] != value:
                raise ValueError("the model assigns a variable twice")
            values[variable] = value
        return all(
            any(values.get(abs(literal)) is (literal > 0) for literal in clause)
            for clause in self.clauses
        )


__all__ = ["CNFFormula"]
