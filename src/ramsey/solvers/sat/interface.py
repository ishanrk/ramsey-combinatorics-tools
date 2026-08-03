from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from time import perf_counter

from ramsey.solvers.sat.formula import CNFFormula
from ramsey.solvers.sat.model import SatResult, SatStatus
from ramsey.solvers.sat.pysat_backend import (
    PYSAT_BACKENDS,
    pysat_available,
    solve_with_pysat,
)

BUILTIN_BACKENDS = {"builtin", "dpll"}


def _normalize_assumptions(
    formula: CNFFormula,
    assumptions: Iterable[int],
) -> tuple[int, ...]:
    normalized = tuple(assumptions)
    for literal in normalized:
        if type(literal) is not int:
            raise TypeError("assumptions must be ordinary integer literals")
        if literal == 0 or abs(literal) > formula.variable_count:
            raise ValueError("an assumption contains an invalid literal")
    return normalized


def _assign_literal(
    clauses: tuple[tuple[int, ...], ...],
    assignment: list[int],
    literal: int,
) -> tuple[tuple[int, ...], ...] | None:
    variable = abs(literal)
    value = 1 if literal > 0 else -1
    if assignment[variable] == -value:
        return None
    if assignment[variable] == value:
        return clauses
    assignment[variable] = value

    reduced: list[tuple[int, ...]] = []
    opposite = -literal
    for clause in clauses:
        if literal in clause:
            continue
        if opposite not in clause:
            reduced.append(clause)
            continue
        new_clause = tuple(item for item in clause if item != opposite)
        if not new_clause:
            return None
        reduced.append(new_clause)
    return tuple(reduced)


def _propagate(
    clauses: tuple[tuple[int, ...], ...],
    assignment: list[int],
) -> tuple[tuple[int, ...], ...] | None:
    current = clauses
    while True:
        unit = next((clause[0] for clause in current if len(clause) == 1), None)
        if unit is not None:
            current = _assign_literal(current, assignment, unit)
            if current is None:
                return None
            continue

        signs: dict[int, int] = {}
        for clause in current:
            for literal in clause:
                variable = abs(literal)
                sign = 1 if literal > 0 else -1
                previous = signs.get(variable, 0)
                if previous == 0:
                    signs[variable] = sign
                elif previous != sign:
                    signs[variable] = 2
        pure_literal = next(
            (
                variable if sign == 1 else -variable
                for variable, sign in signs.items()
                if sign in (-1, 1)
            ),
            None,
        )
        if pure_literal is None:
            return current
        current = _assign_literal(current, assignment, pure_literal)
        if current is None:
            return None


def _choose_literal(clauses: tuple[tuple[int, ...], ...]) -> int:
    shortest_size = min(map(len, clauses))
    shortest = (clause for clause in clauses if len(clause) == shortest_size)
    counts = Counter(abs(literal) for clause in shortest for literal in clause)
    variable = min(
        counts,
        key=lambda item: (-counts[item], item),
    )
    positive = sum(clause.count(variable) for clause in clauses)
    negative = sum(clause.count(-variable) for clause in clauses)
    return variable if positive >= negative else -variable


def _dpll(
    clauses: tuple[tuple[int, ...], ...],
    assignment: list[int],
) -> list[int] | None:
    current = _propagate(clauses, assignment)
    if current is None:
        return None
    if not current:
        return assignment

    literal = _choose_literal(current)
    for choice in (literal, -literal):
        branch_assignment = assignment.copy()
        branch_clauses = _assign_literal(current, branch_assignment, choice)
        if branch_clauses is None:
            continue
        result = _dpll(branch_clauses, branch_assignment)
        if result is not None:
            return result
    return None


def _solve_builtin(
    formula: CNFFormula,
    *,
    backend: str,
    assumptions: tuple[int, ...],
) -> SatResult:
    started = perf_counter()
    clauses = formula.clauses + tuple((literal,) for literal in assumptions)
    if any(not clause for clause in clauses):
        return SatResult(
            status=SatStatus.UNSAT,
            backend=backend,
            model=None,
            elapsed_seconds=perf_counter() - started,
        )
    assignment = [0] * (formula.variable_count + 1)
    try:
        solved = _dpll(clauses, assignment)
    except RecursionError:
        return SatResult(
            status=SatStatus.ERROR,
            backend=backend,
            model=None,
            elapsed_seconds=perf_counter() - started,
            message="the builtin dpll recursion limit was reached",
        )
    if solved is None:
        return SatResult(
            status=SatStatus.UNSAT,
            backend=backend,
            model=None,
            elapsed_seconds=perf_counter() - started,
        )
    model = tuple(
        variable if solved[variable] == 1 else -variable
        for variable in range(1, formula.variable_count + 1)
    )
    return SatResult(
        status=SatStatus.SAT,
        backend=backend,
        model=model,
        elapsed_seconds=perf_counter() - started,
    )


def available_sat_backends() -> tuple[str, ...]:
    backends = ["dpll", "builtin"]
    if pysat_available():
        backends.extend(sorted(PYSAT_BACKENDS))
    return tuple(backends)


def solve_cnf(
    formula: CNFFormula,
    *,
    backend: str = "dpll",
    assumptions: Iterable[int] = (),
) -> SatResult:
    """solve a cnf formula with the selected registered backend."""
    if not isinstance(formula, CNFFormula):
        raise TypeError("formula must be a CNFFormula")
    if not isinstance(backend, str):
        raise TypeError("backend must be a string")
    backend = backend.lower()
    normalized_assumptions = _normalize_assumptions(formula, assumptions)
    if backend in BUILTIN_BACKENDS:
        return _solve_builtin(
            formula,
            backend=backend,
            assumptions=normalized_assumptions,
        )
    if backend in PYSAT_BACKENDS:
        return solve_with_pysat(
            formula,
            backend=backend,
            assumptions=normalized_assumptions,
        )
    choices = sorted(BUILTIN_BACKENDS | set(PYSAT_BACKENDS))
    raise ValueError(f"unknown sat backend {backend!r}; choose from {choices}")


__all__ = ["available_sat_backends", "solve_cnf"]
