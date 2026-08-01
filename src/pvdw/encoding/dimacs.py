"""Streaming DIMACS writing plus strict DIMACS and SAT-model parsing."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DimacsFormula:
    variable_count: int
    clauses: tuple[tuple[int, ...], ...]

    @property
    def clause_count(self) -> int:
        return len(self.clauses)


def _validate_clause(clause: Sequence[int], variable_count: int) -> list[int]:
    materialized = list(clause)
    for literal in materialized:
        if type(literal) is not int or literal == 0:
            raise ValueError("DIMACS clauses require nonzero ordinary integer literals")
        if abs(literal) > variable_count:
            raise ValueError(
                f"literal {literal} exceeds declared variable count {variable_count}"
            )
    return materialized


def _write_clause(handle: object, clause: Sequence[int]) -> None:
    line = " ".join(str(literal) for literal in clause)
    handle.write(f"{line} 0\n" if line else "0\n")  # type: ignore[attr-defined]


def write_dimacs(
    path: str | os.PathLike[str],
    variable_count: int,
    clauses: Iterable[Sequence[int]],
    *,
    clause_count: int | None = None,
) -> None:
    """Write DIMACS, streaming directly when the clause count is known.

    If ``clause_count`` is omitted for a one-shot iterable, clauses are first
    streamed to a temporary body file so the final header remains correct.
    """

    if type(variable_count) is not int or variable_count < 0:
        raise ValueError("variable_count must be a nonnegative ordinary integer")
    output = Path(path)
    if clause_count is None and isinstance(clauses, Sequence):
        clause_count = len(clauses)
    if clause_count is not None:
        if type(clause_count) is not int or clause_count < 0:
            raise ValueError("clause_count must be a nonnegative ordinary integer")
        actual = 0
        try:
            with output.open("w", encoding="ascii", newline="\n") as handle:
                handle.write(f"p cnf {variable_count} {clause_count}\n")
                for clause in clauses:
                    _write_clause(handle, _validate_clause(clause, variable_count))
                    actual += 1
            if actual != clause_count:
                raise ValueError(
                    f"generated {actual} clauses, expected {clause_count}"
                )
        except Exception:
            output.unlink(missing_ok=True)
            raise
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="ascii",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".body",
            delete=False,
        ) as body:
            temporary_name = body.name
            actual = 0
            for clause in clauses:
                _write_clause(body, _validate_clause(clause, variable_count))
                actual += 1
        with output.open("w", encoding="ascii", newline="\n") as handle:
            handle.write(f"p cnf {variable_count} {actual}\n")
            assert temporary_name is not None
            with open(temporary_name, "r", encoding="ascii") as body:
                for chunk in iter(lambda: body.read(1024 * 1024), ""):
                    handle.write(chunk)
    except Exception:
        output.unlink(missing_ok=True)
        raise
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def parse_dimacs(path: str | os.PathLike[str]) -> DimacsFormula:
    """Parse a DIMACS CNF file, including clauses split across lines."""

    variable_count: int | None = None
    declared_clauses: int | None = None
    clauses: list[tuple[int, ...]] = []
    pending: list[int] = []
    with Path(path).open("r", encoding="ascii") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p"):
                if variable_count is not None or pending or clauses:
                    raise ValueError(f"misplaced or duplicate header on line {line_number}")
                fields = line.split()
                if len(fields) != 4 or fields[:2] != ["p", "cnf"]:
                    raise ValueError(f"invalid DIMACS header on line {line_number}")
                try:
                    variable_count = int(fields[2])
                    declared_clauses = int(fields[3])
                except ValueError as error:
                    raise ValueError("DIMACS header counts must be integers") from error
                if variable_count < 0 or declared_clauses < 0:
                    raise ValueError("DIMACS header counts must be nonnegative")
                continue
            if variable_count is None:
                raise ValueError("DIMACS clauses precede the header")
            for token in line.split():
                try:
                    literal = int(token)
                except ValueError as error:
                    raise ValueError(
                        f"invalid DIMACS token {token!r} on line {line_number}"
                    ) from error
                if literal == 0:
                    clauses.append(tuple(pending))
                    pending.clear()
                else:
                    if abs(literal) > variable_count:
                        raise ValueError(
                            f"literal {literal} exceeds declared variable count"
                        )
                    pending.append(literal)
    if variable_count is None or declared_clauses is None:
        raise ValueError("DIMACS header is missing")
    if pending:
        raise ValueError("last DIMACS clause has no terminating zero")
    if len(clauses) != declared_clauses:
        raise ValueError(
            f"DIMACS declares {declared_clauses} clauses but contains {len(clauses)}"
        )
    return DimacsFormula(variable_count=variable_count, clauses=tuple(clauses))


def parse_sat_model(text: str) -> tuple[int, ...] | None:
    """Parse conventional ``s`` and multi-line ``v`` SAT solver output."""

    if not isinstance(text, str):
        raise TypeError("SAT model text must be a string")
    status: bool | None = None
    model: list[int] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("c"):
            continue
        fields = line.split()
        if fields[0] == "s":
            normalized = " ".join(fields[1:]).upper()
            if "UNSATISFIABLE" in normalized:
                current = False
            elif "SATISFIABLE" in normalized:
                current = True
            else:
                raise ValueError(f"unrecognized SAT status on line {line_number}")
            if status is not None and status != current:
                raise ValueError("solver output contains contradictory statuses")
            status = current
        elif fields[0] == "v":
            for token in fields[1:]:
                try:
                    literal = int(token)
                except ValueError as error:
                    raise ValueError(
                        f"invalid model literal {token!r} on line {line_number}"
                    ) from error
                if literal != 0:
                    model.append(literal)
    if status is False:
        if model:
            raise ValueError("UNSATISFIABLE output must not contain a model")
        return None
    if status is None and not model:
        raise ValueError("solver output contains neither status nor model")
    return tuple(model)
