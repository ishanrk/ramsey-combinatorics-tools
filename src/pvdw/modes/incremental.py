"""Monotone incremental scans over the interval size N."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from pysat.card import CardEnc, EncType

from pvdw.backends.base import DecodeSpec, EncodedProblem, SolveOptions
from pvdw.distances import generate_distances
from pvdw.encoding.binary import bits_for_colors
from pvdw.encoding.common import BinaryVariableMap, OneHotVariableMap
from pvdw.encoding.onehot import AtMostOneEncoding
from pvdw.model import InstanceSpec, ModelScope, PolynomialSpec, SolveStatus
from pvdw.modes.direct import EncodingOptions


class IncrementalBackend(Protocol):
    name: str
    capabilities: object

    def add_clause(self, clause: list[int] | tuple[int, ...]) -> None:
        ...

    def solve_current(self, options: SolveOptions) -> bool | None:
        ...

    def get_model(self) -> list[int] | None:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class IncrementalStep:
    n: int
    status: SolveStatus
    elapsed_seconds: float
    added_clauses: int
    cumulative_clauses: int
    coloring: tuple[int, ...] | None


@dataclass(frozen=True)
class IncrementalScanResult:
    steps: tuple[IncrementalStep, ...]
    last_verified_coloring: tuple[int, ...] | None
    first_unsat_n: int | None
    cumulative_clauses: int


def _card_encoding(amo: AtMostOneEncoding) -> int:
    return {
        AtMostOneEncoding.SEQUENTIAL: EncType.seqcounter,
        AtMostOneEncoding.LADDER: EncType.ladder,
        AtMostOneEncoding.BITWISE: EncType.bitwise,
    }[amo]


def scan_incremental(
    polynomial: PolynomialSpec,
    colors: int,
    n_min: int,
    n_max: int,
    backend: IncrementalBackend,
    encoding_options: EncodingOptions,
    solve_options: SolveOptions | None = None,
    *,
    input_domain: str = "all_nonzero",
) -> IncrementalScanResult:
    """Reuse one complete incremental solver and all learned clauses across N."""

    if not 1 <= n_min <= n_max:
        raise ValueError("incremental N range is invalid")
    capabilities = backend.capabilities
    if not getattr(capabilities, "complete", False) or not getattr(
        capabilities, "incremental", False
    ):
        raise ValueError("incremental scanning requires an incremental complete backend")
    options = solve_options or SolveOptions()
    maximum_instance = InstanceSpec(
        polynomial, colors, n_max, input_domain  # type: ignore[arg-type]
    )
    distances = generate_distances(maximum_instance).values
    clauses: list[tuple[int, ...]] = []
    steps: list[IncrementalStep] = []
    last_coloring: tuple[int, ...] | None = None
    first_unsat: int | None = None
    top_id = n_max * colors
    onehot_variables = OneHotVariableMap(n_max, colors)
    bits = bits_for_colors(colors)
    binary_variables = BinaryVariableMap(n_max, bits)

    def add(clause: list[int] | tuple[int, ...]) -> None:
        materialized = tuple(clause)
        backend.add_clause(materialized)
        clauses.append(materialized)

    try:
        for vertex in range(n_max):
            before = len(clauses)
            if encoding_options.encoding == "onehot":
                color_variables = [
                    onehot_variables.var(vertex, color) for color in range(colors)
                ]
                add(color_variables)
                if encoding_options.amo is AtMostOneEncoding.PAIRWISE:
                    for left in range(colors):
                        for right in range(left + 1, colors):
                            add(
                                [
                                    -onehot_variables.var(vertex, left),
                                    -onehot_variables.var(vertex, right),
                                ]
                            )
                else:
                    encoded = CardEnc.atmost(
                        lits=color_variables,
                        bound=1,
                        top_id=top_id,
                        encoding=_card_encoding(encoding_options.amo),
                    )
                    for clause in encoded.clauses:
                        add(clause)
                    top_id = max(top_id, encoded.nv)
                for distance in distances:
                    if distance > vertex:
                        break
                    other = vertex - distance
                    for color in range(colors):
                        add(
                            [
                                -onehot_variables.var(other, color),
                                -onehot_variables.var(vertex, color),
                            ]
                        )
                if vertex == 0 and encoding_options.fix_first_color:
                    add([onehot_variables.var(0, 0)])
            else:
                for invalid_code in range(colors, 1 << bits):
                    add(
                        [
                            -binary_variables.var(vertex, bit)
                            if (invalid_code >> bit) & 1
                            else binary_variables.var(vertex, bit)
                            for bit in range(bits)
                        ]
                    )
                for distance in distances:
                    if distance > vertex:
                        break
                    other = vertex - distance
                    for color in range(colors):
                        clause: list[int] = []
                        for endpoint in (other, vertex):
                            clause.extend(
                                -binary_variables.var(endpoint, bit)
                                if (color >> bit) & 1
                                else binary_variables.var(endpoint, bit)
                                for bit in range(bits)
                            )
                        add(clause)
                if vertex == 0 and encoding_options.fix_first_color:
                    for bit in range(bits):
                        add([-binary_variables.var(0, bit)])
            n = vertex + 1
            if n < n_min:
                continue
            started = time.perf_counter()
            solved = backend.solve_current(options)
            elapsed = time.perf_counter() - started
            coloring: tuple[int, ...] | None = None
            if solved is True:
                model = backend.get_model()
                if model is None:
                    status = SolveStatus.ERROR
                else:
                    current_instance = InstanceSpec(
                        polynomial, colors, n, input_domain  # type: ignore[arg-type]
                    )
                    variable_count = max(
                        (abs(literal) for clause in clauses for literal in clause),
                        default=0,
                    )
                    problem = EncodedProblem(
                        current_instance,
                        variable_count,
                        tuple(clauses),
                        sum(len(clause) for clause in clauses),
                        (
                            "binary"
                            if encoding_options.encoding == "binary"
                            else f"onehot-{encoding_options.amo.value}"
                        ),
                        ModelScope.FULL,
                        DecodeSpec(
                            (
                                "binary"
                                if encoding_options.encoding == "binary"
                                else f"onehot-{encoding_options.amo.value}"
                            ),
                            n,
                            colors,
                        ),
                    )
                    try:
                        coloring = problem.decode_model(model)
                    except ValueError:
                        status = SolveStatus.ERROR
                    else:
                        status = (
                            SolveStatus.FOUND_WITNESS
                            if problem.verify(coloring).valid
                            else SolveStatus.ERROR
                        )
                        if status is SolveStatus.FOUND_WITNESS:
                            last_coloring = coloring
            elif solved is False:
                status = SolveStatus.UNSAT_FULL_MODEL
                first_unsat = n
            else:
                status = SolveStatus.TIMEOUT
            steps.append(
                IncrementalStep(
                    n,
                    status,
                    elapsed,
                    len(clauses) - before,
                    len(clauses),
                    coloring,
                )
            )
            if status in (
                SolveStatus.UNSAT_FULL_MODEL,
                SolveStatus.TIMEOUT,
                SolveStatus.ERROR,
            ):
                break
    finally:
        backend.close()
    return IncrementalScanResult(
        tuple(steps), last_coloring, first_unsat, len(clauses)
    )
