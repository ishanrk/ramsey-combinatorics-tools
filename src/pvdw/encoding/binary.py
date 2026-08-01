"""Compact binary-code CNF for exact graph coloring."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from pvdw.encoding.common import (
    BinaryVariableMap,
    ClauseSink,
    CnfStatistics,
    ListClauseSink,
    counting_sink,
)
from pvdw.graph import DistanceGraph
from pvdw.model import InstanceSpec


@dataclass(frozen=True)
class BinaryEncodingResult:
    variables: BinaryVariableMap
    statistics: CnfStatistics


def bits_for_colors(colors: int) -> int:
    if type(colors) is not int or colors < 2:
        raise ValueError("colors must be an ordinary integer at least 2")
    return (colors - 1).bit_length()


def _different_from_code(
    variables: BinaryVariableMap,
    vertex: int,
    code: int,
) -> list[int]:
    return [
        -variables.var(vertex, bit) if (code >> bit) & 1 else variables.var(vertex, bit)
        for bit in range(variables.bits)
    ]


def iter_binary_clauses(
    instance: InstanceSpec,
    graph: DistanceGraph,
    *,
    fix_first_color: bool = True,
) -> Iterator[list[int]]:
    """Yield binary clauses in a deterministic, streaming order."""

    if graph.n != instance.n:
        raise ValueError("graph and instance vertex counts differ")
    variables = BinaryVariableMap(instance.n, bits_for_colors(instance.colors))
    for vertex in range(instance.n):
        for invalid_code in range(instance.colors, 1 << variables.bits):
            yield _different_from_code(variables, vertex, invalid_code)
    for u, v in graph.iter_edges():
        for color in range(instance.colors):
            yield _different_from_code(variables, u, color) + _different_from_code(
                variables, v, color
            )
    if fix_first_color:
        for bit in range(variables.bits):
            yield [-variables.var(0, bit)]


def binary_statistics(
    instance: InstanceSpec,
    graph: DistanceGraph,
    *,
    fix_first_color: bool = True,
) -> CnfStatistics:
    if graph.n != instance.n:
        raise ValueError("graph and instance vertex counts differ")
    bits = bits_for_colors(instance.colors)
    invalid = instance.n * ((1 << bits) - instance.colors)
    edge_clauses = graph.edge_count * instance.colors
    symmetry = bits if fix_first_color else 0
    clauses = invalid + edge_clauses + symmetry
    literals = invalid * bits + edge_clauses * (2 * bits) + symmetry
    return CnfStatistics(
        variables=instance.n * bits,
        clauses=clauses,
        literals=literals,
        encoding="binary",
    )


def encode_binary(
    instance: InstanceSpec,
    graph: DistanceGraph,
    sink: ClauseSink,
    *,
    fix_first_color: bool = True,
) -> BinaryEncodingResult:
    """Encode valid color codes and inequality at every graph edge."""

    expected = binary_statistics(
        instance, graph, fix_first_color=fix_first_color
    )
    variables = BinaryVariableMap(instance.n, bits_for_colors(instance.colors))
    counted = counting_sink(sink)
    before_clauses = len(sink.clauses) if isinstance(sink, ListClauseSink) else None
    before_literals = (
        sum(len(clause) for clause in sink.clauses)
        if isinstance(sink, ListClauseSink)
        else None
    )
    for clause in iter_binary_clauses(
        instance, graph, fix_first_color=fix_first_color
    ):
        counted.add_clause(clause)
    assert counted.clauses == expected.clauses
    assert counted.literals == expected.literals
    if isinstance(sink, ListClauseSink):
        assert before_clauses is not None and before_literals is not None
        assert len(sink.clauses) - before_clauses == expected.clauses
        assert sum(len(clause) for clause in sink.clauses) - before_literals == expected.literals
    return BinaryEncodingResult(variables=variables, statistics=expected)


def decode_binary_model(
    model: Iterable[int],
    instance: InstanceSpec,
) -> tuple[int, ...]:
    """Decode a complete signed assignment, rejecting omissions and bad codes."""

    variables = BinaryVariableMap(instance.n, bits_for_colors(instance.colors))
    assignments: dict[int, bool] = {}
    for literal in model:
        if type(literal) is not int or literal == 0:
            raise ValueError("model literals must be nonzero ordinary integers")
        variable = abs(literal)
        if not 1 <= variable <= variables.variables:
            raise ValueError(f"model variable {variable} is outside the binary map")
        if variable in assignments:
            raise ValueError(f"model assigns variable {variable} more than once")
        assignments[variable] = literal > 0
    if len(assignments) != variables.variables:
        missing = sorted(set(range(1, variables.variables + 1)) - assignments.keys())
        raise ValueError(f"model omits binary variables: {missing}")
    coloring: list[int] = []
    for vertex in range(instance.n):
        code = sum(
            int(assignments[variables.var(vertex, bit)]) << bit
            for bit in range(variables.bits)
        )
        if code >= instance.colors:
            raise ValueError(f"vertex {vertex} has invalid color code {code}")
        coloring.append(code)
    return tuple(coloring)
