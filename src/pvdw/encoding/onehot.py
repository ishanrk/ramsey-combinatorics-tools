"""One-hot graph-coloring CNF with safe color-name symmetry breaking."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum

from pysat.card import CardEnc, EncType

from pvdw.encoding.common import (
    ClauseSink,
    CnfStatistics,
    ListClauseSink,
    OneHotVariableMap,
    counting_sink,
)
from pvdw.graph import DistanceGraph
from pvdw.model import InstanceSpec


class AtMostOneEncoding(str, Enum):
    PAIRWISE = "pairwise"
    SEQUENTIAL = "sequential"
    LADDER = "ladder"
    BITWISE = "bitwise"


@dataclass(frozen=True)
class OneHotEncodingResult:
    variables: OneHotVariableMap
    statistics: CnfStatistics
    auxiliary_variables: int
    anchored_clique: tuple[int, ...]
    immediate_noncolorability: bool = False


def _validate_graph(instance: InstanceSpec, graph: DistanceGraph) -> None:
    if graph.n != instance.n:
        raise ValueError("graph and instance vertex counts differ")


def _pairwise_base_clauses(
    instance: InstanceSpec,
    graph: DistanceGraph,
) -> Iterator[list[int]]:
    variables = OneHotVariableMap(instance.n, instance.colors)
    for vertex in range(instance.n):
        yield [variables.var(vertex, color) for color in range(instance.colors)]
        for left in range(instance.colors):
            for right in range(left + 1, instance.colors):
                yield [-variables.var(vertex, left), -variables.var(vertex, right)]
    for u, v in graph.iter_edges():
        for color in range(instance.colors):
            yield [-variables.var(u, color), -variables.var(v, color)]


def iter_onehot_pairwise_clauses(
    instance: InstanceSpec,
    graph: DistanceGraph,
    *,
    fix_first_color: bool = True,
) -> Iterator[list[int]]:
    """Yield the non-auxiliary one-hot formula for streaming DIMACS output."""

    _validate_graph(instance, graph)
    yield from _pairwise_base_clauses(instance, graph)
    if fix_first_color:
        yield [OneHotVariableMap(instance.n, instance.colors).var(0, 0)]


def onehot_pairwise_statistics(
    instance: InstanceSpec,
    graph: DistanceGraph,
    *,
    fix_first_color: bool = True,
) -> CnfStatistics:
    """Compute exact pairwise counts without generating any clauses."""

    _validate_graph(instance, graph)
    colors = instance.colors
    clauses = (
        instance.n
        + instance.n * colors * (colors - 1) // 2
        + graph.edge_count * colors
        + int(fix_first_color)
    )
    literals = (
        instance.n * colors
        + instance.n * colors * (colors - 1)
        + graph.edge_count * colors * 2
        + int(fix_first_color)
    )
    return CnfStatistics(
        variables=instance.n * colors,
        clauses=clauses,
        literals=literals,
        encoding="onehot-pairwise",
    )


def _greedy_clique(
    graph: DistanceGraph,
    limit: int,
    *,
    required_first: int | None = None,
) -> tuple[int, ...]:
    adjacency = [set(neighbors) for neighbors in graph.build_adjacency()]
    starts = [required_first] if required_first is not None else list(range(graph.n))
    best: tuple[int, ...] = ()
    for start in starts:
        clique = [start]
        candidates = set(adjacency[start])
        while candidates and len(clique) < limit:
            vertex = max(
                candidates,
                key=lambda candidate: (
                    len(candidates & adjacency[candidate]),
                    len(adjacency[candidate]),
                    -candidate,
                ),
            )
            clique.append(vertex)
            candidates.intersection_update(adjacency[vertex])
        candidate_clique = tuple(clique)
        if len(candidate_clique) > len(best):
            best = candidate_clique
        if len(best) == limit:
            break
    return best


def _cardinality_encoding(amo: AtMostOneEncoding) -> int:
    return {
        AtMostOneEncoding.SEQUENTIAL: EncType.seqcounter,
        AtMostOneEncoding.LADDER: EncType.ladder,
        AtMostOneEncoding.BITWISE: EncType.bitwise,
    }[amo]


def encode_onehot(
    instance: InstanceSpec,
    graph: DistanceGraph,
    sink: ClauseSink,
    *,
    amo: AtMostOneEncoding,
    fix_first_color: bool = True,
    anchor_clique: bool = False,
) -> OneHotEncodingResult:
    """Encode exact graph coloring with no geometric or periodic assumptions."""

    _validate_graph(instance, graph)
    if not isinstance(amo, AtMostOneEncoding):
        try:
            amo = AtMostOneEncoding(amo)
        except ValueError as error:
            raise ValueError(f"unknown at-most-one encoding: {amo!r}") from error
    variables = OneHotVariableMap(instance.n, instance.colors)
    counted = counting_sink(sink)
    before_clauses = len(sink.clauses) if isinstance(sink, ListClauseSink) else None
    before_literals = (
        sum(len(clause) for clause in sink.clauses)
        if isinstance(sink, ListClauseSink)
        else None
    )

    if anchor_clique:
        obstruction = _greedy_clique(graph, instance.colors + 1)
        if len(obstruction) > instance.colors:
            counted.add_clause([])
            statistics = CnfStatistics(
                variables=variables.primary_variables,
                clauses=1,
                literals=0,
                encoding=f"onehot-{amo.value}",
            )
            return OneHotEncodingResult(
                variables=variables,
                statistics=statistics,
                auxiliary_variables=0,
                anchored_clique=obstruction,
                immediate_noncolorability=True,
            )

    top_id = variables.primary_variables
    if amo is AtMostOneEncoding.PAIRWISE:
        for clause in _pairwise_base_clauses(instance, graph):
            counted.add_clause(clause)
    else:
        encoding = _cardinality_encoding(amo)
        for vertex in range(instance.n):
            color_variables = [
                variables.var(vertex, color) for color in range(instance.colors)
            ]
            counted.add_clause(color_variables)
            cardinality = CardEnc.atmost(
                lits=color_variables,
                bound=1,
                top_id=top_id,
                encoding=encoding,
            )
            for clause in cardinality.clauses:
                counted.add_clause(clause)
            top_id = max(top_id, cardinality.nv)
        for u, v in graph.iter_edges():
            for color in range(instance.colors):
                counted.add_clause(
                    [-variables.var(u, color), -variables.var(v, color)]
                )

    anchored: tuple[int, ...] = ()
    if anchor_clique:
        # If color 0 is also fixed at vertex 0, anchor a clique containing 0.
        # A single global permutation of color names then makes all units safe.
        anchored = _greedy_clique(
            graph,
            instance.colors,
            required_first=0 if fix_first_color else None,
        )
        for color, vertex in enumerate(anchored):
            counted.add_clause([variables.var(vertex, color)])
    if fix_first_color and (not anchored or anchored[0] != 0):
        counted.add_clause([variables.var(0, 0)])

    statistics = CnfStatistics(
        variables=top_id,
        clauses=counted.clauses,
        literals=counted.literals,
        encoding=f"onehot-{amo.value}",
    )
    if isinstance(sink, ListClauseSink):
        assert before_clauses is not None and before_literals is not None
        assert len(sink.clauses) - before_clauses == statistics.clauses
        assert (
            sum(len(clause) for clause in sink.clauses) - before_literals
            == statistics.literals
        )
    return OneHotEncodingResult(
        variables=variables,
        statistics=statistics,
        auxiliary_variables=top_id - variables.primary_variables,
        anchored_clique=anchored,
    )
