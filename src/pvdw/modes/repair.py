"""Repair of positive-energy structured backbones with complete reduced SAT."""

from __future__ import annotations

from dataclasses import replace

from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

from pvdw.backends.base import DecodeSpec, EncodedProblem, ExplicitGraph, SearchBackend, SolveOptions
from pvdw.distances import generate_distances
from pvdw.encoding.common import OneHotVariableMap
from pvdw.graph import DistanceGraph
from pvdw.model import InstanceSpec, ModelScope, SolveResult, SolveStatus
from pvdw.modes.direct import EncodingOptions, encode_graph_problem
from pvdw.verify import conflict_count, verify_coloring


def find_bad_edges(
    graph: DistanceGraph | ExplicitGraph,
    coloring: tuple[int, ...] | list[int],
) -> tuple[tuple[int, int], ...]:
    if len(coloring) != graph.n:
        raise ValueError("backbone coloring length differs from graph size")
    return tuple(
        (u, v) for u, v in graph.iter_edges() if coloring[u] == coloring[v]
    )


def greedy_bad_edge_vertex_cover(
    bad_edges: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> tuple[int, ...]:
    remaining = set(bad_edges)
    cover: list[int] = []
    while remaining:
        counts: dict[int, int] = {}
        for u, v in remaining:
            counts[u] = counts.get(u, 0) + 1
            counts[v] = counts.get(v, 0) + 1
        vertex = max(counts, key=lambda candidate: (counts[candidate], -candidate))
        cover.append(vertex)
        remaining = {edge for edge in remaining if vertex not in edge}
    return tuple(sorted(cover))


def exact_bad_edge_vertex_cover(
    bad_edges: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> tuple[int, ...]:
    vertices = sorted({vertex for edge in bad_edges for vertex in edge})
    if not vertices:
        return ()
    variable = {vertex: index + 1 for index, vertex in enumerate(vertices)}
    formula = WCNF()
    for u, v in bad_edges:
        formula.append([variable[u], variable[v]])
    for vertex in vertices:
        formula.append([-variable[vertex]], weight=1)
    with RC2(formula) as solver:
        model = solver.compute()
    selected = {literal for literal in model if literal > 0}
    return tuple(vertex for vertex in vertices if variable[vertex] in selected)


def expand_editable_set(
    graph: DistanceGraph | ExplicitGraph,
    editable_vertices: tuple[int, ...] | set[int],
) -> tuple[int, ...]:
    expanded = set(editable_vertices)
    adjacency = graph.build_adjacency()
    for vertex in tuple(expanded):
        expanded.update(adjacency[vertex])
    return tuple(sorted(expanded))


def build_repair_cnf(
    instance: InstanceSpec,
    graph: DistanceGraph,
    backbone: tuple[int, ...] | list[int],
    editable_vertices: tuple[int, ...] | set[int],
    encoding_options: EncodingOptions | None = None,
) -> EncodedProblem:
    """Encode only editable vertices, with frozen-neighbor unit forbids."""

    backbone_tuple = tuple(backbone)
    if len(backbone_tuple) != instance.n:
        raise ValueError("repair backbone has the wrong length")
    editable = tuple(sorted(set(editable_vertices)))
    if not editable:
        raise ValueError("repair requires at least one editable vertex")
    if any(not 0 <= vertex < instance.n for vertex in editable):
        raise ValueError("editable vertex is outside the instance")
    options = encoding_options or EncodingOptions()
    if options.encoding != "onehot":
        raise ValueError("repair currently requires one-hot encoding")
    options = replace(options, fix_first_color=False, anchor_clique=False)
    local = {vertex: index for index, vertex in enumerate(editable)}
    editable_set = set(editable)
    local_edges: list[tuple[int, int]] = []
    unit_forbids: set[tuple[int, int]] = set()
    for u, v in graph.iter_edges():
        u_editable = u in editable_set
        v_editable = v in editable_set
        if u_editable and v_editable:
            left, right = sorted((local[u], local[v]))
            local_edges.append((left, right))
        elif u_editable:
            unit_forbids.add((local[u], backbone_tuple[v]))
        elif v_editable:
            unit_forbids.add((local[v], backbone_tuple[u]))
        elif backbone_tuple[u] == backbone_tuple[v]:
            raise RuntimeError(
                f"frozen-frozen conflict {(u, v)}: editable set misses a bad edge"
            )
    quotient_instance = InstanceSpec(
        instance.polynomial,
        instance.colors,
        len(editable),
        instance.input_domain,
    )
    local_graph = ExplicitGraph(len(editable), tuple(local_edges))
    decode_spec = DecodeSpec(
        encoding=f"onehot-{options.amo.value}",
        assignment_vertices=len(editable),
        colors=instance.colors,
        mode="repair",
        output_n=instance.n,
        backbone=backbone_tuple,
        editable_vertices=editable,
    )
    base = encode_graph_problem(
        instance,
        quotient_instance,
        local_graph,
        options,
        scope=ModelScope.REPAIR,
        decode_spec=decode_spec,
        potts_constraints=(),
        metadata={
            "mode": "repair",
            "editable_vertex_count": len(editable),
            "editable_vertices": list(editable),
        },
    )
    variable_map = OneHotVariableMap(len(editable), instance.colors)
    clauses = list(base.clauses)
    for vertex, color in sorted(unit_forbids):
        clauses.append((-variable_map.var(vertex, color),))
    return EncodedProblem(
        instance,
        base.variable_count,
        tuple(clauses),
        sum(len(clause) for clause in clauses),
        base.encoding,
        ModelScope.REPAIR,
        decode_spec,
        local_graph,
        (),
        {
            **dict(base.metadata),
            "unit_forbids": len(unit_forbids),
        },
    )


def solve_repair(
    instance: InstanceSpec,
    backbone: tuple[int, ...] | list[int],
    backend: SearchBackend,
    encoding_options: EncodingOptions,
    solve_options: SolveOptions,
    *,
    editable_strategy: str = "greedy_vertex_cover",
    max_expansions: int = 3,
    backbone_metadata: dict[str, object] | None = None,
) -> SolveResult:
    if not backend.capabilities.complete:
        raise ValueError("repair requires a complete SAT backend")
    if max_expansions < 0:
        raise ValueError("max_expansions must be nonnegative")
    backbone_tuple = tuple(backbone)
    data = generate_distances(instance)
    graph = DistanceGraph(instance.n, data.values)
    initial_energy = conflict_count(instance, backbone_tuple)
    if initial_energy == 0:
        if not verify_coloring(instance, backbone_tuple).valid:
            raise RuntimeError("zero-energy repair backbone failed verification")
        return SolveResult(
            SolveStatus.FOUND_WITNESS,
            ModelScope.REPAIR,
            0.0,
            backend.name,
            backbone_tuple,
            {
                "mode": "repair",
                "initial_energy": 0,
                "editable_vertex_count": 0,
                "number_changed_colors": 0,
                "expansion_rounds": 0,
                **(backbone_metadata or {}),
            },
            best_coloring=backbone_tuple,
            best_energy=0,
        )
    bad_edges = find_bad_edges(graph, backbone_tuple)
    if editable_strategy == "all_endpoints":
        editable = tuple(sorted({vertex for edge in bad_edges for vertex in edge}))
    elif editable_strategy == "greedy_vertex_cover":
        editable = greedy_bad_edge_vertex_cover(bad_edges)
    elif editable_strategy == "exact_vertex_cover":
        editable = exact_bad_edge_vertex_cover(bad_edges)
    else:
        raise ValueError(f"unknown editable strategy {editable_strategy!r}")
    if any(u not in editable and v not in editable for u, v in bad_edges):
        raise RuntimeError("editable-set strategy failed to hit every bad edge")
    last_result: SolveResult | None = None
    initial_editable_count = len(editable)
    for expansion_round in range(max_expansions + 1):
        problem = build_repair_cnf(
            instance, graph, backbone_tuple, editable, encoding_options
        )
        result = backend.solve(problem, solve_options)
        common_metadata = {
            **dict(result.metadata),
            **(backbone_metadata or {}),
            "mode": "repair",
            "initial_energy": initial_energy,
            "editable_strategy": editable_strategy,
            "initial_editable_vertex_count": initial_editable_count,
            "editable_vertex_count": len(editable),
            "reduced_variables": problem.variable_count,
            "reduced_clauses": problem.clause_count,
            "expansion_rounds": expansion_round,
        }
        if result.status is SolveStatus.FOUND_WITNESS:
            assert result.coloring is not None
            if not verify_coloring(instance, result.coloring).valid:
                raise RuntimeError("expanded repair coloring failed verification")
            common_metadata["number_changed_colors"] = sum(
                left != right for left, right in zip(backbone_tuple, result.coloring)
            )
            return SolveResult(
                SolveStatus.FOUND_WITNESS,
                ModelScope.REPAIR,
                result.elapsed_seconds,
                result.backend,
                result.coloring,
                common_metadata,
                best_coloring=result.coloring,
                best_energy=0,
            )
        last_result = SolveResult(
            result.status,
            ModelScope.REPAIR,
            result.elapsed_seconds,
            result.backend,
            None,
            common_metadata,
            best_coloring=backbone_tuple,
            best_energy=initial_energy,
        )
        if expansion_round < max_expansions:
            expanded = expand_editable_set(graph, editable)
            if expanded == editable:
                break
            editable = expanded
    assert last_result is not None
    if last_result.status is SolveStatus.UNSAT_FULL_MODEL:
        raise RuntimeError("repair backend incorrectly reported full-model UNSAT")
    return last_result
