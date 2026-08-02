"""Deterministic DSATUR backtracking for tiny full-model instances."""

from __future__ import annotations

import time

from pvdw.backends.base import BackendCapabilities, EncodedProblem, SolveOptions
from pvdw.distances import generate_distances
from pvdw.graph import DistanceGraph
from pvdw.model import InstanceSpec, ModelScope, SolveResult, SolveStatus
from pvdw.verify import verify_coloring


class BruteforceBackend:
    """SearchBackend adapter around the tiny Phase 1 DSATUR implementation."""

    name = "bruteforce"
    capabilities = BackendCapabilities(
        complete=True,
        incremental=False,
        accepts_dimacs=False,
        supports_assumptions=False,
        stochastic=False,
    )

    def __init__(self, size_limit: int = 30) -> None:
        self.size_limit = size_limit

    def solve(self, problem: EncodedProblem, options: SolveOptions) -> SolveResult:
        if problem.scope is not ModelScope.FULL:
            raise ValueError("bruteforce adapter supports direct full models only")
        result = solve_bruteforce(problem.instance, size_limit=self.size_limit)
        metadata = {
            **dict(result.metadata),
            "backend_version": "native",
            "seed": options.seed,
            "formula_variables": problem.variable_count,
            "formula_clauses": problem.clause_count,
            "formula_literals": problem.literal_count,
            "return_code": None,
            "command": None,
            "stdout_tail": "",
            "stderr_tail": "",
            "model_parsing": "direct_assignment"
            if result.coloring is not None
            else "not_applicable_unsat",
            "verification": "valid"
            if result.coloring is not None
            else "not_applicable_unsat",
        }
        return SolveResult(
            result.status,
            result.scope,
            result.elapsed_seconds,
            self.name,
            result.coloring,
            metadata,
            best_coloring=result.coloring,
            best_energy=0 if result.coloring is not None else None,
        )


def solve_bruteforce(
    instance: InstanceSpec,
    *,
    size_limit: int = 30,
) -> SolveResult:
    """Color a tiny graph by DSATUR, or prove its full model uncolorable."""

    if type(size_limit) is not int or size_limit < 1:
        raise ValueError("size_limit must be a positive ordinary integer")
    if instance.n > size_limit:
        raise ValueError(
            f"bruteforce backend limit is n <= {size_limit}; got n={instance.n}"
        )
    started = time.perf_counter()
    data = generate_distances(instance)
    graph = DistanceGraph(instance.n, data.values)
    adjacency = graph.build_adjacency()
    colors = [-1] * instance.n
    nodes = 0
    backtracks = 0

    def choose_vertex() -> int:
        uncolored = (vertex for vertex, color in enumerate(colors) if color < 0)
        return max(
            uncolored,
            key=lambda vertex: (
                len({colors[neighbor] for neighbor in adjacency[vertex] if colors[neighbor] >= 0}),
                len(adjacency[vertex]),
                -vertex,
            ),
        )

    def search(colored_count: int, used_count: int) -> bool:
        nonlocal nodes, backtracks
        nodes += 1
        if colored_count == instance.n:
            return True
        vertex = choose_vertex()
        forbidden = {
            colors[neighbor] for neighbor in adjacency[vertex] if colors[neighbor] >= 0
        }
        choices = [color for color in range(used_count) if color not in forbidden]
        if used_count < instance.colors and used_count not in forbidden:
            choices.append(used_count)
        for color in choices:
            colors[vertex] = color
            if search(colored_count + 1, max(used_count, color + 1)):
                return True
            colors[vertex] = -1
        backtracks += 1
        return False

    found = search(0, 0)
    elapsed = time.perf_counter() - started
    metadata = {
        "nodes": nodes,
        "backtracks": backtracks,
        "edge_count": graph.edge_count,
        "size_limit": size_limit,
    }
    if found:
        coloring = tuple(colors)
        verification = verify_coloring(instance, coloring)
        if not verification.valid:
            raise RuntimeError(
                "bruteforce produced a coloring that failed independent verification"
            )
        return SolveResult(
            SolveStatus.FOUND_WITNESS,
            ModelScope.FULL,
            elapsed,
            "bruteforce",
            coloring,
            metadata,
        )
    return SolveResult(
        SolveStatus.UNSAT_FULL_MODEL,
        ModelScope.FULL,
        elapsed,
        "bruteforce",
        None,
        metadata,
    )
