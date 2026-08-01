"""Deterministic DSATUR backtracking for tiny full-model instances."""

from __future__ import annotations

import time

from pvdw.distances import generate_distances
from pvdw.graph import DistanceGraph
from pvdw.model import InstanceSpec, ModelScope, SolveResult, SolveStatus
from pvdw.verify import verify_coloring


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
