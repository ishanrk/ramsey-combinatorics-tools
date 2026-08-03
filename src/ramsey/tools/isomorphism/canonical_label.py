from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from ramsey.tools.graphs.graph import Graph

Partition = tuple[tuple[int, ...], ...]


def _initial_partition(graph: Graph) -> Partition:
    by_degree: dict[int, list[int]] = defaultdict(list)
    for vertex in graph:
        by_degree[graph.degree(vertex)].append(vertex)
    return tuple(tuple(by_degree[degree]) for degree in sorted(by_degree))


def _refine(graph: Graph, partition: Partition) -> Partition:
    current = partition
    while True:
        cell_masks = tuple(sum(1 << vertex for vertex in cell) for cell in current)
        refined: list[tuple[int, ...]] = []
        changed = False
        for cell in current:
            groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
            for vertex in cell:
                neighbors = graph.neighbor_mask(vertex)
                signature = tuple(
                    (neighbors & cell_mask).bit_count() for cell_mask in cell_masks
                )
                groups[signature].append(vertex)
            if len(groups) > 1:
                changed = True
            for signature in sorted(groups):
                refined.append(tuple(groups[signature]))
        current = tuple(refined)
        if not changed:
            return current


def _are_twins(graph: Graph, first: int, second: int) -> bool:
    first_without_second = graph.neighbor_mask(first) & ~(1 << second)
    second_without_first = graph.neighbor_mask(second) & ~(1 << first)
    return first_without_second == second_without_first


def _branch_representatives(graph: Graph, cell: tuple[int, ...]) -> tuple[int, ...]:
    representatives: list[int] = []
    for vertex in cell:
        if not any(_are_twins(graph, vertex, other) for other in representatives):
            representatives.append(vertex)
    return tuple(representatives)


def _adjacency_code(graph: Graph, order: Sequence[int]) -> int:
    code = 0
    for left_index, left in enumerate(order):
        neighbors = graph.neighbor_mask(left)
        for right_index in range(left_index + 1, len(order)):
            right = order[right_index]
            code = (code << 1) | int(bool(neighbors & (1 << right)))
    return code


def _canonical_order(graph: Graph, partition: Partition) -> tuple[int, tuple[int, ...]]:
    partition = _refine(graph, partition)
    expanded: list[tuple[int, ...]] = []
    for cell in partition:
        if len(cell) > 1 and all(
            _are_twins(graph, cell[0], vertex) for vertex in cell[1:]
        ):
            expanded.extend((vertex,) for vertex in cell)
        else:
            expanded.append(cell)
    partition = tuple(expanded)
    branch_index = next(
        (index for index, cell in enumerate(partition) if len(cell) > 1),
        None,
    )
    if branch_index is None:
        order = tuple(cell[0] for cell in partition)
        return _adjacency_code(graph, order), order

    cell = partition[branch_index]
    best_code: int | None = None
    best_order: tuple[int, ...] | None = None
    for vertex in _branch_representatives(graph, cell):
        rest = tuple(other for other in cell if other != vertex)
        child = partition[:branch_index] + ((vertex,), rest) + partition[branch_index + 1 :]
        code, order = _canonical_order(graph, child)
        if best_code is None or code < best_code or (
            code == best_code and best_order is not None and order < best_order
        ):
            best_code = code
            best_order = order
    if best_code is None or best_order is None:
        raise RuntimeError("canonical labeling reached an empty branch")
    return best_code, best_order


def canonical_order(graph: Graph) -> tuple[int, ...]:
    """return old vertices in their canonical new order."""
    return _canonical_order(graph, _initial_partition(graph))[1]


def canonical_permutation(graph: Graph) -> tuple[int, ...]:
    """return the canonical old-to-new vertex permutation."""
    order = canonical_order(graph)
    permutation = [0] * graph.vertex_count
    for new, old in enumerate(order):
        permutation[old] = new
    return tuple(permutation)


def canonical_code(graph: Graph) -> tuple[int, int]:
    """return an exact comparable code containing order and adjacency."""
    code, _ = _canonical_order(graph, _initial_partition(graph))
    return graph.vertex_count, code


def canonical_label(graph: Graph) -> Graph:
    """return a canonically relabeled copy of the graph."""
    return graph.relabel(canonical_permutation(graph))


__all__ = [
    "canonical_code",
    "canonical_label",
    "canonical_order",
    "canonical_permutation",
]
