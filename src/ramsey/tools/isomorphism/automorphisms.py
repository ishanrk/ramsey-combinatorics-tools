from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence

from ramsey.tools.graphs.graph import Graph, _iter_bits


def _joint_color_refinement(
    first: Graph,
    second: Graph,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    first_colors = tuple(first.degree(vertex) for vertex in first)
    second_colors = tuple(second.degree(vertex) for vertex in second)
    while True:
        first_signatures = tuple(
            (
                first_colors[vertex],
                tuple(sorted(first_colors[neighbor] for neighbor in first.neighbors(vertex))),
            )
            for vertex in first
        )
        second_signatures = tuple(
            (
                second_colors[vertex],
                tuple(
                    sorted(second_colors[neighbor] for neighbor in second.neighbors(vertex))
                ),
            )
            for vertex in second
        )
        palette = {
            signature: color
            for color, signature in enumerate(
                sorted(set(first_signatures) | set(second_signatures))
            )
        }
        new_first = tuple(palette[signature] for signature in first_signatures)
        new_second = tuple(palette[signature] for signature in second_signatures)
        if Counter(new_first) != Counter(new_second):
            return None
        if new_first == first_colors and new_second == second_colors:
            return new_first, new_second
        first_colors = new_first
        second_colors = new_second


def _basic_invariants_match(first: Graph, second: Graph) -> bool:
    if first.vertex_count != second.vertex_count:
        return False
    if first.edge_count != second.edge_count:
        return False
    if first.degree_sequence() != second.degree_sequence():
        return False
    first_components = sorted(map(len, first.connected_components()))
    second_components = sorted(map(len, second.connected_components()))
    return first_components == second_components


def iter_isomorphisms(
    first: Graph,
    second: Graph,
    *,
    limit: int | None = None,
) -> Iterator[tuple[int, ...]]:
    """yield exact old-to-new vertex maps from first to second."""
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise TypeError("limit must be an ordinary integer or none")
        if limit < 0:
            raise ValueError("limit must be nonnegative")
    if limit == 0 or not _basic_invariants_match(first, second):
        return
    refined = _joint_color_refinement(first, second)
    if refined is None:
        return
    first_colors, second_colors = refined
    vertex_count = first.vertex_count
    color_count = max(first_colors, default=-1) + 1
    first_color_masks = [0] * color_count
    second_color_masks = [0] * color_count
    for vertex, color in enumerate(first_colors):
        first_color_masks[color] |= 1 << vertex
    for vertex, color in enumerate(second_colors):
        second_color_masks[color] |= 1 << vertex

    mapping = [-1] * vertex_count
    used_first = 0
    used_second = 0
    produced = 0

    def candidates(vertex: int) -> int:
        available = second_color_masks[first_colors[vertex]] & ~used_second
        first_neighbors = first.neighbor_mask(vertex)
        expected_mapped_neighbors = 0
        for old_neighbor in _iter_bits(first_neighbors & used_first):
            expected_mapped_neighbors |= 1 << mapping[old_neighbor]
        kept = 0
        for candidate in _iter_bits(available):
            second_neighbors = second.neighbor_mask(candidate)
            if second_neighbors & used_second != expected_mapped_neighbors:
                continue
            compatible = True
            for color in range(color_count):
                first_future = (
                    first_neighbors & first_color_masks[color] & ~used_first
                ).bit_count()
                second_future = (
                    second_neighbors & second_color_masks[color] & ~used_second
                ).bit_count()
                if first_future != second_future:
                    compatible = False
                    break
            if compatible:
                kept |= 1 << candidate
        return kept

    def search(depth: int) -> Iterator[tuple[int, ...]]:
        nonlocal produced, used_first, used_second
        if limit is not None and produced >= limit:
            return
        if depth == vertex_count:
            produced += 1
            yield tuple(mapping)
            return

        best_vertex = -1
        best_key: tuple[int, int, int, int] | None = None
        for vertex in range(vertex_count):
            if used_first & (1 << vertex):
                continue
            mapped_neighbors = (first.neighbor_mask(vertex) & used_first).bit_count()
            remaining_in_color = (
                first_color_masks[first_colors[vertex]] & ~used_first
            ).bit_count()
            key = (
                -mapped_neighbors,
                remaining_in_color,
                -first.degree(vertex),
                vertex,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_vertex = vertex

        best_candidates = candidates(best_vertex)
        if not best_candidates:
            return

        first_bit = 1 << best_vertex
        mapping[best_vertex] = -1
        used_first |= first_bit
        for candidate in _iter_bits(best_candidates):
            if limit is not None and produced >= limit:
                break
            second_bit = 1 << candidate
            mapping[best_vertex] = candidate
            used_second |= second_bit
            yield from search(depth + 1)
            used_second ^= second_bit
        used_first ^= first_bit
        mapping[best_vertex] = -1

    yield from search(0)


def find_isomorphism(first: Graph, second: Graph) -> tuple[int, ...] | None:
    """return one exact vertex map, or none when the graphs differ."""
    return next(iter_isomorphisms(first, second, limit=1), None)


def are_isomorphic(first: Graph, second: Graph) -> bool:
    return find_isomorphism(first, second) is not None


def is_isomorphism(
    first: Graph,
    second: Graph,
    mapping: Sequence[int],
) -> bool:
    """check a proposed old-to-new vertex map without trusting it."""
    vertex_count = first.vertex_count
    if second.vertex_count != vertex_count or len(mapping) != vertex_count:
        return False
    if any(not isinstance(v, int) or isinstance(v, bool) for v in mapping):
        return False
    if set(mapping) != set(range(vertex_count)):
        return False
    for vertex in range(vertex_count):
        image_neighbors = 0
        for neighbor in first.neighbors(vertex):
            image_neighbors |= 1 << mapping[neighbor]
        if image_neighbors != second.neighbor_mask(mapping[vertex]):
            return False
    return True


def iter_automorphisms(
    graph: Graph,
    *,
    limit: int | None = None,
) -> Iterator[tuple[int, ...]]:
    """yield graph automorphisms as old-to-new permutations."""
    yield from iter_isomorphisms(graph, graph, limit=limit)


def automorphisms(
    graph: Graph,
    *,
    limit: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    return tuple(iter_automorphisms(graph, limit=limit))


def automorphism_count(graph: Graph, *, limit: int | None = None) -> int:
    """count automorphisms, stopping at limit when one is supplied."""
    return sum(1 for _ in iter_automorphisms(graph, limit=limit))


is_isomorphic = are_isomorphic

__all__ = [
    "are_isomorphic",
    "automorphism_count",
    "automorphisms",
    "find_isomorphism",
    "is_isomorphic",
    "is_isomorphism",
    "iter_automorphisms",
    "iter_isomorphisms",
]
