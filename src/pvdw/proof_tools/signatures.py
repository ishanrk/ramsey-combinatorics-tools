"""Discovery enumeration for small-graph color-class signatures."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass

from pvdw.graph import DistanceGraph
from pvdw.proof_tools.gadgets import SmallGraph, graph_index_data
from pvdw.proof_tools.local import maximum_independent_set, maximum_independent_set_size


@dataclass(frozen=True)
class ColorSignature:
    class_sizes: tuple[int, ...]
    used_colors: int


def enumerate_proper_colorings(
    graph: SmallGraph | DistanceGraph,
    colors: int,
    *,
    modulo_color_permutation: bool = True,
) -> Iterator[tuple[int, ...]]:
    """Enumerate proper colorings by DSATUR and restricted growth strings."""

    if type(colors) is not int or colors < 1:
        raise ValueError("colors must be a positive ordinary integer")
    labels, _, adjacency = graph_index_data(graph)
    assigned = [-1] * len(labels)
    uncolored = set(range(len(labels)))

    def choose_vertex() -> int:
        return max(
            uncolored,
            key=lambda vertex: (
                len({assigned[n] for n in adjacency[vertex] if assigned[n] >= 0}),
                len(adjacency[vertex]),
                -vertex,
            ),
        )

    def search(used_colors: int) -> Iterator[tuple[int, ...]]:
        if not uncolored:
            yield tuple(assigned)
            return
        vertex = choose_vertex()
        forbidden = {assigned[neighbor] for neighbor in adjacency[vertex]}
        if modulo_color_permutation:
            candidates = range(min(colors, used_colors + 1))
        else:
            candidates = range(colors)
        uncolored.remove(vertex)
        for color in candidates:
            if color in forbidden:
                continue
            assigned[vertex] = color
            yield from search(max(used_colors, color + 1))
        assigned[vertex] = -1
        uncolored.add(vertex)

    yield from search(0)


def color_signatures(
    graph: SmallGraph | DistanceGraph,
    colors: int,
) -> Counter[ColorSignature]:
    signatures: Counter[ColorSignature] = Counter()
    for coloring in enumerate_proper_colorings(graph, colors):
        sizes = Counter(coloring)
        signature = ColorSignature(tuple(sorted(sizes.values())), len(sizes))
        signatures[signature] += 1
    return signatures


__all__ = [
    "ColorSignature",
    "color_signatures",
    "enumerate_proper_colorings",
    "maximum_independent_set",
    "maximum_independent_set_size",
]
