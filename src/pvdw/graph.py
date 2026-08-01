"""Implicit polynomial-distance graphs with linear-in-edge generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class DistanceGraph:
    n: int
    distances: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.n) is not int or self.n < 1:
            raise ValueError("n must be a positive ordinary integer")
        distances = tuple(self.distances)
        if distances != tuple(sorted(set(distances))):
            raise ValueError("distances must be sorted and unique")
        if any(type(distance) is not int or not 0 < distance < self.n for distance in distances):
            raise ValueError("every distance must satisfy 0 < distance < n")
        object.__setattr__(self, "distances", distances)

    @property
    def edge_count(self) -> int:
        return sum(self.n - distance for distance in self.distances)

    def iter_edges(self) -> Iterator[tuple[int, int]]:
        """Yield each edge once without scanning all vertex pairs."""

        for distance in self.distances:
            for u in range(self.n - distance):
                yield u, u + distance

    def build_adjacency(self) -> list[list[int]]:
        adjacency: list[list[int]] = [[] for _ in range(self.n)]
        for u, v in self.iter_edges():
            adjacency[u].append(v)
            adjacency[v].append(u)
        for neighbors in adjacency:
            neighbors.sort()
            if any(left == right for left, right in zip(neighbors, neighbors[1:])):
                raise AssertionError("duplicate neighbor generated from unique distances")
        return adjacency


def estimate_adjacency_bytes(
    graph: DistanceGraph,
    *,
    pointer_bytes: int = 8,
    list_overhead_bytes: int = 56,
) -> int:
    """Conservatively estimate list storage before building adjacency."""

    if pointer_bytes <= 0 or list_overhead_bytes < 0:
        raise ValueError("memory-size assumptions must be nonnegative")
    return graph.n * list_overhead_bytes + 2 * graph.edge_count * pointer_bytes


def estimate_adjacency_memory(graph: DistanceGraph) -> int:
    """Compatibility spelling for the default byte estimate."""

    return estimate_adjacency_bytes(graph)
