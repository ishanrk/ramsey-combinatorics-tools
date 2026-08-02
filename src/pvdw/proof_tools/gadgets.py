"""Finite distance graphs and small rigid graph embeddings."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from pvdw.graph import DistanceGraph


@dataclass(frozen=True)
class FiniteDistanceGraph(DistanceGraph):
    """A distance graph on an explicit, possibly nonconsecutive point set."""

    points: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        points = tuple(self.points)
        if not points or any(type(point) is not int for point in points):
            raise ValueError("finite graph points must be ordinary integers")
        if points != tuple(sorted(set(points))):
            raise ValueError("finite graph points must be sorted and unique")
        distances = tuple(self.distances)
        if distances != tuple(sorted(set(distances))):
            raise ValueError("distances must be sorted and unique")
        if any(type(distance) is not int or distance <= 0 for distance in distances):
            raise ValueError("distances must be positive ordinary integers")
        if self.n != len(points):
            raise ValueError("finite graph size must equal its point count")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "distances", distances)

    @property
    def edge_count(self) -> int:
        return sum(1 for _ in self.iter_edges())

    def iter_edges(self) -> Iterator[tuple[int, int]]:
        index = {point: vertex for vertex, point in enumerate(self.points)}
        for left, point in enumerate(self.points):
            for distance in self.distances:
                right = index.get(point + distance)
                if right is not None:
                    yield left, right


@dataclass(frozen=True)
class SmallGraph:
    """A deterministic undirected graph with short symbolic vertex names."""

    vertices: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        vertices = tuple(self.vertices)
        if not vertices or any(not isinstance(vertex, str) or not vertex for vertex in vertices):
            raise ValueError("small graph vertices must be nonempty strings")
        if len(set(vertices)) != len(vertices):
            raise ValueError("small graph vertices must be unique")
        positions = {vertex: index for index, vertex in enumerate(vertices)}
        normalized: set[tuple[str, str]] = set()
        for left, right in self.edges:
            if left not in positions or right not in positions:
                raise ValueError("small graph edge uses an unknown vertex")
            if left == right:
                raise ValueError("small graph loops are not supported")
            normalized.add(
                (left, right)
                if positions[left] < positions[right]
                else (right, left)
            )
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(
            self,
            "edges",
            tuple(sorted(normalized, key=lambda edge: (positions[edge[0]], positions[edge[1]]))),
        )


@dataclass(frozen=True)
class GraphEmbedding:
    """An integer image aligned with the target's declared vertex order."""

    target_vertices: tuple[str, ...]
    images: tuple[int, ...]
    edge_differences: tuple[int, ...]
    reflected: bool = False

    def __post_init__(self) -> None:
        if len(self.target_vertices) != len(self.images):
            raise ValueError("embedding image length differs from target size")
        if len(set(self.images)) != len(self.images):
            raise ValueError("embedding must be injective")

    @property
    def mapping(self) -> dict[str, int]:
        return dict(zip(self.target_vertices, self.images))

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(sorted(self.images))


MOSER_SPINDLE = SmallGraph(
    vertices=("u", "a", "b", "c", "d", "e", "f"),
    edges=(
        ("u", "a"),
        ("u", "b"),
        ("u", "c"),
        ("u", "d"),
        ("a", "b"),
        ("a", "f"),
        ("b", "f"),
        ("c", "d"),
        ("c", "e"),
        ("d", "e"),
        ("e", "f"),
    ),
)

MOSER_ASSIGNMENTS: dict[str, tuple[int, ...]] = {
    "A": (3, 1, 2, 8, 10, 15, 0),
    "B": (7, 0, 2, 12, 14, 13, 1),
    "C": (7, 5, 12, 6, 8, 1, 0),
    "D0": (12, 0, 7, 10, 5, 3, 2),
}

MOSER_SHAPES: dict[str, tuple[int, ...]] = {
    name: tuple(sorted(images)) for name, images in MOSER_ASSIGNMENTS.items()
}
MOSER_SHAPES.update(
    {
        "E": tuple(sorted(12 - point for point in MOSER_SHAPES["D0"])),
        "F": tuple(sorted(12 - point for point in MOSER_SHAPES["C"])),
        "G": tuple(sorted(15 - point for point in MOSER_SHAPES["A"])),
    }
)


def finite_distance_graph(
    vertices: Iterable[int],
    allowed_differences: Iterable[int],
) -> DistanceGraph:
    """Build the induced distance graph on exactly the supplied integer points."""

    points = tuple(sorted(set(vertices)))
    differences = tuple(sorted(set(allowed_differences)))
    return FiniteDistanceGraph(len(points), differences, points)


def interval_distance_graph(
    max_vertex: int,
    allowed_differences: Iterable[int],
) -> DistanceGraph:
    """Build a distance graph on the inclusive interval ``0..max_vertex``."""

    if type(max_vertex) is not int or max_vertex < 0:
        raise ValueError("max_vertex must be a nonnegative ordinary integer")
    supplied = tuple(allowed_differences)
    if any(type(value) is not int or value <= 0 for value in supplied):
        raise ValueError("allowed differences must be positive ordinary integers")
    differences = tuple(
        difference
        for difference in sorted(set(supplied))
        if difference <= max_vertex
    )
    return DistanceGraph(max_vertex + 1, differences)


def graph_vertex_values(graph: DistanceGraph) -> tuple[int, ...]:
    return graph.points if isinstance(graph, FiniteDistanceGraph) else tuple(range(graph.n))


def graph_index_data(
    graph: SmallGraph | DistanceGraph,
) -> tuple[tuple[str | int, ...], tuple[tuple[int, int], ...], tuple[frozenset[int], ...]]:
    """Return stable labels, indexed edges, and indexed adjacency."""

    if isinstance(graph, SmallGraph):
        labels: tuple[str | int, ...] = graph.vertices
        index = {label: position for position, label in enumerate(graph.vertices)}
        edges = tuple((index[left], index[right]) for left, right in graph.edges)
    else:
        labels = graph_vertex_values(graph)
        edges = tuple(graph.iter_edges())
    adjacency = [set() for _ in labels]
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    return labels, edges, tuple(frozenset(neighbors) for neighbors in adjacency)


def verify_embedding(
    target: SmallGraph,
    embedding: GraphEmbedding,
    allowed_differences: Iterable[int],
) -> bool:
    """Independently check injectivity and every mapped target edge."""

    if embedding.target_vertices != target.vertices:
        return False
    mapping = embedding.mapping
    allowed = set(allowed_differences)
    used = tuple(
        sorted({abs(mapping[left] - mapping[right]) for left, right in target.edges})
    )
    return (
        len(set(mapping.values())) == len(mapping)
        and used == embedding.edge_differences
        and all(difference in allowed for difference in used)
    )


def embedding_from_images(
    target: SmallGraph,
    images: Iterable[int],
    *,
    reflected: bool = False,
) -> GraphEmbedding:
    image_tuple = tuple(images)
    if len(image_tuple) != len(target.vertices):
        raise ValueError("embedding image length differs from target size")
    mapping = dict(zip(target.vertices, image_tuple))
    differences = tuple(
        sorted({abs(mapping[left] - mapping[right]) for left, right in target.edges})
    )
    return GraphEmbedding(target.vertices, image_tuple, differences, reflected)


def find_embeddings(
    target: SmallGraph,
    host: DistanceGraph,
    *,
    induced: bool = False,
    normalize_translation: bool = True,
    identify_reflections: bool = True,
) -> list[GraphEmbedding]:
    """Find injective target embeddings by neighborhood-set intersections."""

    _, target_edges, target_adjacency = graph_index_data(target)
    host_values, _, host_adjacency = graph_index_data(host)
    target_order = sorted(
        range(len(target.vertices)),
        key=lambda vertex: (-len(target_adjacency[vertex]), vertex),
    )
    assigned: dict[int, int] = {}
    used: set[int] = set()
    found: dict[tuple[int, ...], GraphEmbedding] = {}

    def record() -> None:
        raw = tuple(host_values[assigned[index]] for index in range(len(target.vertices)))
        images = raw
        if normalize_translation:
            minimum = min(images)
            images = tuple(value - minimum for value in images)
        reflected = False
        if identify_reflections:
            axis_sum = min(images) + max(images)
            mirror = tuple(axis_sum - value for value in images)
            if mirror < images:
                images = mirror
                reflected = True
        differences = tuple(
            sorted({abs(images[left] - images[right]) for left, right in target_edges})
        )
        found.setdefault(
            images,
            GraphEmbedding(target.vertices, images, differences, reflected),
        )

    def search(depth: int) -> None:
        if depth == len(target_order):
            record()
            return
        target_vertex = target_order[depth]
        mapped_neighbors = [
            assigned[neighbor]
            for neighbor in target_adjacency[target_vertex]
            if neighbor in assigned
        ]
        if mapped_neighbors:
            candidates = set(host_adjacency[mapped_neighbors[0]])
            for neighbor in mapped_neighbors[1:]:
                candidates.intersection_update(host_adjacency[neighbor])
        else:
            candidates = set(range(host.n))
        candidates.difference_update(used)
        for candidate in sorted(candidates, key=lambda index: host_values[index]):
            if induced and any(
                ((other in target_adjacency[target_vertex])
                 != (assigned_host in host_adjacency[candidate]))
                for other, assigned_host in assigned.items()
            ):
                continue
            assigned[target_vertex] = candidate
            used.add(candidate)
            search(depth + 1)
            used.remove(candidate)
            del assigned[target_vertex]

    search(0)
    return sorted(found.values(), key=lambda embedding: embedding.images)


def _maximal_cliques(adjacency: tuple[frozenset[int], ...]) -> Iterator[frozenset[int]]:
    def bron_kerbosch(r: set[int], p: set[int], x: set[int]) -> Iterator[frozenset[int]]:
        if not p and not x:
            yield frozenset(r)
            return
        pivot = max(p | x, key=lambda vertex: len(p & adjacency[vertex]), default=None)
        candidates = p - (set() if pivot is None else set(adjacency[pivot]))
        for vertex in sorted(candidates):
            neighbors = set(adjacency[vertex])
            yield from bron_kerbosch(r | {vertex}, p & neighbors, x & neighbors)
            p.remove(vertex)
            x.add(vertex)

    yield from bron_kerbosch(set(), set(range(len(adjacency))), set())


def find_distance_cliques(graph: DistanceGraph, size: int) -> list[tuple[int, ...]]:
    """Enumerate fixed-size cliques via pivoted Bron--Kerbosch maximal cliques."""

    if type(size) is not int or size < 1:
        raise ValueError("clique size must be a positive ordinary integer")
    values, _, adjacency = graph_index_data(graph)
    cliques: set[tuple[int, ...]] = set()
    for maximal in _maximal_cliques(adjacency):
        if len(maximal) < size:
            continue
        for selected in itertools.combinations(sorted(maximal), size):
            cliques.add(tuple(sorted(int(values[index]) for index in selected)))
    return sorted(cliques)
