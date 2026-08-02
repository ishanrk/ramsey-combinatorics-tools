"""Difference constraints and independently checked negative-cycle witnesses."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class DifferenceEdge:
    source: int
    target: int
    weight: int
    distance: int
    direction: int

    def __post_init__(self) -> None:
        if self.source == self.target:
            raise ValueError("difference edge must have distinct endpoints")
        if self.distance != abs(self.target - self.source):
            raise ValueError("difference edge distance differs from its endpoints")
        expected_direction = 1 if self.target > self.source else -1
        if self.direction != expected_direction:
            raise ValueError("difference edge direction differs from its endpoints")


@dataclass(frozen=True)
class NegativeCycleWitness:
    vertices: tuple[int, ...]
    edges: tuple[DifferenceEdge, ...]
    total_weight: int


def drift_bounds(distance: int, drift: int) -> tuple[int, int]:
    numerator = distance + 3 * drift - 1
    if numerator & 1:
        raise ValueError("distance and drift parities do not define integer bounds")
    lower = numerator // 2
    return lower, lower + 1


def build_difference_edges(
    n: int,
    q_values: Mapping[int, int],
) -> tuple[DifferenceEdge, ...]:
    if type(n) is not int or n < 1:
        raise ValueError("difference-constraint size must be positive")
    edges: list[DifferenceEdge] = []
    for distance, drift in sorted(q_values.items()):
        if type(distance) is not int or not 0 < distance < n:
            raise ValueError("difference constraint has an invalid distance")
        lower, upper = drift_bounds(distance, drift)
        for source in range(n - distance):
            target = source + distance
            edges.append(DifferenceEdge(source, target, upper, distance, 1))
            edges.append(DifferenceEdge(target, source, -lower, distance, -1))
    return tuple(edges)


def _witness_from_edges(edges: Sequence[DifferenceEdge]) -> NegativeCycleWitness:
    if not edges:
        raise ValueError("cycle must contain an edge")
    vertices = (edges[0].source, *(edge.target for edge in edges))
    return NegativeCycleWitness(vertices, tuple(edges), sum(edge.weight for edge in edges))


def verify_negative_cycle(witness: NegativeCycleWitness) -> bool:
    """Check closure, edge metadata, implied drifts, and negative total weight."""

    if len(witness.vertices) != len(witness.edges) + 1:
        return False
    if not witness.vertices or witness.vertices[0] != witness.vertices[-1]:
        return False
    inferred_drifts: dict[int, int] = {}
    total = 0
    for index, edge in enumerate(witness.edges):
        if (edge.source, edge.target) != witness.vertices[index : index + 2]:
            return False
        if edge.distance != abs(edge.target - edge.source):
            return False
        if edge.direction == 1:
            numerator = 2 * edge.weight - edge.distance - 1
        elif edge.direction == -1:
            numerator = -2 * edge.weight - edge.distance + 1
        else:
            return False
        if numerator % 3:
            return False
        drift = numerator // 3
        previous = inferred_drifts.setdefault(edge.distance, drift)
        if previous != drift:
            return False
        lower, upper = drift_bounds(edge.distance, drift)
        expected_weight = upper if edge.direction == 1 else -lower
        if edge.weight != expected_weight:
            return False
        total += edge.weight
    return total == witness.total_weight and total < 0


def canonicalize_cycle(witness: NegativeCycleWitness) -> NegativeCycleWitness:
    if not verify_negative_cycle(witness):
        raise ValueError("cannot canonicalize an invalid negative cycle")
    rotations = []
    edge_count = len(witness.edges)
    for offset in range(edge_count):
        rotated_edges = witness.edges[offset:] + witness.edges[:offset]
        rotations.append(_witness_from_edges(rotated_edges))
    reversed_edges = tuple(
        DifferenceEdge(
            edge.target,
            edge.source,
            1 - edge.weight,
            edge.distance,
            -edge.direction,
        )
        for edge in reversed(witness.edges)
    )
    reversed_witness = _witness_from_edges(reversed_edges)
    if verify_negative_cycle(reversed_witness):
        for offset in range(edge_count):
            rotated_edges = reversed_edges[offset:] + reversed_edges[:offset]
            rotations.append(_witness_from_edges(rotated_edges))
    return min(
        rotations,
        key=lambda candidate: (
            candidate.vertices[0],
            candidate.vertices,
            tuple((edge.distance, edge.direction, edge.weight) for edge in candidate.edges),
        ),
    )


def simplify_negative_cycle(witness: NegativeCycleWitness) -> NegativeCycleWitness:
    """Remove repeated closed subwalks whenever negativity is preserved."""

    current = canonicalize_cycle(witness)
    changed = True
    while changed:
        changed = False
        positions: dict[int, int] = {}
        for right, vertex in enumerate(current.vertices[:-1]):
            left = positions.get(vertex)
            if left is None:
                positions[vertex] = right
                continue
            subcycle = _witness_from_edges(current.edges[left:right])
            remaining_edges = current.edges[:left] + current.edges[right:]
            candidates = [
                candidate
                for candidate in (
                    subcycle,
                    _witness_from_edges(remaining_edges) if remaining_edges else None,
                )
                if candidate is not None and verify_negative_cycle(candidate)
            ]
            if candidates:
                current = min(candidates, key=lambda candidate: len(candidate.edges))
                current = canonicalize_cycle(current)
                changed = True
                break
    return current


def find_negative_cycle(
    nodes: Iterable[int],
    edges: Iterable[DifferenceEdge],
) -> NegativeCycleWitness | None:
    """Extract a negative cycle with Bellman--Ford and a zero super-source."""

    node_tuple = tuple(sorted(set(nodes)))
    if not node_tuple:
        raise ValueError("negative-cycle search needs at least one node")
    node_set = set(node_tuple)
    edge_tuple = tuple(edges)
    if any(edge.source not in node_set or edge.target not in node_set for edge in edge_tuple):
        raise ValueError("difference edge uses a node outside the search set")
    distances = {node: 0 for node in node_tuple}
    predecessors: dict[int, DifferenceEdge] = {}
    relaxed: int | None = None
    for _ in range(len(node_tuple)):
        relaxed = None
        for edge in edge_tuple:
            candidate = distances[edge.source] + edge.weight
            if candidate < distances[edge.target]:
                distances[edge.target] = candidate
                predecessors[edge.target] = edge
                relaxed = edge.target
        if relaxed is None:
            return None
    assert relaxed is not None
    cycle_vertex = relaxed
    for _ in range(len(node_tuple)):
        predecessor = predecessors.get(cycle_vertex)
        if predecessor is None:
            raise RuntimeError("Bellman--Ford predecessor chain left the relaxed region")
        cycle_vertex = predecessor.source
    reversed_edges: list[DifferenceEdge] = []
    current = cycle_vertex
    while True:
        predecessor = predecessors[current]
        reversed_edges.append(predecessor)
        current = predecessor.source
        if current == cycle_vertex:
            break
        if len(reversed_edges) > len(node_tuple):
            raise RuntimeError("negative-cycle predecessor extraction did not close")
    witness = _witness_from_edges(tuple(reversed(reversed_edges)))
    if not verify_negative_cycle(witness):
        raise RuntimeError("Bellman--Ford extracted a nonnegative or malformed cycle")
    return simplify_negative_cycle(witness)


def walk_negative_cycle(
    walk: Sequence[int],
    q_values: Mapping[int, int],
) -> NegativeCycleWitness:
    if len(walk) < 2 or walk[0] != walk[-1]:
        raise ValueError("known cycle walk must be explicitly closed")
    edges = []
    for source, target in zip(walk, walk[1:]):
        distance = abs(target - source)
        if distance not in q_values:
            raise ValueError(f"walk step uses undeclared distance {distance}")
        lower, upper = drift_bounds(distance, q_values[distance])
        direction = 1 if target > source else -1
        edges.append(
            DifferenceEdge(
                source,
                target,
                upper if direction == 1 else -lower,
                distance,
                direction,
            )
        )
    witness = _witness_from_edges(edges)
    if not verify_negative_cycle(witness):
        raise ValueError(f"walk has nonnegative total weight {witness.total_weight}")
    return witness


CUBE_WALK_CASE_A = (0, 512, 296, 80, 53, 26, 27, 0)
CUBE_WALK_CASE_B = (
    0, 343, 127, 470, 254, 38, 381, 165, 508, 383, 258, 133, 8, 0,
)
CUBE_WALK_CASE_C = (
    0, 512, 169, 385, 42, 258, 474, 131, 347, 4, 516, 173, 389, 46, 19, 27, 0,
)
CUBE_WALK_CASE_D = (
    0, 216, 432, 89, 305, 521, 9, 225, 441, 505, 513, 1, 217,
    433, 90, 306, 431, 88, 304, 520, 8, 224, 440, 504, 512, 0,
)
KNOWN_CUBE_WALKS = (
    CUBE_WALK_CASE_A,
    CUBE_WALK_CASE_B,
    CUBE_WALK_CASE_C,
    CUBE_WALK_CASE_D,
)
