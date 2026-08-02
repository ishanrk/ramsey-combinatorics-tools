"""Exact local obstruction searches on small finite distance graphs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from pvdw.graph import DistanceGraph
from pvdw.proof_tools.gadgets import SmallGraph, graph_index_data


GraphVertex = str | int


def maximum_independent_set(graph: SmallGraph | DistanceGraph) -> tuple[GraphVertex, ...]:
    """Return an exact maximum independent set using bitset branch-and-bound."""

    labels, _, adjacency = graph_index_data(graph)
    neighbor_masks = tuple(
        sum(1 << neighbor for neighbor in neighbors) for neighbors in adjacency
    )
    best_mask = 0

    def search(candidates: int, selected: int) -> None:
        nonlocal best_mask
        if selected.bit_count() + candidates.bit_count() <= best_mask.bit_count():
            return
        if candidates == 0:
            if selected.bit_count() > best_mask.bit_count():
                best_mask = selected
            return
        candidate_indices = tuple(
            index for index in range(len(labels)) if candidates & (1 << index)
        )
        vertex = max(
            candidate_indices,
            key=lambda index: (neighbor_masks[index] & candidates).bit_count(),
        )
        bit = 1 << vertex
        search(candidates & ~bit & ~neighbor_masks[vertex], selected | bit)
        search(candidates & ~bit, selected)

    search((1 << len(labels)) - 1, 0)
    return tuple(labels[index] for index in range(len(labels)) if best_mask & (1 << index))


def maximum_independent_set_size(graph: SmallGraph | DistanceGraph) -> int:
    return len(maximum_independent_set(graph))


def common_neighborhood(
    graph: SmallGraph | DistanceGraph,
    u: GraphVertex,
    v: GraphVertex,
) -> SmallGraph:
    """Return the graph induced by the common neighbors of two vertices."""

    labels, edges, adjacency = graph_index_data(graph)
    positions = {label: index for index, label in enumerate(labels)}
    if u not in positions or v not in positions or u == v:
        raise ValueError("common-neighborhood endpoints must be distinct graph vertices")
    selected = adjacency[positions[u]] & adjacency[positions[v]]
    names = tuple(str(labels[index]) for index in sorted(selected))
    local = {host_index: names[position] for position, host_index in enumerate(sorted(selected))}
    induced_edges = tuple(
        (local[left], local[right])
        for left, right in edges
        if left in selected and right in selected
    )
    return SmallGraph(names, induced_edges)


def find_odd_cycle(graph: SmallGraph | DistanceGraph) -> tuple[GraphVertex, ...] | None:
    """Find and explicitly close an odd cycle using bipartite BFS parents."""

    labels, _, adjacency = graph_index_data(graph)
    colors: dict[int, int] = {}
    parents: dict[int, int | None] = {}
    depths: dict[int, int] = {}
    for root in range(len(labels)):
        if root in colors:
            continue
        colors[root] = 0
        parents[root] = None
        depths[root] = 0
        queue: deque[int] = deque([root])
        while queue:
            left = queue.popleft()
            for right in adjacency[left]:
                if right not in colors:
                    colors[right] = 1 - colors[left]
                    parents[right] = left
                    depths[right] = depths[left] + 1
                    queue.append(right)
                    continue
                if colors[right] != colors[left]:
                    continue
                left_path: list[int] = []
                right_path: list[int] = []
                a, b = left, right
                while depths[a] > depths[b]:
                    left_path.append(a)
                    parent = parents[a]
                    assert parent is not None
                    a = parent
                while depths[b] > depths[a]:
                    right_path.append(b)
                    parent = parents[b]
                    assert parent is not None
                    b = parent
                while a != b:
                    left_path.append(a)
                    right_path.append(b)
                    parent_a = parents[a]
                    parent_b = parents[b]
                    assert parent_a is not None and parent_b is not None
                    a, b = parent_a, parent_b
                left_path.append(a)
                cycle_indices = left_path + list(reversed(right_path)) + [left]
                cycle = tuple(labels[index] for index in cycle_indices)
                if (len(cycle) - 1) % 2 != 1:
                    raise RuntimeError("odd-cycle extraction produced even length")
                return cycle
    return None


@dataclass(frozen=True)
class CommonNeighborhoodOddCycleWitness:
    u: GraphVertex
    v: GraphVertex
    common_graph: SmallGraph
    cycle: tuple[GraphVertex, ...]


def find_common_neighborhood_odd_cycle(
    graph: SmallGraph | DistanceGraph,
) -> CommonNeighborhoodOddCycleWitness | None:
    labels, _, _ = graph_index_data(graph)
    for left_index in range(len(labels)):
        for right_index in range(left_index + 1, len(labels)):
            induced = common_neighborhood(graph, labels[left_index], labels[right_index])
            cycle = find_odd_cycle(induced)
            if cycle is not None:
                return CommonNeighborhoodOddCycleWitness(
                    labels[left_index], labels[right_index], induced, cycle
                )
    return None


def verify_common_neighborhood_odd_cycle(
    graph: SmallGraph | DistanceGraph,
    witness: CommonNeighborhoodOddCycleWitness,
) -> bool:
    try:
        induced = common_neighborhood(graph, witness.u, witness.v)
    except ValueError:
        return False
    if induced != witness.common_graph or witness.cycle[0] != witness.cycle[-1]:
        return False
    if (len(witness.cycle) - 1) % 2 != 1:
        return False
    edges = {frozenset(edge) for edge in induced.edges}
    return all(
        frozenset((left, right)) in edges
        for left, right in zip(witness.cycle, witness.cycle[1:])
    )


@dataclass(frozen=True)
class AvoidingSubsetResult:
    n: int
    distances: tuple[int, ...]
    size: int
    subset: tuple[int, ...]
    peak_states: int


def maximum_avoiding_subset_dp(
    n: int,
    distances: tuple[int, ...] | list[int],
    *,
    state_limit: int = 1_000_000,
) -> AvoidingSubsetResult:
    """Solve the interval gap problem with a last-``max(D)`` bitmask state."""

    if type(n) is not int or n < 0:
        raise ValueError("n must be a nonnegative ordinary integer")
    normalized = tuple(sorted(set(distances)))
    if any(type(distance) is not int or distance <= 0 for distance in normalized):
        raise ValueError("avoided distances must be positive ordinary integers")
    if state_limit < 1:
        raise ValueError("state_limit must be positive")
    if n == 0:
        return AvoidingSubsetResult(0, normalized, 0, (), 1)
    relevant = tuple(distance for distance in normalized if distance < n)
    span = max(relevant, default=0)
    mask_limit = (1 << span) - 1 if span else 0
    scores: dict[int, int] = {0: 0}
    parents: list[dict[int, tuple[int, bool]]] = []
    peak_states = 1
    for position in range(n):
        next_scores: dict[int, int] = {}
        layer_parents: dict[int, tuple[int, bool]] = {}
        for state, score in scores.items():
            excluded = (state << 1) & mask_limit
            if score > next_scores.get(excluded, -1):
                next_scores[excluded] = score
                layer_parents[excluded] = (state, False)
            allowed = all(
                distance > position or not (state & (1 << (distance - 1)))
                for distance in relevant
            )
            if allowed:
                included = ((state << 1) | 1) & mask_limit
                if score + 1 > next_scores.get(included, -1):
                    next_scores[included] = score + 1
                    layer_parents[included] = (state, True)
        if len(next_scores) > state_limit:
            raise ValueError(
                f"distance-avoiding DP exceeded state limit {state_limit} at position {position}"
            )
        parents.append(layer_parents)
        scores = next_scores
        peak_states = max(peak_states, len(scores))
    final_state = max(scores, key=lambda state: (scores[state], -state))
    selected: list[int] = []
    state = final_state
    for position in range(n - 1, -1, -1):
        previous, included = parents[position][state]
        if included:
            selected.append(position)
        state = previous
    selected.reverse()
    result = AvoidingSubsetResult(
        n, normalized, scores[final_state], tuple(selected), peak_states
    )
    if any(
        right - left in set(relevant)
        for index, left in enumerate(result.subset)
        for right in result.subset[index + 1 :]
    ):
        raise RuntimeError("reconstructed subset violates an avoided distance")
    return result


def reconstruct_avoiding_subset(result: AvoidingSubsetResult) -> tuple[int, ...]:
    """Return the independently checked subset retained by the DP witness."""

    if len(result.subset) != result.size:
        raise ValueError("avoiding-subset witness size is inconsistent")
    avoided = set(result.distances)
    if any(
        right - left in avoided
        for index, left in enumerate(result.subset)
        for right in result.subset[index + 1 :]
    ):
        raise ValueError("avoiding-subset witness contains a forbidden gap")
    return result.subset
