import itertools
import random

from ramsey.tools.graphs import Graph
from ramsey.tools.isomorphism import (
    are_isomorphic,
    automorphism_count,
    canonical_code,
    canonical_label,
    canonical_permutation,
    find_isomorphism,
    is_isomorphic,
    is_isomorphism,
    iter_automorphisms,
    iter_isomorphisms,
)


def _cycle(order: int) -> Graph:
    return Graph(order, ((vertex, (vertex + 1) % order) for vertex in range(order)))


def _brute_isomorphic(first: Graph, second: Graph) -> bool:
    if first.vertex_count != second.vertex_count:
        return False
    return any(
        is_isomorphism(first, second, permutation)
        for permutation in itertools.permutations(range(first.vertex_count))
    )


def test_find_isomorphism_returns_a_checked_map() -> None:
    first = Graph(7, [(0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (4, 5)])
    permutation = (4, 0, 6, 2, 1, 5, 3)
    second = first.relabel(permutation)

    found = find_isomorphism(first, second)
    assert found is not None
    assert is_isomorphism(first, second, found)
    assert are_isomorphic(first, second)
    assert is_isomorphic(first, second)
    assert first.is_isomorphic_to(second)
    assert first.find_isomorphism(second) == found
    assert tuple(iter_isomorphisms(first, second, limit=1)) == (found,)


def test_nonisomorphic_regular_graphs_are_rejected() -> None:
    cycle = _cycle(6)
    two_triangles = Graph(
        6,
        [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)],
    )

    assert cycle.degree_sequence() == two_triangles.degree_sequence()
    assert cycle.edge_count == two_triangles.edge_count
    assert not are_isomorphic(cycle, two_triangles)
    assert find_isomorphism(cycle, two_triangles) is None


def test_mapping_checker_rejects_bad_maps() -> None:
    graph = _cycle(5)

    assert not is_isomorphism(graph, graph, (0, 1))
    assert not is_isomorphism(graph, graph, (0, 1, 2, 3, 3))
    assert not is_isomorphism(graph, graph, (0, 1, 2, 4, 3))
    assert not is_isomorphism(graph, Graph(4), (0, 1, 2, 3, 4))


def test_automorphism_helpers_are_exact() -> None:
    assert automorphism_count(Graph(0)) == 1
    assert automorphism_count(Graph.complete(4)) == 24
    assert automorphism_count(Graph(5, [(0, 1), (1, 2), (2, 3), (3, 4)])) == 2
    assert automorphism_count(_cycle(5)) == 10
    assert len(tuple(iter_automorphisms(Graph.complete(5), limit=7))) == 7


def test_canonical_forms_ignore_vertex_names() -> None:
    rng = random.Random(23041)
    for order in range(9):
        for _ in range(4):
            edges = [
                (u, v)
                for u in range(order)
                for v in range(u + 1, order)
                if rng.random() < 0.38
            ]
            graph = Graph(order, edges)
            permutation = list(range(order))
            rng.shuffle(permutation)
            relabeled = graph.relabel(permutation)

            assert canonical_code(graph) == canonical_code(relabeled)
            assert canonical_label(graph) == canonical_label(relabeled)
            assert graph.canonical_label() == canonical_label(graph)
            canonical_map = canonical_permutation(graph)
            assert is_isomorphism(graph, canonical_label(graph), canonical_map)


def test_isomorphism_search_matches_brute_force() -> None:
    rng = random.Random(8719)
    for order in range(1, 7):
        for _ in range(5):
            first = Graph(
                order,
                (
                    (u, v)
                    for u in range(order)
                    for v in range(u + 1, order)
                    if rng.random() < 0.4
                ),
            )
            second = Graph(
                order,
                (
                    (u, v)
                    for u in range(order)
                    for v in range(u + 1, order)
                    if rng.random() < 0.4
                ),
            )
            assert are_isomorphic(first, second) == _brute_isomorphic(first, second)


def test_symmetric_canonical_labeling_stays_practical() -> None:
    complete = Graph.complete(24)
    empty = Graph.empty(24)

    assert canonical_label(complete) == complete
    assert canonical_label(empty) == empty
