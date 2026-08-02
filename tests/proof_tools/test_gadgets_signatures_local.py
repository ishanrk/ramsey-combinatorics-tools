from __future__ import annotations

import itertools

from pvdw.proof_tools.gadgets import (
    MOSER_ASSIGNMENTS,
    MOSER_SHAPES,
    MOSER_SPINDLE,
    FiniteDistanceGraph,
    GraphEmbedding,
    SmallGraph,
    find_distance_cliques,
    find_embeddings,
    embedding_from_images,
    finite_distance_graph,
    interval_distance_graph,
    verify_embedding,
)
from pvdw.proof_tools.local import (
    common_neighborhood,
    find_common_neighborhood_odd_cycle,
    find_odd_cycle,
    maximum_avoiding_subset_dp,
    maximum_independent_set,
    maximum_independent_set_size,
    reconstruct_avoiding_subset,
    verify_common_neighborhood_odd_cycle,
)
from pvdw.proof_tools.signatures import ColorSignature, color_signatures


DIFFERENCES = (1, 2, 5, 7, 12, 15)


def test_explicit_finite_distance_graph_uses_only_supplied_points() -> None:
    graph = finite_distance_graph((-2, 0, 3), (2, 3, 99))
    assert isinstance(graph, FiniteDistanceGraph)
    assert graph.points == (-2, 0, 3)
    assert tuple(graph.iter_edges()) == ((0, 1), (1, 2))
    assert graph.edge_count == 2


def test_embedding_search_finds_moser_fixture_and_cliques() -> None:
    host = interval_distance_graph(16, DIFFERENCES)
    embeddings = find_embeddings(MOSER_SPINDLE, host)
    assert any(embedding.images == MOSER_ASSIGNMENTS["A"] for embedding in embeddings)
    assert all(verify_embedding(MOSER_SPINDLE, embedding, DIFFERENCES) for embedding in embeddings)
    triangle_graph = interval_distance_graph(4, (1, 2))
    assert (0, 1, 2) in find_distance_cliques(triangle_graph, 3)


def test_all_seven_supplied_spindle_shapes_verify() -> None:
    assignments = dict(MOSER_ASSIGNMENTS)
    assignments.update(
        {
            "E": tuple(12 - value for value in MOSER_ASSIGNMENTS["D0"]),
            "F": tuple(12 - value for value in MOSER_ASSIGNMENTS["C"]),
            "G": tuple(15 - value for value in MOSER_ASSIGNMENTS["A"]),
        }
    )
    for name, images in assignments.items():
        embedding = embedding_from_images(MOSER_SPINDLE, images)
        assert verify_embedding(MOSER_SPINDLE, embedding, DIFFERENCES)
        assert tuple(sorted(images)) == MOSER_SHAPES[name]


def test_induced_embedding_preserves_nonedges() -> None:
    target = SmallGraph(("a", "b", "c"), (("a", "b"), ("b", "c")))
    triangle = interval_distance_graph(2, (1, 2))
    assert find_embeddings(target, triangle, induced=False)
    assert not find_embeddings(target, triangle, induced=True)


def test_moser_signature_and_independence_reason() -> None:
    signatures = color_signatures(MOSER_SPINDLE, 4)
    assert set(signatures) == {ColorSignature((1, 2, 2, 2), 4)}
    independent = maximum_independent_set(MOSER_SPINDLE)
    assert len(independent) == 2
    assert maximum_independent_set_size(MOSER_SPINDLE) == 2


def test_odd_cycle_and_common_neighborhood_extraction() -> None:
    triangle = SmallGraph(("a", "b", "c"), (("a", "b"), ("b", "c"), ("c", "a")))
    cycle = find_odd_cycle(triangle)
    assert cycle is not None and cycle[0] == cycle[-1]
    assert (len(cycle) - 1) % 2 == 1

    graph = SmallGraph(
        ("u", "v", "a", "b", "c"),
        (
            ("u", "a"), ("u", "b"), ("u", "c"),
            ("v", "a"), ("v", "b"), ("v", "c"),
            ("a", "b"), ("b", "c"), ("c", "a"),
        ),
    )
    induced = common_neighborhood(graph, "u", "v")
    assert set(induced.vertices) == {"a", "b", "c"}
    witness = find_common_neighborhood_odd_cycle(graph)
    assert witness is not None
    assert witness.cycle[0] == witness.cycle[-1]
    assert verify_common_neighborhood_odd_cycle(graph, witness)


def test_distance_avoiding_dp_matches_bruteforce() -> None:
    for n in range(1, 10):
        for distances in ((1,), (2,), (1, 3), (2, 4)):
            result = maximum_avoiding_subset_dp(n, distances)
            brute = max(
                (
                    subset
                    for size in range(n + 1)
                    for subset in itertools.combinations(range(n), size)
                    if all(
                        right - left not in distances
                        for index, left in enumerate(subset)
                        for right in subset[index + 1 :]
                    )
                ),
                key=len,
            )
            assert result.size == len(brute)
            assert reconstruct_avoiding_subset(result) == result.subset
