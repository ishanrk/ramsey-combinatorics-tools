from __future__ import annotations

from hypothesis import given, strategies as st
import pytest

from pvdw.graph import DistanceGraph, estimate_adjacency_bytes


@given(
    st.integers(min_value=1, max_value=50),
    st.data(),
)
def test_edges_match_quadratic_reference(n: int, data: st.DataObject) -> None:
    distances = tuple(
        sorted(
            data.draw(
                st.sets(st.integers(min_value=1, max_value=max(1, n - 1)), max_size=8)
            )
        )
    )
    distances = tuple(distance for distance in distances if distance < n)
    graph = DistanceGraph(n, distances)
    reference = {
        (u, v)
        for u in range(n)
        for v in range(u + 1, n)
        if v - u in distances
    }
    assert set(graph.iter_edges()) == reference
    assert graph.edge_count == len(reference)
    adjacency = graph.build_adjacency()
    assert len(adjacency) == n
    for vertex, neighbors in enumerate(adjacency):
        assert neighbors == sorted(set(neighbors))
        assert set(neighbors) == {
            other
            for edge in reference
            if vertex in edge
            for other in edge
            if other != vertex
        }
    assert estimate_adjacency_bytes(graph) >= 2 * graph.edge_count * 8


def test_graph_rejects_invalid_or_duplicate_distances() -> None:
    for distances in ((0,), (3,), (1, 1), (2, 1)):
        with pytest.raises(ValueError):
            DistanceGraph(3, distances)
