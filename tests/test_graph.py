import pytest

from ramsey.tools.graphs import BitGraph, Graph


def test_graph_uses_simple_undirected_edges() -> None:
    graph = Graph(6, [(0, 1), (1, 2), (2, 0), (3, 4), (1, 0)])

    assert BitGraph is Graph
    assert graph.vertex_count == 6
    assert graph.order == 6
    assert graph.edge_count == 4
    assert graph.size == 4
    assert tuple(graph.iter_edges()) == ((0, 1), (0, 2), (1, 2), (3, 4))
    assert tuple(graph.neighbors(0)) == (1, 2)
    assert graph.neighbor_mask(0) == 0b110
    assert graph.degree(0) == 2
    assert graph.degree_sequence() == (2, 2, 2, 1, 1, 0)
    assert graph.has_edge(1, 0)
    assert 5 in graph
    assert 6 not in graph


def test_graph_mutation_keeps_counts_and_masks_in_sync() -> None:
    graph = Graph.empty(3)

    assert graph.add_edge(0, 2)
    assert not graph.add_edge(2, 0)
    assert graph.edge_count == 1
    assert graph.remove_edge(2, 0)
    assert not graph.remove_edge(0, 2)
    assert graph.edge_count == 0
    assert tuple(graph.add_vertices(2)) == (3, 4)
    assert graph.vertex_count == 5
    graph.add_edge(3, 4)
    graph.clear_edges()
    assert graph.adjacency_masks == (0, 0, 0, 0, 0)


def test_fast_constructors_and_complement() -> None:
    complete = Graph.complete(5)

    assert complete.edge_count == 10
    assert complete.degree_sequence() == (4, 4, 4, 4, 4)
    assert complete.complement() == Graph.empty(5)
    assert Graph.from_edges([(2, 4)]).vertex_count == 5
    assert Graph.from_adjacency_masks((0b110, 0b001, 0b001)) == Graph(
        3, [(0, 1), (0, 2)]
    )


def test_components_subgraphs_and_relabeling() -> None:
    graph = Graph(6, [(0, 1), (1, 2), (2, 3), (3, 4)])

    assert graph.connected_components() == ((0, 1, 2, 3, 4), (5,))
    assert not graph.is_connected()
    assert Graph(0).is_connected()
    assert graph.induced_subgraph([4, 3, 2]) == Graph(3, [(0, 1), (1, 2)])

    triangle = Graph(4, [(0, 1), (1, 2), (0, 2)])
    relabeled = triangle.relabel((2, 0, 3, 1))
    assert set(relabeled.iter_edges()) == {(0, 2), (0, 3), (2, 3)}
    assert relabeled.degree(1) == 0
    assert triangle.copy() == triangle
    assert triangle.copy() is not triangle


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda: Graph(-1), ValueError),
        (lambda: Graph(True), TypeError),
        (lambda: Graph(2, [(0, 0)]), ValueError),
        (lambda: Graph(2, [(0, 2)]), IndexError),
        (lambda: Graph.from_adjacency_masks((0b10, 0)), ValueError),
        (lambda: Graph.from_adjacency_masks((0b100, 0)), ValueError),
        (lambda: Graph(3).induced_subgraph([0, 0]), ValueError),
        (lambda: Graph(3).relabel((0, 1, 1)), ValueError),
    ],
)
def test_invalid_graph_inputs_are_rejected(call: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        call()  # type: ignore[operator]
