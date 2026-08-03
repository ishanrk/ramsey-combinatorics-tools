from __future__ import annotations

from ramsey.arithmetic.polynomial_vdw.distance_graph import build_distance_graph
from ramsey.arithmetic.polynomial_vdw.instance import (
    PolynomialVanDerWaerdenInstance,
)
from ramsey.solvers.sat import CNFFormula
from ramsey.tools.graphs import Graph


def color_variable(vertex: int, color: int, colors: int) -> int:
    """map one vertex and color to a stable one-based cnf variable."""
    if type(vertex) is not int or vertex < 0:
        raise ValueError("vertex must be a nonnegative ordinary integer")
    if type(colors) is not int or colors < 2:
        raise ValueError("colors must be an ordinary integer of at least two")
    if type(color) is not int or not 0 <= color < colors:
        raise ValueError("color is outside the available range")
    return vertex * colors + color + 1


def encode_graph_coloring_cnf(
    graph: Graph,
    colors: int,
    *,
    fix_first_color: bool = True,
) -> CNFFormula:
    """encode a graph coloring with pairwise one-hot clauses."""
    if not isinstance(graph, Graph):
        raise TypeError("graph must be a Graph")
    if type(colors) is not int:
        raise TypeError("colors must be an ordinary integer")
    if colors < 2:
        raise ValueError("colors must be at least two")
    if type(fix_first_color) is not bool:
        raise TypeError("fix_first_color must be a bool")

    clauses: list[tuple[int, ...]] = []
    for vertex in graph:
        variables = tuple(
            color_variable(vertex, color, colors) for color in range(colors)
        )
        clauses.append(variables)
        for first_color in range(colors):
            for second_color in range(first_color + 1, colors):
                clauses.append(
                    (-variables[first_color], -variables[second_color])
                )

    for first, second in graph.iter_edges():
        for color in range(colors):
            clauses.append(
                (
                    -color_variable(first, color, colors),
                    -color_variable(second, color, colors),
                )
            )

    if fix_first_color and graph.vertex_count:
        clauses.append((color_variable(0, 0, colors),))
    return CNFFormula.from_clauses(graph.vertex_count * colors, clauses)


def encode_polynomial_vdw_cnf(
    instance: PolynomialVanDerWaerdenInstance,
    *,
    fix_first_color: bool = True,
) -> CNFFormula:
    """build the exact polynomial graph and encode its coloring problem."""
    if not isinstance(instance, PolynomialVanDerWaerdenInstance):
        raise TypeError("instance must be a PolynomialVanDerWaerdenInstance")
    graph = build_distance_graph(instance)
    return encode_graph_coloring_cnf(
        graph,
        instance.colors,
        fix_first_color=fix_first_color,
    )


encode_cnf = encode_polynomial_vdw_cnf

__all__ = [
    "color_variable",
    "encode_cnf",
    "encode_graph_coloring_cnf",
    "encode_polynomial_vdw_cnf",
]
