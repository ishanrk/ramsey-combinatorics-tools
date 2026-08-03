from __future__ import annotations

from collections.abc import Sequence

from ramsey.arithmetic.polynomial_vdw.distance_graph import build_distance_graph
from ramsey.arithmetic.polynomial_vdw.instance import (
    PolynomialVanDerWaerdenInstance,
)


def verify_coloring(
    instance: PolynomialVanDerWaerdenInstance,
    coloring: Sequence[int],
) -> bool:
    """check a coloring against a freshly rebuilt polynomial graph."""
    if not isinstance(instance, PolynomialVanDerWaerdenInstance):
        raise TypeError("instance must be a PolynomialVanDerWaerdenInstance")
    if len(coloring) != instance.n:
        return False
    if any(
        type(color) is not int or not 0 <= color < instance.colors
        for color in coloring
    ):
        return False
    graph = build_distance_graph(instance)
    return all(coloring[first] != coloring[second] for first, second in graph.iter_edges())


__all__ = ["verify_coloring"]
