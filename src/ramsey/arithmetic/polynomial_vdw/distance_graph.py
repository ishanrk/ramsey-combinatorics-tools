from __future__ import annotations

from ramsey.arithmetic.polynomial_vdw.distance_sets import (
    generate_forbidden_distances,
)
from ramsey.arithmetic.polynomial_vdw.instance import (
    InputDomain,
    PolynomialVanDerWaerdenInstance,
)
from ramsey.arithmetic.polynomial_vdw.polynomial import Polynomial
from ramsey.tools.graphs import Graph


def build_distance_graph(instance: PolynomialVanDerWaerdenInstance) -> Graph:
    """build the graph whose edges are the forbidden polynomial distances."""
    if not isinstance(instance, PolynomialVanDerWaerdenInstance):
        raise TypeError("instance must be a PolynomialVanDerWaerdenInstance")

    distances = generate_forbidden_distances(instance).distances
    masks = [0] * instance.n
    for distance in distances:
        for lower in range(instance.n - distance):
            upper = lower + distance
            masks[lower] |= 1 << upper
            masks[upper] |= 1 << lower
    return Graph.from_adjacency_masks(masks)


def build_polynomial_vdw_graph(
    polynomial: Polynomial,
    colors: int,
    n: int,
    *,
    input_domain: InputDomain = "all_nonzero",
) -> Graph:
    """make an instance from polynomial, colors, and n, then build its graph."""
    instance = PolynomialVanDerWaerdenInstance(
        polynomial=polynomial,
        colors=colors,
        n=n,
        input_domain=input_domain,
    )
    return build_distance_graph(instance)


build_graph = build_distance_graph

__all__ = [
    "build_distance_graph",
    "build_graph",
    "build_polynomial_vdw_graph",
]
