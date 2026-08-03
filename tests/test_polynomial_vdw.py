import random

import pytest

from ramsey.arithmetic.polynomial_vdw import (
    Polynomial,
    PolynomialVanDerWaerdenInstance,
    build_distance_graph,
    build_polynomial_vdw_graph,
    certified_input_bound,
    forbidden_distances,
    generate_forbidden_distances,
)
from ramsey.tools.graphs import Graph


def test_polynomial_normalizes_and_evaluates_exactly() -> None:
    polynomial = Polynomial((0, -11, 3, 0, 0))

    assert polynomial.coefficients == (0, -11, 3)
    assert polynomial.degree == 2
    assert polynomial.leading_coefficient == 3
    assert polynomial(5) == 20
    assert polynomial.evaluate(-2) == 34


@pytest.mark.parametrize(
    "coefficients",
    [
        (),
        (0,),
        (0, 0),
        (1, 2),
        (0, 1.5),
        (0, True),
    ],
)
def test_polynomial_rejects_invalid_coefficients(
    coefficients: tuple[object, ...],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        Polynomial(coefficients)  # type: ignore[arg-type]


def test_instance_validation() -> None:
    polynomial = Polynomial((0, 1))

    assert PolynomialVanDerWaerdenInstance(polynomial, 3, 9).max_difference == 8
    with pytest.raises(ValueError):
        PolynomialVanDerWaerdenInstance(polynomial, 1, 9)
    with pytest.raises(ValueError):
        PolynomialVanDerWaerdenInstance(polynomial, 3, 0)
    with pytest.raises(TypeError):
        PolynomialVanDerWaerdenInstance(polynomial, True, 9)
    with pytest.raises(ValueError):
        PolynomialVanDerWaerdenInstance(
            polynomial,
            3,
            9,
            input_domain="zero_too",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("coefficients", "n", "expected"),
    [
        ((0, 0, 1), 58, (1, 4, 9, 16, 25, 36, 49)),
        ((0, 0, 0, 1), 522, (1, 8, 27, 64, 125, 216, 343, 512)),
        ((0, 1, 1), 97, (2, 6, 12, 20, 30, 42, 56, 72, 90)),
    ],
)
def test_known_forbidden_distance_sets(
    coefficients: tuple[int, ...],
    n: int,
    expected: tuple[int, ...],
) -> None:
    instance = PolynomialVanDerWaerdenInstance(Polynomial(coefficients), 4, n)

    assert forbidden_distances(instance) == expected


def test_input_domain_and_preimages_are_preserved() -> None:
    polynomial = Polynomial((0, 2, 1))
    all_inputs = PolynomialVanDerWaerdenInstance(polynomial, 3, 7)
    positive_inputs = PolynomialVanDerWaerdenInstance(
        polynomial,
        3,
        7,
        input_domain="positive",
    )

    data = generate_forbidden_distances(all_inputs)
    assert data.distances == (1, 3)
    assert data.preimages[1] == (-1,)
    assert data.preimages[3] == (-3, 1)
    assert forbidden_distances(positive_inputs) == (3,)


def test_distance_graph_has_exactly_the_polynomial_edges() -> None:
    instance = PolynomialVanDerWaerdenInstance(Polynomial((0, 0, 1)), 3, 10)
    graph = build_distance_graph(instance)
    expected_edges = {
        (lower, lower + distance)
        for distance in (1, 4, 9)
        for lower in range(10 - distance)
    }

    assert isinstance(graph, Graph)
    assert graph.vertex_count == 10
    assert graph.edge_count == 16
    assert set(graph.iter_edges()) == expected_edges


def test_convenience_builder_validates_colors_and_returns_the_same_graph() -> None:
    polynomial = Polynomial((0, -1, 1))
    instance = PolynomialVanDerWaerdenInstance(polynomial, 5, 20)

    assert build_polynomial_vdw_graph(polynomial, 5, 20) == build_distance_graph(
        instance
    )
    assert build_polynomial_vdw_graph(polynomial, 2, 20) == build_polynomial_vdw_graph(
        polynomial,
        9,
        20,
    )
    with pytest.raises(ValueError):
        build_polynomial_vdw_graph(polynomial, 1, 20)


def test_single_vertex_instance_returns_an_empty_graph() -> None:
    graph = build_polynomial_vdw_graph(Polynomial((0, 7)), 2, 1)

    assert graph == Graph.empty(1)


def test_certified_bound_matches_a_much_wider_brute_force_search() -> None:
    rng = random.Random(70413)
    for _ in range(120):
        degree = rng.randint(1, 4)
        coefficients = [0]
        coefficients.extend(rng.randint(-4, 4) for _ in range(degree - 1))
        leading = 0
        while leading == 0:
            leading = rng.randint(-4, 4)
        coefficients.append(leading)
        polynomial = Polynomial(tuple(coefficients))
        n = rng.randint(1, 45)
        instance = PolynomialVanDerWaerdenInstance(polynomial, 3, n)
        data = generate_forbidden_distances(instance)
        bound = certified_input_bound(polynomial, n - 1)

        brute = {
            abs(polynomial(value))
            for value in range(-4 * bound - 20, 4 * bound + 21)
            if value != 0 and 0 < abs(polynomial(value)) < n
        }
        assert set(data.distances) == brute
        assert data.input_bound == bound
        for magnitude in range(bound, bound + 20):
            assert abs(polynomial(magnitude)) >= n
            assert abs(polynomial(-magnitude)) >= n
