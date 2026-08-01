from __future__ import annotations

from hypothesis import given, settings, strategies as st

from pvdw.distances import certified_input_bound, generate_distances
from pvdw.model import InstanceSpec, PolynomialSpec


def instance(
    coefficients: tuple[int, ...],
    n: int,
    domain: str = "all_nonzero",
) -> InstanceSpec:
    return InstanceSpec(PolynomialSpec(coefficients), 3, n, domain)  # type: ignore[arg-type]


def test_required_square_distances() -> None:
    data = generate_distances(instance((0, 0, 1), 58))
    assert set(data.values) == {1, 4, 9, 16, 25, 36, 49}
    assert data.preimages[4] == (-2, 2)


def test_required_cube_distances() -> None:
    data = generate_distances(instance((0, 0, 0, 1), 522))
    assert set(data.values) == {1, 8, 27, 64, 125, 216, 343, 512}


def test_required_quadratic_distances_with_negative_inputs() -> None:
    data = generate_distances(instance((0, 1, 1), 97))
    assert set(data.values) == {2, 6, 12, 20, 30, 42, 56, 72, 90}
    assert data.preimages[2] == (-2, 1)


def test_positive_domain_changes_preimages_and_can_change_values() -> None:
    all_inputs = generate_distances(instance((0, -3, 1), 5))
    positive = generate_distances(instance((0, -3, 1), 5, "positive"))
    assert all(value > 0 for value in positive.values)
    assert all(d > 0 for inputs in positive.preimages.values() for d in inputs)
    assert set(positive.values) <= set(all_inputs.values)


@st.composite
def polynomial_instances(draw: st.DrawFn) -> InstanceSpec:
    degree = draw(st.integers(min_value=1, max_value=4))
    middle = draw(
        st.lists(
            st.integers(min_value=-4, max_value=4),
            min_size=max(0, degree - 1),
            max_size=max(0, degree - 1),
        )
    )
    leading = draw(st.integers(min_value=-4, max_value=4).filter(lambda x: x != 0))
    n = draw(st.integers(min_value=1, max_value=35))
    domain = draw(st.sampled_from(["all_nonzero", "positive"]))
    return InstanceSpec(
        PolynomialSpec(tuple([0, *middle, leading])),
        colors=draw(st.integers(min_value=2, max_value=5)),
        n=n,
        input_domain=domain,
    )


@given(polynomial_instances())
@settings(max_examples=100, deadline=None)
def test_certified_bound_matches_much_larger_bruteforce(instance: InstanceSpec) -> None:
    data = generate_distances(instance)
    radius = data.input_bound * 4 + 12
    inputs = range(1, radius + 1) if instance.input_domain == "positive" else range(-radius, radius + 1)
    reference: dict[int, list[int]] = {}
    for d in inputs:
        if d == 0:
            continue
        value = instance.polynomial.evaluate(d)
        if value and abs(value) < instance.n:
            reference.setdefault(abs(value), []).append(d)
    assert data.values == tuple(sorted(reference))
    assert {key: tuple(value) for key, value in reference.items()} == dict(data.preimages)

    outside = [data.input_bound + 1]
    if instance.input_domain == "all_nonzero":
        outside.append(-data.input_bound - 1)
    assert all(
        abs(instance.polynomial.evaluate(d)) > instance.n - 1
        for d in outside
    )


def test_bound_uses_exact_integer_growth_search() -> None:
    polynomial = PolynomialSpec((0, 0, 0, 7))
    bound = certified_input_bound(polynomial, 10**30)
    assert 7 * bound**3 > 2 * 10**30
    assert bound == 1 or 7 * (bound - 1) ** 3 <= 2 * 10**30
