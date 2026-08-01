"""Certified generation of all polynomial differences relevant to an interval."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from pvdw.model import InstanceSpec, PolynomialSpec


@dataclass(frozen=True)
class DistanceData:
    values: tuple[int, ...]
    preimages: Mapping[int, tuple[int, ...]]
    input_bound: int

    def __post_init__(self) -> None:
        values = tuple(self.values)
        if values != tuple(sorted(set(values))) or any(value <= 0 for value in values):
            raise ValueError("distance values must be sorted, unique, and positive")
        normalized = {
            distance: tuple(sorted(inputs)) for distance, inputs in self.preimages.items()
        }
        if tuple(sorted(normalized)) != values:
            raise ValueError("preimages must have exactly the distance-value keys")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "preimages", MappingProxyType(normalized))


def _smallest_growth_bound(leading: int, degree: int, twice_bound: int) -> int:
    """Return the least positive t with leading * t**degree > twice_bound."""

    high = 1
    while leading * high**degree <= twice_bound:
        high *= 2
    low = 1
    while low < high:
        middle = (low + high) // 2
        if leading * middle**degree > twice_bound:
            high = middle
        else:
            low = middle + 1
    return low


def certified_input_bound(poly: PolynomialSpec, max_difference: int) -> int:
    """Return a complete finite input radius for differences at most ``B``.

    Write ``p(x) = a_k x^k + ... + a_0`` and
    ``S = sum(i < k, abs(a_i))``.  For every integer ``|d| >= 1``, the
    lower-degree terms have magnitude at most ``S |d|^(k-1)``.  Once
    ``|a_k||d| >= 2S``, the reverse triangle inequality gives
    ``|p(d)| >= |a_k||d|^k / 2``.  The exact integer growth bound also
    ensures ``|a_k||d|^k > 2B``, hence ``|p(d)| > B``.  Taking the maximum
    of the two least such bounds proves every relevant input lies in
    ``[-R, R]``; enumeration includes the endpoints for simplicity.
    """

    if type(max_difference) is not int:
        raise TypeError("max_difference must be an ordinary Python integer")
    if max_difference < 0:
        raise ValueError("max_difference must be nonnegative")
    leading = abs(poly.leading_coefficient)
    degree = poly.degree
    lower_sum = sum(abs(value) for value in poly.coefficients[:-1])
    dominance_bound = max(1, (2 * lower_sum + leading - 1) // leading)
    growth_bound = _smallest_growth_bound(leading, degree, 2 * max_difference)
    return max(dominance_bound, growth_bound)


def generate_distances(instance: InstanceSpec) -> DistanceData:
    """Generate every nonzero polynomial distance that fits the instance."""

    max_difference = instance.n - 1
    bound = certified_input_bound(instance.polynomial, max_difference)
    inputs = (
        range(1, bound + 1)
        if instance.input_domain == "positive"
        else range(-bound, bound + 1)
    )
    preimages: dict[int, list[int]] = {}
    for d in inputs:
        if d == 0:
            continue
        value = instance.polynomial.evaluate(d)
        if value == 0:
            continue
        distance = abs(value)
        if distance <= max_difference:
            preimages.setdefault(distance, []).append(d)
    values = tuple(sorted(preimages))
    return DistanceData(
        values=values,
        preimages={distance: tuple(sorted(preimages[distance])) for distance in values},
        input_bound=bound,
    )
