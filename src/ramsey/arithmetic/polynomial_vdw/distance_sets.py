from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ramsey.arithmetic.polynomial_vdw.instance import (
    PolynomialVanDerWaerdenInstance,
)
from ramsey.arithmetic.polynomial_vdw.polynomial_values import (
    certified_input_bound,
)


@dataclass(frozen=True, slots=True)
class ForbiddenDistanceData:
    """the exact distances and polynomial inputs behind one graph."""

    distances: tuple[int, ...]
    preimages: Mapping[int, tuple[int, ...]]
    input_bound: int
    inspected_input_count: int

    def __post_init__(self) -> None:
        distances = tuple(self.distances)
        if distances != tuple(sorted(set(distances))):
            raise ValueError("distances must be sorted and unique")
        if any(type(distance) is not int or distance <= 0 for distance in distances):
            raise ValueError("distances must be positive ordinary integers")
        normalized = {
            distance: tuple(sorted(inputs))
            for distance, inputs in self.preimages.items()
        }
        if tuple(sorted(normalized)) != distances:
            raise ValueError("preimages must match the distance set")
        object.__setattr__(self, "distances", distances)
        object.__setattr__(self, "preimages", MappingProxyType(normalized))


def generate_forbidden_distances(
    instance: PolynomialVanDerWaerdenInstance,
) -> ForbiddenDistanceData:
    """generate every absolute nonzero polynomial distance fitting the interval."""
    if not isinstance(instance, PolynomialVanDerWaerdenInstance):
        raise TypeError("instance must be a PolynomialVanDerWaerdenInstance")

    max_difference = instance.max_difference
    bound = certified_input_bound(instance.polynomial, max_difference)
    if instance.input_domain == "positive":
        inputs = range(1, bound + 1)
        inspected_input_count = bound
    else:
        inputs = range(-bound, bound + 1)
        inspected_input_count = 2 * bound

    preimages: dict[int, list[int]] = {}
    for value in inputs:
        if value == 0:
            continue
        polynomial_value = instance.polynomial(value)
        if polynomial_value == 0:
            continue
        distance = abs(polynomial_value)
        if distance <= max_difference:
            preimages.setdefault(distance, []).append(value)

    distances = tuple(sorted(preimages))
    return ForbiddenDistanceData(
        distances=distances,
        preimages={
            distance: tuple(sorted(preimages[distance])) for distance in distances
        },
        input_bound=bound,
        inspected_input_count=inspected_input_count,
    )


def forbidden_distances(
    instance: PolynomialVanDerWaerdenInstance,
) -> tuple[int, ...]:
    return generate_forbidden_distances(instance).distances


__all__ = [
    "ForbiddenDistanceData",
    "forbidden_distances",
    "generate_forbidden_distances",
]
