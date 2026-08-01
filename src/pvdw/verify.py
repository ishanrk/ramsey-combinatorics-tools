"""Independent verification against regenerated polynomial distances."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from pvdw.distances import DistanceData, generate_distances
from pvdw.graph import DistanceGraph
from pvdw.model import InstanceSpec


@dataclass(frozen=True)
class ColoringViolation:
    u: int
    v: int
    color: int
    difference: int
    polynomial_inputs: tuple[int, ...]


@dataclass(frozen=True)
class VerificationResult:
    valid: bool
    violation: ColoringViolation | None
    violation_count: int
    error: str | None = None


def _validate_coloring_shape(
    instance: InstanceSpec,
    coloring: Sequence[int],
) -> tuple[int, ...]:
    values = tuple(coloring)
    if len(values) != instance.n:
        raise ValueError(
            f"coloring length {len(values)} does not equal instance size {instance.n}"
        )
    for index, color in enumerate(values):
        if type(color) is not int or not 0 <= color < instance.colors:
            raise ValueError(
                f"color at vertex {index} must be an integer in 0..{instance.colors - 1}"
            )
    return values


def _violations_from_data(
    instance: InstanceSpec,
    coloring: tuple[int, ...],
    data: DistanceData,
) -> Iterator[ColoringViolation]:
    graph = DistanceGraph(instance.n, data.values)
    for u, v in graph.iter_edges():
        if coloring[u] == coloring[v]:
            difference = v - u
            yield ColoringViolation(
                u=u,
                v=v,
                color=coloring[u],
                difference=difference,
                polynomial_inputs=data.preimages[difference],
            )


def iter_violations(
    instance: InstanceSpec,
    coloring: Sequence[int],
) -> Iterator[ColoringViolation]:
    """Yield every conflict after independently regenerating the distances."""

    values = _validate_coloring_shape(instance, coloring)
    data = generate_distances(instance)
    yield from _violations_from_data(instance, values, data)


def verify_coloring(
    instance: InstanceSpec,
    coloring: Sequence[int],
) -> VerificationResult:
    """Validate shape, color range, and every polynomial-distance edge."""

    try:
        values = _validate_coloring_shape(instance, coloring)
    except (TypeError, ValueError) as error:
        return VerificationResult(False, None, 0, str(error))
    data = generate_distances(instance)
    first: ColoringViolation | None = None
    count = 0
    for violation in _violations_from_data(instance, values, data):
        if first is None:
            first = violation
        count += 1
    return VerificationResult(first is None, first, count)


def conflict_count(instance: InstanceSpec, coloring: Sequence[int]) -> int:
    """Count conflicts, raising when the coloring itself is malformed."""

    return sum(1 for _ in iter_violations(instance, coloring))
