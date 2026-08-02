"""Exact three-color signed-drift domains and partition constraints."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PartitionRelation:
    total: int
    parts: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.total) is not int or self.total <= 0:
            raise ValueError("relation total must be a positive ordinary integer")
        parts = tuple(self.parts)
        if not parts or any(type(part) is not int or part <= 0 for part in parts):
            raise ValueError("relation parts must be positive ordinary integers")
        if sum(parts) != self.total:
            raise ValueError(
                f"partition relation does not add up: {self.total} != {sum(parts)}"
            )
        object.__setattr__(self, "parts", parts)


@dataclass(frozen=True)
class DriftAssignment:
    values: tuple[tuple[int, int], ...]

    @property
    def mapping(self) -> dict[int, int]:
        return dict(self.values)


@dataclass(frozen=True)
class DriftEnumerationResult:
    distances: tuple[int, ...]
    domains: tuple[tuple[int, tuple[int, ...]], ...]
    relations: tuple[PartitionRelation, ...]
    survivor_counts: tuple[int, ...]
    assignments: tuple[DriftAssignment, ...]


def drift_domain(distance: int) -> tuple[int, ...]:
    if type(distance) is not int or distance <= 0:
        raise ValueError("distance must be a positive ordinary integer")
    limit = (distance + 1) // 3 + 1
    parity = (distance + 1) & 1
    return tuple(
        value
        for value in range(-limit, limit + 1)
        if 3 * abs(value) <= distance + 1 and (value & 1) == parity
    )


def partition_error_domain(part_count: int) -> tuple[int, ...]:
    if type(part_count) is not int or part_count < 1:
        raise ValueError("part_count must be a positive ordinary integer")
    limit = (part_count + 1) // 3 + 1
    parity = (part_count - 1) & 1
    return tuple(
        value
        for value in range(-limit, limit + 1)
        if abs(3 * value) <= part_count + 1 and (value & 1) == parity
    )


def relation_holds(
    q_values: Mapping[int, int],
    relation: PartitionRelation,
) -> bool:
    try:
        difference = q_values[relation.total] - sum(
            q_values[part] for part in relation.parts
        )
    except KeyError as error:
        raise ValueError(f"drift assignment omits distance {error.args[0]}") from error
    return difference in partition_error_domain(len(relation.parts))


def verify_drift_assignment(
    assignment: DriftAssignment,
    relations: Sequence[PartitionRelation],
    *,
    positive_drift: int | None = None,
) -> bool:
    values = assignment.mapping
    return (
        all(value in drift_domain(distance) for distance, value in values.items())
        and all(relation_holds(values, relation) for relation in relations)
        and (
            positive_drift is None
            or positive_drift in values
            and values[positive_drift] > 0
        )
    )


def enumerate_drift_assignments(
    distances: Sequence[int],
    relations: Sequence[PartitionRelation],
    *,
    positive_drift: int | None = None,
) -> DriftEnumerationResult:
    """Incrementally extend assignments and filter after every relation."""

    normalized_distances = tuple(sorted(set(distances)))
    if any(type(distance) is not int or distance <= 0 for distance in normalized_distances):
        raise ValueError("distances must be positive ordinary integers")
    relation_tuple = tuple(relations)
    used_distances = {
        value
        for relation in relation_tuple
        for value in (relation.total, *relation.parts)
    }
    if not used_distances <= set(normalized_distances):
        raise ValueError("a relation uses a distance outside the declared domain")
    if positive_drift is not None and positive_drift not in normalized_distances:
        raise ValueError("positive_drift must be one of the declared distances")
    domains = {distance: drift_domain(distance) for distance in normalized_distances}
    assignments: list[dict[int, int]] = [{}]
    survivor_counts: list[int] = []
    for relation in relation_tuple:
        mentioned = tuple(dict.fromkeys((relation.total, *relation.parts)))
        extended: list[dict[int, int]] = []
        for assignment in assignments:
            missing = tuple(value for value in mentioned if value not in assignment)
            choices = tuple(
                tuple(
                    drift
                    for drift in domains[distance]
                    if distance != positive_drift or drift > 0
                )
                for distance in missing
            )
            for values in itertools.product(*choices):
                candidate = {**assignment, **dict(zip(missing, values))}
                if relation_holds(candidate, relation):
                    extended.append(candidate)
        assignments = extended
        survivor_counts.append(len(assignments))
    unmentioned = tuple(
        distance
        for distance in normalized_distances
        if all(distance not in assignment for assignment in assignments)
    )
    if unmentioned:
        completed = []
        for assignment in assignments:
            choices = tuple(
                tuple(
                    drift
                    for drift in domains[distance]
                    if distance != positive_drift or drift > 0
                )
                for distance in unmentioned
            )
            for values in itertools.product(*choices):
                completed.append({**assignment, **dict(zip(unmentioned, values))})
        assignments = completed
    canonical = tuple(
        DriftAssignment(tuple((distance, assignment[distance]) for distance in normalized_distances))
        for assignment in sorted(
            assignments,
            key=lambda values: tuple(values[distance] for distance in normalized_distances),
        )
    )
    return DriftEnumerationResult(
        normalized_distances,
        tuple((distance, domains[distance]) for distance in normalized_distances),
        relation_tuple,
        tuple(survivor_counts),
        canonical,
    )


CUBE_DISTANCES = (1, 8, 27, 64, 125, 216, 343, 512)
CUBE_PARTITION_RELATIONS = (
    PartitionRelation(27, (8, 8, 8, 1, 1, 1)),
    PartitionRelation(64, (8,) * 8),
    PartitionRelation(64, (27, 27, 8, 1, 1)),
    PartitionRelation(125, (27, 27, 27, 27, 8, 8, 1)),
    PartitionRelation(216, (27, 64, 125)),
    PartitionRelation(216, (27,) * 8),
    PartitionRelation(216, (8,) * 27),
    PartitionRelation(343, (216, 125, 1, 1)),
    PartitionRelation(512, (64,) * 8),
    PartitionRelation(512, (343, 125, 27, 8, 8, 1)),
    PartitionRelation(512, (125, 125, 125, 125, 8, 1, 1, 1, 1)),
)


def cube_drift_assignments() -> DriftEnumerationResult:
    result = enumerate_drift_assignments(
        CUBE_DISTANCES, CUBE_PARTITION_RELATIONS, positive_drift=8
    )
    expected = (
        (0, 1, 4, 9, 18, 31, 48, 71),
        (0, 1, 4, 9, 18, 31, 48, 73),
        (0, 1, 4, 9, 18, 31, 50, 73),
        (0, 1, 4, 9, 18, 31, 50, 75),
    )
    observed = tuple(
        tuple(assignment.mapping[distance] for distance in CUBE_DISTANCES)
        for assignment in result.assignments
    )
    if observed != expected:
        raise RuntimeError(f"cube drift regression differs: observed {observed}")
    return result
