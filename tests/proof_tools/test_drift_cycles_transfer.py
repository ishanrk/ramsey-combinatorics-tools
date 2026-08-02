from __future__ import annotations

from pvdw.polynomial import parse_polynomial
from pvdw.proof_tools.difference_cycles import (
    KNOWN_CUBE_WALKS,
    build_difference_edges,
    canonicalize_cycle,
    find_negative_cycle,
    verify_negative_cycle,
    walk_negative_cycle,
)
from pvdw.proof_tools.drift import (
    CUBE_DISTANCES,
    CUBE_PARTITION_RELATIONS,
    PartitionRelation,
    cube_drift_assignments,
    drift_domain,
    partition_error_domain,
    relation_holds,
    verify_drift_assignment,
)
from pvdw.proof_tools.transfer import search_common_scale, verify_transfer_witness


def test_drift_domains_and_partition_relation() -> None:
    assert drift_domain(1) == (0,)
    assert drift_domain(8) == (-3, -1, 1, 3)
    assert partition_error_domain(6) == (-1, 1)
    relation = PartitionRelation(27, (8, 8, 8, 1, 1, 1))
    assert relation_holds({1: 0, 8: 1, 27: 4}, relation)


def test_cube_relations_derive_exactly_four_cases() -> None:
    result = cube_drift_assignments()
    assert result.relations == CUBE_PARTITION_RELATIONS
    assert result.survivor_counts == (3, 9, 6, 18, 18, 15, 10, 20, 80, 5, 4)
    observed = tuple(
        tuple(assignment.mapping[distance] for distance in CUBE_DISTANCES)
        for assignment in result.assignments
    )
    assert observed == (
        (0, 1, 4, 9, 18, 31, 48, 71),
        (0, 1, 4, 9, 18, 31, 48, 73),
        (0, 1, 4, 9, 18, 31, 50, 73),
        (0, 1, 4, 9, 18, 31, 50, 75),
    )
    assert all(
        verify_drift_assignment(
            assignment, result.relations, positive_drift=8
        )
        for assignment in result.assignments
    )


def test_known_and_automatic_negative_cycles_for_every_case() -> None:
    result = cube_drift_assignments()
    for assignment, walk in zip(result.assignments, KNOWN_CUBE_WALKS):
        known = walk_negative_cycle(walk, assignment.mapping)
        assert known.total_weight == -1
        assert verify_negative_cycle(known)
        automatic = find_negative_cycle(
            range(522), build_difference_edges(522, assignment.mapping)
        )
        assert automatic is not None
        assert verify_negative_cycle(automatic)
        assert verify_negative_cycle(canonicalize_cycle(automatic))


def test_common_scale_transfer_regression() -> None:
    witness = search_common_scale(
        parse_polynomial("x^2+x"),
        (1, 2, 5, 7, 12, 15),
        max_scale=100,
        input_bound=100,
    )
    assert witness is not None
    assert witness.scale == 6
    assert tuple(witness.inputs[difference] for difference in witness.source_differences) == (
        2, 3, 5, 6, 8, 9
    )
    assert verify_transfer_witness(witness)
