from __future__ import annotations

import itertools

from hypothesis import given, settings, strategies as st

from pvdw.backends.base import SolveOptions
from pvdw.backends.pysat_backend import PySatBackend, available_pysat_solvers
from pvdw.model import InstanceSpec, PolynomialSpec, SolveStatus
from pvdw.modes.direct import EncodingOptions, solve_direct
from pvdw.modes.incremental import scan_incremental
from pvdw.modes.periodic import (
    build_periodic_model,
    lift_periodic,
    periodic_pattern_satisfies_model,
)
from pvdw.modes.repair import (
    build_repair_cnf,
    find_bad_edges,
    greedy_bad_edge_vertex_cover,
    solve_repair,
)
from pvdw.modes.twisted import (
    build_twisted_model,
    lift_twisted,
    twisted_pattern_satisfies_model,
)
from pvdw.runner import repair_from_backbone_search
from pvdw.distances import generate_distances
from pvdw.graph import DistanceGraph
from pvdw.verify import verify_coloring


def _solver_name() -> str:
    return available_pysat_solvers()[0]


def test_incremental_matches_separately_rebuilt_formulas() -> None:
    polynomial = PolynomialSpec((0, 0, 1))
    incremental = scan_incremental(
        polynomial,
        3,
        1,
        12,
        PySatBackend(_solver_name()),
        EncodingOptions(),
        SolveOptions(),
    )
    assert [step.n for step in incremental.steps] == list(range(1, 13))
    for step in incremental.steps:
        separate = solve_direct(
            InstanceSpec(polynomial, 3, step.n),
            EncodingOptions(),
            PySatBackend(_solver_name()),
            SolveOptions(),
        )
        assert step.status is separate.status


def test_incremental_stops_at_first_unsat() -> None:
    result = scan_incremental(
        PolynomialSpec((0, 1)),
        2,
        1,
        6,
        PySatBackend(_solver_name()),
        EncodingOptions(),
        SolveOptions(),
    )
    assert result.first_unsat_n == 3
    assert result.steps[-1].status is SolveStatus.UNSAT_FULL_MODEL


@st.composite
def periodic_cases(draw: st.DrawFn):
    n = draw(st.integers(min_value=1, max_value=10))
    colors = draw(st.integers(min_value=2, max_value=4))
    period = draw(st.integers(min_value=1, max_value=5))
    coefficient = draw(st.sampled_from([-2, -1, 1, 2]))
    instance = InstanceSpec(PolynomialSpec((0, coefficient)), colors, n)
    pattern = tuple(
        draw(
            st.lists(
                st.integers(min_value=0, max_value=colors - 1),
                min_size=period,
                max_size=period,
            )
        )
    )
    return instance, period, pattern


@given(periodic_cases())
@settings(max_examples=100, deadline=None)
def test_periodic_quotient_matches_explicit_lift(case) -> None:
    instance, period, pattern = case
    model = build_periodic_model(instance, period)
    quotient_valid = periodic_pattern_satisfies_model(model, pattern)
    lifted_valid = verify_coloring(
        instance, lift_periodic(pattern, instance.n)
    ).valid
    assert quotient_valid is lifted_valid


def test_periodic_distance_divisible_by_period_is_impossible() -> None:
    instance = InstanceSpec(PolynomialSpec((0, 1)), 3, 3)
    model = build_periodic_model(instance, 2)
    assert model.immediate_impossibility
    assert (0, 0) in model.edges


def test_twisted_exhaustive_quotient_lift_equivalence() -> None:
    for n in range(1, 8):
        for colors in range(2, 4):
            instance = InstanceSpec(PolynomialSpec((0, 0, 1)), colors, n)
            for period in range(1, 5):
                for twist in range(colors):
                    model = build_twisted_model(instance, period, twist)
                    for pattern in itertools.product(range(colors), repeat=period):
                        quotient_valid = twisted_pattern_satisfies_model(model, pattern)
                        lifted_valid = verify_coloring(
                            instance,
                            lift_twisted(
                                pattern, n, colors, period, twist
                            ),
                        ).valid
                        assert quotient_valid is lifted_valid


def test_twisted_self_shift_special_cases() -> None:
    instance = InstanceSpec(PolynomialSpec((0, 1)), 2, 2)
    zero_shift = build_twisted_model(instance, period=1, twist=0)
    assert zero_shift.immediate_impossibility
    nonzero_shift = build_twisted_model(instance, period=1, twist=1)
    assert not nonzero_shift.immediate_impossibility
    assert nonzero_shift.constraints == ()
    assert twisted_pattern_satisfies_model(nonzero_shift, (0,))
    assert verify_coloring(instance, lift_twisted((0,), 2, 2, 1, 1)).valid


def test_repair_identifies_covers_and_repairs_bad_edges() -> None:
    instance = InstanceSpec(PolynomialSpec((0, 1)), 4, 4)
    valid = (0, 1, 2, 3)
    corrupted = (0, 0, 2, 3)
    graph = DistanceGraph(instance.n, generate_distances(instance).values)
    bad_edges = find_bad_edges(graph, corrupted)
    assert bad_edges == ((0, 1),)
    editable = greedy_bad_edge_vertex_cover(bad_edges)
    assert all(u in editable or v in editable for u, v in bad_edges)
    reduced = build_repair_cnf(
        instance, graph, corrupted, editable, EncodingOptions()
    )
    assert reduced.decode_spec.backbone == corrupted
    result = solve_repair(
        instance,
        corrupted,
        PySatBackend(_solver_name()),
        EncodingOptions(),
        SolveOptions(),
        editable_strategy="greedy_vertex_cover",
    )
    assert result.status is SolveStatus.FOUND_WITNESS
    assert result.coloring is not None
    assert verify_coloring(instance, result.coloring).valid
    assert result.metadata["number_changed_colors"] >= 1


def test_positive_energy_backbone_is_retained_for_repair() -> None:
    instance = InstanceSpec(PolynomialSpec((0, 1)), 4, 4)
    corrupted = (0, 0, 2, 3)
    result = solve_repair(
        instance,
        corrupted,
        PySatBackend(_solver_name()),
        EncodingOptions(),
        SolveOptions(),
        backbone_metadata={
            "backbone_source": "periodic",
            "backbone_search_status": SolveStatus.UNKNOWN.value,
            "backbone_search_best_energy": 1,
        },
    )
    assert result.status is SolveStatus.FOUND_WITNESS
    assert result.metadata["backbone_search_status"] == SolveStatus.UNKNOWN.value
    assert result.metadata["initial_energy"] == 1


def test_positive_energy_periodic_search_is_passed_into_repair() -> None:
    instance = InstanceSpec(PolynomialSpec((0, 0, 1)), 3, 28)
    result = repair_from_backbone_search(
        instance,
        "periodic",
        PySatBackend(_solver_name()),
        EncodingOptions(),
        SolveOptions(timeout_seconds=5, seed=7),
        period=6,
        max_expansions=3,
        potts_restarts=1,
        potts_steps=1,
    )
    assert result.status is SolveStatus.FOUND_WITNESS
    assert result.metadata["backbone_source"] == "periodic"
    assert result.metadata["backbone_search_status"] == SolveStatus.UNKNOWN.value
    assert result.metadata["backbone_search_best_energy"] > 0
    assert result.metadata["initial_energy"] > 0
    assert result.coloring is not None
    assert verify_coloring(instance, result.coloring).valid
