from __future__ import annotations

import os

import pytest

from pvdw.backends.base import SolveOptions
from pvdw.backends.pysat_backend import PySatBackend, available_pysat_solvers
from pvdw.model import InstanceSpec, PolynomialSpec, SolveStatus
from pvdw.modes.direct import EncodingOptions, solve_direct
from pvdw.modes.periodic import solve_periodic


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("PVDW_RUN_SLOW") != "1", reason="set PVDW_RUN_SLOW=1"
    ),
]


def exact_backend() -> PySatBackend:
    return PySatBackend(available_pysat_solvers()[0])


@pytest.mark.parametrize(
    ("instance", "expected"),
    [
        (InstanceSpec(PolynomialSpec((0, 0, 1)), 4, 58), SolveStatus.UNSAT_FULL_MODEL),
        (InstanceSpec(PolynomialSpec((0, 0, 0, 1)), 3, 522), SolveStatus.UNSAT_FULL_MODEL),
    ],
)
def test_optional_exact_boundaries(instance, expected) -> None:
    result = solve_direct(
        instance,
        EncodingOptions(),
        exact_backend(),
        SolveOptions(timeout_seconds=30),
    )
    assert result.status is expected


@pytest.mark.parametrize(
    "instance",
    [
        InstanceSpec(PolynomialSpec((0, 0, 1)), 4, 57),
        InstanceSpec(PolynomialSpec((0, 0, 0, 1)), 3, 521),
    ],
)
def test_optional_large_lower_witnesses(instance) -> None:
    result = solve_direct(
        instance,
        EncodingOptions(),
        exact_backend(),
        SolveOptions(timeout_seconds=30, seed=0),
    )
    assert result.status is SolveStatus.FOUND_WITNESS


@pytest.mark.parametrize(
    ("instance", "period"),
    [
        (InstanceSpec(PolynomialSpec((0, 0, 0, 1)), 5, 9261), 63),
        (InstanceSpec(PolynomialSpec((0, 0, 1)), 8, 841), 29),
        (InstanceSpec(PolynomialSpec((0, 0, 0, 1)), 4, 2197), 13),
    ],
)
def test_optional_period_regressions(instance, period) -> None:
    result = solve_periodic(
        instance,
        period,
        EncodingOptions(),
        exact_backend(),
        SolveOptions(timeout_seconds=30),
    )
    assert result.status is SolveStatus.FOUND_WITNESS
