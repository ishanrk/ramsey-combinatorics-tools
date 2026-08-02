from __future__ import annotations

import os
import random
import stat
import textwrap

import pytest

from pvdw.backends.base import SolveOptions
from pvdw.backends.external_sat import ExternalSatBackend, generic_dimacs_config
from pvdw.backends.potts import PottsState
from pvdw.backends.pysat_backend import PySatBackend, available_pysat_solvers
from pvdw.backends.ubcsat import UbcsatBackend, UbcsatOptions
from pvdw.model import InstanceSpec, PolynomialSpec, SolveStatus
from pvdw.modes.direct import EncodingOptions, build_direct_problem, solve_direct


def _script(tmp_path, body: str):
    path = tmp_path / "solver"
    path.write_text("#!/bin/sh\n" + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _complete_triangle_problem():
    instance = InstanceSpec(PolynomialSpec((0, 1)), 3, 3)
    return instance, build_direct_problem(instance, EncodingOptions())


def test_pysat_backend_probes_and_solves_sat_and_unsat() -> None:
    available = available_pysat_solvers()
    assert available
    backend = PySatBackend(available[0])
    sat_instance = InstanceSpec(PolynomialSpec((0, 0, 1)), 3, 10)
    sat = solve_direct(
        sat_instance, EncodingOptions(), backend, SolveOptions(timeout_seconds=3)
    )
    assert sat.status is SolveStatus.FOUND_WITNESS
    assert sat.metadata["verification"] == "valid"

    unsat_instance = InstanceSpec(PolynomialSpec((0, 1)), 2, 3)
    unsat = solve_direct(
        unsat_instance,
        EncodingOptions(),
        PySatBackend(available[0]),
        SolveOptions(timeout_seconds=3),
    )
    assert unsat.status is SolveStatus.UNSAT_FULL_MODEL


def test_external_multiline_sat_model(tmp_path) -> None:
    executable = _script(
        tmp_path,
        """
        echo 's SATISFIABLE'
        echo 'v 1 -2 -3 -4 5'
        echo 'v -6 -7 -8 9 0'
        exit 10
        """,
    )
    instance, problem = _complete_triangle_problem()
    backend = ExternalSatBackend(generic_dimacs_config(executable, name="mock"))
    result = backend.solve(problem, SolveOptions(timeout_seconds=2))
    assert result.status is SolveStatus.FOUND_WITNESS
    assert result.coloring == (0, 1, 2)
    assert result.metadata["model_parsing"] == "complete"


def test_external_unsat_and_malformed_responses(tmp_path) -> None:
    _, problem = _complete_triangle_problem()
    unsat = _script(tmp_path, "echo 's UNSATISFIABLE'\nexit 20\n")
    result = ExternalSatBackend(generic_dimacs_config(unsat)).solve(
        problem, SolveOptions(timeout_seconds=2)
    )
    assert result.status is SolveStatus.UNSAT_FULL_MODEL

    missing_model = _script(tmp_path, "echo 's SATISFIABLE'\nexit 10\n")
    result = ExternalSatBackend(generic_dimacs_config(missing_model)).solve(
        problem, SolveOptions(timeout_seconds=2)
    )
    assert result.status is SolveStatus.ERROR
    assert "model" in str(result.metadata["model_parsing"]).lower()

    bad_code = _script(tmp_path, "echo 'c no status'\nexit 7\n")
    result = ExternalSatBackend(generic_dimacs_config(bad_code)).solve(
        problem, SolveOptions(timeout_seconds=2)
    )
    assert result.status is SolveStatus.ERROR


def test_external_timeout_terminates_process_group(tmp_path) -> None:
    executable = _script(tmp_path, "sleep 5\n")
    _, problem = _complete_triangle_problem()
    result = ExternalSatBackend(generic_dimacs_config(executable)).solve(
        problem, SolveOptions(timeout_seconds=0.05)
    )
    assert result.status is SolveStatus.TIMEOUT


def test_ubcsat_sat_and_exhaustion_never_unsat(tmp_path) -> None:
    _, problem = _complete_triangle_problem()
    sat = _script(
        tmp_path,
        "echo 's SATISFIABLE'\necho 'v 1 -2 -3 -4 5 -6 -7 -8 9 0'\nexit 0\n",
    )
    backend = UbcsatBackend(sat, UbcsatOptions(), probe=False)
    result = backend.solve(problem, SolveOptions(timeout_seconds=2))
    assert result.status is SolveStatus.FOUND_WITNESS

    exhausted = _script(tmp_path, "echo 'c exhausted'\nexit 0\n")
    result = UbcsatBackend(exhausted, UbcsatOptions(), probe=False).solve(
        problem, SolveOptions(timeout_seconds=2)
    )
    assert result.status is SolveStatus.UNKNOWN
    assert result.status is not SolveStatus.UNSAT_FULL_MODEL

    claimed_unsat = _script(tmp_path, "echo 's UNSATISFIABLE'\nexit 20\n")
    result = UbcsatBackend(claimed_unsat, UbcsatOptions(), probe=False).solve(
        problem, SolveOptions(timeout_seconds=2)
    )
    assert result.status is SolveStatus.UNKNOWN


def test_potts_incremental_energy_invariant_for_shifted_constraints() -> None:
    random_source = random.Random(12345)
    constraints = tuple(
        (left, right, random_source.randrange(4))
        for left in range(10)
        for right in range(left + 1, 10)
        if random_source.random() < 0.3
    )
    state = PottsState(
        10,
        4,
        constraints,
        [random_source.randrange(4) for _ in range(10)],
    )
    assert state.energy == state.full_energy()
    for _ in range(500):
        vertex = random_source.randrange(10)
        state.recolor(vertex, random_source.randrange(4))
        assert state.energy == state.full_energy()


@pytest.mark.skipif("UBCSAT_BIN" not in os.environ, reason="UBCSAT_BIN is not set")
def test_ubcsat_optional_integration() -> None:
    _, problem = _complete_triangle_problem()
    backend = UbcsatBackend(os.environ["UBCSAT_BIN"])
    result = backend.solve(problem, SolveOptions(timeout_seconds=5))
    assert result.status in {
        SolveStatus.FOUND_WITNESS,
        SolveStatus.UNKNOWN,
        SolveStatus.TIMEOUT,
    }
