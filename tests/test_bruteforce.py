from __future__ import annotations

import pytest

from pvdw.backends.bruteforce import solve_bruteforce
from pvdw.model import InstanceSpec, PolynomialSpec, SolveStatus
from pvdw.verify import verify_coloring


def square_instance(n: int) -> InstanceSpec:
    return InstanceSpec(PolynomialSpec((0, 0, 1)), 3, n)


def test_square_difference_regression_28_has_witness() -> None:
    result = solve_bruteforce(square_instance(28))
    assert result.status is SolveStatus.FOUND_WITNESS
    assert result.coloring is not None
    assert verify_coloring(square_instance(28), result.coloring).valid


def test_square_difference_regression_29_is_unsat() -> None:
    result = solve_bruteforce(square_instance(29))
    assert result.status is SolveStatus.UNSAT_FULL_MODEL
    assert result.coloring is None


def test_bruteforce_size_guard() -> None:
    with pytest.raises(ValueError, match="limit"):
        solve_bruteforce(square_instance(5), size_limit=4)
