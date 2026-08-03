import itertools
import random

import pytest

from ramsey.solvers.sat import (
    CNFFormula,
    SatStatus,
    available_sat_backends,
    solve_cnf,
)


def test_cnf_validation_and_normalization() -> None:
    formula = CNFFormula.from_clauses(
        2,
        [
            (1, 1),
            (1, -1),
            (),
        ],
    )

    assert formula.clauses == ((1,), ())
    assert formula.clause_count == 2
    assert formula.literal_count == 1
    with pytest.raises(ValueError):
        CNFFormula.from_clauses(2, [(0,)])
    with pytest.raises(ValueError):
        CNFFormula.from_clauses(2, [(3,)])


def test_builtin_solver_returns_a_complete_checked_model() -> None:
    formula = CNFFormula.from_clauses(
        3,
        [
            (1, 2),
            (-1, 2),
            (1, -2),
        ],
    )

    result = solve_cnf(formula, backend="dpll")
    assert result.status is SatStatus.SAT
    assert result.model is not None
    assert len(result.model) == 3
    assert formula.is_satisfied_by(result.model)
    assert solve_cnf(formula, backend="builtin").is_sat


def test_builtin_solver_proves_unsat_and_honors_assumptions() -> None:
    unsat = CNFFormula.from_clauses(1, [(1,), (-1,)])
    assert solve_cnf(unsat).status is SatStatus.UNSAT

    formula = CNFFormula.from_clauses(2, [(1, 2)])
    assert solve_cnf(formula, assumptions=(-1, -2)).is_unsat
    assert solve_cnf(formula, assumptions=(1,)).is_sat


def test_empty_formula_is_sat() -> None:
    result = solve_cnf(CNFFormula.from_clauses(0, []))

    assert result.is_sat
    assert result.model == ()


def test_backend_selection_is_explicit() -> None:
    assert {"dpll", "builtin"} <= set(available_sat_backends())
    with pytest.raises(ValueError):
        solve_cnf(CNFFormula.from_clauses(0, []), backend="not-a-solver")


def test_missing_optional_pysat_returns_an_error() -> None:
    if "pysat" in available_sat_backends():
        pytest.skip("python-sat is installed here")

    result = solve_cnf(CNFFormula.from_clauses(1, [(1,)]), backend="pysat")
    assert result.status is SatStatus.ERROR
    assert result.message == "python-sat is not installed"


def test_builtin_solver_matches_brute_force_on_random_formulas() -> None:
    rng = random.Random(40191)
    for _ in range(180):
        variable_count = rng.randint(0, 7)
        clauses = []
        for _ in range(rng.randint(0, 14)):
            if variable_count == 0:
                clauses.append(())
                continue
            clauses.append(
                tuple(
                    rng.choice((-1, 1)) * rng.randint(1, variable_count)
                    for _ in range(rng.randint(0, 4))
                )
            )
        formula = CNFFormula.from_clauses(variable_count, clauses)
        brute_sat = any(
            formula.is_satisfied_by(
                tuple(
                    variable + 1 if value else -(variable + 1)
                    for variable, value in enumerate(values)
                )
            )
            for values in itertools.product((False, True), repeat=variable_count)
        )

        result = solve_cnf(formula)
        assert result.is_sat == brute_sat
        if result.model is not None:
            assert formula.is_satisfied_by(result.model)
