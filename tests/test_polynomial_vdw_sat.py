import itertools

import pytest

from ramsey.arithmetic.polynomial_vdw import (
    Polynomial,
    PolynomialVanDerWaerdenInstance,
    color_variable,
    decode_coloring_model,
    encode_polynomial_vdw_cnf,
    solve_polynomial_vdw,
    verify_coloring,
)
from ramsey.solvers.sat import SatStatus


def _model_for_coloring(
    instance: PolynomialVanDerWaerdenInstance,
    coloring: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        variable
        if coloring[vertex] == color
        else -variable
        for vertex in range(instance.n)
        for color in range(instance.colors)
        for variable in (color_variable(vertex, color, instance.colors),)
    )


def test_polynomial_graph_encoding_has_the_expected_shape() -> None:
    instance = PolynomialVanDerWaerdenInstance(Polynomial((0, 1)), 3, 3)
    formula = encode_polynomial_vdw_cnf(instance)

    assert formula.variable_count == 9
    assert formula.clause_count == 22
    assert formula.clauses[-1] == (1,)
    assert formula.is_satisfied_by(_model_for_coloring(instance, (0, 1, 2)))
    assert not formula.is_satisfied_by(_model_for_coloring(instance, (0, 0, 1)))


def test_solver_finds_and_independently_checks_a_coloring() -> None:
    instance = PolynomialVanDerWaerdenInstance(Polynomial((0, 1)), 3, 3)
    result = solve_polynomial_vdw(instance, solver="dpll")

    assert result.status is SatStatus.SAT
    assert result.backend == "dpll"
    assert result.coloring is not None
    assert result.coloring[0] == 0
    assert verify_coloring(instance, result.coloring)
    assert result.sat_result.model is not None
    assert result.formula.is_satisfied_by(result.sat_result.model)


def test_solver_proves_a_small_instance_unsat() -> None:
    instance = PolynomialVanDerWaerdenInstance(Polynomial((0, 1)), 3, 4)
    result = solve_polynomial_vdw(instance, solver="builtin")

    assert result.status is SatStatus.UNSAT
    assert result.coloring is None


def test_decoding_rejects_malformed_models() -> None:
    instance = PolynomialVanDerWaerdenInstance(Polynomial((0, 0, 1)), 3, 5)

    with pytest.raises(ValueError):
        decode_coloring_model(instance, ())
    with pytest.raises(ValueError):
        decode_coloring_model(instance, (1, -1))
    with pytest.raises(ValueError):
        decode_coloring_model(instance, (100,))


def test_verifier_rejects_bad_colorings() -> None:
    instance = PolynomialVanDerWaerdenInstance(Polynomial((0, 1)), 3, 3)

    assert verify_coloring(instance, (0, 1, 2))
    assert not verify_coloring(instance, (0, 0, 1))
    assert not verify_coloring(instance, (0, 1))
    assert not verify_coloring(instance, (0, 1, 3))


def test_encoding_matches_every_small_coloring() -> None:
    instance = PolynomialVanDerWaerdenInstance(Polynomial((0, 0, 1)), 3, 5)
    formula = encode_polynomial_vdw_cnf(instance, fix_first_color=False)

    for coloring in itertools.product(range(instance.colors), repeat=instance.n):
        assert formula.is_satisfied_by(
            _model_for_coloring(instance, coloring)
        ) is verify_coloring(instance, coloring)
