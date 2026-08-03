from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ramsey.arithmetic.polynomial_vdw.encode import (
    color_variable,
    encode_polynomial_vdw_cnf,
)
from ramsey.arithmetic.polynomial_vdw.instance import (
    PolynomialVanDerWaerdenInstance,
)
from ramsey.arithmetic.polynomial_vdw.verify import verify_coloring
from ramsey.solvers.sat import CNFFormula, SatResult, SatStatus, solve_cnf


def decode_coloring_model(
    instance: PolynomialVanDerWaerdenInstance,
    model: Iterable[int],
) -> tuple[int, ...]:
    """turn a one-hot dimacs model back into vertex colors."""
    if not isinstance(instance, PolynomialVanDerWaerdenInstance):
        raise TypeError("instance must be a PolynomialVanDerWaerdenInstance")
    values: dict[int, bool] = {}
    variable_count = instance.n * instance.colors
    for literal in model:
        if type(literal) is not int:
            raise TypeError("model literals must be ordinary integers")
        variable = abs(literal)
        if literal == 0 or variable > variable_count:
            raise ValueError("the model contains an invalid literal")
        value = literal > 0
        if variable in values and values[variable] != value:
            raise ValueError("the model assigns a variable twice")
        values[variable] = value

    coloring: list[int] = []
    for vertex in range(instance.n):
        selected = [
            color
            for color in range(instance.colors)
            if values.get(color_variable(vertex, color, instance.colors), False)
        ]
        if len(selected) != 1:
            raise ValueError("the model does not select exactly one color per vertex")
        coloring.append(selected[0])
    return tuple(coloring)


@dataclass(frozen=True, slots=True)
class PolynomialVDWSatResult:
    """a sat result together with its decoded polynomial coloring."""

    instance: PolynomialVanDerWaerdenInstance
    formula: CNFFormula
    sat_result: SatResult
    coloring: tuple[int, ...] | None

    @property
    def status(self) -> SatStatus:
        return self.sat_result.status

    @property
    def backend(self) -> str:
        return self.sat_result.backend


def solve_polynomial_vdw(
    instance: PolynomialVanDerWaerdenInstance,
    *,
    solver: str = "dpll",
    fix_first_color: bool = True,
) -> PolynomialVDWSatResult:
    """encode and solve one polynomial vdw instance with a chosen sat solver."""
    if not isinstance(instance, PolynomialVanDerWaerdenInstance):
        raise TypeError("instance must be a PolynomialVanDerWaerdenInstance")
    formula = encode_polynomial_vdw_cnf(
        instance,
        fix_first_color=fix_first_color,
    )
    sat_result = solve_cnf(formula, backend=solver)
    coloring = None
    if sat_result.status is SatStatus.SAT:
        if sat_result.model is None:
            raise RuntimeError("the sat backend returned no model")
        coloring = decode_coloring_model(instance, sat_result.model)
        if not verify_coloring(instance, coloring):
            raise RuntimeError("the sat backend returned an invalid coloring")
    return PolynomialVDWSatResult(
        instance=instance,
        formula=formula,
        sat_result=sat_result,
        coloring=coloring,
    )


solve = solve_polynomial_vdw

__all__ = [
    "PolynomialVDWSatResult",
    "decode_coloring_model",
    "solve",
    "solve_polynomial_vdw",
]
