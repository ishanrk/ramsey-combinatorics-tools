from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ramsey.arithmetic.polynomial_vdw.polynomial import Polynomial

InputDomain = Literal["all_nonzero", "positive"]


@dataclass(frozen=True, slots=True)
class PolynomialVanDerWaerdenInstance:
    """a polynomial difference-coloring problem on zero through n minus one."""

    polynomial: Polynomial
    colors: int
    n: int
    input_domain: InputDomain = "all_nonzero"

    def __post_init__(self) -> None:
        if not isinstance(self.polynomial, Polynomial):
            raise TypeError("polynomial must be a Polynomial")
        if type(self.colors) is not int:
            raise TypeError("colors must be an ordinary integer")
        if self.colors < 2:
            raise ValueError("colors must be at least two")
        if type(self.n) is not int:
            raise TypeError("n must be an ordinary integer")
        if self.n < 1:
            raise ValueError("n must be at least one")
        if self.input_domain not in ("all_nonzero", "positive"):
            raise ValueError("input_domain must be all_nonzero or positive")

    @property
    def max_difference(self) -> int:
        return self.n - 1


PolynomialVDWInstance = PolynomialVanDerWaerdenInstance

__all__ = [
    "InputDomain",
    "PolynomialVDWInstance",
    "PolynomialVanDerWaerdenInstance",
]
