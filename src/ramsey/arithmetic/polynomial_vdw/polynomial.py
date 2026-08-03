from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Polynomial:
    """an exact integer polynomial with coefficients stored low degree first."""

    coefficients: tuple[int, ...]
    degree: int = field(init=False)

    def __post_init__(self) -> None:
        try:
            coefficients = tuple(self.coefficients)
        except TypeError as exc:
            raise TypeError("coefficients must be an iterable of integers") from exc
        if any(type(coefficient) is not int for coefficient in coefficients):
            raise TypeError("coefficients must be ordinary integers")
        while coefficients and coefficients[-1] == 0:
            coefficients = coefficients[:-1]
        if len(coefficients) < 2:
            raise ValueError("the polynomial must be nonzero and have positive degree")
        if coefficients[0] != 0:
            raise ValueError("the constant coefficient must be zero")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "degree", len(coefficients) - 1)

    @property
    def leading_coefficient(self) -> int:
        return self.coefficients[-1]

    def evaluate(self, value: int) -> int:
        """evaluate exactly with horner arithmetic."""
        if type(value) is not int:
            raise TypeError("the polynomial input must be an ordinary integer")
        result = 0
        for coefficient in reversed(self.coefficients):
            result = result * value + coefficient
        return result

    def __call__(self, value: int) -> int:
        return self.evaluate(value)


__all__ = ["Polynomial"]
