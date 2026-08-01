"""Immutable core specifications and solver result types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Literal, Mapping


def _require_plain_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an ordinary Python integer")
    return value


@dataclass(frozen=True)
class PolynomialSpec:
    """An integer polynomial, with coefficients stored low degree first."""

    coefficients: tuple[int, ...]

    def __post_init__(self) -> None:
        coefficients = tuple(self.coefficients)
        for coefficient in coefficients:
            _require_plain_int(coefficient, "coefficient")
        while coefficients and coefficients[-1] == 0:
            coefficients = coefficients[:-1]
        if len(coefficients) < 2:
            raise ValueError("polynomial must be nonzero and have degree at least one")
        if coefficients[0] != 0:
            raise ValueError("the constant coefficient must be zero")
        object.__setattr__(self, "coefficients", coefficients)

    @property
    def degree(self) -> int:
        return len(self.coefficients) - 1

    @property
    def leading_coefficient(self) -> int:
        return self.coefficients[-1]

    def evaluate(self, d: int) -> int:
        """Evaluate exactly with integer Horner arithmetic."""

        _require_plain_int(d, "polynomial input")
        value = 0
        for coefficient in reversed(self.coefficients):
            value = value * d + coefficient
        return value


@dataclass(frozen=True)
class InstanceSpec:
    """A coloring instance on the vertices ``0, ..., n - 1``."""

    polynomial: PolynomialSpec
    colors: int
    n: int
    input_domain: Literal["all_nonzero", "positive"] = "all_nonzero"

    def __post_init__(self) -> None:
        if not isinstance(self.polynomial, PolynomialSpec):
            raise TypeError("polynomial must be a PolynomialSpec")
        _require_plain_int(self.colors, "colors")
        _require_plain_int(self.n, "n")
        if self.colors < 2:
            raise ValueError("colors must be at least 2")
        if self.n < 1:
            raise ValueError("n must be at least 1")
        if self.input_domain not in ("all_nonzero", "positive"):
            raise ValueError("input_domain must be 'all_nonzero' or 'positive'")


class ModelScope(str, Enum):
    FULL = "full"
    PERIODIC = "periodic"
    TWISTED = "twisted"
    REPAIR = "repair"


class SolveStatus(str, Enum):
    FOUND_WITNESS = "found_witness"
    UNSAT_FULL_MODEL = "unsat_full_model"
    NO_WITNESS_IN_RESTRICTED_MODEL = "no_witness_in_restricted_model"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"
    ERROR = "error"


@dataclass(frozen=True)
class SolveResult:
    """The outcome of a search, with scope kept distinct from status."""

    status: SolveStatus
    scope: ModelScope
    elapsed_seconds: float
    backend: str
    coloring: tuple[int, ...] | None
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.status, SolveStatus):
            raise TypeError("status must be a SolveStatus")
        if not isinstance(self.scope, ModelScope):
            raise TypeError("scope must be a ModelScope")
        if not isinstance(self.elapsed_seconds, (int, float)) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be nonnegative")
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("backend must be a nonempty string")
        if self.coloring is not None:
            coloring = tuple(self.coloring)
            for color in coloring:
                _require_plain_int(color, "color")
            object.__setattr__(self, "coloring", coloring)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
