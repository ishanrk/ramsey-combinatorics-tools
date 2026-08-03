from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SatStatus(str, Enum):
    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class SatResult:
    """the normalized result returned by any sat backend."""

    status: SatStatus
    backend: str
    model: tuple[int, ...] | None
    elapsed_seconds: float
    message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, SatStatus):
            raise TypeError("status must be a SatStatus")
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("backend must be a nonempty string")
        if not isinstance(self.elapsed_seconds, (int, float)):
            raise TypeError("elapsed_seconds must be numeric")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be nonnegative")
        if self.status is SatStatus.SAT and self.model is None:
            raise ValueError("a sat result needs a model")
        if self.status is not SatStatus.SAT and self.model is not None:
            raise ValueError("only a sat result may contain a model")
        if self.model is not None:
            model = tuple(self.model)
            if any(type(literal) is not int or literal == 0 for literal in model):
                raise ValueError("model literals must be nonzero ordinary integers")
            object.__setattr__(self, "model", model)

    @property
    def is_sat(self) -> bool:
        return self.status is SatStatus.SAT

    @property
    def is_unsat(self) -> bool:
        return self.status is SatStatus.UNSAT


__all__ = ["SatResult", "SatStatus"]
