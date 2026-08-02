"""Validated JSON witnesses and compact base-36 coloring words."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from pvdw.model import InstanceSpec, ModelScope, PolynomialSpec
from pvdw.verify import verify_coloring


FORMAT_VERSION = 1
COMPACT_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class WitnessFormatError(ValueError):
    """Raised for malformed, inconsistent, or invalid witness data."""


@dataclass(frozen=True)
class Witness:
    instance: InstanceSpec
    scope: ModelScope
    coloring: tuple[int, ...]
    backend: str
    seed: int
    elapsed_seconds: float
    scope_metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": FORMAT_VERSION,
            "instance": {
                "coefficients": list(self.instance.polynomial.coefficients),
                "colors": self.instance.colors,
                "n": self.instance.n,
                "input_domain": self.instance.input_domain,
            },
            "scope": {"mode": self.scope.value, **dict(self.scope_metadata)},
            "coloring": list(self.coloring),
            "search": {
                "backend": self.backend,
                "seed": self.seed,
                "elapsed_seconds": self.elapsed_seconds,
            },
            "verification": {"valid": True},
        }


def coloring_to_word(coloring: tuple[int, ...] | list[int], colors: int) -> str:
    if type(colors) is not int or not 2 <= colors <= len(COMPACT_ALPHABET):
        raise ValueError("compact words require 2 <= colors <= 36")
    word: list[str] = []
    for color in coloring:
        if type(color) is not int or not 0 <= color < colors:
            raise ValueError(f"compact-word color must lie in 0..{colors - 1}")
        word.append(COMPACT_ALPHABET[color])
    return "".join(word)


def word_to_coloring(word: str, colors: int) -> tuple[int, ...]:
    if type(colors) is not int or not 2 <= colors <= len(COMPACT_ALPHABET):
        raise ValueError("compact words require 2 <= colors <= 36")
    if not isinstance(word, str):
        raise TypeError("compact word must be a string")
    lookup = {symbol: value for value, symbol in enumerate(COMPACT_ALPHABET)}
    result: list[int] = []
    for symbol in word.strip().upper():
        if symbol not in lookup or lookup[symbol] >= colors:
            raise ValueError(f"invalid compact-word symbol {symbol!r}")
        result.append(lookup[symbol])
    return tuple(result)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WitnessFormatError(f"{name} must be an object")
    return value


def _plain_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise WitnessFormatError(f"{name} must be an ordinary integer")
    return value


def validate_witness_data(data: object) -> Witness:
    """Validate the complete schema and independently check the coloring."""

    root = _mapping(data, "witness")
    if _plain_int(root.get("format_version"), "format_version") != FORMAT_VERSION:
        raise WitnessFormatError(f"only format_version {FORMAT_VERSION} is supported")
    instance_data = _mapping(root.get("instance"), "instance")
    coefficients_data = instance_data.get("coefficients")
    if not isinstance(coefficients_data, list):
        raise WitnessFormatError("instance.coefficients must be an array")
    coefficients = tuple(
        _plain_int(value, f"instance.coefficients[{index}]")
        for index, value in enumerate(coefficients_data)
    )
    try:
        instance = InstanceSpec(
            PolynomialSpec(coefficients),
            _plain_int(instance_data.get("colors"), "instance.colors"),
            _plain_int(instance_data.get("n"), "instance.n"),
            instance_data.get("input_domain", "all_nonzero"),
        )
    except (TypeError, ValueError) as error:
        raise WitnessFormatError(str(error)) from error
    scope_data = _mapping(root.get("scope"), "scope")
    try:
        scope = ModelScope(scope_data.get("mode"))
    except (TypeError, ValueError) as error:
        raise WitnessFormatError("scope.mode is invalid") from error
    coloring_data = root.get("coloring")
    if not isinstance(coloring_data, list):
        raise WitnessFormatError("coloring must be an array")
    coloring = tuple(
        _plain_int(value, f"coloring[{index}]")
        for index, value in enumerate(coloring_data)
    )
    search = _mapping(root.get("search"), "search")
    backend = search.get("backend")
    if not isinstance(backend, str) or not backend:
        raise WitnessFormatError("search.backend must be a nonempty string")
    seed = _plain_int(search.get("seed"), "search.seed")
    elapsed = search.get("elapsed_seconds")
    if type(elapsed) not in (int, float) or elapsed < 0:
        raise WitnessFormatError("search.elapsed_seconds must be nonnegative")
    verification = _mapping(root.get("verification"), "verification")
    if verification.get("valid") is not True:
        raise WitnessFormatError("verification.valid must be true")
    checked = verify_coloring(instance, coloring)
    if not checked.valid:
        detail = checked.error or repr(checked.violation)
        raise WitnessFormatError(f"witness coloring fails independent verification: {detail}")
    scope_metadata = {
        key: value for key, value in scope_data.items() if key != "mode"
    }
    return Witness(
        instance,
        scope,
        coloring,
        backend,
        seed,
        float(elapsed),
        scope_metadata,
    )


def create_witness(
    instance: InstanceSpec,
    coloring: tuple[int, ...] | list[int],
    *,
    backend: str,
    seed: int = 0,
    elapsed_seconds: float = 0.0,
    scope: ModelScope = ModelScope.FULL,
    scope_metadata: Mapping[str, object] | None = None,
) -> Witness:
    """Construct a witness only after independent verification succeeds."""

    candidate = Witness(
        instance=instance,
        scope=scope,
        coloring=tuple(coloring),
        backend=backend,
        seed=seed,
        elapsed_seconds=float(elapsed_seconds),
        scope_metadata=dict(scope_metadata or {}),
    )
    return validate_witness_data(candidate.to_dict())


def write_witness(path: str | os.PathLike[str], witness: Witness) -> None:
    """Write a witness atomically after revalidating its coloring and schema."""

    checked = validate_witness_data(witness.to_dict())
    output = Path(path)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(checked.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def read_witness(path: str | os.PathLike[str]) -> Witness:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as error:
        raise WitnessFormatError(f"invalid witness JSON: {error}") from error
    return validate_witness_data(data)
