"""Affine/common-scale transfers into polynomial-distance graphs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from pvdw.model import PolynomialSpec


@dataclass(frozen=True)
class TransferWitness:
    polynomial: PolynomialSpec
    source_differences: tuple[int, ...]
    scale: int
    inputs: Mapping[int, int]
    realized_values: Mapping[int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))
        object.__setattr__(
            self, "realized_values", MappingProxyType(dict(self.realized_values))
        )


def verify_transfer_witness(witness: TransferWitness) -> bool:
    if witness.scale <= 0 or set(witness.inputs) != set(witness.source_differences):
        return False
    if set(witness.realized_values) != set(witness.source_differences):
        return False
    for difference in witness.source_differences:
        polynomial_input = witness.inputs[difference]
        if polynomial_input == 0:
            return False
        value = abs(witness.polynomial.evaluate(polynomial_input))
        if value != witness.realized_values[difference]:
            return False
        if value != witness.scale * difference:
            return False
    return True


def search_common_scale(
    polynomial: PolynomialSpec,
    source_differences: Iterable[int],
    max_scale: int,
    input_bound: int,
) -> TransferWitness | None:
    """Intersect exact candidate-scale sets after one polynomial evaluation pass."""

    differences = tuple(sorted(set(source_differences)))
    if not differences or any(type(value) is not int or value <= 0 for value in differences):
        raise ValueError("source differences must be positive ordinary integers")
    if type(max_scale) is not int or max_scale < 1:
        raise ValueError("max_scale must be a positive ordinary integer")
    if type(input_bound) is not int or input_bound < 1:
        raise ValueError("input_bound must be a positive ordinary integer")
    preimages: defaultdict[int, list[int]] = defaultdict(list)
    for polynomial_input in range(-input_bound, input_bound + 1):
        if polynomial_input == 0:
            continue
        value = abs(polynomial.evaluate(polynomial_input))
        if value:
            preimages[value].append(polynomial_input)
    scale_sets = []
    for difference in differences:
        scale_sets.append(
            {
                value // difference
                for value in preimages
                if value % difference == 0 and 1 <= value // difference <= max_scale
            }
        )
    common_scales = set.intersection(*scale_sets)
    if not common_scales:
        return None
    scale = min(common_scales)
    inputs: dict[int, int] = {}
    realized: dict[int, int] = {}
    for difference in differences:
        value = scale * difference
        candidates = preimages[value]
        if not candidates:
            raise RuntimeError("scale intersection retained a missing polynomial value")
        polynomial_input = min(
            candidates,
            key=lambda candidate: (abs(candidate), candidate < 0, candidate),
        )
        inputs[difference] = polynomial_input
        realized[difference] = value
    witness = TransferWitness(polynomial, differences, scale, inputs, realized)
    if not verify_transfer_witness(witness):
        raise RuntimeError("common-scale search produced an invalid transfer")
    return witness
