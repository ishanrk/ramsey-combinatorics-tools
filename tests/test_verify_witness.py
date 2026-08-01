from __future__ import annotations

import json

import pytest

from pvdw.model import InstanceSpec, PolynomialSpec
from pvdw.verify import conflict_count, verify_coloring
from pvdw.witness import (
    WitnessFormatError,
    coloring_to_word,
    create_witness,
    read_witness,
    word_to_coloring,
    write_witness,
)


def complete_three_vertex_instance() -> InstanceSpec:
    return InstanceSpec(PolynomialSpec((0, 1)), 3, 3)


def test_verifier_accepts_valid_and_identifies_first_violation() -> None:
    instance = complete_three_vertex_instance()
    assert verify_coloring(instance, (0, 1, 2)).valid
    result = verify_coloring(instance, (0, 0, 2))
    assert not result.valid
    assert result.violation is not None
    assert (result.violation.u, result.violation.v) == (0, 1)
    assert result.violation.difference == 1
    assert result.violation.polynomial_inputs == (-1, 1)
    assert conflict_count(instance, (0, 0, 2)) == 1


@pytest.mark.parametrize("coloring", [(0, 1), (0, 1, 2, 0), (0, 1, 3), (0, 1, -1)])
def test_verifier_rejects_malformed_colorings(coloring: tuple[int, ...]) -> None:
    result = verify_coloring(complete_three_vertex_instance(), coloring)
    assert not result.valid
    assert result.error is not None


def test_witness_round_trip_and_compact_word(tmp_path) -> None:
    instance = complete_three_vertex_instance()
    witness = create_witness(instance, (0, 1, 2), backend="bruteforce")
    path = tmp_path / "witness.json"
    write_witness(path, witness)
    assert read_witness(path) == witness
    assert coloring_to_word([0, 1, 2, 10, 35], 36) == "012AZ"
    assert word_to_coloring("012az", 36) == (0, 1, 2, 10, 35)


def test_unverified_witness_is_never_written(tmp_path) -> None:
    with pytest.raises(WitnessFormatError):
        create_witness(complete_three_vertex_instance(), (0, 0, 2), backend="bad")
    path = tmp_path / "bad.json"
    valid = create_witness(complete_three_vertex_instance(), (0, 1, 2), backend="ok")
    data = valid.to_dict()
    data["coloring"] = [0, 0, 2]
    path.write_text(json.dumps(data))
    with pytest.raises(WitnessFormatError):
        read_witness(path)


@pytest.mark.parametrize("word,colors", [("G", 16), ("-", 3), ("0", 37)])
def test_compact_word_rejections(word: str, colors: int) -> None:
    with pytest.raises(ValueError):
        word_to_coloring(word, colors)
