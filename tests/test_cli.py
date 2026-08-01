from __future__ import annotations

import json

from pvdw.cli import main
from pvdw.encoding.dimacs import parse_dimacs


def test_inspect_json(capsys) -> None:
    assert main(["inspect", "--poly", "x^2", "--colors", "4", "--N", "58", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["coefficients"] == [0, 0, 1]
    assert payload["distances"] == [1, 4, 9, 16, 25, 36, 49]
    assert payload["edge_count"] == sum(58 - distance for distance in payload["distances"])


def test_encode_and_verify_workflow(tmp_path, capsys) -> None:
    cnf = tmp_path / "instance.cnf"
    assert main(
        [
            "encode",
            "--poly",
            "x^2+x",
            "--colors",
            "4",
            "--N",
            "97",
            "--encoding",
            "onehot",
            "--amo",
            "pairwise",
            "--output",
            str(cnf),
            "--json",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    formula = parse_dimacs(cnf)
    assert formula.variable_count == payload["variables"]
    assert formula.clause_count == payload["clauses"]

    witness = tmp_path / "witness.json"
    assert main(
        [
            "solve",
            "--poly",
            "x^2",
            "--colors",
            "3",
            "--N",
            "10",
            "--backend",
            "bruteforce",
            "--output",
            str(witness),
            "--json",
        ]
    ) == 0
    capsys.readouterr()
    assert main(["verify", str(witness), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True


def test_cli_invalid_input_exit_code(capsys) -> None:
    assert main(
        ["inspect", "--poly", '__import__("os")', "--colors", "3", "--N", "10"]
    ) == 2
    assert "invalid input" in capsys.readouterr().err
