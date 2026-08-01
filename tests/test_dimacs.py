from __future__ import annotations

import pytest

from pvdw.encoding.dimacs import parse_dimacs, parse_sat_model, write_dimacs


def test_streaming_dimacs_round_trip(tmp_path) -> None:
    path = tmp_path / "formula.cnf"
    clauses = ([1, -2], [], [3])
    write_dimacs(path, 3, (clause for clause in clauses), clause_count=3)
    formula = parse_dimacs(path)
    assert formula.variable_count == 3
    assert formula.clauses == ((1, -2), (), (3,))
    assert path.read_text().startswith("p cnf 3 3\n")


def test_unknown_count_uses_a_counted_body(tmp_path) -> None:
    path = tmp_path / "formula.cnf"
    write_dimacs(path, 2, (clause for clause in ([1], [-1, 2])))
    assert parse_dimacs(path).clause_count == 2


def test_dimacs_parser_supports_split_clauses_and_comments(tmp_path) -> None:
    path = tmp_path / "split.cnf"
    path.write_text("c example\np cnf 3 2\n1 -2\n3 0\n0\n")
    assert parse_dimacs(path).clauses == ((1, -2, 3), ())


@pytest.mark.parametrize(
    "text",
    [
        "p cnf 2 1\n3 0\n",
        "p cnf 2 2\n1 0\n",
        "p cnf 2 1\n1\n",
        "1 0\n",
    ],
)
def test_dimacs_parser_rejects_malformed_files(tmp_path, text: str) -> None:
    path = tmp_path / "bad.cnf"
    path.write_text(text)
    with pytest.raises(ValueError):
        parse_dimacs(path)


def test_sat_model_parser() -> None:
    text = "c result\ns SATISFIABLE\nv 1 -2 3\nv -4 0\n"
    assert parse_sat_model(text) == (1, -2, 3, -4)
    assert parse_sat_model("s UNSATISFIABLE\n") is None
    assert parse_sat_model("v 1 -2\n") == (1, -2)


@pytest.mark.parametrize(
    "text",
    [
        "c only\n",
        "s UNKNOWN\n",
        "s SATISFIABLE\ns UNSATISFIABLE\n",
        "s UNSATISFIABLE\nv 1 0\n",
        "s SATISFIABLE\nv nope 0\n",
    ],
)
def test_sat_model_parser_rejects_malformed_output(text: str) -> None:
    with pytest.raises(ValueError):
        parse_sat_model(text)
