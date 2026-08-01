from __future__ import annotations

import pytest

from pvdw.model import InstanceSpec, PolynomialSpec
from pvdw.polynomial import PolynomialParseError, parse_coefficients, parse_polynomial


def test_polynomial_normalization_and_horner_evaluation() -> None:
    polynomial = PolynomialSpec((0, -11, 3, 0, 0))
    assert polynomial.coefficients == (0, -11, 3)
    assert polynomial.degree == 2
    assert polynomial.evaluate(-4) == 92
    assert polynomial.evaluate(5) == 20


@pytest.mark.parametrize(
    "coefficients",
    [(), (0,), (0, 0, 0), (1, 2), (0, 1.0), (0, True)],
)
def test_polynomial_rejects_invalid_coefficients(coefficients: tuple[object, ...]) -> None:
    with pytest.raises((TypeError, ValueError)):
        PolynomialSpec(coefficients)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("x^2", (0, 0, 1)),
        ("3*x^2-11*x", (0, -11, 3)),
        ("-(x*x) + 4*x", (0, 4, -1)),
        ("x**0*x", (0, 1)),
        ("(x+x)*(x-2)", (0, -4, 2)),
    ],
)
def test_safe_parser_accepts_restricted_polynomials(
    text: str, expected: tuple[int, ...]
) -> None:
    assert parse_polynomial(text).coefficients == expected


@pytest.mark.parametrize(
    "text",
    [
        '__import__("os")',
        "x / 2",
        "x ** -1",
        "sin(x)",
        "y + x",
        "1.5*x",
        "x.__class__",
        "[x][0]",
        "x if 1 else 0",
        "x ** (1 + 1)",
        "",
    ],
)
def test_safe_parser_rejects_invalid_or_malicious_input(text: str) -> None:
    with pytest.raises(PolynomialParseError):
        parse_polynomial(text)


def test_coefficient_parser_is_primary_machine_interface() -> None:
    assert parse_coefficients("0, 0, 1, 0").coefficients == (0, 0, 1)
    with pytest.raises(PolynomialParseError):
        parse_coefficients("0,1.5")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"colors": 1, "n": 2},
        {"colors": 2, "n": 0},
        {"colors": 2.0, "n": 2},
        {"colors": 2, "n": True},
        {"colors": 2, "n": 2, "input_domain": "zero_too"},
    ],
)
def test_instance_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        InstanceSpec(PolynomialSpec((0, 1)), **kwargs)  # type: ignore[arg-type]
