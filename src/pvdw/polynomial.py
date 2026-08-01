"""Safe parsing and exact arithmetic for integer polynomials."""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence

from pvdw.model import PolynomialSpec


class PolynomialParseError(ValueError):
    """Raised when input is outside the supported polynomial grammar."""


Coefficients = tuple[int, ...]


def _trim(coefficients: Sequence[int]) -> Coefficients:
    result = tuple(coefficients)
    while len(result) > 1 and result[-1] == 0:
        result = result[:-1]
    return result or (0,)


def _add(left: Coefficients, right: Coefficients) -> Coefficients:
    size = max(len(left), len(right))
    return _trim(
        tuple(
            (left[i] if i < len(left) else 0)
            + (right[i] if i < len(right) else 0)
            for i in range(size)
        )
    )


def _negate(poly: Coefficients) -> Coefficients:
    return tuple(-coefficient for coefficient in poly)


def _multiply(left: Coefficients, right: Coefficients) -> Coefficients:
    result = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] += a * b
    return _trim(result)


def _power(poly: Coefficients, exponent: int) -> Coefficients:
    result: Coefficients = (1,)
    base = poly
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = _multiply(result, base)
        remaining >>= 1
        if remaining:
            base = _multiply(base, base)
    return result


def _nonnegative_exponent(node: ast.AST) -> int:
    if isinstance(node, ast.Constant) and type(node.value) is int and node.value >= 0:
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.UAdd)
        and isinstance(node.operand, ast.Constant)
        and type(node.operand.value) is int
        and node.operand.value >= 0
    ):
        return node.operand.value
    raise PolynomialParseError("powers require a nonnegative integer exponent")


def _from_ast(node: ast.AST) -> Coefficients:
    if isinstance(node, ast.Constant):
        if type(node.value) is not int:
            raise PolynomialParseError("only integer constants are allowed")
        return (node.value,)
    if isinstance(node, ast.Name):
        if node.id != "x":
            raise PolynomialParseError("the only permitted symbol is x")
        return (0, 1)
    if isinstance(node, ast.UnaryOp):
        operand = _from_ast(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return _negate(operand)
        raise PolynomialParseError("unsupported unary operator")
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Add):
            return _add(_from_ast(node.left), _from_ast(node.right))
        if isinstance(node.op, ast.Sub):
            return _add(_from_ast(node.left), _negate(_from_ast(node.right)))
        if isinstance(node.op, ast.Mult):
            return _multiply(_from_ast(node.left), _from_ast(node.right))
        if isinstance(node.op, ast.Pow):
            return _power(_from_ast(node.left), _nonnegative_exponent(node.right))
        raise PolynomialParseError("only +, -, *, and nonnegative powers are allowed")
    raise PolynomialParseError(f"unsupported syntax: {type(node).__name__}")


def parse_polynomial(text: str) -> PolynomialSpec:
    """Parse the restricted expression grammar without evaluating Python code."""

    if not isinstance(text, str) or not text.strip():
        raise PolynomialParseError("polynomial expression must be nonempty")
    try:
        tree = ast.parse(text.replace("^", "**"), mode="eval")
    except (SyntaxError, ValueError) as error:
        raise PolynomialParseError("invalid polynomial syntax") from error
    try:
        return PolynomialSpec(_from_ast(tree.body))
    except (TypeError, ValueError) as error:
        if isinstance(error, PolynomialParseError):
            raise
        raise PolynomialParseError(str(error)) from error


def parse_coefficients(text: str | Sequence[int]) -> PolynomialSpec:
    """Parse comma-separated coefficients in low-degree-first order."""

    if isinstance(text, str):
        parts = [part.strip() for part in text.split(",")]
        if not parts or any(
            re.fullmatch(r"[+-]?[0-9]+", part) is None for part in parts
        ):
            raise PolynomialParseError("coefficients must be comma-separated integers")
        coefficients = tuple(int(part, 10) for part in parts)
    else:
        coefficients = tuple(text)
    try:
        return PolynomialSpec(coefficients)
    except (TypeError, ValueError) as error:
        raise PolynomialParseError(str(error)) from error
