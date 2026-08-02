"""Exact packed-integer arithmetic for polynomials over GF(2)."""

from __future__ import annotations

from collections.abc import Iterable


_REVERSED_BYTES = bytes(
    int(f"{value:08b}"[::-1], 2) for value in range(256)
)


def _check_polynomial(polynomial: int) -> int:
    if type(polynomial) is not int or polynomial < 0:
        raise TypeError("a GF(2) polynomial must be a nonnegative ordinary integer")
    return polynomial


def gf2_degree(polynomial: int) -> int:
    """Return -1 for zero and otherwise the highest nonzero exponent."""

    return _check_polynomial(polynomial).bit_length() - 1


def gf2_add(left: int, right: int) -> int:
    return _check_polynomial(left) ^ _check_polynomial(right)


def gf2_monomial(exponent: int) -> int:
    if type(exponent) is not int or exponent < 0:
        raise ValueError("monomial exponent must be a nonnegative ordinary integer")
    return 1 << exponent


def gf2_from_exponents(exponents: Iterable[int]) -> int:
    polynomial = 0
    for exponent in exponents:
        polynomial ^= gf2_monomial(exponent)
    return polynomial


def gf2_to_exponents(polynomial: int) -> tuple[int, ...]:
    remaining = _check_polynomial(polynomial)
    exponents: list[int] = []
    while remaining:
        lowest = remaining & -remaining
        exponents.append(lowest.bit_length() - 1)
        remaining ^= lowest
    return tuple(exponents)


def gf2_mul(left: int, right: int) -> int:
    """Multiply by shifting once per set bit of the sparser operand."""

    left = _check_polynomial(left)
    right = _check_polynomial(right)
    if left.bit_count() > right.bit_count():
        left, right = right, left
    product = 0
    while left:
        lowest = left & -left
        product ^= right << (lowest.bit_length() - 1)
        left ^= lowest
    return product


def _gf2_mul_fast(left: int, right: int) -> int:
    """Karatsuba helper for dense internal extended-Euclid coefficients."""

    if not left or not right:
        return 0
    width = max(left.bit_length(), right.bit_length())
    if width <= 512 or min(left.bit_count(), right.bit_count()) <= 64:
        return gf2_mul(left, right)
    split = width // 2
    mask = (1 << split) - 1
    left_low, left_high = left & mask, left >> split
    right_low, right_high = right & mask, right >> split
    low = _gf2_mul_fast(left_low, right_low)
    high = _gf2_mul_fast(left_high, right_high)
    middle = (
        _gf2_mul_fast(left_low ^ left_high, right_low ^ right_high)
        ^ low
        ^ high
    )
    return low ^ (middle << split) ^ (high << (2 * split))


def _reverse_bits(value: int, width: int) -> int:
    if width == 0:
        return 0
    byte_count = (width + 7) // 8
    reversed_value = int.from_bytes(
        value.to_bytes(byte_count, "little").translate(_REVERSED_BYTES)[::-1],
        "little",
    )
    return reversed_value >> (8 * byte_count - width)


def _inverse_series(polynomial: int, precision: int) -> int:
    """Invert a constant-one polynomial modulo ``X**precision``."""

    if not polynomial & 1:
        raise ValueError("power-series inverse requires constant coefficient one")
    inverse = 1
    known = 1
    while known < precision:
        extended = min(2 * known, precision)
        mask = (1 << extended) - 1
        product = _gf2_mul_fast(polynomial & mask, inverse)
        error = (product >> known) & ((1 << (extended - known)) - 1)
        inverse ^= (_gf2_mul_fast(inverse, error) << known) & mask
        known = extended
    return inverse & ((1 << precision) - 1)


def _gf2_divmod_fast(dividend: int, divisor: int) -> tuple[int, int]:
    dividend_degree = gf2_degree(dividend)
    divisor_degree = gf2_degree(divisor)
    if dividend_degree < divisor_degree:
        return 0, dividend
    quotient_width = dividend_degree - divisor_degree + 1
    reversed_divisor = _reverse_bits(divisor, divisor_degree + 1)
    reversed_dividend = _reverse_bits(dividend, dividend_degree + 1)
    low_dividend = reversed_dividend & ((1 << quotient_width) - 1)
    inverse = _inverse_series(reversed_divisor, quotient_width)
    reversed_quotient = (
        _gf2_mul_fast(low_dividend, inverse) & ((1 << quotient_width) - 1)
    )
    quotient = _reverse_bits(reversed_quotient, quotient_width)
    remainder = dividend ^ _gf2_mul_fast(quotient, divisor)
    if remainder and gf2_degree(remainder) >= divisor_degree:
        raise RuntimeError("fast GF(2) division produced an oversized remainder")
    return quotient, remainder


def gf2_divmod(dividend: int, divisor: int) -> tuple[int, int]:
    dividend = _check_polynomial(dividend)
    divisor = _check_polynomial(divisor)
    if divisor == 0:
        raise ZeroDivisionError("GF(2) polynomial division by zero")
    quotient = 0
    divisor_degree = gf2_degree(divisor)
    remainder = dividend
    while remainder and gf2_degree(remainder) >= divisor_degree:
        shift = gf2_degree(remainder) - divisor_degree
        quotient ^= 1 << shift
        remainder ^= divisor << shift
    return quotient, remainder


def gf2_div_exact(dividend: int, divisor: int) -> int:
    quotient, remainder = gf2_divmod(dividend, divisor)
    if remainder:
        raise ValueError(
            f"GF(2) division is not exact; remainder exponents={gf2_to_exponents(remainder)}"
        )
    return quotient


def gf2_div_exact_fast(dividend: int, divisor: int) -> int:
    """Exact reciprocal-series division for very large internal relations."""

    dividend = _check_polynomial(dividend)
    divisor = _check_polynomial(divisor)
    if divisor == 0:
        raise ZeroDivisionError("GF(2) polynomial division by zero")
    quotient, remainder = _gf2_divmod_fast(dividend, divisor)
    if remainder:
        raise ValueError(
            f"GF(2) division is not exact; remainder degree={gf2_degree(remainder)}"
        )
    return quotient


def gf2_gcd(left: int, right: int) -> int:
    left = _check_polynomial(left)
    right = _check_polynomial(right)
    while right:
        _, remainder = gf2_divmod(left, right)
        left, right = right, remainder
    return left


def gf2_xgcd(left: int, right: int) -> tuple[int, int, int]:
    """Return ``(g, s, t)`` with ``s*left + t*right == g``."""

    old_r, r = _check_polynomial(left), _check_polynomial(right)
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        quotient, remainder = gf2_divmod(old_r, r)
        old_r, r = r, remainder
        old_s, s = s, old_s ^ _gf2_mul_fast(quotient, s)
        old_t, t = t, old_t ^ _gf2_mul_fast(quotient, t)
    return old_r, old_s, old_t


def gf2_mod_inverse(polynomial: int, modulus: int) -> int:
    modulus = _check_polynomial(modulus)
    if gf2_degree(modulus) < 1:
        raise ValueError("modulus must have positive degree")
    old_remainder, remainder = _check_polynomial(polynomial), modulus
    old_coefficient, coefficient = 1, 0
    while remainder:
        quotient, new_remainder = gf2_divmod(old_remainder, remainder)
        old_remainder, remainder = remainder, new_remainder
        old_coefficient, coefficient = (
            coefficient,
            old_coefficient ^ _gf2_mul_fast(quotient, coefficient),
        )
    if old_remainder != 1:
        raise ValueError("GF(2) polynomial is not invertible modulo the modulus")
    return gf2_divmod(old_coefficient, modulus)[1]


def gf2_mul_mod(left: int, right: int, modulus: int) -> int:
    """Multiply dense operands efficiently and reduce modulo ``modulus``."""

    modulus = _check_polynomial(modulus)
    if modulus == 0:
        raise ZeroDivisionError("GF(2) polynomial reduction by zero")
    product = _gf2_mul_fast(_check_polynomial(left), _check_polynomial(right))
    return _gf2_divmod_fast(product, modulus)[1]


def gf2_eval_at_one(polynomial: int) -> int:
    return _check_polynomial(polynomial).bit_count() & 1


def gf2_derivative(polynomial: int) -> int:
    result = 0
    for exponent in gf2_to_exponents(polynomial):
        if exponent & 1:
            result ^= 1 << (exponent - 1)
    return result


def gf2_reciprocal(polynomial: int) -> int:
    polynomial = _check_polynomial(polynomial)
    if polynomial == 0:
        return 0
    degree = gf2_degree(polynomial)
    return gf2_from_exponents(degree - exponent for exponent in gf2_to_exponents(polynomial))
