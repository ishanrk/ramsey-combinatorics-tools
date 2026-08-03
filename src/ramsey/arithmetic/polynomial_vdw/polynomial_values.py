from __future__ import annotations

from ramsey.arithmetic.polynomial_vdw.polynomial import Polynomial


def _smallest_growth_bound(
    leading_magnitude: int,
    degree: int,
    twice_max_difference: int,
) -> int:
    high = 1
    while leading_magnitude * high**degree <= twice_max_difference:
        high *= 2
    low = 1
    while low < high:
        middle = (low + high) // 2
        if leading_magnitude * middle**degree > twice_max_difference:
            high = middle
        else:
            low = middle + 1
    return low


def certified_input_bound(polynomial: Polynomial, max_difference: int) -> int:
    """find a proven finite radius containing every relevant input.

    write the polynomial as a_k x^k plus its lower terms and let s be the
    sum of the lower coefficient magnitudes. once abs(a_k) times abs(d) is
    at least 2s, the leading term leaves at least half its magnitude after
    cancellation. an exact integer growth bound then makes that half bigger
    than max_difference. taking the larger bound proves no skipped input can
    create another graph edge.
    """
    if not isinstance(polynomial, Polynomial):
        raise TypeError("polynomial must be a Polynomial")
    if type(max_difference) is not int:
        raise TypeError("max_difference must be an ordinary integer")
    if max_difference < 0:
        raise ValueError("max_difference must be nonnegative")

    leading_magnitude = abs(polynomial.leading_coefficient)
    lower_sum = sum(abs(value) for value in polynomial.coefficients[:-1])
    dominance_bound = max(
        1,
        (2 * lower_sum + leading_magnitude - 1) // leading_magnitude,
    )
    growth_bound = _smallest_growth_bound(
        leading_magnitude,
        polynomial.degree,
        2 * max_difference,
    )
    return max(dominance_bound, growth_bound)


__all__ = ["certified_input_bound"]
