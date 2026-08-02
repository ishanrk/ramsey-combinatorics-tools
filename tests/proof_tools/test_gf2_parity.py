from __future__ import annotations

import random

import pytest
from hypothesis import given, settings, strategies as st

from pvdw.proof_tools.gf2poly import (
    gf2_add,
    gf2_degree,
    gf2_derivative,
    gf2_div_exact,
    gf2_divmod,
    gf2_eval_at_one,
    gf2_from_exponents,
    gf2_gcd,
    gf2_mod_inverse,
    gf2_monomial,
    gf2_mul,
    gf2_reciprocal,
    gf2_to_exponents,
    gf2_xgcd,
)
from pvdw.proof_tools.parity_cover import (
    A_POINTS,
    B_POINTS,
    C_POINTS,
    find_parity_cover_bounded_span,
    find_multi_shape_bezout_relation,
    find_two_shape_bezout_relation,
    enumerate_square_k4_shapes,
    is_perfect_square,
    is_square_distance_clique,
    square_submillion_parity_witness,
    verify_bezout_relation,
    verify_multi_shape_bezout_relation,
    verify_parity_cover_witness,
    verify_square_parity_witness,
)


def _slow_mul(left: int, right: int) -> int:
    left_bits = [(left >> index) & 1 for index in range(left.bit_length())]
    right_bits = [(right >> index) & 1 for index in range(right.bit_length())]
    product = [0] * max(1, len(left_bits) + len(right_bits))
    for left_index, left_bit in enumerate(left_bits):
        for right_index, right_bit in enumerate(right_bits):
            product[left_index + right_index] ^= left_bit & right_bit
    return sum(bit << index for index, bit in enumerate(product))


@given(st.integers(min_value=0, max_value=2**32 - 1), st.integers(min_value=0, max_value=2**32 - 1))
@settings(max_examples=100, deadline=None)
def test_gf2_multiplication_matches_list_reference(left: int, right: int) -> None:
    assert gf2_mul(left, right) == _slow_mul(left, right)


@given(
    st.integers(min_value=0, max_value=2**80 - 1),
    st.integers(min_value=1, max_value=2**40 - 1),
)
@settings(max_examples=100, deadline=None)
def test_gf2_division_identity(dividend: int, divisor: int) -> None:
    quotient, remainder = gf2_divmod(dividend, divisor)
    assert gf2_add(gf2_mul(quotient, divisor), remainder) == dividend
    assert remainder == 0 or gf2_degree(remainder) < gf2_degree(divisor)


def test_gf2_operations_and_algebraic_properties() -> None:
    polynomial = gf2_from_exponents((0, 1, 4, 9))
    assert gf2_to_exponents(polynomial) == (0, 1, 4, 9)
    assert gf2_monomial(9) == 1 << 9
    assert gf2_degree(0) == -1
    assert gf2_eval_at_one(polynomial) == 0
    assert gf2_reciprocal(gf2_reciprocal(polynomial)) == polynomial
    left = gf2_from_exponents((0, 2, 5))
    right = gf2_from_exponents((0, 1, 3))
    assert gf2_derivative(gf2_mul(left, right)) == (
        gf2_mul(gf2_derivative(left), right)
        ^ gf2_mul(left, gf2_derivative(right))
    )
    gcd, coefficient_left, coefficient_right = gf2_xgcd(left, right)
    assert gf2_mul(coefficient_left, left) ^ gf2_mul(coefficient_right, right) == gcd
    if gcd == 1:
        inverse = gf2_mod_inverse(left, right)
        assert gf2_divmod(gf2_mul(left, inverse), right)[1] == 1
    with pytest.raises(ValueError):
        gf2_div_exact(0b100, 0b11)


def test_bounded_parity_cover_and_bezout_verification() -> None:
    witness = find_parity_cover_bounded_span({"a": (0, 1), "b": (0, 2)}, 4)
    assert witness.verified and witness.odd_block_count
    assert not any(witness.point_parities)
    assert verify_parity_cover_witness(witness)
    bezout = find_two_shape_bezout_relation((0, 1), (0, 2))
    assert verify_bezout_relation(bezout)
    multi = find_multi_shape_bezout_relation(((0, 1), (0, 2), (0, 3)))
    assert verify_multi_shape_bezout_relation(multi)


def test_square_shapes_and_large_exact_relation() -> None:
    assert is_perfect_square(0)
    assert is_perfect_square(485809)
    assert not is_perfect_square(2)
    assert all(is_square_distance_clique(shape) for shape in (A_POINTS, B_POINTS, C_POINTS))
    assert list(
        enumerate_square_k4_shapes(
            max(A_POINTS), candidate_points=A_POINTS
        )
    ) == [A_POINTS]
    witness = square_submillion_parity_witness()
    assert gf2_degree(witness.q) == 32
    assert gf2_degree(witness.gcd_ab) == 33
    assert witness.relation_zero
    assert witness.odd_block_count
    assert witness.expanded_even_cover
    assert witness.maximum_point == 971584
    assert witness.interval_size == 971585
    assert witness.peak_memory_bytes > 0
    assert witness.packed_polynomial_bytes > 0
    assert verify_square_parity_witness(witness)
