"""Even-cover searches and the exact submillion square-parity relation."""

from __future__ import annotations

import hashlib
import resource
import sys
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache, reduce
from math import isqrt

from ortools.sat.python import cp_model

from pvdw.proof_tools.block_cover import Block, enumerate_translated_blocks
from pvdw.proof_tools.gadgets import find_distance_cliques, interval_distance_graph
from pvdw.proof_tools.gf2poly import (
    gf2_degree,
    gf2_div_exact_fast,
    gf2_eval_at_one,
    gf2_from_exponents,
    gf2_gcd,
    gf2_mod_inverse,
    gf2_mul,
    gf2_mul_mod,
    gf2_to_exponents,
    gf2_xgcd,
)


@dataclass(frozen=True)
class ParityCoverWitness:
    selected_blocks: tuple[Block, ...]
    max_point: int
    point_parities: tuple[int, ...]
    odd_block_count: bool
    verified: bool


@dataclass(frozen=True)
class BezoutRelation:
    left: int
    right: int
    gcd: int
    left_coefficient: int
    right_coefficient: int
    verified: bool


@dataclass(frozen=True)
class MultiShapeBezoutRelation:
    polynomials: tuple[int, ...]
    gcd: int
    coefficients: tuple[int, ...]
    verified: bool


@dataclass(frozen=True)
class SquareParityWitness:
    shapes: tuple[tuple[int, ...], ...]
    shape_polynomials: tuple[int, ...]
    q: int
    gcd_ab: int
    u: int
    v: int
    relation_zero: bool
    odd_block_count: bool
    expanded_even_cover: bool
    maximum_point: int
    interval_size: int
    normalized_degrees: tuple[int, int, int]
    coefficient_degrees: tuple[int, int, int]
    product_degrees: tuple[int, int, int]
    elapsed_seconds: float
    peak_memory_bytes: int
    packed_polynomial_bytes: int


A_POINTS = (0, 23409, 34225, 485809)
B_POINTS = (0, 451584, 462400, 485809)
C_POINTS = (0, 270400, 284089, 855625)


def is_perfect_square(n: int) -> bool:
    if type(n) is not int or n < 0:
        return False
    root = isqrt(n)
    return root * root == n


def is_square_distance_clique(points: Sequence[int]) -> bool:
    normalized = tuple(points)
    return len(set(normalized)) == len(normalized) and all(
        is_perfect_square(abs(right - left))
        for index, left in enumerate(normalized)
        for right in normalized[index + 1 :]
    )


def enumerate_square_k4_shapes(
    max_span: int,
    *,
    normalize_translation: bool = True,
    identify_reflections: bool = True,
    candidate_points: Sequence[int] | None = None,
) -> Iterator[tuple[int, int, int, int]]:
    """Enumerate square-distance four-cliques without scanning four-subsets."""

    if type(max_span) is not int or max_span < 1:
        raise ValueError("max_span must be a positive ordinary integer")
    squares = tuple(square for root in range(1, isqrt(max_span) + 1) if (square := root * root) <= max_span)
    if candidate_points is None:
        graph = interval_distance_graph(max_span, squares)
    else:
        from pvdw.proof_tools.gadgets import finite_distance_graph

        supplied = tuple(sorted(set(candidate_points)))
        if not supplied:
            return iter(())
        graph = finite_distance_graph(supplied, squares)
    shapes: set[tuple[int, int, int, int]] = set()
    for clique in find_distance_cliques(graph, 4):
        shape = tuple(sorted(clique))
        if max(shape) - min(shape) > max_span:
            continue
        if normalize_translation:
            minimum = min(shape)
            shape = tuple(point - minimum for point in shape)
        if identify_reflections:
            maximum = max(shape)
            reflected = tuple(sorted(maximum - point for point in shape))
            shape = min(shape, reflected)
        assert len(shape) == 4
        shapes.add(shape)
    return iter(sorted(shapes))


def _solve_gf2(equations: Iterable[tuple[int, int]], variable_count: int) -> int | None:
    """Solve packed Boolean linear equations and return one assignment mask."""

    pivots: dict[int, tuple[int, int]] = {}
    for supplied_mask, supplied_rhs in equations:
        mask, rhs = supplied_mask, supplied_rhs & 1
        while mask:
            pivot = (mask & -mask).bit_length() - 1
            previous = pivots.get(pivot)
            if previous is None:
                pivots[pivot] = (mask, rhs)
                break
            mask ^= previous[0]
            rhs ^= previous[1]
        if mask == 0 and rhs:
            return None
    assignment = 0
    for pivot in sorted(pivots, reverse=True):
        mask, rhs = pivots[pivot]
        other = mask & ~(1 << pivot)
        value = rhs ^ ((assignment & other).bit_count() & 1)
        if value:
            assignment |= 1 << pivot
    if assignment >> variable_count:
        raise RuntimeError("GF(2) elimination set an out-of-range variable")
    return assignment


def find_parity_cover_bounded_span(
    shapes: Mapping[str, Iterable[int]] | Iterable[Iterable[int]],
    max_point: int,
    *,
    minimize_block_count: bool = False,
) -> ParityCoverWitness:
    """Find an odd collection of translated shapes with even point incidence."""

    if type(max_point) is not int or max_point < 0:
        raise ValueError("max_point must be a nonnegative ordinary integer")
    blocks = tuple(
        enumerate_translated_blocks(
            shapes, 0, max_point, include_reflections=False
        )
    )
    if not blocks:
        raise ValueError("no shape translation fits the proposed span")
    equations: list[tuple[int, int]] = []
    for point in range(max_point + 1):
        mask = sum(
            1 << index for index, block in enumerate(blocks) if point in block.points
        )
        equations.append((mask, 0))
    equations.append(((1 << len(blocks)) - 1, 1))
    assignment = _solve_gf2(equations, len(blocks))
    if assignment is None:
        raise ValueError(f"no odd parity cover exists through point {max_point}")
    if minimize_block_count:
        model = cp_model.CpModel()
        variables = [model.new_bool_var(f"translation_{index}") for index in range(len(blocks))]
        for point in range(max_point + 1):
            incident = [
                variables[index]
                for index, block in enumerate(blocks)
                if point in block.points
            ]
            half = model.new_int_var(0, len(incident) // 2, f"point_half_{point}")
            model.add(sum(incident) == 2 * half)
        total_half = model.new_int_var(0, len(blocks) // 2, "total_half")
        model.add(sum(variables) == 2 * total_half + 1)
        model.minimize(sum(variables))
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        status = solver.solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError("GF(2)-feasible cover was unexpectedly CP-SAT infeasible")
        assignment = sum(
            (1 << index) * int(solver.value(variable))
            for index, variable in enumerate(variables)
        )
    selected = tuple(
        block for index, block in enumerate(blocks) if assignment & (1 << index)
    )
    parities = [0] * (max_point + 1)
    for block in selected:
        for point in block.points:
            parities[point] ^= 1
    verified = not any(parities) and len(selected) % 2 == 1
    if not verified:
        raise RuntimeError("linear-algebra result failed expanded parity verification")
    return ParityCoverWitness(selected, max_point, tuple(parities), True, True)


def verify_parity_cover_witness(witness: ParityCoverWitness) -> bool:
    parities = [0] * (witness.max_point + 1)
    for block in witness.selected_blocks:
        if min(block.points) < 0 or max(block.points) > witness.max_point:
            return False
        for point in block.points:
            parities[point] ^= 1
    return (
        witness.verified
        and not any(parities)
        and tuple(parities) == witness.point_parities
        and len(witness.selected_blocks) % 2 == 1
        and witness.odd_block_count
    )


def _shape_polynomial(points: Iterable[int]) -> int:
    normalized = tuple(sorted(set(points)))
    if not normalized:
        raise ValueError("parity shape must be nonempty")
    minimum = min(normalized)
    return gf2_from_exponents(point - minimum for point in normalized)


def find_two_shape_bezout_relation(
    left_shape: Iterable[int],
    right_shape: Iterable[int],
) -> BezoutRelation:
    left = _shape_polynomial(left_shape)
    right = _shape_polynomial(right_shape)
    gcd, left_coefficient, right_coefficient = gf2_xgcd(left, right)
    verified = (
        gf2_mul(left_coefficient, left) ^ gf2_mul(right_coefficient, right)
    ) == gcd
    if not verified:
        raise RuntimeError("extended-Euclid relation failed exact verification")
    return BezoutRelation(
        left, right, gcd, left_coefficient, right_coefficient, True
    )


def verify_bezout_relation(witness: BezoutRelation) -> bool:
    return (
        witness.verified
        and gf2_gcd(witness.left, witness.right) == witness.gcd
        and (
            gf2_mul(witness.left_coefficient, witness.left)
            ^ gf2_mul(witness.right_coefficient, witness.right)
        )
        == witness.gcd
    )


def find_multi_shape_bezout_relation(
    shapes: Mapping[str, Iterable[int]] | Iterable[Iterable[int]],
) -> MultiShapeBezoutRelation:
    """Construct a gcd certificate iteratively across several shapes."""

    supplied = tuple(shapes.values()) if isinstance(shapes, Mapping) else tuple(shapes)
    polynomials = tuple(_shape_polynomial(shape) for shape in supplied)
    if not polynomials:
        raise ValueError("multi-shape Bézout construction needs at least one shape")
    current_gcd = polynomials[0]
    coefficients = [1]
    for polynomial in polynomials[1:]:
        new_gcd, scale_existing, coefficient_new = gf2_xgcd(
            current_gcd, polynomial
        )
        coefficients = [
            gf2_mul(scale_existing, coefficient) for coefficient in coefficients
        ] + [coefficient_new]
        current_gcd = new_gcd
    verified = (
        reduce(
            int.__xor__,
            (
                gf2_mul(coefficient, polynomial)
                for coefficient, polynomial in zip(coefficients, polynomials)
            ),
            0,
        )
        == current_gcd
    )
    if not verified:
        raise RuntimeError("iterative multi-shape Bézout certificate failed")
    return MultiShapeBezoutRelation(
        polynomials, current_gcd, tuple(coefficients), True
    )


def verify_multi_shape_bezout_relation(
    witness: MultiShapeBezoutRelation,
) -> bool:
    combined = 0
    for coefficient, polynomial in zip(witness.coefficients, witness.polynomials):
        combined ^= gf2_mul(coefficient, polynomial)
    gcd = 0
    for polynomial in witness.polynomials:
        gcd = gf2_gcd(gcd, polynomial)
    return witness.verified and combined == witness.gcd == gcd


def find_multi_shape_syzygy(
    shapes: Mapping[str, Iterable[int]] | Iterable[Iterable[int]],
    maximum_point: int,
) -> ParityCoverWitness:
    """Use bounded-degree packed linear algebra for a multi-shape syzygy."""

    return find_parity_cover_bounded_span(shapes, maximum_point)


def minimize_relation_degree(
    shapes: Mapping[str, Iterable[int]] | Iterable[Iterable[int]],
    maximum_point: int,
) -> ParityCoverWitness:
    """Return the first feasible odd relation in increasing ambient degree."""

    named = tuple(
        (str(name), tuple(points))
        for name, points in (
            shapes.items() if isinstance(shapes, Mapping) else enumerate(shapes)
        )
    )
    spans = [max(points) - min(points) for _, points in named]
    for candidate in range(max(spans), maximum_point + 1):
        try:
            return find_parity_cover_bounded_span(
                dict(named), candidate, minimize_block_count=True
            )
        except ValueError:
            continue
    raise ValueError(f"no odd parity relation exists through point {maximum_point}")


def _series(length: int) -> int:
    if length < 1:
        raise ValueError("series length must be positive")
    return (1 << length) - 1


def _expanded_relation_is_even(
    shapes: tuple[tuple[int, ...], ...],
    coefficients: tuple[int, ...],
    maximum_point: int,
) -> bool:
    incidence = bytearray(maximum_point + 1)
    for shape, coefficient in zip(shapes, coefficients):
        for translation in gf2_to_exponents(coefficient):
            for point in shape:
                incidence[translation + point] ^= 1
    return not any(incidence)


@lru_cache(maxsize=1)
def square_submillion_parity_witness() -> SquareParityWitness:
    """Construct and independently verify the supplied square K4 relation."""

    started = time.perf_counter()
    shapes = (A_POINTS, B_POINTS, C_POINTS)
    if not all(is_square_distance_clique(shape) for shape in shapes):
        raise RuntimeError("a supplied square-parity shape is not a square-distance K4")
    a, b, c = tuple(gf2_from_exponents(shape) for shape in shapes)
    q = gf2_mul(_series(9), _series(25))
    g = gf2_mul(0b11, q)
    if gf2_degree(q) != 32 or gf2_degree(g) != 33:
        raise RuntimeError("supplied q/g degree identity failed")
    if gf2_eval_at_one(q) != 1:
        raise RuntimeError("supplied q does not evaluate to one at X=1")
    gcd_ab = gf2_gcd(a, b)
    if gcd_ab != g:
        raise RuntimeError("supplied identity gcd(A,B)=g failed")
    a0 = gf2_div_exact_fast(a, g)
    b0 = gf2_div_exact_fast(b, g)
    t = gf2_div_exact_fast(c, 0b11)
    inverse = gf2_mod_inverse(a0, b0)
    u = gf2_mul_mod(t, inverse, b0)
    v = gf2_div_exact_fast(t ^ gf2_mul(u, a0), b0)
    relation = gf2_mul(u, a) ^ gf2_mul(v, b) ^ gf2_mul(q, c)
    if relation:
        raise RuntimeError(
            f"supplied square relation failed at degree {gf2_degree(relation)}"
        )
    parity = gf2_eval_at_one(u) ^ gf2_eval_at_one(v) ^ gf2_eval_at_one(q)
    if parity != 1:
        raise RuntimeError("supplied square relation has even block count")
    maximum_point = max(
        gf2_degree(gf2_mul(u, a)),
        gf2_degree(gf2_mul(v, b)),
        gf2_degree(gf2_mul(q, c)),
    )
    if maximum_point != 971584:
        raise RuntimeError(
            f"supplied square relation maximum is {maximum_point}, expected 971584"
        )
    expanded = _expanded_relation_is_even(shapes, (u, v, q), maximum_point)
    if not expanded:
        raise RuntimeError("expanded square relation has an odd-incidence point")
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = int(peak_rss if sys.platform == "darwin" else peak_rss * 1024)
    packed_bytes = sum(
        max(1, (polynomial.bit_length() + 7) // 8)
        for polynomial in (a, b, c, q, gcd_ab, u, v)
    )
    return SquareParityWitness(
        shapes,
        (a, b, c),
        q,
        gcd_ab,
        u,
        v,
        True,
        True,
        True,
        maximum_point,
        maximum_point + 1,
        (gf2_degree(a0), gf2_degree(b0), gf2_degree(t)),
        (gf2_degree(u), gf2_degree(v), gf2_degree(q)),
        (
            gf2_degree(gf2_mul(u, a)),
            gf2_degree(gf2_mul(v, b)),
            gf2_degree(gf2_mul(q, c)),
        ),
        time.perf_counter() - started,
        peak,
        packed_bytes,
    )


def polynomial_summary(polynomial: int) -> dict[str, int | str]:
    """Summarize a packed polynomial without printing a dense coefficient list."""

    byte_count = max(1, (polynomial.bit_length() + 7) // 8)
    digest = hashlib.sha256(polynomial.to_bytes(byte_count, "little")).hexdigest()
    return {
        "degree": gf2_degree(polynomial),
        "bit_count": polynomial.bit_count(),
        "parity_at_one": gf2_eval_at_one(polynomial),
        "sha256": digest,
    }


def verify_square_parity_witness(witness: SquareParityWitness) -> bool:
    if not all(is_square_distance_clique(shape) for shape in witness.shapes):
        return False
    expected_polynomials = tuple(
        gf2_from_exponents(shape) for shape in witness.shapes
    )
    if expected_polynomials != witness.shape_polynomials:
        return False
    a, b, c = witness.shape_polynomials
    expected_q = gf2_mul(_series(9), _series(25))
    if witness.q != expected_q:
        return False
    relation = (
        gf2_mul(witness.u, a)
        ^ gf2_mul(witness.v, b)
        ^ gf2_mul(witness.q, c)
    )
    maximum = max(
        gf2_degree(gf2_mul(witness.u, a)),
        gf2_degree(gf2_mul(witness.v, b)),
        gf2_degree(gf2_mul(witness.q, c)),
    )
    return (
        witness.relation_zero
        and witness.odd_block_count
        and witness.expanded_even_cover
        and relation == 0
        and gf2_gcd(a, b) == witness.gcd_ab
        and (gf2_eval_at_one(witness.u) ^ gf2_eval_at_one(witness.v) ^ gf2_eval_at_one(witness.q)) == 1
        and maximum == witness.maximum_point
        and witness.interval_size == maximum + 1
        and _expanded_relation_is_even(
            witness.shapes, (witness.u, witness.v, witness.q), maximum
        )
    )
