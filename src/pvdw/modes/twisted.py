"""Exact finite-interval twisted-periodic quotient models."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from pvdw.backends.base import DecodeSpec, EncodedProblem, ExplicitGraph, SearchBackend, SolveOptions
from pvdw.distances import generate_distances
from pvdw.encoding.common import BinaryVariableMap, OneHotVariableMap
from pvdw.model import InstanceSpec, ModelScope, SolveResult, SolveStatus
from pvdw.modes.direct import EncodingOptions, encode_graph_problem
from pvdw.verify import VerificationResult, verify_coloring


@dataclass(frozen=True, order=True)
class TwistedConstraint:
    left_residue: int
    right_residue: int
    shift: int


@dataclass(frozen=True)
class TwistedModel:
    period: int
    twist: int
    colors: int
    constraints: tuple[TwistedConstraint, ...]
    immediate_impossibility: bool = False

    @property
    def effective_period(self) -> int:
        return self.period * self.colors // math.gcd(self.colors, self.twist)


def _canonical_constraint(
    left: int,
    right: int,
    shift: int,
    colors: int,
) -> TwistedConstraint:
    direct = TwistedConstraint(left, right, shift % colors)
    reverse = TwistedConstraint(right, left, (-shift) % colors)
    return min(direct, reverse)


def build_twisted_model(
    instance: InstanceSpec,
    period: int,
    twist: int,
) -> TwistedModel:
    if type(period) is not int or period < 1:
        raise ValueError("period must be a positive ordinary integer")
    if type(twist) is not int:
        raise TypeError("twist must be an ordinary integer")
    twist %= instance.colors
    constraints: set[TwistedConstraint] = set()
    impossible = False
    for distance in generate_distances(instance).values:
        start_count = instance.n - distance
        residues = range(period) if start_count >= period else range(start_count)
        for residue in residues:
            right = (residue + distance) % period
            carry = (residue + distance) // period
            shift = (twist * carry) % instance.colors
            if residue == right:
                if shift == 0:
                    impossible = True
                # A nonzero self-shift says a != a + shift and is tautological.
                continue
            constraints.add(
                _canonical_constraint(
                    residue, right, shift, instance.colors
                )
            )
    return TwistedModel(
        period,
        twist,
        instance.colors,
        tuple(sorted(constraints)),
        impossible,
    )


def lift_twisted(
    pattern: tuple[int, ...] | list[int],
    n: int,
    colors: int,
    period: int,
    twist: int,
) -> tuple[int, ...]:
    if len(pattern) != period or period < 1:
        raise ValueError("twisted pattern length must equal the positive period")
    if n < 1 or colors < 2:
        raise ValueError("twisted lift requires n >= 1 and colors >= 2")
    if any(type(color) is not int or not 0 <= color < colors for color in pattern):
        raise ValueError("twisted base color is out of range")
    return tuple(
        (pattern[index % period] + twist * (index // period)) % colors
        for index in range(n)
    )


def verify_twisted_pattern(
    instance: InstanceSpec,
    pattern: tuple[int, ...] | list[int],
    period: int,
    twist: int,
) -> VerificationResult:
    return verify_coloring(
        instance,
        lift_twisted(pattern, instance.n, instance.colors, period, twist),
    )


def twisted_pattern_satisfies_model(
    model: TwistedModel,
    pattern: tuple[int, ...] | list[int],
) -> bool:
    if len(pattern) != model.period:
        raise ValueError("twisted pattern length differs from its period")
    if model.immediate_impossibility:
        return False
    return all(
        pattern[constraint.left_residue]
        != (pattern[constraint.right_residue] + constraint.shift) % model.colors
        for constraint in model.constraints
    )


def build_twisted_problem(
    instance: InstanceSpec,
    period: int,
    twist: int,
    encoding_options: EncodingOptions,
) -> tuple[TwistedModel, EncodedProblem | None]:
    model = build_twisted_model(instance, period, twist)
    if model.immediate_impossibility:
        return model, None
    quotient_instance = InstanceSpec(
        instance.polynomial,
        instance.colors,
        period,
        instance.input_domain,
    )
    graph = ExplicitGraph(period, ())
    potts_constraints = tuple(
        (
            constraint.left_residue,
            constraint.right_residue,
            constraint.shift,
        )
        for constraint in model.constraints
    )
    decode_spec = DecodeSpec(
        encoding=(
            "binary"
            if encoding_options.encoding == "binary"
            else f"onehot-{encoding_options.amo.value}"
        ),
        assignment_vertices=period,
        colors=instance.colors,
        mode="twisted",
        output_n=instance.n,
        period=period,
        twist=model.twist,
    )
    base = encode_graph_problem(
        instance,
        quotient_instance,
        graph,
        encoding_options,
        scope=ModelScope.TWISTED,
        decode_spec=decode_spec,
        potts_constraints=potts_constraints,
        metadata={
            "mode": "twisted",
            "period": period,
            "twist": model.twist,
            "effective_period": model.effective_period,
            "finite_interval": True,
            "quotient_constraints": len(model.constraints),
        },
    )
    clauses = list(base.clauses)
    if encoding_options.encoding == "onehot":
        variables = OneHotVariableMap(period, instance.colors)
        for constraint in model.constraints:
            for color in range(instance.colors):
                clauses.append(
                    (
                        -variables.var(constraint.left_residue, color),
                        -variables.var(
                            constraint.right_residue,
                            (color - constraint.shift) % instance.colors,
                        ),
                    )
                )
    else:
        bits = (instance.colors - 1).bit_length()
        variables = BinaryVariableMap(period, bits)

        def differs(vertex: int, code: int) -> tuple[int, ...]:
            return tuple(
                -variables.var(vertex, bit)
                if (code >> bit) & 1
                else variables.var(vertex, bit)
                for bit in range(bits)
            )

        for constraint in model.constraints:
            for color in range(instance.colors):
                right_color = (color - constraint.shift) % instance.colors
                clauses.append(
                    differs(constraint.left_residue, color)
                    + differs(constraint.right_residue, right_color)
                )
    literal_count = sum(len(clause) for clause in clauses)
    encoding_options.limits.check(base.variable_count, len(clauses), literal_count)
    return model, EncodedProblem(
        instance=instance,
        variable_count=base.variable_count,
        clauses=tuple(clauses),
        literal_count=literal_count,
        encoding=base.encoding,
        scope=ModelScope.TWISTED,
        decode_spec=decode_spec,
        graph=graph,
        potts_constraints=potts_constraints,
        metadata=base.metadata,
    )


def solve_twisted(
    instance: InstanceSpec,
    period: int,
    twist: int,
    encoding_options: EncodingOptions,
    backend: SearchBackend,
    solve_options: SolveOptions,
) -> SolveResult:
    model, problem = build_twisted_problem(
        instance, period, twist, encoding_options
    )
    if problem is None:
        return SolveResult(
            SolveStatus.NO_WITNESS_IN_RESTRICTED_MODEL,
            ModelScope.TWISTED,
            0.0,
            backend.name,
            None,
            {
                "mode": "twisted",
                "period": period,
                "twist": model.twist,
                "effective_period": model.effective_period,
                "finite_interval": True,
                "immediate_impossibility": True,
                "quotient_constraints": len(model.constraints) + 1,
                "formula_variables": 0,
                "formula_clauses": 1,
                "formula_literals": 0,
                "seed": solve_options.seed,
            },
        )
    result = backend.solve(problem, solve_options)
    if result.status is SolveStatus.UNSAT_FULL_MODEL:
        raise RuntimeError("twisted backend incorrectly reported full-model UNSAT")
    return result


@dataclass(frozen=True)
class TwistAttempt:
    period: int
    twist: int
    variables: int
    constraints: int
    immediate_impossibility: bool
    status: SolveStatus
    elapsed_seconds: float
    best_energy: int | None


@dataclass(frozen=True)
class TwistScanResult:
    attempts: tuple[TwistAttempt, ...]
    witness: SolveResult | None


def scan_twists(
    instance: InstanceSpec,
    period_min: int,
    period_max: int,
    twists: tuple[int, ...],
    encoding_options: EncodingOptions,
    backend: SearchBackend,
    solve_options: SolveOptions,
) -> TwistScanResult:
    if not 1 <= period_min <= period_max or not twists:
        raise ValueError("twisted scan range is invalid")
    attempts: list[TwistAttempt] = []
    witness: SolveResult | None = None
    for period in range(period_min, period_max + 1):
        for twist in twists:
            started = time.perf_counter()
            result = solve_twisted(
                instance,
                period,
                twist,
                encoding_options,
                backend,
                solve_options,
            )
            metadata = dict(result.metadata)
            attempts.append(
                TwistAttempt(
                    period,
                    twist % instance.colors,
                    int(metadata.get("formula_variables", 0)),
                    int(metadata.get("quotient_constraints", 0)),
                    bool(metadata.get("immediate_impossibility", False)),
                    result.status,
                    time.perf_counter() - started,
                    result.best_energy,
                )
            )
            if result.status is SolveStatus.FOUND_WITNESS:
                witness = result
                return TwistScanResult(tuple(attempts), witness)
    return TwistScanResult(tuple(attempts), witness)
