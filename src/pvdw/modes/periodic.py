"""Exact finite-interval periodic quotient models."""

from __future__ import annotations

import time
from dataclasses import dataclass

from pvdw.backends.base import (
    DecodeSpec,
    EncodedProblem,
    ExplicitGraph,
    SearchBackend,
    SolveOptions,
)
from pvdw.distances import generate_distances
from pvdw.model import InstanceSpec, ModelScope, SolveResult, SolveStatus
from pvdw.modes.direct import EncodingOptions, encode_graph_problem
from pvdw.verify import VerificationResult, verify_coloring


@dataclass(frozen=True)
class PeriodicModel:
    period: int
    edges: tuple[tuple[int, int], ...]

    @property
    def immediate_impossibility(self) -> bool:
        return any(left == right for left, right in self.edges)


def build_periodic_model(instance: InstanceSpec, period: int) -> PeriodicModel:
    """Build only residue edges realized by the finite interval."""

    if type(period) is not int or period < 1:
        raise ValueError("period must be a positive ordinary integer")
    data = generate_distances(instance)
    edges: set[tuple[int, int]] = set()
    for distance in data.values:
        start_count = instance.n - distance
        residues = range(period) if start_count >= period else range(start_count)
        for residue in residues:
            other = (residue + distance) % period
            edges.add((min(residue, other), max(residue, other)))
    return PeriodicModel(period, tuple(sorted(edges)))


def lift_periodic(pattern: tuple[int, ...] | list[int], n: int) -> tuple[int, ...]:
    if not pattern:
        raise ValueError("periodic pattern must be nonempty")
    if type(n) is not int or n < 1:
        raise ValueError("lift length must be positive")
    return tuple(pattern[index % len(pattern)] for index in range(n))


def verify_periodic_pattern(
    instance: InstanceSpec,
    pattern: tuple[int, ...] | list[int],
) -> VerificationResult:
    return verify_coloring(instance, lift_periodic(pattern, instance.n))


def periodic_pattern_satisfies_model(
    model: PeriodicModel,
    pattern: tuple[int, ...] | list[int],
) -> bool:
    if len(pattern) != model.period:
        raise ValueError("periodic pattern length differs from its period")
    return all(pattern[left] != pattern[right] for left, right in model.edges)


def build_periodic_problem(
    instance: InstanceSpec,
    period: int,
    encoding_options: EncodingOptions,
) -> tuple[PeriodicModel, EncodedProblem | None]:
    model = build_periodic_model(instance, period)
    if model.immediate_impossibility:
        return model, None
    graph = ExplicitGraph(period, model.edges)
    quotient_instance = InstanceSpec(
        instance.polynomial,
        instance.colors,
        period,
        instance.input_domain,
    )
    problem = encode_graph_problem(
        instance,
        quotient_instance,
        graph,
        encoding_options,
        scope=ModelScope.PERIODIC,
        decode_spec=DecodeSpec(
            encoding=(
                "binary"
                if encoding_options.encoding == "binary"
                else f"onehot-{encoding_options.amo.value}"
            ),
            assignment_vertices=period,
            colors=instance.colors,
            mode="periodic",
            output_n=instance.n,
            period=period,
        ),
        metadata={
            "mode": "periodic",
            "period": period,
            "finite_interval": True,
            "quotient_constraints": len(model.edges),
        },
    )
    return model, problem


def solve_periodic(
    instance: InstanceSpec,
    period: int,
    encoding_options: EncodingOptions,
    backend: SearchBackend,
    solve_options: SolveOptions,
) -> SolveResult:
    model, problem = build_periodic_problem(instance, period, encoding_options)
    if problem is None:
        metadata = {
            "mode": "periodic",
            "period": period,
            "finite_interval": True,
            "immediate_impossibility": True,
            "quotient_constraints": len(model.edges),
            "formula_variables": 0,
            "formula_clauses": 1,
            "formula_literals": 0,
            "seed": solve_options.seed,
        }
        return SolveResult(
            SolveStatus.NO_WITNESS_IN_RESTRICTED_MODEL,
            ModelScope.PERIODIC,
            0.0,
            backend.name,
            None,
            metadata,
        )
    result = backend.solve(problem, solve_options)
    if result.status is SolveStatus.UNSAT_FULL_MODEL:
        raise RuntimeError("periodic backend incorrectly reported full-model UNSAT")
    return result


@dataclass(frozen=True)
class PeriodAttempt:
    period: int
    variables: int
    constraints: int
    immediate_impossibility: bool
    status: SolveStatus
    elapsed_seconds: float
    best_energy: int | None


@dataclass(frozen=True)
class PeriodScanResult:
    attempts: tuple[PeriodAttempt, ...]
    witness: SolveResult | None


def scan_periods(
    instance: InstanceSpec,
    period_min: int,
    period_max: int,
    encoding_options: EncodingOptions,
    backend: SearchBackend,
    solve_options: SolveOptions,
) -> PeriodScanResult:
    if not 1 <= period_min <= period_max:
        raise ValueError("period range is invalid")
    attempts: list[PeriodAttempt] = []
    witness: SolveResult | None = None
    for period in range(period_min, period_max + 1):
        started = time.perf_counter()
        result = solve_periodic(
            instance, period, encoding_options, backend, solve_options
        )
        metadata = dict(result.metadata)
        attempts.append(
            PeriodAttempt(
                period=period,
                variables=int(metadata.get("formula_variables", 0)),
                constraints=int(metadata.get("quotient_constraints", 0)),
                immediate_impossibility=bool(
                    metadata.get("immediate_impossibility", False)
                ),
                status=result.status,
                elapsed_seconds=time.perf_counter() - started,
                best_energy=result.best_energy,
            )
        )
        if result.status is SolveStatus.FOUND_WITNESS:
            witness = result
            break
    return PeriodScanResult(tuple(attempts), witness)
