"""Unrestricted direct encoding and solving."""

from __future__ import annotations

from dataclasses import dataclass, field

from pysat.card import CardEnc, EncType

from pvdw.backends.base import DecodeSpec, EncodedProblem, SearchBackend, SolveOptions
from pvdw.distances import generate_distances
from pvdw.encoding.binary import binary_statistics, encode_binary
from pvdw.encoding.common import ListClauseSink
from pvdw.encoding.onehot import (
    AtMostOneEncoding,
    encode_onehot,
    onehot_pairwise_statistics,
)
from pvdw.graph import DistanceGraph
from pvdw.model import InstanceSpec, ModelScope, SolveResult, SolveStatus
from pvdw.verify import verify_coloring


@dataclass(frozen=True)
class FormulaLimits:
    max_variables: int | None = 2_000_000
    max_clauses: int | None = 20_000_000
    max_estimated_bytes: int | None = 1_000_000_000

    def check(self, variables: int, clauses: int, literals: int) -> None:
        if self.max_variables is not None and variables > self.max_variables:
            raise ValueError(
                f"formula estimate {variables} variables exceeds limit {self.max_variables}"
            )
        if self.max_clauses is not None and clauses > self.max_clauses:
            raise ValueError(
                f"formula estimate {clauses} clauses exceeds limit {self.max_clauses}"
            )
        estimated_bytes = clauses * 64 + literals * 8
        if self.max_estimated_bytes is not None and estimated_bytes > self.max_estimated_bytes:
            raise ValueError(
                f"formula estimate {estimated_bytes} bytes exceeds limit "
                f"{self.max_estimated_bytes}"
            )


@dataclass(frozen=True)
class EncodingOptions:
    encoding: str = "onehot"
    amo: AtMostOneEncoding = AtMostOneEncoding.PAIRWISE
    fix_first_color: bool = True
    anchor_clique: bool = False
    limits: FormulaLimits = field(default_factory=FormulaLimits)

    def __post_init__(self) -> None:
        if self.encoding not in ("onehot", "binary"):
            raise ValueError("encoding must be 'onehot' or 'binary'")
        if not isinstance(self.amo, AtMostOneEncoding):
            object.__setattr__(self, "amo", AtMostOneEncoding(self.amo))


def _preflight_estimate(
    instance: InstanceSpec,
    graph: object,
    options: EncodingOptions,
) -> tuple[int, int, int]:
    if options.encoding == "binary":
        statistics = binary_statistics(
            instance, graph, fix_first_color=options.fix_first_color  # type: ignore[arg-type]
        )
    elif options.amo is AtMostOneEncoding.PAIRWISE:
        statistics = onehot_pairwise_statistics(
            instance, graph, fix_first_color=options.fix_first_color  # type: ignore[arg-type]
        )
    else:
        encoding = {
            AtMostOneEncoding.SEQUENTIAL: EncType.seqcounter,
            AtMostOneEncoding.LADDER: EncType.ladder,
            AtMostOneEncoding.BITWISE: EncType.bitwise,
        }[options.amo]
        sample = CardEnc.atmost(
            lits=list(range(1, instance.colors + 1)),
            bound=1,
            top_id=instance.colors,
            encoding=encoding,
        )
        auxiliaries_per_vertex = sample.nv - instance.colors
        variables = instance.n * (
            instance.colors + auxiliaries_per_vertex
        )
        clauses = (
            instance.n * (1 + len(sample.clauses))
            + graph.edge_count * instance.colors  # type: ignore[attr-defined]
            + int(options.fix_first_color)
            + (instance.colors if options.anchor_clique else 0)
        )
        literals = (
            instance.n * instance.colors
            + instance.n * sum(len(clause) for clause in sample.clauses)
            + graph.edge_count * instance.colors * 2  # type: ignore[attr-defined]
            + int(options.fix_first_color)
            + (instance.colors if options.anchor_clique else 0)
        )
        options.limits.check(variables, clauses, literals)
        return variables, clauses, literals
    options.limits.check(
        statistics.variables, statistics.clauses, statistics.literals
    )
    return statistics.variables, statistics.clauses, statistics.literals


def encode_graph_problem(
    original_instance: InstanceSpec,
    assignment_instance: InstanceSpec,
    graph: object,
    encoding_options: EncodingOptions,
    *,
    scope: ModelScope,
    decode_spec: DecodeSpec,
    potts_constraints: tuple[tuple[int, int, int], ...] | None = None,
    metadata: dict[str, object] | None = None,
) -> EncodedProblem:
    """Encode a direct or quotient graph through the Phase 1 encoders."""

    _preflight_estimate(assignment_instance, graph, encoding_options)
    sink = ListClauseSink()
    if encoding_options.encoding == "binary":
        result = encode_binary(
            assignment_instance,
            graph,  # type: ignore[arg-type]
            sink,
            fix_first_color=encoding_options.fix_first_color,
        )
        statistics = result.statistics
    else:
        result = encode_onehot(
            assignment_instance,
            graph,  # type: ignore[arg-type]
            sink,
            amo=encoding_options.amo,
            fix_first_color=encoding_options.fix_first_color,
            anchor_clique=encoding_options.anchor_clique,
        )
        statistics = result.statistics
    encoding_options.limits.check(
        statistics.variables, statistics.clauses, statistics.literals
    )
    constraints = potts_constraints
    if constraints is None:
        constraints = tuple((u, v, 0) for u, v in graph.iter_edges())  # type: ignore[attr-defined]
    return EncodedProblem(
        instance=original_instance,
        variable_count=statistics.variables,
        clauses=tuple(tuple(clause) for clause in sink.clauses),
        literal_count=statistics.literals,
        encoding=statistics.encoding,
        scope=scope,
        decode_spec=decode_spec,
        graph=graph,
        potts_constraints=constraints,
        metadata=metadata or {},
    )


def build_direct_problem(
    instance: InstanceSpec,
    encoding_options: EncodingOptions,
) -> EncodedProblem:
    data = generate_distances(instance)
    graph = DistanceGraph(instance.n, data.values)
    return encode_graph_problem(
        instance,
        instance,
        graph,
        encoding_options,
        scope=ModelScope.FULL,
        decode_spec=DecodeSpec(
            encoding=(
                "binary"
                if encoding_options.encoding == "binary"
                else f"onehot-{encoding_options.amo.value}"
            ),
            assignment_vertices=instance.n,
            colors=instance.colors,
        ),
        metadata={
            "mode": "direct",
            "edge_count": graph.edge_count,
            "distances": list(data.values),
        },
    )


def solve_direct(
    instance: InstanceSpec,
    encoding_options: EncodingOptions,
    backend: SearchBackend,
    solve_options: SolveOptions,
) -> SolveResult:
    """Build and solve the unrestricted full coloring model."""

    problem = build_direct_problem(instance, encoding_options)
    result = backend.solve(problem, solve_options)
    if result.status is SolveStatus.FOUND_WITNESS:
        if result.coloring is None or not verify_coloring(instance, result.coloring).valid:
            raise RuntimeError("backend witness failed direct independent verification")
    if (
        result.status is SolveStatus.UNSAT_FULL_MODEL
        and not backend.capabilities.complete
    ):
        raise RuntimeError("an incomplete backend attempted to report full UNSAT")
    return result
