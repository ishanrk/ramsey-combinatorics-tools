"""Shared encoded-problem and backend interfaces."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Protocol

from pvdw.encoding.binary import bits_for_colors
from pvdw.encoding.common import BinaryVariableMap, OneHotVariableMap
from pvdw.model import InstanceSpec, ModelScope, SolveResult, SolveStatus
from pvdw.verify import VerificationResult, verify_coloring


@dataclass(frozen=True)
class BackendCapabilities:
    complete: bool
    incremental: bool
    accepts_dimacs: bool
    supports_assumptions: bool
    stochastic: bool


@dataclass(frozen=True)
class SolveOptions:
    timeout_seconds: float | None = None
    seed: int = 0
    extra_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when supplied")
        if type(self.seed) is not int:
            raise TypeError("seed must be an ordinary integer")
        object.__setattr__(self, "extra_args", tuple(self.extra_args))


@dataclass(frozen=True)
class ExplicitGraph:
    """A deterministic explicit graph compatible with the Phase 1 encoders."""

    n: int
    edges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if type(self.n) is not int or self.n < 1:
            raise ValueError("explicit graph size must be positive")
        normalized = tuple(sorted(set(self.edges)))
        if any(not (0 <= u < v < self.n) for u, v in normalized):
            raise ValueError("explicit graph edges must satisfy 0 <= u < v < n")
        object.__setattr__(self, "edges", normalized)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def iter_edges(self) -> Iterator[tuple[int, int]]:
        yield from self.edges

    def build_adjacency(self) -> list[list[int]]:
        adjacency = [[] for _ in range(self.n)]
        for u, v in self.edges:
            adjacency[u].append(v)
            adjacency[v].append(u)
        for neighbors in adjacency:
            neighbors.sort()
        return adjacency


@dataclass(frozen=True)
class DecodeSpec:
    """Pickle-safe instructions for turning a SAT assignment into a coloring."""

    encoding: str
    assignment_vertices: int
    colors: int
    mode: str = "direct"
    output_n: int | None = None
    period: int | None = None
    twist: int = 0
    backbone: tuple[int, ...] | None = None
    editable_vertices: tuple[int, ...] = ()


@dataclass(frozen=True)
class EncodedProblem:
    """A CNF plus enough structure to decode and independently verify a model."""

    instance: InstanceSpec
    variable_count: int
    clauses: tuple[tuple[int, ...], ...]
    literal_count: int
    encoding: str
    scope: ModelScope
    decode_spec: DecodeSpec
    graph: object | None = None
    potts_constraints: tuple[tuple[int, int, int], ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.variable_count < 0 or self.literal_count < 0:
            raise ValueError("formula counts must be nonnegative")
        clauses = tuple(tuple(clause) for clause in self.clauses)
        if any(
            type(literal) is not int
            or literal == 0
            or abs(literal) > self.variable_count
            for clause in clauses
            for literal in clause
        ):
            raise ValueError("encoded problem contains an invalid literal")
        if sum(len(clause) for clause in clauses) != self.literal_count:
            raise ValueError("literal_count does not match the clauses")
        object.__setattr__(self, "clauses", clauses)
        object.__setattr__(self, "potts_constraints", tuple(self.potts_constraints))
        # A plain copied dict keeps EncodedProblem pickle-safe for process portfolios.
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def clause_count(self) -> int:
        return len(self.clauses)

    def decode_model(self, model: Sequence[int]) -> tuple[int, ...]:
        base = decode_assignment(model, self.decode_spec)
        spec = self.decode_spec
        if spec.mode == "direct":
            return base
        if spec.mode == "periodic":
            assert spec.period is not None and spec.output_n is not None
            return tuple(base[index % spec.period] for index in range(spec.output_n))
        if spec.mode == "twisted":
            assert spec.period is not None and spec.output_n is not None
            return tuple(
                (base[index % spec.period] + spec.twist * (index // spec.period))
                % spec.colors
                for index in range(spec.output_n)
            )
        if spec.mode == "repair":
            if spec.backbone is None:
                raise ValueError("repair decoder is missing its backbone")
            coloring = list(spec.backbone)
            if len(base) != len(spec.editable_vertices):
                raise ValueError("repair assignment has the wrong length")
            for vertex, color in zip(spec.editable_vertices, base):
                coloring[vertex] = color
            return tuple(coloring)
        raise ValueError(f"unknown decode mode {spec.mode!r}")

    def verify(self, coloring: Sequence[int]) -> VerificationResult:
        return verify_coloring(self.instance, coloring)


class SearchBackend(Protocol):
    name: str
    capabilities: BackendCapabilities

    def solve(self, problem: EncodedProblem, options: SolveOptions) -> SolveResult:
        ...


def _signed_assignment(model: Sequence[int], variable_count: int) -> dict[int, bool]:
    assignments: dict[int, bool] = {}
    for literal in model:
        if type(literal) is not int or literal == 0:
            raise ValueError("model literals must be nonzero ordinary integers")
        variable = abs(literal)
        if variable > variable_count:
            continue
        value = literal > 0
        if variable in assignments and assignments[variable] != value:
            raise ValueError(f"model contradicts itself on variable {variable}")
        assignments[variable] = value
    return assignments


def decode_assignment(model: Sequence[int], spec: DecodeSpec) -> tuple[int, ...]:
    """Strictly decode the primary one-hot or binary assignment."""

    if spec.encoding.startswith("onehot"):
        variables = OneHotVariableMap(spec.assignment_vertices, spec.colors)
        assignments = _signed_assignment(model, variables.primary_variables)
        missing = set(range(1, variables.primary_variables + 1)) - assignments.keys()
        if missing:
            raise ValueError(f"model omits one-hot variables: {sorted(missing)}")
        coloring: list[int] = []
        for vertex in range(spec.assignment_vertices):
            selected = [
                color
                for color in range(spec.colors)
                if assignments[variables.var(vertex, color)]
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"vertex {vertex} has {len(selected)} selected one-hot colors"
                )
            coloring.append(selected[0])
        return tuple(coloring)
    if spec.encoding == "binary":
        variable_map = BinaryVariableMap(
            spec.assignment_vertices, bits_for_colors(spec.colors)
        )
        assignments = _signed_assignment(model, variable_map.variables)
        missing = set(range(1, variable_map.variables + 1)) - assignments.keys()
        if missing:
            raise ValueError(f"model omits binary variables: {sorted(missing)}")
        coloring = []
        for vertex in range(spec.assignment_vertices):
            code = sum(
                int(assignments[variable_map.var(vertex, bit)]) << bit
                for bit in range(variable_map.bits)
            )
            if code >= spec.colors:
                raise ValueError(f"vertex {vertex} has invalid color code {code}")
            coloring.append(code)
        return tuple(coloring)
    raise ValueError(f"unsupported assignment encoding {spec.encoding!r}")


def negative_status(scope: ModelScope) -> SolveStatus:
    return (
        SolveStatus.UNSAT_FULL_MODEL
        if scope is ModelScope.FULL
        else SolveStatus.NO_WITNESS_IN_RESTRICTED_MODEL
    )


def base_metadata(
    problem: EncodedProblem,
    options: SolveOptions,
    *,
    backend_version: str | None = None,
) -> dict[str, object]:
    return {
        "backend_version": backend_version,
        "seed": options.seed,
        "formula_variables": problem.variable_count,
        "formula_clauses": problem.clause_count,
        "formula_literals": problem.literal_count,
        "return_code": None,
        "command": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "model_parsing": "not_attempted",
        "verification": "not_attempted",
        **dict(problem.metadata),
    }
