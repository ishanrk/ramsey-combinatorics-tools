"""Optional stochastic UBCSAT execution; exhaustion is never an UNSAT proof."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pvdw.backends.base import BackendCapabilities, EncodedProblem, SolveOptions
from pvdw.backends.external_sat import ExternalSatBackend, ExternalSolverConfig
from pvdw.model import SolveResult, SolveStatus


SUPPORTED_ALGORITHMS = (
    "walksat",
    "walksat-tabu",
    "adaptnovelty+",
    "g2wsat",
    "saps",
)


@dataclass(frozen=True)
class UbcsatOptions:
    algorithm: str = "walksat"
    runs: int = 20
    cutoff: int = 1_000_000
    extra_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(f"unsupported UBCSAT algorithm {self.algorithm!r}")
        if self.runs < 1 or self.cutoff < 1:
            raise ValueError("UBCSAT runs and cutoff must be positive")
        object.__setattr__(self, "extra_args", tuple(self.extra_args))


def probe_ubcsat_algorithms(executable: str | os.PathLike[str]) -> set[str]:
    located = shutil.which(str(executable)) or str(executable)
    for argument in ("-algorithms", "-help"):
        try:
            completed = subprocess.run(
                [located, argument],
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        output = (completed.stdout + "\n" + completed.stderr).lower()
        discovered = {
            algorithm for algorithm in SUPPORTED_ALGORITHMS if algorithm in output
        }
        if discovered:
            return discovered
    return set()


class UbcsatBackend:
    capabilities = BackendCapabilities(
        complete=False,
        incremental=False,
        accepts_dimacs=True,
        supports_assumptions=False,
        stochastic=True,
    )

    def __init__(
        self,
        executable: str | os.PathLike[str] = "ubcsat",
        options: UbcsatOptions | None = None,
        *,
        probe: bool = True,
    ) -> None:
        self.executable = Path(executable)
        self.ubcsat_options = options or UbcsatOptions()
        self.name = f"ubcsat-{self.ubcsat_options.algorithm}"
        if probe:
            algorithms = probe_ubcsat_algorithms(executable)
            if algorithms and self.ubcsat_options.algorithm not in algorithms:
                raise ValueError(
                    f"installed UBCSAT does not list {self.ubcsat_options.algorithm!r}; "
                    f"available: {sorted(algorithms)}"
                )

    def solve(self, problem: EncodedProblem, options: SolveOptions) -> SolveResult:
        ubcsat = self.ubcsat_options
        command_template = (
            "{executable}",
            "-alg",
            ubcsat.algorithm,
            "-i",
            "{cnf}",
            "-solve",
            "-runs",
            str(ubcsat.runs),
            "-cutoff",
            str(ubcsat.cutoff),
            "-seed",
            "{seed}",
        ) + ubcsat.extra_args
        delegate = ExternalSatBackend(
            ExternalSolverConfig(
                self.name,
                self.executable,
                command_template,
                # Exit 0 also means ordinary exhausted runs; require an explicit
                # SAT status (or conventional 10) before attempting a model.
                sat_exit_codes=frozenset({10}),
                unsat_exit_codes=frozenset(),
            )
        )
        result = delegate.solve(problem, options)
        if result.status in (
            SolveStatus.UNSAT_FULL_MODEL,
            SolveStatus.NO_WITNESS_IN_RESTRICTED_MODEL,
        ):
            return SolveResult(
                SolveStatus.UNKNOWN,
                result.scope,
                result.elapsed_seconds,
                self.name,
                None,
                {**dict(result.metadata), "exhausted": True},
                best_coloring=result.best_coloring,
                best_energy=result.best_energy,
            )
        # UBCSAT commonly exits 0 after exhausting runs with no status/model.
        if result.status is SolveStatus.ERROR:
            metadata = dict(result.metadata)
            if (
                metadata.get("return_code") == 0
                and metadata.get("model_parsing") == "unrecognized_solver_response"
            ):
                metadata["exhausted"] = True
                return SolveResult(
                    SolveStatus.UNKNOWN,
                    result.scope,
                    result.elapsed_seconds,
                    self.name,
                    None,
                    metadata,
                )
        return SolveResult(
            result.status,
            result.scope,
            result.elapsed_seconds,
            self.name,
            result.coloring,
            result.metadata,
            best_coloring=result.best_coloring,
            best_energy=result.best_energy,
        )
