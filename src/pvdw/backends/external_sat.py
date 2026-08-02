"""Complete external DIMACS solver execution with process-group timeouts."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from pvdw.backends.base import (
    BackendCapabilities,
    EncodedProblem,
    SolveOptions,
    base_metadata,
    negative_status,
)
from pvdw.encoding.dimacs import parse_sat_model, write_dimacs
from pvdw.model import SolveResult, SolveStatus


@dataclass(frozen=True)
class ExternalSolverConfig:
    name: str
    executable: Path
    command_template: tuple[str, ...]
    sat_exit_codes: frozenset[int] = frozenset({10})
    unsat_exit_codes: frozenset[int] = frozenset({20})

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("external solver name must be nonempty")
        object.__setattr__(self, "executable", Path(self.executable))
        object.__setattr__(self, "command_template", tuple(self.command_template))


def kissat_config(executable: str | os.PathLike[str] = "kissat") -> ExternalSolverConfig:
    return ExternalSolverConfig(
        "kissat",
        Path(executable),
        ("{executable}", "--seed={seed}", "{cnf}"),
    )


def cadical_config(executable: str | os.PathLike[str] = "cadical") -> ExternalSolverConfig:
    return ExternalSolverConfig(
        "cadical",
        Path(executable),
        ("{executable}", "--seed={seed}", "{cnf}"),
    )


def generic_dimacs_config(
    executable: str | os.PathLike[str],
    *,
    name: str = "external-dimacs",
    command_template: tuple[str, ...] = ("{executable}", "{cnf}"),
    sat_exit_codes: frozenset[int] = frozenset({10}),
    unsat_exit_codes: frozenset[int] = frozenset({20}),
) -> ExternalSolverConfig:
    return ExternalSolverConfig(
        name,
        Path(executable),
        command_template,
        sat_exit_codes,
        unsat_exit_codes,
    )


def _status_from_output(text: str) -> str | None:
    status: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip().upper()
        if not line.startswith("S "):
            continue
        current = "unsat" if "UNSATISFIABLE" in line else (
            "sat" if "SATISFIABLE" in line else None
        )
        if current is not None and status is not None and current != status:
            raise ValueError("solver output contains contradictory statuses")
        if current is not None:
            status = current
    return status


class ExternalSatBackend:
    """Run a complete command-line SAT solver against a temporary DIMACS file."""

    capabilities = BackendCapabilities(
        complete=True,
        incremental=False,
        accepts_dimacs=True,
        supports_assumptions=False,
        stochastic=False,
    )

    def __init__(self, config: ExternalSolverConfig) -> None:
        self.config = config
        self.name = config.name

    def _resolved_executable(self) -> Path:
        executable = self.config.executable
        if executable.is_absolute() or executable.parent != Path("."):
            resolved = executable
        else:
            located = shutil.which(str(executable))
            if located is None:
                raise FileNotFoundError(f"external solver not found: {executable}")
            resolved = Path(located)
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise FileNotFoundError(f"external solver is not executable: {resolved}")
        return resolved

    def version(self) -> str | None:
        try:
            executable = self._resolved_executable()
            completed = subprocess.run(
                [str(executable), "--version"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        text = (completed.stdout or completed.stderr).strip()
        return text.splitlines()[0][:300] if text else None

    def _command(
        self,
        executable: Path,
        cnf: Path,
        options: SolveOptions,
    ) -> list[str]:
        replacements = {
            "executable": str(executable),
            "cnf": str(cnf),
            "seed": str(options.seed),
            "timeout": "" if options.timeout_seconds is None else str(options.timeout_seconds),
        }
        command = [part.format(**replacements) for part in self.config.command_template]
        return [part for part in command if part] + list(options.extra_args)

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen[str]) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            process.wait()

    def solve(self, problem: EncodedProblem, options: SolveOptions) -> SolveResult:
        started = time.perf_counter()
        metadata = base_metadata(problem, options, backend_version=self.version())
        try:
            executable = self._resolved_executable()
        except OSError as error:
            metadata["error"] = str(error)
            return SolveResult(
                SolveStatus.ERROR,
                problem.scope,
                time.perf_counter() - started,
                self.name,
                None,
                metadata,
            )
        with tempfile.TemporaryDirectory(prefix="pvdw-sat-") as directory:
            cnf = Path(directory) / "instance.cnf"
            write_dimacs(
                cnf,
                problem.variable_count,
                problem.clauses,
                clause_count=problem.clause_count,
            )
            command = self._command(executable, cnf, options)
            metadata["command"] = command
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                try:
                    stdout, stderr = process.communicate(timeout=options.timeout_seconds)
                except subprocess.TimeoutExpired:
                    self._terminate_process_group(process)
                    stdout, stderr = process.communicate()
                    metadata.update(
                        {
                            "return_code": process.returncode,
                            "stdout_tail": stdout[-8192:],
                            "stderr_tail": stderr[-8192:],
                            "model_parsing": "not_attempted_timeout",
                        }
                    )
                    return SolveResult(
                        SolveStatus.TIMEOUT,
                        problem.scope,
                        time.perf_counter() - started,
                        self.name,
                        None,
                        metadata,
                    )
            except OSError as error:
                metadata["error"] = repr(error)
                return SolveResult(
                    SolveStatus.ERROR,
                    problem.scope,
                    time.perf_counter() - started,
                    self.name,
                    None,
                    metadata,
                )
            metadata.update(
                {
                    "return_code": process.returncode,
                    "stdout_tail": stdout[-8192:],
                    "stderr_tail": stderr[-8192:],
                }
            )
            try:
                output_status = _status_from_output(stdout)
            except ValueError as error:
                metadata["model_parsing"] = f"error: {error}"
                return SolveResult(
                    SolveStatus.ERROR,
                    problem.scope,
                    time.perf_counter() - started,
                    self.name,
                    None,
                    metadata,
                )
            exit_status = (
                "sat"
                if process.returncode in self.config.sat_exit_codes
                else "unsat"
                if process.returncode in self.config.unsat_exit_codes
                else None
            )
            if output_status and exit_status and output_status != exit_status:
                metadata["model_parsing"] = "status_exit_code_mismatch"
                return SolveResult(
                    SolveStatus.ERROR,
                    problem.scope,
                    time.perf_counter() - started,
                    self.name,
                    None,
                    metadata,
                )
            status = output_status or exit_status
            if status == "unsat":
                metadata["model_parsing"] = "not_applicable_unsat"
                metadata["verification"] = "not_applicable_unsat"
                return SolveResult(
                    negative_status(problem.scope),
                    problem.scope,
                    time.perf_counter() - started,
                    self.name,
                    None,
                    metadata,
                )
            if status != "sat":
                metadata["model_parsing"] = "unrecognized_solver_response"
                return SolveResult(
                    SolveStatus.ERROR,
                    problem.scope,
                    time.perf_counter() - started,
                    self.name,
                    None,
                    metadata,
                )
            try:
                model = parse_sat_model(stdout)
                if model is None:
                    raise ValueError("SAT response parsed as UNSAT")
                coloring = problem.decode_model(model)
            except ValueError as error:
                metadata["model_parsing"] = f"error: {error}"
                return SolveResult(
                    SolveStatus.ERROR,
                    problem.scope,
                    time.perf_counter() - started,
                    self.name,
                    None,
                    metadata,
                )
            metadata["model_parsing"] = "complete"
            verification = problem.verify(coloring)
            metadata["verification"] = "valid" if verification.valid else "invalid"
            if not verification.valid:
                metadata["verification_error"] = verification.error or repr(
                    verification.violation
                )
                return SolveResult(
                    SolveStatus.ERROR,
                    problem.scope,
                    time.perf_counter() - started,
                    self.name,
                    None,
                    metadata,
                )
            return SolveResult(
                SolveStatus.FOUND_WITNESS,
                problem.scope,
                time.perf_counter() - started,
                self.name,
                coloring,
                metadata,
                best_coloring=coloring,
                best_energy=0,
            )
