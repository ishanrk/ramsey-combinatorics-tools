"""Complete in-process SAT solving through the installed PySAT build."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import multiprocessing
import queue
import threading
import time
from collections.abc import Sequence

from pysat.solvers import NoSuchSolverError, Solver, SolverNames

from pvdw.backends.base import (
    BackendCapabilities,
    EncodedProblem,
    SolveOptions,
    base_metadata,
    negative_status,
)
from pvdw.model import SolveResult, SolveStatus


PREFERRED_SOLVERS = (
    "cadical195",
    "cadical153",
    "glucose4",
    "glucose3",
    "minisat22",
)

_KNOWN_SOLVERS = PREFERRED_SOLVERS + (
    "cadical103",
    "glucose42",
    "gluecard4",
    "gluecard3",
    "lingeling",
    "maplechrono",
    "maplecm",
    "maplesat",
    "mergesat3",
    "minicard",
    "minisatgh",
    "kissat404",
    "cadical300",
    "cryptosat",
    "minisatep",
)


def available_pysat_solvers() -> list[str]:
    """Probe solver constructors instead of trusting optional build names."""

    available: list[str] = []
    documented = tuple(
        attribute
        for attribute in dir(SolverNames)
        if not attribute.startswith("_")
        and isinstance((aliases := getattr(SolverNames, attribute)), tuple)
        and aliases
    )
    for name in dict.fromkeys(_KNOWN_SOLVERS + documented):
        if name == "cryptosat" and importlib.util.find_spec("pycryptosat") is None:
            continue
        try:
            solver = Solver(name=name)
        except (NoSuchSolverError, OSError, RuntimeError, AssertionError):
            continue
        else:
            solver.delete()
            available.append(name)
    return available


def _supports_limited_interrupt(solver_name: str) -> bool:
    # PySAT documents solve_limited/interrupt as a MiniSat-like feature and
    # explicitly excludes CaDiCaL and Lingeling.  Constructor probing is safe;
    # calling unsupported limited APIs merely to probe them is not (some
    # optional wrappers can terminate the interpreter).
    return solver_name not in {
        "cadical103",
        "cadical153",
        "cadical195",
        "cadical300",
        "lingeling",
        "kissat404",
    }


def _fixed_formula_worker(
    solver_name: str,
    clauses: tuple[tuple[int, ...], ...],
    result_queue: multiprocessing.Queue[object],
) -> None:
    try:
        with Solver(name=solver_name, bootstrap_with=clauses) as solver:
            solved = solver.solve()
            result_queue.put((solved, solver.get_model() if solved else None, None))
    except Exception as error:
        result_queue.put((None, None, repr(error)))


def _subprocess_solve(
    solver_name: str,
    clauses: tuple[tuple[int, ...], ...],
    timeout_seconds: float,
) -> tuple[bool | None, list[int] | None, str | None, bool]:
    methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in methods else "spawn")
    result_queue: multiprocessing.Queue[object] = context.Queue()
    process = context.Process(
        target=_fixed_formula_worker,
        args=(solver_name, clauses, result_queue),
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(1.0)
        if process.is_alive():
            process.kill()
            process.join()
        return None, None, None, True
    try:
        solved, model, error = result_queue.get_nowait()
    except queue.Empty:
        return None, None, "solver subprocess returned no result", False
    return solved, model, error, False  # type: ignore[return-value]


class PySatBackend:
    """A configurable complete CDCL backend with reusable incremental state."""

    def __init__(self, solver_name: str | None = None) -> None:
        available = available_pysat_solvers()
        if solver_name is None:
            solver_name = next(
                (candidate for candidate in PREFERRED_SOLVERS if candidate in available),
                None,
            )
        if solver_name is None:
            raise ValueError(f"no PySAT solver is available; probed: {available}")
        if solver_name not in available:
            try:
                probe = Solver(name=solver_name)
            except (NoSuchSolverError, OSError, RuntimeError, AssertionError) as error:
                raise ValueError(
                    f"PySAT solver {solver_name!r} is unavailable; available: {available}"
                ) from error
            else:
                probe.delete()
        self.solver_name = solver_name
        self.name = solver_name
        incremental = not solver_name.startswith("kissat")
        self.capabilities = BackendCapabilities(
            complete=True,
            incremental=incremental,
            accepts_dimacs=False,
            supports_assumptions=incremental,
            stochastic=False,
        )
        self.supports_interrupt_timeout = (
            incremental and _supports_limited_interrupt(solver_name)
        )
        self._solver: Solver | None = None
        self._clauses: list[tuple[int, ...]] = []
        self._last_model: list[int] | None = None
        self._last_timed_out = False

    @property
    def version(self) -> str:
        try:
            package_version = importlib.metadata.version("python-sat")
        except importlib.metadata.PackageNotFoundError:
            package_version = "unknown"
        return f"python-sat {package_version} ({self.solver_name})"

    def _ensure_solver(self) -> Solver:
        if self._solver is None:
            self._solver = Solver(name=self.solver_name, bootstrap_with=self._clauses)
        return self._solver

    def add_clause(self, clause: Sequence[int]) -> None:
        if not self.capabilities.incremental:
            raise RuntimeError(f"{self.solver_name} is fixed-formula only")
        materialized = tuple(clause)
        if any(type(literal) is not int or literal == 0 for literal in materialized):
            raise ValueError("clauses require nonzero ordinary integer literals")
        self._clauses.append(materialized)
        self._ensure_solver().add_clause(materialized)

    def solve_current(
        self,
        options: SolveOptions,
        assumptions: Sequence[int] = (),
    ) -> bool | None:
        """Solve current clauses; ``None`` means interruption/unknown."""

        solver = self._ensure_solver()
        self._last_model = None
        self._last_timed_out = False
        if options.timeout_seconds is None:
            solved = solver.solve(assumptions=list(assumptions))
            self._last_model = solver.get_model() if solved else None
            return solved
        if not self.capabilities.incremental:
            solved, model, error, timed_out = _subprocess_solve(
                self.solver_name, tuple(self._clauses), options.timeout_seconds
            )
            if error:
                raise RuntimeError(error)
            self._last_timed_out = timed_out
            self._last_model = model
            return solved
        if not self.supports_interrupt_timeout or not all(
            hasattr(solver, attribute)
            for attribute in ("interrupt", "clear_interrupt", "solve_limited")
        ):
            raise RuntimeError(
                f"{self.solver_name} cannot safely enforce in-process timeouts"
            )
        deadline_fired = threading.Event()

        def interrupt() -> None:
            deadline_fired.set()
            solver.interrupt()

        timer = threading.Timer(options.timeout_seconds, interrupt)
        timer.daemon = True
        timer.start()
        try:
            solved = solver.solve_limited(
                assumptions=list(assumptions), expect_interrupt=True
            )
        finally:
            timer.cancel()
            solver.clear_interrupt()
        self._last_timed_out = deadline_fired.is_set() and solved is None
        self._last_model = solver.get_model() if solved else None
        return solved

    def get_model(self) -> list[int] | None:
        return list(self._last_model) if self._last_model is not None else None

    def close(self) -> None:
        if self._solver is not None:
            self._solver.delete()
            self._solver = None

    def solve_cnf(
        self,
        problem: EncodedProblem,
        options: SolveOptions,
    ) -> SolveResult:
        return self.solve(problem, options)

    def solve(self, problem: EncodedProblem, options: SolveOptions) -> SolveResult:
        started = time.perf_counter()
        metadata = base_metadata(problem, options, backend_version=self.version)
        self.close()
        self._clauses = list(problem.clauses)
        try:
            if options.timeout_seconds is not None and (
                not self.capabilities.incremental or not self.supports_interrupt_timeout
            ):
                solved, model, error, timed_out = _subprocess_solve(
                    self.solver_name, problem.clauses, options.timeout_seconds
                )
                if error is not None:
                    raise RuntimeError(error)
                self._last_timed_out = timed_out
                self._last_model = model
            else:
                solved = self.solve_current(options)
            elapsed = time.perf_counter() - started
            if self._last_timed_out:
                metadata["model_parsing"] = "not_attempted_timeout"
                return SolveResult(
                    SolveStatus.TIMEOUT,
                    problem.scope,
                    elapsed,
                    self.name,
                    None,
                    metadata,
                )
            if solved is False:
                metadata["model_parsing"] = "not_applicable_unsat"
                metadata["verification"] = "not_applicable_unsat"
                return SolveResult(
                    negative_status(problem.scope),
                    problem.scope,
                    elapsed,
                    self.name,
                    None,
                    metadata,
                )
            if solved is not True:
                return SolveResult(
                    SolveStatus.UNKNOWN,
                    problem.scope,
                    elapsed,
                    self.name,
                    None,
                    metadata,
                )
            model = self.get_model()
            if model is None:
                metadata["model_parsing"] = "missing_model"
                return SolveResult(
                    SolveStatus.ERROR,
                    problem.scope,
                    elapsed,
                    self.name,
                    None,
                    metadata,
                )
            try:
                coloring = problem.decode_model(model)
            except ValueError as error:
                metadata["model_parsing"] = f"error: {error}"
                return SolveResult(
                    SolveStatus.ERROR,
                    problem.scope,
                    elapsed,
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
                    elapsed,
                    self.name,
                    None,
                    metadata,
                )
            return SolveResult(
                SolveStatus.FOUND_WITNESS,
                problem.scope,
                elapsed,
                self.name,
                coloring,
                metadata,
                best_coloring=coloring,
                best_energy=0,
            )
        except Exception as error:
            metadata["error"] = repr(error)
            return SolveResult(
                SolveStatus.ERROR,
                problem.scope,
                time.perf_counter() - started,
                self.name,
                None,
                metadata,
            )
        finally:
            self.close()
