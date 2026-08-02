"""Deterministically seeded portfolios with proof-status safeguards."""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import replace

from pvdw.backends.base import (
    BackendCapabilities,
    EncodedProblem,
    SearchBackend,
    SolveOptions,
)
from pvdw.model import SolveResult, SolveStatus


def _portfolio_worker(
    backend: SearchBackend,
    problem: EncodedProblem,
    options: SolveOptions,
) -> SolveResult:
    return backend.solve(problem, options)


class PortfolioBackend:
    """Try stochastic searches first and reserve negative proofs for complete SAT."""

    name = "portfolio"

    def __init__(
        self,
        backends: tuple[SearchBackend, ...] | list[SearchBackend],
        *,
        parallel: bool = False,
        max_workers: int | None = None,
    ) -> None:
        self.backends = tuple(backends)
        if not self.backends:
            raise ValueError("portfolio requires at least one backend")
        self.parallel = parallel
        self.max_workers = max_workers
        self.capabilities = BackendCapabilities(
            complete=any(backend.capabilities.complete for backend in self.backends),
            incremental=False,
            accepts_dimacs=any(
                backend.capabilities.accepts_dimacs for backend in self.backends
            ),
            supports_assumptions=False,
            stochastic=any(backend.capabilities.stochastic for backend in self.backends),
        )

    @staticmethod
    def _derived_options(
        options: SolveOptions,
        index: int,
        backend: SearchBackend,
    ) -> SolveOptions:
        timeout = options.timeout_seconds
        if backend.capabilities.stochastic and timeout is not None:
            # Keep a proof-capable solver in the budget for restricted scans.
            timeout = min(timeout, 2.0)
        return replace(
            options,
            seed=options.seed + 1_000_003 * index,
            timeout_seconds=timeout,
        )

    @staticmethod
    def _safe_result(backend: SearchBackend, result: SolveResult) -> SolveResult:
        if not backend.capabilities.complete and result.status in (
            SolveStatus.UNSAT_FULL_MODEL,
            SolveStatus.NO_WITNESS_IN_RESTRICTED_MODEL,
        ):
            return SolveResult(
                SolveStatus.UNKNOWN,
                result.scope,
                result.elapsed_seconds,
                result.backend,
                None,
                {**dict(result.metadata), "discarded_incomplete_negative": True},
                best_coloring=result.best_coloring,
                best_energy=result.best_energy,
            )
        return result

    def _wrap(
        self,
        result: SolveResult,
        attempts: list[dict[str, object]],
        elapsed: float,
    ) -> SolveResult:
        metadata = {
            **dict(result.metadata),
            "portfolio_winner": result.backend,
            "portfolio_attempts": attempts,
        }
        return SolveResult(
            result.status,
            result.scope,
            elapsed,
            self.name,
            result.coloring,
            metadata,
            best_coloring=result.best_coloring,
            best_energy=result.best_energy,
        )

    def solve(self, problem: EncodedProblem, options: SolveOptions) -> SolveResult:
        started = time.perf_counter()
        stochastic = [
            backend for backend in self.backends if backend.capabilities.stochastic
        ]
        complete = [
            backend for backend in self.backends if backend.capabilities.complete
        ]
        other = [
            backend
            for backend in self.backends
            if backend not in stochastic and backend not in complete
        ]
        attempts: list[dict[str, object]] = []
        results: list[SolveResult] = []
        if self.parallel and len(stochastic) > 1:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers or len(stochastic)
            ) as executor:
                futures = {
                    executor.submit(
                        _portfolio_worker,
                        backend,
                        problem,
                        self._derived_options(options, index, backend),
                    ): backend
                    for index, backend in enumerate(stochastic)
                }
                for future in concurrent.futures.as_completed(futures):
                    backend = futures[future]
                    try:
                        result = self._safe_result(backend, future.result())
                    except Exception as error:
                        attempts.append(
                            {
                                "backend": backend.name,
                                "status": SolveStatus.ERROR.value,
                                "error": repr(error),
                            }
                        )
                        continue
                    attempts.append(
                        {
                            "backend": backend.name,
                            "status": result.status.value,
                            "elapsed_seconds": result.elapsed_seconds,
                            "seed": result.metadata.get("seed"),
                            "best_energy": result.best_energy,
                        }
                    )
                    results.append(result)
                    if result.status is SolveStatus.FOUND_WITNESS:
                        for pending in futures:
                            pending.cancel()
                        return self._wrap(
                            result, attempts, time.perf_counter() - started
                        )
        else:
            for index, backend in enumerate(stochastic):
                result = self._safe_result(
                    backend,
                    backend.solve(
                        problem, self._derived_options(options, index, backend)
                    ),
                )
                attempts.append(
                    {
                        "backend": backend.name,
                        "status": result.status.value,
                        "elapsed_seconds": result.elapsed_seconds,
                        "seed": result.metadata.get("seed"),
                        "best_energy": result.best_energy,
                    }
                )
                results.append(result)
                if result.status is SolveStatus.FOUND_WITNESS:
                    return self._wrap(result, attempts, time.perf_counter() - started)
        offset = len(stochastic)
        for index, backend in enumerate(other + complete, start=offset):
            result = self._safe_result(
                backend,
                backend.solve(problem, self._derived_options(options, index, backend)),
            )
            attempts.append(
                {
                    "backend": backend.name,
                    "status": result.status.value,
                    "elapsed_seconds": result.elapsed_seconds,
                    "seed": result.metadata.get("seed"),
                    "best_energy": result.best_energy,
                }
            )
            results.append(result)
            if result.status in (
                SolveStatus.FOUND_WITNESS,
                SolveStatus.UNSAT_FULL_MODEL,
                SolveStatus.NO_WITNESS_IN_RESTRICTED_MODEL,
            ):
                return self._wrap(result, attempts, time.perf_counter() - started)
        best = min(
            (result for result in results if result.best_energy is not None),
            key=lambda result: result.best_energy,  # type: ignore[arg-type]
            default=results[-1] if results else None,
        )
        if best is None:
            raise RuntimeError("portfolio produced no result")
        status = (
            SolveStatus.TIMEOUT
            if results and all(result.status is SolveStatus.TIMEOUT for result in results)
            else SolveStatus.UNKNOWN
        )
        fallback = SolveResult(
            status,
            problem.scope,
            time.perf_counter() - started,
            best.backend,
            None,
            best.metadata,
            best_coloring=best.best_coloring,
            best_energy=best.best_energy,
        )
        return self._wrap(fallback, attempts, time.perf_counter() - started)
