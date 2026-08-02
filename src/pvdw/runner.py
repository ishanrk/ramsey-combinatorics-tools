"""Backend discovery and high-level mode orchestration."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from pvdw.backends.base import SearchBackend, SolveOptions
from pvdw.backends.bruteforce import BruteforceBackend
from pvdw.backends.external_sat import (
    ExternalSatBackend,
    cadical_config,
    kissat_config,
)
from pvdw.backends.portfolio import PortfolioBackend
from pvdw.backends.potts import PottsBackend, PottsOptions
from pvdw.backends.pysat_backend import PySatBackend, available_pysat_solvers
from pvdw.backends.ubcsat import UbcsatBackend, UbcsatOptions
from pvdw.model import InstanceSpec, SolveResult
from pvdw.modes.direct import EncodingOptions, solve_direct
from pvdw.modes.periodic import build_periodic_problem, solve_periodic
from pvdw.modes.repair import solve_repair
from pvdw.modes.twisted import build_twisted_problem, solve_twisted


def create_backend(
    name: str,
    *,
    timeout_seconds: float | None = None,
    seed: int = 0,
    algorithm: str = "walksat",
    runs: int = 20,
    cutoff: int = 1_000_000,
    extra_args: tuple[str, ...] = (),
    potts_restarts: int = 8,
    potts_steps: int | None = 100_000,
    parallel_portfolio: bool = False,
) -> SearchBackend:
    """Resolve a CLI backend name without hard-coded executable paths."""

    available = available_pysat_solvers()
    if name in available:
        return PySatBackend(name)
    if name == "bruteforce":
        return BruteforceBackend()
    if name == "potts":
        return PottsBackend(
            PottsOptions(
                restarts=potts_restarts,
                max_steps=potts_steps,
                timeout_seconds=timeout_seconds or 10.0,
                seed=seed,
            )
        )
    if name == "ubcsat":
        executable = os.environ.get("UBCSAT_BIN", "ubcsat")
        return UbcsatBackend(
            executable,
            UbcsatOptions(algorithm, runs, cutoff, extra_args),
        )
    if name == "kissat":
        if "kissat404" in available:
            return PySatBackend("kissat404")
        executable = os.environ.get("KISSAT_BIN") or shutil.which("kissat")
        if executable is None:
            raise ValueError("Kissat is unavailable (set KISSAT_BIN or install it)")
        return ExternalSatBackend(kissat_config(executable))
    if name == "cadical":
        executable = os.environ.get("CADICAL_BIN") or shutil.which("cadical")
        if executable is None:
            if "cadical195" in available:
                return PySatBackend("cadical195")
            raise ValueError("CaDiCaL is unavailable (set CADICAL_BIN or install it)")
        return ExternalSatBackend(cadical_config(executable))
    if name == "portfolio":
        members: list[SearchBackend] = [
            PottsBackend(
                PottsOptions(
                    restarts=potts_restarts,
                    max_steps=potts_steps,
                    timeout_seconds=timeout_seconds or 10.0,
                    seed=seed,
                )
            )
        ]
        ubcsat_executable = os.environ.get("UBCSAT_BIN") or shutil.which("ubcsat")
        if ubcsat_executable:
            members.append(
                UbcsatBackend(
                    ubcsat_executable,
                    UbcsatOptions(algorithm, runs, cutoff, extra_args),
                )
            )
        preferred = next(
            (
                candidate
                for candidate in (
                    "cadical195",
                    "cadical153",
                    "glucose4",
                    "glucose3",
                    "minisat22",
                )
                if candidate in available
            ),
            None,
        )
        if preferred:
            members.append(PySatBackend(preferred))
        return PortfolioBackend(members, parallel=parallel_portfolio)
    if name.startswith("external:"):
        executable = Path(name.partition(":")[2])
        from pvdw.backends.external_sat import generic_dimacs_config

        return ExternalSatBackend(generic_dimacs_config(executable))
    try:
        return PySatBackend(name)
    except ValueError as error:
        raise ValueError(f"unknown backend {name!r}") from error


def backend_inventory() -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for name in available_pysat_solvers():
        backend = PySatBackend(name)
        inventory.append(
            {
                "name": name,
                "available": True,
                "version": backend.version,
                "complete": True,
                "incremental": backend.capabilities.incremental,
                "timeout_in_process": backend.supports_interrupt_timeout,
                "stochastic": False,
            }
        )
    inventory.append(
        {
            "name": "potts",
            "available": True,
            "version": "native",
            "complete": False,
            "incremental": False,
            "stochastic": True,
        }
    )
    for name, environment, executable_name in (
        ("kissat-external", "KISSAT_BIN", "kissat"),
        ("cadical-external", "CADICAL_BIN", "cadical"),
        ("ubcsat", "UBCSAT_BIN", "ubcsat"),
    ):
        path = os.environ.get(environment) or shutil.which(executable_name)
        inventory.append(
            {
                "name": name,
                "available": bool(path),
                "path": path,
                "complete": name != "ubcsat",
                "incremental": False,
                "stochastic": name == "ubcsat",
            }
        )
    return inventory


def solve_mode(
    instance: InstanceSpec,
    mode: str,
    backend: SearchBackend,
    encoding_options: EncodingOptions,
    solve_options: SolveOptions,
    *,
    period: int | None = None,
    twist: int = 0,
) -> SolveResult:
    if mode == "direct":
        return solve_direct(instance, encoding_options, backend, solve_options)
    if mode == "periodic":
        if period is None:
            raise ValueError("periodic mode requires --period")
        return solve_periodic(
            instance, period, encoding_options, backend, solve_options
        )
    if mode == "twisted":
        if period is None:
            raise ValueError("twisted mode requires --period")
        return solve_twisted(
            instance, period, twist, encoding_options, backend, solve_options
        )
    raise ValueError(f"unknown solve mode {mode!r}")


def repair_from_backbone_search(
    instance: InstanceSpec,
    backbone_source: str,
    repair_backend: SearchBackend,
    encoding_options: EncodingOptions,
    solve_options: SolveOptions,
    *,
    period: int | None = None,
    twist: int = 0,
    editable_strategy: str = "greedy_vertex_cover",
    max_expansions: int = 3,
    potts_restarts: int = 8,
    potts_steps: int | None = 100_000,
) -> SolveResult:
    """Keep the best positive-energy backbone and pass it into reduced repair."""

    potts = PottsBackend(
        PottsOptions(
            restarts=potts_restarts,
            max_steps=potts_steps,
            timeout_seconds=solve_options.timeout_seconds or 10.0,
            seed=solve_options.seed,
        )
    )
    if backbone_source == "direct":
        backbone_result = solve_direct(
            instance, encoding_options, potts, solve_options
        )
    elif backbone_source == "periodic":
        if period is None:
            raise ValueError("periodic repair backbone requires --period")
        model, problem = build_periodic_problem(instance, period, encoding_options)
        if problem is None:
            raise ValueError(f"period {period} is immediately impossible")
        backbone_result = potts.solve(problem, solve_options)
    elif backbone_source == "twisted":
        if period is None:
            raise ValueError("twisted repair backbone requires --period")
        model, problem = build_twisted_problem(
            instance, period, twist, encoding_options
        )
        if problem is None:
            raise ValueError(
                f"period {period}, twist {twist} is immediately impossible"
            )
        backbone_result = potts.solve(problem, solve_options)
    else:
        raise ValueError(f"unknown backbone source {backbone_source!r}")
    backbone = backbone_result.best_coloring or backbone_result.coloring
    if backbone is None:
        raise RuntimeError("backbone search did not retain a candidate coloring")
    return solve_repair(
        instance,
        backbone,
        repair_backend,
        encoding_options,
        solve_options,
        editable_strategy=editable_strategy,
        max_expansions=max_expansions,
        backbone_metadata={
            "backbone_source": backbone_source,
            "backbone_period": period,
            "backbone_twist": twist if backbone_source == "twisted" else None,
            "backbone_search_status": backbone_result.status.value,
            "backbone_search_best_energy": backbone_result.best_energy,
        },
    )
