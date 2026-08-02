"""Reproducible project regression benchmark suites and artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from pvdw.backends.base import SolveOptions
from pvdw.model import InstanceSpec, PolynomialSpec, SolveStatus
from pvdw.modes.direct import EncodingOptions
from pvdw.runner import create_backend, solve_mode
from pvdw.witness import create_witness, write_witness


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    coefficients: tuple[int, ...]
    colors: int
    n: int
    mode: str = "direct"
    period: int | None = None
    twist: int = 0


PROJECT_REGRESSIONS = (
    BenchmarkCase("square-c3-n28", (0, 0, 1), 3, 28),
    BenchmarkCase("square-c4-n57", (0, 0, 1), 4, 57),
    BenchmarkCase("square-c5-n180", (0, 0, 1), 5, 180),
    BenchmarkCase("square-c6-n400-p40", (0, 0, 1), 6, 400, "periodic", 40),
    BenchmarkCase("square-c8-n841-p29", (0, 0, 1), 8, 841, "periodic", 29),
    BenchmarkCase("cube-c3-n521", (0, 0, 0, 1), 3, 521),
    BenchmarkCase("cube-c4-n2197-p13", (0, 0, 0, 1), 4, 2197, "periodic", 13),
    BenchmarkCase("cube-c5-n9261-p63", (0, 0, 0, 1), 5, 9261, "periodic", 63),
)


def benchmark_suite(name: str) -> tuple[BenchmarkCase, ...]:
    if name != "project-regressions":
        raise ValueError(f"unknown benchmark suite {name!r}")
    return PROJECT_REGRESSIONS


def run_benchmark_suite(
    suite: str,
    output_dir: str | Path,
    *,
    timeout_seconds: float = 30.0,
    backend_name: str = "portfolio",
    seed: int = 0,
) -> tuple[dict[str, object], ...]:
    """Run a suite and write JSONL, CSV, and verified witnesses."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    witness_directory = output / "witnesses"
    witness_directory.mkdir(exist_ok=True)
    records: list[dict[str, object]] = []
    for index, case in enumerate(benchmark_suite(suite)):
        instance = InstanceSpec(
            PolynomialSpec(case.coefficients), case.colors, case.n
        )
        options = SolveOptions(timeout_seconds, seed + index)
        backend = create_backend(
            backend_name,
            timeout_seconds=timeout_seconds,
            seed=seed + index,
        )
        result = solve_mode(
            instance,
            case.mode,
            backend,
            EncodingOptions(),
            options,
            period=case.period,
            twist=case.twist,
        )
        witness_path: str | None = None
        if result.status is SolveStatus.FOUND_WITNESS:
            assert result.coloring is not None
            witness = create_witness(
                instance,
                result.coloring,
                backend=result.backend,
                seed=seed + index,
                elapsed_seconds=result.elapsed_seconds,
                scope=result.scope,
                scope_metadata={
                    key: result.metadata[key]
                    for key in ("period", "twist", "finite_interval")
                    if key in result.metadata
                },
            )
            destination = witness_directory / f"{case.name}.json"
            write_witness(destination, witness)
            witness_path = str(destination)
        records.append(
            {
                "suite": suite,
                "case": case.name,
                "coefficients": list(case.coefficients),
                "colors": case.colors,
                "n": case.n,
                "mode": case.mode,
                "period": case.period,
                "twist": case.twist,
                "backend": result.backend,
                "status": result.status.value,
                "scope": result.scope.value,
                "elapsed_seconds": result.elapsed_seconds,
                "variables": result.metadata.get("formula_variables"),
                "clauses": result.metadata.get("formula_clauses"),
                "best_energy": result.best_energy,
                "verified_witness": witness_path,
            }
        )
    jsonl = output / "results.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    csv_path = output / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return tuple(records)
