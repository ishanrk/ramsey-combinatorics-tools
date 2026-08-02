"""Command-line interface for finite polynomial difference-coloring searches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from pvdw.backends.base import SolveOptions
from pvdw.backends.bruteforce import BruteforceBackend
from pvdw.distances import generate_distances
from pvdw.encoding.binary import (
    binary_statistics,
    encode_binary,
    iter_binary_clauses,
)
from pvdw.encoding.common import ListClauseSink
from pvdw.encoding.dimacs import write_dimacs
from pvdw.encoding.onehot import (
    AtMostOneEncoding,
    encode_onehot,
    iter_onehot_pairwise_clauses,
    onehot_pairwise_statistics,
)
from pvdw.graph import DistanceGraph, estimate_adjacency_bytes
from pvdw.model import InstanceSpec, ModelScope, SolveResult, SolveStatus
from pvdw.modes.direct import EncodingOptions
from pvdw.modes.incremental import scan_incremental
from pvdw.modes.periodic import scan_periods
from pvdw.modes.twisted import scan_twists
from pvdw.polynomial import PolynomialParseError, parse_coefficients, parse_polynomial
from pvdw.verify import verify_coloring
from pvdw.runner import (
    backend_inventory,
    create_backend,
    repair_from_backbone_search,
    solve_mode,
)
from pvdw.witness import (
    WitnessFormatError,
    coloring_to_word,
    create_witness,
    read_witness,
    write_witness,
)


EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_INVALID_INPUT = 2
EXIT_UNSAT = 10
EXIT_INCONCLUSIVE = 20


def _add_instance_arguments(parser: argparse.ArgumentParser) -> None:
    polynomial = parser.add_mutually_exclusive_group(required=True)
    polynomial.add_argument("--coefficients", help="low-degree-first integers, e.g. 0,0,1")
    polynomial.add_argument("--poly", help="restricted polynomial expression in x")
    parser.add_argument("--colors", type=int, required=True)
    parser.add_argument("--N", dest="n", type=int, required=True)
    parser.add_argument(
        "--input-domain",
        choices=("all_nonzero", "positive"),
        default="all_nonzero",
    )


def _add_polynomial_arguments(parser: argparse.ArgumentParser) -> None:
    polynomial = parser.add_mutually_exclusive_group(required=True)
    polynomial.add_argument("--coefficients", help="low-degree-first integers, e.g. 0,0,1")
    polynomial.add_argument("--poly", help="restricted polynomial expression in x")
    parser.add_argument("--colors", type=int, required=True)
    parser.add_argument(
        "--input-domain",
        choices=("all_nonzero", "positive"),
        default="all_nonzero",
    )


def _polynomial_from_args(args: argparse.Namespace):
    return (
        parse_coefficients(args.coefficients)
        if args.coefficients is not None
        else parse_polynomial(args.poly)
    )


def _instance_from_args(args: argparse.Namespace) -> InstanceSpec:
    polynomial = _polynomial_from_args(args)
    return InstanceSpec(polynomial, args.colors, args.n, args.input_domain)


def _encoding_options(args: argparse.Namespace) -> EncodingOptions:
    return EncodingOptions(
        encoding=getattr(args, "encoding", "onehot"),
        amo=AtMostOneEncoding(getattr(args, "amo", "pairwise")),
        fix_first_color=not getattr(args, "no_fix_first_color", False),
        anchor_clique=getattr(args, "anchor_clique", False),
    )


def _solve_options(args: argparse.Namespace) -> SolveOptions:
    return SolveOptions(
        timeout_seconds=getattr(args, "timeout", None),
        seed=getattr(args, "seed", 0),
        extra_args=tuple(getattr(args, "extra_arg", ()) or ()),
    )


def _backend_from_args(args: argparse.Namespace):
    if args.backend == "bruteforce":
        return BruteforceBackend(getattr(args, "size_limit", 30))
    return create_backend(
        args.backend,
        timeout_seconds=getattr(args, "timeout", None),
        seed=getattr(args, "seed", 0),
        algorithm=getattr(args, "algorithm", "walksat"),
        runs=getattr(args, "runs", 20),
        cutoff=getattr(args, "cutoff", 1_000_000),
        extra_args=tuple(getattr(args, "extra_arg", ()) or ()),
        potts_restarts=getattr(args, "potts_restarts", 8),
        potts_steps=getattr(args, "potts_steps", 100_000),
        parallel_portfolio=getattr(args, "parallel_portfolio", False),
    )


def _emit(payload: dict[str, Any], *, as_json: bool, lines: Sequence[str]) -> None:
    if as_json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print("\n".join(lines))


def _inspect(args: argparse.Namespace) -> int:
    instance = _instance_from_args(args)
    data = generate_distances(instance)
    graph = DistanceGraph(instance.n, data.values)
    onehot = onehot_pairwise_statistics(instance, graph)
    binary = binary_statistics(instance, graph)
    inspected = data.input_bound if instance.input_domain == "positive" else 2 * data.input_bound
    payload = {
        "coefficients": list(instance.polynomial.coefficients),
        "colors": instance.colors,
        "n": instance.n,
        "input_domain": instance.input_domain,
        "input_bound": data.input_bound,
        "distances": list(data.values),
        "polynomial_inputs_inspected": inspected,
        "edge_count": graph.edge_count,
        "estimated_adjacency_bytes": estimate_adjacency_bytes(graph),
        "onehot_pairwise": {
            "variables": onehot.variables,
            "clauses": onehot.clauses,
            "literals": onehot.literals,
        },
        "binary": {
            "variables": binary.variables,
            "clauses": binary.clauses,
            "literals": binary.literals,
        },
    }
    _emit(
        payload,
        as_json=args.json,
        lines=(
            f"normalized coefficients: {payload['coefficients']}",
            f"input bound: {data.input_bound}",
            f"distance set: {list(data.values)}",
            f"polynomial inputs inspected: {inspected}",
            f"edge count: {graph.edge_count}",
            f"estimated adjacency bytes: {payload['estimated_adjacency_bytes']}",
            f"one-hot pairwise: {onehot.variables} variables, {onehot.clauses} clauses, {onehot.literals} literals",
            f"binary: {binary.variables} variables, {binary.clauses} clauses, {binary.literals} literals",
        ),
    )
    return EXIT_SUCCESS


def _encode(args: argparse.Namespace) -> int:
    instance = _instance_from_args(args)
    data = generate_distances(instance)
    graph = DistanceGraph(instance.n, data.values)
    fix_first = not args.no_fix_first_color
    output = Path(args.output)
    if args.encoding == "binary":
        statistics = binary_statistics(instance, graph, fix_first_color=fix_first)
        write_dimacs(
            output,
            statistics.variables,
            iter_binary_clauses(instance, graph, fix_first_color=fix_first),
            clause_count=statistics.clauses,
        )
        anchored: list[int] = []
        immediate = False
    else:
        amo = AtMostOneEncoding(args.amo)
        if amo is AtMostOneEncoding.PAIRWISE and not args.anchor_clique:
            statistics = onehot_pairwise_statistics(
                instance, graph, fix_first_color=fix_first
            )
            write_dimacs(
                output,
                statistics.variables,
                iter_onehot_pairwise_clauses(
                    instance, graph, fix_first_color=fix_first
                ),
                clause_count=statistics.clauses,
            )
            anchored = []
            immediate = False
        else:
            sink = ListClauseSink()
            result = encode_onehot(
                instance,
                graph,
                sink,
                amo=amo,
                fix_first_color=fix_first,
                anchor_clique=args.anchor_clique,
            )
            statistics = result.statistics
            write_dimacs(
                output,
                statistics.variables,
                sink.clauses,
                clause_count=statistics.clauses,
            )
            anchored = list(result.anchored_clique)
            immediate = result.immediate_noncolorability
    payload = {
        "output": str(output),
        "encoding": statistics.encoding,
        "variables": statistics.variables,
        "clauses": statistics.clauses,
        "literals": statistics.literals,
        "edge_count": graph.edge_count,
        "anchored_clique": anchored,
        "immediate_noncolorability": immediate,
    }
    _emit(
        payload,
        as_json=args.json,
        lines=(
            f"wrote: {output}",
            f"encoding: {statistics.encoding}",
            f"variables: {statistics.variables}",
            f"clauses: {statistics.clauses}",
            f"literals: {statistics.literals}",
            f"edge count: {graph.edge_count}",
        ),
    )
    return EXIT_UNSAT if immediate else EXIT_SUCCESS


def _result_payload(result: SolveResult) -> dict[str, Any]:
    return {
        "status": result.status.value,
        "scope": result.scope.value,
        "elapsed_seconds": result.elapsed_seconds,
        "backend": result.backend,
        "coloring": list(result.coloring) if result.coloring is not None else None,
        "best_coloring": (
            list(result.best_coloring) if result.best_coloring is not None else None
        ),
        "best_energy": result.best_energy,
        "metadata": dict(result.metadata),
    }


def _negative_scope_lines(result: SolveResult) -> list[str]:
    if result.status is SolveStatus.UNSAT_FULL_MODEL:
        return ["No coloring exists in the unrestricted full model."]
    if result.status is SolveStatus.NO_WITNESS_IN_RESTRICTED_MODEL:
        if result.scope is ModelScope.PERIODIC:
            period = result.metadata.get("period")
            first = (
                "No coloring exists in the finite-interval periodic model "
                f"with period {period}."
            )
        elif result.scope is ModelScope.TWISTED:
            first = (
                "No coloring exists in the finite-interval twisted model with "
                f"period {result.metadata.get('period')} and twist "
                f"{result.metadata.get('twist')}."
            )
        else:
            first = "No witness exists in the restricted repair model."
        return [first, "This does not imply that the unrestricted instance is uncolorable."]
    return []


def _result_exit_code(result: SolveResult) -> int:
    if result.status is SolveStatus.FOUND_WITNESS:
        return EXIT_SUCCESS
    if result.status is SolveStatus.UNSAT_FULL_MODEL:
        return EXIT_UNSAT
    if result.status is SolveStatus.NO_WITNESS_IN_RESTRICTED_MODEL:
        return EXIT_SUCCESS
    if result.status in (SolveStatus.TIMEOUT, SolveStatus.UNKNOWN):
        return EXIT_INCONCLUSIVE
    return EXIT_ERROR


def _present_result(
    instance: InstanceSpec,
    result: SolveResult,
    args: argparse.Namespace,
) -> int:
    payload = _result_payload(result)
    lines = [
        f"status: {result.status.value}",
        f"scope: {result.scope.value}",
        f"backend: {result.backend}",
        f"elapsed seconds: {result.elapsed_seconds:.6f}",
    ]
    lines.extend(_negative_scope_lines(result))
    if result.best_energy is not None:
        lines.append(f"best energy: {result.best_energy}")
    if result.coloring is not None:
        verification = verify_coloring(instance, result.coloring)
        if not verification.valid:
            raise RuntimeError("solver result failed independent verification")
        payload["verification"] = {"valid": True}
        if instance.colors <= 36:
            word = coloring_to_word(list(result.coloring), instance.colors)
            payload["word"] = word
            lines.append(f"word: {word}")
        if getattr(args, "output", None):
            witness = create_witness(
                instance,
                result.coloring,
                backend=result.backend,
                seed=getattr(args, "seed", 0),
                elapsed_seconds=result.elapsed_seconds,
                scope=result.scope,
                scope_metadata={
                    key: result.metadata[key]
                    for key in ("period", "twist", "finite_interval")
                    if key in result.metadata
                },
            )
            write_witness(args.output, witness)
            payload["witness"] = args.output
            lines.append(f"witness: {args.output}")
    _emit(payload, as_json=args.json, lines=lines)
    return _result_exit_code(result)


def _solve(args: argparse.Namespace) -> int:
    instance = _instance_from_args(args)
    backend = _backend_from_args(args)
    result = solve_mode(
        instance,
        args.mode,
        backend,
        _encoding_options(args),
        _solve_options(args),
        period=args.period,
        twist=args.twist,
    )
    return _present_result(instance, result, args)


def _scan(args: argparse.Namespace) -> int:
    backend = _backend_from_args(args)
    result = scan_incremental(
        _polynomial_from_args(args),
        args.colors,
        args.n_min,
        args.n_max,
        backend,
        _encoding_options(args),
        _solve_options(args),
        input_domain=args.input_domain,
    )
    payload: dict[str, Any] = {
        "backend": args.backend,
        "first_unsat_n": result.first_unsat_n,
        "last_verified_coloring": (
            list(result.last_verified_coloring)
            if result.last_verified_coloring is not None
            else None
        ),
        "cumulative_clauses": result.cumulative_clauses,
        "steps": [
            {
                "n": step.n,
                "status": step.status.value,
                "elapsed_seconds": step.elapsed_seconds,
                "added_clauses": step.added_clauses,
                "cumulative_clauses": step.cumulative_clauses,
            }
            for step in result.steps
        ],
    }
    lines = [
        f"backend: {args.backend}",
        f"tested N values: {len(result.steps)}",
        f"first full-model UNSAT N: {result.first_unsat_n}",
        f"cumulative clauses: {result.cumulative_clauses}",
    ]
    _emit(payload, as_json=args.json, lines=lines)
    if result.first_unsat_n is not None:
        return EXIT_UNSAT
    if result.steps and result.steps[-1].status in (
        SolveStatus.TIMEOUT,
        SolveStatus.UNKNOWN,
    ):
        return EXIT_INCONCLUSIVE
    return EXIT_SUCCESS


def _periods(args: argparse.Namespace) -> int:
    instance = _instance_from_args(args)
    result = scan_periods(
        instance,
        args.period_min,
        args.period_max,
        _encoding_options(args),
        _backend_from_args(args),
        _solve_options(args),
    )
    payload = {
        "scope": "periodic",
        "witness_found": result.witness is not None,
        "attempts": [
            {
                "period": attempt.period,
                "variables": attempt.variables,
                "constraints": attempt.constraints,
                "immediate_impossibility": attempt.immediate_impossibility,
                "status": attempt.status.value,
                "elapsed_seconds": attempt.elapsed_seconds,
                "best_energy": attempt.best_energy,
            }
            for attempt in result.attempts
        ],
    }
    lines = [
        f"periods attempted: {len(result.attempts)}",
        f"witness found: {str(result.witness is not None).lower()}",
    ]
    if result.witness is None:
        lines.append("No attempted periodic model produced a witness.")
        lines.append("This does not imply that the unrestricted instance is uncolorable.")
    _emit(payload, as_json=args.json, lines=lines)
    return EXIT_SUCCESS if result.witness is not None else EXIT_INCONCLUSIVE


def _twists(args: argparse.Namespace) -> int:
    instance = _instance_from_args(args)
    twists = (
        tuple(range(instance.colors))
        if args.twists == "all"
        else tuple(int(value) for value in args.twists.split(","))
    )
    result = scan_twists(
        instance,
        args.period_min,
        args.period_max,
        twists,
        _encoding_options(args),
        _backend_from_args(args),
        _solve_options(args),
    )
    payload = {
        "scope": "twisted",
        "witness_found": result.witness is not None,
        "attempts": [
            {
                "period": attempt.period,
                "twist": attempt.twist,
                "variables": attempt.variables,
                "constraints": attempt.constraints,
                "immediate_impossibility": attempt.immediate_impossibility,
                "status": attempt.status.value,
                "elapsed_seconds": attempt.elapsed_seconds,
                "best_energy": attempt.best_energy,
            }
            for attempt in result.attempts
        ],
    }
    lines = [
        f"period/twist pairs attempted: {len(result.attempts)}",
        f"witness found: {str(result.witness is not None).lower()}",
    ]
    if result.witness is None:
        lines.append("No attempted twisted model produced a witness.")
        lines.append("This does not imply that the unrestricted instance is uncolorable.")
    _emit(payload, as_json=args.json, lines=lines)
    return EXIT_SUCCESS if result.witness is not None else EXIT_INCONCLUSIVE


def _repair(args: argparse.Namespace) -> int:
    instance = _instance_from_args(args)
    result = repair_from_backbone_search(
        instance,
        args.backbone,
        _backend_from_args(args),
        _encoding_options(args),
        _solve_options(args),
        period=args.period,
        twist=args.twist,
        editable_strategy=args.editable_strategy,
        max_expansions=args.max_expansions,
        potts_restarts=args.potts_restarts,
        potts_steps=args.potts_steps,
    )
    return _present_result(instance, result, args)


def _backends(args: argparse.Namespace) -> int:
    inventory = backend_inventory()
    _emit(
        {"backends": inventory},
        as_json=args.json,
        lines=tuple(
            f"{item['name']}: {'available' if item['available'] else 'unavailable'}"
            + (f" ({item.get('version')})" if item.get("version") else "")
            for item in inventory
        ),
    )
    return EXIT_SUCCESS


def _benchmark(args: argparse.Namespace) -> int:
    from pvdw.benchmarks import run_benchmark_suite

    records = run_benchmark_suite(
        args.suite,
        args.output_dir,
        timeout_seconds=args.timeout,
        backend_name=args.backend,
        seed=args.seed,
    )
    payload = {"suite": args.suite, "output_dir": args.output_dir, "records": records}
    _emit(
        payload,
        as_json=args.json,
        lines=(
            f"suite: {args.suite}",
            f"runs: {len(records)}",
            f"output directory: {args.output_dir}",
            *(
                f"{record['case']}: {record['status']} ({record['elapsed_seconds']:.3f}s)"
                for record in records
            ),
        ),
    )
    return EXIT_SUCCESS


def _verify(args: argparse.Namespace) -> int:
    witness = read_witness(args.witness)
    result = verify_coloring(witness.instance, witness.coloring)
    if not result.valid:
        raise WitnessFormatError(result.error or repr(result.violation))
    payload = {
        "valid": True,
        "n": witness.instance.n,
        "colors": witness.instance.colors,
        "coefficients": list(witness.instance.polynomial.coefficients),
        "scope": witness.scope.value,
        "backend": witness.backend,
        "violation_count": result.violation_count,
    }
    _emit(
        payload,
        as_json=args.json,
        lines=(
            "valid: true",
            f"vertices: {witness.instance.n}",
            f"colors: {witness.instance.colors}",
            f"scope: {witness.scope.value}",
            f"backend: {witness.backend}",
        ),
    )
    return EXIT_SUCCESS


def _add_encoding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--encoding", choices=("onehot", "binary"), default="onehot")
    parser.add_argument(
        "--amo",
        choices=tuple(encoding.value for encoding in AtMostOneEncoding),
        default=AtMostOneEncoding.PAIRWISE.value,
    )
    parser.add_argument("--no-fix-first-color", action="store_true")
    parser.add_argument("--anchor-clique", action="store_true")


def _add_search_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", required=True)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--algorithm", default="walksat")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--cutoff", type=int, default=1_000_000)
    parser.add_argument("--potts-restarts", type=int, default=8)
    parser.add_argument("--potts-steps", type=int, default=100_000)
    parser.add_argument("--parallel-portfolio", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pvdw",
        description="Finite polynomial van der Waerden difference-coloring tools",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="inspect an exact instance")
    _add_instance_arguments(inspect_parser)
    inspect_parser.add_argument("--json", action="store_true")
    inspect_parser.set_defaults(handler=_inspect)

    encode_parser = subparsers.add_parser("encode", help="write a DIMACS encoding")
    _add_instance_arguments(encode_parser)
    encode_parser.add_argument("--encoding", choices=("onehot", "binary"), required=True)
    encode_parser.add_argument(
        "--amo",
        choices=tuple(encoding.value for encoding in AtMostOneEncoding),
        default=AtMostOneEncoding.PAIRWISE.value,
    )
    encode_parser.add_argument("--output", required=True)
    encode_parser.add_argument("--no-fix-first-color", action="store_true")
    encode_parser.add_argument("--anchor-clique", action="store_true")
    encode_parser.add_argument("--json", action="store_true")
    encode_parser.set_defaults(handler=_encode)

    solve_parser = subparsers.add_parser("solve", help="solve a full or structured model")
    _add_instance_arguments(solve_parser)
    _add_encoding_arguments(solve_parser)
    _add_search_arguments(solve_parser)
    solve_parser.add_argument(
        "--mode", choices=("direct", "periodic", "twisted"), default="direct"
    )
    solve_parser.add_argument("--period", type=int)
    solve_parser.add_argument("--twist", type=int, default=0)
    solve_parser.add_argument("--size-limit", type=int, default=30)
    solve_parser.add_argument("--output", help="write a verified JSON witness")
    solve_parser.add_argument("--json", action="store_true")
    solve_parser.set_defaults(handler=_solve)

    scan_parser = subparsers.add_parser("scan", help="incrementally scan over N")
    _add_polynomial_arguments(scan_parser)
    _add_encoding_arguments(scan_parser)
    _add_search_arguments(scan_parser)
    scan_parser.add_argument("--N-min", dest="n_min", type=int, required=True)
    scan_parser.add_argument("--N-max", dest="n_max", type=int, required=True)
    scan_parser.add_argument("--json", action="store_true")
    scan_parser.set_defaults(handler=_scan)

    periods_parser = subparsers.add_parser("periods", help="scan finite periods")
    _add_instance_arguments(periods_parser)
    _add_encoding_arguments(periods_parser)
    _add_search_arguments(periods_parser)
    periods_parser.add_argument("--period-min", type=int, required=True)
    periods_parser.add_argument("--period-max", type=int, required=True)
    periods_parser.add_argument("--json", action="store_true")
    periods_parser.set_defaults(handler=_periods)

    twists_parser = subparsers.add_parser("twists", help="scan twisted periods")
    _add_instance_arguments(twists_parser)
    _add_encoding_arguments(twists_parser)
    _add_search_arguments(twists_parser)
    twists_parser.add_argument("--period-min", type=int, required=True)
    twists_parser.add_argument("--period-max", type=int, required=True)
    twists_parser.add_argument("--twists", required=True)
    twists_parser.add_argument("--json", action="store_true")
    twists_parser.set_defaults(handler=_twists)

    repair_parser = subparsers.add_parser("repair", help="repair a structured backbone")
    _add_instance_arguments(repair_parser)
    _add_encoding_arguments(repair_parser)
    _add_search_arguments(repair_parser)
    repair_parser.add_argument(
        "--backbone", choices=("direct", "periodic", "twisted"), required=True
    )
    repair_parser.add_argument("--period", type=int)
    repair_parser.add_argument("--twist", type=int, default=0)
    repair_parser.add_argument(
        "--editable-strategy",
        choices=("all_endpoints", "greedy_vertex_cover", "exact_vertex_cover"),
        default="greedy_vertex_cover",
    )
    repair_parser.add_argument("--max-expansions", type=int, default=3)
    repair_parser.add_argument("--output")
    repair_parser.add_argument("--json", action="store_true")
    repair_parser.set_defaults(handler=_repair)

    backends_parser = subparsers.add_parser("backends", help="list discovered backends")
    backends_parser.add_argument("--json", action="store_true")
    backends_parser.set_defaults(handler=_backends)

    benchmark_parser = subparsers.add_parser("benchmark", help="run a benchmark suite")
    benchmark_parser.add_argument("--suite", required=True)
    benchmark_parser.add_argument("--timeout", type=float, default=30.0)
    benchmark_parser.add_argument("--backend", default="portfolio")
    benchmark_parser.add_argument("--seed", type=int, default=0)
    benchmark_parser.add_argument("--output-dir", default="benchmark-results")
    benchmark_parser.add_argument("--json", action="store_true")
    benchmark_parser.set_defaults(handler=_benchmark)

    verify_parser = subparsers.add_parser("verify", help="verify a JSON witness")
    verify_parser.add_argument("witness")
    verify_parser.add_argument("--json", action="store_true")
    verify_parser.set_defaults(handler=_verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    try:
        args = parser.parse_args(arguments)
        return int(args.handler(args))
    except (PolynomialParseError, WitnessFormatError, ValueError) as error:
        if "--json" in arguments:
            print(json.dumps({"error": str(error), "exit_code": EXIT_INVALID_INPUT}))
        else:
            print(f"pvdw: invalid input: {error}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    except Exception as error:
        if "--json" in arguments:
            print(json.dumps({"error": str(error), "exit_code": EXIT_ERROR}))
        else:
            print(f"pvdw: internal error: {error}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
