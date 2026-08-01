"""Command-line interface for Phase 1 polynomial difference-coloring tools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from pvdw.backends.bruteforce import solve_bruteforce
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
from pvdw.model import InstanceSpec, SolveStatus
from pvdw.polynomial import PolynomialParseError, parse_coefficients, parse_polynomial
from pvdw.verify import verify_coloring
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


def _instance_from_args(args: argparse.Namespace) -> InstanceSpec:
    polynomial = (
        parse_coefficients(args.coefficients)
        if args.coefficients is not None
        else parse_polynomial(args.poly)
    )
    return InstanceSpec(polynomial, args.colors, args.n, args.input_domain)


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


def _solve(args: argparse.Namespace) -> int:
    instance = _instance_from_args(args)
    result = solve_bruteforce(instance, size_limit=args.size_limit)
    payload: dict[str, Any] = {
        "status": result.status.value,
        "scope": result.scope.value,
        "elapsed_seconds": result.elapsed_seconds,
        "backend": result.backend,
        "coloring": list(result.coloring) if result.coloring is not None else None,
        "metadata": dict(result.metadata),
    }
    lines = [
        f"status: {result.status.value}",
        f"scope: {result.scope.value}",
        f"backend: {result.backend}",
        f"elapsed seconds: {result.elapsed_seconds:.6f}",
    ]
    if result.coloring is not None:
        verification = verify_coloring(instance, result.coloring)
        if not verification.valid:
            raise RuntimeError("solver result failed independent verification")
        payload["verification"] = {"valid": True}
        if instance.colors <= 36:
            word = coloring_to_word(list(result.coloring), instance.colors)
            payload["word"] = word
            lines.append(f"word: {word}")
        if args.output:
            witness = create_witness(
                instance,
                result.coloring,
                backend=result.backend,
                seed=0,
                elapsed_seconds=result.elapsed_seconds,
                scope=result.scope,
            )
            write_witness(args.output, witness)
            payload["witness"] = args.output
            lines.append(f"witness: {args.output}")
    _emit(payload, as_json=args.json, lines=lines)
    if result.status is SolveStatus.FOUND_WITNESS:
        return EXIT_SUCCESS
    if result.status is SolveStatus.UNSAT_FULL_MODEL:
        return EXIT_UNSAT
    if result.status in (SolveStatus.TIMEOUT, SolveStatus.UNKNOWN):
        return EXIT_INCONCLUSIVE
    return EXIT_ERROR


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

    solve_parser = subparsers.add_parser("solve", help="solve a tiny instance")
    _add_instance_arguments(solve_parser)
    solve_parser.add_argument("--backend", choices=("bruteforce",), required=True)
    solve_parser.add_argument("--size-limit", type=int, default=30)
    solve_parser.add_argument("--output", help="write a verified JSON witness")
    solve_parser.add_argument("--json", action="store_true")
    solve_parser.set_defaults(handler=_solve)

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
