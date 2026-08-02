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
from pvdw.proof_tools.block_cover import (
    check_one_exception_signature_obstruction,
    enumerate_translated_blocks,
    find_exact_cover,
    spindle_sevenfold_blocks,
    verify_uniform_cover,
)
from pvdw.proof_tools.difference_cycles import (
    KNOWN_CUBE_WALKS,
    build_difference_edges,
    find_negative_cycle,
    verify_negative_cycle,
    walk_negative_cycle,
)
from pvdw.proof_tools.drift import CUBE_DISTANCES, cube_drift_assignments
from pvdw.proof_tools.export import ProofArtifact, to_jsonable, write_proof_artifact
from pvdw.proof_tools.gadgets import (
    MOSER_ASSIGNMENTS,
    MOSER_SHAPES,
    MOSER_SPINDLE,
    SmallGraph,
    find_embeddings,
    embedding_from_images,
    interval_distance_graph,
    verify_embedding,
)
from pvdw.proof_tools.gf2poly import gf2_degree, gf2_to_exponents
from pvdw.proof_tools.local import (
    find_common_neighborhood_odd_cycle,
    maximum_avoiding_subset_dp,
    maximum_independent_set,
)
from pvdw.proof_tools.parity_cover import (
    polynomial_summary,
    square_submillion_parity_witness,
)
from pvdw.proof_tools.signatures import color_signatures
from pvdw.proof_tools.transfer import search_common_scale, verify_transfer_witness
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


def _parse_integer_list(text: str, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as error:
        raise ValueError(f"{name} must be a comma-separated integer list") from error
    if not values:
        raise ValueError(f"{name} must not be empty")
    return values


def _load_small_graph(target: str) -> tuple[str, SmallGraph]:
    if target.lower() == "moser":
        return "Moser spindle", MOSER_SPINDLE
    path = Path(target)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("target edge-list JSON must be an object")
    vertices = payload.get("vertices")
    edges = payload.get("edges")
    if not isinstance(vertices, list) or not isinstance(edges, list):
        raise ValueError("target JSON requires vertices and edges arrays")
    try:
        graph = SmallGraph(
            tuple(str(vertex) for vertex in vertices),
            tuple((str(edge[0]), str(edge[1])) for edge in edges),
        )
    except (IndexError, TypeError) as error:
        raise ValueError("every target edge must contain two vertex names") from error
    return str(payload.get("name", path.stem)), graph


def _write_cli_artifact(
    args: argparse.Namespace,
    artifact: ProofArtifact,
) -> int:
    files = write_proof_artifact(
        args.output_prefix, artifact, include_latex=getattr(args, "latex", False)
    )
    payload = {
        "title": artifact.title,
        "verified": artifact.verified,
        "proof": to_jsonable(artifact.data),
        "files": {
            "json": str(files.json_path),
            "text": str(files.text_path),
            "latex": str(files.latex_path) if files.latex_path else None,
        },
    }
    lines = [
        artifact.text.rstrip(),
        f"JSON: {files.json_path}",
        f"Text: {files.text_path}",
    ]
    if files.latex_path:
        lines.append(f"LaTeX: {files.latex_path}")
    _emit(payload, as_json=args.json, lines=lines)
    return EXIT_SUCCESS


def _gadget(args: argparse.Namespace) -> int:
    target_name, target = _load_small_graph(args.target)
    differences = _parse_integer_list(args.differences, "differences")
    host = interval_distance_graph(args.ambient_max, differences)
    embeddings = find_embeddings(
        target,
        host,
        induced=args.induced,
        normalize_translation=True,
        identify_reflections=True,
    )
    if not embeddings:
        raise ValueError("no target embedding exists in the supplied distance graph")
    preferred = (
        MOSER_ASSIGNMENTS["A"] if target == MOSER_SPINDLE else embeddings[0].images
    )
    embedding = next(
        (candidate for candidate in embeddings if candidate.images == preferred),
        embeddings[0],
    )
    verified = verify_embedding(target, embedding, differences)
    if not verified:
        raise RuntimeError("selected graph embedding failed independent edge checking")
    signatures = color_signatures(target, args.colors)
    alpha_set = maximum_independent_set(target)
    signature_rows = [
        {
            "class_sizes": list(signature.class_sizes),
            "used_colors": signature.used_colors,
            "colorings_modulo_permutation": count,
        }
        for signature, count in sorted(
            signatures.items(), key=lambda item: (item[0].class_sizes, item[0].used_colors)
        )
    ]
    mapping_text = ", ".join(
        f"{vertex}={image}" for vertex, image in embedding.mapping.items()
    )
    signature_reason = (
        (
            "every color class is independent",
            "every independent set has size at most two",
            "seven vertices need four colors",
            "the class sizes are (2,2,2,1)",
        )
        if target == MOSER_SPINDLE and len(alpha_set) == 2
        else (
            "every color class is independent",
            f"the computed independence number is {len(alpha_set)}",
        )
    )
    text = "\n".join(
        (
            f"Target: {target_name}",
            f"Host points: 0..{args.ambient_max}",
            f"Allowed differences: {','.join(map(str, sorted(set(differences))))}",
            "Embedding:",
            mapping_text,
            f"All {len(target.edges)} target edges verified.",
            f"Normalized embeddings: {len(embeddings)}",
            f"Maximum independent set: {','.join(map(str, alpha_set))}",
            "Color-class signatures: "
            + "; ".join(
                ",".join(map(str, row["class_sizes"])) for row in signature_rows
            ),
            "Mathematical signature reason:",
            *(f"{line};" for line in signature_reason),
        )
    )
    artifact = ProofArtifact(
        f"{target_name} distance-graph embedding",
        {
            "target": target_name,
            "target_vertices": target.vertices,
            "target_edges": target.edges,
            "host": [0, args.ambient_max],
            "allowed_differences": tuple(sorted(set(differences))),
            "embedding": embedding,
            "embedding_count": len(embeddings),
            "all_edges_verified": True,
            "maximum_independent_set": alpha_set,
            "independence_number": len(alpha_set),
            "color_signatures": signature_rows,
            "signature_reason": signature_reason,
        },
        text,
        verified,
        latex=(
            r"\textbf{Embedding: }" + mapping_text.replace(", ", r",\ ")
            + rf".\quad \alpha={len(alpha_set)}."
        ),
    )
    return _write_cli_artifact(args, artifact)


def _spindle_fixture_embeddings_verified(differences: tuple[int, ...]) -> bool:
    assignments = dict(MOSER_ASSIGNMENTS)
    assignments.update(
        {
            "E": tuple(12 - value for value in MOSER_ASSIGNMENTS["D0"]),
            "F": tuple(12 - value for value in MOSER_ASSIGNMENTS["C"]),
            "G": tuple(15 - value for value in MOSER_ASSIGNMENTS["A"]),
        }
    )
    return all(
        verify_embedding(
            MOSER_SPINDLE,
            embedding_from_images(MOSER_SPINDLE, images),
            differences,
        )
        and tuple(sorted(images)) == MOSER_SHAPES[name]
        for name, images in assignments.items()
    )


def _block_cover(args: argparse.Namespace) -> int:
    if args.example is not None:
        if args.example != "x2-plus-x-spindle":
            raise ValueError(f"unknown block-cover example {args.example!r}")
        differences = (1, 2, 5, 7, 12, 15)
        selected = spindle_sevenfold_blocks()
        ambient = tuple(range(17))
        if not _spindle_fixture_embeddings_verified(differences):
            raise RuntimeError("a supplied spindle fixture embedding failed")
        cover_verified = verify_uniform_cover(selected, ambient, 7)
        obstruction = check_one_exception_signature_obstruction()
        signatures = color_signatures(MOSER_SPINDLE, 4)
        transfer = search_common_scale(
            parse_polynomial("x^2+x"), differences, 100, 100
        )
        verified = (
            cover_verified
            and obstruction.impossible
            and len(signatures) == 1
            and transfer is not None
            and verify_transfer_witness(transfer)
        )
        if not verified:
            raise RuntimeError("spindle block-cover example failed independent checking")
        text = "\n".join(
            (
                "Target: Moser spindle transferred into p(x)=x^2+x at scale 6",
                f"Selected blocks: {len(selected)}",
                "Ambient points: 0..16",
                "Multiplicity of every point: 7",
                "Modular color-count contradiction:",
                "each singleton count is 6 modulo 7;",
                "four singleton counts therefore total at least 24;",
                "but only 17 blocks exist.",
            )
        )
        artifact = ProofArtifact(
            "x^2+x spindle seven-fold cover obstruction",
            {
                "differences": differences,
                "shapes": MOSER_SHAPES,
                "selected_blocks": selected,
                "selected_block_count": len(selected),
                "ambient_points": ambient,
                "multiplicity": 7,
                "uniform_cover_verified": True,
                "color_signatures": [
                    {
                        "class_sizes": signature.class_sizes,
                        "used_colors": signature.used_colors,
                        "count": count,
                    }
                    for signature, count in signatures.items()
                ],
                "obstruction": obstruction,
                "transfer": transfer,
            },
            text,
            True,
            latex=(
                r"34-n_j\equiv0\pmod 7,\quad n_j\in\{6,13\}. "
                r"\text{ Hence }\sum_{j=1}^4n_j\ge24>17."
            ),
        )
        return _write_cli_artifact(args, artifact)

    if args.target is None or args.differences is None or args.ambient_max is None:
        raise ValueError(
            "generic block-cover search requires --target, --differences, and --ambient-max"
        )
    if args.multiplicity is None:
        raise ValueError("generic block-cover search requires --multiplicity")
    target_name, target = _load_small_graph(args.target)
    differences = _parse_integer_list(args.differences, "differences")
    host = interval_distance_graph(args.ambient_max, differences)
    embeddings = find_embeddings(target, host)
    shapes = {
        f"embedding-{index:03d}": shape
        for index, shape in enumerate(sorted({embedding.shape for embedding in embeddings}))
    }
    candidates = enumerate_translated_blocks(
        shapes, 0, args.ambient_max, include_reflections=True
    )
    cover = find_exact_cover(
        candidates,
        range(args.ambient_max + 1),
        multiplicity=args.multiplicity,
        multiplicity_range=None,
        minimize_block_count=args.minimize_blocks,
    )
    signatures = color_signatures(target, args.colors)
    obstruction = None
    if len(signatures) == 1:
        signature = next(iter(signatures))
        size_counts: dict[int, int] = {}
        for size in signature.class_sizes:
            size_counts[size] = size_counts.get(size, 0) + 1
        if len(size_counts) == 2:
            exceptional_size, ordinary_size = sorted(size_counts)
            if size_counts[exceptional_size] == 1:
                obstruction = check_one_exception_signature_obstruction(
                    colors=args.colors,
                    block_count=len(cover.selected_blocks),
                    multiplicity=cover.multiplicity,
                    exceptional_size=exceptional_size,
                    ordinary_size=ordinary_size,
                )
    obstruction_lines = (
        (
            "Modular color-count contradiction:",
            *obstruction.explanation,
        )
        if obstruction is not None and obstruction.impossible
        else ()
    )
    text = "\n".join(
        (
            f"Target: {target_name}",
            f"Normalized embeddings: {len(embeddings)}",
            f"Candidate translated blocks: {len(candidates)}",
            f"Selected blocks: {len(cover.selected_blocks)}",
            f"Ambient points: 0..{args.ambient_max}",
            f"Multiplicity of every point: {cover.multiplicity}",
            "Expanded block incidence independently verified.",
            *obstruction_lines,
        )
    )
    artifact = ProofArtifact(
        f"{target_name} exact translated-block cover",
        {
            "target": target_name,
            "differences": differences,
            "embedding_count": len(embeddings),
            "shape_count": len(shapes),
            "cover": cover,
            "color_signatures": [
                {
                    "class_sizes": signature.class_sizes,
                    "used_colors": signature.used_colors,
                    "count": count,
                }
                for signature, count in signatures.items()
            ],
            "one_exception_obstruction": obstruction,
        },
        text,
        cover.verified,
    )
    return _write_cli_artifact(args, artifact)


def _parity_cover(args: argparse.Namespace) -> int:
    if args.example != "square-submillion":
        raise ValueError(f"unknown parity-cover example {args.example!r}")
    witness = square_submillion_parity_witness()
    a, b, c = witness.shape_polynomials
    summaries = {
        "A": polynomial_summary(a),
        "B": polynomial_summary(b),
        "C": polynomial_summary(c),
        "q": polynomial_summary(witness.q),
        "gcd_A_B": polynomial_summary(witness.gcd_ab),
        "U": polynomial_summary(witness.u),
        "V": polynomial_summary(witness.v),
    }
    text = "\n".join(
        (
            "Example: square-submillion",
            "All three four-point shapes have only square pairwise differences.",
            f"degree(q) = {gf2_degree(witness.q)}",
            f"degree(gcd(A,B)) = {gf2_degree(witness.gcd_ab)}",
            f"degrees(A0,B0,T) = {witness.normalized_degrees}",
            f"degrees(U,V,q) = {witness.coefficient_degrees}",
            f"degrees(UA,VB,qC) = {witness.product_degrees}",
            "UA + VB + qC = 0: verified",
            "U(1) + V(1) + q(1) = 1: verified",
            "Expanded incidence parity: even at every point",
            f"Maximum translated point: {witness.maximum_point}",
            f"Interval size: {witness.interval_size}",
            f"Peak resident memory: {witness.peak_memory_bytes} bytes",
            f"Packed witness-polynomial storage: {witness.packed_polynomial_bytes} bytes",
            f"Runtime: {witness.elapsed_seconds:.6f} seconds",
        )
    )
    artifact = ProofArtifact(
        "Submillion square-distance odd parity cover",
        {
            "shapes": witness.shapes,
            "polynomials": summaries,
            "gcd_exponents": gf2_to_exponents(witness.gcd_ab),
            "relation_zero": witness.relation_zero,
            "odd_block_count": witness.odd_block_count,
            "expanded_even_cover": witness.expanded_even_cover,
            "normalized_degrees_A0_B0_T": witness.normalized_degrees,
            "coefficient_degrees_U_V_q": witness.coefficient_degrees,
            "product_degrees_UA_VB_qC": witness.product_degrees,
            "maximum_translated_point": witness.maximum_point,
            "interval_size": witness.interval_size,
            "runtime_seconds": witness.elapsed_seconds,
            "peak_memory_bytes": witness.peak_memory_bytes,
            "packed_polynomial_bytes": witness.packed_polynomial_bytes,
        },
        text,
        witness.relation_zero and witness.odd_block_count and witness.expanded_even_cover,
        latex=(
            r"U A+V B+q C=0,\qquad U(1)+V(1)+q(1)=1,"
            rf"\qquad \max\deg={witness.maximum_point}."
        ),
    )
    return _write_cli_artifact(args, artifact)


def _drift(args: argparse.Namespace) -> int:
    if args.example != "cubes-3color-522":
        raise ValueError(f"unknown drift example {args.example!r}")
    enumeration = cube_drift_assignments()
    cases = []
    text_lines = [
        "Example: cubes-3color-522",
        "Drift domains:",
        *(
            f"q_{distance}: {','.join(map(str, domain))}"
            for distance, domain in enumeration.domains
        ),
        "Surviving assignments after each relation: "
        + ",".join(map(str, enumeration.survivor_counts)),
        f"Final cases: {len(enumeration.assignments)}",
    ]
    for index, (assignment, known_walk) in enumerate(
        zip(enumeration.assignments, KNOWN_CUBE_WALKS), start=1
    ):
        known = walk_negative_cycle(known_walk, assignment.mapping)
        edges = build_difference_edges(522, assignment.mapping)
        found = find_negative_cycle(range(522), edges)
        if found is None or not verify_negative_cycle(found):
            raise RuntimeError(f"cube drift case {index} has no verified negative cycle")
        if known.total_weight != -1:
            raise RuntimeError(
                f"supplied cube walk {index} has weight {known.total_weight}, expected -1"
            )
        running = 0
        edge_rows = []
        text_lines.append(f"Case {index}: {assignment.mapping}")
        for edge in found.edges:
            running += edge.weight
            edge_rows.append(
                {
                    "source": edge.source,
                    "target": edge.target,
                    "distance": edge.distance,
                    "direction": edge.direction,
                    "weight": edge.weight,
                    "running_weight": running,
                }
            )
            text_lines.append(
                f"  {edge.source} -> {edge.target}: d={edge.distance}, "
                f"direction={edge.direction:+d}, weight={edge.weight}, running={running}"
            )
        text_lines.append(f"  total weight: {found.total_weight}")
        cases.append(
            {
                "assignment": assignment,
                "automatic_cycle": {
                    "vertices": found.vertices,
                    "edges": edge_rows,
                    "total_weight": found.total_weight,
                },
                "known_walk": known,
            }
        )
    artifact = ProofArtifact(
        "Cube three-color drift negative-cycle obstruction",
        {
            "distances": CUBE_DISTANCES,
            "domains": enumeration.domains,
            "survivor_counts": enumeration.survivor_counts,
            "cases": cases,
        },
        "\n".join(text_lines),
        True,
    )
    return _write_cli_artifact(args, artifact)


def _transfer(args: argparse.Namespace) -> int:
    polynomial = _polynomial_from_args(args)
    differences = _parse_integer_list(args.source_differences, "source differences")
    witness = search_common_scale(
        polynomial, differences, args.max_scale, args.input_bound
    )
    if witness is None:
        raise ValueError("no common scale exists within the supplied bounds")
    if not verify_transfer_witness(witness):
        raise RuntimeError("common-scale witness failed independent verification")
    equations = tuple(
        f"|p({witness.inputs[difference]})| = {witness.realized_values[difference]} "
        f"= {witness.scale}*{difference}"
        for difference in witness.source_differences
    )
    text = "\n".join(
        (
            f"Polynomial coefficients: {list(polynomial.coefficients)}",
            f"Source differences: {','.join(map(str, witness.source_differences))}",
            f"Common scale: {witness.scale}",
            *equations,
            "Every equality independently verified with exact integer evaluation.",
        )
    )
    artifact = ProofArtifact(
        "Polynomial common-scale transfer",
        {
            "coefficients": polynomial.coefficients,
            "source_differences": witness.source_differences,
            "scale": witness.scale,
            "inputs": witness.inputs,
            "realized_values": witness.realized_values,
        },
        text,
        True,
        latex="\\\n".join(equations),
    )
    return _write_cli_artifact(args, artifact)


def _local_proof(args: argparse.Namespace) -> int:
    differences = _parse_integer_list(args.differences, "differences")
    if args.local_command == "gaps":
        result = maximum_avoiding_subset_dp(
            args.n, differences, state_limit=args.state_limit
        )
        data = {"mode": "gaps", "result": result}
        text = "\n".join(
            (
                f"Distance-avoiding subset on [0,{args.n - 1}]",
                f"Differences: {','.join(map(str, differences))}",
                f"Maximum size: {result.size}",
                f"Subset: {','.join(map(str, result.subset))}",
                f"Peak DP states: {result.peak_states}",
            )
        )
    else:
        graph = interval_distance_graph(args.ambient_max, differences)
        if args.local_command == "independence":
            independent = maximum_independent_set(graph)
            data = {
                "mode": "independence",
                "maximum_independent_set": independent,
                "independence_number": len(independent),
            }
            text = "\n".join(
                (
                    f"Host points: 0..{args.ambient_max}",
                    f"Differences: {','.join(map(str, differences))}",
                    f"Independence number: {len(independent)}",
                    f"Maximum independent set: {','.join(map(str, independent))}",
                )
            )
        else:
            witness = find_common_neighborhood_odd_cycle(graph)
            if witness is None:
                raise ValueError("no common-neighborhood odd cycle was found")
            data = {"mode": "common-odd-cycle", "witness": witness}
            text = "\n".join(
                (
                    f"Host points: 0..{args.ambient_max}",
                    f"Common-neighborhood endpoints: {witness.u},{witness.v}",
                    "Odd cycle: " + " -> ".join(map(str, witness.cycle)),
                )
            )
    artifact = ProofArtifact(f"Local {args.local_command} proof", data, text, True)
    return _write_cli_artifact(args, artifact)


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


def _add_proof_output_arguments(
    parser: argparse.ArgumentParser,
    default_name: str,
) -> None:
    parser.add_argument(
        "--output-prefix",
        default=f"proof-artifacts/{default_name}",
        help="path prefix for verified .json/.txt proof files",
    )
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--json", action="store_true")


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

    gadget_parser = subparsers.add_parser(
        "gadget", help="find a rigid target graph in a finite distance graph"
    )
    gadget_parser.add_argument("--target", required=True)
    gadget_parser.add_argument("--differences", required=True)
    gadget_parser.add_argument("--ambient-max", type=int, required=True)
    gadget_parser.add_argument("--colors", type=int, default=4)
    gadget_parser.add_argument("--induced", action="store_true")
    _add_proof_output_arguments(gadget_parser, "gadget")
    gadget_parser.set_defaults(handler=_gadget)

    block_parser = subparsers.add_parser(
        "block-cover", help="search for an exact translated gadget cover"
    )
    block_parser.add_argument("--example")
    block_parser.add_argument("--target")
    block_parser.add_argument("--differences")
    block_parser.add_argument("--ambient-max", type=int)
    block_parser.add_argument("--multiplicity", type=int)
    block_parser.add_argument("--colors", type=int, default=4)
    block_parser.add_argument("--minimize-blocks", action="store_true")
    _add_proof_output_arguments(block_parser, "block-cover")
    block_parser.set_defaults(handler=_block_cover)

    parity_parser = subparsers.add_parser(
        "parity-cover", help="construct an odd GF(2) translated-shape cover"
    )
    parity_parser.add_argument("--example", required=True)
    _add_proof_output_arguments(parity_parser, "parity-cover")
    parity_parser.set_defaults(handler=_parity_cover)

    drift_parser = subparsers.add_parser(
        "drift", help="derive signed drifts and negative-cycle obstructions"
    )
    drift_parser.add_argument("--example", required=True)
    _add_proof_output_arguments(drift_parser, "drift")
    drift_parser.set_defaults(handler=_drift)

    transfer_parser = subparsers.add_parser(
        "transfer", help="find a common-scale polynomial realization"
    )
    transfer_polynomial = transfer_parser.add_mutually_exclusive_group(required=True)
    transfer_polynomial.add_argument("--coefficients")
    transfer_polynomial.add_argument("--poly")
    transfer_parser.add_argument("--source-differences", required=True)
    transfer_parser.add_argument("--max-scale", type=int, required=True)
    transfer_parser.add_argument("--input-bound", type=int, required=True)
    _add_proof_output_arguments(transfer_parser, "transfer")
    transfer_parser.set_defaults(handler=_transfer)

    local_parser = subparsers.add_parser(
        "local-proof", help="run a small exact local obstruction search"
    )
    local_subparsers = local_parser.add_subparsers(
        dest="local_command", required=True
    )
    for command in ("independence", "common-odd-cycle"):
        child = local_subparsers.add_parser(command)
        child.add_argument("--differences", required=True)
        child.add_argument("--ambient-max", type=int, required=True)
        _add_proof_output_arguments(child, f"local-{command}")
        child.set_defaults(handler=_local_proof)
    gaps_parser = local_subparsers.add_parser("gaps")
    gaps_parser.add_argument("--differences", required=True)
    gaps_parser.add_argument("--N", dest="n", type=int, required=True)
    gaps_parser.add_argument("--state-limit", type=int, default=1_000_000)
    _add_proof_output_arguments(gaps_parser, "local-gaps")
    gaps_parser.set_defaults(handler=_local_proof)

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
