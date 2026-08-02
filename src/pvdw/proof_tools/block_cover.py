"""Exact translated-block covers and the spindle modular obstruction."""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ortools.sat.python import cp_model

from pvdw.proof_tools.gadgets import MOSER_SHAPES


@dataclass(frozen=True, order=True)
class Block:
    points: tuple[int, ...]
    shape_name: str
    translation: int
    reflected: bool

    def __post_init__(self) -> None:
        if self.points != tuple(sorted(set(self.points))) or not self.points:
            raise ValueError("block points must be sorted, unique, and nonempty")
        if any(type(point) is not int for point in self.points):
            raise TypeError("block points must be ordinary integers")
        if not self.shape_name:
            raise ValueError("block shape_name must be nonempty")


@dataclass(frozen=True)
class BlockCoverWitness:
    selected_blocks: tuple[Block, ...]
    ambient_points: tuple[int, ...]
    multiplicity: int
    point_counts: tuple[tuple[int, int], ...]
    verified: bool
    candidate_block_count: int


@dataclass(frozen=True)
class OneExceptionObstruction:
    colors: int
    block_count: int
    multiplicity: int
    residue: int
    permitted_counts: tuple[int, ...]
    minimum_total: int
    impossible: bool
    explanation: tuple[str, ...]


def _named_shapes(
    shapes: Mapping[str, Iterable[int]] | Iterable[Iterable[int]],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    entries = shapes.items() if isinstance(shapes, Mapping) else (
        (f"shape-{index}", points) for index, points in enumerate(shapes)
    )
    normalized: list[tuple[str, tuple[int, ...]]] = []
    for name, supplied in entries:
        points = tuple(sorted(set(supplied)))
        if not points or any(type(point) is not int for point in points):
            raise ValueError("shape points must be nonempty ordinary-integer sets")
        minimum = min(points)
        normalized.append((str(name), tuple(point - minimum for point in points)))
    return tuple(normalized)


def enumerate_translated_blocks(
    shapes: Mapping[str, Iterable[int]] | Iterable[Iterable[int]],
    ambient_min: int,
    ambient_max: int,
    *,
    include_reflections: bool,
) -> list[Block]:
    """Enumerate all translated orientations contained in an ambient interval."""

    if not ambient_min <= ambient_max:
        raise ValueError("ambient interval is empty")
    blocks: set[Block] = set()
    for name, shape in _named_shapes(shapes):
        span = max(shape)
        orientations = [(shape, False)]
        if include_reflections:
            reflected = tuple(sorted(span - point for point in shape))
            if reflected != shape:
                orientations.append((reflected, True))
        for oriented, reflected_flag in orientations:
            for translation in range(ambient_min, ambient_max - span + 1):
                points = tuple(point + translation for point in oriented)
                blocks.add(Block(points, name, translation, reflected_flag))
    return sorted(blocks)


def cover_counts(
    blocks: Iterable[Block],
    ambient_points: Iterable[int],
) -> dict[int, int]:
    ambient = tuple(sorted(set(ambient_points)))
    counts = {point: 0 for point in ambient}
    for block in blocks:
        for point in block.points:
            if point not in counts:
                raise ValueError(f"selected block contains nonambient point {point}")
            counts[point] += 1
    return counts


def verify_uniform_cover(
    blocks: Iterable[Block],
    ambient_points: Iterable[int],
    multiplicity: int = 7,
) -> bool:
    if type(multiplicity) is not int or multiplicity < 0:
        raise ValueError("cover multiplicity must be a nonnegative integer")
    return all(
        count == multiplicity
        for count in cover_counts(blocks, ambient_points).values()
    )


def verify_block_cover_witness(witness: BlockCoverWitness) -> bool:
    counts = cover_counts(witness.selected_blocks, witness.ambient_points)
    return (
        witness.verified
        and verify_uniform_cover(
            witness.selected_blocks, witness.ambient_points, witness.multiplicity
        )
        and tuple(sorted(counts.items())) == witness.point_counts
    )


def _new_solver() -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    return solver


def find_exact_cover(
    blocks: Iterable[Block],
    ambient_points: Iterable[int],
    *,
    multiplicity: int | None,
    multiplicity_range: tuple[int, int] | None,
    binary_selection: bool = True,
    minimize_block_count: bool = False,
) -> BlockCoverWitness:
    """Solve a uniform incidence cover and independently recount every point."""

    candidates = tuple(sorted(set(blocks)))
    ambient = tuple(sorted(set(ambient_points)))
    if not candidates or not ambient:
        raise ValueError("exact-cover search needs blocks and ambient points")
    if (multiplicity is None) == (multiplicity_range is None):
        raise ValueError("supply exactly one of multiplicity or multiplicity_range")
    if multiplicity is not None and (type(multiplicity) is not int or multiplicity < 0):
        raise ValueError("multiplicity must be a nonnegative ordinary integer")
    if multiplicity_range is not None:
        lower, upper = multiplicity_range
        if not 0 <= lower <= upper:
            raise ValueError("multiplicity_range is invalid")
    else:
        assert multiplicity is not None
        lower = upper = multiplicity
    ambient_set = set(ambient)
    if any(not set(block.points) <= ambient_set for block in candidates):
        raise ValueError("candidate block extends outside ambient_points")

    model = cp_model.CpModel()
    selections = [
        model.new_bool_var(f"block_{index}")
        if binary_selection
        else model.new_int_var(0, upper, f"block_{index}")
        for index in range(len(candidates))
    ]
    multiplicity_variable = (
        model.new_int_var(lower, upper, "multiplicity")
        if multiplicity_range is not None
        else None
    )
    target = multiplicity if multiplicity_variable is None else multiplicity_variable
    for point in ambient:
        incident = [
            selections[index]
            for index, block in enumerate(candidates)
            if point in block.points
        ]
        if not incident:
            raise ValueError(f"ambient point {point} belongs to no candidate block")
        model.add(sum(incident) == target)
    total_blocks = sum(selections)

    if minimize_block_count:
        model.minimize(total_blocks)
        solver = _new_solver()
        status = solver.solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise ValueError("no exact block cover exists")
        optimum_blocks = int(solver.value(total_blocks))
        model.add(total_blocks == optimum_blocks)
        model.clear_objective()
    if multiplicity_variable is not None:
        model.minimize(multiplicity_variable)
        solver = _new_solver()
        status = solver.solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise ValueError("no exact block cover exists")
        optimum_multiplicity = solver.value(multiplicity_variable)
        model.add(multiplicity_variable == optimum_multiplicity)
        model.clear_objective()

    if minimize_block_count and binary_selection:
        # With the preceding objectives fixed, prefer the earliest feasible
        # selected column at each position, which fixes the selected index list
        # lexicographically by repeated feasibility solves.
        for selection in selections:
            model.add_assumption(selection)
            trial_solver = _new_solver()
            feasible = trial_solver.solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)
            model.clear_assumptions()
            if feasible:
                model.add(selection == 1)
            else:
                model.add(selection == 0)

    solver = _new_solver()
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise ValueError("no exact block cover exists")
    chosen: list[Block] = []
    for index, block in enumerate(candidates):
        chosen.extend([block] * int(solver.value(selections[index])))
    solved_multiplicity = (
        int(multiplicity)
        if multiplicity_variable is None
        else int(solver.value(multiplicity_variable))
    )
    counts = cover_counts(chosen, ambient)
    verified = all(count == solved_multiplicity for count in counts.values())
    if not verified:
        raise RuntimeError("CP-SAT cover failed independent incidence verification")
    return BlockCoverWitness(
        tuple(chosen),
        ambient,
        solved_multiplicity,
        tuple(sorted(counts.items())),
        True,
        len(candidates),
    )


def find_lexicographic_exact_cover(
    shapes: Mapping[str, Iterable[int]] | Iterable[Iterable[int]],
    ambient_min: int,
    ambient_max_limit: int,
    *,
    multiplicity: int | None,
    multiplicity_range: tuple[int, int] | None,
    include_reflections: bool = True,
) -> BlockCoverWitness:
    """Optimize span, block count, multiplicity, then selected block list."""

    if ambient_min > ambient_max_limit:
        raise ValueError("ambient span search range is empty")
    materialized = (
        {name: tuple(points) for name, points in shapes.items()}
        if isinstance(shapes, Mapping)
        else tuple(tuple(points) for points in shapes)
    )
    for ambient_max in range(ambient_min, ambient_max_limit + 1):
        blocks = enumerate_translated_blocks(
            materialized,
            ambient_min,
            ambient_max,
            include_reflections=include_reflections,
        )
        if not blocks:
            continue
        try:
            return find_exact_cover(
                blocks,
                range(ambient_min, ambient_max + 1),
                multiplicity=multiplicity,
                multiplicity_range=multiplicity_range,
                minimize_block_count=True,
            )
        except ValueError:
            continue
    raise ValueError("no exact cover exists within the ambient span limit")


SPINDLE_SEVENFOLD_SELECTIONS: dict[str, tuple[int, ...]] = {
    "A": (0, 1),
    "B": (0, 1, 2),
    "C": (0, 4),
    "D0": (0, 3, 4),
    "E": (0, 1, 4),
    "F": (0, 4),
    "G": (0, 1),
}


def spindle_sevenfold_blocks() -> tuple[Block, ...]:
    blocks = []
    for name, translations in SPINDLE_SEVENFOLD_SELECTIONS.items():
        for translation in translations:
            blocks.append(
                Block(
                    tuple(point + translation for point in MOSER_SHAPES[name]),
                    name,
                    translation,
                    name in {"E", "F", "G"},
                )
            )
    return tuple(blocks)


def check_one_exception_signature_obstruction(
    colors: int = 4,
    block_count: int = 17,
    multiplicity: int = 7,
    exceptional_size: int = 1,
    ordinary_size: int = 2,
) -> OneExceptionObstruction:
    """Check the modular singleton-count obstruction for a uniform cover."""

    values = (colors, block_count, multiplicity, exceptional_size, ordinary_size)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("obstruction parameters must be positive ordinary integers")
    coefficient = ordinary_size - exceptional_size
    if coefficient <= 0:
        raise ValueError("ordinary_size must exceed exceptional_size")
    permitted = tuple(
        count
        for count in range(block_count + 1)
        if (ordinary_size * block_count - coefficient * count) % multiplicity == 0
    )
    residue = permitted[0] % multiplicity if permitted else -1
    minimum_total = colors * min(permitted) if permitted else block_count + 1
    impossible = not any(
        sum(counts) == block_count
        for counts in itertools.product(permitted, repeat=colors)
    )
    return OneExceptionObstruction(
        colors,
        block_count,
        multiplicity,
        residue,
        permitted,
        minimum_total,
        impossible,
        (
            f"n_j ≡ {residue} mod {multiplicity}",
            f"n_j ∈ {{{','.join(map(str, permitted))}}}",
            f"sum_j n_j >= {minimum_total} > {block_count}",
        ),
    )
