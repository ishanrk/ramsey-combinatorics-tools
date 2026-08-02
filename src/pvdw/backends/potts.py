"""Native Potts/min-conflicts search with exact incremental energy updates."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from pvdw.backends.base import (
    BackendCapabilities,
    EncodedProblem,
    SolveOptions,
    base_metadata,
)
from pvdw.model import SolveResult, SolveStatus
from pvdw.verify import conflict_count


@dataclass(frozen=True)
class PottsOptions:
    restarts: int = 8
    max_steps: int | None = 100_000
    timeout_seconds: float = 10.0
    seed: int = 0
    initial_temperature: float = 2.0
    final_temperature: float = 0.02
    greedy_probability: float = 0.9
    tabu_min: int = 3
    tabu_max: int = 12
    stagnation_steps: int = 5_000

    def __post_init__(self) -> None:
        if self.restarts < 1:
            raise ValueError("Potts restarts must be positive")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("Potts max_steps must be positive or None")
        if self.timeout_seconds <= 0:
            raise ValueError("Potts timeout_seconds must be positive")
        if self.initial_temperature <= 0 or self.final_temperature <= 0:
            raise ValueError("Potts temperatures must be positive")
        if not 0 <= self.greedy_probability <= 1:
            raise ValueError("greedy_probability must lie in [0, 1]")
        if not 0 <= self.tabu_min <= self.tabu_max:
            raise ValueError("tabu tenures are invalid")
        if self.stagnation_steps < 1:
            raise ValueError("stagnation_steps must be positive")


class PottsState:
    """Incremental energy state for constraints ``left != right + shift``."""

    def __init__(
        self,
        vertex_count: int,
        colors: int,
        constraints: tuple[tuple[int, int, int], ...],
        coloring: list[int] | tuple[int, ...],
    ) -> None:
        if len(coloring) != vertex_count:
            raise ValueError("initial Potts coloring has the wrong length")
        self.vertex_count = vertex_count
        self.colors_count = colors
        self.constraints = tuple(constraints)
        self.colors = list(coloring)
        if any(not 0 <= color < colors for color in self.colors):
            raise ValueError("initial Potts color is out of range")
        self.incident: list[list[tuple[int, bool]]] = [
            [] for _ in range(vertex_count)
        ]
        self.neighbor_color_count = [
            [0 for _ in range(colors)] for _ in range(vertex_count)
        ]
        self.energy = 0
        for index, (left, right, shift) in enumerate(self.constraints):
            if not (0 <= left < vertex_count and 0 <= right < vertex_count):
                raise ValueError("Potts constraint endpoint is out of range")
            shift %= colors
            if left == right:
                if shift == 0:
                    raise ValueError("impossible self-constraint in Potts problem")
                continue
            self.incident[left].append((index, True))
            self.incident[right].append((index, False))
            self.neighbor_color_count[left][(self.colors[right] + shift) % colors] += 1
            self.neighbor_color_count[right][(self.colors[left] - shift) % colors] += 1
            if self.colors[left] == (self.colors[right] + shift) % colors:
                self.energy += 1
        self.conflicted = {
            vertex
            for vertex in range(vertex_count)
            if self.neighbor_color_count[vertex][self.colors[vertex]] > 0
        }

    def full_energy(self) -> int:
        return sum(
            self.colors[left] == (self.colors[right] + shift) % self.colors_count
            for left, right, shift in self.constraints
            if left != right
        )

    def move_delta(self, vertex: int, new_color: int) -> int:
        return (
            self.neighbor_color_count[vertex][new_color]
            - self.neighbor_color_count[vertex][self.colors[vertex]]
        )

    def recolor(self, vertex: int, new_color: int) -> int:
        old_color = self.colors[vertex]
        if new_color == old_color:
            return 0
        if not 0 <= new_color < self.colors_count:
            raise ValueError("new Potts color is out of range")
        delta = self.move_delta(vertex, new_color)
        affected = {vertex}
        for constraint_index, is_left in self.incident[vertex]:
            left, right, shift = self.constraints[constraint_index]
            shift %= self.colors_count
            if is_left:
                other = right
                old_forbidden = (old_color - shift) % self.colors_count
                new_forbidden = (new_color - shift) % self.colors_count
            else:
                other = left
                old_forbidden = (old_color + shift) % self.colors_count
                new_forbidden = (new_color + shift) % self.colors_count
            self.neighbor_color_count[other][old_forbidden] -= 1
            self.neighbor_color_count[other][new_forbidden] += 1
            affected.add(other)
        self.colors[vertex] = new_color
        self.energy += delta
        for candidate in affected:
            if self.neighbor_color_count[candidate][self.colors[candidate]]:
                self.conflicted.add(candidate)
            else:
                self.conflicted.discard(candidate)
        return delta


class PottsBackend:
    capabilities = BackendCapabilities(
        complete=False,
        incremental=False,
        accepts_dimacs=False,
        supports_assumptions=False,
        stochastic=True,
    )
    name = "potts"

    def __init__(
        self,
        options: PottsOptions | None = None,
        *,
        initial_coloring: tuple[int, ...] | None = None,
    ) -> None:
        self.potts_options = options or PottsOptions()
        self.initial_coloring = initial_coloring

    @staticmethod
    def _temperature(options: PottsOptions, step: int, max_steps: int) -> float:
        fraction = min(1.0, step / max(1, max_steps - 1))
        return options.initial_temperature * (
            options.final_temperature / options.initial_temperature
        ) ** fraction

    def solve(self, problem: EncodedProblem, options: SolveOptions) -> SolveResult:
        config = self.potts_options
        timeout = options.timeout_seconds or config.timeout_seconds
        seed = options.seed if options.seed != 0 else config.seed
        random_source = random.Random(seed)
        started = time.perf_counter()
        deadline = started + timeout
        constraints = problem.potts_constraints
        vertex_count = problem.decode_spec.assignment_vertices
        colors = problem.decode_spec.colors
        metadata = base_metadata(problem, options, backend_version="native")
        metadata.update(
            {
                "seed": seed,
                "restarts": config.restarts,
                "max_steps": config.max_steps,
                "quotient_constraints": len(constraints),
            }
        )
        if not constraints:
            if problem.decode_spec.encoding.startswith("onehot"):
                empty_model = [
                    (vertex * colors + color + 1)
                    * (1 if color == 0 else -1)
                    for vertex in range(vertex_count)
                    for color in range(colors)
                ]
            else:
                bits = (colors - 1).bit_length()
                empty_model = [
                    -(vertex * bits + bit + 1)
                    for vertex in range(vertex_count)
                    for bit in range(bits)
                ]
            coloring = problem.decode_model(empty_model)
            verification = problem.verify(coloring)
            if verification.valid:
                metadata.update(
                    {"model_parsing": "direct_assignment", "verification": "valid"}
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
        best_assignment: tuple[int, ...] | None = None
        best_quotient_energy = len(constraints) + 1
        total_steps = 0
        completed_restarts = 0
        max_steps = config.max_steps or 1_000_000_000
        initial = self.initial_coloring
        if initial is not None and len(initial) != vertex_count:
            raise ValueError("Potts initial coloring has the wrong quotient length")

        for restart in range(config.restarts):
            if time.perf_counter() >= deadline:
                break
            completed_restarts += 1
            coloring = (
                list(initial)
                if restart == 0 and initial is not None
                else [random_source.randrange(colors) for _ in range(vertex_count)]
            )
            state = PottsState(vertex_count, colors, constraints, coloring)
            tabu_until = [[0 for _ in range(colors)] for _ in range(vertex_count)]
            last_improvement = 0
            if state.energy < best_quotient_energy:
                best_quotient_energy = state.energy
                best_assignment = tuple(state.colors)
            for step in range(max_steps):
                total_steps += 1
                if state.energy == 0 or time.perf_counter() >= deadline:
                    break
                if step - last_improvement >= config.stagnation_steps:
                    candidates = sorted(state.conflicted)
                    random_source.shuffle(candidates)
                    count = max(1, len(candidates) // 10)
                    for vertex in candidates[:count]:
                        alternatives = [
                            color
                            for color in range(colors)
                            if color != state.colors[vertex]
                        ]
                        if alternatives:
                            state.recolor(vertex, random_source.choice(alternatives))
                    last_improvement = step
                    continue
                vertices = tuple(state.conflicted)
                weights = [
                    state.neighbor_color_count[vertex][state.colors[vertex]]
                    for vertex in vertices
                ]
                vertex = random_source.choices(vertices, weights=weights, k=1)[0]
                old_color = state.colors[vertex]
                moves = []
                for color in range(colors):
                    if color == old_color:
                        continue
                    delta = state.move_delta(vertex, color)
                    if tabu_until[vertex][color] > total_steps and state.energy + delta > 0:
                        continue
                    moves.append((color, delta))
                if not moves:
                    continue
                minimum = min(delta for _, delta in moves)
                if random_source.random() < config.greedy_probability:
                    choices = [color for color, delta in moves if delta == minimum]
                    new_color = random_source.choice(choices)
                else:
                    temperature = self._temperature(config, step, max_steps)
                    baseline = min(delta for _, delta in moves)
                    move_weights = [
                        math.exp(-(delta - baseline) / max(temperature, 1e-12))
                        for _, delta in moves
                    ]
                    new_color = random_source.choices(
                        [color for color, _ in moves], weights=move_weights, k=1
                    )[0]
                state.recolor(vertex, new_color)
                tabu_until[vertex][old_color] = total_steps + random_source.randint(
                    config.tabu_min, config.tabu_max
                )
                if state.energy < best_quotient_energy:
                    best_quotient_energy = state.energy
                    best_assignment = tuple(state.colors)
                    last_improvement = step
            if state.energy == 0:
                best_assignment = tuple(state.colors)
                best_quotient_energy = 0
                break

        elapsed = time.perf_counter() - started
        if best_assignment is None:
            best_assignment = tuple(0 for _ in range(vertex_count))
        # Encode the base assignment through the same strict decoder used by SAT.
        if problem.decode_spec.encoding.startswith("onehot"):
            model = [
                (vertex * colors + color + 1)
                * (1 if best_assignment[vertex] == color else -1)
                for vertex in range(vertex_count)
                for color in range(colors)
            ]
        else:
            bits = (colors - 1).bit_length()
            model = [
                (vertex * bits + bit + 1)
                * (1 if (best_assignment[vertex] >> bit) & 1 else -1)
                for vertex in range(vertex_count)
                for bit in range(bits)
            ]
        best_coloring = problem.decode_model(model)
        full_energy = conflict_count(problem.instance, best_coloring)
        verification = problem.verify(best_coloring)
        metadata.update(
            {
                "completed_restarts": completed_restarts,
                "steps": total_steps,
                "quotient_best_energy": best_quotient_energy,
                "best_energy": full_energy,
                "best_assignment": list(best_assignment),
                "model_parsing": "direct_assignment",
                "verification": "valid" if verification.valid else "positive_energy",
            }
        )
        if full_energy == 0 and verification.valid:
            return SolveResult(
                SolveStatus.FOUND_WITNESS,
                problem.scope,
                elapsed,
                self.name,
                best_coloring,
                metadata,
                best_coloring=best_coloring,
                best_energy=0,
            )
        return SolveResult(
            SolveStatus.TIMEOUT if time.perf_counter() >= deadline else SolveStatus.UNKNOWN,
            problem.scope,
            elapsed,
            self.name,
            None,
            metadata,
            best_coloring=best_coloring,
            best_energy=full_energy,
        )
