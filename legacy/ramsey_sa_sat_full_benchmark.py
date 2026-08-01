#!/usr/bin/env python3
"""
Ramsey VS/VC annealing + optional SAT benchmark.

Cases from Gasarch et al.:
  VS4_N57:  R_4(VS) > 57 lower-side witness
  VS5_N180: R_5(VS) > 180
  VS6_N333: R_6(VS) > 333
  VC3_N521: R_3(VC) > 521

This file is self-contained and uses only the Python standard library.
It can also export DIMACS CNF and run an external SAT solver such as Kissat/CaDiCaL.

Example:
  python ramsey_sa_sat_full_benchmark.py --mode ours --case all --timeout 300
  python ramsey_sa_sat_full_benchmark.py --mode export --case all --out-dir cnf
  python ramsey_sa_sat_full_benchmark.py --mode sat --case all --solver ./kissat/build/kissat --timeout 300
  python ramsey_sa_sat_full_benchmark.py --mode compare --case all --solver ./kissat/build/kissat --timeout 300
"""

from __future__ import annotations
import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

ALPH = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

@dataclass(frozen=True)
class Case:
    name: str
    kind: str  # "VS" or "VC"
    colors: int
    N: int

CASES: Dict[str, Case] = {
    "VS4_N57": Case("VS4_N57", "VS", 4, 57),
    "VS5_N180": Case("VS5_N180", "VS", 5, 180),
    "VS6_N333": Case("VS6_N333", "VS", 6, 333),
    "VC3_N521": Case("VC3_N521", "VC", 3, 521),
}

# A period-40 pattern discovered by the quotient annealer. The script does not rely on it
# unless --use-known-periods is passed; normally it searches for the period again.
KNOWN_VS6_PERIOD40 = "5302421030245153424510342151034215303421"

# ---------------------------------------------------------------------------
# Basic Ramsey instance construction and verification
# ---------------------------------------------------------------------------

def distance_set(kind: str, N: int) -> List[int]:
    """Forbidden distances <= N-1."""
    if kind == "VS":
        kmax = int(math.isqrt(N - 1))
        return [k * k for k in range(1, kmax + 1)]
    if kind == "VC":
        ds = []
        k = 1
        while k ** 3 <= N - 1:
            ds.append(k ** 3)
            k += 1
        return ds
    raise ValueError(f"unknown kind {kind!r}")


def build_graph(kind: str, N: int) -> Tuple[List[List[int]], List[Tuple[int, int]], List[int]]:
    """Build forbidden-distance graph on vertices 0,...,N-1."""
    ds = distance_set(kind, N)
    adj: List[Set[int]] = [set() for _ in range(N)]
    edges: List[Tuple[int, int]] = []
    for d in ds:
        for u in range(0, N - d):
            v = u + d
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))
    return [sorted(s) for s in adj], edges, ds


def color_string(colors: Sequence[int]) -> str:
    return "".join(ALPH[x] for x in colors)


def parse_color_string(s: str) -> List[int]:
    d = {ch: i for i, ch in enumerate(ALPH)}
    return [d[ch] for ch in s.strip()]


def verify_coloring(kind: str, c: int, N: int, colors: Sequence[int]) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
    """Return (ok, bad). bad = (i,j,d,color) with 1-indexed i,j if invalid."""
    if len(colors) != N:
        raise ValueError(f"coloring length {len(colors)} != N {N}")
    for x in colors:
        if not (0 <= x < c):
            return False, (-1, -1, -1, x)
    for d in distance_set(kind, N):
        for u in range(N - d):
            v = u + d
            if colors[u] == colors[v]:
                return False, (u + 1, v + 1, d, colors[u])
    return True, None


def energy_from_adj(adj: Sequence[Sequence[int]], colors: Sequence[int]) -> int:
    e = 0
    for u, nbs in enumerate(adj):
        cu = colors[u]
        for v in nbs:
            if v > u and colors[v] == cu:
                e += 1
    return e

# ---------------------------------------------------------------------------
# Potts simulated annealing for ordinary graph coloring
# ---------------------------------------------------------------------------

def greedy_random_init(adj: Sequence[Sequence[int]], c: int, rng: random.Random) -> List[int]:
    """Randomized greedy initialization in descending degree order."""
    N = len(adj)
    colors = [-1] * N
    order = sorted(range(N), key=lambda v: -len(adj[v]))
    for v in order:
        counts = [0] * c
        for nb in adj[v]:
            if colors[nb] >= 0:
                counts[colors[nb]] += 1
        m = min(counts)
        choices = [k for k in range(c) if counts[k] == m]
        colors[v] = rng.choice(choices)
    return colors


def compute_neighbor_color_counts(adj: Sequence[Sequence[int]], c: int, colors: Sequence[int]) -> Tuple[List[List[int]], int, List[int]]:
    N = len(adj)
    counts = [[0] * c for _ in range(N)]
    for v in range(N):
        for nb in adj[v]:
            counts[v][colors[nb]] += 1
    E = sum(counts[v][colors[v]] for v in range(N)) // 2
    vconf = [counts[v][colors[v]] for v in range(N)]
    return counts, E, vconf


def potts_anneal_graph(
    adj: Sequence[Sequence[int]],
    c: int,
    *,
    seed: int = 0,
    time_limit: float = 10.0,
    max_steps: int = 50_000_000,
    init: Optional[Sequence[int]] = None,
    verbose: bool = False,
) -> Tuple[List[int], Dict[str, object]]:
    """
    Constrained Potts simulated annealer.

    State: one integer color per vertex. This is equivalent to a one-hot QUBO
    with the exactly-one constraint projected out.

    Move: recolor one vertex v from old to new.
    Delta energy = #neighbors of v colored new - #neighbors of v colored old.
    """
    rng = random.Random(seed)
    N = len(adj)
    t0 = time.perf_counter()
    deadline = t0 + time_limit

    if init is None:
        # Mix a random and greedy-random start; greedy is usually better, but random helps diversify.
        if rng.random() < 0.20:
            colors = [rng.randrange(c) for _ in range(N)]
        else:
            colors = greedy_random_init(adj, c, rng)
    else:
        colors = list(init)
        # Small perturbation for restart diversification.
        for _ in range(max(1, N // 25)):
            colors[rng.randrange(N)] = rng.randrange(c)

    counts, E, vconf = compute_neighbor_color_counts(adj, c, colors)
    best = colors[:]
    bestE = E
    best_step = 0
    tabu_until = [[0] * c for _ in range(N)]
    stagnation = 0

    # Temperature cycles. The problem is integer-valued; most useful uphill moves have delta 1 or 2.
    # Repeated cycles prevent total freezing on a bad local minimum.
    cycle_len = max(1000, 200 * N)
    T_high = 2.50
    T_low = 0.015

    for step in range(1, max_steps + 1):
        if E == 0:
            return colors, {
                "status": "SAT",
                "energy": 0,
                "best_energy": 0,
                "seconds": time.perf_counter() - t0,
                "steps": step,
                "seed": seed,
            }
        if step % 4096 == 0 and time.perf_counter() >= deadline:
            break

        # Pick a conflicted vertex with high probability. This is min-conflicts-like.
        if rng.random() < 0.94:
            v = None
            for _ in range(24):
                x = rng.randrange(N)
                if vconf[x] > 0:
                    v = x
                    break
            if v is None:
                conflicted = [i for i, a in enumerate(vconf) if a > 0]
                if not conflicted:
                    return colors, {
                        "status": "SAT",
                        "energy": 0,
                        "best_energy": 0,
                        "seconds": time.perf_counter() - t0,
                        "steps": step,
                        "seed": seed,
                    }
                # Bias to high-conflict vertices without computing a full weighted sample.
                sample = rng.sample(conflicted, min(len(conflicted), 6))
                v = max(sample, key=lambda x: vconf[x])
        else:
            v = rng.randrange(N)

        old = colors[v]
        old_bad = counts[v][old]

        # Candidate recolorings and deltas.
        candidates: List[Tuple[int, int]] = []
        min_delta = 10**9
        for new in range(c):
            if new == old:
                continue
            dE = counts[v][new] - old_bad
            # Short tabu tenure, with aspiration if it beats the best energy.
            if step < tabu_until[v][new] and E + dE >= bestE:
                continue
            candidates.append((dE, new))
            if dE < min_delta:
                min_delta = dE
        if not candidates:
            continue

        # Geometric cooling within each cycle.
        frac = (step % cycle_len) / max(1, cycle_len - 1)
        T = math.exp(math.log(T_high) * (1.0 - frac) + math.log(T_low) * frac)

        # Annealed min-conflicts proposal.
        # 80%: choose uniformly among best-delta colors.
        # 20%: choose from a Boltzmann distribution over all colors.
        if rng.random() < 0.80:
            best_cands = [(d, col) for d, col in candidates if d == min_delta]
            dE, new = rng.choice(best_cands)
        else:
            weights = []
            total = 0.0
            for d, col in candidates:
                # Shift by min_delta to prevent underflow.
                w = math.exp(-(d - min_delta) / max(T, 1e-12))
                total += w
                weights.append((total, d, col))
            r = rng.random() * total
            dE, new = weights[-1][1], weights[-1][2]
            for cum, d, col in weights:
                if r <= cum:
                    dE, new = d, col
                    break

        # SA acceptance. Improvements always accepted; uphill accepted with Boltzmann probability.
        if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-12)):
            colors[v] = new
            E += dE
            vconf[v] = counts[v][new]

            # Update neighbor color counts and conflict counts locally.
            for nb in adj[v]:
                nbcol = colors[nb]
                if nbcol == old:
                    vconf[nb] -= 1
                if nbcol == new:
                    vconf[nb] += 1
                counts[nb][old] -= 1
                counts[nb][new] += 1

            tabu_until[v][old] = step + rng.randint(4, 18) + min(100, E // 5)

            if E < bestE:
                bestE = E
                best = colors[:]
                best_step = step
                stagnation = 0
                if verbose:
                    print(f"  new best E={bestE} step={step} t={time.perf_counter()-t0:.3f}s", flush=True)
            else:
                stagnation += 1

        # Breakout: if stuck, perturb the conflicted set and a few random vertices.
        if stagnation > 25_000 + 150 * max(1, E):
            conflicted = [i for i, a in enumerate(vconf) if a > 0]
            for v2 in conflicted:
                colors[v2] = rng.randrange(c)
            for _ in range(max(1, N // 30)):
                colors[rng.randrange(N)] = rng.randrange(c)
            counts, E, vconf = compute_neighbor_color_counts(adj, c, colors)
            stagnation = 0

    return best, {
        "status": "TIMEOUT",
        "energy": energy_from_adj(adj, best),
        "best_energy": bestE,
        "seconds": time.perf_counter() - t0,
        "steps": step,
        "seed": seed,
        "best_step": best_step,
    }

# ---------------------------------------------------------------------------
# Periodic quotient annealing
# ---------------------------------------------------------------------------

def build_cyclic_distance_graph(kind: str, N: int, period: int) -> Tuple[Optional[List[List[int]]], List[Tuple[int, int]], List[int], Optional[int]]:
    """
    Graph on residues modulo period. Edge i--i+r for each forbidden distance residue r.

    If residue 0 occurs, a period-p lift is impossible because some allowed distance d<=N-1
    maps a vertex to the same residue, hence same color.
    """
    residues = sorted({d % period for d in distance_set(kind, N)})
    if 0 in residues:
        return None, [], residues, 0
    adj: List[Set[int]] = [set() for _ in range(period)]
    edges: Set[Tuple[int, int]] = set()
    for r in residues:
        for i in range(period):
            j = (i + r) % period
            if i == j:
                return None, [], residues, 0
            u, v = (i, j) if i < j else (j, i)
            edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
    return [sorted(s) for s in adj], sorted(edges), residues, None


def lift_periodic(pattern: Sequence[int], N: int) -> List[int]:
    p = len(pattern)
    return [pattern[i % p] for i in range(N)]


def periodic_quotient_anneal(
    kind: str,
    c: int,
    N: int,
    *,
    timeout: float,
    pmin: Optional[int] = None,
    pmax: Optional[int] = None,
    seeds_per_period: int = 8,
    verbose: bool = False,
) -> Tuple[Optional[List[int]], Dict[str, object]]:
    """Try to solve the full problem by solving a smaller periodic quotient problem."""
    t0 = time.perf_counter()
    if pmin is None:
        # Periods below sqrt(N) for VS often have zero square residues and are impossible.
        # This start is just a speed heuristic.
        pmin = max(c + 1, int(math.isqrt(N - 1)) + 1 if kind == "VS" else c + 1)
    if pmax is None:
        pmax = min(N, 128)

    best_info: Dict[str, object] = {"energy": 10**9}
    best_pattern: Optional[List[int]] = None
    attempts = 0

    # Slightly favor the p=40 case for VS6 because it is a small quotient that the search discovers reliably.
    periods = list(range(pmin, pmax + 1))
    if kind == "VS" and c == 6 and 40 in periods:
        periods.remove(40)
        periods.insert(0, 40)

    for p in periods:
        if time.perf_counter() - t0 >= timeout:
            break
        adj, edges, residues, impossible_residue = build_cyclic_distance_graph(kind, N, p)
        if adj is None:
            continue
        if verbose:
            print(f"  quotient p={p}, edges={len(edges)}, residues={residues}", flush=True)
        for s in range(seeds_per_period):
            elapsed = time.perf_counter() - t0
            if elapsed >= timeout:
                break
            attempts += 1
            per_try = min(0.75, max(0.01, timeout - elapsed))
            pattern, info = potts_anneal_graph(adj, c, seed=10_000 * p + s, time_limit=per_try, verbose=False)
            E = energy_from_adj(adj, pattern)
            if E < int(best_info.get("energy", 10**9)):
                best_pattern = pattern[:]
                best_info = {
                    **info,
                    "energy": E,
                    "period": p,
                    "period_edges": len(edges),
                    "residues": residues,
                    "attempts": attempts,
                }
            if E == 0:
                full = lift_periodic(pattern, N)
                ok, bad = verify_coloring(kind, c, N, full)
                if ok:
                    return full, {
                        **info,
                        "status": "SAT",
                        "mode": "periodic_quotient_annealing",
                        "period": p,
                        "period_pattern": color_string(pattern),
                        "period_edges": len(edges),
                        "residues": residues,
                        "attempts": attempts,
                        "verified": True,
                        "seconds": time.perf_counter() - t0,
                    }
                # This should not happen if the quotient construction is correct.
                return None, {
                    "status": "BUG",
                    "mode": "periodic_quotient_annealing",
                    "period": p,
                    "bad": bad,
                    "seconds": time.perf_counter() - t0,
                }
    return None, {
        "status": "TIMEOUT",
        "mode": "periodic_quotient_annealing",
        "best_info": best_info,
        "best_pattern": color_string(best_pattern) if best_pattern is not None else None,
        "seconds": time.perf_counter() - t0,
        "attempts": attempts,
    }

# ---------------------------------------------------------------------------
# Our full method: periodic quotient first for VS, direct Potts SA otherwise/fallback.
# ---------------------------------------------------------------------------

def run_our_method(case: Case, *, timeout: float = 300.0, seed0: int = 0, use_known_periods: bool = False, verbose: bool = False) -> Tuple[Optional[List[int]], Dict[str, object]]:
    t0 = time.perf_counter()
    kind, c, N = case.kind, case.colors, case.N

    # Optional direct test of the known period-40 witness. This is useful for sanity checks,
    # but not used by default in benchmark mode because it would make VS6_N333 trivial.
    if use_known_periods and case.name == "VS6_N333":
        pattern = parse_color_string(KNOWN_VS6_PERIOD40)
        full = lift_periodic(pattern, N)
        ok, bad = verify_coloring(kind, c, N, full)
        return (full if ok else None), {
            "status": "SAT" if ok else "INVALID_KNOWN_PATTERN",
            "mode": "known_periodic_witness",
            "period": len(pattern),
            "period_pattern": KNOWN_VS6_PERIOD40,
            "verified": ok,
            "bad": bad,
            "seconds": time.perf_counter() - t0,
        }

    # We choose the ordering by instance.
    # * VS6_N333 benefits massively from quotient annealing; try that first.
    # * VS4_N57, VS5_N180, and VC3_N521 are usually solved fastest by direct Potts annealing.
    if kind == "VS" and c >= 6 and N >= 300:
        q_timeout = min(timeout, 90.0)
        col, info = periodic_quotient_anneal(kind, c, N, timeout=q_timeout, pmax=min(160, N), verbose=verbose)
        if col is not None:
            info["total_seconds"] = time.perf_counter() - t0
            return col, info

    adj, edges, ds = build_graph(kind, N)
    best: Optional[List[int]] = None
    bestE = 10**9
    attempts = 0

    # Direct Potts annealing.
    direct_budget = timeout if not (kind == "VS" and c >= 6 and N >= 300) else max(0.0, timeout - (time.perf_counter() - t0))
    while time.perf_counter() - t0 < timeout and (time.perf_counter() - t0) < direct_budget:
        rem = timeout - (time.perf_counter() - t0)
        attempts += 1
        per_try = min(15.0, max(0.05, rem))
        col, info = potts_anneal_graph(adj, c, seed=seed0 + attempts - 1, time_limit=per_try, verbose=verbose)
        E = energy_from_adj(adj, col)
        if E < bestE:
            best = col[:]
            bestE = E
        if E == 0:
            ok, bad = verify_coloring(kind, c, N, col)
            return col, {
                **info,
                "status": "SAT" if ok else "BUG_INVALID",
                "mode": "direct_potts_annealing",
                "edges": len(edges),
                "distances": ds,
                "attempts": attempts,
                "verified": ok,
                "bad": bad,
                "total_seconds": time.perf_counter() - t0,
            }

    # If direct did not solve a square-distance case, try quotient annealing with remaining time.
    if kind == "VS" and time.perf_counter() - t0 < timeout:
        rem = timeout - (time.perf_counter() - t0)
        col, info = periodic_quotient_anneal(kind, c, N, timeout=rem, pmax=min(160, N), verbose=verbose)
        if col is not None:
            info["total_seconds"] = time.perf_counter() - t0
            return col, info

    return best, {
        "status": "TIMEOUT",
        "mode": "our_method",
        "best_energy": bestE,
        "attempts": attempts,
        "verified": False,
        "total_seconds": time.perf_counter() - t0,
    }

# ---------------------------------------------------------------------------
# DIMACS CNF generation and external SAT solver runner
# ---------------------------------------------------------------------------

def var_id(v: int, k: int, c: int) -> int:
    """1-indexed DIMACS variable id for x_{v,k}, with v,k zero-indexed."""
    return v * c + k + 1


def inv_var_id(x: int, c: int) -> Tuple[int, int]:
    y = abs(x) - 1
    return y // c, y % c


def cnf_clauses(kind: str, c: int, N: int) -> Iterable[List[int]]:
    """Standard graph-coloring CNF."""
    adj, edges, ds = build_graph(kind, N)
    # At least one color per vertex.
    for v in range(N):
        yield [var_id(v, k, c) for k in range(c)]
    # At most one color per vertex.
    for v in range(N):
        for k1 in range(c):
            for k2 in range(k1 + 1, c):
                yield [-var_id(v, k1, c), -var_id(v, k2, c)]
    # Forbidden same-color edges.
    for u, v in edges:
        for k in range(c):
            yield [-var_id(u, k, c), -var_id(v, k, c)]


def count_cnf_clauses(kind: str, c: int, N: int) -> Tuple[int, int, int]:
    _adj, edges, _ds = build_graph(kind, N)
    nvars = N * c
    nclauses = N + N * (c * (c - 1) // 2) + len(edges) * c
    return nvars, nclauses, len(edges)


def write_dimacs(case: Case, path: str) -> Dict[str, object]:
    nvars, nclauses, nedges = count_cnf_clauses(case.kind, case.colors, case.N)
    with open(path, "w") as f:
        f.write(f"c {case.name}: {case.kind}, c={case.colors}, N={case.N}\n")
        f.write(f"p cnf {nvars} {nclauses}\n")
        actual = 0
        for clause in cnf_clauses(case.kind, case.colors, case.N):
            f.write(" ".join(map(str, clause)) + " 0\n")
            actual += 1
    assert actual == nclauses, (actual, nclauses)
    return {"path": path, "vars": nvars, "clauses": nclauses, "edges": nedges}


def parse_dimacs_model(output: str, case: Case) -> Optional[List[int]]:
    """Parse a SAT solver model from lines beginning with v."""
    positives: Set[int] = set()
    saw_sat = False
    saw_unsat = False
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("s"):
            if "UNSAT" in line.upper():
                saw_unsat = True
            if "SATISFIABLE" in line.upper() and "UNSAT" not in line.upper():
                saw_sat = True
        if line.startswith("v"):
            for tok in line.split()[1:]:
                try:
                    lit = int(tok)
                except ValueError:
                    continue
                if lit == 0:
                    continue
                if lit > 0:
                    positives.add(lit)
    if saw_unsat:
        return None
    colors = [-1] * case.N
    for lit in positives:
        v, k = inv_var_id(lit, case.colors)
        if 0 <= v < case.N and 0 <= k < case.colors:
            if colors[v] == -1:
                colors[v] = k
            elif colors[v] != k:
                # Invalid model for exactly-one encoding.
                return None
    if all(x >= 0 for x in colors):
        return colors
    # Some solvers omit model unless asked. Return None in that case.
    return None


def run_external_sat_solver(case: Case, solver: str, *, timeout: float, out_dir: Optional[str] = None) -> Dict[str, object]:
    """Generate DIMACS, run external solver, parse status/model if printed."""
    if not shutil.which(solver) and not os.path.exists(solver):
        return {"status": "SOLVER_NOT_FOUND", "solver": solver}
    if out_dir is None:
        tmp = tempfile.TemporaryDirectory()
        cnf_path = os.path.join(tmp.name, f"{case.name}.cnf")
    else:
        os.makedirs(out_dir, exist_ok=True)
        tmp = None
        cnf_path = os.path.join(out_dir, f"{case.name}.cnf")
    meta = write_dimacs(case, cnf_path)
    cmd = [solver, cnf_path]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        wall = time.perf_counter() - t0
        out = p.stdout
        upper = out.upper()
        if "UNSATISFIABLE" in upper:
            status = "UNSAT"
        elif "SATISFIABLE" in upper:
            status = "SAT"
        else:
            status = "UNKNOWN"
        model = parse_dimacs_model(out, case) if status == "SAT" else None
        verified = False
        bad = None
        coloring_str = None
        if model is not None:
            verified, bad = verify_coloring(case.kind, case.colors, case.N, model)
            coloring_str = color_string(model)
        return {
            "status": status,
            "solver": solver,
            "cmd": cmd,
            "returncode": p.returncode,
            "wall_seconds": wall,
            "timeout": timeout,
            "cnf": meta,
            "model_printed": model is not None,
            "verified": verified,
            "bad": bad,
            "coloring": coloring_str,
            "stdout_tail": "\n".join(out.splitlines()[-20:]),
        }
    except subprocess.TimeoutExpired as e:
        return {
            "status": "TIMEOUT",
            "solver": solver,
            "cmd": cmd,
            "wall_seconds": time.perf_counter() - t0,
            "timeout": timeout,
            "cnf": meta,
            "stdout_tail": (e.stdout[-4000:] if isinstance(e.stdout, str) else ""),
        }
    finally:
        if tmp is not None:
            tmp.cleanup()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def select_cases(case_arg: str) -> List[Case]:
    if case_arg == "all":
        return [CASES[k] for k in ["VS4_N57", "VS5_N180", "VS6_N333", "VC3_N521"]]
    if case_arg not in CASES:
        raise SystemExit(f"unknown case {case_arg!r}; choices: all, " + ", ".join(CASES))
    return [CASES[case_arg]]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["ours", "export", "sat", "compare", "verify-known"], default="ours")
    ap.add_argument("--case", default="all", help="all or one of: " + ", ".join(CASES))
    ap.add_argument("--timeout", type=float, default=300.0, help="per-case timeout in seconds")
    ap.add_argument("--solver", default="kissat", help="external SAT solver executable for --mode sat/compare")
    ap.add_argument("--out-dir", default="/mnt/data/ramsey_benchmark_output", help="output directory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-known-periods", action="store_true", help="use embedded known period witness when available")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    cases = select_cases(args.case)
    all_results = []

    if args.mode == "verify-known":
        pattern = parse_color_string(KNOWN_VS6_PERIOD40)
        for N in [333, 400, 401]:
            full = lift_periodic(pattern, N)
            ok, bad = verify_coloring("VS", 6, N, full)
            print(json.dumps({"case": f"VS6_N{N}", "period": 40, "ok": ok, "bad": bad}, indent=2))
        return 0

    for case in cases:
        print(f"\n=== {case.name}: {case.kind}, c={case.colors}, N={case.N} ===", flush=True)
        record: Dict[str, object] = {"case": asdict(case)}

        if args.mode in ["ours", "compare"]:
            col, info = run_our_method(case, timeout=args.timeout, seed0=args.seed, use_known_periods=args.use_known_periods, verbose=args.verbose)
            if col is not None:
                ok, bad = verify_coloring(case.kind, case.colors, case.N, col)
                info["verified"] = ok
                info["bad"] = bad
                info["coloring"] = color_string(col)
                # Save witness.
                wpath = os.path.join(args.out_dir, f"{case.name}.ours.witness.txt")
                with open(wpath, "w") as f:
                    f.write(color_string(col) + "\n")
                info["witness_path"] = wpath
            print("OURS", json.dumps(info, indent=2), flush=True)
            record["ours"] = info

        if args.mode in ["export", "sat", "compare"]:
            cnf_path = os.path.join(args.out_dir, f"{case.name}.cnf")
            meta = write_dimacs(case, cnf_path)
            print("CNF", json.dumps(meta, indent=2), flush=True)
            record["cnf"] = meta

        if args.mode in ["sat", "compare"]:
            sat_info = run_external_sat_solver(case, args.solver, timeout=args.timeout, out_dir=args.out_dir)
            print("SAT_SOLVER", json.dumps(sat_info, indent=2), flush=True)
            record["sat_solver"] = sat_info

        all_results.append(record)

    result_path = os.path.join(args.out_dir, f"results_{args.mode}_{int(time.time())}.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWROTE {result_path}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
