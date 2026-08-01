#!/usr/bin/env python3
"""
Ramsey polynomial van der Waerden experiment matrix.

This driver reuses the Potts annealer and twisted quotient code from the earlier
files in this package:
  - ramsey_sa_sat_full_benchmark.py
  - ramsey_periodic_twist_explorer.py

It benchmarks, with a PER-RUN timeout, the following structure choices:
  direct   : color all N integers directly
  period   : ordinary residue quotient color(n)=a[n mod p]
  twist    : affine/twisted quotient color(qp+r)=a[r]+g*q mod c
  defect   : NEW proposed trick: periodic/twisted backbone + sparse defect repair

and the following engines:
  potts    : simulated annealing only
  kissat   : plain external SAT on the corresponding CNF/reduced CNF
  hybrid   : Potts first; if nonzero, freeze most variables and ask Kissat to repair

The output is JSONL plus a CSV summary and witness files.
"""
from __future__ import annotations

import argparse
import csv
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

# Keep this package self-contained: run from the directory containing the previous scripts,
# or put that directory on PYTHONPATH.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if "/mnt/data" not in sys.path:
    sys.path.insert(0, "/mnt/data")

from ramsey_sa_sat_full_benchmark import (  # type: ignore
    ALPH,
    Case as OldCase,
    build_graph,
    build_cyclic_distance_graph,
    cnf_clauses,
    color_string,
    count_cnf_clauses,
    distance_set,
    energy_from_adj,
    lift_periodic,
    parse_color_string,
    potts_anneal_graph,
    verify_coloring,
    var_id,
    inv_var_id,
)
from ramsey_periodic_twist_explorer import (  # type: ignore
    build_twisted_constraints,
    lift_twisted,
    twisted_anneal,
)

# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchCase:
    name: str
    kind: str       # VS or VC
    c: int
    N: int
    note: str = ""

# Default benchmark targets.  For c=3..8, these include the Gasarch table entries
# for VS where available and the quotient-generated candidate lower witnesses we found.
DEFAULT_CASES: Dict[str, BenchCase] = {
    # Square distances: exact/lower-side known from Gasarch for c<=6; ours for c>=6/7/8.
    "VS3_N28": BenchCase("VS3_N28", "VS", 3, 28, "lower side of R_3(VS)=29"),
    "VS4_N57": BenchCase("VS4_N57", "VS", 4, 57, "lower side of R_4(VS)=58"),
    "VS5_N180": BenchCase("VS5_N180", "VS", 5, 180, "Gasarch lower bound"),
    "VS6_N333": BenchCase("VS6_N333", "VS", 6, 333, "Gasarch lower bound"),
    "VS6_N400": BenchCase("VS6_N400", "VS", 6, 400, "period-40 quotient witness found by us"),
    "VS7_N576": BenchCase("VS7_N576", "VS", 7, 576, "ordinary quotient witness found by us"),
    "VS8_N841": BenchCase("VS8_N841", "VS", 8, 841, "ordinary quotient witness found by us"),

    # Cube distances: Gasarch gives R_3(VC)>521; the rest are quotient-generated targets from our scan.
    "VC3_N521": BenchCase("VC3_N521", "VC", 3, 521, "Gasarch lower bound"),
    "VC4_N2197": BenchCase("VC4_N2197", "VC", 4, 2197, "period-13 quotient witness found by us"),
    "VC5_N9261": BenchCase("VC5_N9261", "VC", 5, 9261, "period-63 quotient witness found by us"),
    "VC6_N50653": BenchCase("VC6_N50653", "VC", 6, 50653, "period-37 quotient witness found by us"),
    "VC7_N79507": BenchCase("VC7_N79507", "VC", 7, 79507, "period-43 quotient witness found by us"),
    "VC8_N753571": BenchCase("VC8_N753571", "VC", 8, 753571, "period-91 quotient witness found by us"),
}

# Good periods discovered in earlier scans.  These are used first in guided mode,
# but the script can also scan blindly with --period-policy blind.
GUIDED_PERIODS: Dict[Tuple[str, int, int], List[int]] = {
    ("VS", 6, 333): [40],
    ("VS", 6, 400): [40],
    ("VS", 7, 576): [96, 24, 48],
    ("VS", 8, 841): [29, 58, 87],
    ("VS", 9, 1681): [41],
    ("VS", 10, 4225): [65],
    ("VC", 3, 521): [49, 37, 43],
    ("VC", 4, 2197): [13],
    ("VC", 5, 9261): [63],
    ("VC", 6, 50653): [37],
    ("VC", 7, 79507): [43],
    ("VC", 8, 753571): [91],
}

# Good twist seeds discovered in previous exploration.  The general twist scanner will
# also search if these do not work.
GUIDED_TWISTS: Dict[Tuple[str, int, int], List[Tuple[int, int]]] = {
    ("VS", 6, 333): [(20, 3), (40, 0)],
    ("VS", 6, 400): [(20, 3), (40, 0)],
    ("VS", 10, 4225): [(13, 2)],
}

# ---------------------------------------------------------------------------
# Utility and verification
# ---------------------------------------------------------------------------

def solver_exists(path: str) -> bool:
    return bool(shutil.which(path) or os.path.exists(path))


def now() -> float:
    return time.perf_counter()


def estimate_full(case: BenchCase) -> Dict[str, int]:
    nvars, nclauses, nedges = count_cnf_clauses(case.kind, case.c, case.N)
    return {"vars": nvars, "clauses": nclauses, "edges": nedges}


def safe_case_as_old(case: BenchCase) -> OldCase:
    return OldCase(case.name, case.kind, case.c, case.N)


def save_witness(out_dir: str, case: BenchCase, tag: str, coloring: Sequence[int], extra: Optional[Dict[str, object]]=None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{case.name}.{tag}.witness.txt")
    with open(path, "w") as f:
        f.write(color_string(coloring) + "\n")
        if extra:
            f.write("# " + json.dumps(extra, sort_keys=True) + "\n")
    return path

# ---------------------------------------------------------------------------
# DIMACS helpers
# ---------------------------------------------------------------------------

def write_full_dimacs(case: BenchCase, path: str) -> Dict[str, object]:
    nvars, nclauses, nedges = count_cnf_clauses(case.kind, case.c, case.N)
    with open(path, "w") as f:
        f.write(f"c full {case.name}: {case.kind}, c={case.c}, N={case.N}\n")
        f.write(f"p cnf {nvars} {nclauses}\n")
        for clause in cnf_clauses(case.kind, case.c, case.N):
            f.write(" ".join(map(str, clause)) + " 0\n")
    return {"path": path, "vars": nvars, "clauses": nclauses, "edges": nedges}


def qvar(v: int, k: int, c: int) -> int:
    return v * c + k + 1


def parse_model_from_output(output: str, nvertices: int, c: int) -> Optional[List[int]]:
    positives: Set[int] = set()
    saw_unsat = False
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("s") and "UNSAT" in line.upper():
            saw_unsat = True
        if line.startswith("v"):
            for tok in line.split()[1:]:
                try:
                    lit = int(tok)
                except ValueError:
                    continue
                if lit > 0:
                    positives.add(lit)
    if saw_unsat:
        return None
    colors = [-1] * nvertices
    for lit in positives:
        y = lit - 1
        v, k = divmod(y, c)
        if 0 <= v < nvertices and 0 <= k < c:
            if colors[v] == -1:
                colors[v] = k
            elif colors[v] != k:
                return None
    if all(x >= 0 for x in colors):
        return colors
    return None


def run_solver_on_cnf(solver: str, cnf_path: str, timeout: float) -> Dict[str, object]:
    if not solver_exists(solver):
        return {"status": "SOLVER_NOT_FOUND", "solver": solver}
    cmd = [solver, cnf_path]
    t0 = now()
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=max(0.001, timeout))
        wall = now() - t0
        out = p.stdout or ""
        upper = out.upper()
        if "UNSATISFIABLE" in upper:
            status = "UNSAT"
        elif "SATISFIABLE" in upper:
            status = "SAT"
        else:
            status = "UNKNOWN"
        return {
            "status": status, "solver": solver, "cmd": cmd, "returncode": p.returncode,
            "wall_seconds": wall, "stdout_tail": "\n".join(out.splitlines()[-20:]),
            "stdout": out,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "status": "TIMEOUT", "solver": solver, "cmd": cmd, "wall_seconds": now() - t0,
            "stdout_tail": (e.stdout[-4000:] if isinstance(e.stdout, str) else ""),
        }


def write_graph_coloring_cnf(nvertices: int, c: int, edges: Sequence[Tuple[int, int]], path: str,
                             unit_forbids: Optional[Sequence[Tuple[int, int]]] = None) -> Dict[str, object]:
    """CNF for proper c-coloring of an ordinary graph.
    unit_forbids is a list of (v,k) saying vertex v may not use color k.
    """
    unit_forbids = list(unit_forbids or [])
    nvars = nvertices * c
    nclauses = nvertices + nvertices * (c * (c - 1) // 2) + len(edges) * c + len(unit_forbids)
    with open(path, "w") as f:
        f.write(f"p cnf {nvars} {nclauses}\n")
        for v in range(nvertices):
            f.write(" ".join(str(qvar(v, k, c)) for k in range(c)) + " 0\n")
            for k1 in range(c):
                for k2 in range(k1 + 1, c):
                    f.write(f"-{qvar(v,k1,c)} -{qvar(v,k2,c)} 0\n")
        for u, v in edges:
            for k in range(c):
                f.write(f"-{qvar(u,k,c)} -{qvar(v,k,c)} 0\n")
        for v, k in unit_forbids:
            f.write(f"-{qvar(v,k,c)} 0\n")
    return {"path": path, "vars": nvars, "clauses": nclauses, "edges": len(edges), "unit_forbids": len(unit_forbids)}


def write_twist_cnf(p: int, c: int, constraints: Sequence[Tuple[int, int, int]], path: str) -> Dict[str, object]:
    """CNF for labeled constraints a_r != a_j + shift mod c."""
    nvars = p * c
    nclauses = p + p * (c * (c - 1) // 2) + len(constraints) * c
    with open(path, "w") as f:
        f.write(f"p cnf {nvars} {nclauses}\n")
        for v in range(p):
            f.write(" ".join(str(qvar(v, k, c)) for k in range(c)) + " 0\n")
            for k1 in range(c):
                for k2 in range(k1 + 1, c):
                    f.write(f"-{qvar(v,k1,c)} -{qvar(v,k2,c)} 0\n")
        for r, j, sh in constraints:
            for kr in range(c):
                kj = (kr - sh) % c
                f.write(f"-{qvar(r,kr,c)} -{qvar(j,kj,c)} 0\n")
    return {"path": path, "vars": nvars, "clauses": nclauses, "constraints": len(constraints)}

# ---------------------------------------------------------------------------
# Structure candidate generation
# ---------------------------------------------------------------------------

def candidate_periods(case: BenchCase, pmax: int, policy: str) -> List[int]:
    base = list(range(max(case.c + 1, 2), min(case.N, pmax) + 1))
    if policy == "blind":
        return base
    guided = GUIDED_PERIODS.get((case.kind, case.c, case.N), [])
    # Descending first-zero heuristic from the earlier exploration.
    def first_zero(kind: str, p: int) -> int:
        power = 2 if kind == "VS" else 3
        k = 1
        while pow(k, power, p) != 0:
            k += 1
            if k > 10_000:
                break
        return k ** power
    rest = sorted(base, key=lambda p: first_zero(case.kind, p), reverse=True)
    out: List[int] = []
    for p in guided + rest:
        if p in base and p not in out:
            out.append(p)
    return out


def candidate_twists(case: BenchCase, pmax: int, policy: str) -> List[Tuple[int, int]]:
    guided = GUIDED_TWISTS.get((case.kind, case.c, case.N), []) if policy != "blind" else []
    out: List[Tuple[int, int]] = []
    for x in guided:
        if x not in out:
            out.append(x)
    for p in candidate_periods(case, pmax, policy):
        for g in range(1, case.c):
            if (p, g) not in out:
                out.append((p, g))
    return out

# ---------------------------------------------------------------------------
# Direct methods
# ---------------------------------------------------------------------------

def run_direct_potts(case: BenchCase, timeout: float, seed: int, max_edges: int) -> Dict[str, object]:
    est = estimate_full(case)
    if est["edges"] > max_edges:
        return {"status": "SKIPPED_TOO_MANY_EDGES", "estimate": est, "max_edges": max_edges}
    t0 = now()
    adj, edges, ds = build_graph(case.kind, case.N)
    best_col = None; bestE = 10**18; attempts = 0
    while now() - t0 < timeout:
        rem = timeout - (now() - t0)
        attempts += 1
        col, info = potts_anneal_graph(adj, case.c, seed=seed + attempts - 1, time_limit=min(20.0, rem))
        E = energy_from_adj(adj, col)
        if E < bestE:
            bestE = E; best_col = col[:]
        if E == 0:
            ok, bad = verify_coloring(case.kind, case.c, case.N, col)
            return {**info, "status": "SAT" if ok else "BUG_INVALID", "structure": "direct", "engine": "potts",
                    "attempts": attempts, "total_seconds": now() - t0, "verified": ok, "bad": bad,
                    "coloring": color_string(col), "edges": len(edges), "distances": ds}
    return {"status": "TIMEOUT", "structure": "direct", "engine": "potts", "best_energy": bestE,
            "attempts": attempts, "total_seconds": now() - t0, "estimate": est,
            "coloring": color_string(best_col) if best_col else None}


def run_direct_kissat(case: BenchCase, solver: str, timeout: float, out_dir: str, max_clauses: int) -> Dict[str, object]:
    est = estimate_full(case)
    if est["clauses"] > max_clauses:
        return {"status": "SKIPPED_TOO_MANY_CLAUSES", "estimate": est, "max_clauses": max_clauses}
    cnf_path = os.path.join(out_dir, f"{case.name}.direct.cnf")
    meta = write_full_dimacs(case, cnf_path)
    sat = run_solver_on_cnf(solver, cnf_path, timeout)
    model = parse_model_from_output(sat.get("stdout", ""), case.N, case.c) if sat.get("status") == "SAT" else None
    ok = False; bad = None; coloring = None
    if model is not None:
        ok, bad = verify_coloring(case.kind, case.c, case.N, model); coloring = color_string(model)
    sat.pop("stdout", None)
    return {**sat, "structure": "direct", "engine": "kissat", "cnf": meta,
            "verified": ok, "bad": bad, "coloring": coloring}

# ---------------------------------------------------------------------------
# Period methods
# ---------------------------------------------------------------------------

def run_period_potts(case: BenchCase, timeout: float, seed: int, pmax: int, policy: str, per_try: float) -> Dict[str, object]:
    t0 = now(); best = None; bestE = 10**18; attempts = 0
    for p in candidate_periods(case, pmax, policy):
        if now() - t0 >= timeout:
            break
        adj, edges, residues, impossible = build_cyclic_distance_graph(case.kind, case.N, p)
        if adj is None:
            continue
        rem = timeout - (now() - t0)
        attempts += 1
        pat, info = potts_anneal_graph(adj, case.c, seed=seed + 100000*p + attempts, time_limit=min(per_try, rem))
        E = energy_from_adj(adj, pat)
        if E < bestE:
            bestE = E; best = {"period": p, "pattern": color_string(pat), "energy": E, "residues": residues, "period_edges": len(edges)}
        if E == 0:
            full = lift_periodic(pat, case.N)
            ok, bad = verify_coloring(case.kind, case.c, case.N, full)
            return {**info, "status": "SAT" if ok else "BUG_INVALID", "structure": "period", "engine": "potts",
                    "period": p, "period_pattern": color_string(pat), "residues": residues, "period_edges": len(edges),
                    "attempts": attempts, "total_seconds": now() - t0, "verified": ok, "bad": bad,
                    "coloring": color_string(full)}
    return {"status": "TIMEOUT", "structure": "period", "engine": "potts", "best_energy": bestE,
            "best": best, "attempts": attempts, "total_seconds": now() - t0}


def run_period_kissat(case: BenchCase, solver: str, timeout: float, out_dir: str, pmax: int, policy: str) -> Dict[str, object]:
    t0 = now(); attempts = 0; last = None
    for p in candidate_periods(case, pmax, policy):
        rem = timeout - (now() - t0)
        if rem <= 0:
            break
        adj, edges, residues, impossible = build_cyclic_distance_graph(case.kind, case.N, p)
        if adj is None:
            continue
        attempts += 1
        cnf_path = os.path.join(out_dir, f"{case.name}.period_p{p}.cnf")
        meta = write_graph_coloring_cnf(p, case.c, edges, cnf_path)
        sat = run_solver_on_cnf(solver, cnf_path, rem)
        last = {**sat, "period": p, "cnf": meta}
        if sat.get("status") == "SAT":
            pat = parse_model_from_output(sat.get("stdout", ""), p, case.c)
            sat.pop("stdout", None)
            if pat is None:
                return {**sat, "structure": "period", "engine": "kissat", "period": p,
                        "attempts": attempts, "total_seconds": now()-t0, "model_printed": False, "cnf": meta}
            full = lift_periodic(pat, case.N)
            ok, bad = verify_coloring(case.kind, case.c, case.N, full)
            return {**sat, "structure": "period", "engine": "kissat", "period": p,
                    "period_pattern": color_string(pat), "residues": residues, "attempts": attempts,
                    "total_seconds": now()-t0, "verified": ok, "bad": bad, "coloring": color_string(full), "cnf": meta}
        sat.pop("stdout", None)
    return {"status": "TIMEOUT", "structure": "period", "engine": "kissat", "attempts": attempts,
            "total_seconds": now()-t0, "last": last}

# ---------------------------------------------------------------------------
# Twisted methods
# ---------------------------------------------------------------------------

def run_twist_potts(case: BenchCase, timeout: float, seed: int, pmax: int, policy: str, per_try: float) -> Dict[str, object]:
    t0 = now(); attempts = 0; best = None; bestE = 10**18
    for p, g in candidate_twists(case, pmax, policy):
        rem = timeout - (now() - t0)
        if rem <= 0:
            break
        attempts += 1
        pat, info = twisted_anneal(case.kind, case.c, case.N, p, g, seed=seed + 100000*p + 100*g + attempts,
                                   time_limit=min(per_try, rem))
        E = int(info.get("energy", 10**9))
        if E < bestE:
            bestE = E; best = {"period": p, "g": g, "pattern": color_string(pat) if pat else None, "energy": E}
        if pat is not None and info.get("status") == "SAT":
            full = lift_twisted(pat, case.N, case.c, p, g)
            ok, bad = verify_coloring(case.kind, case.c, case.N, full)
            return {**info, "status": "SAT" if ok else "BUG_INVALID", "structure": "twist", "engine": "potts",
                    "period": p, "g": g, "period_pattern": color_string(pat), "effective_period": p * (case.c // math.gcd(case.c, g)),
                    "attempts": attempts, "total_seconds": now() - t0, "verified": ok, "bad": bad,
                    "coloring": color_string(full)}
    return {"status": "TIMEOUT", "structure": "twist", "engine": "potts", "best_energy": bestE,
            "best": best, "attempts": attempts, "total_seconds": now() - t0}


def run_twist_kissat(case: BenchCase, solver: str, timeout: float, out_dir: str, pmax: int, policy: str) -> Dict[str, object]:
    t0 = now(); attempts = 0; last = None
    for p, g in candidate_twists(case, pmax, policy):
        rem = timeout - (now() - t0)
        if rem <= 0:
            break
        constraints, ds, meta0 = build_twisted_constraints(case.kind, case.N, p, case.c, g)
        if constraints is None:
            continue
        attempts += 1
        cnf_path = os.path.join(out_dir, f"{case.name}.twist_p{p}_g{g}.cnf")
        meta = write_twist_cnf(p, case.c, constraints, cnf_path)
        sat = run_solver_on_cnf(solver, cnf_path, rem)
        last = {**sat, "period": p, "g": g, "cnf": meta}
        if sat.get("status") == "SAT":
            pat = parse_model_from_output(sat.get("stdout", ""), p, case.c)
            sat.pop("stdout", None)
            if pat is None:
                return {**sat, "structure": "twist", "engine": "kissat", "period": p, "g": g,
                        "attempts": attempts, "total_seconds": now()-t0, "model_printed": False, "cnf": meta}
            full = lift_twisted(pat, case.N, case.c, p, g)
            ok, bad = verify_coloring(case.kind, case.c, case.N, full)
            return {**sat, "structure": "twist", "engine": "kissat", "period": p, "g": g,
                    "period_pattern": color_string(pat), "attempts": attempts, "total_seconds": now()-t0,
                    "verified": ok, "bad": bad, "coloring": color_string(full), "cnf": meta}
        sat.pop("stdout", None)
    return {"status": "TIMEOUT", "structure": "twist", "engine": "kissat", "attempts": attempts,
            "total_seconds": now()-t0, "last": last}

# ---------------------------------------------------------------------------
# NEW proposed algorithm: defect repair on top of a periodic/twisted backbone
# ---------------------------------------------------------------------------

def defect_free_set(kind: str, N: int, colors: Sequence[int], radius: int) -> Tuple[Set[int], List[Tuple[int,int]], int]:
    adj, edges, ds = build_graph(kind, N)
    bad = [(u, v) for u, v in edges if colors[u] == colors[v]]
    free: Set[int] = set()
    frontier: Set[int] = set()
    for u, v in bad:
        free.add(u); free.add(v); frontier.add(u); frontier.add(v)
    for _ in range(radius):
        new: Set[int] = set()
        for x in frontier:
            new.update(adj[x])
        new -= free
        free.update(new)
        frontier = new
    return free, bad, len(edges)


def build_defect_subproblem(case: BenchCase, base: Sequence[int], radius: int) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    """Return a reduced repair problem with frozen boundary.

    Variables are only the defect/free vertices.  Fixed-fixed conflicts cannot be repaired;
    if they occur, caller should enlarge radius or choose a different backbone.
    """
    adj, edges, ds = build_graph(case.kind, case.N)
    free, bad0, nedges = defect_free_set(case.kind, case.N, base, radius)
    if not bad0:
        return {"already_valid": True, "free": [], "sub_edges": [], "forbids": [], "bad0": [], "nedges": nedges}, None
    free_list = sorted(free)
    idx = {v: i for i, v in enumerate(free_list)}
    free_set = set(free_list)
    sub_edges: Set[Tuple[int, int]] = set()
    forbids: List[Tuple[int, int]] = []
    fixed_bad = []
    for u, v in edges:
        iu, iv = u in free_set, v in free_set
        if iu and iv:
            a, b = idx[u], idx[v]
            if a > b: a, b = b, a
            sub_edges.add((a, b))
        elif iu and not iv:
            forbids.append((idx[u], base[v]))
        elif iv and not iu:
            forbids.append((idx[v], base[u]))
        else:
            if base[u] == base[v]:
                fixed_bad.append((u, v))
    if fixed_bad:
        return None, f"fixed-fixed conflicts remain ({len(fixed_bad)}); enlarge radius"
    return {"already_valid": False, "free": free_list, "sub_edges": sorted(sub_edges), "forbids": forbids,
            "bad0": bad0, "nedges": nedges}, None


def anneal_defect_subproblem(case: BenchCase, sub: Dict[str, object], base: Sequence[int], timeout: float, seed: int) -> Tuple[Optional[List[int]], Dict[str, object]]:
    if sub.get("already_valid"):
        return list(base), {"status": "SAT", "reason": "backbone_already_valid", "free_vertices": 0, "seconds": 0.0}
    free = list(sub["free"])  # type: ignore
    edges = list(sub["sub_edges"])  # type: ignore
    forbids = list(sub["forbids"])  # type: ignore
    m = len(free)
    # Turn frozen boundary forbids into pendant fixed-color penalty vertices would be wasteful;
    # instead, make an augmented graph with c color-forbid penalties by adding c impossible colors.
    # Easier: custom local-search energy for subproblem.
    rng = random.Random(seed)
    t0 = now(); deadline = t0 + timeout
    state = [base[v] for v in free]
    for i in range(m):
        if rng.random() < 0.9:
            state[i] = rng.randrange(case.c)
    adj = [[] for _ in range(m)]
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
    forbid_counts = [[0] * case.c for _ in range(m)]
    for i, k in forbids:
        forbid_counts[i][k] += 1
    def compute():
        vconf = [forbid_counts[i][state[i]] for i in range(m)]
        E = sum(vconf)
        for a, b in edges:
            if state[a] == state[b]:
                E += 1; vconf[a] += 1; vconf[b] += 1
        return E, vconf
    E, vconf = compute(); best = state[:]; bestE = E
    step = 0; tabu = [[0]*case.c for _ in range(m)]; cycle=max(1000,200*max(1,m))
    T_high,T_low=2.5,0.015; stagn=0
    while now() < deadline:
        step += 1
        if E == 0:
            out = list(base)
            for v, i in zip(free, range(m)):
                out[v] = state[i]
            ok, bad = verify_coloring(case.kind, case.c, case.N, out)
            return out, {"status": "SAT" if ok else "BUG_INVALID", "free_vertices": m,
                         "initial_bad_edges": len(sub.get("bad0", [])), "seconds": now()-t0,
                         "steps": step, "verified": ok, "bad": bad}
        if rng.random() < 0.95:
            cand = [i for i,a in enumerate(vconf) if a > 0]
            i = max(rng.sample(cand, min(len(cand), 6)), key=lambda x: vconf[x]) if cand else rng.randrange(m)
        else:
            i = rng.randrange(m)
        old = state[i]
        oldbad = vconf[i]
        opts=[]; md=10**9
        for new in range(case.c):
            if new == old: continue
            newbad = forbid_counts[i][new] + sum(1 for j in adj[i] if state[j] == new)
            dE = newbad - oldbad
            if step < tabu[i][new] and E + dE >= bestE:
                continue
            opts.append((dE,new)); md=min(md,dE)
        if not opts:
            continue
        frac=(step%cycle)/max(1,cycle-1)
        T=math.exp(math.log(T_high)*(1-frac)+math.log(T_low)*frac)
        if rng.random()<0.8:
            dE,new=rng.choice([x for x in opts if x[0]==md])
        else:
            wheel=[]; total=0.0
            for d,nc in opts:
                w=math.exp(-(d-md)/max(T,1e-12)); total+=w; wheel.append((total,d,nc))
            r=rng.random()*total; dE,new=wheel[-1][1], wheel[-1][2]
            for cum,d,nc in wheel:
                if r<=cum: dE,new=d,nc; break
        if dE <= 0 or rng.random() < math.exp(-dE/max(T,1e-12)):
            state[i] = new; E += dE
            vconf[i] = forbid_counts[i][new] + sum(1 for j in adj[i] if state[j] == new)
            for j in adj[i]:
                if state[j] == old: vconf[j] -= 1
                if state[j] == new: vconf[j] += 1
            tabu[i][old]=step+rng.randint(4,18)
            if E < bestE:
                bestE=E; best=state[:]; stagn=0
            else:
                stagn += 1
        if stagn > 25000 + 150*max(1,E):
            for i,a in enumerate(vconf):
                if a > 0: state[i]=rng.randrange(case.c)
            E,vconf=compute(); stagn=0
    out=list(base)
    for v,i in zip(free, range(m)):
        out[v]=best[i]
    ok,bad=verify_coloring(case.kind,case.c,case.N,out)
    return out, {"status":"TIMEOUT", "best_energy":bestE, "free_vertices":m,
                 "initial_bad_edges":len(sub.get("bad0", [])), "seconds":now()-t0,
                 "steps":step, "verified":ok, "bad":bad}


def choose_backbone(case: BenchCase, timeout: float, seed: int, pmax: int, policy: str, per_try: float) -> Tuple[Optional[List[int]], Dict[str, object]]:
    """Find a periodic or twisted low-conflict/full coloring to use as a backbone."""
    # Try period first briefly.
    res = run_period_potts(case, timeout * 0.5, seed, pmax, policy, per_try)
    if res.get("status") == "SAT" and res.get("coloring"):
        return parse_color_string(str(res["coloring"])), {"backbone_mode":"period", **res}
    # If period failed, try twist.
    res2 = run_twist_potts(case, timeout * 0.5, seed+1234567, pmax, policy, per_try)
    if res2.get("status") == "SAT" and res2.get("coloring"):
        return parse_color_string(str(res2["coloring"])), {"backbone_mode":"twist", **res2}
    # Fall back to best direct if instance small enough.
    res3 = run_direct_potts(case, timeout * 0.5, seed+7654321, max_edges=500_000)
    if res3.get("coloring"):
        return parse_color_string(str(res3["coloring"])), {"backbone_mode":"direct", **res3}
    return None, {"status":"NO_BACKBONE", "period_result":res, "twist_result":res2, "direct_result":res3}


def run_defect_potts(case: BenchCase, timeout: float, seed: int, pmax: int, policy: str, per_try: float, radius_max: int) -> Dict[str, object]:
    t0 = now()
    base, binfo = choose_backbone(case, min(timeout, max(1.0, timeout*0.35)), seed, pmax, policy, per_try)
    if base is None:
        return {"status":"NO_BACKBONE", "structure":"defect", "engine":"potts", "total_seconds":now()-t0, "backbone_info":binfo}
    ok, bad = verify_coloring(case.kind, case.c, case.N, base)
    if ok:
        return {"status":"SAT", "structure":"defect", "engine":"potts", "reason":"backbone_already_valid",
                "total_seconds":now()-t0, "verified":True, "bad":None, "coloring":color_string(base), "backbone_info":binfo}
    last = None
    for radius in range(radius_max + 1):
        rem = timeout - (now() - t0)
        if rem <= 0:
            break
        sub, err = build_defect_subproblem(case, base, radius)
        if sub is None:
            last = {"radius":radius, "error":err}
            continue
        out, info = anneal_defect_subproblem(case, sub, base, min(rem, per_try*4), seed+1000*radius)
        last = {"radius":radius, **info}
        if out is not None and info.get("status") == "SAT":
            return {**info, "structure":"defect", "engine":"potts", "radius":radius, "total_seconds":now()-t0,
                    "coloring":color_string(out), "backbone_info":binfo}
    return {"status":"TIMEOUT", "structure":"defect", "engine":"potts", "total_seconds":now()-t0,
            "last":last, "backbone_info":binfo}


def run_defect_kissat(case: BenchCase, solver: str, timeout: float, out_dir: str, seed: int, pmax: int, policy: str, per_try: float, radius_max: int) -> Dict[str, object]:
    t0 = now()
    base, binfo = choose_backbone(case, min(timeout, max(1.0, timeout*0.25)), seed, pmax, policy, per_try)
    if base is None:
        return {"status":"NO_BACKBONE", "structure":"defect", "engine":"kissat", "total_seconds":now()-t0, "backbone_info":binfo}
    ok, bad = verify_coloring(case.kind, case.c, case.N, base)
    if ok:
        return {"status":"SAT", "structure":"defect", "engine":"kissat", "reason":"backbone_already_valid",
                "total_seconds":now()-t0, "verified":True, "bad":None, "coloring":color_string(base), "backbone_info":binfo}
    last = None
    for radius in range(radius_max + 1):
        rem = timeout - (now() - t0)
        if rem <= 0:
            break
        sub, err = build_defect_subproblem(case, base, radius)
        if sub is None:
            last = {"radius":radius, "error":err}; continue
        if sub.get("already_valid"):
            return {"status":"SAT", "structure":"defect", "engine":"kissat", "reason":"backbone_already_valid_after_build",
                    "total_seconds":now()-t0, "verified":True, "bad":None, "coloring":color_string(base), "backbone_info":binfo}
        free = list(sub["free"])  # type: ignore
        cnf_path = os.path.join(out_dir, f"{case.name}.defect_r{radius}.cnf")
        meta = write_graph_coloring_cnf(len(free), case.c, sub["sub_edges"], cnf_path, sub["forbids"])  # type: ignore
        sat = run_solver_on_cnf(solver, cnf_path, rem)
        last = {**sat, "radius":radius, "cnf":meta, "free_vertices":len(free)}
        if sat.get("status") == "SAT":
            repair = parse_model_from_output(sat.get("stdout", ""), len(free), case.c)
            sat.pop("stdout", None)
            if repair is None:
                return {**sat, "structure":"defect", "engine":"kissat", "radius":radius, "model_printed":False,
                        "total_seconds":now()-t0, "cnf":meta, "backbone_info":binfo}
            out = list(base)
            for v, col in zip(free, repair):
                out[v] = col
            ok, bad = verify_coloring(case.kind, case.c, case.N, out)
            return {**sat, "structure":"defect", "engine":"kissat", "radius":radius, "total_seconds":now()-t0,
                    "verified":ok, "bad":bad, "coloring":color_string(out), "free_vertices":len(free), "cnf":meta,
                    "backbone_info":binfo}
        sat.pop("stdout", None)
    return {"status":"TIMEOUT", "structure":"defect", "engine":"kissat", "total_seconds":now()-t0,
            "last":last, "backbone_info":binfo}

# Hybrid: for this matrix, hybrid means "try Potts version first, then Kissat version if not solved".

def run_hybrid(case: BenchCase, structure: str, solver: str, timeout: float, out_dir: str, seed: int,
               pmax: int, policy: str, per_try: float, radius_max: int, max_edges: int, max_clauses: int) -> Dict[str, object]:
    t0 = now()
    if structure == "direct":
        first = run_direct_potts(case, min(timeout*0.5, timeout), seed, max_edges)
        if first.get("status") == "SAT":
            return {**first, "engine":"hybrid", "potts_phase":first, "total_seconds":now()-t0}
        rem = timeout - (now()-t0)
        second = run_direct_kissat(case, solver, rem, out_dir, max_clauses) if rem > 0 else {"status":"NO_TIME"}
    elif structure == "period":
        first = run_period_potts(case, min(timeout*0.5, timeout), seed, pmax, policy, per_try)
        if first.get("status") == "SAT":
            return {**first, "engine":"hybrid", "potts_phase":first, "total_seconds":now()-t0}
        rem = timeout - (now()-t0)
        second = run_period_kissat(case, solver, rem, out_dir, pmax, policy) if rem > 0 else {"status":"NO_TIME"}
    elif structure == "twist":
        first = run_twist_potts(case, min(timeout*0.5, timeout), seed, pmax, policy, per_try)
        if first.get("status") == "SAT":
            return {**first, "engine":"hybrid", "potts_phase":first, "total_seconds":now()-t0}
        rem = timeout - (now()-t0)
        second = run_twist_kissat(case, solver, rem, out_dir, pmax, policy) if rem > 0 else {"status":"NO_TIME"}
    elif structure == "defect":
        first = run_defect_potts(case, min(timeout*0.5, timeout), seed, pmax, policy, per_try, radius_max)
        if first.get("status") == "SAT":
            return {**first, "engine":"hybrid", "potts_phase":first, "total_seconds":now()-t0}
        rem = timeout - (now()-t0)
        second = run_defect_kissat(case, solver, rem, out_dir, seed+99999, pmax, policy, per_try, radius_max) if rem > 0 else {"status":"NO_TIME"}
    else:
        return {"status":"UNKNOWN_STRUCTURE", "structure":structure}
    status = second.get("status")
    out = {**second, "engine":"hybrid", "structure":structure, "potts_phase":first, "sat_phase":second,
           "total_seconds":now()-t0}
    return out

# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def select_cases(spec: str) -> List[BenchCase]:
    if spec == "default" or spec == "all":
        keys = ["VS3_N28","VS4_N57","VS5_N180","VS6_N333","VS6_N400","VS7_N576","VS8_N841",
                "VC3_N521","VC4_N2197","VC5_N9261","VC6_N50653","VC7_N79507","VC8_N753571"]
        return [DEFAULT_CASES[k] for k in keys]
    out = []
    for k in spec.split(','):
        k = k.strip()
        if not k:
            continue
        if k not in DEFAULT_CASES:
            raise SystemExit(f"unknown case {k}; known: {', '.join(DEFAULT_CASES)}")
        out.append(DEFAULT_CASES[k])
    return out


def run_one(case: BenchCase, structure: str, engine: str, args) -> Dict[str, object]:
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = now()
    try:
        if engine == "potts":
            if structure == "direct":
                res = run_direct_potts(case, args.timeout, args.seed, args.max_full_edges)
            elif structure == "period":
                res = run_period_potts(case, args.timeout, args.seed, args.pmax, args.period_policy, args.per_try)
            elif structure == "twist":
                res = run_twist_potts(case, args.timeout, args.seed, args.pmax, args.period_policy, args.per_try)
            elif structure == "defect":
                res = run_defect_potts(case, args.timeout, args.seed, args.pmax, args.period_policy, args.per_try, args.defect_radius_max)
            else:
                res = {"status":"UNKNOWN_STRUCTURE"}
        elif engine == "kissat":
            if structure == "direct":
                res = run_direct_kissat(case, args.solver, args.timeout, args.out_dir, args.max_full_clauses)
            elif structure == "period":
                res = run_period_kissat(case, args.solver, args.timeout, args.out_dir, args.pmax, args.period_policy)
            elif structure == "twist":
                res = run_twist_kissat(case, args.solver, args.timeout, args.out_dir, args.pmax, args.period_policy)
            elif structure == "defect":
                res = run_defect_kissat(case, args.solver, args.timeout, args.out_dir, args.seed, args.pmax, args.period_policy, args.per_try, args.defect_radius_max)
            else:
                res = {"status":"UNKNOWN_STRUCTURE"}
        elif engine == "hybrid":
            res = run_hybrid(case, structure, args.solver, args.timeout, args.out_dir, args.seed, args.pmax,
                             args.period_policy, args.per_try, args.defect_radius_max, args.max_full_edges,
                             args.max_full_clauses)
        else:
            res = {"status":"UNKNOWN_ENGINE"}
    except Exception as e:
        res = {"status":"EXCEPTION", "exception":repr(e)}
    res.setdefault("case", asdict(case))
    res.setdefault("structure", structure)
    res.setdefault("engine", engine)
    res.setdefault("total_seconds", now()-t0)
    if res.get("status") == "SAT" and res.get("coloring"):
        try:
            col = parse_color_string(str(res["coloring"]))
            ok, bad = verify_coloring(case.kind, case.c, case.N, col)
            res["verified"] = ok; res["bad"] = bad
            if ok:
                res["witness_path"] = save_witness(args.out_dir, case, f"{structure}.{engine}", col,
                                                     {"structure":structure,"engine":engine})
        except Exception as e:
            res["witness_save_error"] = repr(e)
    return res


def main(argv: Optional[Sequence[str]]=None) -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--cases", default="default", help="default/all or comma-separated case names")
    ap.add_argument("--structures", default="direct,period,twist,defect", help="comma list: direct,period,twist,defect")
    ap.add_argument("--engines", default="potts,kissat,hybrid", help="comma list: potts,kissat,hybrid")
    ap.add_argument("--timeout", type=float, default=300.0, help="per case/structure/engine timeout seconds")
    ap.add_argument("--solver", default="./kissat/build/kissat")
    ap.add_argument("--out-dir", default="runs/poly_matrix")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pmax", type=int, default=160)
    ap.add_argument("--period-policy", choices=["guided","blind"], default="guided")
    ap.add_argument("--per-try", type=float, default=1.0, help="per anneal candidate cap inside period/twist scans")
    ap.add_argument("--defect-radius-max", type=int, default=2)
    ap.add_argument("--max-full-edges", type=int, default=2_000_000, help="skip direct Potts if graph has more edges")
    ap.add_argument("--max-full-clauses", type=int, default=8_000_000, help="skip full direct CNF if more clauses")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    cases = select_cases(args.cases)
    structures = [x.strip() for x in args.structures.split(',') if x.strip()]
    engines = [x.strip() for x in args.engines.split(',') if x.strip()]

    jsonl_path = os.path.join(args.out_dir, f"results_{int(time.time())}.jsonl")
    csv_path = os.path.join(args.out_dir, "summary.csv")
    all_records: List[Dict[str, object]] = []

    print(f"Writing JSONL to {jsonl_path}", flush=True)
    with open(jsonl_path, "w") as jf:
        for case in cases:
            est = estimate_full(case)
            print(f"\n### {case.name}: {case.kind}, c={case.c}, N={case.N}, est={est} ###", flush=True)
            for structure in structures:
                for engine in engines:
                    print(f"RUN {case.name} structure={structure} engine={engine}", flush=True)
                    rec = run_one(case, structure, engine, args)
                    all_records.append(rec)
                    jf.write(json.dumps(rec) + "\n"); jf.flush()
                    # concise terminal line
                    print(json.dumps({
                        "case": case.name, "structure": structure, "engine": engine,
                        "status": rec.get("status"), "seconds": rec.get("total_seconds", rec.get("wall_seconds")),
                        "verified": rec.get("verified"), "period": rec.get("period"), "g": rec.get("g"),
                        "witness_path": rec.get("witness_path")
                    }), flush=True)

    with open(csv_path, "w", newline="") as cf:
        fields = ["case","kind","c","N","structure","engine","status","verified","seconds","period","g","energy","best_energy","attempts","witness_path"]
        w = csv.DictWriter(cf, fieldnames=fields)
        w.writeheader()
        for rec in all_records:
            case_dict = rec.get("case", {}) if isinstance(rec.get("case"), dict) else {}
            w.writerow({
                "case": case_dict.get("name"), "kind": case_dict.get("kind"), "c": case_dict.get("c"), "N": case_dict.get("N"),
                "structure": rec.get("structure"), "engine": rec.get("engine"), "status": rec.get("status"),
                "verified": rec.get("verified"), "seconds": rec.get("total_seconds", rec.get("wall_seconds")),
                "period": rec.get("period"), "g": rec.get("g"), "energy": rec.get("energy"),
                "best_energy": rec.get("best_energy"), "attempts": rec.get("attempts"), "witness_path": rec.get("witness_path")
            })
    print(f"\nWROTE {jsonl_path}\nWROTE {csv_path}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
