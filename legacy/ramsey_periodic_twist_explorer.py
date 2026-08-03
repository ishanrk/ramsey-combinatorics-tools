#!/usr/bin/env python3
"""
Exploratory Potts/quotient search for polynomial van der Waerden lower bounds.

Supports two structured ansatzes for distance Ramsey problems on [N]:

1. ordinary periodic quotient:
       color(n) = a[n mod p]

2. affine/twisted quotient over cyclic color group Z_c:
       color(n) = a[n mod p] + g * floor(n / p)  (mod c)
   where n is zero-indexed.  This can compress a long period p*ord(g) into p
   variables and can sometimes avoid self-loop obstructions at the smaller p.

The script is self-contained except for importing utility functions from
ramsey_sa_sat_full_benchmark.py in the same directory.
"""
from __future__ import annotations
import argparse, json, math, random, sys, time
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.append('/mnt/data')
from ramsey_sa_sat_full_benchmark import (  # type: ignore
    ALPH, build_cyclic_distance_graph, build_graph, color_string,
    distance_set, energy_from_adj, lift_periodic, potts_anneal_graph,
    verify_coloring,
)


def first_zero_N(kind: str, period: int) -> int:
    """Largest ordinary periodic witness can only go up to this N.

    If d is a forbidden distance and d == 0 mod period, an ordinary period-p
    coloring has a self-loop.  The first such distance is k^power.  Since [N]
    only contains distances <= N-1, the ordinary period is potentially valid up
    to N = k^power, but never N = k^power + 1.
    """
    power = 2 if kind == 'VS' else 3
    k = 1
    while pow(k, power, period) != 0:
        k += 1
    return k ** power


def scan_ordinary_periods(kind: str, c: int, pmax: int, total_timeout: float,
                          per_try: float = 0.05, seeds: int = 1,
                          order: str = 'desc') -> Dict[str, object]:
    """Search ordinary period-p quotient colorings for p <= pmax."""
    t0 = time.perf_counter()
    candidates = [(first_zero_N(kind, p), p) for p in range(c + 1, pmax + 1)]
    candidates.sort(reverse=(order == 'desc'))
    best: Optional[Dict[str, object]] = None
    solved: List[Dict[str, object]] = []
    tried = 0
    for N, p in candidates:
        if time.perf_counter() - t0 >= total_timeout:
            break
        adj, edges, residues, impossible = build_cyclic_distance_graph(kind, N, p)
        if adj is None:
            continue
        for s in range(seeds):
            remaining = total_timeout - (time.perf_counter() - t0)
            if remaining <= 0:
                break
            pat, info = potts_anneal_graph(adj, c, seed=100000*c + 1000*p + s,
                                           time_limit=min(per_try, remaining))
            tried += 1
            E = energy_from_adj(adj, pat)
            if E == 0:
                rec = {
                    'kind': kind, 'c': c, 'mode': 'ordinary_periodic',
                    'period': p, 'N': N, 'quotient_vertices': p,
                    'quotient_edges': len(edges), 'residue_count': len(residues),
                    'residues': residues, 'pattern': color_string(pat),
                    'elapsed': time.perf_counter() - t0, 'attempt': tried,
                }
                solved.append(rec)
                if best is None or N > int(best['N']):
                    best = rec
                break
    return {
        'kind': kind, 'c': c, 'pmax': pmax, 'total_timeout': total_timeout,
        'per_try': per_try, 'seeds': seeds, 'tried': tried,
        'best': best, 'solved_count': len(solved), 'elapsed': time.perf_counter() - t0,
        'solved_tail': solved[-10:],
    }


# -------------------------- twisted quotient --------------------------

def build_twisted_constraints(kind: str, N: int, p: int, c: int, g: int) -> Tuple[Optional[List[Tuple[int,int,int]]], List[int], Dict[str, object]]:
    """Constraints for color(n)=a[n mod p]+g*floor(n/p) mod c.

    For a forbidden distance d = m p + s, the pair from residue r to r+s has
    block carry m + carry(r,s).  If the base colors are a_r and a_j, equality
    in the lifted coloring occurs iff
          a_r == a_j + g*(m+carry)  (mod c).
    So each constraint is (r, j, shift) meaning forbid
          a_r == a_j + shift mod c.
    """
    constraints: List[Tuple[int,int,int]] = []
    seen = set()
    ds = distance_set(kind, N)
    for d in ds:
        m, s = divmod(d, p)
        for r in range(p):
            j = (r + s) % p
            carry = 1 if r + s >= p else 0
            shift = (g * (m + carry)) % c
            if r == j and shift == 0:
                return None, ds, {'impossible_self_loop': {'distance': d, 'residue': r, 'shift': shift}}
            key = (r, j, shift)
            if key not in seen:
                seen.add(key)
                constraints.append(key)
    return constraints, ds, {'constraints': len(constraints)}


def init_twisted_counts(p: int, c: int, constraints: Sequence[Tuple[int,int,int]], colors: Sequence[int]):
    counts = [[0] * c for _ in range(p)]
    out = [[] for _ in range(p)]  # v appears as left endpoint: (right, shift)
    inn = [[] for _ in range(p)]  # v appears as right endpoint: (left, shift)
    for r, j, sh in constraints:
        out[r].append((j, sh))
        inn[j].append((r, sh))
        counts[r][(colors[j] + sh) % c] += 1
        counts[j][(colors[r] - sh) % c] += 1
    E = sum(counts[v][colors[v]] for v in range(p)) // 2
    vconf = [counts[v][colors[v]] for v in range(p)]
    return counts, E, vconf, out, inn


def greedy_twisted_init(p: int, c: int, constraints: Sequence[Tuple[int,int,int]], rng: random.Random) -> List[int]:
    deg = [0] * p
    for r, j, _ in constraints:
        deg[r] += 1; deg[j] += 1
    order = sorted(range(p), key=lambda v: -deg[v])
    colors = [-1] * p
    assigned = [False] * p
    for v in order:
        score = [0] * c
        for r, j, sh in constraints:
            if r == v and assigned[j]:
                score[(colors[j] + sh) % c] += 1
            elif j == v and assigned[r]:
                score[(colors[r] - sh) % c] += 1
        m = min(score)
        colors[v] = rng.choice([k for k, x in enumerate(score) if x == m])
        assigned[v] = True
    return colors


def twisted_anneal(kind: str, c: int, N: int, p: int, g: int,
                   seed: int = 0, time_limit: float = 1.0) -> Tuple[Optional[List[int]], Dict[str, object]]:
    constraints, ds, meta = build_twisted_constraints(kind, N, p, c, g)
    if constraints is None:
        return None, {'status': 'IMPOSSIBLE_SELF_LOOP', 'kind': kind, 'c': c, 'N': N, 'period': p, 'g': g, **meta}
    rng = random.Random(seed)
    t0 = time.perf_counter(); deadline = t0 + time_limit
    colors = greedy_twisted_init(p, c, constraints, rng) if rng.random() > 0.2 else [rng.randrange(c) for _ in range(p)]
    counts, E, vconf, out, inn = init_twisted_counts(p, c, constraints, colors)
    best = colors[:]; bestE = E; best_step = 0; stagn = 0
    tabu = [[0] * c for _ in range(p)]
    T_high, T_low = 2.5, 0.015
    cycle = max(1000, 200 * p)
    step = 0
    while True:
        step += 1
        if E == 0:
            return colors, {'status': 'SAT', 'energy': 0, 'kind': kind, 'c': c, 'N': N, 'period': p, 'g': g,
                            'constraints': len(constraints), 'distances': ds, 'seconds': time.perf_counter() - t0,
                            'steps': step}
        if step % 4096 == 0 and time.perf_counter() >= deadline:
            break
        if rng.random() < 0.94:
            v = None
            for _ in range(20):
                x = rng.randrange(p)
                if vconf[x] > 0:
                    v = x; break
            if v is None:
                conflicted = [i for i, a in enumerate(vconf) if a > 0]
                if not conflicted:
                    return colors, {'status': 'SAT', 'energy': 0, 'kind': kind, 'c': c, 'N': N, 'period': p, 'g': g,
                                    'constraints': len(constraints), 'distances': ds, 'seconds': time.perf_counter() - t0,
                                    'steps': step}
                v = max(rng.sample(conflicted, min(6, len(conflicted))), key=lambda x: vconf[x])
        else:
            v = rng.randrange(p)
        old = colors[v]; old_bad = counts[v][old]
        cands = []
        min_delta = 10**9
        for new in range(c):
            if new == old: continue
            dE = counts[v][new] - old_bad
            if step < tabu[v][new] and E + dE >= bestE:
                continue
            cands.append((dE, new))
            min_delta = min(min_delta, dE)
        if not cands:
            continue
        frac = (step % cycle) / max(1, cycle - 1)
        T = math.exp(math.log(T_high) * (1-frac) + math.log(T_low) * frac)
        if rng.random() < 0.82:
            dE, new = rng.choice([x for x in cands if x[0] == min_delta])
        else:
            total = 0.0; wheel = []
            for d, nw in cands:
                w = math.exp(-(d - min_delta) / max(T, 1e-12))
                total += w; wheel.append((total, d, nw))
            r = rng.random() * total
            dE, new = wheel[-1][1], wheel[-1][2]
            for cum, d, nw in wheel:
                if r <= cum:
                    dE, new = d, nw; break
        if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-12)):
            colors[v] = new; E += dE; vconf[v] = counts[v][new]
            for j, sh in out[v]:
                old_forb = (old - sh) % c; new_forb = (new - sh) % c
                if colors[j] == old_forb: vconf[j] -= 1
                if colors[j] == new_forb: vconf[j] += 1
                counts[j][old_forb] -= 1; counts[j][new_forb] += 1
            for r0, sh in inn[v]:
                old_forb = (old + sh) % c; new_forb = (new + sh) % c
                if colors[r0] == old_forb: vconf[r0] -= 1
                if colors[r0] == new_forb: vconf[r0] += 1
                counts[r0][old_forb] -= 1; counts[r0][new_forb] += 1
            tabu[v][old] = step + rng.randint(4, 18) + min(100, E // 5)
            if E < bestE:
                bestE = E; best = colors[:]; best_step = step; stagn = 0
            else:
                stagn += 1
        if stagn > 25000 + 150 * max(1, E):
            for i, a in enumerate(vconf):
                if a > 0 and rng.random() < 0.8:
                    colors[i] = rng.randrange(c)
            for _ in range(max(1, p // 30)):
                colors[rng.randrange(p)] = rng.randrange(c)
            counts, E, vconf, out, inn = init_twisted_counts(p, c, constraints, colors)
            stagn = 0
    return best, {'status': 'TIMEOUT', 'energy': bestE, 'kind': kind, 'c': c, 'N': N, 'period': p, 'g': g,
                  'constraints': len(constraints), 'distances': ds, 'seconds': time.perf_counter() - t0,
                  'steps': step, 'best_step': best_step}


def lift_twisted(pattern: Sequence[int], N: int, c: int, p: int, g: int) -> List[int]:
    return [(pattern[n % p] + g * (n // p)) % c for n in range(N)]


def verify_twisted(kind: str, c: int, N: int, p: int, g: int, pattern: Sequence[int]):
    return verify_coloring(kind, c, N, lift_twisted(pattern, N, c, p, g))


def scan_twisted(kind: str, c: int, N: int, periods: Sequence[int], total_timeout: float,
                 per_try: float = 0.2, seeds: int = 1) -> Dict[str, object]:
    t0 = time.perf_counter(); best = None; solved = []; tried = 0
    for p in periods:
        for g in range(1, c):
            for s in range(seeds):
                rem = total_timeout - (time.perf_counter() - t0)
                if rem <= 0:
                    return {'kind': kind, 'c': c, 'N': N, 'mode': 'twisted_scan', 'best': best,
                            'solved_count': len(solved), 'solved_tail': solved[-10:], 'tried': tried,
                            'elapsed': time.perf_counter() - t0}
                pat, info = twisted_anneal(kind, c, N, p, g, seed=100000*c+1000*p+100*g+s,
                                           time_limit=min(per_try, rem))
                tried += 1
                E = info.get('energy', 10**9)
                if best is None or E < best.get('energy', 10**9):
                    best = {**info, 'pattern': color_string(pat) if pat else None}
                if pat is not None and info['status'] == 'SAT':
                    ok, bad = verify_twisted(kind, c, N, p, g, pat)
                    rec = {**info, 'verified': ok, 'bad': bad, 'pattern': color_string(pat),
                           'quotient_vertices': p, 'effective_period': p * (c // math.gcd(c, g)),
                           'elapsed': time.perf_counter() - t0}
                    solved.append(rec)
                    return {'kind': kind, 'c': c, 'N': N, 'mode': 'twisted_scan', 'best': rec,
                            'solved_count': len(solved), 'solved_tail': solved[-10:], 'tried': tried,
                            'elapsed': time.perf_counter() - t0}
    return {'kind': kind, 'c': c, 'N': N, 'mode': 'twisted_scan', 'best': best,
            'solved_count': len(solved), 'solved_tail': solved[-10:], 'tried': tried,
            'elapsed': time.perf_counter() - t0}


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p1 = sub.add_parser('ordinary-scan')
    p1.add_argument('kind'); p1.add_argument('c', type=int); p1.add_argument('--pmax', type=int, default=160)
    p1.add_argument('--timeout', type=float, default=120); p1.add_argument('--per-try', type=float, default=0.05)
    p1.add_argument('--seeds', type=int, default=1); p1.add_argument('--order', default='desc')
    p2 = sub.add_parser('twisted-scan')
    p2.add_argument('kind'); p2.add_argument('c', type=int); p2.add_argument('N', type=int)
    p2.add_argument('--periods', default='')
    p2.add_argument('--pmin', type=int, default=2); p2.add_argument('--pmax', type=int, default=100)
    p2.add_argument('--timeout', type=float, default=120); p2.add_argument('--per-try', type=float, default=0.2)
    p2.add_argument('--seeds', type=int, default=1)
    args = ap.parse_args()
    if args.cmd == 'ordinary-scan':
        res = scan_ordinary_periods(args.kind, args.c, args.pmax, args.timeout, args.per_try, args.seeds, args.order)
    else:
        if args.periods:
            periods = [int(x) for x in args.periods.split(',') if x]
        else:
            periods = list(range(args.pmin, args.pmax + 1))
        res = scan_twisted(args.kind, args.c, args.N, periods, args.timeout, args.per_try, args.seeds)
    print(json.dumps(res, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())


