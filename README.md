# Ramsey Theory Tools
`ramsey-combinatorics-tools` is a focused Python 3.11+ package for finite
polynomial van der Waerden difference-coloring problems. Given an integer
polynomial `p` with `p(0) = 0`, `c >= 2`, and `N >= 1`, it studies colorings of
`0, ..., N - 1` in which endpoints at a distance `|p(d)|` always receive
different colors.

The package remains deliberately specific to polynomial-distance coloring. It
provides unrestricted SAT and Potts search, monotone incremental scans,
finite-interval periodic and twisted-periodic quotients, and structured-backbone
repair. Restricted failures are always reported separately from unrestricted
noncolorability, and stochastic searches never report UNSAT.

## Install

```console
python -m pip install -e ".[dev]"
```

The package depends on `python-sat` for its established cardinality encodings.
`pytest` and `hypothesis` are installed by the `dev` extra.

## CLI

The primary machine input is a low-degree-first coefficient list:

```console
pvdw inspect --coefficients 0,0,1 --colors 4 --N 58
```

A deliberately restricted expression parser is also available. It accepts
integer constants, `x`, unary signs, addition, subtraction, multiplication, and
nonnegative integer powers; it never calls `eval`.

```console
pvdw inspect --poly "x^2" --colors 4 --N 58
pvdw encode --poly "x^2" --colors 4 --N 58 \
  --encoding onehot --amo pairwise --output instance.cnf
pvdw encode --poly "x^2+x" --colors 4 --N 97 \
  --encoding binary --output instance-binary.cnf
pvdw solve --poly "x^2" --colors 3 --N 28 \
  --mode direct --backend cadical195 --output square-28.json
pvdw verify square-28.json
```

Discover the exact PySAT and optional executable backends present on the host:

```console
pvdw backends
```

Further search modes are available without changing the polynomial convention:

```console
pvdw solve --poly "x^2" --colors 4 --N 57 \
  --mode direct --backend potts --timeout 10

pvdw scan --poly "x^2" --colors 4 --N-min 50 --N-max 60 \
  --backend glucose4 --encoding onehot

pvdw solve --poly "x^3" --colors 5 --N 9261 \
  --mode periodic --period 63 --backend cadical195

pvdw solve --poly "x^2" --colors 6 --N 400 \
  --mode twisted --period 20 --twist 3 --backend potts

pvdw periods --poly "x^2" --colors 8 --N 841 \
  --period-min 2 --period-max 160 --backend portfolio --timeout 30

pvdw twists --poly "x^2" --colors 6 --N 400 \
  --period-min 2 --period-max 80 --twists all --backend portfolio

pvdw repair --poly "x^2" --colors 8 --N 841 \
  --backbone periodic --period 29 --backend cadical195 \
  --editable-strategy greedy_vertex_cover --max-expansions 3
```

Add `--json` to any subcommand for machine-readable output. Exit codes are 0
for success or a witness, 10 for a proved full-model UNSAT result, 20 for an
inconclusive result, 2 for invalid input, and 1 for an internal failure.

## Exact finite distance generation

For `B = N - 1`, only values `0 < |p(d)| <= B` matter. If `a_k` is the leading
coefficient and `S` is the sum of absolute values of all lower coefficients,
the generator computes the least integer dominance and growth bounds from

```text
|a_k| t >= 2S
|a_k| t^k > 2B
```

using integer arithmetic and binary search. Beyond their maximum, the leading
term dominates and `|p(d)| > B`. Enumeration over the resulting certified
interval is therefore complete, with all signed preimages retained.

Edges are generated directly as `(u, u + delta)` for every certified distance
and `u in range(N - delta)`; there is no quadratic all-pairs scan.

## Backends and status semantics

`cadical195`, `cadical153`, `glucose4`, `glucose3`, and `minisat22` are preferred
in that order when present in PySAT. Other installed PySAT solver names can be
selected directly. Fixed-formula Kissat is accepted when available, but is not
used for incremental scans. Solvers without safe in-process interruption are
isolated in a subprocess whenever a fixed-formula timeout is requested.

External Kissat and CaDiCaL are discovered through `PATH`, `KISSAT_BIN`, and
`CADICAL_BIN`; a generic executable can be selected as `external:/path/to/solver`.
UBCSAT is optional and configured with `UBCSAT_BIN`. Supported algorithm names
are `walksat`, `walksat-tabu`, `adaptnovelty+`, `g2wsat`, and `saps`; additional
model-report flags can be passed repeatedly with `--extra-arg`.

The native Potts backend maintains constraint energy incrementally, retains the
best positive-energy coloring across restarts, and returns `unknown` or
`timeout` when it does not reach zero. A portfolio tries stochastic searches
before a complete CDCL solver; only the latter may supply a negative proof.

An unrestricted complete UNSAT result uses `unsat_full_model`. Periodic,
twisted, and repair UNSAT results use `no_witness_in_restricted_model`, with the
period/twist and finite-interval scope retained in metadata and witness files.

## Finite structured models

Periodic and twisted quotients include only residue constraints realized by an
actual edge in `0, ..., N-1`; they do not impose a full cyclic graph. A periodic
self-edge is immediately impossible. A twisted self-constraint with zero shift
is impossible, while a nonzero self-shift is tautological and skipped.

Repair retains the best nonzero-energy structured or direct Potts candidate,
selects editable vertices that hit every current bad edge, and encodes only
those vertices. Editable–frozen edges become deduplicated color forbids;
frozen–frozen conflicts are treated as construction errors. Failed reduced
models can expand by graph-neighborhood layers before returning a restricted
negative result.

## Encodings and witnesses

One-hot supports pairwise, sequential-counter, ladder, and bitwise at-most-one
constraints. Binary encoding forbids unused codes and expands endpoint
inequality exactly over every valid color. Both use stable, documented variable
maps. Pairwise one-hot and binary DIMACS files stream after an exact header is
computed; auxiliary encodings are materialized before writing.

JSON witnesses include their complete instance, full/restricted scope label,
search provenance, and verification result. The writer and reader independently
regenerate the polynomial distances and reject invalid colorings. Colorings with
at most 36 colors also have a compact `0-9A-Z` representation.

Historical exploratory scripts are preserved under [`legacy/`](legacy/) and are
not imported by the package.

## Benchmarks and optional integrations

```console
pvdw benchmark --suite project-regressions --timeout 30 \
  --output-dir benchmark-results
```

The benchmark writes `results.jsonl`, `summary.csv`, and independently verified
JSON witnesses. No witness is written for an invalid or absent model. Large
solver regressions are marked `slow` and run only when requested:

```console
PVDW_RUN_SLOW=1 pytest -q -m slow
```
