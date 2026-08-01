# ramsey-combinatorics-tools

`ramsey-combinatorics-tools` is a focused Python 3.11+ package for finite
polynomial van der Waerden difference-coloring problems. Given an integer
polynomial `p` with `p(0) = 0`, `c >= 2`, and `N >= 1`, it studies colorings of
`0, ..., N - 1` in which endpoints at a distance `|p(d)|` always receive
different colors.

Phase 1 deliberately contains no generic Ramsey framework, periodic or twisted
models, repair heuristics, Potts/annealing searches, UBCSAT integration, or
external solver execution. Its only search backend is a deterministic DSATUR
backtracker guarded by a small-instance size limit.

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
  --backend bruteforce --output square-28.json
pvdw verify square-28.json
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
