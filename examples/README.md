# Proof-search examples

Phase 3 proof commands write independently checked JSON and plain-text
artifacts. Reproduce the bundled regressions from the repository root:

```console
pvdw gadget --target moser --differences 1,2,5,7,12,15 --ambient-max 16
pvdw block-cover --example x2-plus-x-spindle
pvdw parity-cover --example square-submillion
pvdw drift --example cubes-3color-522
pvdw transfer --poly "x^2+x" \
  --source-differences 1,2,5,7,12,15 --max-scale 100 --input-bound 100
```

Use `--output-prefix PATH` to choose the `.json` and `.txt` destinations and
`--latex` where a command offers a concise mathematical rendering. The default
destination is the ignored `proof-artifacts/` directory.

[`moser-spindle.json`](moser-spindle.json) demonstrates the edge-list format
accepted anywhere `--target` permits a JSON file.
