from __future__ import annotations

import json

from pvdw.cli import main


def test_gadget_and_transfer_commands_export_verified_artifacts(tmp_path) -> None:
    gadget = tmp_path / "gadget"
    assert main(
        [
            "gadget",
            "--target", "moser",
            "--differences", "1,2,5,7,12,15",
            "--ambient-max", "16",
            "--output-prefix", str(gadget),
        ]
    ) == 0
    assert json.loads(gadget.with_suffix(".json").read_text())["verified"] is True

    transfer = tmp_path / "transfer"
    assert main(
        [
            "transfer",
            "--poly", "x^2+x",
            "--source-differences", "1,2,5,7,12,15",
            "--max-scale", "100",
            "--input-bound", "100",
            "--output-prefix", str(transfer),
        ]
    ) == 0
    payload = json.loads(transfer.with_suffix(".json").read_text())
    assert payload["verified"] is True
    assert payload["proof"]["scale"] == 6


def test_block_cover_and_drift_commands_export_verified_artifacts(tmp_path) -> None:
    cover = tmp_path / "cover"
    assert main(
        [
            "block-cover",
            "--example", "x2-plus-x-spindle",
            "--output-prefix", str(cover),
        ]
    ) == 0
    assert json.loads(cover.with_suffix(".json").read_text())["verified"] is True

    searched_cover = tmp_path / "searched-cover"
    assert main(
        [
            "block-cover",
            "--target", "moser",
            "--differences", "1,2,5,7,12,15",
            "--ambient-max", "16",
            "--multiplicity", "7",
            "--output-prefix", str(searched_cover),
        ]
    ) == 0
    searched_payload = json.loads(searched_cover.with_suffix(".json").read_text())
    assert searched_payload["proof"]["cover"]["verified"] is True
    assert searched_payload["proof"]["one_exception_obstruction"]["impossible"] is True

    drift = tmp_path / "drift"
    assert main(
        [
            "drift",
            "--example", "cubes-3color-522",
            "--output-prefix", str(drift),
            "--json",
        ]
    ) == 0
    payload = json.loads(drift.with_suffix(".json").read_text())
    assert payload["verified"] is True
    assert len(payload["proof"]["cases"]) == 4
