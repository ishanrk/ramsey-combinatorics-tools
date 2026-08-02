from __future__ import annotations

import json

import pytest

from pvdw.proof_tools.export import ProofArtifact, write_proof_artifact


def test_verified_artifact_writes_json_text_and_latex(tmp_path) -> None:
    artifact = ProofArtifact(
        "tiny proof", {"identity": [1, 2, 3]}, "identity verified", True, "1+2=3"
    )
    files = write_proof_artifact(tmp_path / "proof", artifact, include_latex=True)
    assert json.loads(files.json_path.read_text())["verified"] is True
    assert files.text_path.read_text() == "identity verified\n"
    assert files.latex_path is not None and files.latex_path.read_text() == "1+2=3\n"


def test_unverified_artifact_is_never_written(tmp_path) -> None:
    with pytest.raises(ValueError):
        write_proof_artifact(
            tmp_path / "bad", ProofArtifact("bad", {}, "bad", False)
        )
