"""Verified JSON, text, and optional LaTeX proof-artifact exports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProofArtifact:
    title: str
    data: Mapping[str, object]
    text: str
    verified: bool
    latex: str | None = None


@dataclass(frozen=True)
class ExportedProofFiles:
    json_path: Path
    text_path: Path
    latex_path: Path | None


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: to_jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot serialize proof value of type {type(value).__name__}")


def write_proof_artifact(
    prefix: str | Path,
    artifact: ProofArtifact,
    *,
    include_latex: bool = False,
) -> ExportedProofFiles:
    """Write only independently verified proof artifacts."""

    if not artifact.verified:
        raise ValueError("refusing to export an unverified proof artifact")
    base = Path(prefix)
    base.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(f"{base}.json")
    text_path = Path(f"{base}.txt")
    payload = {
        "format_version": 1,
        "title": artifact.title,
        "verified": True,
        "proof": to_jsonable(artifact.data),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    text_path.write_text(artifact.text.rstrip() + "\n", encoding="utf-8")
    latex_path: Path | None = None
    if include_latex:
        if artifact.latex is None:
            raise ValueError("LaTeX output was requested but the artifact has none")
        latex_path = Path(f"{base}.tex")
        latex_path.write_text(artifact.latex.rstrip() + "\n", encoding="utf-8")
    return ExportedProofFiles(json_path, text_path, latex_path)
