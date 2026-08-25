#!/usr/bin/env python3
"""Compare R and Python long-run artifacts using the combined-MCSE gate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from core import REQUIRED_METHODS, compare_parity_summaries, load_manifest


def compare_method_artifacts(
    r_artifact: Mapping[str, Any],
    python_artifacts: Sequence[Mapping[str, Any]],
    *,
    multiplier: float = 6.0,
) -> dict[str, Any]:
    """Validate provenance and compare every supplied Python precision."""
    method = r_artifact.get("method")
    if method not in REQUIRED_METHODS:
        raise ValueError("R parity artifact has an unknown method")
    if r_artifact.get("implementation") != "bspcov":
        raise ValueError("reference parity artifact must come from bspcov")
    if r_artifact.get("version") != "1.0.3":
        raise ValueError("reference parity artifact must use bspcov 1.0.3")
    fixture_hash = r_artifact.get("fixture_sha256")
    if not isinstance(fixture_hash, str) or len(fixture_hash) != 64:
        raise ValueError("R parity artifact requires a fixture SHA-256")
    revision = str(r_artifact.get("git_revision", ""))
    if len(revision) != 40:
        raise ValueError("R parity artifact requires a full git revision")
    if r_artifact.get("git_dirty") is not False:
        raise ValueError("R parity artifact requires a clean git worktree")

    result: dict[str, Any] = {}
    for artifact in python_artifacts:
        if artifact.get("method") != method:
            raise ValueError("R and Python parity methods differ")
        if artifact.get("implementation") != "pybspcov":
            raise ValueError("candidate parity artifact must come from pybspcov")
        if artifact.get("fixture_sha256") != fixture_hash:
            raise ValueError("R and Python parity fixture hashes differ")
        if artifact.get("git_revision") != revision:
            raise ValueError("R and Python parity revisions differ")
        if artifact.get("git_dirty") is not False:
            raise ValueError("Python parity artifact requires a clean git worktree")
        dtype = artifact.get("dtype")
        if dtype not in {"float32", "float64"}:
            raise ValueError("Python parity dtype must be float32 or float64")
        if dtype in result:
            raise ValueError(f"duplicate Python parity artifact for {dtype}")
        comparison = compare_parity_summaries(
            r_artifact["summary"], artifact["summary"], multiplier=multiplier
        )
        result[dtype] = {
            "verdict": comparison["verdict"],
            "comparison": comparison,
            "r_version": r_artifact["version"],
            "python_version": artifact.get("version"),
        }
    if set(result) != {"float32", "float64"}:
        raise ValueError(
            "both float32 and float64 Python parity artifacts are required"
        )
    return result


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def main() -> None:
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=script_directory / "manifest.json"
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = load_manifest(arguments.manifest)

    methods: dict[str, Any] = {}
    revisions: set[str] = set()
    for method in sorted(REQUIRED_METHODS):
        r_artifact = _read(arguments.artifact_dir / f"r-{method}.json")
        revisions.add(str(r_artifact.get("git_revision", "")))
        details = compare_method_artifacts(
            r_artifact,
            [
                _read(arguments.artifact_dir / f"python-{method}-float64.json"),
                _read(arguments.artifact_dir / f"python-{method}-float32.json"),
            ],
            multiplier=float(manifest["parity"]["mcse_multiplier"]),
        )
        methods[method] = {
            "float64": details["float64"]["verdict"],
            "float32": details["float32"]["verdict"],
            "details": details,
        }
    if len(revisions) != 1:
        raise ValueError("parity artifact revisions do not match across methods")
    result = {"schema_version": "1.0", "revision": revisions.pop(), "methods": methods}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
