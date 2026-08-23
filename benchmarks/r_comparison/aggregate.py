#!/usr/bin/env python3
"""Validate and aggregate a complete cached R/Python comparison matrix."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from core import (
    geometric_mean,
    load_manifest,
    select_headline_dtype,
    summarize_timings,
    validate_timing_record,
)
from run_matrix import Cell, build_cells


def _record_key(record: Mapping[str, object]) -> tuple[object, ...]:
    return (
        record.get("method"),
        record.get("dimension"),
        record.get("implementation"),
        record.get("device"),
        record.get("dtype"),
        record.get("configuration"),
    )


def _cell_key(cell: Cell) -> tuple[object, ...]:
    return (
        cell.method,
        cell.dimension,
        cell.implementation,
        cell.device,
        cell.dtype,
        cell.configuration,
    )


def _median(record: Mapping[str, object]) -> float:
    summary = summarize_timings(
        record["warm_seconds"], retained_draws=int(record["retained_draws"])
    )
    return float(summary["median_seconds"])


def _expected_draws(method: str, parallelism: int, manifest: Mapping[str, Any]) -> int:
    configuration = manifest["methods"][method]
    samples = (
        configuration["samples"]
        if method in {"bm", "sbm"}
        else configuration["samples_per_batch"]
    )
    return int(samples * parallelism)


def _validate_parity(parity: Mapping[str, object]) -> Mapping[str, Any]:
    if parity.get("schema_version") != "1.0":
        raise ValueError("parity results must use schema version 1.0")
    revision = str(parity.get("revision", ""))
    if len(revision) != 40:
        raise ValueError("parity results require a full git revision")
    methods = parity.get("methods")
    if not isinstance(methods, Mapping):
        raise TypeError("parity methods must be an object")
    expected = {"bm", "sbm", "bandppp", "thresholdppp"}
    if set(methods) != expected:
        raise ValueError("parity results must contain all four methods")
    for method, result in methods.items():
        if not isinstance(result, Mapping):
            raise TypeError(f"{method} parity result must be an object")
        if result.get("float64") != "pass":
            raise ValueError(f"{method} float64 parity did not pass")
        if result.get("float32") not in {"pass", "fail"}:
            raise ValueError(f"{method} float32 parity must be pass or fail")
    return methods


def aggregate_records(
    records: Sequence[Mapping[str, object]],
    parity: Mapping[str, object],
    manifest: Mapping[str, Any],
) -> dict[str, object]:
    """Validate all cells and compute the README-facing summary."""
    parity_methods = _validate_parity(parity)
    expected_cells = {_cell_key(cell): cell for cell in build_cells(manifest)}
    selected: dict[tuple[object, ...], Mapping[str, object]] = {}
    revisions: set[str] = set()
    fixtures: dict[tuple[str, int], str] = {}
    for record in records:
        validate_timing_record(record)
        key = _record_key(record)
        if key not in expected_cells:
            raise ValueError(f"unexpected benchmark cell: {key}")
        if key in selected:
            raise ValueError(f"duplicate benchmark cell: {key}")
        cell = expected_cells[key]
        if record.get("parallelism") != cell.parallelism:
            raise ValueError(f"parallelism differs for benchmark cell: {key}")
        if record.get("cpu_cores") != cell.cpu_cores:
            raise ValueError(f"CPU core allocation differs for benchmark cell: {key}")
        if record.get("n_observations") != manifest["n_factor"] * cell.dimension:
            raise ValueError(f"observation count differs for benchmark cell: {key}")
        if record.get("seed") != manifest["seed"]:
            raise ValueError(f"seed differs for benchmark cell: {key}")
        if record.get("retained_draws") != _expected_draws(
            cell.method, cell.parallelism, manifest
        ):
            raise ValueError(f"retained draw count differs for benchmark cell: {key}")
        if cell.implementation == "bspcov" and record.get("version") != "1.0.3":
            raise ValueError("R benchmark records must use bspcov 1.0.3")
        fixture_key = (cell.method, cell.dimension)
        fixture = str(record["fixture_sha256"])
        previous_fixture = fixtures.setdefault(fixture_key, fixture)
        if previous_fixture != fixture:
            raise ValueError(f"fixture differs for {fixture_key}")
        revisions.add(str(record["git_revision"]))
        selected[key] = record

    missing = expected_cells.keys() - selected.keys()
    if missing:
        raise ValueError(f"missing benchmark cells: {sorted(missing)}")
    if len(revisions) != 1:
        raise ValueError("benchmark git revisions do not match")
    revision = revisions.pop()
    if parity.get("revision") != revision:
        raise ValueError("parity revision does not match timing revision")

    method_results: dict[str, object] = {}
    dimensions = [int(value) for value in manifest["dimensions"]]
    large_dimension = max(dimensions)
    for method in manifest["methods"]:
        r_optimized = {
            dimension: selected[
                (method, dimension, "bspcov", "cpu", "float64", "optimized")
            ]
            for dimension in dimensions
        }
        python_gpu = {
            dtype: {
                dimension: selected[
                    (method, dimension, "pybspcov", "gpu", dtype, "optimized")
                ]
                for dimension in dimensions
            }
            for dtype in ("float32", "float64")
        }
        parity_result = parity_methods[method]
        headline_dtype = select_headline_dtype(
            float64_seconds=_median(python_gpu["float64"][large_dimension]),
            float32_seconds=_median(python_gpu["float32"][large_dimension]),
            float64_parity=parity_result["float64"] == "pass",
            float32_parity=parity_result["float32"] == "pass",
        )
        speedups = [
            _median(r_optimized[dimension])
            / _median(python_gpu[headline_dtype][dimension])
            for dimension in dimensions
        ]
        r_large = r_optimized[large_dimension]
        python_large = python_gpu[headline_dtype][large_dimension]
        r_cpu = selected[
            (method, large_dimension, "bspcov", "cpu", "float64", "cpu_baseline")
        ]
        python_cpu = selected[
            (
                method,
                large_dimension,
                "pybspcov",
                "cpu",
                "float64",
                "cpu_baseline",
            )
        ]
        r_seconds = _median(r_large)
        python_seconds = _median(python_large)
        r_cpu_seconds = _median(r_cpu)
        python_cpu_seconds = _median(python_cpu)
        method_results[method] = {
            "headline_dtype": headline_dtype,
            "geometric_mean_speedup": geometric_mean(speedups),
            "large": {
                "r_optimized_seconds": r_seconds,
                "python_gpu_seconds": python_seconds,
                "optimized_speedup": r_seconds / python_seconds,
                "cold_speedup": float(r_large["cold_end_to_end_seconds"])
                / float(python_large["cold_end_to_end_seconds"]),
                "r_cpu_baseline_seconds": r_cpu_seconds,
                "python_cpu_baseline_seconds": python_cpu_seconds,
                "cpu_baseline_speedup": r_cpu_seconds / python_cpu_seconds,
            },
        }

    return {
        "schema_version": "1.0",
        "complete": True,
        "revision": revision,
        "dimensions": dimensions,
        "methods": method_results,
    }


def load_records(directory: Path) -> list[dict[str, object]]:
    """Load one JSON object from each raw JSONL cell file."""
    records = []
    for path in sorted(directory.glob("*.jsonl")):
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        if len(lines) != 1:
            raise ValueError(f"{path} must contain exactly one JSON object")
        record = json.loads(lines[0])
        if not isinstance(record, dict):
            raise TypeError(f"{path} must contain a JSON object")
        records.append(record)
    return records


def main() -> None:
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=script_directory / "manifest.json"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = load_manifest(arguments.manifest)
    parity = json.loads(arguments.parity.read_text(encoding="utf-8"))
    result = aggregate_records(load_records(arguments.input_dir), parity, manifest)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
