#!/usr/bin/env python3
"""Generate fixtures and execute the pre-registered R/Python timing matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

from core import load_manifest
from fixtures import generate_fixture, write_fixture


class Cell(NamedTuple):
    method: str
    dimension: int
    implementation: str
    device: str
    dtype: str
    configuration: str
    parallelism: int
    cpu_cores: int

    @property
    def key(self) -> str:
        return "-".join(
            (
                self.method,
                f"p{self.dimension}",
                self.implementation,
                self.device,
                self.dtype,
                self.configuration,
            )
        )


def build_cells(manifest: Mapping[str, Any]) -> list[Cell]:
    """Return the complete pre-registered matrix with no adaptive cell choice."""
    cells: list[Cell] = []
    optimized = manifest["optimized"]
    baseline = manifest["cpu_baseline"]
    for method in manifest["methods"]:
        for dimension in manifest["dimensions"]:
            cells.append(
                Cell(
                    method,
                    dimension,
                    "bspcov",
                    "cpu",
                    "float64",
                    "optimized",
                    optimized["parallelism"],
                    optimized["cpu_cores"],
                )
            )
            for dtype in optimized["python_dtypes"]:
                cells.append(
                    Cell(
                        method,
                        dimension,
                        "pybspcov",
                        "gpu",
                        dtype,
                        "optimized",
                        optimized["parallelism"],
                        optimized["cpu_cores"],
                    )
                )
            for implementation in ("bspcov", "pybspcov"):
                cells.append(
                    Cell(
                        method,
                        dimension,
                        implementation,
                        "cpu",
                        baseline["dtype"],
                        "cpu_baseline",
                        baseline["parallelism"],
                        baseline["cpu_cores"],
                    )
                )
    return cells


def generate_all_fixtures(manifest: Mapping[str, Any], root: Path) -> None:
    """Write one sparse and one banded input for every registered dimension."""
    for dimension in manifest["dimensions"]:
        n_observations = int(manifest["n_factor"] * dimension)
        seed = int(manifest["seed"] + dimension)
        sparse = generate_fixture(
            dimension=dimension,
            n_observations=n_observations,
            seed=seed,
            kind="sparse",
            density=float(manifest["sparse_density"]),
        )
        write_fixture(root / f"sparse-p{dimension}", sparse)
        divisor = int(manifest["methods"]["bandppp"]["bandwidth_divisor"])
        banded = generate_fixture(
            dimension=dimension,
            n_observations=n_observations,
            seed=seed,
            kind="banded",
            bandwidth=max(1, dimension // divisor),
        )
        write_fixture(root / f"banded-p{dimension}", banded)


def _fixture_directory(root: Path, cell: Cell) -> Path:
    kind = "banded" if cell.method == "bandppp" else "sparse"
    return root / f"{kind}-p{cell.dimension}"


def command_for_cell(
    cell: Cell,
    *,
    script_directory: Path,
    manifest_path: Path,
    fixture_root: Path,
    output_directory: Path,
) -> list[str]:
    """Build one resource-pinned subprocess command."""
    cpu_list = "0" if cell.cpu_cores == 1 else f"0-{cell.cpu_cores - 1}"
    output = output_directory / f"{cell.key}.jsonl"
    fixture = _fixture_directory(fixture_root, cell)
    common = [
        "--manifest",
        str(manifest_path),
        "--fixture-dir",
        str(fixture),
        "--method",
        cell.method,
        "--parallelism",
        str(cell.parallelism),
        "--configuration",
        cell.configuration,
        "--cpu-cores",
        str(cell.cpu_cores),
        "--output",
        str(output),
    ]
    if cell.implementation == "pybspcov":
        uv = ["uv", "run", "--frozen"]
        if cell.device == "gpu":
            uv.extend(("--extra", "cuda12"))
        return [
            "taskset",
            "-c",
            cpu_list,
            *uv,
            "python",
            str(script_directory / "run_pybspcov.py"),
            *common,
            "--dtype",
            cell.dtype,
            "--device",
            cell.device,
        ]
    return [
        "taskset",
        "-c",
        cpu_list,
        "Rscript",
        "--vanilla",
        str(script_directory / "run_bspcov.R"),
        *common,
    ]


def main() -> None:
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=script_directory / "manifest.json")
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generate-fixtures", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    manifest = load_manifest(arguments.manifest)
    if arguments.generate_fixtures:
        generate_all_fixtures(manifest, arguments.fixture_root)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
        }
    )
    for cell in build_cells(manifest):
        command = command_for_cell(
            cell,
            script_directory=script_directory,
            manifest_path=arguments.manifest,
            fixture_root=arguments.fixture_root,
            output_directory=arguments.output_dir,
        )
        if arguments.dry_run:
            print(json.dumps({"key": cell.key, "command": command}))
        else:
            subprocess.run(command, check=True, env=environment)


if __name__ == "__main__":
    main()
