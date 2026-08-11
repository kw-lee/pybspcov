from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).parents[1]
ORCHESTRATOR_PATH = PROJECT_ROOT / "benchmarks" / "run_v0_1_baseline.py"
RENDERER_PATH = PROJECT_ROOT / "benchmarks" / "render_v0_1_baseline.py"


def _load_module(path: Path, name: str) -> ModuleType:
    assert path.exists(), f"{path} is required"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _python_record(
    *,
    estimator: str,
    device: str,
    dtype: str,
    dimension: int,
) -> dict[str, object]:
    execution_model = "parallel" if device == "cpu" else "sequential"
    repetitions = []
    for index in range(10):
        chain_seconds = [2.0 + index / 10.0]
        if device == "gpu":
            chain_seconds = [2.0 + index / 10.0] * 4
        total = sum(chain_seconds)
        normalized = total / 4
        repetitions.append(
            {
                "repetition": index,
                "execution_model": execution_model,
                "raw_wall_seconds": chain_seconds,
                "total_wall_seconds": total,
                "normalized_wall_seconds_per_chain": normalized,
                "chains_per_second": 1.0 / normalized,
                "retained_draws": 200,
                "posterior_mean_finite": True,
                "posterior_mean_symmetric": True,
                "posterior_mean_spd": True,
                "truth_relative_frobenius_error": 0.18 + index / 1000.0,
                "accepted_sweeps": 400,
                "rejected_sweeps": 0,
            }
        )
    return {
        "benchmark": f"{estimator}-public-scaling",
        "schema_version": "2.0",
        "estimator": estimator,
        "dimension": dimension,
        "density": 0.05,
        "n_observations": dimension * 3,
        "burnin": 50,
        "samples": 50,
        "chain_count": 4,
        "repetitions": 10,
        "seed": 20260803,
        "device": device,
        "dtype": dtype,
        "execution_model": execution_model,
        "fixture_sha256": f"{dimension:064x}",
        "git": {"revision": "abc123", "dirty": False},
        "environment": {"python": "3.12", "jax": "0.11"},
        "compile_plus_execution_seconds": 5.0,
        "measured_repetitions": repetitions,
        "timing_summary": {
            "median": 0.6125 if device == "cpu" else 2.45,
            "q1": 0.55625 if device == "cpu" else 2.225,
            "q3": 0.66875 if device == "cpu" else 2.675,
            "min": 0.5 if device == "cpu" else 2.0,
            "max": 0.725 if device == "cpu" else 2.9,
        },
    }


def _all_python_records() -> list[dict[str, object]]:
    return [
        _python_record(
            estimator=estimator,
            device=device,
            dtype=dtype,
            dimension=dimension,
        )
        for estimator in ("bm", "sbm")
        for device in ("cpu", "gpu")
        for dtype in ("float32", "float64")
        for dimension in (25, 50, 100, 200)
    ]


def _r_document() -> dict[str, object]:
    repetitions = [
        {
            "repetition": index,
            "total_wall_seconds": 8.0 + index / 10.0,
            "normalized_wall_seconds_per_chain": 2.0 + index / 40.0,
            "chains_per_second": 1.0 / (2.0 + index / 40.0),
            "retained_draws": 200,
            "posterior_mean_finite": True,
            "posterior_mean_symmetric": True,
            "posterior_mean_spd": True,
            "truth_relative_frobenius_error": 0.2 + index / 1000.0,
        }
        for index in range(10)
    ]
    normalized = [
        float(repetition["normalized_wall_seconds_per_chain"])
        for repetition in repetitions
    ]
    return {
        "schema_version": "2.0",
        "configuration": {
            "burnin": 50,
            "chain_count": 4,
            "n_samples": 50,
            "repetitions": 10,
        },
        "fixture": {"fixture_sha256": f"{200:064x}"},
        "r": {
            "package_version": "1.0.3",
            "measured_repetitions": repetitions,
            "timing_summary": {
                "median": 2.1125,
                "q1": 2.05625,
                "q3": 2.16875,
                "min": min(normalized),
                "max": max(normalized),
            },
        },
    }


def test_lane_commands_pin_cpu_and_gpu_to_separate_numa_nodes(tmp_path: Path) -> None:
    """Catch co-scheduling both benchmark lanes on the GPU-local socket."""
    orchestrator = _load_module(ORCHESTRATOR_PATH, "run_v0_1_baseline")
    configuration = orchestrator.BenchmarkConfiguration(
        project_root=PROJECT_ROOT,
        output_dir=tmp_path / "output",
        r_library=tmp_path / "r-library",
        python_executable=Path("/venv/bin/python"),
        dimensions=(25, 50, 100, 200),
        density=0.05,
        n_factor=3,
        burnin=50,
        samples=50,
        chains=4,
        repetitions=10,
        seed=20260803,
    )

    lanes = orchestrator.build_lane_commands(configuration)

    assert len(lanes.cpu_python) == 4
    assert len(lanes.gpu_python) == 4
    assert all(
        command[:3] == ("numactl", "--cpunodebind=1", "--membind=1")
        for command in (*lanes.cpu_python, lanes.r)
    )
    assert all(
        command[:3] == ("numactl", "--cpunodebind=0", "--membind=0")
        for command in lanes.gpu_python
    )
    assert all(
        "--chains" in command and "10" in command for command in lanes.cpu_python
    )
    assert "--repetitions" in lanes.r


def test_renderer_requires_all_32_long_chain_cells_and_ten_runs() -> None:
    """Catch accepting missing cells, old short chains, or fewer repetitions."""
    renderer = _load_module(RENDERER_PATH, "render_v0_1_baseline")
    records = _all_python_records()

    validated = renderer.validate_python_records(records)

    assert len(validated) == 32
    assert all(len(record["measured_repetitions"]) == 10 for record in validated)

    records[0] = {**records[0], "burnin": 1}
    with pytest.raises(ValueError, match="burnin=50"):
        renderer.validate_python_records(records)


def test_plot_series_preserve_every_raw_timing_and_error_point() -> None:
    """Catch plotting summaries instead of all ten observed repetitions."""
    renderer = _load_module(RENDERER_PATH, "render_v0_1_baseline")
    records = renderer.validate_python_records(_all_python_records())

    series = renderer.build_plot_series(records)

    assert len(series) == 32
    assert all(len(item.wall_seconds) == 10 for item in series)
    assert all(len(item.errors) == 10 for item in series)


def test_render_writes_human_readable_report_and_raw_point_plots(
    tmp_path: Path,
) -> None:
    """Catch a benchmark that validates but cannot produce reviewable artifacts."""
    renderer = _load_module(RENDERER_PATH, "render_v0_1_baseline")
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    records = _all_python_records()
    for filename in renderer.PYTHON_FILENAMES:
        estimator, device, dtype = filename.removesuffix(".jsonl").split("-")
        selected = [
            record
            for record in records
            if record["estimator"] == estimator
            and record["device"] == device
            and record["dtype"] == dtype
        ]
        (input_dir / filename).write_text(
            "".join(json.dumps(record) + "\n" for record in selected),
            encoding="utf-8",
        )
    (input_dir / "p200-r-only.json").write_text(
        json.dumps(_r_document()),
        encoding="utf-8",
    )
    (input_dir / "lane-metadata.json").write_text(
        json.dumps({"topology": {"cpu_lane_node": 1, "gpu_lane_node": 0}}),
        encoding="utf-8",
    )

    renderer.render(input_dir, output_dir)

    report = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "50 burn-in sweeps" in report
    assert "p=200 comparison with R bspcov 1.0.3" in report
    assert (output_dir / "wall-time-boxplots.svg").is_file()
    assert (output_dir / "error-boxplots.svg").is_file()
    comparison = json.loads(
        (output_dir / "p200-r-comparison.json").read_text(encoding="utf-8")
    )
    assert len(comparison["python"]) == 4
