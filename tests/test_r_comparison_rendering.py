import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[1]
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks" / "r_comparison"
sys.path.insert(0, str(BENCHMARK_DIR))


def _module(filename: str, name: str):
    path = BENCHMARK_DIR / filename
    assert path.is_file(), f"{path.relative_to(PROJECT_ROOT)} is required"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _parity(*, float32: str = "pass") -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "revision": "a" * 40,
        "methods": {
            method: {"float64": "pass", "float32": float32}
            for method in ("bm", "sbm", "bandppp", "thresholdppp")
        },
    }


def _record(cell, *, seconds: float) -> dict[str, object]:
    fixture_digit = {50: "1", 100: "2", 200: "3"}[cell.dimension]
    samples = 50 if cell.method in {"bm", "sbm"} else 100
    return {
        "schema_version": "1.0",
        "method": cell.method,
        "dimension": cell.dimension,
        "n_observations": 3 * cell.dimension,
        "seed": 20260803,
        "fixture_sha256": fixture_digit * 64,
        "implementation": cell.implementation,
        "version": "1.0.3" if cell.implementation == "bspcov" else "0.1.0.dev0",
        "device": cell.device,
        "actual_platform": "cuda" if cell.device == "gpu" else "cpu",
        "dtype": cell.dtype,
        "execution": (
            "vmap"
            if cell.device == "gpu"
            else "parallel"
            if cell.parallelism > 1
            else "single"
        ),
        "configuration": cell.configuration,
        "parallelism": cell.parallelism,
        "cpu_cores": cell.cpu_cores,
        "retained_draws": samples * cell.parallelism,
        "cold_end_to_end_seconds": (
            1.5 * seconds if cell.implementation == "bspcov" else 3.0 * seconds
        ),
        "warm_seconds": [0.99 * seconds, seconds, 1.01 * seconds],
        "posterior_mean_finite": True,
        "posterior_mean_symmetric": True,
        "posterior_mean_spd": True,
        "rejected_sweeps": 0,
        "git_revision": "a" * 40,
        "git_dirty": False,
    }


def _records() -> list[dict[str, object]]:
    matrix = _module("run_matrix.py", "r_comparison_test_matrix")
    core = _module("core.py", "r_comparison_test_core")
    manifest = core.load_manifest(BENCHMARK_DIR / "manifest.json")
    records = []
    for cell in matrix.build_cells(manifest):
        reference = cell.dimension / 10.0
        if cell.configuration == "optimized":
            if cell.implementation == "bspcov":
                seconds = reference
            elif cell.dtype == "float32":
                seconds = reference / 4.0
            else:
                seconds = reference / 2.0
        elif cell.implementation == "bspcov":
            seconds = reference * 2.0
        else:
            seconds = reference
        records.append(_record(cell, seconds=seconds))
    return records


def test_aggregate_selects_validated_fast_dtype_and_all_size_geomean() -> None:
    aggregate = _module("aggregate.py", "r_comparison_aggregate")
    core = _module("core.py", "r_comparison_aggregate_core")
    manifest = core.load_manifest(BENCHMARK_DIR / "manifest.json")

    result = aggregate.aggregate_records(_records(), _parity(), manifest)

    assert result["revision"] == "a" * 40
    assert result["complete"] is True
    bm = result["methods"]["bm"]
    assert bm["headline_dtype"] == "float32"
    assert bm["large"] == {
        "r_optimized_seconds": 20.0,
        "python_gpu_seconds": 5.0,
        "optimized_speedup": 4.0,
        "cold_speedup": 2.0,
        "r_cpu_baseline_seconds": 40.0,
        "python_cpu_baseline_seconds": 20.0,
        "cpu_baseline_speedup": 2.0,
    }
    assert bm["geometric_mean_speedup"] == pytest.approx(4.0)


def test_aggregate_falls_back_to_float64_when_float32_parity_fails() -> None:
    aggregate = _module("aggregate.py", "r_comparison_aggregate_f64")
    core = _module("core.py", "r_comparison_aggregate_core_f64")
    manifest = core.load_manifest(BENCHMARK_DIR / "manifest.json")

    result = aggregate.aggregate_records(_records(), _parity(float32="fail"), manifest)

    bm = result["methods"]["bm"]
    assert bm["headline_dtype"] == "float64"
    assert bm["large"]["optimized_speedup"] == pytest.approx(2.0)
    assert bm["geometric_mean_speedup"] == pytest.approx(2.0)


def test_aggregate_rejects_missing_cells_and_fixture_mismatches() -> None:
    aggregate = _module("aggregate.py", "r_comparison_aggregate_reject")
    core = _module("core.py", "r_comparison_aggregate_core_reject")
    manifest = core.load_manifest(BENCHMARK_DIR / "manifest.json")
    records = _records()

    with pytest.raises(ValueError, match="missing benchmark cells"):
        aggregate.aggregate_records(records[:-1], _parity(), manifest)

    mismatched = [dict(record) for record in records]
    mismatched[1]["fixture_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="fixture"):
        aggregate.aggregate_records(mismatched, _parity(), manifest)

    stale_parity = {**_parity(), "revision": "b" * 40}
    with pytest.raises(ValueError, match="parity revision"):
        aggregate.aggregate_records(records, stale_parity, manifest)


def test_rendered_readme_leads_with_large_optimized_result_and_discloses_modes() -> (
    None
):
    aggregate = _module("aggregate.py", "r_comparison_render_aggregate")
    renderer = _module("render_readme.py", "r_comparison_renderer")
    core = _module("core.py", "r_comparison_render_core")
    manifest = core.load_manifest(BENCHMARK_DIR / "manifest.json")
    summary = aggregate.aggregate_records(_records(), _parity(), manifest)

    rendered = renderer.render_section(summary, baseline="test-baseline")

    assert "All four float64 parity gates passed" in rendered
    assert "R CPU 8-worker" in rendered
    assert "Python GPU vmap-8" in rendered
    assert "| BM | float32 | 20.000 | 5.000 | 4.000x | 4.000x |" in rendered
    assert "single-core float64" in rendered
    assert "host- and workload-specific" in rendered
    assert "test-baseline" in rendered


def test_sync_readme_replaces_only_generated_markers_and_checks_staleness(
    tmp_path: Path,
) -> None:
    renderer = _module("render_readme.py", "r_comparison_renderer_sync")
    readme = tmp_path / "README.md"
    readme.write_text(
        "before\n<!-- r-python-benchmark:start -->\nold\n"
        "<!-- r-python-benchmark:end -->\nafter\n",
        encoding="utf-8",
    )

    renderer.sync_readme(readme, "new\n", check=False)

    assert readme.read_text(encoding="utf-8") == (
        "before\n<!-- r-python-benchmark:start -->\nnew\n"
        "<!-- r-python-benchmark:end -->\nafter\n"
    )
    renderer.sync_readme(readme, "new\n", check=True)
    with pytest.raises(RuntimeError, match="out of date"):
        renderer.sync_readme(readme, "different\n", check=True)
