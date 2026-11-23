from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).parents[1]
RENDERER_PATH = PROJECT_ROOT / "benchmarks" / "render_docs_benchmarks.py"
METHODS = ("bm", "sbm", "bandppp", "thresholdppp")


def _load_renderer() -> ModuleType:
    assert RENDERER_PATH.exists(), f"{RENDERER_PATH} is required"
    spec = importlib.util.spec_from_file_location(
        "render_docs_benchmarks", RENDERER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _method(scale: float) -> dict[str, object]:
    return {
        "geometric_mean_speedup": 1.5 * scale,
        "headline_dtype": "float32",
        "large": {
            "cold_speedup": 1.25 * scale,
            "cpu_baseline_speedup": 3.0 * scale,
            "optimized_speedup": 2.0 * scale,
            "python_cpu_baseline_seconds": 2.0,
            "python_gpu_seconds": 4.0,
            "r_cpu_baseline_seconds": 6.0 * scale,
            "r_optimized_seconds": 8.0 * scale,
        },
    }


def _summary() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "complete": True,
        "dimensions": [50, 100, 200],
        "revision": "abc123",
        "methods": {
            method: _method(float(index))
            for index, method in enumerate(METHODS, start=1)
        },
    }


def test_render_summary_reports_both_comparisons_for_every_method() -> None:
    """Catch swapping the GPU and single-core columns or reversing a speedup."""
    renderer = _load_renderer()

    rendered = renderer.render_summary(_summary(), baseline="0.1.0.dev0")

    assert "Cached baseline `0.1.0.dev0`" in rendered
    assert "revision `abc123`" in rendered
    assert "| BM | float32 | 8.000 | 4.000 | 2.000x | 1.500x |" in rendered
    assert "| SBM | float32 | 16.000 | 4.000 | 4.000x | 3.000x |" in rendered
    assert "| BandPPP | 18.000 | 2.000 | 9.000x |" in rendered
    assert "| ThresholdPPP | 24.000 | 2.000 | 12.000x |" in rendered
    assert not any(line.endswith(" ") for line in rendered.splitlines())
    assert (
        "https://github.com/kw-lee/pybspcov/blob/main/benchmarks/"
        "baselines/0.1.0.dev0/r-python/README.md"
    ) in rendered


def test_render_summary_rejects_a_summary_missing_a_method() -> None:
    """Catch publishing a table after one method's result disappears."""
    renderer = _load_renderer()
    summary = _summary()
    methods = summary["methods"]
    assert isinstance(methods, dict)
    del methods["thresholdppp"]

    with pytest.raises(ValueError, match="all four methods"):
        renderer.render_summary(summary, baseline="0.1.0.dev0")


def test_render_summary_rejects_an_incomplete_summary() -> None:
    """Catch rendering a run whose parity or timing gates did not all pass."""
    renderer = _load_renderer()
    summary = _summary()
    summary["complete"] = False

    with pytest.raises(ValueError, match="complete schema 1.0 summary"):
        renderer.render_summary(summary, baseline="0.1.0.dev0")


def test_load_summary_reads_the_tracked_json_object(tmp_path: Path) -> None:
    """Catch reading the summary from the wrong file or as the wrong type."""
    renderer = _load_renderer()
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(_summary()), encoding="utf-8")

    assert renderer.load_summary(path) == _summary()

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="JSON object"):
        renderer.load_summary(path)


def test_sync_output_check_rejects_stale_generated_document(tmp_path: Path) -> None:
    """Catch CI accepting generated documentation that no longer matches data."""
    renderer = _load_renderer()
    output = tmp_path / "benchmark-summary.md"
    output.write_text("stale\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="out of date"):
        renderer.sync_output(output, "fresh\n", check=True)

    assert output.read_text(encoding="utf-8") == "stale\n"

    renderer.sync_output(output, "fresh\n", check=False)

    assert output.read_text(encoding="utf-8") == "fresh\n"
