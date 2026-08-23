from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).parents[1]
RENDERER_PATH = PROJECT_ROOT / "benchmarks" / "render_docs_benchmarks.py"


def _load_renderer() -> ModuleType:
    assert RENDERER_PATH.exists(), f"{RENDERER_PATH} is required"
    spec = importlib.util.spec_from_file_location(
        "render_docs_benchmarks", RENDERER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(
    *,
    estimator: str,
    device: str,
    dtype: str,
    median: float,
) -> dict[str, object]:
    return {
        "schema_version": "2.0",
        "estimator": estimator,
        "device": device,
        "dtype": dtype,
        "dimension": 200,
        "n_observations": 600,
        "density": 0.05,
        "burnin": 1,
        "samples": 2,
        "chain_count": 1,
        "repetitions": 3,
        "git": {"revision": "abc123", "dirty": False},
        "environment": {
            "python": "3.13.6",
            "jax": "0.11.0",
            "jaxlib": "0.11.0",
        },
        "timing_summary": {"median": median},
    }


def _complete_records() -> list[dict[str, object]]:
    return [
        _record(estimator="bm", device="cpu", dtype="float32", median=4.0),
        _record(estimator="bm", device="gpu", dtype="float32", median=2.0),
        _record(estimator="bm", device="cpu", dtype="float64", median=6.0),
        _record(estimator="bm", device="gpu", dtype="float64", median=3.0),
        _record(estimator="sbm", device="cpu", dtype="float32", median=5.0),
        _record(estimator="sbm", device="gpu", dtype="float32", median=4.0),
        _record(estimator="sbm", device="cpu", dtype="float64", median=9.0),
        _record(estimator="sbm", device="gpu", dtype="float64", median=6.0),
    ]


def test_render_summary_pairs_cpu_and_gpu_and_computes_speedup() -> None:
    """Catch selecting the wrong p=200 row or reversing CPU/GPU speedup."""
    renderer = _load_renderer()

    rendered = renderer.render_summary(_complete_records(), baseline="0.1.0.dev1")

    assert "Cached baseline `0.1.0.dev1`" in rendered
    assert "revision `abc123`" in rendered
    assert "| BM | float32 | 4.000 | 2.000 | 2.000x |" in rendered
    assert "| BM | float64 | 6.000 | 3.000 | 2.000x |" in rendered
    assert "| SBM | float32 | 5.000 | 4.000 | 1.250x |" in rendered
    assert "| SBM | float64 | 9.000 | 6.000 | 1.500x |" in rendered
    assert "1 burn-in sweep, 2 retained samples, 1 chain, and 3 repetitions" in rendered
    assert "Python 3.13.6, JAX/JAXLIB 0.11.0" in rendered
    assert not any(line.endswith(" ") for line in rendered.splitlines())
    assert (
        "https://github.com/kw-lee/pybspcov/blob/main/benchmarks/"
        "baselines/0.1.0.dev1/README.md"
    ) in rendered


def test_render_summary_rejects_an_incomplete_p200_matrix() -> None:
    """Catch publishing a table after one required device result disappears."""
    renderer = _load_renderer()

    with pytest.raises(ValueError, match="missing p=200 benchmark cells"):
        renderer.render_summary(_complete_records()[:-1], baseline="0.1.0.dev1")


def test_load_records_reads_jsonl_objects(tmp_path: Path) -> None:
    """Catch silently skipping a cached JSONL record used by the table."""
    renderer = _load_renderer()
    records = _complete_records()
    (tmp_path / "first.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records[:4]) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "second.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records[4:]) + "\n",
        encoding="utf-8",
    )

    loaded = renderer.load_records(tmp_path)

    assert loaded == records


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
