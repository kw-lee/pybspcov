#!/usr/bin/env python3
"""Render the validated R/Python comparison into README.md markers."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

START_MARKER = "<!-- r-python-benchmark:start -->"
END_MARKER = "<!-- r-python-benchmark:end -->"
METHOD_LABELS = {
    "bm": "BM",
    "sbm": "SBM",
    "bandppp": "BandPPP",
    "thresholdppp": "ThresholdPPP",
}


def render_section(
    summary: Mapping[str, object], *, baseline: str, execution_note: str
) -> str:
    """Render only a complete, validated four-method summary."""
    if not execution_note.strip():
        raise ValueError("README rendering requires an execution note")
    if summary.get("schema_version") != "1.0" or summary.get("complete") is not True:
        raise ValueError("README rendering requires a complete schema 1.0 summary")
    methods = summary.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != set(METHOD_LABELS):
        raise ValueError("README rendering requires all four methods")
    revision = str(summary.get("revision"))
    lines = [
        "### R `bspcov` comparison",
        "",
        (
            "All four float64 parity gates passed against `bspcov` 1.0.3. "
            "The headline compares total wall time for R CPU 8-worker execution "
            "with Python GPU vmap-8 execution on the same inputs."
        ),
        "",
        execution_note.strip(),
        "",
        "| Method | Validated Python dtype | R CPU 8-worker (s) | Python GPU vmap-8 (s) | p=200 speedup | p=50/100/200 geometric mean |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for method, label in METHOD_LABELS.items():
        result = methods[method]
        large = result["large"]
        lines.append(
            f"| {label} | {result['headline_dtype']} | "
            f"{large['r_optimized_seconds']:.3f} | "
            f"{large['python_gpu_seconds']:.3f} | "
            f"{large['optimized_speedup']:.3f}x | "
            f"{result['geometric_mean_speedup']:.3f}x |"
        )
    lines.extend(
        [
            "",
            (
                "The matching single-core float64 baseline separates implementation "
                "differences from GPU and multi-chain acceleration."
            ),
            "",
            "| Method | R CPU (s) | Python CPU (s) | Python speedup | Cold optimized speedup |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for method, label in METHOD_LABELS.items():
        large = methods[method]["large"]
        lines.append(
            f"| {label} | {large['r_cpu_baseline_seconds']:.3f} | "
            f"{large['python_cpu_baseline_seconds']:.3f} | "
            f"{large['cpu_baseline_speedup']:.3f}x | "
            f"{large['cold_speedup']:.3f}x |"
        )
    report_url = (
        "https://github.com/kw-lee/pybspcov/blob/main/benchmarks/baselines/"
        f"{baseline}/r-python/README.md"
    )
    lines.extend(
        [
            "",
            (
                f"These host- and workload-specific results were recorded at revision "
                f"`{revision}`. They do not establish universal performance superiority."
            ),
            "",
            f"[Full protocol, raw timings, environment, and limitations]({report_url})",
            "",
        ]
    )
    return "\n".join(lines)


def sync_readme(path: Path, rendered: str, *, check: bool) -> None:
    """Replace only the generated README region or detect staleness."""
    current = path.read_text(encoding="utf-8")
    if current.count(START_MARKER) != 1 or current.count(END_MARKER) != 1:
        raise ValueError(
            "README must contain exactly one R/Python benchmark marker pair"
        )
    prefix, remainder = current.split(START_MARKER, maxsplit=1)
    _, suffix = remainder.split(END_MARKER, maxsplit=1)
    updated = f"{prefix}{START_MARKER}\n{rendered}{END_MARKER}{suffix}"
    if current == updated:
        return
    if check:
        raise RuntimeError(f"{path} is out of date; rerun render_readme.py")
    path.write_text(updated, encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--execution-note", required=True)
    parser.add_argument("--readme", type=Path, default=project_root / "README.md")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    summary = json.loads(arguments.summary.read_text(encoding="utf-8"))
    sync_readme(
        arguments.readme,
        render_section(
            summary,
            baseline=arguments.baseline,
            execution_note=arguments.execution_note,
        ),
        check=arguments.check,
    )


if __name__ == "__main__":
    main()
