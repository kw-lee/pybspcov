"""Validate, summarize, and plot the v0.1 long-chain baseline records."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import NamedTuple

import numpy as np

ESTIMATORS = ("bm", "sbm")
DEVICES = ("cpu", "gpu")
DTYPES = ("float32", "float64")
DIMENSIONS = (25, 50, 100, 200)
PYTHON_FILENAMES = tuple(
    f"{estimator}-{device}-{dtype}.jsonl"
    for estimator, device, dtype in product(ESTIMATORS, DEVICES, DTYPES)
)


class PlotSeries(NamedTuple):
    """Ten raw wall-time and error observations for one benchmark cell."""

    estimator: str
    device: str
    dtype: str
    dimension: int
    wall_seconds: tuple[float, ...]
    errors: tuple[float, ...]


def _summary(values: Sequence[float]) -> dict[str, float]:
    q1, q3 = np.quantile(np.asarray(values, dtype=np.float64), [0.25, 0.75])
    return {
        "median": float(statistics.median(values)),
        "q1": float(q1),
        "q3": float(q3),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def load_python_records(directory: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for filename in PYTHON_FILENAMES:
        path = directory / filename
        if not path.is_file():
            raise ValueError(f"missing Python result file: {filename}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    raise TypeError(f"{filename} must contain JSON objects")
                records.append(parsed)
    return records


def _require_close(actual: object, expected: float, label: str) -> None:
    if not np.isclose(float(actual), expected, rtol=1e-12, atol=0.0):
        raise ValueError(f"invalid {label}")


def validate_python_records(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Require the complete 32-cell, ten-repetition long-chain matrix."""
    expected_cells = set(product(ESTIMATORS, DEVICES, DTYPES, DIMENSIONS))
    validated: list[dict[str, object]] = []
    observed_cells: set[tuple[str, str, str, int]] = set()
    fixture_hashes: dict[int, str] = {}
    for source in records:
        record = dict(source)
        cell = (
            str(record.get("estimator")),
            str(record.get("device")),
            str(record.get("dtype")),
            int(record.get("dimension", -1)),
        )
        if cell not in expected_cells or cell in observed_cells:
            raise ValueError(f"unexpected or duplicate Python cell: {cell}")
        observed_cells.add(cell)
        _, device, _, dimension = cell
        if record.get("schema_version") != "2.0":
            raise ValueError("Python records must use schema 2.0")
        if int(record.get("burnin", -1)) != 50:
            raise ValueError("every Python cell must use burnin=50")
        if int(record.get("samples", -1)) != 50:
            raise ValueError("every Python cell must use samples=50")
        if int(record.get("chain_count", -1)) != 4:
            raise ValueError("every Python cell must use chain_count=4")
        if int(record.get("repetitions", -1)) != 10:
            raise ValueError("every Python cell must use repetitions=10")
        if int(record.get("n_observations", -1)) != dimension * 3:
            raise ValueError("every Python cell must use n=3p")
        if float(record.get("density", -1.0)) != 0.05:
            raise ValueError("every Python cell must use density=0.05")
        if int(record.get("seed", -1)) != 20260803:
            raise ValueError("every Python cell must use seed=20260803")
        expected_model = "parallel" if device == "cpu" else "sequential"
        if record.get("execution_model") != expected_model:
            raise ValueError(f"invalid execution model for {device}")
        fixture_hash = str(record.get("fixture_sha256"))
        previous_hash = fixture_hashes.setdefault(dimension, fixture_hash)
        if fixture_hash != previous_hash:
            raise ValueError(f"fixture hash mismatch at p={dimension}")
        git = record.get("git")
        if not isinstance(git, dict) or git.get("dirty") is not False:
            raise ValueError("Python records require clean git provenance")

        repetitions = record.get("measured_repetitions")
        if not isinstance(repetitions, list) or len(repetitions) != 10:
            raise ValueError("every Python cell must contain ten measured repetitions")
        normalized_values: list[float] = []
        for index, repetition_source in enumerate(repetitions):
            if not isinstance(repetition_source, dict):
                raise TypeError("Python repetitions must be JSON objects")
            repetition = repetition_source
            if int(repetition.get("repetition", -1)) != index:
                raise ValueError(
                    "Python repetition indices must be contiguous from zero"
                )
            if repetition.get("execution_model") != expected_model:
                raise ValueError("Python repetition execution model mismatch")
            raw = repetition.get("raw_wall_seconds")
            expected_raw_count = 1 if device == "cpu" else 4
            if not isinstance(raw, list) or len(raw) != expected_raw_count:
                raise ValueError(
                    f"{device} repetitions require {expected_raw_count} raw times"
                )
            raw_values = [float(value) for value in raw]
            total = float(repetition["total_wall_seconds"])
            _require_close(total, sum(raw_values), "Python total wall time")
            normalized = float(repetition["normalized_wall_seconds_per_chain"])
            _require_close(normalized, total / 4.0, "Python normalized wall time")
            _require_close(
                repetition["chains_per_second"],
                1.0 / normalized,
                "Python throughput",
            )
            if int(repetition.get("retained_draws", -1)) != 200:
                raise ValueError(
                    "every Python repetition must contain 200 retained draws"
                )
            if not all(
                repetition.get(field) is True
                for field in (
                    "posterior_mean_finite",
                    "posterior_mean_symmetric",
                    "posterior_mean_spd",
                )
            ):
                raise ValueError(
                    "every Python repetition requires a valid posterior mean"
                )
            error = float(repetition["truth_relative_frobenius_error"])
            if not np.isfinite(error) or error < 0.0:
                raise ValueError("Python errors must be finite and non-negative")
            normalized_values.append(normalized)
        timing_summary = record.get("timing_summary")
        if not isinstance(timing_summary, dict):
            raise TypeError("Python timing_summary must be an object")
        for label, expected in _summary(normalized_values).items():
            _require_close(timing_summary[label], expected, f"Python timing {label}")
        validated.append(record)

    missing = expected_cells - observed_cells
    if missing:
        raise ValueError(f"missing Python cells: {sorted(missing)}")
    return sorted(
        validated,
        key=lambda record: (
            str(record["estimator"]),
            str(record["dtype"]),
            int(record["dimension"]),
            str(record["device"]),
        ),
    )


def build_plot_series(records: Sequence[Mapping[str, object]]) -> list[PlotSeries]:
    """Preserve all measured points in plotting order."""
    series: list[PlotSeries] = []
    for record in records:
        repetitions = record["measured_repetitions"]
        if not isinstance(repetitions, list):
            raise TypeError("measured_repetitions must be a list")
        series.append(
            PlotSeries(
                estimator=str(record["estimator"]),
                device=str(record["device"]),
                dtype=str(record["dtype"]),
                dimension=int(record["dimension"]),
                wall_seconds=tuple(
                    float(repetition["normalized_wall_seconds_per_chain"])
                    for repetition in repetitions
                ),
                errors=tuple(
                    float(repetition["truth_relative_frobenius_error"])
                    for repetition in repetitions
                ),
            )
        )
    return series


def validate_r_result(
    document: Mapping[str, object],
    python_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Require ten valid R repetitions on the Python p=200 fixture."""
    result = dict(document)
    if result.get("schema_version") != "2.0":
        raise ValueError("R result must use schema 2.0")
    configuration = result.get("configuration")
    if not isinstance(configuration, dict) or configuration != {
        "burnin": 50,
        "chain_count": 4,
        "n_samples": 50,
        "repetitions": 10,
    }:
        raise ValueError("R result must use 50/50, four chains, and ten repetitions")
    fixture = result.get("fixture")
    if not isinstance(fixture, dict):
        raise TypeError("R fixture must be an object")
    hashes = {
        str(record["fixture_sha256"])
        for record in python_records
        if int(record["dimension"]) == 200
    }
    if hashes != {str(fixture.get("fixture_sha256"))}:
        raise ValueError("R and Python p=200 fixture hashes must match")
    r_result = result.get("r")
    if not isinstance(r_result, dict) or r_result.get("package_version") != "1.0.3":
        raise ValueError("R result requires bspcov 1.0.3")
    repetitions = r_result.get("measured_repetitions")
    if not isinstance(repetitions, list) or len(repetitions) != 10:
        raise ValueError("R result requires ten measured repetitions")
    for index, repetition in enumerate(repetitions):
        if not isinstance(repetition, dict) or repetition.get("repetition") != index:
            raise ValueError("R repetition indices must be contiguous from zero")
        if repetition.get("retained_draws") != 200:
            raise ValueError("every R repetition must contain 200 retained draws")
        if not all(
            repetition.get(field) is True
            for field in (
                "posterior_mean_finite",
                "posterior_mean_symmetric",
                "posterior_mean_spd",
            )
        ):
            raise ValueError("every R repetition requires a valid posterior mean")
        total = float(repetition["total_wall_seconds"])
        normalized = float(repetition["normalized_wall_seconds_per_chain"])
        _require_close(normalized, total / 4.0, "R normalized wall time")
    return result


def _p200_comparison(
    records: Sequence[Mapping[str, object]],
    r_document: Mapping[str, object],
) -> dict[str, object]:
    result = dict(r_document)
    result["python"] = [
        dict(record)
        for record in records
        if record["estimator"] == "bm" and record["dimension"] == 200
    ]
    return result


def _median_error(record: Mapping[str, object]) -> float:
    repetitions = record["measured_repetitions"]
    if not isinstance(repetitions, list):
        raise TypeError("measured_repetitions must be a list")
    return float(
        statistics.median(
            float(repetition["truth_relative_frobenius_error"])
            for repetition in repetitions
        )
    )


def _report(
    records: Sequence[Mapping[str, object]],
    r_document: Mapping[str, object],
    lane_metadata: Mapping[str, object],
) -> str:
    keyed = {
        (
            str(record["estimator"]),
            str(record["dtype"]),
            int(record["dimension"]),
            str(record["device"]),
        ): record
        for record in records
    }
    lines = [
        "# 0.1.0.dev0 long-chain baseline",
        "",
        "This baseline uses 50 burn-in sweeps, 50 retained samples, four chains,",
        "and ten independently keyed repetitions in every Python cell.",
        "",
        "## Python BM and SBM scaling",
        "",
        "Times are normalized wall seconds per chain. IQR is shown as q1–q3.",
        "A GPU speedup above 1.0 means GPU was faster.",
        "",
        "| Estimator | Dtype | p | CPU median (IQR) | GPU median (IQR) | GPU speedup | CPU error | GPU error |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for estimator, dtype, dimension in product(ESTIMATORS, DTYPES, DIMENSIONS):
        cpu = keyed[(estimator, dtype, dimension, "cpu")]
        gpu = keyed[(estimator, dtype, dimension, "gpu")]
        cpu_timing = cpu["timing_summary"]
        gpu_timing = gpu["timing_summary"]
        if not isinstance(cpu_timing, dict) or not isinstance(gpu_timing, dict):
            raise TypeError("timing summaries must be objects")
        cpu_median = float(cpu_timing["median"])
        gpu_median = float(gpu_timing["median"])
        lines.append(
            f"| {estimator.upper()} | {dtype} | {dimension} | "
            f"{cpu_median:.3f} ({float(cpu_timing['q1']):.3f}–{float(cpu_timing['q3']):.3f}) | "
            f"{gpu_median:.3f} ({float(gpu_timing['q1']):.3f}–{float(gpu_timing['q3']):.3f}) | "
            f"{cpu_median / gpu_median:.3f}x | {_median_error(cpu):.5f} | {_median_error(gpu):.5f} |"
        )

    r_result = r_document["r"]
    if not isinstance(r_result, dict):
        raise TypeError("R result must be an object")
    r_timing = r_result["timing_summary"]
    if not isinstance(r_timing, dict):
        raise TypeError("R timing summary must be an object")
    r_median = float(r_timing["median"])
    lines.extend(
        [
            "",
            "## p=200 comparison with R bspcov 1.0.3",
            "",
            "| Implementation | Device | Dtype | Median wall / chain (s) | R-relative speedup |",
            "| --- | --- | --- | ---: | ---: |",
            f"| bspcov 1.0.3 | CPU | float64 | {r_median:.3f} | 1.000x |",
        ]
    )
    for device, dtype in product(DEVICES, DTYPES):
        record = keyed[("bm", dtype, 200, device)]
        timing = record["timing_summary"]
        if not isinstance(timing, dict):
            raise TypeError("Python timing summary must be an object")
        median = float(timing["median"])
        lines.append(
            f"| pybspcov | {device.upper()} | {dtype} | {median:.3f} | {r_median / median:.3f}x |"
        )
    topology = lane_metadata.get("topology", {})
    lines.extend(
        [
            "",
            "## Method and limitations",
            "",
            "- CPU used one vmapped four-chain fit; GPU used four sequential single-chain fits.",
            "- Compilation was synchronized, recorded separately, and excluded from primary timings.",
            "- CPU/R work was pinned to NUMA node 1; GPU host work was pinned to NUMA node 0.",
            f"- Recorded topology: `{json.dumps(topology, sort_keys=True)}`.",
            "- The boxplots overlay all ten raw observations; no density estimate is used.",
            "- Results are from one host and one fixture per dimension; they do not establish convergence or universal speedup.",
            "",
            "See [wall-time boxplots](wall-time-boxplots.svg) and [error boxplots](error-boxplots.svg).",
            "",
        ]
    )
    return "\n".join(lines)


def _plot(
    series: Sequence[PlotSeries],
    r_document: Mapping[str, object],
    *,
    attribute: str,
    ylabel: str,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "pybspcov-v0.1"
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {"cpu": "#4C78A8", "gpu": "#F58518", "r": "#54A24B"}
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    lookup = {
        (item.estimator, item.dtype, item.dimension, item.device): item
        for item in series
    }
    for row, estimator in enumerate(ESTIMATORS):
        for column, dtype in enumerate(DTYPES):
            axis = axes[row, column]
            for p_index, dimension in enumerate(DIMENSIONS, start=1):
                for device, offset in (("cpu", -0.16), ("gpu", 0.16)):
                    item = lookup[(estimator, dtype, dimension, device)]
                    values = tuple(getattr(item, attribute))
                    position = p_index + offset
                    axis.boxplot(
                        [values],
                        positions=[position],
                        widths=0.24,
                        patch_artist=True,
                        showfliers=False,
                        boxprops={"facecolor": colors[device], "alpha": 0.45},
                        medianprops={"color": colors[device], "linewidth": 1.5},
                    )
                    axis.scatter(
                        position + np.linspace(-0.04, 0.04, len(values)),
                        values,
                        color=colors[device],
                        s=12,
                        alpha=0.8,
                        zorder=3,
                    )
            if estimator == "bm" and dtype == "float64":
                r_result = r_document["r"]
                if not isinstance(r_result, dict):
                    raise TypeError("R result must be an object")
                repetitions = r_result["measured_repetitions"]
                if not isinstance(repetitions, list):
                    raise TypeError("R repetitions must be a list")
                r_field = (
                    "normalized_wall_seconds_per_chain"
                    if attribute == "wall_seconds"
                    else "truth_relative_frobenius_error"
                )
                values = [float(repetition[r_field]) for repetition in repetitions]
                position = 4.48
                axis.boxplot(
                    [values],
                    positions=[position],
                    widths=0.24,
                    patch_artist=True,
                    showfliers=False,
                    boxprops={"facecolor": colors["r"], "alpha": 0.45},
                    medianprops={"color": colors["r"], "linewidth": 1.5},
                )
                axis.scatter(
                    position + np.linspace(-0.04, 0.04, len(values)),
                    values,
                    color=colors["r"],
                    s=12,
                    alpha=0.8,
                    zorder=3,
                )
            axis.set_title(f"{estimator.upper()} · {dtype}")
            axis.set_xticks(range(1, 5), [str(value) for value in DIMENSIONS])
            axis.set_xlabel("Dimension p")
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=0.2)
    figure.legend(
        handles=[
            Patch(facecolor=colors["cpu"], alpha=0.45, label="Python CPU"),
            Patch(facecolor=colors["gpu"], alpha=0.45, label="Python GPU"),
            Patch(facecolor=colors["r"], alpha=0.45, label="R CPU (p=200)"),
        ],
        loc="upper center",
        ncol=3,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(
        output,
        format="svg",
        metadata={"Date": None, "Creator": "pybspcov"},
    )
    plt.close(figure)


def render(input_dir: Path, output_dir: Path) -> None:
    records = validate_python_records(load_python_records(input_dir))
    r_path = input_dir / "p200-r-only.json"
    r_document = validate_r_result(
        json.loads(r_path.read_text(encoding="utf-8")), records
    )
    lane_path = input_dir / "lane-metadata.json"
    lane_metadata = json.loads(lane_path.read_text(encoding="utf-8"))
    if not isinstance(lane_metadata, dict):
        raise TypeError("lane metadata must be an object")
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in PYTHON_FILENAMES:
        source = input_dir / filename
        target = output_dir / filename
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    shutil.copy2(lane_path, output_dir / "lane-metadata.json")
    (output_dir / "p200-r-comparison.json").write_text(
        json.dumps(_p200_comparison(records, r_document), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        _report(records, r_document, lane_metadata),
        encoding="utf-8",
    )
    series = build_plot_series(records)
    _plot(
        series,
        r_document,
        attribute="wall_seconds",
        ylabel="Normalized wall seconds per chain",
        output=output_dir / "wall-time-boxplots.svg",
    )
    _plot(
        series,
        r_document,
        attribute="errors",
        ylabel="Truth-relative Frobenius error",
        output=output_dir / "error-boxplots.svg",
    )


def check(directory: Path) -> None:
    records = validate_python_records(load_python_records(directory))
    r_path = directory / "p200-r-only.json"
    if not r_path.is_file():
        r_path = directory / "p200-r-comparison.json"
    validate_r_result(json.loads(r_path.read_text(encoding="utf-8")), records)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.check == (arguments.output_dir is not None):
        parser.error("choose exactly one of --check or --output-dir")
    return arguments


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_args(argv)
    if arguments.check:
        check(arguments.input_dir)
    else:
        render(arguments.input_dir, arguments.output_dir)


if __name__ == "__main__":
    main()
