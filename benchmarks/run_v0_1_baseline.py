"""Run the v0.1 long-chain baseline in isolated concurrent CPU/GPU lanes."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple


class BenchmarkConfiguration(NamedTuple):
    """Static workload and host paths for one complete baseline run."""

    project_root: Path
    output_dir: Path
    r_library: Path
    python_executable: Path
    dimensions: tuple[int, ...]
    density: float
    n_factor: int
    burnin: int
    samples: int
    chains: int
    repetitions: int
    seed: int


class LaneCommands(NamedTuple):
    """Commands executed sequentially inside each concurrent lane."""

    cpu_python: tuple[tuple[str, ...], ...]
    gpu_python: tuple[tuple[str, ...], ...]
    r: tuple[str, ...]


def _numa_prefix(node: int) -> tuple[str, ...]:
    return ("numactl", f"--cpunodebind={node}", f"--membind={node}")


def _python_cell_command(
    configuration: BenchmarkConfiguration,
    *,
    estimator: str,
    device: str,
    dtype: str,
) -> tuple[str, ...]:
    node = 1 if device == "cpu" else 0
    platform_name = "cpu" if device == "cpu" else "cuda"
    x64 = "1" if dtype == "float64" else "0"
    output = configuration.output_dir / f"{estimator}-{device}-{dtype}.jsonl"
    script = configuration.project_root / "benchmarks" / "sbm_public_scaling.py"
    return (
        *_numa_prefix(node),
        "env",
        f"JAX_PLATFORMS={platform_name}",
        f"JAX_ENABLE_X64={x64}",
        "XLA_PYTHON_CLIENT_PREALLOCATE=false",
        str(configuration.python_executable),
        str(script),
        "--estimator",
        estimator,
        "--device",
        device,
        "--dtype",
        dtype,
        "--dimensions",
        *(str(value) for value in configuration.dimensions),
        "--density",
        str(configuration.density),
        "--n-factor",
        str(configuration.n_factor),
        "--burnin",
        str(configuration.burnin),
        "--samples",
        str(configuration.samples),
        "--chains",
        str(configuration.chains),
        "--repetitions",
        str(configuration.repetitions),
        "--seed",
        str(configuration.seed),
        "--output",
        str(output),
    )


def build_lane_commands(configuration: BenchmarkConfiguration) -> LaneCommands:
    """Build four sequential Python cells per lane and one trailing R command."""
    cpu_python = tuple(
        _python_cell_command(
            configuration,
            estimator=estimator,
            device="cpu",
            dtype=dtype,
        )
        for estimator in ("bm", "sbm")
        for dtype in ("float64", "float32")
    )
    gpu_python = tuple(
        _python_cell_command(
            configuration,
            estimator=estimator,
            device="gpu",
            dtype=dtype,
        )
        for estimator in ("bm", "sbm")
        for dtype in ("float64", "float32")
    )
    r_script = configuration.project_root / "benchmarks" / "r_scaling" / "run_p200.py"
    r_command = (
        *_numa_prefix(1),
        "env",
        "JAX_PLATFORMS=cpu",
        "JAX_ENABLE_X64=1",
        str(configuration.python_executable),
        str(r_script),
        "--r-library",
        str(configuration.r_library),
        "--output",
        str(configuration.output_dir / "p200-r-only.json"),
        "--dimension",
        "200",
        "--n-factor",
        str(configuration.n_factor),
        "--density",
        str(configuration.density),
        "--burnin",
        str(configuration.burnin),
        "--n-samples",
        str(configuration.samples),
        "--n-chains",
        str(configuration.chains),
        "--repetitions",
        str(configuration.repetitions),
        "--seed",
        str(configuration.seed),
    )
    return LaneCommands(cpu_python=cpu_python, gpu_python=gpu_python, r=r_command)


def _gpu_numa_node() -> tuple[int, str]:
    result = subprocess.run(
        ("nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"),
        check=True,
        capture_output=True,
        text=True,
    )
    bus_id = result.stdout.splitlines()[0].strip().lower()
    parts = bus_id.split(":")
    if len(parts) != 3:
        raise RuntimeError(f"unexpected GPU PCI bus id: {bus_id}")
    sysfs_bus_id = f"{parts[0][-4:]}:{parts[1]}:{parts[2]}"
    node_path = Path("/sys/bus/pci/devices") / sysfs_bus_id / "numa_node"
    node = int(node_path.read_text(encoding="ascii").strip())
    return node, bus_id


def preflight(configuration: BenchmarkConfiguration) -> dict[str, object]:
    """Require the expected dual-NUMA host and working memory binding."""
    if shutil.which("numactl") is None:
        raise RuntimeError("numactl is required")
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi is required")
    if not configuration.r_library.is_dir():
        raise RuntimeError(f"R library does not exist: {configuration.r_library}")
    cpu_lists: dict[str, str] = {}
    for node in (0, 1):
        cpulist = Path(f"/sys/devices/system/node/node{node}/cpulist")
        if not cpulist.is_file():
            raise RuntimeError("two NUMA nodes are required")
        cpu_lists[str(node)] = cpulist.read_text(encoding="ascii").strip()
        subprocess.run((*_numa_prefix(node), "true"), check=True)
    gpu_node, pci_bus_id = _gpu_numa_node()
    if gpu_node != 0:
        raise RuntimeError(f"expected GPU on NUMA node 0, found node {gpu_node}")
    return {
        "gpu_numa_node": gpu_node,
        "gpu_pci_bus_id": pci_bus_id,
        "numa_cpu_lists": cpu_lists,
        "cpu_lane_node": 1,
        "gpu_lane_node": 0,
    }


def _run_lane(name: str, commands: Sequence[tuple[str, ...]]) -> dict[str, object]:
    started = datetime.now(UTC)
    command_records: list[dict[str, object]] = []
    for index, command in enumerate(commands, start=1):
        command_start = time.perf_counter()
        print(
            f"[{name}] starting {index}/{len(commands)}: {shlex.join(command)}",
            flush=True,
        )
        subprocess.run(command, check=True)
        elapsed = time.perf_counter() - command_start
        print(
            f"[{name}] completed {index}/{len(commands)} in {elapsed:.3f}s", flush=True
        )
        command_records.append({"command": list(command), "wall_seconds": elapsed})
    ended = datetime.now(UTC)
    return {
        "name": name,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "wall_seconds": (ended - started).total_seconds(),
        "commands": command_records,
    }


def run(configuration: BenchmarkConfiguration) -> dict[str, object]:
    """Execute both isolated lanes and retain orchestration provenance."""
    topology = preflight(configuration)
    configuration.output_dir.mkdir(parents=True, exist_ok=True)
    if any(configuration.output_dir.iterdir()):
        raise RuntimeError("output directory must be empty")
    lanes = build_lane_commands(configuration)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        cpu_future = executor.submit(_run_lane, "cpu-r", (*lanes.cpu_python, lanes.r))
        gpu_future = executor.submit(_run_lane, "gpu", lanes.gpu_python)
        lane_results = [cpu_future.result(), gpu_future.result()]
    metadata = {
        "schema_version": "1.0",
        "topology": topology,
        "lanes": lane_results,
    }
    (configuration.output_dir / "lane-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--r-library", type=Path, required=True)
    parser.add_argument("--dimensions", nargs="+", type=int, default=[25, 50, 100, 200])
    parser.add_argument("--density", type=float, default=0.05)
    parser.add_argument("--n-factor", type=int, default=3)
    parser.add_argument("--burnin", type=int, default=50)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260803)
    arguments = parser.parse_args(argv)
    if any(value < 2 for value in arguments.dimensions):
        parser.error("dimensions must be at least two")
    if arguments.burnin < 0:
        parser.error("burnin must be non-negative")
    if (
        min(
            arguments.n_factor,
            arguments.samples,
            arguments.chains,
            arguments.repetitions,
        )
        < 1
    ):
        parser.error("n-factor, samples, chains, and repetitions must be positive")
    return arguments


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_args(argv)
    run(
        BenchmarkConfiguration(
            project_root=Path(__file__).resolve().parents[1],
            output_dir=arguments.output_dir.resolve(),
            r_library=arguments.r_library.resolve(),
            python_executable=Path(sys.executable).resolve(),
            dimensions=tuple(arguments.dimensions),
            density=arguments.density,
            n_factor=arguments.n_factor,
            burnin=arguments.burnin,
            samples=arguments.samples,
            chains=arguments.chains,
            repetitions=arguments.repetitions,
            seed=arguments.seed,
        )
    )


if __name__ == "__main__":
    main()
