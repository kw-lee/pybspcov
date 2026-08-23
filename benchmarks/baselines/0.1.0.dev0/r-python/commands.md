# Commands and orchestration

Commands were run from the dedicated worktree on clean revision
`56920dd04f1a56446035a3362a1f2ccc3172bfab`. R commands used:

```bash
export R_LIBS_USER=/tmp/pybspcov-r-lib:/home/kwlee/R/x86_64-pc-linux-gnu-library/4.5
```

## Fixtures

```bash
uv run --frozen python benchmarks/r_comparison/run_matrix.py \
  --fixture-root /tmp/pybspcov-r-comparison-smoke/fixtures \
  --output-dir /tmp/pybspcov-r-comparison-smoke/timings \
  --generate-fixtures --fixtures-only
```

## Parity

For each `METHOD` in `bm sbm bandppp thresholdppp`:

```bash
Rscript --vanilla benchmarks/r_comparison/run_bspcov_parity.R \
  --manifest benchmarks/r_comparison/manifest.json \
  --fixture-dir /tmp/pybspcov-r-comparison-smoke/fixtures/parity-p5 \
  --method METHOD \
  --output /tmp/pybspcov-r-comparison-smoke/final-parity/r-METHOD.json

uv run --frozen python benchmarks/r_comparison/run_pybspcov_parity.py \
  --manifest benchmarks/r_comparison/manifest.json \
  --fixture-dir /tmp/pybspcov-r-comparison-smoke/fixtures/parity-p5 \
  --method METHOD --dtype float64 \
  --output /tmp/pybspcov-r-comparison-smoke/final-parity/python-METHOD-float64.json

uv run --frozen python benchmarks/r_comparison/run_pybspcov_parity.py \
  --manifest benchmarks/r_comparison/manifest.json \
  --fixture-dir /tmp/pybspcov-r-comparison-smoke/fixtures/parity-p5 \
  --method METHOD --dtype float32 \
  --output /tmp/pybspcov-r-comparison-smoke/final-parity/python-METHOD-float32.json
```

The twelve artifacts were compared with:

```bash
uv run --frozen python benchmarks/r_comparison/compare_parity.py \
  --artifact-dir /tmp/pybspcov-r-comparison-smoke/final-parity \
  --output /tmp/pybspcov-r-comparison-smoke/final-parity.json
```

## Concurrent timing matrix

The committed runner's `build_cells()` and `command_for_cell()` functions
created the same pre-registered 60 subprocess commands. A temporary controller
partitioned them as follows and ran the three serial lanes with a
`ThreadPoolExecutor(max_workers=3)`:

```python
gpu_cells = [
    cell for cell in build_cells(manifest)
    if cell.implementation == "pybspcov" and cell.device == "gpu"
]
cpu_cells = [cell for cell in build_cells(manifest) if cell not in gpu_cells]
lanes = [
    ("gpu", gpu_cells, available_cpu_ids[16:24]),
    ("cpu-a", cpu_cells[::2], available_cpu_ids[0:8]),
    ("cpu-b", cpu_cells[1::2], available_cpu_ids[8:16]),
]
```

For each cell the controller ran the list returned by `command_for_cell()` in a
fresh subprocess, timed it with `time.perf_counter()`, and replaced the local
cold time with `record_external_cold_time()`. It reused only artifacts accepted
by `completed_cell_seconds()` for the same revision. All lanes shared one
12-hour wall-clock deadline.

The base child environment came from `benchmark_environment()`. For Python CPU
cells only, the controller additionally set:

```bash
JAX_PLATFORMS=cpu
```

This prevented CPU children from reserving RTX 3090 memory while the GPU lane
was active. GPU commands used `uv run --frozen --extra cuda12`; R commands used
`Rscript --vanilla`. The actual generated subprocess commands and all their
resource fields are recoverable from `manifest.json` and the 60 raw JSONL
records.

## Aggregate and render

```bash
uv run --frozen python benchmarks/r_comparison/aggregate.py \
  --input-dir /tmp/pybspcov-r-comparison-smoke/timings \
  --parity /tmp/pybspcov-r-comparison-smoke/final-parity.json \
  --output /tmp/pybspcov-r-comparison-smoke/summary.json

uv run --frozen python benchmarks/r_comparison/render_readme.py \
  --summary /tmp/pybspcov-r-comparison-smoke/summary.json \
  --baseline 0.1.0.dev0 \
  --execution-note 'Measured under the current system load (load1 approximately 80-93): CPU and GPU lanes ran concurrently until both CPU lanes completed; the remaining GPU cells continued under the same external 80-worker CPU load.'
```
