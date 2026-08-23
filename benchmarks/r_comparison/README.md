# R/Python comparison benchmark

This directory implements the pre-registered comparison between R `bspcov`
1.0.3 and the current `pybspcov` revision. It deliberately separates
statistical parity from performance. A timing result is publishable only when
all four float64 parity gates pass and all 60 timing cells are present.

## Protocol

The immutable settings are in `manifest.json`. BM, SBM, BandPPP, and
ThresholdPPP use identical CSV inputs generated once for each dimension. The
timing matrix uses `p = 50, 100, 200`, `n = 3p`, and a fixed seed. It contains:

- an optimized comparison of R float64 CPU with 8 workers against Python GPU
  vmap with 8 chains/batches in both float32 and float64; and
- a matching single-core CPU float64 baseline for both implementations.

The headline dtype is selected after measurement. Float32 is used only if its
long-run parity gate passes and its `p=200` median is at least 5% faster than
float64. Otherwise float64 is used. The README reports the optimized `p=200`
result and the geometric-mean speedup across all three sizes, while also
showing the single-core baseline.

Each timing cell runs once in a fresh process for cold end-to-end wall time,
then three warm fits. Two more warm fits are added only when the relative range
of the first three exceeds 10%. BLAS libraries are limited to one thread and
each process is pinned to the requested number of CPUs from its actual affinity
mask. GPU requests must resolve to CUDA; silent CPU fallback is rejected.

The parity fixture is intentionally small and long-running. BM and SBM use four
chains with 2,000 burn-in and 2,000 retained draws per chain. PPP methods use
5,000 retained draws. Posterior mean, standard deviation, 2.5%, 50%, and 97.5%
quantiles, plus RMSE, must each agree within six combined Monte Carlo standard
errors. Fifty contiguous batches estimate the MCSEs.

## Run

Run from a clean repository revision. R must provide `bspcov` 1.0.3,
`jsonlite`, and `openssl`; the runners reject other `bspcov` versions. The
optimized Python cells require a working CUDA installation supported by the
project's `cuda12` extra.

Generate the versioned timing fixtures without starting the matrix:

```bash
uv run python benchmarks/r_comparison/run_matrix.py \
  --fixture-root /tmp/pybspcov-r-comparison/fixtures \
  --output-dir /tmp/pybspcov-r-comparison/timings \
  --generate-fixtures --fixtures-only
```

That command also converts the committed upstream R example into
`fixtures/parity-p5`. Replace `METHOD` below with each of `bm`, `sbm`,
`bandppp`, and `thresholdppp`:

```bash
Rscript --vanilla benchmarks/r_comparison/run_bspcov_parity.R \
  --manifest benchmarks/r_comparison/manifest.json \
  --fixture-dir /tmp/pybspcov-r-comparison/fixtures/parity-p5 \
  --method METHOD \
  --output /tmp/pybspcov-r-comparison/parity/r-METHOD.json

uv run python benchmarks/r_comparison/run_pybspcov_parity.py \
  --manifest benchmarks/r_comparison/manifest.json \
  --fixture-dir /tmp/pybspcov-r-comparison/fixtures/parity-p5 \
  --method METHOD --dtype float64 \
  --output /tmp/pybspcov-r-comparison/parity/python-METHOD-float64.json

uv run python benchmarks/r_comparison/run_pybspcov_parity.py \
  --manifest benchmarks/r_comparison/manifest.json \
  --fixture-dir /tmp/pybspcov-r-comparison/fixtures/parity-p5 \
  --method METHOD --dtype float32 \
  --output /tmp/pybspcov-r-comparison/parity/python-METHOD-float32.json
```

Then compare the twelve artifacts:

```bash
uv run python benchmarks/r_comparison/compare_parity.py \
  --artifact-dir /tmp/pybspcov-r-comparison/parity \
  --output /tmp/pybspcov-r-comparison/parity.json
```

Inspect the complete command matrix before execution, then run it:

```bash
uv run python benchmarks/r_comparison/run_matrix.py \
  --fixture-root /tmp/pybspcov-r-comparison/fixtures \
  --output-dir /tmp/pybspcov-r-comparison/timings --dry-run

uv run python benchmarks/r_comparison/run_matrix.py \
  --fixture-root /tmp/pybspcov-r-comparison/fixtures \
  --output-dir /tmp/pybspcov-r-comparison/timings
```

The manifest caps cumulative cell wall time at 12 hours. If an execution is
interrupted, resume only the valid cells from the same clean revision:

```bash
uv run python benchmarks/r_comparison/run_matrix.py \
  --fixture-root /tmp/pybspcov-r-comparison/fixtures \
  --output-dir /tmp/pybspcov-r-comparison/timings --resume
```

Aggregate only after the parity and timing matrices are complete:

```bash
uv run python benchmarks/r_comparison/aggregate.py \
  --input-dir /tmp/pybspcov-r-comparison/timings \
  --parity /tmp/pybspcov-r-comparison/parity.json \
  --output /tmp/pybspcov-r-comparison/summary.json

uv run python benchmarks/r_comparison/render_readme.py \
  --summary /tmp/pybspcov-r-comparison/summary.json \
  --baseline BASELINE-NAME \
  --execution-note 'Describe load, lane concurrency, and other run conditions.'
```

Copy the manifest, raw JSONL, parity artifacts, aggregate summary, commands,
device and precision details, and environment information into the named
baseline directory before committing a generated README result.

## Limitations

The optimized headline intentionally compares each implementation's intended
fast path and therefore includes a GPU only on the Python side. It is not a
hardware-equal comparison; the single-core CPU float64 table provides that
control. Results are specific to one host, fixture family, seed, and short
performance-run chain length. They do not establish universal superiority.

The public R API does not expose a rejected-sweep counter for these calls. The
R runner records zero only after checking that all returned draws are finite;
this limitation must remain in the archived baseline report. No partial,
version-mismatched, dirty-tree, parity-failing, or CPU-fallback result may be
rendered into the project README.
