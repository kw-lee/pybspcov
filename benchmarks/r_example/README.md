# R example parity benchmark

This benchmark runs the committed, centered `bspcov` BM example through R
`bspcov` and JAX `pybspcov`. It records posterior summaries, Monte Carlo
standard errors (MCSEs), timings, and environment metadata. The comparison is
a statistical parity check; timing fields are descriptive measurements with
different scopes, not interchangeable speed claims.

Run all commands from the repository root. Generated files under `results/`
are ignored. Do not commit raw machine results. A reviewed baseline must be
curated deliberately and force-added as a named file with its environment and
run settings documented.

## Requirements

- Use R with `bspcov` exactly at version 1.0.3. Point `R_LIBS` at the library
  containing that version. The R runner rejects another version unless
  `--allow-version-mismatch` is supplied; that flag makes the run unsuitable
  as the 1.0.3 baseline.
- Use Python 3.12 or newer and the locked project environment (`uv sync
  --frozen`). The project constrains JAX to `>=0.11.0,<0.12`, and the runner
  enables float64.
- Select JAX CPU or GPU explicitly with `--device`. GPU runs require a
  CUDA-capable JAX installation, a visible supported GPU, and sufficient free
  device memory. A missing backend or device produces an unavailable-device
  error. For this benchmark host, NVIDIA driver 575 can use a CUDA 12 JAX
  build; that is a host-specific example, not a package requirement. R runs on
  CPU.
- For timing work, keep hardware, load, power settings, and thread environment
  fixed. Both runners record device/CPU and thread provenance in metadata.

## Production run

The explicit settings below match the runner defaults. Use the same burn-in,
draw count, committed fixture, and controlled host for a comparison. Increase
the chain lengths only by changing both commands together.

```sh
R_LIBS=/path/to/r-library Rscript benchmarks/r_example/run_bspcov.R \
  --burnin 1000 --n-samples 1000 \
  --output-dir benchmarks/r_example/results/r

JAX_ENABLE_X64=1 uv run python benchmarks/r_example/run_pybspcov.py \
  --burnin 1000 --n-samples 1000 --repetitions 5 --device cpu \
  --output-dir benchmarks/r_example/results/pybspcov-cpu
```

To measure JAX on GPU, use a separate output directory and replace `cpu` with
`gpu`. Do not overwrite a CPU run or compare timings across unlike hosts
without retaining that distinction.

```sh
JAX_ENABLE_X64=1 uv run python benchmarks/r_example/run_pybspcov.py \
  --burnin 1000 --n-samples 1000 --repetitions 5 --device gpu \
  --output-dir benchmarks/r_example/results/pybspcov-gpu
```

Compare one R run with one selected JAX run:

```sh
uv run python benchmarks/r_example/compare_results.py \
  --r-summary benchmarks/r_example/results/r/r_summary.csv \
  --pybspcov-summary benchmarks/r_example/results/pybspcov-cpu/pybspcov_summary.csv \
  --r-timing benchmarks/r_example/results/r/r_timing.csv \
  --pybspcov-timing benchmarks/r_example/results/pybspcov-cpu/pybspcov_timing.csv \
  --output benchmarks/r_example/results/comparison-cpu.json
```

Read `statistical_verdict` in the JSON; the comparator can exit successfully
while reporting `"fail"`. A pass requires every posterior statistic and RMSE
to differ by no more than six times the combined MCSE. Timing categories do
not affect that verdict.

## Smoke check only

The following 10-burn-in, 20-draw commands validate execution and schemas
only. Their summaries, MCSEs, verdict, and timings are **not statistical or
performance evidence** and must not be published as benchmark results.

```sh
R_LIBS=/path/to/r-library Rscript benchmarks/r_example/run_bspcov.R \
  --burnin 10 --n-samples 20 --output-dir /tmp/r-example-smoke

JAX_ENABLE_X64=1 uv run python benchmarks/r_example/run_pybspcov.py \
  --burnin 10 --n-samples 20 --repetitions 1 --device cpu \
  --output-dir /tmp/r-example-smoke

uv run python benchmarks/r_example/compare_results.py \
  --r-summary /tmp/r-example-smoke/r_summary.csv \
  --pybspcov-summary /tmp/r-example-smoke/pybspcov_summary.csv \
  --r-timing /tmp/r-example-smoke/r_timing.csv \
  --pybspcov-timing /tmp/r-example-smoke/pybspcov_timing.csv \
  --output /tmp/r-example-smoke/comparison.json
```

## Output schemas

Both summary CSVs (`r_summary.csv` and `pybspcov_summary.csv`) contain one row
per covariance element and share these columns:

```text
implementation,row,column,posterior_mean,posterior_mean_mcse,
posterior_sd,posterior_sd_mcse,q025,q025_mcse,q50,q50_mcse,
q975,q975_mcse,truth,rmse,rmse_mcse
```

The metadata CSVs have `name,value` columns. They record package/runtime
versions, platform and device, float64 precision, CPU/core and thread details,
fixture dimensions, chain settings and seeds, batch-MCSE settings, and exact
timing-scope descriptions. R additionally records BLAS, LAPACK, and session
information; JAX records JAX/JAXLIB versions, backend, device ID, and device
kind.

`r_timing.csv` contains:

```text
implementation,sampler_seconds,end_to_end_seconds
```

`pybspcov_timing.csv` contains:

```text
implementation,compile_plus_execution_seconds,steady_state_seconds,
steady_state_min_seconds,steady_state_max_seconds,end_to_end_seconds
```

The timing scopes are:

- R `sampler_seconds`: `bspcov::bmspcov` only.
- R `end_to_end_seconds`: fixture reads through summary and metadata CSV
  writes; excludes the final timing CSV write.
- JAX `compile_plus_execution_seconds`: first jitted BM chain call, including
  compilation, execution, and synchronization.
- JAX steady-state fields: synchronized warmed jitted BM chain calls with
  fresh PRNG keys; `steady_state_seconds` is their median and the other fields
  are their minimum and maximum.
- JAX `end_to_end_seconds`: fixture, first-chain, posterior-summary, metadata,
  and summary/metadata CSV work; excludes warmed repetitions and the final
  timing CSV write.

The comparison JSON contains `schema_version`, `statistical_verdict`,
`standard_error_multiplier`, per-element `posterior_comparisons`, an
`rmse_comparison`, and `timing_categories`. Each statistic comparison records
both values and MCSEs, signed and absolute differences, combined MCSE,
tolerance, pass/fail status, and an optional failure reason.
