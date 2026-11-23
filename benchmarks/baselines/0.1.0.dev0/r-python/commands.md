# Commands

Commands were run from the repository root on clean revision
`0fffd987d91586035820e64743ccd6ca18ef8058`. R came from a micromamba
environment placed first on `PATH`, and float64 was enabled for JAX:

```bash
export PATH=/path/to/micromamba/envs/r/bin:$PATH
export JAX_ENABLE_X64=1
W=$PWD/benchmarks/results/local/r-comparison
```

## Fixtures

```bash
uv run --frozen python benchmarks/r_comparison/run_matrix.py \
  --fixture-root $W/fixtures --output-dir $W/timings \
  --generate-fixtures --fixtures-only
```

## Parity

For each `METHOD` in `bm sbm bandppp thresholdppp`:

```bash
Rscript --vanilla benchmarks/r_comparison/run_bspcov_parity.R \
  --manifest benchmarks/r_comparison/manifest.json \
  --fixture-dir $W/fixtures/parity-p5 \
  --method METHOD --output $W/parity/r-METHOD.json

for DTYPE in float64 float32; do
  uv run --frozen python benchmarks/r_comparison/run_pybspcov_parity.py \
    --manifest benchmarks/r_comparison/manifest.json \
    --fixture-dir $W/fixtures/parity-p5 \
    --method METHOD --dtype $DTYPE \
    --output $W/parity/python-METHOD-$DTYPE.json
done
```

The twelve artifacts were compared with:

```bash
uv run --frozen python benchmarks/r_comparison/compare_parity.py \
  --artifact-dir $W/parity --output $W/parity.json
```

## Timing matrix

```bash
uv run --frozen python benchmarks/r_comparison/run_matrix.py \
  --fixture-root $W/fixtures --output-dir $W/timings
```

The runner executed the 60 cells sequentially in fresh subprocesses, pinned
with `taskset -c 0-7` for optimized cells and `taskset -c 0` for single-core
cells. GPU cells used `uv run --frozen --extra cuda12`; R cells used
`Rscript --vanilla`.

## Aggregate and render

```bash
uv run --frozen python benchmarks/r_comparison/aggregate.py \
  --input-dir $W/timings --parity $W/parity.json --output $W/summary.json

uv run --frozen python benchmarks/r_comparison/render_readme.py \
  --summary $W/summary.json --baseline 0.1.0.dev0 \
  --execution-note 'Measured on a shared host: the 60 cells ran one at a time with the GPU otherwise idle, while 15 unrelated single-threaded CPU jobs ran unpinned (load average about 24 at the start and 18 at the end).'

uv run --frozen python benchmarks/render_docs_benchmarks.py
```
