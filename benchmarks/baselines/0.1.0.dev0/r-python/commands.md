# Commands

Commands were run from the repository root on clean revision
`3ffbdd59f52f70e48cf1158f82058664f1398cd8`. R came from a micromamba
environment placed first on `PATH`, and float64 was enabled for JAX:

```bash
export PATH=/path/to/micromamba/envs/r/bin:$PATH
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
W=$PWD/benchmarks/results/local/r-comparison
```

`XLA_PYTHON_CLIENT_PREALLOCATE=false` stops JAX from reserving most of the GPU
memory at start-up, so other GPU jobs on the shared host were not starved.

## Fixtures

```bash
uv run --frozen python benchmarks/r_comparison/run_matrix.py \
  --fixture-root $W/fixtures --output-dir $W/timings \
  --generate-fixtures --fixtures-only
```

## Parity

The parity runs used `JAX_PLATFORMS=cpu`. For each `METHOD` in
`bm sbm bandppp thresholdppp`:

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
  --fixture-root $W/fixtures --output-dir $W/timings --resume
```

A wrapper started this command only after no other process had used the GPU
for ten minutes. It polled `nvidia-smi` every ten seconds and would have killed
a GPU cell, discarded its record, and resumed later if another GPU process had
appeared. That did not happen: the matrix completed in one attempt.

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
  --execution-note 'Measured on a shared host: the 60 cells ran one at a time, no other process used the GPU during the run, and the load average stayed between 0.1 and 6.6.'

uv run --frozen python benchmarks/render_docs_benchmarks.py
```
