# Benchmarks

The documentation build does not run performance benchmarks. It renders the
summary below from the aggregate tracked at
`benchmarks/baselines/0.1.0.dev0/r-python/summary.json`. CI only verifies that
the rendered summary still matches that file, so a documentation build does not
require R, a GPU, or a long sampler run.

```{include} _generated/benchmark-summary.md
```

## Interpretation and limitations

The first table compares each implementation's fastest path, so only the Python
side uses a GPU. The single-core float64 table is the hardware-equal
comparison. The float32 GPU path is faster than R's 8 workers in every BM and
SBM cell except SBM at `p = 100`.

The chains are too short to assess convergence, and the timings apply only to
the recorded configurations. The measurement host had an Intel Xeon w7-3565X
and an NVIDIA RTX PRO 5000 Blackwell. The host is shared: no other process
used the GPU during the measurement, but other CPU activity was not excluded,
and CPU frequency scaling was not fixed.

The linked report contains the commands, environment, execution conditions,
and per-dimension GPU timings.

## Refreshing the cached summary

After a reviewed benchmark run updates the tracked summary, regenerate the
documentation fragment locally:

```console
uv run python benchmarks/render_docs_benchmarks.py
```

The documentation jobs use `--check`; they never regenerate or rerun the
benchmark.
