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
comparison. For BM and SBM the GPU path is slower than R at `p = 50` and
`p = 100`, which pulls the geometric mean below 1.

The chains are too short to assess convergence, and the timings apply only to
the recorded configurations. The measurement host had an Intel Xeon w7-3565X
and an NVIDIA RTX PRO 5000 Blackwell. Unrelated CPU jobs from another user ran
during the measurement, and CPU frequency scaling was not fixed.

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
