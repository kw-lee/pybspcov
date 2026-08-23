# Benchmarks

The documentation build does not run performance benchmarks. It renders the
small, version-controlled summary below from cached JSONL measurements under
`benchmarks/baselines/`. CI only verifies that the rendered summary still
matches those records, so a documentation build does not require a GPU or a
long sampler run.

```{include} _generated/benchmark-summary.md
```

## Interpretation and limitations

The cached baseline measures dispatch and scaling for intentionally short
chains. It is useful for comparing these recorded configurations, but it does
not establish convergence, scientific equivalence, or a universal CPU/GPU
speedup. The measurement host had two Intel Xeon Gold 5220R CPUs and an NVIDIA
GeForce RTX 3090. CPU affinity, NUMA placement, power state, and system load
were not isolated.

Compilation and steady-state execution are recorded separately. The displayed
table uses only the warmed medians, and the linked full report contains the
commands, full environment, validity checks, and additional dimensions.

## Refreshing the cached summary

After a reviewed benchmark run updates a versioned baseline, regenerate the
small documentation fragment locally:

```console
uv run python benchmarks/render_docs_benchmarks.py
```

The ordinary documentation jobs use `--check`; they never regenerate or rerun
the benchmark implicitly.
