# R/Python comparison: sequential run on a shared workstation

This is the benchmark record behind the generated table in the project README
and the documentation. All 60 timing cells and all eight method/dtype parity
gates passed for revision `0fffd987d91586035820e64743ccd6ca18ef8058`.

## Results

Warm times are medians in seconds. Speedup is R time divided by Python time, so
a value below 1 means R was faster.

| Method | Headline dtype | R CPU 8-worker, p=200 | Python GPU vmap-8, p=200 | Optimized speedup | p=50/100/200 geometric mean |
| --- | --- | ---: | ---: | ---: | ---: |
| BM | float32 | 121.154 | 84.642 | 1.431x | 0.756x |
| SBM | float32 | 53.779 | 78.494 | 0.685x | 0.409x |
| BandPPP | float32 | 1.876 | 0.061 | 30.739x | 39.802x |
| ThresholdPPP | float32 | 2.061 | 0.046 | 45.288x | 50.646x |

| Method | R CPU single-core, p=200 | Python CPU single-core, p=200 | Python speedup | Cold optimized speedup |
| --- | ---: | ---: | ---: | ---: |
| BM | 65.473 | 31.706 | 2.065x | 1.313x |
| SBM | 20.861 | 18.318 | 1.139x | 0.647x |
| BandPPP | 0.601 | 0.558 | 1.076x | 1.681x |
| ThresholdPPP | 0.983 | 0.481 | 2.044x | 1.850x |

Float32 and float64 passed the six-combined-MCSE parity gate for BM, SBM,
BandPPP, and ThresholdPPP. Float32 was selected for each optimized headline by
the pre-registered selection rule.

For BM and SBM the GPU path is slower than R at `p = 50` and `p = 100`, which
is why the geometric means fall below 1. Warm medians in seconds:

| Method | p | R CPU 8-worker | Python GPU float32 | Python GPU float64 |
| --- | ---: | ---: | ---: | ---: |
| BM | 50 | 10.185 | 18.631 | 27.913 |
| BM | 100 | 21.128 | 38.228 | 61.230 |
| BM | 200 | 121.154 | 84.642 | 148.667 |
| SBM | 50 | 6.266 | 17.610 | 26.136 |
| SBM | 100 | 10.483 | 37.381 | 55.185 |
| SBM | 200 | 53.779 | 78.494 | 120.466 |

## Workload

Each cell fits one fixture with `n = 3p` observations and sparse density 0.05,
seed 20260803. BM and SBM run 50 burn-in sweeps and 50 retained samples per
chain; SBM uses correlation screening with retained fraction 0.2. BandPPP and
ThresholdPPP draw 100 samples per batch. Optimized cells run 8 chains or
batches, and single-core cells run one. `benchmarks/r_comparison/manifest.json`
holds the full pre-registered configuration.

## Execution conditions

The matrix was measured on 2026-11-23 between 11:09 and 12:41 KST. The runner
executed the 60 cells one at a time. Optimized cells were pinned to CPUs 0-7
and single-core cells to CPU 0 with `taskset`.

No other process used the GPU during the run. The host was shared: 15
unrelated single-threaded CPU jobs from another user ran throughout without
CPU pinning, so they could share cores with the benchmark. The load average was
about 24 at the start and 18 at the end.

BLAS thread counts were fixed at one for every child. Each cell measured one
fresh-process cold end-to-end time and three warm fits, extended to five when
the first three had a relative range above 10%. Of the 60 cells, 44 used three
warm repetitions and 16 used five.

## Environment

| Component | Value |
| --- | --- |
| Host | HP Z8 Fury G5 Workstation, Ubuntu 24.04.5 LTS, Linux 6.17.0-1032-oem |
| CPU | Intel Xeon w7-3565X, 32 cores, 2 threads/core, 64 logical CPUs |
| GPU | NVIDIA RTX PRO 5000 Blackwell, 48,935 MiB, driver 595.99.02 |
| R | 4.5.3 with OpenBLAS 0.3.34; `bspcov` 1.0.3 |
| Python | 3.12.3; `pybspcov` 0.1.0.dev0 |
| JAX | 0.11.0 with the project `cuda12` extra for GPU cells |
| Precision | R optimized and both CPU baselines float64; Python GPU float32 and float64 |

See `python-freeze.txt` and `r-session-info.txt` for dependency snapshots.

## Retained record

- `summary.json`: the aggregate written by `aggregate.py`. The README and
  documentation tables are rendered from it.
- `commands.md`: fixture, parity, timing, aggregation, and rendering commands.
- `python-freeze.txt`: the Python dependency snapshot with the `cuda12` extra.
- `r-session-info.txt`: the R dependency and platform snapshot.

The per-cell JSONL timing records and the parity artifacts are not tracked in
Git. The commands regenerate them from the tracked runner and manifest. Every
record used for these tables identified the clean revision above; the later
commits that add this record do not change the measured implementation.

## Limitations

These numbers come from one host, one fixture family, one seed, and short
performance chains. The external CPU load was not controlled, so the absolute
times are not idle-host throughput estimates. CPU frequency scaling and power
state were not fixed.

The optimized headline compares R's CPU multi-worker path with Python's GPU
vmap path, so the hardware differs between the two sides. The single-core
float64 table is the implementation control. Cold timing includes process
startup and compilation, which dominates the short PPP cells.

The public R API does not expose a rejected-sweep counter for these calls. The R
runner records zero only after checking that all returned draws are finite.
