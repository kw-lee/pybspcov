# R/Python comparison: sequential run on a shared workstation

This is the benchmark record behind the generated table in the project README
and the documentation. All 60 timing cells and all eight method/dtype parity
gates passed for revision `3ffbdd59f52f70e48cf1158f82058664f1398cd8`.

## Results

Warm times are medians in seconds. Speedup is R time divided by Python time, so
a value below 1 means R was faster.

| Method | Headline dtype | R CPU 8-worker, p=200 | Python GPU vmap-8, p=200 | Optimized speedup | p=50/100/200 geometric mean |
| --- | --- | ---: | ---: | ---: | ---: |
| BM | float32 | 74.655 | 26.401 | 2.828x | 1.818x |
| SBM | float32 | 35.590 | 19.899 | 1.789x | 1.115x |
| BandPPP | float32 | 1.308 | 0.058 | 22.654x | 29.509x |
| ThresholdPPP | float32 | 1.272 | 0.043 | 29.738x | 34.739x |

| Method | R CPU single-core, p=200 | Python CPU single-core, p=200 | Python speedup | Cold optimized speedup |
| --- | ---: | ---: | ---: | ---: |
| BM | 48.107 | 23.360 | 2.059x | 2.488x |
| SBM | 14.472 | 11.551 | 1.253x | 1.504x |
| BandPPP | 0.464 | 0.420 | 1.105x | 1.275x |
| ThresholdPPP | 0.708 | 0.335 | 2.115x | 1.734x |

Float32 and float64 passed the six-combined-MCSE parity gate for BM, SBM,
BandPPP, and ThresholdPPP. Float32 was selected for each optimized headline by
the pre-registered selection rule.

The float32 GPU path is faster than R's 8 workers in every BM and SBM cell
except SBM at `p = 100`. The float64 GPU path is slower than R except for BM at
`p = 200`. Warm medians in seconds:

| Method | p | R CPU 8-worker | Python GPU float32 | Python GPU float64 |
| --- | ---: | ---: | ---: | ---: |
| BM | 50 | 6.103 | 3.615 | 6.831 |
| BM | 100 | 11.698 | 9.295 | 20.076 |
| BM | 200 | 74.655 | 26.401 | 66.922 |
| SBM | 50 | 3.499 | 3.403 | 5.165 |
| SBM | 100 | 5.817 | 7.707 | 13.819 |
| SBM | 200 | 35.590 | 19.899 | 38.606 |

This record replaces the run of 2026-11-23 at revision
`0fffd987d91586035820e64743ccd6ca18ef8058`. Between the two, the scalar GIG
sampler stopped running an unused rejection loop under `vmap`, which cut the BM
and SBM GPU times at `p = 200` from 84.642 s and 78.494 s. The host was also
much quieter for this run, so the R and CPU times fell as well and the two
records are not a controlled before/after comparison.

## Workload

Each cell fits one fixture with `n = 3p` observations and sparse density 0.05,
seed 20260803. BM and SBM run 50 burn-in sweeps and 50 retained samples per
chain; SBM uses correlation screening with retained fraction 0.2. BandPPP and
ThresholdPPP draw 100 samples per batch. Optimized cells run 8 chains or
batches, and single-core cells run one. `benchmarks/r_comparison/manifest.json`
holds the full pre-registered configuration.

## Execution conditions

The matrix was measured on 2026-11-24 between 10:43 and 11:24 KST. The runner
executed the 60 cells one at a time. Optimized cells were pinned to CPUs 0-7
and single-core cells to CPU 0 with `taskset`.

The host is shared and other GPU jobs had priority. The run started after the
GPU had been free of other processes for ten minutes, and a watcher polled
`nvidia-smi` every ten seconds to stop a GPU cell if another GPU process
appeared. None did, so every cell ran in a single uninterrupted attempt. The
load average, sampled every ten minutes and including the benchmark's own
workers, stayed between 0.1 and 6.6. Other CPU activity was not pinned away
from the benchmark cores.

JAX ran with `XLA_PYTHON_CLIENT_PREALLOCATE=false` so that it did not reserve
GPU memory other jobs might need.

BLAS thread counts were fixed at one for every child. Each cell measured one
fresh-process cold end-to-end time and three warm fits, extended to five when
the first three had a relative range above 10%. Of the 60 cells, 46 used three
warm repetitions and 14 used five.

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
performance chains. Other CPU load was light but not controlled, so the absolute
times are not idle-host throughput estimates. CPU frequency scaling and power
state were not fixed.

The optimized headline compares R's CPU multi-worker path with Python's GPU
vmap path, so the hardware differs between the two sides. The single-core
float64 table is the implementation control. Cold timing includes process
startup and compilation, which dominates the short PPP cells.

The public R API does not expose a rejected-sweep counter for these calls. The R
runner records zero only after checking that all returned draws are finite.
