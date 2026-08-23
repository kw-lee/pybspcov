# R/Python comparison: concurrent loaded-host run

This is the complete archived baseline behind the generated table in the project
README. All 60 timing cells and all eight method/dtype parity gates passed for
revision `56920dd04f1a56446035a3362a1f2ccc3172bfab`.

## Results

Warm times are medians in seconds. Speedup is R time divided by Python time, so
a value below 1 means R was faster.

| Method | Headline dtype | R CPU 8-worker, p=200 | Python GPU vmap-8, p=200 | Optimized speedup | p=50/100/200 geometric mean |
| --- | --- | ---: | ---: | ---: | ---: |
| BM | float32 | 322.876 | 251.827 | 1.282x | 1.014x |
| SBM | float32 | 159.317 | 243.186 | 0.655x | 0.756x |
| BandPPP | float32 | 3.747 | 0.102 | 36.901x | 36.407x |
| ThresholdPPP | float32 | 5.821 | 0.071 | 81.905x | 79.326x |

| Method | R CPU single-core, p=200 | Python CPU single-core, p=200 | Python speedup | Cold optimized speedup |
| --- | ---: | ---: | ---: | ---: |
| BM | 195.724 | 107.284 | 1.824x | 1.819x |
| SBM | 58.419 | 71.983 | 0.812x | 0.954x |
| BandPPP | 1.884 | 1.292 | 1.459x | 1.810x |
| ThresholdPPP | 2.151 | 1.154 | 1.864x | 1.883x |

Float32 and float64 passed the six-combined-MCSE parity gate for BM, SBM,
BandPPP, and ThresholdPPP. Float32 was selected for each optimized headline by
the pre-registered selection rule.

## Execution conditions

The final matrix was measured on 2026-08-23 UTC under the current system load,
not on an idle host. An unrelated 80-worker CPU workload remained active, and
the benchmark controller observed load1 values of approximately 80 to 93.

Three lanes started concurrently:

- CPU lane A used affinity IDs 0-7.
- CPU lane B used affinity IDs 8-15.
- The GPU lane used host affinity IDs 16-23 and one RTX 3090.

The two CPU lanes finished while the GPU lane was still running. Consequently,
BM and part of the GPU matrix overlapped the benchmark CPU lanes, while later
GPU cells ran only alongside the external CPU workload. The run is therefore a
contention-conditioned measurement, not a balanced simultaneous-load design.

Python CPU children set `JAX_PLATFORMS=cpu`; this was verified while CPU and GPU
children were alive by confirming that only the GPU child owned CUDA memory.
An earlier diagnostic attempt allowed CPU children to initialize CUDA. Those
artifacts were moved to a separate temporary contamination directory and are
not included here. BLAS thread counts were fixed at one for every child.

Each timing artifact contains one fresh-process cold end-to-end time and three
warm fits, extended to five when the first three had a relative range above
10%. Of the final 60 cells, 42 used three warm repetitions and 18 used five.

## Environment

| Component | Value |
| --- | --- |
| Host | Dell PowerEdge R740, Ubuntu 24.04.2 LTS, Linux 6.8.0-60-generic |
| CPU | 2 x Intel Xeon Gold 5220R, 24 cores/socket, 2 threads/core, 96 logical CPUs |
| GPU | NVIDIA GeForce RTX 3090, 24,576 MiB, driver 575.57.08 |
| R | 4.5.0; `bspcov` 1.0.3 |
| Python | 3.13.6; `pybspcov` 0.1.0.dev0 |
| JAX | 0.11.0 with the project `cuda12` extra for GPU cells |
| Precision | R optimized and both CPU baselines float64; Python GPU float32 and float64 |

See `environment.json`, `python-freeze.txt`, and `r-session-info.txt` for the
machine-readable settings and dependency snapshots.

## Archived evidence

- `manifest.json`: pre-registered dimensions, seeds, repetitions, and gates.
- `timings/`: all 60 raw timing JSONL records.
- `parity/`: four R and eight Python raw parity artifacts.
- `parity.json`: the eight parity verdicts and detailed MCSE comparisons.
- `summary.json`: the validated aggregate used to render the project README.
- `commands.md`: fixture, parity, lane, aggregation, and rendering commands.

All raw timing and parity records identify the clean benchmark revision
`56920dd04f1a56446035a3362a1f2ccc3172bfab`. The later documentation commit does
not change the measured implementation.

## Limitations

These numbers are specific to one heavily loaded host, one fixture family, one
seed, and short performance chains. External CPU contention was neither
controlled nor replayable, and the benchmark CPU lanes did not overlap every
GPU cell. Do not use the absolute times as idle-host throughput estimates.

The optimized headline deliberately compares R's CPU multi-worker path with
Python's GPU vmap path, so it is not a hardware-equal comparison. The
single-core float64 table is the implementation control. Cold timing includes
process startup and compilation, which particularly affects short PPP cells.

The public R API does not expose a rejected-sweep counter for these calls. The R
runner records zero only after checking that all returned draws are finite.
Nothing in this baseline establishes universal superiority of either language
or package.
