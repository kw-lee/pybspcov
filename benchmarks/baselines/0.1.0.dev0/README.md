# 0.1.0.dev0 scaling baseline

This is a short-chain performance baseline for the public `BMSPCov` and
`SBMSPCov` APIs. It measures scaling and dispatch overhead; it does not
establish MCMC convergence, scientific equivalence, or a universal speedup.

## Result

On this host, every first fit was faster on CPU because GPU compilation and
dispatch dominated the short workload. After compilation, GPU acceleration
appeared only for dense BM at `p=200`: 2.09x with float32 and 1.28x with
float64. Screened SBM did not benefit from the GPU through `p=200`; float32 at
`p=200` was close to parity at 0.965x. A speedup above 1.0 below means that the
GPU was faster.

All 32 posterior means were finite, symmetric, and positive definite.
Float32 truth-relative errors were close to float64 in this short run, but the
largest observed increase was 11.3% for SBM at `p=50`. These errors include
short-chain Monte Carlo variation and are not precision-accuracy evidence.

### Warmed median fit time

Times are seconds. Error columns are truth-relative Frobenius errors measured
on CPU; GPU values agree to the displayed precision.

| Estimator | p | CPU f32 | GPU f32 | f32 speedup | CPU f64 | GPU f64 | f64 speedup | error f32 | error f64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM | 25 | 0.157 | 0.277 | 0.567x | 0.140 | 0.472 | 0.296x | 0.2560 | 0.2387 |
| BM | 50 | 0.271 | 0.549 | 0.493x | 0.241 | 0.929 | 0.259x | 0.2030 | 0.2080 |
| BM | 100 | 0.909 | 1.131 | 0.804x | 0.878 | 2.030 | 0.433x | 0.1931 | 0.1900 |
| BM | 200 | 5.259 | 2.513 | 2.092x | 6.552 | 5.135 | 1.276x | 0.1905 | 0.1908 |
| SBM | 25 | 0.119 | 0.278 | 0.429x | 0.120 | 0.469 | 0.256x | 0.1779 | 0.1762 |
| SBM | 50 | 0.197 | 0.524 | 0.375x | 0.207 | 0.905 | 0.229x | 0.1977 | 0.1777 |
| SBM | 100 | 0.376 | 1.035 | 0.363x | 0.423 | 1.806 | 0.234x | 0.1961 | 0.1949 |
| SBM | 200 | 2.040 | 2.115 | 0.965x | 2.326 | 3.830 | 0.607x | 0.1885 | 0.1871 |

## p=200 comparison with R

This longer BM run compares `bspcov` 1.0.3 with `pybspcov` on the same
`p=200`, `n=600` float64 fixture. Each result retains 50 samples after 50
burn-in iterations from each of four chains. R used four parallel workers,
Python CPU used one vmapped four-chain fit, and Python GPU ran four single-chain
fits sequentially as required by the protocol.

The primary timing is normalized wall time per chain: total wall time divided
by four for the parallel or vmapped runs, and the arithmetic mean of the four
single-chain wall times for GPU. Python compilation is excluded from these
steady-state values. R-relative speedup is `R wall/chain` divided by the row's
`wall/chain`; values above 1.0 are faster than R.

| Implementation | Device | Dtype | Execution | Total wall (s) | Wall / chain (s) | Chains / s | R-relative speedup | Truth-relative error |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| bspcov 1.0.3 | CPU | float64 | 4 parallel workers | 488.126 | 122.032 | 0.00819 | 1.000x | 0.18144 |
| pybspcov | CPU | float64 | 4-chain vmap | 463.722 | 115.931 | 0.00863 | 1.053x | 0.18141 |
| pybspcov | GPU | float64 | 4 sequential fits | 663.773 | 165.943 | 0.00603 | 0.735x | 0.18158 |
| pybspcov | CPU | float32 | 4-chain vmap | 394.713 | 98.678 | 0.01013 | 1.237x | 0.17997 |
| pybspcov | GPU | float32 | 4 sequential fits | 321.042 | 80.260 | 0.01246 | 1.520x | 0.18147 |

All five posterior means were finite, symmetric, and positive definite. Their
truth-relative errors differ by at most 0.00161 in this run. Float32 GPU had
the best normalized throughput, while float64 GPU was slower than both R and
Python CPU. This is performance and numerical-sanity evidence for one fixture,
not a convergence or posterior-equivalence claim.

### First fit, including compilation

| Estimator | p | CPU f32 | GPU f32 | CPU f64 | GPU f64 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM | 25 | 7.878 | 17.843 | 7.629 | 21.215 |
| BM | 50 | 7.378 | 18.760 | 7.706 | 20.716 |
| BM | 100 | 7.985 | 18.838 | 8.372 | 24.275 |
| BM | 200 | 12.551 | 20.731 | 13.968 | 27.758 |
| SBM | 25 | 13.045 | 24.441 | 14.303 | 28.551 |
| SBM | 50 | 12.005 | 24.328 | 12.494 | 25.687 |
| SBM | 100 | 11.012 | 25.547 | 12.089 | 29.008 |
| SBM | 200 | 13.817 | 27.172 | 14.617 | 31.868 |

## Method and environment

- Estimator source baseline: merge commit `69e3535`.
- Benchmark runner revision: `80754461bea02d1d7f1deb2b19abfc9a00dac2d1`;
  it changes no files under `src/pybspcov`.
- Workload: `p = 25, 50, 100, 200`, `n = 3p`, target density 0.05,
  burn-in 1, retained samples 2, warmed repetitions 3, seed 20260803.
- CPU: two Intel Xeon Gold 5220R sockets, 48 physical cores, 96 threads.
- GPU: NVIDIA GeForce RTX 3090, 24 GiB, driver 575.57.08.
- Runtime: Linux 6.8.0-60, Python 3.13.6, JAX/JAXLIB 0.11.0,
  NumPy 2.5.1, locked CUDA 12 optional dependencies.
- Each fit used a fresh estimator and PRNG key. All public outputs were
  synchronized before stopping the timer. The first fit and three warmed fits
  were recorded separately.
- The R comparison used `bspcov` 1.0.3, burn-in 50, retained samples 50, and
  four chains. The Python compile-plus-execution measurements were recorded
  separately in `p200-r-comparison.json`; R worker startup remains included in
  its measured `bmspcov()` call.

The following command template produced each JSONL file. Substitute the
estimator, backend, dtype, X64 flag, and output filename. GPU runs also add
`--extra cuda12` to `uv run`.

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 uv run --frozen python \
  benchmarks/sbm_public_scaling.py \
  --estimator bm --device cpu --dtype float32 \
  --dimensions 25 50 100 200 --density 0.05 --n-factor 3 \
  --burnin 1 --samples 2 --repetitions 3 --seed 20260803 \
  --output benchmarks/baselines/0.1.0.dev0/bm-cpu-float32.jsonl
```

## Files and limitations

The eight adjacent JSONL files contain four dimension records each. The
adjacent `p200-r-comparison.json` contains the longer R comparison. These raw
files include timings, fixture hashes, validity checks, and runtime provenance;
they are the authoritative values, while this report rounds numbers for
readability.

This is one run on one dual-socket server. CPU affinity, NUMA placement, power
state, and system load were not isolated. The chains are intentionally too
short for posterior-quality claims. Larger `p`, longer chains, repeated host
sessions, and profiler evidence are required before changing the implementation
solely on these timings.
