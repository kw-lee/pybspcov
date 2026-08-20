# 0.1.0.dev1 short-chain speedup comparison

This artifact repeats the 32-cell short-chain matrix from
`0.1.0.dev0` after the speedup changes. It is a small performance and
numerical-sanity check, not a convergence study or a universal performance
claim.

Across these 32 host-specific cells, 27 had a lower warmed median in dev1.
The geometric mean of `dev0 / dev1` was 1.098x, with individual ratios from
0.755x to 1.745x. The table retains all five regressions: the largest were BM
CPU float64 at p=50 (0.755x) and p=25 (0.838x). Larger BM cells benefited more
consistently. All GPU BM cells improved by 1.035x to 1.220x; GPU SBM remained
near parity to modestly faster at 0.999x to 1.061x.

## Warmed comparison

Times are seconds. `dev0 / dev1` above 1 means dev1 was faster. Dev0 uses
`warmed_fit_seconds.median`; dev1 uses `timing_summary.median`. Both are
one-chain warmed wall times, so no per-chain normalization changes their
meaning.

| Estimator | p | Device | Dtype | dev0 | dev1 | dev0 / dev1 |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| BM | 25 | CPU | float32 | 0.157 | 0.161 | 0.975x |
| BM | 25 | CPU | float64 | 0.140 | 0.167 | 0.838x |
| BM | 25 | GPU | float32 | 0.277 | 0.244 | 1.135x |
| BM | 25 | GPU | float64 | 0.472 | 0.456 | 1.035x |
| BM | 50 | CPU | float32 | 0.271 | 0.202 | 1.345x |
| BM | 50 | CPU | float64 | 0.241 | 0.319 | 0.755x |
| BM | 50 | GPU | float32 | 0.549 | 0.489 | 1.124x |
| BM | 50 | GPU | float64 | 0.929 | 0.884 | 1.051x |
| BM | 100 | CPU | float32 | 0.909 | 0.521 | 1.745x |
| BM | 100 | CPU | float64 | 0.878 | 0.734 | 1.196x |
| BM | 100 | GPU | float32 | 1.131 | 0.995 | 1.136x |
| BM | 100 | GPU | float64 | 2.030 | 1.884 | 1.078x |
| BM | 200 | CPU | float32 | 5.259 | 3.197 | 1.645x |
| BM | 200 | CPU | float64 | 6.552 | 4.152 | 1.578x |
| BM | 200 | GPU | float32 | 2.513 | 2.060 | 1.220x |
| BM | 200 | GPU | float64 | 5.135 | 4.374 | 1.174x |
| SBM | 25 | CPU | float32 | 0.119 | 0.117 | 1.022x |
| SBM | 25 | CPU | float64 | 0.120 | 0.121 | 0.992x |
| SBM | 25 | GPU | float32 | 0.278 | 0.262 | 1.061x |
| SBM | 25 | GPU | float64 | 0.469 | 0.458 | 1.024x |
| SBM | 50 | CPU | float32 | 0.197 | 0.181 | 1.089x |
| SBM | 50 | CPU | float64 | 0.207 | 0.187 | 1.108x |
| SBM | 50 | GPU | float32 | 0.524 | 0.504 | 1.039x |
| SBM | 50 | GPU | float64 | 0.905 | 0.903 | 1.002x |
| SBM | 100 | CPU | float32 | 0.376 | 0.359 | 1.047x |
| SBM | 100 | CPU | float64 | 0.423 | 0.408 | 1.035x |
| SBM | 100 | GPU | float32 | 1.035 | 1.011 | 1.025x |
| SBM | 100 | GPU | float64 | 1.806 | 1.807 | 0.999x |
| SBM | 200 | CPU | float32 | 2.040 | 1.914 | 1.066x |
| SBM | 200 | CPU | float64 | 2.326 | 2.266 | 1.026x |
| SBM | 200 | GPU | float32 | 2.115 | 2.025 | 1.044x |
| SBM | 200 | GPU | float64 | 3.830 | 3.712 | 1.032x |

### First fit, including compilation

These dev1 measurements are recorded separately and are not used in the
speedup ratios.

| Estimator | p | CPU f32 | CPU f64 | GPU f32 | GPU f64 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM | 25 | 7.542 | 7.654 | 18.287 | 20.841 |
| BM | 50 | 7.466 | 7.589 | 18.272 | 20.801 |
| BM | 100 | 7.180 | 7.987 | 19.053 | 24.213 |
| BM | 200 | 10.592 | 11.103 | 20.470 | 26.597 |
| SBM | 25 | 12.383 | 13.322 | 25.033 | 27.854 |
| SBM | 50 | 11.629 | 12.455 | 24.091 | 25.163 |
| SBM | 100 | 10.918 | 11.901 | 25.255 | 28.590 |
| SBM | 200 | 13.037 | 14.105 | 27.317 | 32.072 |

## Protocol and validation

- Workload: p=25, 50, 100, 200; n=3p; target density 0.05;
  burn-in 1; retained samples 2; one chain; three warmed repetitions; seed
  20260803.
- CPU execution model was `parallel` with one chain. GPU execution model was
  explicitly `sequential` with one chain.
- Each dimension used one compile-plus-execution fit followed by three
  synchronized measured fits. Public outputs were blocked before stopping the
  clock.
- Fixture policy was explicitly `estimator`: float32 cells generated the
  fixture in float32 and float64 cells in float64, matching the historical
  dev0 policy. Every corresponding dev0/dev1 SHA256 matched.
- The PRNG policy splits `jax.random.key(seed)` into one warm-up key and one
  key per measured repetition. Sequential fits split each repetition key by
  chain; parallel fits pass one key to the estimator, which derives per-chain
  keys.
- All 96 measured repetitions produced finite, symmetric, positive-definite
  posterior means and finite truth-relative Frobenius errors. Each repetition
  recorded three accepted sweeps and zero rejected sweeps.
- The raw JSONL timing summaries and every displayed ratio were computed from
  full-precision values; tables are rounded only for readability.

## Environment

- Speedup integration base: `23b7736`; benchmark runner revision recorded by
  every dev1 row: `9b7a0adceb75a9d182f8156129ea9222e926f90e`, with a clean
  working tree at measurement start.
- Linux 6.8.0-60-generic, Python 3.13.6, JAX/JAXLIB 0.11.0, NumPy 2.5.1.
- CPU: two Intel Xeon Gold 5220R sockets, 48 physical cores and 96 threads.
- GPU: NVIDIA GeForce RTX 3090, 24576 MiB, driver 575.57.08.
- CPU runs set `JAX_PLATFORMS=cpu`; GPU runs set `JAX_PLATFORMS=cuda` and
  used the locked `cuda12` optional dependencies. `JAX_ENABLE_X64` was 0
  for float32 and 1 for float64.
- `XLA_FLAGS`, `XLA_PYTHON_CLIENT_ALLOCATOR`,
  `XLA_PYTHON_CLIENT_PREALLOCATE`,
  `XLA_PYTHON_CLIENT_MEM_FRACTION`, and `CUDA_VISIBLE_DEVICES` were unset.
- Before GPU measurement, `nvidia-smi` showed 9 MiB used, 0% utilization,
  P8, 42 C, and no compute processes. After each GPU pair, memory returned to
  9 MiB and utilization to 0%; the final observed temperature was 53 C.
  In-process peak allocation was not sampled, so this artifact does not
  quantify XLA preallocation or peak GPU memory.
- The eight configurations ran serially in separate Python processes. This
  avoids overlap between cells but also prevents process-local compilation
  cache reuse between files.

## Commands

The environment was prepared with:

```bash
uv sync --locked --all-groups
```

Each CPU file used this exact command shape:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=<0-or-1> uv run --frozen python \
  benchmarks/sbm_public_scaling.py \
  --estimator <bm-or-sbm> --device cpu --dtype <float32-or-float64> \
  --fixture-dtype-policy estimator --dimensions 25 50 100 200 \
  --density 0.05 --n-factor 3 --burnin 1 --samples 2 --chains 1 \
  --repetitions 3 --seed 20260803 \
  --output benchmarks/baselines/0.1.0.dev1/<estimator>-cpu-<dtype>.jsonl
```

Each GPU file used this exact command shape:

```bash
JAX_PLATFORMS=cuda JAX_ENABLE_X64=<0-or-1> \
  uv run --frozen --extra cuda12 python benchmarks/sbm_public_scaling.py \
  --estimator <bm-or-sbm> --device gpu --dtype <float32-or-float64> \
  --fixture-dtype-policy estimator --dimensions 25 50 100 200 \
  --density 0.05 --n-factor 3 --burnin 1 --samples 2 --chains 1 \
  --repetitions 3 --execution-mode sequential --seed 20260803 \
  --output benchmarks/baselines/0.1.0.dev1/<estimator>-gpu-<dtype>.jsonl
```

For each estimator (`bm`, `sbm`) and dtype (`float32`, `float64`), the
corresponding template was run once. The X64 flag was 0 for float32 and 1 for
float64, producing the eight adjacent JSONL filenames.

## Limitations

The chains are intentionally too short to establish convergence, scientific
equivalence, or precision-accuracy tradeoffs. Matching fixture hashes controls
the input data but not host load, thermal state, power state, or run-order
effects across the two historical sessions. CPU affinity, NUMA placement, and
system load were not isolated. Three warmed repetitions cannot distinguish
small improvements from timing noise, especially the near-parity SBM cells.
The result covers one host and dimensions only through p=200. It does not
justify a universal execution default or conclusions about batching unrelated
datasets.

The eight JSONL files beside this README are authoritative. The immutable
`0.1.0.dev0` directory was not modified.
