# R Example Parity and Performance Benchmark Design

**Status:** Approved
**Date:** 2026-08-01
**Baseline:** `bspcov` 1.0.3 `bmspcov` documentation example

## Goal

Demonstrate that `pybspcov` and the R `bspcov` package produce statistically
similar BM posterior results on the upstream `n = 20`, `p = 5` example, while
making CPU and GPU runtime differences reproducible and explicit.

## Statistical comparison

Both implementations consume the same versioned data matrix, true covariance,
initial covariance, priors, burn-in length, and retained sample count. Random
streams are intentionally independent. The report compares posterior covariance
means, standard deviations, quantiles, and Frobenius RMSE against the true
covariance. Automated parity checks use Monte Carlo-aware tolerances; they do
not require sample-by-sample identity.

## Timing comparison

The R runner records end-to-end sampling time. The JAX runner records first-call
compile-plus-execution time, warmed steady-state sampling time, and end-to-end
time separately. Every result records package versions, dtype, device, `n`,
`p`, burn-in, retained samples, chains, and repetition count. GPU execution is
reported only when a JAX GPU device is actually selected.

Absolute wall-clock thresholds never fail correctness tests. Raw CSV/JSON data
remain machine-readable, and a concise Markdown report explains the hardware
and commands used so timing claims are not detached from their environment.

## Scope

The first benchmark covers the upstream BM example with one chain. The harness
is structured so SBM and dimension/sparsity grids can be added after those
estimators exist. It does not add a runtime R dependency to `pybspcov`.

## Isolation and integration

Benchmark work lives on `bench/r-example-parity` in
`.worktrees/r-example-parity`. Core sampler APIs remain on
`feat/bm-sbm-core`. The benchmark branch may depend on those APIs but must not
modify their production implementation. Integration happens only after both
branches pass their focused tests.
