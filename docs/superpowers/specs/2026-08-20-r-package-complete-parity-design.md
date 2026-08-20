# R Package Complete Public-Feature Parity Design

## Goal

Implement every documented user-facing feature in `bspcov` 1.0.3 while
preserving pybspcov's idiomatic estimator API. The deliverable is a reviewable
local branch named `feat/r-package-complete-parity`; it will not be pushed or
merged by the implementation agent.

## Public API

The existing `BMSPCov` and `SBMSPCov` APIs remain stable. Add `BandPPP` and
`ThresholdPPP` estimators with the same `fit`, `estimate`, `quantile`,
`summary`, and `to_arviz` workflow. Add deterministic
`cross_validate_band_ppp` and `cross_validate_threshold_ppp` functions that
return typed result objects with score tables, best parameters, and plotting
methods.

Expose posterior trace plots through ArviZ. Provide Matplotlib/Seaborn helpers
for posterior-mean heatmaps, quantile heatmaps, uncertainty views, and
cross-validation curves. Plotting functions return figure and axes objects and
do not open a window. A dedicated save helper covers the upstream
`save_quantile_plot` behavior.

Do not reproduce R spellings such as `bandPPP` as Python aliases. The Sphinx
documentation will contain a complete R-to-Python API mapping.

## Numerical Contract

PPP estimators sample the initial posterior
`IW_p(A + X.T @ X, nu + n)` from explicit JAX keys. `BandPPP` applies a
banding operator. `ThresholdPPP` preserves the diagonal and applies hard or
soft thresholding to off-diagonal entries. Both add a diagonal correction when
the processed covariance has minimum eigenvalue below `epsilon`.

Posterior samples retain the existing chain-first representation and the R
column-major packed-lower-triangle convention. Random keys are split
deterministically across chains, samples, folds, and tuning combinations.
Repeated calls with the same inputs and key are reproducible.

Band cross-validation reproduces the upstream leave-one-out posterior
predictive log likelihood. Threshold cross-validation reproduces the upstream
ten-fold spectral-norm loss. Correctness and deterministic execution take
priority over parallel speed in this branch.

`SBMSPCov` keeps shared per-fit screening as its backward-compatible default.
An explicit `screening_scope="chain"` mode supplies upstream per-chain FNR
screening semantics.

## Datasets and Preprocessing

Package the `colon`, `tissues`, and `SP500` data from `bspcov` 1.0.3 as
compressed resources with provenance and checksums. A lightweight
`DatasetBunch` supports both mapping and attribute access. `load_colon` and
`load_sp500` return arrays by default, support `return_X_y` where meaningful,
and optionally return pandas frames.

`load_colon` follows Python's samples-by-features convention and documents the
transpose relative to the R object. `preprocess_colon` performs the log
transform, two-group Welch statistic ranking, and top-50 feature selection.
`preprocess_sp500` computes monthly adjusted returns, estimates the factor
count, and returns POET-style factor residuals and sector metadata. The Python
implementation must not require R packages at runtime.

Base installation supports array loading and all statistical algorithms. A
`data` extra supplies pandas frame integration. The `analysis` extra supplies
ArviZ, Matplotlib, and Seaborn.

## Validation and Errors

Validate dimensions, finite values, prior scale positive-definiteness,
degrees of freedom, bandwidth, thresholds, and epsilon before sampling. Do not
silently center observations. Publish fitted state only after successful
completion. Missing optional dependencies raise actionable `ImportError`
messages, and cross-validation never silently drops a failed combination.

Use test-driven development. Independently generated R 1.0.3 fixtures verify
PPP posterior summaries, cross-validation scores, and preprocessing outputs.
Theoretical inverse-Wishart moments provide an implementation-independent
statistical check. Structural tests cover symmetry, positive definiteness,
banding, threshold support, dtype/device placement, and key reproducibility.
Probabilistic parity is assessed with Monte Carlo uncertainty rather than
draw-for-draw equality across different random-number generators.

## Documentation and Delivery

Sphinx documents every public symbol, complete R-to-Python mappings, estimator
and dataset examples, visualization, dtype/device/key behavior, optional
dependencies, and validated limitations. Final evidence comprises focused
tests, the full test suite, Ruff, strict mypy, package build checks, and a
strict Sphinx build on CPU with float64 enabled where statistical parity needs
it.

The stop condition is a clean, committed local feature branch plus a proposed
PR title and body. No push, PR creation, merge, or protected-branch write is
authorized.
