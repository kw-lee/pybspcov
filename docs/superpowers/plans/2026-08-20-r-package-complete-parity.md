# R Package Complete Public-Feature Parity Implementation Plan

> Execute every behavior change with a failing focused test first. Commit at
> coherent review boundaries and run the full verification only once after the
> integrated tree is unchanged.

## 1. Establish the integration baseline

- Record the approved design and this plan on
  `feat/r-package-complete-parity`.
- Cherry-pick the `feat/bandppp` implementation commit onto the current
  baseline, resolving conflicts in favor of current speed and documentation
  work.
- Run only the BandPPP focused tests and inspect its R fixture provenance.

## 2. Complete the shared PPP estimator contract

- Add failing tests for common validation, packed/full sample shapes,
  posterior summaries, ArviZ conversion, typed-key reproducibility, device
  placement, and atomic refits.
- Refactor the reviewed BandPPP implementation only as required to satisfy the
  common contract.
- Add failing tests for inverse-Wishart sampling, hard/soft thresholding, and
  positive-definite adjustment.
- Implement `ThresholdPPP` and expose it from the package.
- Generate and check in versioned R fixtures and generator scripts for
  threshold PPP parity.

## 3. Implement cross-validation

- Add deterministic unit tests for predictive log density, fold construction,
  spectral norm loss, sorting, and best-parameter selection.
- Implement typed `BandCVResult` and `ThresholdCVResult` containers.
- Implement `cross_validate_band_ppp` and
  `cross_validate_threshold_ppp` with explicit key splitting.
- Add small R 1.0.3 fixtures for independent score validation.

## 4. Package datasets and preprocessing

- Convert the three upstream R data objects into deterministic compressed
  package resources, recording source version and SHA-256 metadata.
- Add failing loader tests for shapes, orientation, names, mapping/attribute
  access, tuple output, optional frames, and installed-resource access.
- Implement `DatasetBunch`, `load_colon`, and `load_sp500`.
- Add R preprocessing fixtures, then implement and test `preprocess_colon` and
  `preprocess_sp500` against them.

## 5. Add visualization and public documentation

- Add non-interactive tests for ArviZ trace selection, posterior heatmaps,
  quantile/uncertainty plots, CV curves, and explicit file saving.
- Implement plotting helpers with actionable optional-dependency errors.
- Update extras and lock data, then document the complete public API and the
  R-to-Python mapping in Sphinx.
- Add short deterministic examples for every feature family; do not add
  notebooks.

## 6. Close parity and deliver

- Add a manifest test showing that every documented R 1.0.3 public function
  and dataset has a Python API, test, and documentation entry.
- Run focused statistical parity tests with `JAX_ENABLE_X64=1` on CPU.
- Run the full pytest suite once, followed by Ruff, strict mypy, package build
  and metadata checks, and strict Sphinx HTML build.
- Review the complete diff, commit all useful changes, ensure a clean worktree,
  and prepare the local PR handoff with commands, device/precision, and known
  limitations.
