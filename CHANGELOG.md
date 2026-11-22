# Changelog

All notable changes to this project will be documented in this file.

## 0.1.0.dev0 - 2026-08-01

- Establish the installable package, test, documentation, and CI scaffold.
- Implement the public `BMSPCov` and `SBMSPCov` estimators.
- Add pooled posterior `estimate`, `quantile`, and `summary` methods for both
  estimators.
- Add an executable BM/SBM quickstart and a wheel installation smoke check.
- Implement the public `BandPPP` and `ThresholdPPP` estimators with the same
  pooled posterior summary methods.
- Add the fail-closed R `bspcov` 1.0.3 comparison harness under
  `benchmarks/r_comparison/` and publish its results in the README.
- Remove the superseded `r_example`, `r_scaling`, and v0.1 baseline benchmark
  runners and the internal plan and design records under `docs/superpowers/`.
- Document upstream provenance, protected-main development, isolated worktrees,
  AI-assisted contribution controls, and GPL-2.0-or-later licensing.
