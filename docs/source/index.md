# pybspcov

`pybspcov` is a pure-Python, JAX-based port of the R package
[`bspcov`](https://github.com/statjs/bspcov).

The development package includes the initial `BMSPCov` and `SBMSPCov`
estimators.

## BMSPCov contract

`BMSPCov` expects a two-dimensional, already-centered data matrix with rows as
observations and columns as variables. Pass one typed master key created by
`jax.random.key`; the estimator splits independent chain keys internally. The
default float64 path requires starting Python with `JAX_ENABLE_X64=1`.
`dtype="float32"` is experimental and should be validated against float64 and
the R reference implementation before scientific use.

For `p` variables, fitted packed covariance draws have shape
`(n_chains, n_samples, p * (p + 1) // 2)`. The `posterior_samples_` property
reconstructs symmetric draws with shape `(n_chains, n_samples, p, p)`, and
`covariance_` is their posterior mean pooled across chains and retained samples.

## SBMSPCov contract

`SBMSPCov` follows the same centered-input, typed-key, sample-layout, dtype, and
device contract as `BMSPCov`. `device=None` uses JAX's default device; explicit
`"cpu"` or `"gpu"` requests fail when that backend is unavailable, and fitted
arrays stay on the selected device. The default FNR screen uses correlation `0.25`,
false-negative rate `0.05`, and `1000` cutoff simulations. Correlation screening
is selected with `cutoff_method="correlation"` and defaults to a retained
fraction of `0.2`.

Screening is computed once per Python fit and the resulting `screening_mask_` is
shared by all chains. This is an intentional reproducibility difference from
`bspcov` 1.0.3, which consumes independent screening randomness per chain in its
FNR path. `screening_cutoff_` is an array for FNR screening and `None` for
correlation screening. Packed and reconstructed posterior draws have the same
shapes as `BMSPCov`; `diagnostics_` additionally records the screening method,
jitter, and active/screened edge counts.

The R-derived tests establish screening and orchestration contracts and compare
the public correlation-screened SBM posterior with a versioned `bspcov 1.0.3`
fixture using combined Monte Carlo uncertainty. Broader scientific parity and
performance claims still require representative datasets, dimensions, and
complete runtime provenance.

```{toctree}
:maxdepth: 2
:caption: Contents

installation
api
development
```
