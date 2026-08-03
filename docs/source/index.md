# pybspcov

`pybspcov` is a pure-Python, JAX-based port of the R package
[`bspcov`](https://github.com/statjs/bspcov).

The development package includes the initial `BMSPCov` estimator. `SBMSPCov`
remains under development.

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

```{toctree}
:maxdepth: 2
:caption: Contents

installation
development
```
