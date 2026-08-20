# pybspcov

`pybspcov` is a pure-Python, JAX-based port of the R package
[`bspcov`](https://github.com/statjs/bspcov).

The development package includes `BandPPP`, `ThresholdPPP`, `BMSPCov`, and `SBMSPCov`.

## BandPPP contract and example

`BandPPP` ports `bspcov::bandPPP`. It draws from the inverse-Wishart posterior,
zeros covariance entries outside the requested band, and shifts each processed
draw when needed so its minimum eigenvalue is at least `epsilon`. Inputs must
already be centered; the posterior scale is `X.T @ X + prior_scale`.

Python's `bandwidth`, `epsilon`, `prior_scale`, and `prior_df` correspond to R's
`k`, `eps`, `A`, and `nu`. The prior defaults are `A=I` and `nu=p+k`; omitting
`epsilon` uses `(log(k) ** 2) * (k + log(p)) / n`. BandPPP draws are independent,
so it has no burn-in parameter.

```python
import arviz as az
import jax
import jax.numpy as jnp

from pybspcov import BandPPP

data_key, fit_key = jax.random.split(jax.random.key(42))
X = jax.random.normal(data_key, shape=(100, 10), dtype=jnp.float32)
X_centered = X - X.mean(axis=0)

model = BandPPP(
    bandwidth=2,
    epsilon=0.01,
    n_samples=2_000,
    n_chains=4,
    dtype="float32",
    device="cpu",
).fit(X_centered, key=fit_key)

posterior_mean = model.estimate()
intervals = model.quantile([0.025, 0.975])
diagnostics = az.summary(model.to_arviz(), var_names=["covariance"])
```

`posterior_samples_packed_` stores the R-compatible lower triangle with shape
`(n_chains, n_samples, p * (p + 1) // 2)`. `posterior_samples_` reconstructs
full symmetric matrices. `adjusted_draws_` identifies draws that required the
diagonal eigenvalue correction.

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

## Posterior summaries

After fitting any estimator, `estimate()` returns the pooled posterior mean,
`quantile(probs)` returns R Type-7 elementwise quantiles with shape
`(n_probs, p, p)`, and `summary(probs)` returns a `PosteriorSummary`. Summary
standard deviations use `ddof=1`, as in the upstream R summary, and all array
results remain on the fitted JAX device. Probability inputs must be non-empty,
real, finite, and between zero and one.

## ArviZ analysis

Install `pybspcov` with the `analysis` extra to add ArviZ and its Matplotlib
plotting backend. All estimators then expose their retained covariance draws as
an ArviZ DataTree:

```python
import arviz as az

inference_data = model.to_arviz()
summary = az.summary(inference_data, var_names=["covariance"])
trace = az.plot_trace(
    inference_data,
    var_names=["covariance"],
    coords={"row": [0], "column": [1]},
    backend="matplotlib",
)
```

The `posterior/covariance` variable has dimensions `chain`, `draw`, `row`,
and `column`. Conversion preserves separate chains for ESS and R-hat
calculation, reconstructs each symmetric covariance matrix, and transfers the
result to host memory. For `BMSPCov` and `SBMSPCov`, sampler acceptance values
include burn-in sweeps and therefore are not exported as an ArviZ
`sample_stats` group.

## SBMSPCov contract

`SBMSPCov` follows the same centered-input, typed-key, sample-layout, dtype, and
device contract as `BMSPCov`. `device=None` uses JAX's default device; explicit
`"cpu"` or `"gpu"` requests fail when that backend is unavailable, and fitted
arrays stay on the selected device. The default FNR screen uses correlation `0.25`,
false-negative rate `0.05`, and `1000` cutoff simulations. Correlation screening
is selected with `cutoff_method="correlation"` and defaults to a retained
fraction of `0.2`.

Screening is computed once per fit by default and the resulting
`screening_mask_` is shared by all chains. Use `screening_scope="chain"` for
`bspcov` 1.0.3 per-chain FNR screening. That mode publishes a `(chain, p, p)`
mask, a `(chain,)` FNR cutoff, per-chain initial covariances, and per-chain
screening diagnostic tuples. Correlation screening is deterministic but honors
the same output shapes when chain scope is requested. Packed and reconstructed
posterior draws retain the same chain-first shapes as `BMSPCov`.

The R-derived tests establish screening and orchestration contracts and compare
the public correlation-screened SBM posterior with a versioned `bspcov 1.0.3`
fixture using combined Monte Carlo uncertainty. Broader scientific parity and
performance claims still require representative datasets, dimensions, and
complete runtime provenance.

```{toctree}
:maxdepth: 2
:caption: Contents

installation
datasets
visualization
parity
api
development
```
