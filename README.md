# pybspcov

`pybspcov` is a pure-Python, JAX-accelerated port of the R package
[`bspcov`](https://github.com/statjs/bspcov) for Bayesian sparse covariance
estimation.

> [!IMPORTANT]
> This development repository contains `BandPPP`, `BMSPCov`, and `SBMSPCov`.
> A published package release is not yet available.

Repository: <https://github.com/kw-lee/pybspcov>

## Initial scope

The package provides `BandPPP`, a post-processed inverse-Wishart posterior for
banded covariance matrices; `BMSPCov`, based on the beta-mixture shrinkage
prior; and `SBMSPCov`, based on the screened beta-mixture shrinkage prior.

The implementation uses JAX and XLA for CPU and NVIDIA GPU execution. The
package itself does not contain custom C, C++, or CUDA extensions.

## R/Python performance comparison

`benchmarks/r_comparison/` compares R `bspcov` 1.0.3 with `pybspcov` on BM,
SBM, BandPPP, and ThresholdPPP at `p = 50, 100, 200`. One comparison runs
Python on the GPU against R on 8 CPU workers. The other runs both on a single
CPU core in float64. The tables below are rendered only when all four float64
parity gates and all 60 timing cells pass. See the
[protocol and reproduction commands](benchmarks/r_comparison/README.md).

<!-- r-python-benchmark:start -->
### R `bspcov` comparison

All four float64 parity gates passed against `bspcov` 1.0.3. The headline compares total wall time for R CPU 8-worker execution with Python GPU vmap-8 execution on the same inputs.

Measured on a shared host: the 60 cells ran one at a time, no other process used the GPU during the run, and the load average stayed between 0.1 and 6.6.

| Method | Validated Python dtype | R CPU 8-worker (s) | Python GPU vmap-8 (s) | p=200 speedup | p=50/100/200 geometric mean |
| --- | --- | ---: | ---: | ---: | ---: |
| BM | float32 | 74.655 | 26.401 | 2.828x | 1.818x |
| SBM | float32 | 35.590 | 19.899 | 1.789x | 1.115x |
| BandPPP | float32 | 1.308 | 0.058 | 22.654x | 29.509x |
| ThresholdPPP | float32 | 1.272 | 0.043 | 29.738x | 34.739x |

The matching single-core float64 baseline separates implementation differences from GPU and multi-chain acceleration.

| Method | R CPU (s) | Python CPU (s) | Python speedup | Cold optimized speedup |
| --- | ---: | ---: | ---: | ---: |
| BM | 48.107 | 23.360 | 2.059x | 2.488x |
| SBM | 14.472 | 11.551 | 1.253x | 1.504x |
| BandPPP | 0.464 | 0.420 | 1.105x | 1.275x |
| ThresholdPPP | 0.708 | 0.335 | 2.115x | 1.734x |

These host- and workload-specific results were recorded at revision `3ffbdd59f52f70e48cf1158f82058664f1398cd8`.

[Full protocol, environment, and limitations](https://github.com/kw-lee/pybspcov/blob/main/benchmarks/baselines/0.1.0.dev0/r-python/README.md)
<!-- r-python-benchmark:end -->

## Quickstart

Run the BM and SBM estimators on a small centered data matrix:

```bash
uv run python examples/quickstart.py
```

The example runs short float32 chains to check the installation; they are too
short for inference. A complete public API call looks like:

```python
import jax
import jax.numpy as jnp
from pybspcov import BMSPCov

data_key, fit_key = jax.random.split(jax.random.key(42))
X = jax.random.normal(data_key, shape=(100, 10), dtype=jnp.float32)
X_centered = X - X.mean(axis=0)

model = BMSPCov(
    n_samples=1_000,
    burnin=1_000,
    dtype="float32",
    device="cpu",
)
model.fit(X_centered, key=fit_key)
posterior_mean = model.covariance_
posterior_quantiles = model.quantile([0.025, 0.5, 0.975])
posterior_summary = model.summary()
```

`BMSPCov` does not center its input. For `p` variables,
`posterior_samples_packed_` has shape
`(n_chains, n_samples, p * (p + 1) // 2)`, while the reconstructed
`posterior_samples_` has shape `(n_chains, n_samples, p, p)`. The
`dtype="float32"` path is experimental and should be compared with the default
float64 and R reference results before scientific use.
Enable the default float64 path before Python starts with
`JAX_ENABLE_X64=1`.

All estimators provide `estimate()`, `quantile()`, and `summary()` after
fitting. These methods pool retained draws across all chains, matching the R
`bspcov` post-processing convention. Quantiles have shape `(n_probs, p, p)`;
`PosteriorSummary` contains the pooled mean, sample standard deviation,
requested quantiles, and fitted chain/sample counts. Returned statistics remain
JAX arrays on the fitted device.

Install the optional analysis dependencies to use ArviZ diagnostics and plots:

```bash
uv sync --extra analysis
```

```python
import arviz as az

inference_data = model.to_arviz()
diagnostics = az.summary(inference_data, var_names=["covariance"])
trace = az.plot_trace(
    inference_data,
    var_names=["covariance"],
    coords={"row": [0], "column": [1]},
    backend="matplotlib",
)
```

`to_arviz()` preserves the fitted chain and retained-draw axes and exposes the
full symmetric matrix as `posterior/covariance` with `row` and `column`
coordinates. Conversion transfers the covariance draws from their JAX device
to host memory. ArviZ supplies effective sample size, R-hat, Monte Carlo
standard errors, intervals, and visualization; `summary()` does not compute
them.

`SBMSPCov` takes the same input, key, dtype, and device arguments as `BMSPCov`
and stores draws in the same packed layout. With `device=None`, JAX selects its
default device; `device="cpu"` and `device="gpu"` request that backend and
raise `RuntimeError` when it is unavailable. Fitted arrays remain on the
selected device.
Its default `cutoff_method="fnr"` uses `fnr_correlation=0.25`,
`false_negative_rate=0.05`, and `n_cutoff_simulations=1000`; the alternative
`cutoff_method="correlation"` uses `retained_fraction=0.2`. Screening runs
once per fit by default, so all chains share `screening_mask_`.
Set `screening_scope="chain"` for the per-chain FNR cutoffs and supports used
by `bspcov` 1.0.3. In that mode, `screening_mask_` has shape `(chain, p, p)`,
`screening_cutoff_` has shape `(chain,)`, and the screening diagnostics contain
per-chain tuples. `screening_cutoff_` is populated only for FNR screening.

The checked-in R fixtures check the screening formulas and the
correlation-screened SBM posterior on the upstream `p=5` example. They are
regression tests on that one example.

## Development installation

The development package can be installed with its CPU dependencies as follows.

```bash
git clone https://github.com/kw-lee/pybspcov.git
cd pybspcov
uv sync --all-groups
uv run python -c "import pybspcov; print(pybspcov.__version__)"
```

For NVIDIA GPU support, use the accelerator-specific JAX installation selected
by the project:

```bash
uv sync --extra cuda12
uv run python -c "import jax; print(jax.devices())"
```

Confirm driver and platform compatibility against the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Development

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Branch,
worktree, parallel ownership, and protected-`main` rules are defined in the
[development workflow](docs/development/workflow.md).

The [documentation layout](docs/README.md) keeps Sphinx sources separate from
maintainer documents.

Repository content is written in English. AI-assisted contributions are
treated as unreviewed suggestions; [CONTRIBUTING.md](CONTRIBUTING.md) lists the
review they must pass.

## Maintainer

Kyeongwon Lee (<kwlee1718@gmail.com>)

## Upstream project

This project is derived from `bspcov`, version 1.0.3 at the time of bootstrap:

- Development repository: <https://github.com/statjs/bspcov>
- CRAN mirror: <https://github.com/cran/bspcov>

The initial port is based on methods described in:

- Kyoungjae Lee, Seongil Jo, and Jaeyong Lee (2022). The beta-mixture shrinkage
  prior for sparse covariances with near-minimax posterior convergence rate.
  *Journal of Multivariate Analysis*, 192, 105067.
  <https://doi.org/10.1016/j.jmva.2022.105067>
- Kyoungjae Lee, Seongil Jo, Kyeongwon Lee, and Jaeyong Lee (2026). Scalable
  and optimal Bayesian inference for sparse covariance matrices via screened
  beta-mixture prior. *Bayesian Analysis*, 21(2).
  <https://doi.org/10.1214/24-BA1495>

## License

`pybspcov` follows the upstream package under the GNU General Public License,
version 2 or any later version (`GPL-2.0-or-later`). See
[`LICENSE.md`](LICENSE.md) for the complete GPL v2 license text.
