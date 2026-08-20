# pybspcov

`pybspcov` is a pure-Python, JAX-accelerated port of the R package
[`bspcov`](https://github.com/statjs/bspcov) for Bayesian sparse covariance
estimation.

> [!IMPORTANT]
> This development repository contains the initial `BMSPCov` and `SBMSPCov`
> estimators. A published package release is not yet available.

Repository: <https://github.com/kw-lee/pybspcov>

## Initial scope

The package provides `BMSPCov`, based on the beta-mixture shrinkage prior, and
`SBMSPCov`, based on the screened beta-mixture shrinkage prior.

The implementation uses JAX and XLA for CPU and NVIDIA GPU execution. The
package itself does not contain custom C, C++, or CUDA extensions.

## Quickstart

Run both estimators on a small centered data matrix:

```bash
uv run python examples/quickstart.py
```

The example uses short float32 chains so that it serves as an installation
smoke test, not as a scientific analysis. A complete public API call looks like:

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

Both estimators provide `estimate()`, `quantile()`, and `summary()` after
fitting. These methods pool retained draws across all chains, matching the R
`bspcov` post-processing convention. Quantiles have shape `(n_probs, p, p)`;
`PosteriorSummary` contains the pooled mean, sample standard deviation,
requested quantiles, and fitted chain/sample counts. Returned statistics remain
JAX arrays on the fitted device.

`SBMSPCov` has the same input, dtype, device, key, and packed-sample contracts.
With `device=None`, JAX selects its default device; `device="cpu"` and
`device="gpu"` request those backends explicitly and fail clearly when the
requested backend is unavailable. Fitted arrays remain on the selected device.
Its default `cutoff_method="fnr"` uses `fnr_correlation=0.25`,
`false_negative_rate=0.05`, and `n_cutoff_simulations=1000`; the alternative
`cutoff_method="correlation"` uses `retained_fraction=0.2`. Screening runs once
per fit and all Python chains share `screening_mask_`. This intentionally differs
from `bspcov` 1.0.3, whose FNR path draws a separate screening cutoff for each
chain. `screening_cutoff_` is populated only for FNR screening, and
`diagnostics_` reports active and screened edge counts.

The checked-in R fixtures validate screening formulas, estimator orchestration,
and the public correlation-screened SBM posterior on the upstream `p=5`
example. This is targeted regression evidence, not a claim of universal
scientific equivalence or speedup.

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

## Design

The approved architecture, numerical contract, validation strategy, benchmark
methodology, repository policy, and initial success criteria are documented in
the [bootstrap and initial port design](docs/superpowers/specs/2026-08-01-pybspcov-bootstrap-design.md).

The [documentation layout](docs/README.md) keeps Sphinx sources separate from
internal design and plan records. The [development workflow](docs/development/workflow.md)
defines protected-branch, worktree, review, and parallel-integration rules.

Repository content is written in English. AI-assisted contributions will be
treated as unreviewed suggestions and will remain subject to human review,
independent scientific validation, provenance checks, and the contribution
policy established during project scaffolding.

## Maintainer

Kyeongwon Lee — <kwlee1718@gmail.com>

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
