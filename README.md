# pybspcov

`pybspcov` is a planned pure-Python, JAX-accelerated port of the R package
[`bspcov`](https://github.com/statjs/bspcov) for Bayesian sparse covariance
estimation.

> [!IMPORTANT]
> This repository is in its bootstrap phase. It does not yet contain an
> installable estimator implementation or a published package release.

Repository: <https://github.com/kw-lee/pybspcov>

## Initial scope

The first functional milestone will provide class-based Python implementations
of:

- `BMSPCov`, based on the beta-mixture shrinkage prior; and
- `SBMSPCov`, based on the screened beta-mixture shrinkage prior.

The implementation will use JAX and XLA for CPU and NVIDIA GPU execution. The
package itself will not contain custom C, C++, or CUDA extensions. It will use
64-bit floating point arithmetic by default and evaluate masked dense and sparse
GPU strategies with reproducible benchmarks.

The intended public API is:

```python
import jax
from pybspcov import BMSPCov

model = BMSPCov(n_samples=2_000, n_chains=4)
model.fit(X, key=jax.random.key(42))
posterior_mean = model.covariance_
```

This example documents the approved target API and is not executable in the
bootstrap commit.

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

- Lee, K., Jo, S., and Lee, J. (2022). The beta-mixture shrinkage prior for
  sparse covariances with near-minimax posterior convergence rate. *Journal of
  Multivariate Analysis*, 192, 105067.
  <https://doi.org/10.1016/j.jmva.2022.105067>
- Lee, K., Jo, S., Lee, K., and Lee, J. (2024). Scalable and optimal Bayesian
  inference for sparse covariance matrices via screened beta-mixture prior.
  *Bayesian Analysis*. <https://doi.org/10.1214/24-BA1495>

## License

`pybspcov` follows the upstream package under the GNU General Public License,
version 2 or any later version (`GPL-2.0-or-later`). See
[`LICENSE.md`](LICENSE.md) for the complete GPL v2 license text.
