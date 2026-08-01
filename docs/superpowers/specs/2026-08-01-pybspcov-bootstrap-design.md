# pybspcov Bootstrap and Initial Port Design

**Status:** Approved design
**Date:** 2026-08-01
**Upstream:** [`statjs/bspcov`](https://github.com/statjs/bspcov), version 1.0.3 at the time of design

## 1. Purpose

`pybspcov` will be the pure-Python, JAX-accelerated port of the R package
`bspcov`. The first functional milestone will port the beta-mixture shrinkage
prior (`bmspcov`) and screened beta-mixture shrinkage prior (`sbmspcov`) for
Bayesian sparse covariance estimation.

The package itself will contain no custom C, C++, or CUDA extensions. JAX and
XLA will provide compilation and accelerator access. The project will preserve
the statistical meaning and default prior definitions of the R package while
offering an idiomatic class-based Python API.

## 2. Goals

- Provide `BMSPCov` and `SBMSPCov` estimators with a consistent Python API.
- Run the MCMC kernels on JAX-supported CPU and NVIDIA GPU devices.
- Use 64-bit floating point arithmetic by default for statistical fidelity and
  numerical stability, with 32-bit arithmetic exposed as experimental.
- Compile the sampling loop with JAX and parallelize independent chains with
  JAX transformations.
- Establish statistical parity with the R implementation using reproducible
  reference fixtures and Monte Carlo-aware comparisons.
- Measure whether masked dense or sparse representations are faster for the
  actual algorithm before making sparse storage part of the primary path.
- Publish the project on GitHub under GPL-2 with clear attribution, contribution
  guidance, scientific validation, and secure automation practices.

## 3. Non-goals for the Initial Milestone

- Porting `bandPPP`, `thresPPP`, cross-validation helpers, plotting, or bundled
  R datasets.
- Matching R and JAX random-number streams sample by sample.
- Adding NumPy, CuPy, PyTorch, or native CUDA implementations as alternative
  compute backends.
- Depending on scikit-learn solely to provide estimator conventions.
- Promising that a sparse representation is faster before representative
  benchmarks demonstrate it.
- Publishing to PyPI as part of the initial bootstrap commit.

## 4. Naming, Language, and Licensing

- Repository name: `pybspcov`
- GitHub repository: `https://github.com/kw-lee/pybspcov`
- Distribution name: `pybspcov`
- Import package: `pybspcov`
- Public estimators: `BMSPCov` and `SBMSPCov`
- Maintainer: Kyeongwon Lee (`kwlee1718@gmail.com`)
- License: GPL-2.0-or-later, inherited from the upstream R package

All repository content will be written in English, including code identifiers,
comments, docstrings, error messages, documentation, issue templates, and
commit messages. Maintainer conversations may use other languages.

The README, package metadata, and source headers where appropriate will credit
the upstream package, authors, and related publications. Reference outputs
derived from the R implementation will record the upstream version and the
script used to produce them.

## 5. Public API

The estimators will use a scikit-learn-like interface without taking a runtime
dependency on scikit-learn:

```python
import jax
from pybspcov import BMSPCov, SBMSPCov

bm = BMSPCov(n_samples=2_000, n_chains=4)
bm.fit(X, key=jax.random.key(42))

posterior_mean = bm.covariance_
samples = bm.posterior_samples_
diagnostics = bm.diagnostics_

sbm = SBMSPCov(n_samples=2_000, n_chains=4, cutoff_method="fnr")
sbm.fit(X, key=jax.random.key(42))
screening_mask = sbm.screening_mask_
```

`fit` will return `self`. Before fitting, attributes whose names end in `_` will
not exist. The public classes will manage input validation, compilation,
execution, and fitted state. They will not be passed through `jax.jit`.

The required random key makes randomness explicit and prevents hidden use of a
global random state. An optional initial covariance may be supplied to `fit`.
When it is absent, the estimator will construct and validate a stabilized sample
covariance. Device arrays will remain on their JAX device unless the user
explicitly asks to convert or serialize them.

## 6. Architecture

The implementation will separate stateful Python orchestration from pure JAX
kernels:

```text
User
  -> BMSPCov / SBMSPCov
       -> host-side validation and configuration
       -> device selection, compilation, and fitted state
       -> pure JAX kernels
            -> common covariance updates
            -> BM update policy
            -> SBM masked update policy
            -> random samplers
```

The initial package boundaries will be:

- `estimators`: public classes and fitted-state management.
- `kernels`: immutable sampler state and pure BM, SBM, and shared update
  functions.
- `sampling`: PRNG-key management and JIT-compatible distribution samplers.
- `diagnostics`: convergence summaries, numerical-repair counts, and runtime
  metadata.
- `validation`: input, configuration, dtype, device, and covariance checks.
- `reference`: scripts and metadata for generating R parity fixtures.
- `benchmarks`: reproducible CPU, GPU, dense, masked, and sparse comparisons.

Kernel state will be represented by immutable JAX PyTrees. Host-level result
objects may use Python dataclasses, but the estimators remain the only primary
public interface in the first milestone.

## 7. JAX Execution Model

The execution path will be:

1. Validate `X`, prior settings, sampling settings, device requirements, and an
   optional initial covariance on the host.
2. Convert inputs to the requested JAX dtype, using `float64` by default.
3. For `SBMSPCov`, run FNR or correlation screening once before MCMC and create
   a fixed-shape boolean mask.
4. Split the user-provided PRNG key into independent chain keys.
5. Use `jax.vmap` for chains, `jax.lax.scan` for MCMC iterations, and
   `jax.lax.fori_loop` for within-iteration column updates.
6. Save only the lower-triangular covariance and prior elements after burn-in.
7. Expose posterior summaries, diagnostics, the screening mask, and execution
   metadata as fitted attributes.

The compiled program will operate on static shapes. Variable-length SBM active
sets will be represented with full-size arrays and boolean masks, avoiding
Python lists and data-dependent array shapes inside JIT-compiled functions.
Compilation cache keys will naturally include shape, dtype, device, and static
sampler configuration.

The original default statistical definitions will be preserved:

- BM: `a = 1/2`, `b = 1/2`, `lambda = 1`, and
  `tau1sq = 10^4 / (n * p^4)`.
- SBM: `a = 1/2`, `b = 1/2`, `lambda = 1`, and
  `tau1sq = log(p) / (p^2 * n)`.
- Both: `burnin = 1000` and `nmc = 1000` unless explicitly changed.
- SBM: FNR screening is the default; correlation screening remains available.

## 8. Sparse Acceleration Strategy

The initial production kernel will use JAX-native masked dense operations. This
choice avoids making the primary implementation depend on
`jax.experimental.sparse`, whose high-level implementations are experimental
and not recommended by JAX for performance-critical use.

Sparse acceleration will be an internal strategy, not a different public API.
Benchmarks will determine whether individual operations should use BCOO, BCSR,
sparse primitives, or a later pure-Python Pallas kernel. A sparse path will be
accepted only when it:

- preserves the statistical and numerical contracts,
- improves steady-state runtime or memory on declared problem regimes,
- includes conversion and recompilation costs in the comparison, and
- retains a tested masked dense fallback.

This design leaves room for modern GPU sparse capabilities without assuming
that sparse storage improves the screened Gibbs updates, which repeatedly use
reduced dense factorizations.

## 9. GIG Sampling

The generalized inverse Gaussian sampler is a central porting risk because the
R implementation uses `GIGrvg::rgig`. `pybspcov` will implement a bounded,
JIT-compatible GIG sampler in pure JAX rather than call R, SciPy during the
compiled loop, or a custom native extension.

The sampler will have its own statistical validation suite. Tests will compare
moments, quantiles, support, and difficult parameter regimes against independent
SciPy or R reference outputs. Rejection loops will have explicit finite bounds
and will return a status that the host can convert into an informative error.

## 10. Numerical and Error-Handling Contract

- `X` must be a finite two-dimensional array with at least two columns.
- An explicitly supplied initial covariance must have shape `(p, p)`, be
  symmetric within a documented tolerance, and be positive definite.
- If `dtype="float64"` is requested while JAX X64 mode is disabled, fitting will
  fail with instructions for enabling X64 rather than silently truncate.
- If a GPU is explicitly requested but unavailable, fitting will fail rather
  than silently fall back to CPU.
- Cholesky inputs will be symmetrized. On failure, a bounded adaptive-jitter
  policy based on the minimum eigenvalue may be applied.
- Every jitter event will be counted in diagnostics. Exhausting the repair
  policy will produce an error with chain, iteration, and column context.
- Non-finite sampler state, invalid GIG output, and sampler rejection-limit
  exhaustion will be detected explicitly.
- JIT kernels will return structured status data where Python exceptions cannot
  be raised safely; the estimator will translate that status into clear public
  exceptions.

The default dtype is `float64`. `float32` will be labeled experimental and must
pass separate stability tests before performance results are presented.

## 11. Testing and Statistical Parity

Testing will have five layers:

1. **Unit tests:** GIG sampling, lower-triangle packing, masks, prior defaults,
   PRNG splitting, and validation.
2. **Numerical invariants:** covariance symmetry and positive definiteness,
   preservation of screened zeros, finite state, and repeatability for the same
   device and key.
3. **R statistical parity:** posterior means, variances, and quantiles on small
   fixed datasets will be compared with versioned R fixtures using tolerances
   that account for Monte Carlo error.
4. **CPU/GPU parity:** posterior summaries and diagnostic statistics will be
   compared statistically rather than bitwise.
5. **Performance benchmarks:** R, JAX CPU, and JAX GPU will be measured on a
   declared grid of `n`, `p`, sparsity, and chain count.

Tests will not require identical R and JAX sample paths. CPU tests will run in
ordinary CI. GPU tests and benchmarks will use separate markers and workflows
so contributors without CUDA can run the complete correctness suite that is
applicable to CPU.

Independent references are essential: generated implementation tests must not
merely restate the implementation. R fixtures will be generated by committed
scripts, mathematical identities will be checked directly, and stochastic
tolerances will be justified in test documentation.

## 12. Benchmark Contract

Each benchmark report will separate:

- compilation time,
- steady-state sampling time after compilation,
- end-to-end time,
- peak device memory, and
- effective samples per second.

Reports will record software versions, dtype, device model, problem dimensions,
screening density, warm-up procedure, and commands. Raw benchmark results will
be retained separately from curated documentation. Initial CI will record
performance without failing on unstable absolute wall-clock thresholds; stable
regression budgets may be added after representative baselines exist.

## 13. Repository Bootstrap

The repository will use Python 3.12 or newer, `uv` for development environment
and lock-file management, `hatchling` as the build backend, and a `src` layout.
The bootstrap will include:

```text
pyproject.toml
README.md
.gitignore
LICENSE.md
CITATION.cff
CHANGELOG.md
CONTRIBUTING.md
SECURITY.md
AGENTS.md
src/pybspcov/
tests/
benchmarks/
reference/r/
docs/README.md
docs/source/
docs/development/
docs/superpowers/specs/
docs/superpowers/plans/
.github/workflows/
.github/dependabot.yml
.github/pull_request_template.md
```

Development tooling will include pytest, Ruff, static type checking, build
validation, and pre-commit hooks. CUDA installation will be optional and
documented separately from the CPU-compatible base installation because JAX
wheels depend on the CUDA environment.

### Documentation separation

Sphinx will use `docs/source/` as its explicit source directory and
`docs/_build/` as generated output. Design records and implementation plans
remain under `docs/superpowers/` and are not placed in the Sphinx toctree.
Maintainer guidance lives under `docs/development/`. Executable examples live
under the repository-level `examples/` tree and may be rendered into Sphinx only
after their public API is stable and their CPU smoke tests pass.

### Branches, worktrees, and parallel development

After the bootstrap commit, `main` is integration-only. Each change uses a
focused topic branch and a separate worktree under the ignored `.worktrees/`
directory unless a platform-native worktree facility is available. Parallel
tracks must own disjoint files and expose explicit interfaces. Tests travel with
their implementation; R reference fixtures, Sphinx infrastructure, benchmark
infrastructure, and independent parity testing may proceed in parallel when
their inputs are stable. Shared files such as `pyproject.toml`, `uv.lock`, public
exports, common state types, and CI workflows have one active owner per wave.
Performance benchmarks require exclusive GPU use.

## 14. AI-Assisted Development Policy

LLM output will be treated as an unreviewed suggestion. Human contributors are
responsible for correctness, provenance, licensing, security, and scientific
claims in every contribution.

The English `CONTRIBUTING.md`, `AGENTS.md`, and
`docs/development/ai-assisted-development.md` will establish these rules:

- Disclose material AI assistance in a pull request without requiring raw chat
  logs or prompts to be committed.
- Do not submit private source code, credentials, patient data, unpublished
  research data, or other restricted information to an external model.
- Verify generated code and documentation against primary sources, the papers,
  and executable tests.
- Check the provenance and license of generated or suggested code. Preserve
  GPL-2.0-or-later and upstream attribution for derived work.
- Do not accept tests generated solely from the same implementation logic as
  evidence of scientific correctness.
- Check JAX APIs and installation commands against current official
  documentation; do not rely on model memory for changing dependencies.
- Report complete benchmark context and do not select only favorable runs.
- Treat issue, pull-request, and dependency text consumed by automation as
  untrusted input that may contain prompt injection.
- Do not grant code-generation or issue-triage automation release credentials
  or unnecessary repository write permissions.

## 15. Public GitHub and Supply-Chain Controls

The public repository is `https://github.com/kw-lee/pybspcov`. An active ruleset
will protect `main` by requiring a pull request, one independent approval,
resolved conversations, an up-to-date branch, and the `quality`, `tests`,
`docs`, and `build` checks. The ruleset will require linear history and block
force pushes and deletion. Normal changes use squash merge. The bootstrap root
commit is the only direct-main exception; no automation receives routine bypass
permission.

- GitHub Actions will declare least-privilege permissions, with ordinary CI
  using `contents: read`.
- Third-party Actions will be pinned to verified full commit SHAs.
- Dependabot will monitor Python dependencies and GitHub Actions.
- Local secrets, `.env` files, caches, build products, and uncurated local
  benchmark output will be ignored.
- Secret scanning and push protection will be enabled before normal public
  development begins.
- A future PyPI release workflow will use OIDC Trusted Publishing and a protected
  release environment instead of a long-lived PyPI API token.
- Pull requests will require human review and passing correctness checks; AI or
  dependency-bot pull requests will not auto-merge by default.

## 16. Initial Success Criteria

The initial BM/SBM port milestone is complete when:

- both public estimators run on JAX CPU and at least one supported NVIDIA GPU;
- the default float64 path passes unit, invariant, and R statistical parity
  tests;
- multi-chain sampling uses explicit independent keys and JAX batching;
- SBM screening masks remain valid throughout sampling;
- errors do not silently change device, precision, or statistical behavior;
- benchmark reports separate compilation from steady-state execution; and
- the English public documentation explains installation, provenance,
  limitations, reproducibility, and AI-assisted contribution expectations.

## 17. References

- Upstream R package: <https://github.com/statjs/bspcov>
- CRAN mirror: <https://github.com/cran/bspcov>
- JAX sparse documentation: <https://docs.jax.dev/en/latest/jax.experimental.sparse.html>
- JAX X64 documentation: <https://docs.jax.dev/en/latest/default_dtypes.html>
- GitHub Actions secure-use guidance: <https://docs.github.com/en/actions/reference/security/secure-use>
- GitHub secret scanning: <https://docs.github.com/en/code-security/concepts/secret-security/secret-scanning>
- GitHub OIDC publishing to PyPI: <https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-pypi>
