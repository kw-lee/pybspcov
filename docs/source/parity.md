# R `bspcov` 1.0.3 feature map

pybspcov exposes every documented user-facing function and bundled dataset
from R `bspcov` 1.0.3 through an idiomatic Python API.

| R API | Python API |
|---|---|
| `bmspcov` | `BMSPCov` |
| `sbmspcov` | `SBMSPCov` |
| `bandPPP` | `BandPPP` |
| `thresPPP` | `ThresholdPPP` |
| `cv.bandPPP` | `cross_validate_band_ppp` |
| `cv.thresPPP` | `cross_validate_threshold_ppp` |
| `estimate` | estimator `estimate()` |
| `summary.bspcov` | estimator `summary()` or ArviZ `summary` |
| `quantile.bspcov` | estimator `quantile()` |
| `plot.bspcov` | `plot_trace` and `plot_cv` |
| `plot.postmean.bspcov` | `plot_posterior_mean` |
| `plot.quantile.bspcov` | `plot_quantiles` |
| `save_quantile_plot` | `save_quantile_plot` |
| `proc_colon` | `preprocess_colon` |
| `proc_SP500` | `preprocess_sp500` |
| `colon`, `tissues` | `load_colon` |
| `SP500` | `load_sp500` |

R and JAX use different random-number generators, so stochastic draws are not
expected to match one-for-one. Versioned R fixtures compare deterministic
transforms exactly and posterior summaries with Monte Carlo uncertainty.

The SP500 preprocessing function implements the complete monthly-return,
factor-reconstruction, and residual-output workflow without R runtime
dependencies. Its automatic factor-count selector is a documented spectral
ratio implementation rather than a byte-for-byte port of `hdbinseg`'s `ah`
criterion; pass `n_factors` to make that choice explicit and reproducible.
