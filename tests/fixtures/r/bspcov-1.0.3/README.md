# bspcov 1.0.3 covariance-column fixture

This fixture reproduces the deterministic covariance and precision updates in
CRAN `bspcov` 1.0.3. The upstream source is
[`R/bmspcov.R`](https://github.com/cran/bspcov/blob/1.0.3/R/bmspcov.R), from
tag `1.0.3` at commit `165106c5ab8f6506e6d69b0b8f94ce5bdc99092f`.

The generator follows `bmspcov()` line 225 for the Schur complement, lines
256-258 for the covariance update, and lines 274-281 for the rank-one
precision update. Run it from the repository root with:

```text
Rscript reference/r/generate_covariance_column_fixture.R
```

All CSV files have no headers or row names. Values are computed with base R
IEEE 754 double-precision arithmetic and written with 17 significant decimal
digits. `parameters.csv` stores the zero-based column followed by `gamma`;
`other_indices.csv` uses zero-based indices in the same order as `beta.csv`.

## SBM screening fixture

The `sbm_screening_*` files capture both screening branches exposed by
`sbmspcov()` in CRAN `bspcov` 1.0.3. The generator calls the exported
`sbmspcov()` function with a deterministic centered `12 x 4` design and one
minimal MCMC iteration, then checks its `INDzero` result against the internal
screening functions in
[`R/hidden.R`](https://github.com/cran/bspcov/blob/1.0.3/R/hidden.R).

The FNR fixture uses the public defaults `rho=0.25`, `FNR=0.05`, and
`nsimdata=1000`, with `seed=314159`. `select_cutoff()` simulates bivariate
normal data with correlation `rho`, computes
`exp(BayesFactor::correlationBF(..., rscale="ultrawide")@bayesFactor$bf)`, and
uses its `FNR` quantile as the cutoff. `BayesCGM.SS()` retains an edge only
when its pairwise Bayes factor is strictly greater than that cutoff. With
RNGversion `4.5.0`, `seed=314159` deterministically produces the internal
`select_cutoff()` seed `910796002`; the generator observes that draw before
calling `sbmspcov()`, whose public `seed` argument resets the RNG and therefore
preserves the upstream random path.

The fixture was generated with R 4.5.0, RNGkind `Mersenne-Twister`, `Inversion`,
and `Rejection`, BayesFactor 0.9.12.4.7, MASS 7.3.65, and IEEE 754 double
precision. Full platform, BLAS, locale, and loaded-package details are in
`sbm_screening_session_info.txt`.

The correlation fixture selects `cutoff=list(method="corr")`, whose actual
implementation default is `thr=0.2`. `BayesCGM.SS.CORR()` takes the
`1-thr` quantile of the absolute lower-triangular sample correlations and
retains an edge only when its absolute correlation is strictly greater than
that cutoff.

Mask names encode their semantics explicitly:

- `*_excluded_mask.csv` is `1` for an edge listed in upstream `INDzero` and
  therefore zeroed before sampling.
- `*_active_mask.csv` is `1` for a retained off-diagonal edge and has a zero
  diagonal. It is the off-diagonal complement of the excluded mask.

The input, raw lower-triangular Bayes factors, full correlation matrix,
initial covariance, screened covariances, and generation parameters are also
committed. R's `NA` spelling is preserved in the unused diagonal and upper
triangle of `sbm_screening_pairwise_bf.csv`. All indices described by the
metadata are zero-based for Python consumers. Tests pin the literal FNR and
correlation cutoffs plus every lower-triangular Bayes factor and correlation,
so dependency or RNG drift fails instead of silently rewriting the reference.

Regenerate from the repository root with exactly `bspcov` 1.0.3 installed:

```text
Rscript reference/r/generate_sbm_screening_fixture.R
```

There is an upstream API/documentation mismatch in 1.0.3: the help page says
the correlation threshold field is `rho`, but `sbmspcov()` reads `thr`.
Passing `rho` to the correlation branch does not override `thr`; Python parity
must follow the implemented `thr` behavior documented by this fixture.
