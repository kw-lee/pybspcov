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
