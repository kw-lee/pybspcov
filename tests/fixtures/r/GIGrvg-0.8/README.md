# GIGrvg 0.8 GIG summary fixture

This fixture summarizes draws from `GIGrvg::rgig()` in CRAN `GIGrvg` 0.8,
the GIG sampler used by `bspcov`. The package source is available from the
[`GIGrvg` CRAN page](https://cran.r-project.org/package=GIGrvg).

The generator uses base R and `GIGrvg` only, with seed `20260801` and 200,000
draws per parameter regime. Run it from the repository root with:

```text
Rscript --vanilla reference/r/generate_gig_summary_fixture.R
```

`gig_summary.csv` records the parameter values, sample count, mean, standard
error of the mean, sample variance, and 10th, 50th, and 90th percentiles.
