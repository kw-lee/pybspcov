options(digits = 17, scipen = 999)

package_name <- "GIGrvg"
if (!requireNamespace(package_name, quietly = TRUE)) {
  stop("GIGrvg must be installed to generate this fixture")
}

package_version <- as.character(packageVersion(package_name))
output_dir <- file.path(
  "tests", "fixtures", "r", paste0(package_name, "-", package_version)
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

sample_count <- 200000L
set.seed(20260801L)

regimes <- data.frame(
  regime = c("ordinary", "small_omega"),
  lambda = c(-2, 0),
  chi = c(2, 0.01),
  psi = c(1, 1),
  stringsAsFactors = FALSE
)

summarize_regime <- function(regime, lambda, chi, psi) {
  draws <- GIGrvg::rgig(
    n = sample_count,
    lambda = lambda,
    chi = chi,
    psi = psi
  )
  quantiles <- quantile(draws, probs = c(0.1, 0.5, 0.9), names = FALSE)
  sample_variance <- var(draws)

  c(
    regime,
    sprintf("%.17g", lambda),
    sprintf("%.17g", chi),
    sprintf("%.17g", psi),
    as.character(sample_count),
    sprintf("%.17g", mean(draws)),
    sprintf("%.17g", sqrt(sample_variance / sample_count)),
    sprintf("%.17g", sample_variance),
    sprintf("%.17g", quantiles[[1L]]),
    sprintf("%.17g", quantiles[[2L]]),
    sprintf("%.17g", quantiles[[3L]])
  )
}

summary_rows <- mapply(
  summarize_regime,
  regimes$regime,
  regimes$lambda,
  regimes$chi,
  regimes$psi,
  SIMPLIFY = FALSE
)
summary_table <- do.call(rbind, summary_rows)
colnames(summary_table) <- c(
  "regime", "lambda", "chi", "psi", "sample_count", "mean",
  "standard_error", "variance", "q10", "q50", "q90"
)

write.table(
  summary_table,
  file = file.path(output_dir, "gig_summary.csv"),
  sep = ",",
  row.names = FALSE,
  col.names = TRUE,
  quote = FALSE
)
