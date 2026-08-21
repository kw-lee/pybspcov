required_version <- "1.0.3"
installed_version <- as.character(utils::packageVersion("bspcov"))
if (installed_version != required_version) {
  stop(sprintf("bspcov %s is required; found %s", required_version, installed_version))
}

fixture_dir <- Sys.getenv(
  "THRESHOLDPPP_FIXTURE_DIR",
  "tests/fixtures/r/bspcov-1.0.3"
)
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)
options(digits = 17)

X <- matrix(
  c(
    -1.5, -0.5, 0.5,
    -0.5, 1.5, -1.5,
    0.5, -1.5, 1.5,
    1.5, 0.5, -0.5
  ),
  nrow = 4,
  byrow = TRUE
)

RNGversion("4.5.0")
set.seed(20260820)
fit <- bspcov::thresPPP(
  X,
  eps = 0.05,
  thres = list(value = 0.4, fun = "hard"),
  nsample = 5000
)

draws <- fit$Sigma
n_batches <- 50L
batch_size <- nrow(draws) %/% n_batches
batch_means <- t(vapply(
  seq_len(n_batches),
  function(batch) {
    start <- (batch - 1L) * batch_size + 1L
    end <- batch * batch_size
    colMeans(draws[start:end, , drop = FALSE])
  },
  numeric(ncol(draws))
))
summary <- data.frame(
  packed_index = seq_len(ncol(draws)) - 1L,
  mean = colMeans(draws),
  standard_deviation = apply(draws, 2L, stats::sd),
  mean_mcse = apply(batch_means, 2L, stats::sd) / sqrt(n_batches)
)

write.table(
  X,
  file = file.path(fixture_dir, "thresholdppp_x.csv"),
  sep = ",",
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)
write.table(
  summary,
  file = file.path(fixture_dir, "thresholdppp_summary.csv"),
  sep = ",",
  row.names = FALSE,
  col.names = TRUE,
  quote = FALSE
)
writeLines(
  capture.output(sessionInfo()),
  file.path(fixture_dir, "thresholdppp_session_info.txt")
)
