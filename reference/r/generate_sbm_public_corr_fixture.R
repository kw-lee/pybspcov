#!/usr/bin/env Rscript

options(digits = 17, scipen = 999)

expected_version <- "1.0.3"
rng_version <- "4.5.0"
RNGversion(rng_version)

required_packages <- c("bspcov", "digest", "jsonlite", "openssl")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1L), quietly = TRUE)
]
if (length(missing_packages) > 0L) {
  stop(
    sprintf("Missing required R packages: %s", paste(missing_packages, collapse = ", ")),
    call. = FALSE
  )
}

actual_version <- as.character(utils::packageVersion("bspcov"))
if (!identical(actual_version, expected_version)) {
  stop(
    sprintf("Expected bspcov %s, but found %s", expected_version, actual_version),
    call. = FALSE
  )
}

data_dir <- file.path("benchmarks", "r_example", "data")
output_dir <- file.path("tests", "fixtures", "r", "bspcov-1.0.3")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

fixture_names <- c(
  "bm_example_x.csv",
  "bm_example_truth.csv",
  "bm_example_initial.csv"
)
fixture_paths <- file.path(data_dir, fixture_names)
if (any(!file.exists(fixture_paths))) {
  stop("The committed bspcov example fixture is incomplete", call. = FALSE)
}

read_matrix <- function(filename) {
  as.matrix(utils::read.csv(
    file.path(data_dir, filename),
    header = FALSE,
    check.names = FALSE
  ))
}

read_raw_file <- function(path) {
  connection <- file(path, open = "rb")
  on.exit(close(connection))
  readBin(connection, what = "raw", n = as.integer(file.info(path)$size))
}

fixture_payload <- raw()
for (index in seq_along(fixture_paths)) {
  fixture_payload <- c(
    fixture_payload,
    charToRaw(fixture_names[[index]]),
    as.raw(0L),
    read_raw_file(fixture_paths[[index]])
  )
}
fixture_sha256 <- unclass(as.character(openssl::sha256(fixture_payload)))
initial_fixture_sha256 <- digest::digest(
  fixture_paths[[3L]],
  algo = "sha256",
  file = TRUE,
  serialize = FALSE
)

x <- read_matrix("bm_example_x.csv")
truth <- read_matrix("bm_example_truth.csv")
initial_covariance <- read_matrix("bm_example_initial.csv")
n <- nrow(x)
p <- ncol(x)
burnin <- 1000L
n_samples <- 1000L
sampler_seed <- 1L
retained_fraction <- 0.2

fit <- bspcov::sbmspcov(
  X = x,
  Sigma = initial_covariance,
  cutoff = list(method = "corr", thr = retained_fraction),
  nsample = list(burnin = burnin, nmc = n_samples),
  nchain = 1L,
  seed = sampler_seed,
  do.parallel = FALSE,
  show_progress = FALSE
)

draws <- fit$Sigma
parameter_count <- as.integer(p * (p + 1L) / 2L)
if (
  !is.matrix(draws) ||
    !identical(dim(draws), c(n_samples, parameter_count)) ||
    any(!is.finite(draws))
) {
  stop("bspcov::sbmspcov returned invalid covariance draws", call. = FALSE)
}

excluded_mask <- matrix(FALSE, nrow = p, ncol = p)
for (column in seq_len(p)) {
  if (length(fit$INDzero[[column]]) > 0L) {
    excluded_mask[fit$INDzero[[column]], column] <- TRUE
  }
}
active_mask <- (!excluded_mask) & !diag(p)
stopifnot(identical(active_mask, t(active_mask)))

correlations <- stats::cor(x)
correlation_lower <- correlations
correlation_lower[upper.tri(correlation_lower, diag = TRUE)] <- NA_real_
screening_cutoff <- as.numeric(stats::quantile(
  abs(correlation_lower),
  prob = 1 - retained_fraction,
  na.rm = TRUE,
  type = 7L
))
reconstructed_support <- abs(correlation_lower) > screening_cutoff
reconstructed_support[upper.tri(reconstructed_support)] <-
  t(reconstructed_support)[upper.tri(reconstructed_support)]
stopifnot(all(active_mask == (reconstructed_support & !diag(p))))

quantile_probabilities <- c(0.025, 0.5, 0.975)
n_batches <- min(20L, floor(n_samples / 2L))
batch_size <- floor(n_samples / n_batches)
trimmed_samples <- n_batches * batch_size
batch_rows <- split(
  seq_len(trimmed_samples),
  rep(seq_len(n_batches), each = batch_size)
)

batch_means <- vapply(
  batch_rows,
  function(rows) colMeans(draws[rows, , drop = FALSE]),
  numeric(parameter_count)
)
batch_sds <- vapply(
  batch_rows,
  function(rows) apply(draws[rows, , drop = FALSE], 2L, stats::sd),
  numeric(parameter_count)
)
batch_quantiles <- lapply(
  quantile_probabilities,
  function(probability) vapply(
    batch_rows,
    function(rows) apply(
      draws[rows, , drop = FALSE],
      2L,
      stats::quantile,
      probs = probability,
      names = FALSE,
      type = 7L
    ),
    numeric(parameter_count)
  )
)
batch_mcse <- function(values) {
  apply(values, 1L, stats::sd) / sqrt(n_batches)
}

vech_to_symmetric <- function(values) {
  result <- matrix(0, nrow = p, ncol = p)
  result[lower.tri(result, diag = TRUE)] <- values
  result[upper.tri(result)] <- t(result)[upper.tri(result)]
  result
}

posterior_mean <- vech_to_symmetric(colMeans(draws))
posterior_sd <- vech_to_symmetric(apply(draws, 2L, stats::sd))
posterior_quantiles <- apply(
  draws,
  2L,
  stats::quantile,
  probs = quantile_probabilities,
  names = FALSE,
  type = 7L
)
q025 <- vech_to_symmetric(posterior_quantiles[1L, ])
q50 <- vech_to_symmetric(posterior_quantiles[2L, ])
q975 <- vech_to_symmetric(posterior_quantiles[3L, ])
posterior_mean_mcse <- vech_to_symmetric(batch_mcse(batch_means))
posterior_sd_mcse <- vech_to_symmetric(batch_mcse(batch_sds))
q025_mcse <- vech_to_symmetric(batch_mcse(batch_quantiles[[1L]]))
q50_mcse <- vech_to_symmetric(batch_mcse(batch_quantiles[[2L]]))
q975_mcse <- vech_to_symmetric(batch_mcse(batch_quantiles[[3L]]))
rmse <- sqrt(mean((posterior_mean - truth)^2))
batch_rmses <- vapply(
  seq_len(n_batches),
  function(batch) {
    sqrt(mean((vech_to_symmetric(batch_means[, batch]) - truth)^2))
  },
  numeric(1L)
)
rmse_mcse <- stats::sd(batch_rmses) / sqrt(n_batches)

summary_output <- data.frame(
  implementation = "bspcov",
  row = rep(seq_len(p), times = p),
  column = rep(seq_len(p), each = p),
  posterior_mean = as.vector(posterior_mean),
  posterior_mean_mcse = as.vector(posterior_mean_mcse),
  posterior_sd = as.vector(posterior_sd),
  posterior_sd_mcse = as.vector(posterior_sd_mcse),
  q025 = as.vector(q025),
  q025_mcse = as.vector(q025_mcse),
  q50 = as.vector(q50),
  q50_mcse = as.vector(q50_mcse),
  q975 = as.vector(q975),
  q975_mcse = as.vector(q975_mcse),
  truth = as.vector(truth),
  rmse = rmse,
  rmse_mcse = rmse_mcse,
  stringsAsFactors = FALSE
)
utils::write.csv(
  summary_output,
  file.path(output_dir, "sbm_public_corr_summary.csv"),
  row.names = FALSE
)

software_versions <- tryCatch(extSoftVersion(), error = function(error) NULL)
blas <- if (is.null(software_versions)) {
  "unavailable"
} else {
  unname(software_versions[["BLAS"]])
}
lapack <- tryCatch(La_library(), error = function(error) "unavailable")
metadata <- list(
  fixture_schema_version = 1L,
  implementation = "bspcov",
  package = "bspcov",
  package_version = actual_version,
  runtime = "R",
  runtime_version = as.character(getRversion()),
  r_platform = R.version$platform,
  blas = blas,
  lapack = lapack,
  session_info = paste(capture.output(utils::sessionInfo()), collapse = "\n"),
  model = "sbm",
  sampler_variant = "bspcov_sbm",
  estimator_api = "bspcov::sbmspcov",
  dtype = "float64",
  source_command = paste(
    "Rscript --vanilla",
    "reference/r/generate_sbm_public_corr_fixture.R"
  ),
  fixture_sha256 = fixture_sha256,
  initial_fixture_sha256 = initial_fixture_sha256,
  n = n,
  p = p,
  burnin = burnin,
  n_samples = n_samples,
  chains = 1L,
  sampler_seed = sampler_seed,
  screening_method = "corr",
  screening_retained_fraction = retained_fraction,
  screening_cutoff = screening_cutoff,
  screening_active_lower = paste(
    as.integer(active_mask[lower.tri(active_mask)]),
    collapse = ""
  ),
  screening_active_edges = sum(active_mask[lower.tri(active_mask)]),
  prior_a = 0.5,
  prior_b = 0.5,
  diagonal_rate = 1.0,
  tau1sq = log(p) / (p^2 * n),
  n_batches = n_batches,
  batch_size = batch_size,
  trimmed_samples = trimmed_samples
)
jsonlite::write_json(
  metadata,
  file.path(output_dir, "sbm_public_corr_metadata.json"),
  auto_unbox = TRUE,
  pretty = TRUE,
  digits = 17
)
