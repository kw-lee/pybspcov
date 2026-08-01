#!/usr/bin/env Rscript

BASELINE_BSPCOV_VERSION <- "1.0.3"
FIXTURE_SEED <- 1L
SAMPLER_SEED <- 1L
QUANTILE_PROBABILITIES <- c(0.025, 0.5, 0.975)

usage <- function() {
  paste(
    "usage: Rscript run_bspcov.R",
    "[--burnin INTEGER] [--n-samples INTEGER]",
    "[--output-dir PATH] [--allow-version-mismatch]"
  )
}

parse_positive_integer <- function(value, option_name, minimum = 1L) {
  parsed <- suppressWarnings(as.integer(value))
  if (is.na(parsed) || as.character(parsed) != value || parsed < minimum) {
    stop(
      sprintf("%s must be an integer greater than or equal to %d", option_name, minimum),
      call. = FALSE
    )
  }
  parsed
}

parse_arguments <- function(arguments, default_output_directory) {
  parsed <- list(
    burnin = 1000L,
    n_samples = 1000L,
    output_directory = default_output_directory,
    allow_version_mismatch = FALSE
  )
  position <- 1L
  while (position <= length(arguments)) {
    option <- arguments[[position]]
    if (option == "--allow-version-mismatch") {
      parsed$allow_version_mismatch <- TRUE
      position <- position + 1L
      next
    }
    if (option %in% c("--burnin", "--n-samples", "--output-dir")) {
      if (position == length(arguments)) {
        stop(sprintf("%s requires a value\n%s", option, usage()), call. = FALSE)
      }
      value <- arguments[[position + 1L]]
      if (option == "--burnin") {
        parsed$burnin <- parse_positive_integer(value, option, minimum = 0L)
      } else if (option == "--n-samples") {
        parsed$n_samples <- parse_positive_integer(value, option, minimum = 2L)
      } else {
        parsed$output_directory <- value
      }
      position <- position + 2L
      next
    }
    stop(sprintf("unknown option: %s\n%s", option, usage()), call. = FALSE)
  }
  parsed
}

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_argument) == 1L) {
  normalizePath(sub("^--file=", "", script_argument), mustWork = TRUE)
} else {
  normalizePath("benchmarks/r_example/run_bspcov.R", mustWork = TRUE)
}
script_directory <- dirname(script_path)
arguments <- parse_arguments(
  commandArgs(trailingOnly = TRUE),
  file.path(script_directory, "results")
)

if (!requireNamespace("bspcov", quietly = TRUE)) {
  stop("run_bspcov.R requires the R package bspcov", call. = FALSE)
}
actual_bspcov_version <- as.character(utils::packageVersion("bspcov"))
if (
  actual_bspcov_version != BASELINE_BSPCOV_VERSION &&
    !arguments$allow_version_mismatch
) {
  stop(
    sprintf(
      paste(
        "run_bspcov.R requires bspcov %s, but found %s;",
        "pass --allow-version-mismatch only for a deliberate non-baseline run"
      ),
      BASELINE_BSPCOV_VERSION,
      actual_bspcov_version
    ),
    call. = FALSE
  )
}

fixture_directory <- file.path(script_directory, "data")
read_matrix <- function(filename) {
  as.matrix(utils::read.csv(
    file.path(fixture_directory, filename),
    header = FALSE,
    check.names = FALSE
  ))
}
x <- read_matrix("bm_example_x.csv")
true_sigma <- read_matrix("bm_example_truth.csv")
initial_sigma <- read_matrix("bm_example_initial.csv")
n <- nrow(x)
p <- ncol(x)

# X is the explicitly centered, committed output of generate_case.R. Do not
# center it again here: the future JAX runner must consume these exact values.
elapsed <- system.time({
  fit <- bspcov::bmspcov(
    X = x,
    Sigma = initial_sigma,
    nsample = list(burnin = arguments$burnin, nmc = arguments$n_samples),
    nchain = 1L,
    seed = SAMPLER_SEED,
    do.parallel = FALSE,
    show_progress = FALSE
  )
})[["elapsed"]]

draws <- fit$Sigma
posterior_mean_vector <- colMeans(draws)
posterior_sd_vector <- apply(draws, MARGIN = 2L, FUN = stats::sd)
posterior_quantiles <- apply(
  draws,
  MARGIN = 2L,
  FUN = stats::quantile,
  probs = QUANTILE_PROBABILITIES,
  names = FALSE,
  type = 7L
)

vech_to_symmetric <- function(value, dimension) {
  result <- matrix(0, nrow = dimension, ncol = dimension)
  result[lower.tri(result, diag = TRUE)] <- value
  result[upper.tri(result)] <- t(result)[upper.tri(result)]
  result
}
posterior_mean <- vech_to_symmetric(posterior_mean_vector, p)
posterior_sd <- vech_to_symmetric(posterior_sd_vector, p)
q025 <- vech_to_symmetric(posterior_quantiles[1L, ], p)
q50 <- vech_to_symmetric(posterior_quantiles[2L, ], p)
q975 <- vech_to_symmetric(posterior_quantiles[3L, ], p)
rmse <- sqrt(mean((posterior_mean - true_sigma)^2))

row_indices <- rep(seq_len(p), times = p)
column_indices <- rep(seq_len(p), each = p)
summary_output <- data.frame(
  implementation = "bspcov",
  row = row_indices,
  column = column_indices,
  posterior_mean = as.vector(posterior_mean),
  posterior_sd = as.vector(posterior_sd),
  q025 = as.vector(q025),
  q50 = as.vector(q50),
  q975 = as.vector(q975),
  truth = as.vector(true_sigma),
  rmse = rmse,
  stringsAsFactors = FALSE
)

timing_output <- data.frame(
  implementation = "bspcov",
  end_to_end_seconds = as.numeric(elapsed),
  stringsAsFactors = FALSE
)

session <- utils::sessionInfo()
metadata_output <- data.frame(
  name = c(
    "implementation",
    "package",
    "package_version",
    "baseline_package_version",
    "r_version",
    "platform",
    "dtype",
    "device",
    "n",
    "p",
    "burnin",
    "n_samples",
    "chains",
    "repetitions",
    "fixture_seed",
    "sampler_seed",
    "fixture_centered",
    "session_info"
  ),
  value = c(
    "bspcov",
    "bspcov",
    actual_bspcov_version,
    BASELINE_BSPCOV_VERSION,
    R.version.string,
    R.version$platform,
    "float64",
    "CPU",
    as.character(n),
    as.character(p),
    as.character(arguments$burnin),
    as.character(arguments$n_samples),
    "1",
    "1",
    as.character(FIXTURE_SEED),
    as.character(SAMPLER_SEED),
    "true",
    paste(utils::capture.output(print(session)), collapse = "\n")
  ),
  stringsAsFactors = FALSE
)

dir.create(arguments$output_directory, recursive = TRUE, showWarnings = FALSE)
utils::write.csv(
  summary_output,
  file.path(arguments$output_directory, "r_summary.csv"),
  row.names = FALSE
)
utils::write.csv(
  timing_output,
  file.path(arguments$output_directory, "r_timing.csv"),
  row.names = FALSE
)
utils::write.csv(
  metadata_output,
  file.path(arguments$output_directory, "r_metadata.csv"),
  row.names = FALSE
)
