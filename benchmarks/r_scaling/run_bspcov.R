#!/usr/bin/env Rscript

EXPECTED_BSPCOV_VERSION <- "1.0.3"

parse_arguments <- function(values) {
  parsed <- list()
  index <- 1L
  while (index <= length(values)) {
    option <- values[[index]]
    if (index == length(values) || !startsWith(option, "--")) {
      stop("options require a following value", call. = FALSE)
    }
    parsed[[substring(option, 3L)]] <- values[[index + 1L]]
    index <- index + 2L
  }
  required <- c(
    "fixture-dir", "fixture-sha256", "burnin", "n-samples", "n-chains", "seed"
  )
  if (!all(required %in% names(parsed))) {
    stop("missing required benchmark options", call. = FALSE)
  }
  parsed
}

read_matrix <- function(directory, filename) {
  as.matrix(utils::read.csv(
    file.path(directory, filename),
    header = FALSE,
    check.names = FALSE
  ))
}

arguments <- parse_arguments(commandArgs(trailingOnly = TRUE))
if (!requireNamespace("bspcov", quietly = TRUE)) {
  stop("run_bspcov.R requires bspcov", call. = FALSE)
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("run_bspcov.R requires jsonlite", call. = FALSE)
}
actual_version <- as.character(utils::packageVersion("bspcov"))
if (!identical(actual_version, EXPECTED_BSPCOV_VERSION)) {
  stop(
    sprintf("expected bspcov %s, found %s", EXPECTED_BSPCOV_VERSION, actual_version),
    call. = FALSE
  )
}

fixture_directory <- arguments[["fixture-dir"]]
x <- read_matrix(fixture_directory, "observations.csv")
truth <- read_matrix(fixture_directory, "truth_covariance.csv")
initial <- read_matrix(fixture_directory, "initial_covariance.csv")
burnin <- as.integer(arguments[["burnin"]])
n_samples <- as.integer(arguments[["n-samples"]])
chain_count <- as.integer(arguments[["n-chains"]])
seed <- as.integer(arguments[["seed"]])

fit_wall_seconds <- system.time({
  fit <- bspcov::bmspcov(
    X = x,
    Sigma = initial,
    nsample = list(burnin = burnin, nmc = n_samples),
    nchain = chain_count,
    seed = seed,
    do.parallel = TRUE,
    show_progress = FALSE
  )
})[["elapsed"]]

draws <- as.matrix(fit$Sigma)
expected_draws <- n_samples * chain_count
if (nrow(draws) != expected_draws) {
  stop(
    sprintf("expected %d retained draws, found %d", expected_draws, nrow(draws)),
    call. = FALSE
  )
}
dimension <- ncol(x)
posterior_mean <- matrix(0, nrow = dimension, ncol = dimension)
posterior_mean[!upper.tri(posterior_mean)] <- colMeans(draws)
posterior_mean[upper.tri(posterior_mean)] <- t(posterior_mean)[
  upper.tri(posterior_mean)
]
finite <- all(is.finite(posterior_mean))
symmetric <- isTRUE(all.equal(posterior_mean, t(posterior_mean)))
spd <- finite && symmetric && min(eigen(
  posterior_mean,
  symmetric = TRUE,
  only.values = TRUE
)$values) > 0
relative_error <- sqrt(sum((posterior_mean - truth)^2)) / sqrt(sum(truth^2))
normalized_seconds <- as.numeric(fit_wall_seconds) / chain_count

result <- list(
  implementation = "bspcov",
  package_version = actual_version,
  r_version = R.version.string,
  platform = R.version$platform,
  fixture_sha256 = arguments[["fixture-sha256"]],
  device = "cpu",
  dtype = "float64",
  execution_model = "parallel",
  burnin = burnin,
  n_samples = n_samples,
  chain_count = chain_count,
  retained_draws = nrow(draws),
  raw_wall_seconds = list(as.numeric(fit_wall_seconds)),
  total_wall_seconds = as.numeric(fit_wall_seconds),
  normalized_wall_seconds_per_chain = normalized_seconds,
  chains_per_second = 1 / normalized_seconds,
  posterior_mean_finite = finite,
  posterior_mean_symmetric = symmetric,
  posterior_mean_spd = spd,
  truth_relative_frobenius_error = as.numeric(relative_error)
)
cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = 17L), "\n", sep = "")
