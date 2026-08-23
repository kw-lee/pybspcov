#!/usr/bin/env Rscript

expected_bspcov_version <- "1.0.3"

usage <- function() {
  paste(
    "usage: Rscript --vanilla run_bspcov_parity.R",
    "--method bm|sbm|bandppp|thresholdppp",
    "--manifest PATH --fixture-dir PATH --output PATH",
    sprintf("(requires bspcov %s)", expected_bspcov_version)
  )
}

values <- commandArgs(trailingOnly = TRUE)
if (identical(values, "--help")) {
  cat(usage(), "\n")
  quit(status = 0L)
}

parse_arguments <- function(values) {
  parsed <- list()
  index <- 1L
  while (index <= length(values)) {
    option <- values[[index]]
    if (index == length(values) || !startsWith(option, "--")) {
      stop(sprintf("options require values\n%s", usage()), call. = FALSE)
    }
    parsed[[substring(option, 3L)]] <- values[[index + 1L]]
    index <- index + 2L
  }
  if (!all(c("manifest", "fixture-dir", "method", "output") %in% names(parsed))) {
    stop(sprintf("missing required options\n%s", usage()), call. = FALSE)
  }
  parsed
}

arguments <- parse_arguments(values)
required_packages <- c("bspcov", "jsonlite", "openssl")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1L), quietly = TRUE)
]
if (length(missing_packages) > 0L) {
  stop(
    sprintf("missing required R packages: %s", paste(missing_packages, collapse = ", ")),
    call. = FALSE
  )
}
actual_version <- as.character(utils::packageVersion("bspcov"))
if (!identical(actual_version, expected_bspcov_version)) {
  stop(
    sprintf("expected bspcov %s, found %s", expected_bspcov_version, actual_version),
    call. = FALSE
  )
}

manifest <- jsonlite::fromJSON(arguments[["manifest"]], simplifyVector = FALSE)
method <- arguments[["method"]]
if (!method %in% c("bm", "sbm", "bandppp", "thresholdppp")) {
  stop("unknown parity method", call. = FALSE)
}
fixture_directory <- arguments[["fixture-dir"]]
read_matrix <- function(filename) {
  as.matrix(utils::read.csv(
    file.path(fixture_directory, filename),
    header = FALSE,
    check.names = FALSE
  ))
}
x <- read_matrix("observations.csv")
truth <- read_matrix("truth_covariance.csv")
initial <- read_matrix("initial_covariance.csv")
p <- ncol(x)

fixture_payload <- raw()
for (entry in list(
  c("observations", "observations.csv"),
  c("truth_covariance", "truth_covariance.csv"),
  c("initial_covariance", "initial_covariance.csv")
)) {
  path <- file.path(fixture_directory, entry[[2L]])
  connection <- file(path, open = "rb")
  bytes <- readBin(connection, what = "raw", n = as.integer(file.info(path)$size))
  close(connection)
  fixture_payload <- c(fixture_payload, charToRaw(entry[[1L]]), as.raw(0L), bytes)
}
fixture_sha256 <- as.character(openssl::sha256(fixture_payload))
metadata <- jsonlite::fromJSON(file.path(fixture_directory, "metadata.json"))
if (!identical(fixture_sha256, metadata$sha256)) {
  stop("fixture SHA-256 does not match metadata", call. = FALSE)
}

parity <- manifest$parity
configuration <- manifest$methods[[method]]
seed <- as.integer(manifest$seed)
set.seed(seed)
fit <- if (method == "bm") {
  bspcov::bmspcov(
    X = x,
    Sigma = initial,
    nsample = list(
      burnin = as.integer(parity$bm_sbm_burnin),
      nmc = as.integer(parity$bm_sbm_samples_per_chain)
    ),
    nchain = as.integer(parity$bm_sbm_chains),
    seed = seed,
    do.parallel = TRUE,
    show_progress = FALSE
  )
} else if (method == "sbm") {
  bspcov::sbmspcov(
    X = x,
    Sigma = initial,
    cutoff = list(method = "corr", thr = as.numeric(configuration$retained_fraction)),
    nsample = list(
      burnin = as.integer(parity$bm_sbm_burnin),
      nmc = as.integer(parity$bm_sbm_samples_per_chain)
    ),
    nchain = as.integer(parity$bm_sbm_chains),
    seed = seed,
    do.parallel = TRUE,
    show_progress = FALSE
  )
} else if (method == "bandppp") {
  bspcov::bandPPP(
    x,
    k = 1L,
    eps = as.numeric(configuration$epsilon),
    nsample = as.integer(parity$ppp_total_samples)
  )
} else {
  bspcov::thresPPP(
    x,
    eps = as.numeric(configuration$epsilon),
    thres = list(
      value = as.numeric(configuration$threshold),
      fun = configuration$method
    ),
    nsample = as.integer(parity$ppp_total_samples)
  )
}

draws <- if (is.list(fit$Sigma)) do.call(rbind, fit$Sigma) else as.matrix(fit$Sigma)
if (any(!is.finite(draws))) {
  stop("bspcov returned nonfinite parity draws", call. = FALSE)
}
n_batches <- as.integer(parity$batches)
if (nrow(draws) < 2L * n_batches) {
  stop("parity summaries require at least two draws per batch", call. = FALSE)
}
batch_size <- nrow(draws) %/% n_batches
trimmed_samples <- n_batches * batch_size
trimmed_draws <- draws[seq_len(trimmed_samples), , drop = FALSE]
batch_rows <- split(
  seq_len(trimmed_samples),
  rep(seq_len(n_batches), each = batch_size)
)
parameter_count <- ncol(draws)
batch_means <- vapply(
  batch_rows,
  function(rows) colMeans(trimmed_draws[rows, , drop = FALSE]),
  numeric(parameter_count)
)
batch_sds <- vapply(
  batch_rows,
  function(rows) apply(
    trimmed_draws[rows, , drop = FALSE], 2L, stats::sd
  ),
  numeric(parameter_count)
)
probabilities <- c(0.025, 0.5, 0.975)
posterior_quantiles <- apply(
  draws, 2L, stats::quantile, probs = probabilities, names = FALSE, type = 7L
)
batch_quantiles <- lapply(
  probabilities,
  function(probability) vapply(
    batch_rows,
    function(rows) apply(
      trimmed_draws[rows, , drop = FALSE],
      2L,
      stats::quantile,
      probs = probability,
      names = FALSE,
      type = 7L
    ),
    numeric(parameter_count)
  )
)
batch_mcse <- function(value) {
  apply(value, 1L, stats::sd) / sqrt(n_batches)
}
vech_to_symmetric <- function(value) {
  result <- matrix(0, nrow = p, ncol = p)
  result[lower.tri(result, diag = TRUE)] <- value
  result[upper.tri(result)] <- t(result)[upper.tri(result)]
  result
}

posterior_mean <- vech_to_symmetric(colMeans(draws))
posterior_sd <- vech_to_symmetric(apply(draws, 2L, stats::sd))
batch_rmses <- vapply(
  seq_len(n_batches),
  function(batch) sqrt(mean((vech_to_symmetric(batch_means[, batch]) - truth)^2)),
  numeric(1L)
)
summary <- list(
  truth = truth,
  posterior_mean = posterior_mean,
  posterior_mean_mcse = vech_to_symmetric(batch_mcse(batch_means)),
  posterior_sd = posterior_sd,
  posterior_sd_mcse = vech_to_symmetric(batch_mcse(batch_sds)),
  q025 = vech_to_symmetric(posterior_quantiles[1L, ]),
  q025_mcse = vech_to_symmetric(batch_mcse(batch_quantiles[[1L]])),
  q50 = vech_to_symmetric(posterior_quantiles[2L, ]),
  q50_mcse = vech_to_symmetric(batch_mcse(batch_quantiles[[2L]])),
  q975 = vech_to_symmetric(posterior_quantiles[3L, ]),
  q975_mcse = vech_to_symmetric(batch_mcse(batch_quantiles[[3L]])),
  rmse = sqrt(mean((posterior_mean - truth)^2)),
  rmse_mcse = stats::sd(batch_rmses) / sqrt(n_batches),
  n_batches = n_batches,
  batch_size = batch_size,
  trimmed_samples = trimmed_samples
)

git_revision <- system2("git", c("rev-parse", "HEAD"), stdout = TRUE, stderr = FALSE)[[1L]]
git_status <- system2("git", c("status", "--porcelain"), stdout = TRUE, stderr = FALSE)
if (length(git_status) > 0L) {
  stop("parity artifacts require a clean git worktree", call. = FALSE)
}
result <- list(
  schema_version = "1.0",
  method = method,
  implementation = "bspcov",
  version = actual_version,
  dtype = "float64",
  device = "cpu",
  fixture_sha256 = fixture_sha256,
  git_revision = git_revision,
  git_dirty = FALSE,
  summary = summary
)
output <- arguments[["output"]]
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
writeLines(
  jsonlite::toJSON(result, auto_unbox = TRUE, digits = 17L, null = "null"),
  output
)
