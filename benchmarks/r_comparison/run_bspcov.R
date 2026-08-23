#!/usr/bin/env Rscript

process_start <- proc.time()[["elapsed"]]
expected_bspcov_version <- "1.0.3"

usage <- function() {
  paste(
    "usage: Rscript --vanilla run_bspcov.R",
    "--method bm|sbm|bandppp|thresholdppp",
    "--manifest PATH --fixture-dir PATH --parallelism INTEGER",
    "--configuration optimized|cpu_baseline --cpu-cores INTEGER --output PATH",
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
  required <- c(
    "manifest", "fixture-dir", "method", "parallelism", "configuration",
    "cpu-cores", "output"
  )
  if (!all(required %in% names(parsed))) {
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
  stop("unknown benchmark method", call. = FALSE)
}
parallelism <- as.integer(arguments[["parallelism"]])
cpu_cores <- as.integer(arguments[["cpu-cores"]])
if (is.na(parallelism) || parallelism < 1L || is.na(cpu_cores) || cpu_cores < 1L) {
  stop("parallelism and cpu-cores must be positive integers", call. = FALSE)
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
metadata <- jsonlite::fromJSON(file.path(fixture_directory, "metadata.json"))
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
  fixture_payload <- c(
    fixture_payload,
    charToRaw(entry[[1L]]),
    as.raw(0L),
    bytes
  )
}
fixture_sha256 <- as.character(openssl::sha256(fixture_payload))
if (!identical(fixture_sha256, metadata$sha256)) {
  stop("fixture SHA-256 does not match metadata", call. = FALSE)
}

method_configuration <- manifest$methods[[method]]
fit_once <- function(seed) {
  set.seed(seed)
  if (method == "bm") {
    return(bspcov::bmspcov(
      X = x,
      Sigma = initial,
      nsample = list(
        burnin = as.integer(method_configuration$burnin),
        nmc = as.integer(method_configuration$samples)
      ),
      nchain = parallelism,
      seed = seed,
      do.parallel = parallelism > 1L,
      show_progress = FALSE
    ))
  }
  if (method == "sbm") {
    return(bspcov::sbmspcov(
      X = x,
      Sigma = initial,
      cutoff = list(
        method = "corr",
        thr = as.numeric(method_configuration$retained_fraction)
      ),
      nsample = list(
        burnin = as.integer(method_configuration$burnin),
        nmc = as.integer(method_configuration$samples)
      ),
      nchain = parallelism,
      seed = seed,
      do.parallel = parallelism > 1L,
      show_progress = FALSE
    ))
  }

  fit_ppp <- function(batch) {
    set.seed(seed + batch)
    if (method == "bandppp") {
      bspcov::bandPPP(
        x,
        k = max(1L, p %/% as.integer(method_configuration$bandwidth_divisor)),
        eps = as.numeric(method_configuration$epsilon),
        nsample = as.integer(method_configuration$samples_per_batch)
      )
    } else {
      bspcov::thresPPP(
        x,
        eps = as.numeric(method_configuration$epsilon),
        thres = list(
          value = as.numeric(method_configuration$threshold),
          fun = method_configuration$method
        ),
        nsample = as.integer(method_configuration$samples_per_batch)
      )
    }
  }
  batches <- if (parallelism == 1L) {
    list(fit_ppp(1L))
  } else {
    parallel::mclapply(seq_len(parallelism), fit_ppp, mc.cores = parallelism)
  }
  list(Sigma = do.call(rbind, lapply(batches, function(fit) fit$Sigma)))
}

extract_draws <- function(fit) {
  draws <- if (is.list(fit$Sigma)) do.call(rbind, fit$Sigma) else as.matrix(fit$Sigma)
  if (any(!is.finite(draws))) {
    stop("bspcov returned nonfinite posterior draws", call. = FALSE)
  }
  draws
}

timed_fit <- function(seed) {
  elapsed <- system.time(fit <- fit_once(seed))[["elapsed"]]
  list(seconds = as.numeric(elapsed), fit = fit)
}

seed <- as.integer(manifest$seed)
cold <- timed_fit(seed)
cold_end_to_end_seconds <- as.numeric(proc.time()[["elapsed"]] - process_start)
warm <- lapply(seq_len(3L), function(index) timed_fit(seed + index))
warm_seconds <- vapply(warm, function(value) value$seconds, numeric(1L))
relative_range <- (max(warm_seconds) - min(warm_seconds)) / stats::median(warm_seconds)
if (relative_range > as.numeric(manifest$timing$relative_range_threshold)) {
  extra <- lapply(4:5, function(index) timed_fit(seed + index))
  warm <- c(warm, extra)
  warm_seconds <- vapply(warm, function(value) value$seconds, numeric(1L))
}

draws <- extract_draws(warm[[length(warm)]]$fit)
posterior_mean <- matrix(0, nrow = p, ncol = p)
posterior_mean[lower.tri(posterior_mean, diag = TRUE)] <- colMeans(draws)
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

git_revision <- system2(
  "git", c("rev-parse", "HEAD"), stdout = TRUE, stderr = FALSE
)[[1L]]
git_status <- system2(
  "git", c("status", "--porcelain"), stdout = TRUE, stderr = FALSE
)

result <- list(
  schema_version = "1.0",
  method = method,
  dimension = as.integer(metadata$dimension),
  n_observations = as.integer(metadata$n_observations),
  seed = seed,
  fixture_sha256 = fixture_sha256,
  implementation = "bspcov",
  version = actual_version,
  device = "cpu",
  actual_platform = "cpu",
  dtype = "float64",
  execution = if (parallelism > 1L) "parallel" else "single",
  configuration = arguments[["configuration"]],
  parallelism = parallelism,
  cpu_cores = cpu_cores,
  retained_draws = nrow(draws),
  cold_fit_seconds = cold$seconds,
  cold_end_to_end_seconds = cold_end_to_end_seconds,
  warm_seconds = as.list(warm_seconds),
  posterior_mean_finite = finite,
  posterior_mean_symmetric = symmetric,
  posterior_mean_spd = spd,
  rejected_sweeps = 0L,
  truth_relative_frobenius_error = as.numeric(relative_error),
  git_revision = git_revision,
  git_dirty = length(git_status) > 0L,
  r_version = R.version.string,
  blas = unname(extSoftVersion()[["BLAS"]]),
  lapack = La_library()
)
if (result$git_dirty) {
  stop("publishable timing records require a clean git worktree", call. = FALSE)
}
if (!finite || !symmetric || !spd) {
  stop("posterior mean failed finite, symmetric, or SPD validation", call. = FALSE)
}

output <- arguments[["output"]]
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
writeLines(
  jsonlite::toJSON(result, auto_unbox = TRUE, digits = 17L, null = "null"),
  output
)
