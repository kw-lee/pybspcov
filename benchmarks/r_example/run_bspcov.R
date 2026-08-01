#!/usr/bin/env Rscript

BASELINE_BSPCOV_VERSION <- "1.0.3"
FIXTURE_SEED <- 1L
SAMPLER_SEED <- 1L
QUANTILE_PROBABILITIES <- c(0.025, 0.5, 0.975)
THREAD_ENVIRONMENT_VARIABLES <- c(
  "OMP_NUM_THREADS",
  "OPENBLAS_NUM_THREADS",
  "MKL_NUM_THREADS",
  "VECLIB_MAXIMUM_THREADS",
  "BLIS_NUM_THREADS"
)

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
        parsed$n_samples <- parse_positive_integer(value, option, minimum = 4L)
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

first_available <- function(value, fallback = "unavailable") {
  usable <- value[!is.na(value) & nzchar(value)]
  if (length(usable) == 0L) fallback else as.character(usable[[1L]])
}

detect_cpu_model <- function() {
  system_name <- first_available(unname(Sys.info()["sysname"]))
  if (system_name == "Linux" && file.exists("/proc/cpuinfo")) {
    cpuinfo <- tryCatch(
      readLines("/proc/cpuinfo", warn = FALSE),
      error = function(error) character()
    )
    model_lines <- grep("^model name[[:space:]]*:", cpuinfo, value = TRUE)
    if (length(model_lines) > 0L) {
      return(trimws(sub("^[^:]*:", "", model_lines[[1L]])))
    }
  }
  if (system_name == "Darwin" && nzchar(Sys.which("sysctl"))) {
    model <- tryCatch(
      system2(
        "sysctl",
        c("-n", "machdep.cpu.brand_string"),
        stdout = TRUE,
        stderr = FALSE
      ),
      error = function(error) character()
    )
    if (length(model) > 0L && nzchar(model[[1L]])) return(model[[1L]])
  }
  if (system_name == "Windows") {
    model <- Sys.getenv("PROCESSOR_IDENTIFIER", unset = "")
    if (nzchar(model)) return(model)
  }
  architecture <- first_available(unname(Sys.info()["machine"]))
  sprintf("unavailable (architecture: %s)", architecture)
}

detect_linux_physical_cores <- function() {
  if (
    first_available(unname(Sys.info()["sysname"])) != "Linux" ||
      !file.exists("/proc/cpuinfo")
  ) {
    return(NA_integer_)
  }
  cpuinfo <- tryCatch(
    readLines("/proc/cpuinfo", warn = FALSE),
    error = function(error) character()
  )
  if (length(cpuinfo) == 0L) return(NA_integer_)
  records <- strsplit(paste(cpuinfo, collapse = "\n"), "\n\n", fixed = TRUE)[[1L]]
  core_pairs <- vapply(
    records,
    function(record) {
      lines <- strsplit(record, "\n", fixed = TRUE)[[1L]]
      physical_id <- grep("^physical id[[:space:]]*:", lines, value = TRUE)
      core_id <- grep("^core id[[:space:]]*:", lines, value = TRUE)
      if (length(physical_id) == 0L || length(core_id) == 0L) {
        return(NA_character_)
      }
      paste(
        trimws(sub("^[^:]*:", "", physical_id[[1L]])),
        trimws(sub("^[^:]*:", "", core_id[[1L]])),
        sep = ":"
      )
    },
    character(1L)
  )
  core_pairs <- unique(core_pairs[!is.na(core_pairs)])
  if (length(core_pairs) == 0L) NA_integer_ else length(core_pairs)
}

format_core_count <- function(value) {
  if (length(value) == 1L && !is.na(value) && value > 0L) {
    as.character(as.integer(value))
  } else {
    "unavailable"
  }
}

detect_hardware <- function() {
  logical_cores <- suppressWarnings(parallel::detectCores(logical = TRUE))
  physical_cores <- detect_linux_physical_cores()
  if (
    is.na(physical_cores) &&
      first_available(unname(Sys.info()["sysname"])) != "Linux"
  ) {
    physical_cores <- suppressWarnings(parallel::detectCores(logical = FALSE))
  }
  software_versions <- tryCatch(extSoftVersion(), error = function(error) NULL)
  blas <- if (is.null(software_versions)) {
    "unavailable"
  } else {
    first_available(unname(software_versions["BLAS"]))
  }
  lapack <- first_available(
    tryCatch(La_library(), error = function(error) character())
  )
  list(
    cpu_model = detect_cpu_model(),
    logical_cores = format_core_count(logical_cores),
    physical_cores = format_core_count(physical_cores),
    blas = blas,
    lapack = lapack,
    thread_environment = Sys.getenv(
      THREAD_ENVIRONMENT_VARIABLES,
      unset = "<unset>"
    )
  )
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

end_to_end_start <- proc.time()[["elapsed"]]
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
sampler_seconds <- system.time({
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

n_batches <- min(20L, floor(nrow(draws) / 2L))
batch_size <- floor(nrow(draws) / n_batches)
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
    trimmed_draws[rows, , drop = FALSE],
    MARGIN = 2L,
    FUN = stats::sd
  ),
  numeric(parameter_count)
)
batch_quantiles <- lapply(
  QUANTILE_PROBABILITIES,
  function(probability) vapply(
    batch_rows,
    function(rows) apply(
      trimmed_draws[rows, , drop = FALSE],
      MARGIN = 2L,
      FUN = stats::quantile,
      probs = probability,
      names = FALSE,
      type = 7L
    ),
    numeric(parameter_count)
  )
)
batch_mcse <- function(value) {
  apply(value, MARGIN = 1L, FUN = stats::sd) / sqrt(n_batches)
}

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
posterior_mean_mcse <- vech_to_symmetric(batch_mcse(batch_means), p)
posterior_sd_mcse <- vech_to_symmetric(batch_mcse(batch_sds), p)
q025_mcse <- vech_to_symmetric(batch_mcse(batch_quantiles[[1L]]), p)
q50_mcse <- vech_to_symmetric(batch_mcse(batch_quantiles[[2L]]), p)
q975_mcse <- vech_to_symmetric(batch_mcse(batch_quantiles[[3L]]), p)
batch_rmses <- vapply(
  seq_len(n_batches),
  function(batch) sqrt(mean(
    (vech_to_symmetric(batch_means[, batch], p) - true_sigma)^2
  )),
  numeric(1L)
)
rmse_mcse <- stats::sd(batch_rmses) / sqrt(n_batches)

row_indices <- rep(seq_len(p), times = p)
column_indices <- rep(seq_len(p), each = p)
summary_output <- data.frame(
  implementation = "bspcov",
  row = row_indices,
  column = column_indices,
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
  truth = as.vector(true_sigma),
  rmse = rmse,
  rmse_mcse = rmse_mcse,
  stringsAsFactors = FALSE
)

session <- utils::sessionInfo()
hardware <- detect_hardware()
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
    "cpu_model",
    "logical_cores",
    "physical_cores",
    "blas",
    "lapack",
    THREAD_ENVIRONMENT_VARIABLES,
    "n",
    "p",
    "burnin",
    "n_samples",
    "n_batches",
    "batch_size",
    "trimmed_samples",
    "chains",
    "repetitions",
    "fixture_seed",
    "sampler_seed",
    "fixture_centered",
    "sampler_timing_scope",
    "end_to_end_timing_scope",
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
    hardware$cpu_model,
    hardware$logical_cores,
    hardware$physical_cores,
    hardware$blas,
    hardware$lapack,
    unname(hardware$thread_environment),
    as.character(n),
    as.character(p),
    as.character(arguments$burnin),
    as.character(arguments$n_samples),
    as.character(n_batches),
    as.character(batch_size),
    as.character(trimmed_samples),
    "1",
    "1",
    as.character(FIXTURE_SEED),
    as.character(SAMPLER_SEED),
    "true",
    "bspcov::bmspcov only",
    paste(
      "fixture reads through summary and metadata CSV writes;",
      "excludes the final timing CSV write"
    ),
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
  metadata_output,
  file.path(arguments$output_directory, "r_metadata.csv"),
  row.names = FALSE
)
end_to_end_seconds <- proc.time()[["elapsed"]] - end_to_end_start
timing_output <- data.frame(
  implementation = "bspcov",
  sampler_seconds = as.numeric(sampler_seconds),
  end_to_end_seconds = as.numeric(end_to_end_seconds),
  stringsAsFactors = FALSE
)
utils::write.csv(
  timing_output,
  file.path(arguments$output_directory, "r_timing.csv"),
  row.names = FALSE
)
