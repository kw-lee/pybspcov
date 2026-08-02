options(digits = 17, scipen = 999)

rng_version <- "4.5.0"
RNGversion(rng_version)
r_version <- as.character(getRversion())
rng_kind <- RNGkind()
bayes_factor_version <- as.character(utils::packageVersion("BayesFactor"))
mass_version <- as.character(utils::packageVersion("MASS"))

expected_version <- "1.0.3"
actual_version <- as.character(utils::packageVersion("bspcov"))
if (!identical(actual_version, expected_version)) {
  stop(
    sprintf(
      "Expected bspcov %s, but found %s",
      expected_version,
      actual_version
    )
  )
}

output_dir <- file.path("tests", "fixtures", "r", "bspcov-1.0.3")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

write_csv <- function(values, filename) {
  if (is.null(dim(values))) {
    values <- matrix(values, ncol = 1)
  }
  formatted <- ifelse(is.na(values), "NA", sprintf("%.17g", values))
  dim(formatted) <- dim(values)
  write.table(
    formatted,
    file = file.path(output_dir, filename),
    sep = ",",
    row.names = FALSE,
    col.names = FALSE,
    quote = FALSE
  )
}

excluded_mask_from_indices <- function(indices, p) {
  mask <- matrix(FALSE, nrow = p, ncol = p)
  for (column in seq_len(p)) {
    if (length(indices[[column]]) > 0) {
      mask[indices[[column]], column] <- TRUE
    }
  }
  stopifnot(identical(mask, t(mask)), !any(diag(mask)))
  mask
}

apply_screening <- function(Sigma, mask) {
  Sigma[mask] <- 0
  Sigma
}

# Discrete cosine vectors provide a deterministic, exactly centered design.
# The linear combinations set the six sample correlations to approximately
# 0.65, 0, -0.45, 0, -0.2925, and 0 without sampling input data.
n <- 12L
p <- 4L
observation <- 0:(n - 1L)
basis <- vapply(
  1:p,
  function(k) cos(pi * (observation + 0.5) * k / n),
  numeric(n)
)
basis <- apply(basis, 2, function(column) column / sqrt(sum(column^2)))
X <- cbind(
  basis[, 1],
  0.65 * basis[, 1] + sqrt(1 - 0.65^2) * basis[, 2],
  basis[, 3],
  -0.45 * basis[, 1] + sqrt(1 - 0.45^2) * basis[, 4]
)
stopifnot(max(abs(colMeans(X))) < 1e-15)

initial_covariance <- matrix(
  c(
    1.40, 0.12, 0.14, 0.16,
    0.12, 1.30, 0.18, 0.20,
    0.14, 0.18, 1.20, 0.22,
    0.16, 0.20, 0.22, 1.10
  ),
  nrow = p,
  byrow = TRUE
)
invisible(chol(initial_covariance))

seed <- 314159L
# Observe the seed that select_cutoff() derives internally. The subsequent
# public sbmspcov() call resets to seed before repeating the same draw.
set.seed(seed)
internal_cutoff_seed <- sample(.Machine$integer.max, 1)
sampler_settings <- list(burnin = 0L, nmc = 1L)

# Calling the exported function verifies the default FNR branch, including
# select_cutoff()'s internal reseeding behavior and its default parameters.
fnr_result <- bspcov::sbmspcov(
  X = X,
  Sigma = initial_covariance,
  seed = seed,
  nsample = sampler_settings,
  show_progress = FALSE
)

# Selecting only the method exercises the implementation's default thr=0.2.
corr_result <- bspcov::sbmspcov(
  X = X,
  Sigma = initial_covariance,
  cutoff = list(method = "corr"),
  seed = seed,
  nsample = sampler_settings,
  show_progress = FALSE
)

pairwise_bf <- getFromNamespace("pairwise.Jeffreys", "bspcov")(X)
correlations <- stats::cor(X)
fnr_excluded_mask <- excluded_mask_from_indices(fnr_result$INDzero, p)
corr_excluded_mask <- excluded_mask_from_indices(corr_result$INDzero, p)
fnr_active_mask <- (!fnr_excluded_mask) & !diag(p)
corr_active_mask <- (!corr_excluded_mask) & !diag(p)

# Independently reconstruct each support decision from the recorded scalar
# cutoff and verify it agrees with the public sbmspcov() result.
fnr_support <- pairwise_bf > as.numeric(fnr_result$cutoff)
fnr_support[upper.tri(fnr_support)] <- t(fnr_support)[upper.tri(fnr_support)]
stopifnot(identical(fnr_excluded_mask, (!fnr_support) & !diag(p)))

corr_lower <- correlations
corr_lower[upper.tri(corr_lower, diag = TRUE)] <- NA
corr_cutoff <- as.numeric(
  stats::quantile(abs(corr_lower), prob = 0.8, na.rm = TRUE)
)
corr_support <- abs(corr_lower) > corr_cutoff
corr_support[upper.tri(corr_support)] <-
  t(corr_support)[upper.tri(corr_support)]
stopifnot(identical(corr_excluded_mask, (!corr_support) & !diag(p)))

write_csv(X, "sbm_screening_x.csv")
write_csv(initial_covariance, "sbm_screening_initial_covariance.csv")
write_csv(pairwise_bf, "sbm_screening_pairwise_bf.csv")
write_csv(correlations, "sbm_screening_correlations.csv")
write_csv(fnr_excluded_mask, "sbm_screening_fnr_excluded_mask.csv")
write_csv(corr_excluded_mask, "sbm_screening_corr_excluded_mask.csv")
write_csv(fnr_active_mask, "sbm_screening_fnr_active_mask.csv")
write_csv(corr_active_mask, "sbm_screening_corr_active_mask.csv")
write_csv(
  apply_screening(initial_covariance, fnr_excluded_mask),
  "sbm_screening_fnr_covariance.csv"
)
write_csv(
  apply_screening(initial_covariance, corr_excluded_mask),
  "sbm_screening_corr_covariance.csv"
)

session_details <- c(
  capture.output(utils::sessionInfo()),
  "",
  sprintf("Pinned RNGversion: %s", rng_version),
  sprintf("RNGkind: %s", paste(rng_kind, collapse = ", ")),
  sprintf("Observed select_cutoff internal seed: %d", internal_cutoff_seed),
  "Numeric precision: IEEE 754 double"
)
session_details <- sub("[[:blank:]]+$", "", session_details)
writeLines(
  session_details,
  file.path(output_dir, "sbm_screening_session_info.txt")
)

metadata <- c(
  "{",
  '  "package": "bspcov",',
  sprintf('  "package_version": "%s",', actual_version),
  '  "source_tag": "1.0.3",',
  '  "source_commit": "165106c5ab8f6506e6d69b0b8f94ce5bdc99092f",',
  '  "index_base": 0,',
  sprintf('  "n": %d,', n),
  sprintf('  "p": %d,', p),
  '  "generation_environment": {',
  sprintf('    "r_version": "%s",', r_version),
  sprintf('    "rng_version": "%s",', rng_version),
  '    "rng_kind": [',
  sprintf('      "%s",', rng_kind[[1]]),
  sprintf('      "%s",', rng_kind[[2]]),
  sprintf('      "%s"', rng_kind[[3]]),
  "    ],",
  sprintf('    "BayesFactor_version": "%s",', bayes_factor_version),
  sprintf('    "MASS_version": "%s",', mass_version),
  '    "numeric_precision": "IEEE 754 double"',
  "  },",
  '  "mask_semantics": {',
  '    "excluded": "true_for_upstream_INDzero",',
  '    "active": "true_for_retained_off_diagonal_edge"',
  "  },",
  '  "fnr": {',
  '    "method": "FNR",',
  sprintf('    "seed": %d,', seed),
  sprintf('    "internal_cutoff_seed": %d,', internal_cutoff_seed),
  '    "rho": 0.25,',
  '    "FNR": 0.05,',
  '    "nsimdata": 1000,',
  sprintf('    "cutoff": %.17g', as.numeric(fnr_result$cutoff)),
  "  },",
  '  "corr": {',
  '    "method": "corr",',
  '    "thr": 0.2,',
  '    "quantile_probability": 0.8,',
  sprintf('    "cutoff": %.17g', corr_cutoff),
  "  }",
  "}"
)
writeLines(metadata, file.path(output_dir, "sbm_screening_metadata.json"))
