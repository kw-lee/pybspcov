#!/usr/bin/env Rscript

# Generate the bspcov 1.0.3 bmspcov documentation example as a versioned CSV
# fixture. The upstream example samples X from a zero-mean distribution, while
# bmspcov's input contract requires the realized columns to have mean zero. We
# therefore center X exactly once here; both R and JAX runners consume this same
# committed matrix.

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_argument) == 1L) {
  normalizePath(sub("^--file=", "", script_argument), mustWork = TRUE)
} else {
  normalizePath("benchmarks/r_example/generate_case.R", mustWork = TRUE)
}

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) > 1L) {
  stop("usage: Rscript generate_case.R [output-directory]", call. = FALSE)
}
output_directory <- if (length(arguments) == 1L) {
  arguments[[1L]]
} else {
  file.path(dirname(script_path), "data")
}
dir.create(output_directory, recursive = TRUE, showWarnings = FALSE)

if (!requireNamespace("MASS", quietly = TRUE)) {
  stop("generate_case.R requires the R package MASS", call. = FALSE)
}

set.seed(1)
n <- 20L
p <- 5L

true_sigma <- matrix(0, nrow = p, ncol = p)
diag(true_sigma) <- 1
values <- -runif(n = p * (p - 1L) / 2L, min = 0.2, max = 0.8)
nonzero_indices <- which(
  rbinom(n = p * (p - 1L) / 2L, size = 1L, prob = 1 / p) == 1L
)
zero_indices <- (seq_len(p * (p - 1L) / 2L))[-nonzero_indices]
values[zero_indices] <- 0
true_sigma[lower.tri(true_sigma)] <- values
true_sigma[upper.tri(true_sigma)] <- t(true_sigma)[upper.tri(true_sigma)]
minimum_eigenvalue <- min(eigen(true_sigma, symmetric = TRUE)$values)
if (minimum_eigenvalue <= 0) {
  delta <- -minimum_eigenvalue + 1.0e-5
  true_sigma <- true_sigma + delta * diag(p)
}

x <- MASS::mvrnorm(n = n, mu = rep(0, p), Sigma = true_sigma)
x <- sweep(x, MARGIN = 2L, STATS = colMeans(x), FUN = "-")
initial_sigma <- diag(diag(stats::cov(x)))

write_matrix <- function(value, filename) {
  previous_digits <- getOption("digits")
  on.exit(options(digits = previous_digits), add = TRUE)
  options(digits = 17L)
  write.table(
    value,
    file = file.path(output_directory, filename),
    sep = ",",
    row.names = FALSE,
    col.names = FALSE,
    quote = FALSE
  )
}

write_matrix(x, "bm_example_x.csv")
write_matrix(true_sigma, "bm_example_truth.csv")
write_matrix(initial_sigma, "bm_example_initial.csv")
