options(digits = 17, scipen = 999)

required_version <- "1.0.3"
installed_version <- as.character(packageVersion("bspcov"))
if (installed_version != required_version) {
  stop(
    sprintf(
      "Expected bspcov %s, found %s",
      required_version,
      installed_version
    )
  )
}

output_dir <- file.path("tests", "fixtures", "r", "bspcov-1.0.3")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

write_csv <- function(values, filename) {
  if (is.null(dim(values))) {
    values <- matrix(values, ncol = 1)
  }
  values <- matrix(
    sprintf("%.17g", as.numeric(values)),
    nrow = nrow(values),
    ncol = ncol(values)
  )
  write.table(
    values,
    file = file.path(output_dir, filename),
    sep = ",",
    row.names = FALSE,
    col.names = FALSE,
    quote = FALSE
  )
}

X <- matrix(
  c(
    -1.0, 0.5, 0.2, -0.1,
    -0.4, -0.7, 0.1, 0.3,
    0.2, 0.1, -0.8, 0.4,
    0.5, -0.2, 0.6, -0.5,
    0.9, 0.4, -0.3, 0.2,
    -0.2, -0.1, 0.2, -0.3
  ),
  ncol = 4,
  byrow = TRUE
)
Sigma <- matrix(
  c(
    2.00, 0.20, 0.00, 0.10,
    0.20, 1.50, -0.15, 0.00,
    0.00, -0.15, 1.20, 0.00,
    0.10, 0.00, 0.00, 1.80
  ),
  ncol = 4,
  byrow = TRUE
)
tau <- matrix(
  c(
    0.40, 0.30, 0.20, 0.50,
    0.30, 0.40, 0.60, 0.70,
    0.20, 0.60, 0.40, 0.80,
    0.50, 0.70, 0.80, 0.40
  ),
  ncol = 4,
  byrow = TRUE
)
invisible(chol(Sigma))

n <- nrow(X)
p <- ncol(X)
i <- 2L
ind_noi <- seq_len(p)[-i]
lambda <- 1.0
gam <- 0.9
active_mask <- Sigma != 0
diag(active_mask) <- FALSE
active <- active_mask[ind_noi, i]
active_positions <- which(active)

# bspcov 1.0.3 R/sbmspcov.R partial-column calculations, padded back to p - 1.
S <- crossprod(X)
C <- solve(Sigma)
C11 <- C[ind_noi, ind_noi]
C12 <- C[ind_noi, i]
S11 <- S[ind_noi, ind_noi]
S12 <- S[ind_noi, i]
invSig11 <- C11 - tcrossprod(C12) / C[i, i]
reduced_rows <- invSig11[active_positions, , drop = FALSE]
reduced_scatter <- drop(reduced_rows %*% S12)
W1 <- reduced_rows %*% S11 %*% t(reduced_rows)
beta <- Sigma[ind_noi[active_positions], i, drop = FALSE]
chi <- drop(
  crossprod(beta, W1 %*% beta) -
    2 * crossprod(beta, reduced_scatter) +
    S[i, i]
)
W <- W1 / gam +
  diag(1 / tau[ind_noi[active_positions], i]) +
  lambda * invSig11[active_positions, active_positions]
W <- (W + t(W)) / 2
W_chol <- chol(W)
mu_i <- backsolve(
  W_chol,
  forwardsolve(t(W_chol), reduced_scatter)
) / gam

padded_scatter <- numeric(p - 1L)
padded_scatter[active_positions] <- reduced_scatter
padded_quadratic <- matrix(0, nrow = p - 1L, ncol = p - 1L)
padded_quadratic[active_positions, active_positions] <- W1
padded_beta_precision <- diag(as.numeric(!active))
padded_beta_precision[active_positions, active_positions] <- W
padded_beta_mean <- numeric(p - 1L)
padded_beta_mean[active_positions] <- drop(mu_i)

write_csv(X, "sbm_column_x.csv")
write_csv(Sigma, "sbm_column_covariance.csv")
write_csv(C, "sbm_column_precision.csv")
write_csv(tau, "sbm_column_tau.csv")
write_csv(active_mask, "sbm_column_active_mask.csv")
write_csv(c(i - 1L, n, lambda, gam), "sbm_column_parameters.csv")
write_csv(ind_noi - 1L, "sbm_column_other_indices.csv")
write_csv(active, "sbm_column_expected_active.csv")
write_csv(invSig11, "sbm_column_expected_conditional_precision.csv")
write_csv(padded_scatter, "sbm_column_expected_conditional_scatter.csv")
write_csv(padded_quadratic, "sbm_column_expected_quadratic.csv")
write_csv(c(1 - n / 2, chi, lambda), "sbm_column_expected_gamma_parameters.csv")
write_csv(padded_beta_precision, "sbm_column_expected_beta_precision.csv")
write_csv(padded_beta_mean, "sbm_column_expected_beta_mean.csv")
