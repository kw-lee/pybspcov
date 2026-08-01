options(digits = 17, scipen = 999)

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
    -1.0, 0.5, 0.2,
    -0.4, -0.7, 0.1,
    0.2, 0.1, -0.8,
    0.5, -0.2, 0.6,
    0.9, 0.4, -0.3,
    -0.2, -0.1, 0.2
  ),
  ncol = 3,
  byrow = TRUE
)
Sigma <- matrix(
  c(
    1.40, 0.25, -0.15,
    0.25, 1.10, 0.20,
    -0.15, 0.20, 0.90
  ),
  ncol = 3,
  byrow = TRUE
)
tau <- matrix(
  c(
    1.0, 1.3, 0.7,
    1.3, 1.0, 1.8,
    0.7, 1.8, 1.0
  ),
  ncol = 3,
  byrow = TRUE
)
invisible(chol(Sigma))

n <- nrow(X)
p <- ncol(X)
i <- 2L
ind_noi <- seq_len(p)[-i]
lambda <- 1.25
gam <- 0.85

# bspcov 1.0.3 R/bmspcov.R blocked-Gibbs column calculations.
S <- crossprod(X)
C <- solve(Sigma)
C11 <- C[ind_noi, ind_noi]
C12 <- C[ind_noi, i]
S11 <- S[ind_noi, ind_noi]
S12 <- S[ind_noi, i]
invSig11 <- C11 - tcrossprod(C12) / C[i, i]
invSig11S12 <- invSig11 %*% S12
W1 <- invSig11 %*% S11 %*% invSig11
beta <- Sigma[ind_noi, i, drop = FALSE]
chi <- drop(
  crossprod(beta, W1 %*% beta) -
    2 * crossprod(beta, invSig11S12) +
    S[i, i]
)
W <- W1 / gam + diag(1 / tau[ind_noi, i]) + lambda * invSig11
W <- (W + t(W)) / 2
W_chol <- chol(W)
mu_i <- backsolve(
  W_chol,
  forwardsolve(t(W_chol), invSig11S12)
) / gam

write_csv(X, "bm_x.csv")
write_csv(Sigma, "bm_covariance.csv")
write_csv(C, "bm_precision.csv")
write_csv(tau, "bm_tau.csv")
write_csv(c(i - 1L, n, lambda, gam), "bm_parameters.csv")
write_csv(ind_noi - 1L, "bm_other_indices.csv")
write_csv(invSig11, "bm_expected_conditional_precision.csv")
write_csv(drop(invSig11S12), "bm_expected_conditional_scatter.csv")
write_csv(W1, "bm_expected_quadratic.csv")
write_csv(c(1 - n / 2, chi, lambda), "bm_expected_gamma_parameters.csv")
write_csv(W, "bm_expected_beta_precision.csv")
write_csv(drop(mu_i), "bm_expected_beta_mean.csv")
