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

Sigma <- matrix(
  c(
    2.00, 0.25, -0.10, 0.15,
    0.25, 1.60, 0.30, -0.20,
    -0.10, 0.30, 1.40, 0.18,
    0.15, -0.20, 0.18, 1.20
  ),
  nrow = 4,
  byrow = TRUE
)
invisible(chol(Sigma))

i <- 3L
ind_noi <- seq_len(nrow(Sigma))[-i]
beta <- matrix(c(0.31, -0.27, 0.22), ncol = 1)
gam <- 0.85

C <- solve(Sigma)
input_Sigma <- Sigma
input_C <- C

# bspcov 1.0.3 R/bmspcov.R line 225.
invSig11 <- C[ind_noi, ind_noi] - tcrossprod(C[ind_noi, i]) / C[i, i]

# bspcov 1.0.3 R/bmspcov.R lines 256-258.
Sigma[ind_noi, i] <- beta
Sigma[i, ind_noi] <- beta
Sigma[i, i] <- gam + crossprod(beta, invSig11 %*% beta)

# bspcov 1.0.3 R/bmspcov.R lines 274-281.
invSig11beta <- invSig11 %*% beta
C[ind_noi, ind_noi] <- invSig11 + tcrossprod(invSig11beta) / gam
C12 <- -invSig11beta / gam
C[ind_noi, i] <- C12
C[i, ind_noi] <- t(C12)
C[i, i] <- 1 / gam

write_csv(c(i - 1L, gam), "parameters.csv")
write_csv(ind_noi - 1L, "other_indices.csv")
write_csv(drop(beta), "beta.csv")
write_csv(input_Sigma, "covariance.csv")
write_csv(input_C, "precision.csv")
write_csv(Sigma, "expected_covariance.csv")
write_csv(C, "expected_precision.csv")
