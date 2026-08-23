fixture_sha256 <- function(directory) {
  if (!requireNamespace("openssl", quietly = TRUE)) {
    stop("fixture hashing requires the R package openssl", call. = FALSE)
  }
  payload <- raw()
  for (entry in list(
    c("observations", "observations.csv"),
    c("truth_covariance", "truth_covariance.csv"),
    c("initial_covariance", "initial_covariance.csv")
  )) {
    path <- file.path(directory, entry[[2L]])
    connection <- file(path, open = "rb")
    bytes <- readBin(connection, what = "raw", n = as.integer(file.info(path)$size))
    close(connection)
    payload <- c(payload, charToRaw(entry[[1L]]), as.raw(0L), bytes)
  }
  hash <- as.character(openssl::sha256(payload))
  tolower(gsub(":", "", hash, fixed = TRUE))
}
