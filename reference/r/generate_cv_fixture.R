args <- commandArgs(trailingOnly = TRUE)
output_directory <- if (length(args)) args[[1]] else "tests/fixtures/r/bspcov-1.0.3"
library_directory <- if (length(args) >= 2) args[[2]] else NULL
if (!is.null(library_directory)) .libPaths(c(library_directory, .libPaths()))
stopifnot(as.character(packageVersion("bspcov")) == "1.0.3")
dir.create(output_directory, recursive = TRUE, showWarnings = FALSE)

set.seed(11)
band_x <- scale(matrix(rnorm(30), ncol = 3), center = TRUE, scale = FALSE)
set.seed(7)
band <- bspcov::cv.bandPPP(
  band_x,
  kvec = c(1, 2),
  epsvec = c(0.01, 0.05),
  nsample = 1000,
  ncores = 1
)$elpd
future::plan(future::sequential)

set.seed(13)
threshold_x <- scale(matrix(rnorm(60), ncol = 3), center = TRUE, scale = FALSE)
set.seed(17)
threshold <- bspcov::cv.thresPPP(
  threshold_x,
  thresvec = c(0.05, 0.2),
  epsvec = c(0.01, 0.05),
  nsample = 1000,
  ncores = 1
)$error
future::plan(future::sequential)

write.csv(band_x, file.path(output_directory, "band_cv_x.csv"), row.names = FALSE)
write.csv(band, file.path(output_directory, "band_cv_scores.csv"), row.names = FALSE)
write.csv(
  threshold_x,
  file.path(output_directory, "threshold_cv_x.csv"),
  row.names = FALSE
)
write.csv(
  threshold,
  file.path(output_directory, "threshold_cv_scores.csv"),
  row.names = FALSE
)
writeLines(
  sub("[[:space:]]+$", "", capture.output(sessionInfo())),
  file.path(output_directory, "cv_session_info.txt")
)
