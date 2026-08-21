args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("usage: export_package_datasets.R BSPCOV_SOURCE OUTPUT_DIRECTORY")
}

source_directory <- normalizePath(args[[1]], mustWork = TRUE)
output_directory <- normalizePath(args[[2]], mustWork = TRUE)

load(file.path(source_directory, "data", "colon.rda"))
load(file.path(source_directory, "data", "tissues.rda"))
load(file.path(source_directory, "data", "SP500.rda"))

write.csv(colon, file.path(output_directory, "colon.csv"), row.names = FALSE)
write.csv(
  data.frame(tissues = tissues),
  file.path(output_directory, "tissues.csv"),
  row.names = FALSE
)
write.csv(SP500, file.path(output_directory, "SP500.csv"), row.names = FALSE)
