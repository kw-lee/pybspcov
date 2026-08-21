args <- commandArgs(trailingOnly = TRUE)
output_directory <- if (length(args)) args[[1]] else "tests/fixtures/r/bspcov-1.0.3"
dir.create(output_directory, recursive = TRUE, showWarnings = FALSE)

records <- data.frame(
  symbol = c("AAA", "AAA", "AAA", "BBB", "BBB", "BBB"),
  date = as.Date(c(
    "2020-01-01", "2020-02-01", "2020-03-01",
    "2020-01-01", "2020-02-01", "2020-03-01"
  )),
  adjusted = c(10, 11, 12.1, 20, 18, 19.8),
  sector = c("Tech", "Tech", "Tech", "Energy", "Energy", "Energy")
)

symbols <- unique(records[c("sector", "symbol")])
symbols <- symbols[order(symbols$sector, symbols$symbol), ]
monthly <- lapply(symbols$symbol, function(symbol) {
  rows <- records[records$symbol == symbol, ]
  prices <- xts::xts(rows$adjusted, order.by = rows$date)
  result <- quantmod::periodReturn(prices, period = "monthly")
  data.frame(month = as.Date(zoo::index(result)), value = as.numeric(result))
})
names(monthly) <- symbols$symbol

aligned <- Reduce(function(left, right) merge(left, right, by = "month"), monthly)
colnames(aligned) <- c("month", symbols$symbol)
returns <- scale(as.matrix(aligned[-1]), center = TRUE, scale = FALSE)
decomposition <- svd(returns)
factor_part <- decomposition$u[, 1, drop = FALSE] %*%
  (decomposition$d[1] * t(decomposition$v[, 1, drop = FALSE]))
residuals <- returns - factor_part

write.csv(returns, file.path(output_directory, "sp500_fixed_returns.csv"), row.names = FALSE)
write.csv(factor_part, file.path(output_directory, "sp500_fixed_factor.csv"), row.names = FALSE)
write.csv(residuals, file.path(output_directory, "sp500_fixed_residuals.csv"), row.names = FALSE)
write.csv(symbols, file.path(output_directory, "sp500_fixed_columns.csv"), row.names = FALSE)
writeLines(
  sub("[[:space:]]+$", "", capture.output(sessionInfo())),
  file.path(output_directory, "sp500_session_info.txt")
)
