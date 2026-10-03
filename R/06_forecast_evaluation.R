library(forecast)
library(xts)

# =============================================================
# 1. LOAD DATA
# =============================================================
df <- read.csv("data/processed/eurusd_merged.csv")
df$Date <- as.Date(df$Date)
ts_ret <- xts(df$ret_eurusd, order.by = df$Date)

n <- length(ts_ret)
n_train <- floor(0.9 * n)
train <- ts_ret[1:n_train]
test  <- ts_ret[(n_train+1):n]

# =============================================================
# 2. BASELINE (NAIVE)
# =============================================================
naive_fc <- naive(train, h = length(test))
metrics_naive <- accuracy(naive_fc, test)

# =============================================================
# 3. ARIMA
# =============================================================
arima_fit <- auto.arima(train, seasonal = FALSE)
arima_fc  <- forecast(arima_fit, h = length(test))
metrics_arima <- accuracy(arima_fc, test)

# =============================================================
# 4. ETS
# =============================================================
ets_fit <- ets(train)
ets_fc  <- forecast(ets_fit, h = length(test))
metrics_ets <- accuracy(ets_fc, test)

# =============================================================
# 5. COMPARISON TABLE
# =============================================================
comparison <- rbind(
  Naive  = metrics_naive[2, c("RMSE","MAE","MAPE")],
  ARIMA  = metrics_arima[2, c("RMSE","MAE","MAPE")],
  ETS    = metrics_ets[2,   c("RMSE","MAE","MAPE")]
)
print(round(comparison, 5))
write.csv(comparison, "figures/metrics_comparison.csv")

# =============================================================
# 6. FINAL PLOT
# =============================================================
png("figures/15_final_comparison.png", width = 1200, height = 600)

# Plot the actual series
plot(index(test), as.numeric(test), type = "l", col = "black", lwd = 2,
     main = "Forecast Comparison - EUR/USD (Test Set)",
     xlab = "Date", ylab = "Log-Returns")

# Add forecast lines
lines(index(test), as.numeric(arima_fc$mean), col = "blue",  lwd = 2)
lines(index(test), as.numeric(ets_fc$mean),   col = "red",   lwd = 2)
lines(index(test), as.numeric(naive_fc$mean), col = "green", lwd = 2, lty = 2)

# Legend
legend("topright",
       legend = c("Actual", "ARIMA", "ETS", "Naive"),
       col    = c("black", "blue", "red", "green"),
       lwd    = 2, lty = c(1, 1, 1, 2))

grid()
dev.off()

cat("✅ Plot saved: figures/15_final_comparison.png\n")