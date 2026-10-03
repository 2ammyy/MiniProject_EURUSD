library(forecast)
library(xts)
library(tseries)

# =============================================================
# 1. LOAD DATA
# =============================================================
df <- read.csv("data/processed/eurusd_merged.csv")
df$Date <- as.Date(df$Date)
ts_ret <- xts(df$ret_eurusd, order.by = df$Date)

# Train/test split 90/10
n <- length(ts_ret)
n_train <- floor(0.9 * n)
train <- ts_ret[1:n_train]
test  <- ts_ret[(n_train+1):n]

cat("Train size:", length(train), "| Test size:", length(test), "\n\n")

# =============================================================
# 2. AUTO ARIMA
# =============================================================
cat("=== Auto ARIMA ===\n")
fit_arima <- auto.arima(train, seasonal = FALSE, stepwise = FALSE,
                        approximation = FALSE)
print(summary(fit_arima))

# =============================================================
# 3. RESIDUAL DIAGNOSTICS
# =============================================================
cat("\n=== Ljung-Box Test ===\n")
print(Box.test(residuals(fit_arima), lag = 20, type = "Ljung-Box"))

cat("\n=== Jarque-Bera Test ===\n")
library(moments)
res <- as.numeric(residuals(fit_arima))
print(jarque.test(res))

cat("\n=== Shapiro-Wilk Test ===\n")
print(shapiro.test(res))

png("figures/04_arima_residuals.png", width = 1000, height = 800)
checkresiduals(fit_arima)
dev.off()

# =============================================================
# 4. FORECAST
# =============================================================
h <- 10
fc <- forecast(fit_arima, h = h)
print(fc)

png("figures/05_arima_forecast.png", width = 1000, height = 600)
plot(fc, main = "ARIMA Forecast - EUR/USD Log-Returns")
dev.off()

# =============================================================
# 5. SARIMA (on hourly data for intraday seasonality)
# =============================================================
cat("\n=== SARIMA on Hourly Data ===\n")
df_h <- read.csv("data/raw/eurusd_hourly.csv")
# Clean hourly
if ("Close" %in% names(df_h)) {
  df_h$Date <- as.POSIXct(df_h$Date, tz = "UTC")
  ts_h <- xts(df_h$Close, order.by = df_h$Date)
  ts_h_ret <- na.omit(diff(log(ts_h)))
  
  # SARIMA with s=24 (24 hours)
  fit_sarima <- auto.arima(ts_h_ret, seasonal = TRUE, D = 1)
  print(summary(fit_sarima))
  
  png("figures/06_sarima.png", width = 1000, height = 600)
  checkresiduals(fit_sarima)
  dev.off()
}

# =============================================================
# 6. EVALUATION METRICS
# =============================================================
compute_metrics <- function(actual, predicted) {
  mse  <- mean((actual - predicted)^2, na.rm = TRUE)
  rmse <- sqrt(mse)
  mae  <- mean(abs(actual - predicted), na.rm = TRUE)
  mape <- mean(abs((actual - predicted) / actual), na.rm = TRUE) * 100
  return(c(MSE = mse, RMSE = rmse, MAE = mae, MAPE = mape))
}

pred_arima <- as.numeric(forecast(fit_arima, h = length(test))$mean)
metrics_arima <- compute_metrics(as.numeric(test), pred_arima)
cat("\n=== ARIMA Metrics ===\n")
print(metrics_arima)
write.csv(metrics_arima, "figures/metrics_arima.csv")