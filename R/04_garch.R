library(rugarch)
library(xts)

# =============================================================
# 1. LOAD DATA
# =============================================================
df <- read.csv("data/processed/eurusd_merged.csv")
df$Date <- as.Date(df$Date)
returns <- xts(df$ret_eurusd * 100, order.by = df$Date)  # in %
returns <- na.omit(returns)

# =============================================================
# 2. ARCH(1) MODEL
# =============================================================
cat("=== ARCH(1) ===\n")
spec_arch1 <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 0)),
  mean.model     = list(armaOrder = c(0, 0))
)
fit_arch1 <- ugarchfit(spec = spec_arch1, data = returns)
print(fit_arch1)

# =============================================================
# 3. ARCH(2) MODEL
# =============================================================
cat("\n=== ARCH(2) ===\n")
spec_arch2 <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(2, 0)),
  mean.model     = list(armaOrder = c(0, 0))
)
fit_arch2 <- ugarchfit(spec = spec_arch2, data = returns)
print(fit_arch2)

# =============================================================
# 4. GARCH(1,1) MODEL
# =============================================================
cat("\n=== GARCH(1,1) ===\n")
spec_garch <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model     = list(armaOrder = c(0, 0))
)
fit_garch <- ugarchfit(spec = spec_garch, data = returns)
print(fit_garch)
cat("\nFull summary:\n")
print(summary(fit_garch))

# =============================================================
# 5. GARCH FORECAST (manual plot)
# =============================================================
garch_fc <- ugarchforecast(fit_garch, n.ahead = 20)

# Extract forecast values
fc_series <- as.numeric(fitted(garch_fc))    # Forecast of the mean (returns)
fc_sigma  <- as.numeric(sigma(garch_fc))     # Forecast of conditional volatility

# Plot manually
png("figures/07_garch_forecast.png", width = 1000, height = 800)
par(mfrow = c(2, 1))

# Top panel: forecast of the series (returns)
plot(1:20, fc_series, type = "b", pch = 19, col = "steelblue",
     main = "GARCH Forecast - Series (next 20 days)",
     xlab = "Horizon (days)", ylab = "Forecast (log-returns %)")
abline(h = 0, lty = 2, col = "gray")
grid()

# Bottom panel: forecast of conditional volatility
plot(1:20, fc_sigma, type = "b", pch = 19, col = "darkred",
     main = "GARCH Forecast - Conditional Volatility",
     xlab = "Horizon (days)", ylab = "Forecast σ_t")
grid()

par(mfrow = c(1, 1))
dev.off()

cat("\n=== GARCH Forecast (next 5 days) ===\n")
print(data.frame(
  Horizon   = 1:5,
  Mean      = round(fc_series[1:5], 6),
  Volatility = round(fc_sigma[1:5], 6)
))
# =============================================================
# 6. DIAGNOSTICS
# =============================================================
# Extract residuals and squared residuals
resid_garch <- residuals(fit_garch)
resid_sq    <- resid_garch^2

png("figures/08_garch_diagnostics.png", width = 1000, height = 800)
par(mfrow = c(2, 2))
plot(resid_garch, main = "GARCH Residuals", col = "steelblue", type = "l")
acf(resid_garch, main = "ACF Residuals")
acf(resid_sq,    main = "ACF Squared Residuals")
hist(resid_garch, breaks = 50, main = "Histogram Residuals", col = "lightblue")
par(mfrow = c(1, 1))
dev.off()

# Ljung-Box on squared residuals
cat("\n=== Ljung-Box on Squared Residuals ===\n")
print(Box.test(resid_sq, lag = 20, type = "Ljung-Box"))

# =============================================================
# 7. EXPORT VOLATILITY
# =============================================================
vol <- sigma(fit_garch)
write.csv(data.frame(Date = index(returns), Volatility = as.numeric(vol)),
          "data/processed/garch_volatility.csv", row.names = FALSE)