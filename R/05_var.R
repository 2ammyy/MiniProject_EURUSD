library(vars)
library(xts)
library(tseries)

# =============================================================
# 1. LOAD DATA
# =============================================================
df <- read.csv("data/processed/eurusd_merged.csv")
df$Date <- as.Date(df$Date)

# Build multivariate series (log-returns)
var_data <- ts.union(
  EURUSD = ts(df$ret_eurusd),
  DXY    = ts(df$ret_dxy),
  US10Y  = ts(df$ret_us10y)
)
var_data <- na.omit(var_data)

cat("VAR data dimensions:", dim(var_data), "\n\n")

# =============================================================
# 2. STATIONARITY TESTS
# =============================================================
cat("=== ADF Tests ===\n")
for (i in 1:ncol(var_data)) {
  cat(colnames(var_data)[i], ": ")
  print(adf.test(var_data[, i])$p.value)
}

# =============================================================
# 3. LAG SELECTION
# =============================================================
cat("\n=== Lag Selection ===\n")
lag_sel <- VARselect(var_data, lag.max = 10, type = "const")
print(lag_sel$selection)

# Choose lag based on BIC
p_opt <- lag_sel$selection["BIC(n)"]
if (is.na(p_opt) || p_opt < 1) p_opt <- 2
cat("Selected lag (BIC):", p_opt, "\n\n")

# =============================================================
# 4. ESTIMATE VAR
# =============================================================
var_model <- VAR(var_data, p = p_opt, type = "const")
print(summary(var_model))

# =============================================================
# 5. STABILITY CHECK
# =============================================================
cat("\n=== Stability ===\n")
roots <- roots(var_model)
cat("Roots modulus:", round(Mod(roots), 3), "\n")
cat("Stable:", all(Mod(roots) < 1), "\n\n")

# =============================================================
# 6. GRANGER CAUSALITY
# =============================================================
cat("=== Granger Causality ===\n")
cat("\nDXY -> EURUSD:\n")
print(causality(var_model, cause = "DXY"))

cat("\nUS10Y -> EURUSD:\n")
print(causality(var_model, cause = "US10Y"))

cat("\nEURUSD -> DXY:\n")
print(causality(var_model, cause = "EURUSD"))

# =============================================================
# 7. RESIDUAL DIAGNOSTICS
# =============================================================
cat("\n=== Residual Diagnostics ===\n")
# Serial correlation
print(serial.test(var_model, lags.pt = 20, type = "PT.asymptotic"))

# Normality
print(normality.test(var_model))

# ARCH effects
print(arch.test(var_model, lags.multi = 5))

# =============================================================
# 8. FORECAST
# =============================================================
var_fc <- predict(var_model, n.ahead = 10, ci = 0.95)
png("figures/09_var_forecast.png", width = 1200, height = 800)
plot(var_fc)
dev.off()

# =============================================================
# 9. IRF (Impulse Response Functions)
# =============================================================
irf_res <- irf(var_model, n.ahead = 20, boot = TRUE, ci = 0.95)
png("figures/10_irf.png", width = 1200, height = 800)
plot(irf_res)
dev.off()

# =============================================================
# 10. FEVD (Forecast Error Variance Decomposition)
# =============================================================
fevd_res <- fevd(var_model, n.ahead = 20)
png("figures/11_fevd.png", width = 1200, height = 800)
plot(fevd_res)
dev.off()
print(fevd_res)

# =============================================================
# SAVE PLOTS
# =============================================================
# VAR Forecast
png("figures/09_var_forecast.png", width = 1200, height = 800)
plot(predict(var_model, n.ahead = 10))
dev.off()

# IRF
png("figures/10_irf.png", width = 1200, height = 800)
plot(irf(var_model, n.ahead = 20, boot = TRUE, ci = 0.95))
dev.off()

# FEVD
png("figures/11_fevd.png", width = 1200, height = 800)
plot(fevd(var_model, n.ahead = 20))
dev.off()

# =============================================================
# SAVE PLOTS
# =============================================================
# VAR Forecast
png("figures/09_var_forecast.png", width = 1200, height = 800)
plot(predict(var_model, n.ahead = 10))
dev.off()

# IRF
png("figures/10_irf.png", width = 1200, height = 800)
plot(irf(var_model, n.ahead = 20, boot = TRUE, ci = 0.95))
dev.off()

# FEVD
png("figures/11_fevd.png", width = 1200, height = 800)
plot(fevd(var_model, n.ahead = 20))
dev.off()