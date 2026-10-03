# install.packages(c("xts","zoo","forecast","tseries","ggplot2",
#                    "gridExtra","urca","tsoutliers"))

library(xts)
library(zoo)
library(forecast)
library(tseries)
library(ggplot2)
library(gridExtra)
library(urca)
library(tsoutliers)

# =============================================================
# 1. RELOAD DATA
# =============================================================
df <- read.csv("data/processed/eurusd_merged.csv", stringsAsFactors = FALSE)
df$Date <- as.Date(df$Date)
rownames(df) <- df$Date

ts_eurusd <- xts(df$EURUSD,     order.by = df$Date)
ts_dxy    <- xts(df$DXY,        order.by = df$Date)
ts_us10y  <- xts(df$US10Y,      order.by = df$Date)
ts_ret    <- xts(df$ret_eurusd, order.by = df$Date)

cat("Period:", format(start(ts_eurusd)), "->", format(end(ts_eurusd)), "\n")
cat("Observations:", nrow(df), "\n\n")

# =============================================================
# 2. VISUALIZATION
# =============================================================
png("figures/01_series.png", width = 1200, height = 900, res = 100)
par(mfrow = c(3,1), mar = c(4,5,3,2))

plot(index(ts_eurusd), as.numeric(ts_eurusd), type = "l",
     col = "steelblue", lwd = 1.2,
     main = "EUR/USD (Close)", xlab = "", ylab = "Price")
grid()

plot(index(ts_dxy), as.numeric(ts_dxy), type = "l",
     col = "darkorange", lwd = 1.2,
     main = "US Dollar Index (DXY)", xlab = "", ylab = "Index")
grid()

plot(index(ts_us10y), as.numeric(ts_us10y), type = "l",
     col = "darkred", lwd = 1.2,
     main = "US 10Y Treasury Yield", xlab = "Date", ylab = "%")
grid()

par(mfrow = c(1,1))
dev.off()
# =============================================================
# 3. DESCRIPTIVE STATISTICS
# =============================================================
cat("=== Descriptive Statistics ===\n")
print(summary(ts_eurusd))
cat("\nSkewness:", skewness(ts_eurusd), "\n")
cat("Kurtosis:", kurtosis(ts_eurusd), "\n\n")

# =============================================================
# 4. STATIONARITY TEST (ADF)
# =============================================================
cat("=== ADF Test on Price ===\n")
print(adf.test(ts_eurusd))

cat("\n=== ADF Test on Log-Returns ===\n")
print(adf.test(na.omit(ts_ret)))

# =============================================================
# 5. ACF / PACF
# =============================================================
png("figures/02_acf_pacf.png", width = 1000, height = 600)
par(mfrow = c(1,2))
acf(na.omit(ts_ret),  lag.max = 40, main = "ACF - EUR/USD Log-Returns")
pacf(na.omit(ts_ret), lag.max = 40, main = "PACF - EUR/USD Log-Returns")
dev.off()

# =============================================================
# 6. STL DECOMPOSITION
# =============================================================
# Extract only EURUSD column as a univariate series
ts_sub <- window(ts_eurusd, start = "2023-01-01")
ts_sub_univ <- ts(as.numeric(ts_sub), frequency = 252)
decomp <- stl(ts_sub_univ, s.window = "periodic")
png("figures/03_decomposition.png", width = 1000, height = 800)
plot(decomp, main = "STL Decomposition - EUR/USD")
dev.off()

# =============================================================
# 7. OUTLIER DETECTION (fast version)
# =============================================================
cat("\n=== Outlier Detection ===\n")

# Fast outlier detection using z-scores on log-returns
ret_vec <- na.omit(as.numeric(ts_ret))
z_scores <- abs((ret_vec - mean(ret_vec)) / sd(ret_vec))
outlier_idx <- which(z_scores > 4)   # outliers beyond 4 std devs

cat("Number of outliers detected (|z| > 4):", length(outlier_idx), "\n\n")

if (length(outlier_idx) > 0) {
  outlier_dates <- index(ts_ret)[outlier_idx]
  outlier_df <- data.frame(
    Date    = outlier_dates,
    Return  = round(ret_vec[outlier_idx], 6),
    Z_score = round(z_scores[outlier_idx], 2)
  )
  print(outlier_df)
  write.csv(outlier_df, "figures/outliers.csv", row.names = FALSE)
}

# =============================================================
# 8. CORRELATIONS
# =============================================================
cat("\n=== Correlations ===\n")
cor_matrix <- cor(na.omit(cbind(ts_ret, diff(ts_dxy), diff(ts_us10y))))
colnames(cor_matrix) <- rownames(cor_matrix) <- c("EURUSD_ret","DXY_ret","US10Y_ret")
print(round(cor_matrix, 3))