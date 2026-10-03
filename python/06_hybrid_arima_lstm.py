
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

# -------------------------------------------------------------
# Manual helpers (no sklearn)
# -------------------------------------------------------------
def minmax_fit(x):
    return x.min(axis=0), x.max(axis=0)

def minmax_transform(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)

def minmax_inverse(x_scaled, xmin, xmax):
    return x_scaled * (xmax - xmin) + xmin

def mse(a, p):  return float(np.mean((a - p) ** 2))
def rmse(a, p): return float(np.sqrt(mse(a, p)))
def mae(a, p):  return float(np.mean(np.abs(a - p)))
def mape(a, p): return float(np.mean(np.abs((a - p) / a)) * 100)

def print_metrics(actual, pred, name):
    m1, m2, m3, m4 = mse(actual, pred), rmse(actual, pred), mae(actual, pred), mape(actual, pred)
    print(f"{name:15s} -> MSE: {m1:.6e} | RMSE: {m2:.6f} | MAE: {m3:.6f} | MAPE: {m4:.4f}%")
    return {"Model": name, "MSE": m1, "RMSE": m2, "MAE": m3, "MAPE": m4}

# -------------------------------------------------------------
# 1. LOAD DATA — use raw numpy values (no index issues)
# -------------------------------------------------------------
df = pd.read_csv("data/processed/eurusd_merged.csv", parse_dates=["Date"])
df = df.set_index("Date")

series_values = df["EURUSD"].values.astype(float)
n = len(series_values)
n_train = int(0.9 * n)

train_values = series_values[:n_train]
test_values  = series_values[n_train:]
test_index   = df.index[n_train:]

print(f"Total  : {n}")
print(f"Train  : {len(train_values)}")
print(f"Test   : {len(test_values)}")

# -------------------------------------------------------------
# 2. FIT ARIMA ON TRAIN (raw numpy values — no index)
# -------------------------------------------------------------
print("\n=== ARIMA Model ===")
arima_model = ARIMA(train_values, order=(1, 1, 1)).fit()
arima_fitted = arima_model.fittedvalues
arima_resid  = train_values - arima_fitted

print(arima_model.summary())

# -------------------------------------------------------------
# 3. TRAIN LSTM ON ARIMA RESIDUALS
# -------------------------------------------------------------
print("\n=== LSTM on ARIMA Residuals ===")

resid = arima_resid.reshape(-1, 1)
xmin, xmax = minmax_fit(resid)
resid_scaled = minmax_transform(resid, xmin, xmax)

LAG = 20
def create_sequences(data, lag):
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i-lag:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(resid_scaled, LAG)
X = X.reshape((X.shape[0], X.shape[1], 1))

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(LAG, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
model.fit(X, y, epochs=100, batch_size=32,
          validation_split=0.2, callbacks=[early_stop], verbose=1)

# -------------------------------------------------------------
# 4. HYBRID FORECAST
# -------------------------------------------------------------
h = len(test_values)
arima_forecast = np.array(arima_model.forecast(steps=h))

# Predict residuals for the test horizon using recursive LSTM
last_resid = resid_scaled[-LAG:].reshape(1, LAG, 1)
lstm_resid_pred = []
current = last_resid.copy()
for _ in range(h):
    p = model.predict(current, verbose=0)[0, 0]
    lstm_resid_pred.append(p)
    current = np.append(current[:, 1:, :], [[[p]]], axis=1)

lstm_resid_pred = minmax_inverse(np.array(lstm_resid_pred).reshape(-1, 1), xmin, xmax).flatten()
hybrid_forecast = arima_forecast + lstm_resid_pred

# -------------------------------------------------------------
# 5. METRICS
# -------------------------------------------------------------
print("\n=== Metrics ===")
results = [
    print_metrics(test_values, arima_forecast,  "ARIMA"),
    print_metrics(test_values, hybrid_forecast, "ARIMA-LSTM")
]
pd.DataFrame(results).to_csv("figures/metrics_hybrid.csv", index=False)

# -------------------------------------------------------------
# 6. PLOT
# -------------------------------------------------------------
plt.figure(figsize=(14, 6))
plt.plot(test_index, test_values, label="Actual", color="black", lw=2)
plt.plot(test_index, arima_forecast, label="ARIMA", color="blue", alpha=0.7)
plt.plot(test_index, hybrid_forecast, label="ARIMA-LSTM Hybrid", color="red", alpha=0.8)
plt.legend()
plt.title("Hybrid ARIMA-LSTM Forecast - EUR/USD (Test Set)")
plt.xlabel("Date")
plt.ylabel("EUR/USD Price")
plt.tight_layout()
plt.savefig("figures/14_hybrid_forecast.png", dpi=120)
plt.close()

print("\n✅ Done. Files saved in figures/")