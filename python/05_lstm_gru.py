import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

# -------------------------------------------------------------
# Manual scaler (equivalent to MinMaxScaler, no sklearn)
# -------------------------------------------------------------
def minmax_fit(x):
    return x.min(axis=0), x.max(axis=0)

def minmax_transform(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)

def minmax_inverse(x_scaled, xmin, xmax):
    return x_scaled * (xmax - xmin) + xmin

# -------------------------------------------------------------
# Metrics (manual, no sklearn)
# -------------------------------------------------------------
def mse(a, p):  return np.mean((a - p) ** 2)
def rmse(a, p): return np.sqrt(mse(a, p))
def mae(a, p):  return np.mean(np.abs(a - p))
def mape(a, p): return np.mean(np.abs((a - p) / a)) * 100

# -------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------
df = pd.read_csv("data/processed/eurusd_merged.csv", parse_dates=["Date"])
df = df.set_index("Date")
series = df["EURUSD"].values.reshape(-1, 1)
print(f"Series length: {len(series)}")

# -------------------------------------------------------------
# 2. TRAIN/TEST SPLIT (90/10)
# -------------------------------------------------------------
n = len(series)
n_train = int(0.9 * n)
train_raw = series[:n_train]
test_raw  = series[n_train:]

# Manual scaling
xmin, xmax = minmax_fit(train_raw)
train_scaled = minmax_transform(train_raw, xmin, xmax)
test_scaled  = minmax_transform(test_raw, xmin, xmax)

# -------------------------------------------------------------
# 3. CREATE SEQUENCES
# -------------------------------------------------------------
def create_sequences(data, lag=30):
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i-lag:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

LAG = 30
X_train, y_train = create_sequences(train_scaled, LAG)
X_test,  y_test  = create_sequences(test_scaled,  LAG)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test  = X_test.reshape((X_test.shape[0],  X_test.shape[1],  1))
print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

# -------------------------------------------------------------
# 4. BUILD MODELS
# -------------------------------------------------------------
def build_lstm(lag):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lag, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_gru(lag):
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=(lag, 1)),
        Dropout(0.2),
        GRU(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# -------------------------------------------------------------
# 5. TRAIN
# -------------------------------------------------------------
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

print("\n=== Training LSTM ===\n")
model_lstm = build_lstm(LAG)
history_lstm = model_lstm.fit(X_train, y_train,
                              epochs=100, batch_size=32,
                              validation_split=0.2,
                              callbacks=[early_stop], verbose=1)

print("\n=== Training GRU ===\n")
model_gru = build_gru(LAG)
history_gru = model_gru.fit(X_train, y_train,
                            epochs=100, batch_size=32,
                            validation_split=0.2,
                            callbacks=[early_stop], verbose=1)

# -------------------------------------------------------------
# 6. LOSS CURVES
# -------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(history_lstm.history["loss"], label="LSTM Train")
plt.plot(history_lstm.history["val_loss"], label="LSTM Val")
plt.plot(history_gru.history["loss"], label="GRU Train")
plt.plot(history_gru.history["val_loss"], label="GRU Val")
plt.legend()
plt.title("Training Loss Comparison - LSTM vs GRU")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.savefig("figures/12_lstm_gru_loss.png", dpi=120)
plt.close()

# -------------------------------------------------------------
# 7. PREDICT
# -------------------------------------------------------------
pred_lstm = model_lstm.predict(X_test, verbose=0)
pred_gru  = model_gru.predict(X_test,  verbose=0)

pred_lstm_inv = minmax_inverse(pred_lstm, xmin, xmax)
pred_gru_inv  = minmax_inverse(pred_gru,  xmin, xmax)
y_test_inv    = minmax_inverse(y_test.reshape(-1, 1), xmin, xmax)

# -------------------------------------------------------------
# 8. METRICS
# -------------------------------------------------------------
def print_metrics(actual, pred, name):
    m1, m2, m3, m4 = mse(actual, pred), rmse(actual, pred), mae(actual, pred), mape(actual, pred)
    print(f"{name:5s} -> MSE: {m1:.6e} | RMSE: {m2:.6f} | MAE: {m3:.6f} | MAPE: {m4:.4f}%")
    return {"Model": name, "MSE": m1, "RMSE": m2, "MAE": m3, "MAPE": m4}

print("\n=== Metrics ===")
results = [
    print_metrics(y_test_inv, pred_lstm_inv, "LSTM"),
    print_metrics(y_test_inv, pred_gru_inv,  "GRU")
]
pd.DataFrame(results).to_csv("figures/metrics_lstm_gru.csv", index=False)

# -------------------------------------------------------------
# 9. PLOT FORECAST
# -------------------------------------------------------------
plt.figure(figsize=(14, 6))
plt.plot(y_test_inv, label="Actual", color="black")
plt.plot(pred_lstm_inv, label="LSTM", color="blue", alpha=0.7)
plt.plot(pred_gru_inv,  label="GRU",  color="red",  alpha=0.7)
plt.legend()
plt.title("LSTM vs GRU - EUR/USD Forecast (Test Set)")
plt.xlabel("Time")
plt.ylabel("EUR/USD Price")
plt.savefig("figures/13_lstm_gru_forecast.png", dpi=120)
plt.close()

print("\n✅ Done. Files saved in figures/")