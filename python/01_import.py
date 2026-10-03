# pip install yfinance pandas numpy

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

# Create directories
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

# =============================================================
# 1. DOWNLOAD DATA
# =============================================================
print("Downloading data from Yahoo Finance...\n")

# EUR/USD daily
eurusd = yf.download("EURUSD=X", start="2015-01-01", end="2025-11-01",
                     interval="1d", auto_adjust=False, progress=False)
eurusd.to_csv("data/raw/eurusd_daily.csv")
print(f"EUR/USD daily : {eurusd.shape[0]} rows")

# =============================================================
# EUR/USD hourly (for SARIMA intraday)
# =============================================================
# Yahoo Finance limits hourly data to the last 730 days
from datetime import datetime, timedelta

end_dt = datetime.now()
start_dt = end_dt - timedelta(days=720)  # safe margin under 730

print(f"Downloading EUR/USD hourly from {start_dt.date()} to {end_dt.date()}...")

eurusd_h = yf.download(
    "EURUSD=X",
    start=start_dt.strftime("%Y-%m-%d"),
    end=end_dt.strftime("%Y-%m-%d"),
    interval="1h",
    auto_adjust=False,
    progress=False
)
eurusd_h.to_csv("data/raw/eurusd_hourly.csv")
print(f"✅ EUR/USD hourly : {eurusd_h.shape[0]} rows")
# DXY (Dollar Index)
dxy = yf.download("DX-Y.NYB", start="2015-01-01", end="2025-11-01",
                  interval="1d", auto_adjust=False, progress=False)
dxy.to_csv("data/raw/dxy_daily.csv")
print(f"DXY : {dxy.shape[0]} rows")

# US 10Y Treasury yield (proxy for FED rate)
tnx = yf.download("^TNX", start="2015-01-01", end="2025-11-01",
                  interval="1d", auto_adjust=False, progress=False)
tnx.to_csv("data/raw/tnx_daily.csv")
print(f"US 10Y : {tnx.shape[0]} rows")

# =============================================================
# 2. CLEANING
# =============================================================
print("\nCleaning data...\n")

def clean_yf(df, col_name):
    """Clean yfinance DataFrame: keep Close, rename, drop NaN."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].copy()
    df.columns = [col_name]
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    return df

eurusd_c = clean_yf(eurusd, "EURUSD")
dxy_c    = clean_yf(dxy,    "DXY")
tnx_c    = clean_yf(tnx,    "US10Y")

# =============================================================
# 3. MERGE
# =============================================================
print("Merging series...\n")

df = eurusd_c.join([dxy_c, tnx_c], how="outer")
df = df.dropna()

# =============================================================
# 4. FEATURE ENGINEERING
# =============================================================
print("Creating features...\n")

# Log-returns
df["ret_eurusd"] = np.log(df["EURUSD"] / df["EURUSD"].shift(1))
df["ret_dxy"]    = np.log(df["DXY"]    / df["DXY"].shift(1))
df["ret_us10y"]  = np.log(df["US10Y"]  / df["US10Y"].shift(1))

# Rolling volatility (30 days, annualized)
df["vol_30d"] = df["ret_eurusd"].rolling(window=30).std() * np.sqrt(252)

# Spread EUR/USD - DXY
df["spread"] = df["EURUSD"] - df["DXY"]

df = df.dropna()

# =============================================================
# 5. EXPORT
# =============================================================
df.to_csv("data/processed/eurusd_merged.csv")
print(f"Merged file: data/processed/eurusd_merged.csv")
print(f"Final shape: {df.shape}")
print(f"Period: {df.index.min().date()} -> {df.index.max().date()}")
print(f"\nPreview:\n{df.head()}")
print(f"\nDescriptive stats:\n{df.describe()}")