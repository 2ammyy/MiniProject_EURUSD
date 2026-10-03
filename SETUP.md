# Setup — MiniProject_EURUSD

Everything needed before running the project. Two toolchains: **Python** (data + LSTM/GRU + hybrid) and **R** (EDA, ARIMA/SARIMA, GARCH, VAR, evaluation, report).

---

## 1. Prerequisites (install once)

| Tool | Why | Check |
|---|---|---|
| Python 3.11 or 3.12 | All python scripts. **Do not use 3.14** — TensorFlow has no wheels for it. | `python --version` |
| R 4.3+ | All `R/*.R` scripts | `R --version` |
| RStudio Desktop | Runs the `.R` scripts and renders `report/report.Rmd` | rstudio.com |
| Pandoc | Required by R Markdown to knit `report.Rmd`. Bundled with RStudio, so a separate install is usually unnecessary. | `pandoc --version` |
| Git | Optional, for version control | `git --version` |

---

## 2. Python environment

From the repo root (`MiniProject_EURUSD`):

```powershell
# 2.1 Create + activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2.2 Install all Python packages
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Equivalent one-liner without the requirements file:

```powershell
pip install yfinance pandas numpy matplotlib scikit-learn statsmodels tensorflow jupyterlab ipykernel
```

Verify:

```powershell
python -c "import yfinance, pandas, numpy, matplotlib, sklearn, statsmodels, tensorflow as tf; print('OK', tf.__version__)"
```

> Windows note: if PowerShell blocks the activation script, run
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.

---

## 3. R environment

### 3.1 Packages

Required by `R/02_eda.R` → `R/06_forecast_evaluation.R`:

```r
install.packages(c(
  "xts", "zoo",            # time-series containers
  "forecast", "tseries",   # ARIMA / SARIMA, unit-root & stationarity tests
  "urca",                  # unit root tests (ADF, KPSS, Zivot)
  "tsoutliers",            # outlier detection  <- not yet installed
  "moments",               # skewness / kurtosis
  "ggplot2", "gridExtra",  # plots
  "rugarch",               # GARCH / GJR-GARCH  <- not yet installed
  "vars"                   # VAR / VECM       <- not yet installed
))

# Report generation
install.packages(c("rmarkdown", "knitr"))
```

`rugarch` and `vars` need a C/C++ compiler on Windows. Easiest path is Rtools45:
download from <https://cran.r-project.org/bin/windows/Rtools/> and install
`rtools45-x86_64.exe`, then run `install.packages("pkgbuild"); pkgbuild::has_build_tools(debug = TRUE)`.

Batch install from the command line (no RStudio needed):

```powershell
Rscript -e "install.packages(c('xts','zoo','forecast','tseries','urca','tsoutliers','moments','ggplot2','gridExtra','rugarch','vars','rmarkdown','knitr'), repos='https://cloud.r-project.org')"
```

### 3.2 Project file structure (already present)

```
data/raw/          <- created by 01_import.py
data/processed/    <- created by 01_import.py (eurusd_merged.csv)
figures/           <- PNG plots + metrics_*.csv written by the scripts
```

---

## 4. Run order

All commands are run **from the repo root**, because the scripts use relative paths (`data/...`, `figures/...`).

```powershell
# 4.1 Download + clean + merge data (must run first)
python python\01_import.py

# 4.2 Deep learning models
jupyter lab python\05_lstm_gru.ipynb     # or: jupyter notebook python\05_lstm_gru.ipynb
python python\06_hybrid_arima_lstm.py

# 4.3 R analyses (order matters, each re-reads data/processed)
#   In RStudio: open each file and press Ctrl+Enter / Source.
#   Or from the terminal:
Rscript R\02_eda.R
Rscript R\03_arima_sarima.R
Rscript R\04_garch.R
Rscript R\05_var.R
Rscript R\06_forecast_evaluation.R

# 4.4 Render the report (writes report/report.html)
Rscript -e "rmarkdown::render('report/report.Rmd')"
```

---

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'tensorflow'` | You are on Python 3.14. Create the venv with `python3.12 -m venv .venv`. |
| `ERROR: Package 'rugarch' is not available` / compilation fails | Install Rtools45 (needs a C++ compiler). |
| `pandoc not found` when rendering `report.Rmd` | Install Pandoc and restart RStudio, or render from inside RStudio. |
| `FileNotFoundError: data/processed/eurusd_merged.csv` | You skipped step 4.1, or you ran the script from inside `python/` instead of the repo root. |
| Empty data from Yahoo Finance | Rate-limited or throttled. Wait a few minutes and re-run `01_import.py`. |
