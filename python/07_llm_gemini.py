# =============================================================
# Purpose : Use Google Gemini API (new SDK) to generate hypotheses,
#           vulgarized explanations, and investment recommendations
#           for EUR/USD time series analysis.
# =============================================================
# pip install google-genai

import os
import pandas as pd
from google import genai

# -------------------------------------------------------------
# 1. SETUP API KEY
# -------------------------------------------------------------
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("ERROR: GEMINI_API_KEY not set.")
    print("Run in PowerShell: setx GEMINI_API_KEY 'AIzaSy...'")
    print("Then close and reopen VS Code.")
    exit(1)

client = genai.Client(api_key=API_KEY)

# -------------------------------------------------------------
# 2. LOAD DATA STATS
# -------------------------------------------------------------
df = pd.read_csv("data/processed/eurusd_merged.csv", parse_dates=["Date"])
df = df.set_index("Date")

stats = {
    "period":          f"{df.index.min().date()} to {df.index.max().date()}",
    "n_obs":           len(df),
    "mean_price":      float(df["EURUSD"].mean()),
    "std_price":       float(df["EURUSD"].std()),
    "min_price":       float(df["EURUSD"].min()),
    "max_price":       float(df["EURUSD"].max()),
    "skewness_return": float(df["ret_eurusd"].skew()),
    "kurtosis_return": float(df["ret_eurusd"].kurtosis() + 3),
}

print("=== Descriptive Statistics ===\n")
for k, v in stats.items():
    print(f"{k:20s}: {v}")

# -------------------------------------------------------------
# 3. OUTPUT DIRECTORY
# -------------------------------------------------------------
os.makedirs("llm_outputs", exist_ok=True)

# -------------------------------------------------------------
# 4. HELPER FUNCTION
# -------------------------------------------------------------
def ask_gemini(prompt, filename):
    """Send prompt to Gemini and save response."""
    print(f"\n=== Sending to Gemini: {filename} ===\n")
    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash", 
            contents=prompt
)
        text = response.text
        print(text)
        with open(f"llm_outputs/{filename}", "w", encoding="utf-8") as f:
            f.write(text)
        return text
    except Exception as e:
        print(f"ERROR: {e}")
        return None

# -------------------------------------------------------------
# 5. PROMPT 1 — HYPOTHESIS GENERATION
# -------------------------------------------------------------
prompt1 = f"""
You are a time series expert. Context: EUR/USD daily data {stats['period']}
({stats['n_obs']} observations).

Descriptive statistics:
- Mean price: {stats['mean_price']:.4f}
- Std price: {stats['std_price']:.4f}
- Min price: {stats['min_price']:.4f}
- Max price: {stats['max_price']:.4f}
- Skewness of log-returns: {stats['skewness_return']:.4f}
- Kurtosis of log-returns: {stats['kurtosis_return']:.4f}
- ADF on price: non-stationary (p > 0.05)
- ADF on returns: stationary (p < 0.01)
- Correlation EUR/USD vs DXY: strongly negative

Task: Propose 5 time series models to test, ranked by relevance,
with statistical justification and expected limitations.
Keep the answer under 400 words.
"""

text1 = ask_gemini(prompt1, "gemini_hypotheses.txt")

# -------------------------------------------------------------
# 6. PROMPT 2 — VULGARIZED EXPLANATION
# -------------------------------------------------------------
prompt2 = """
You are explaining time series results to a non-specialist.

Here are my ARIMA results on EUR/USD:
- Model: ARIMA(1,1,1)
- ar1 = 0.2358, ma1 = -0.2351
- Ljung-Box p-value = 0.99 (residuals not autocorrelated)
- Jarque-Bera p-value < 0.01 (residuals NOT normal -> fat tails)

Explain these results in 5 sentences maximum. Use simple analogies.
Explicitly mention that EUR/USD predictability is very low and why.
Keep it under 200 words.
"""

text2 = ask_gemini(prompt2, "gemini_explanation.txt")

# -------------------------------------------------------------
# 7. PROMPT 3 — INVESTMENT RECOMMENDATIONS WITH RISKS
# -------------------------------------------------------------
prompt3 = """
Based on my ARIMA-GARCH forecast on EUR/USD:
- Central forecast (10 days ahead): 1.0850
- 95% confidence interval: [1.0720, 1.0980]
- Forecast volatility: 8.5% annualized
- Context: ECB on hold, FED cutting rates

Generate 3 SIMULATED investment recommendations.

MANDATORY for each recommendation:
1. Explicitly state associated risks
2. Specify model limitations (random walk nature, regime changes)
3. Remind that this is NOT real financial advice
4. Mention uncaptured factors (geopolitics, surprise announcements)

Present them as a simple markdown table with columns:
Recommendation | Justification | Risks | Limits
Keep it under 400 words.
"""

text3 = ask_gemini(prompt3, "gemini_recommendations.txt")

# -------------------------------------------------------------
# 8. DONE
# -------------------------------------------------------------
print("\n" + "="*60)
print("All responses saved in llm_outputs/")
print("  - gemini_hypotheses.txt")
print("  - gemini_explanation.txt")
print("  - gemini_recommendations.txt")
print("="*60)