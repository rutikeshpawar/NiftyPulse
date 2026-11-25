import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGS = ROOT / "reports" / "figures"

STEP = 0.25
REGIME_NAMES = {1: "Calm", 2: "Elevated", 3: "Stressed"}

st.set_page_config(page_title="NiftyPulse", page_icon="📈", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_data
def load_data():
    return pd.read_csv(DATA / "model_dataset.csv", parse_dates=["date"])


@st.cache_resource
def fit_ridge(_df, features, alpha):
    """Production Ridge: refit on all rows whose 5-day future volatility is known."""
    train = _df.dropna(subset=["target_vol_5d"])
    X = train[features]
    y = np.log(train["target_vol_5d"] / train["vix_close"])
    scaler = StandardScaler().fit(X)
    ridge = Ridge(alpha=alpha).fit(scaler.transform(X), y)
    ratio = train["target_vol_5d"].mean() / train["vix_close"].mean()   # VIX-rescaled baseline factor
    return scaler, ridge, ratio


def assign_regime(rows, params):
    x = (rows[params["features"]].to_numpy(float) - np.array(params["mean"])) / np.array(params["scale"])
    centers = np.array(params["centers"])
    nearest = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    return np.array([params["rank"][str(c)] for c in nearest])


def suggested_exposure(forecast, target_vol):
    raw = min(max(target_vol / forecast, 0.0), 1.0)
    return round(raw / STEP) * STEP


# ---------------------------------------------------------------- setup
df = load_data().copy()
features = json.load(open(DATA / "selected_features.json"))["features"]
alpha = json.load(open(DATA / "best_params_vol.json"))["Ridge"]["alpha"]
regime_params = json.load(open(MODELS / "regime_params.json"))

scaler, ridge, ratio = fit_ridge(df, features, alpha)
df["regime"] = assign_regime(df, regime_params)
latest = df.iloc[-1]

x_latest = scaler.transform(df[features].iloc[[-1]])
log_ratio = float(ridge.predict(x_latest)[0])
forecast_ridge = latest["vix_close"] * np.exp(log_ratio)
forecast_vix = latest["vix_close"] * ratio

# ---------------------------------------------------------------- sidebar
st.sidebar.title("NiftyPulse")
choice = st.sidebar.radio("Volatility forecast to use",
                          ["VIX rescaled (default)", "Ridge (challenger)"])
target_vol = st.sidebar.slider("Target volatility (% a year)", 6, 20, 12)
years_back = st.sidebar.slider("Chart history (years)", 1, 11, 4)

forecast = forecast_vix if choice.startswith("VIX") else forecast_ridge
exposure = suggested_exposure(forecast, target_vol)

# ---------------------------------------------------------------- header
st.title("📈 NiftyPulse: Nifty 50 regime and volatility dashboard")
st.warning("Educational research tool. Not investment advice. No orders are placed. "
           "Past results do not guarantee future results.")

age_days = (pd.Timestamp.today().normalize() - latest["date"]).days
if age_days > 4:
    st.info(f"Data is {age_days} days old (latest trading day in file: {latest['date'].date()}). "
            "Refresh with `python -m src.update_data` and redeploy.")

tab_today, tab_regime, tab_why, tab_evidence, tab_about = st.tabs(
    ["Today", "Regimes", "Why this forecast", "Evidence", "About"])

# ---------------------------------------------------------------- tab 1
with tab_today:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data as of", str(latest["date"].date()))
    c2.metric("Nifty 50 close", f"{latest['nifty_close']:,.0f}")
    c3.metric("India VIX", f"{latest['vix_close']:.2f}")
    c4.metric("Market regime", f"{int(latest['regime'])} · {REGIME_NAMES[int(latest['regime'])]}")

    st.subheader("Next 5-day volatility forecast (annualised %)")
    f1, f2, f3 = st.columns(3)
    f1.metric("VIX rescaled (default)", f"{forecast_vix:.1f}%")
    f2.metric("Ridge (challenger)", f"{forecast_ridge:.1f}%")
    f3.metric("Suggested exposure (educational)", f"{exposure:.0%}",
              help="Target volatility ÷ forecast volatility, capped at 100%, in 25% steps.")
    st.caption("The two forecasts can differ. In the sealed 2025-26 test they were statistically "
               "about equal, so the simpler VIX-rescaled forecast is the default.")

    chart_df = df[df["date"] >= df["date"].max() - pd.DateOffset(years=years_back)]
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, 3))
    fig, ax = plt.subplots(figsize=(11, 4))
    for r in (1, 2, 3):
        part = chart_df[chart_df["regime"] == r]
        if len(part):
            ax.scatter(part["date"], part["nifty_close"], s=8, color=colors[r - 1],
                       label=f"{r} · {REGIME_NAMES[r]}")
    ax.set_ylabel("Nifty 50 close")
    ax.legend(markerscale=2)
    st.pyplot(fig)

# ---------------------------------------------------------------- tab 2
with tab_regime:
    st.markdown("Regimes come from **K-Means (k = 3)** on VIX, 21-day volatility, drawdown and trend "
                "measures, fitted on 2015-2022 data. Regime 1 is the calmest.")
    profile = df.groupby("regime").agg(
        days_pct=("date", lambda s: len(s) / len(df) * 100),
        avg_VIX=("vix_close", "mean"),
        avg_vol_21d=("vol_21d", "mean"),
        avg_drawdown_pct=("drawdown", lambda s: s.mean() * 100),
        avg_future_5d_vol=("target_vol_5d", "mean"),
    ).round(2)
    profile.index = [f"{r} · {REGIME_NAMES[r]}" for r in profile.index]
    st.dataframe(profile)
    st.caption("Future volatility rises with the regime number, which is what makes the regimes useful. "
               "Boundaries were learned on 2015-2022 and should be refit periodically.")

# ---------------------------------------------------------------- tab 3
with tab_why:
    st.markdown("How each feature moved today's **Ridge** forecast, relative to the usual ratio of realised "
                "volatility to VIX. For a linear model this equals the SHAP value.")
    contrib = pd.Series(ridge.coef_ * x_latest[0], index=features)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10).iloc[::-1]
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.barh(top.index, top.values, color=["tab:red" if v > 0 else "tab:blue" for v in top.values])
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Push on log(volatility ÷ VIX): red = higher volatility, blue = lower")
    st.pyplot(fig2)
    st.caption(f"Typical ratio ≈ {np.exp(ridge.intercept_):.2f}; today's Ridge ratio = {np.exp(log_ratio):.2f}. "
               "SHAP describes model behaviour, not causes.")
    img = FIGS / "14_shap_volatility_beeswarm.png"
    if img.exists():
        st.image(str(img), caption="Overall SHAP pattern (whole history)")

# ---------------------------------------------------------------- tab 4
with tab_evidence:
    st.subheader("Volatility models: mean absolute error (lower is better)")
    st.table(pd.DataFrame({
        "VIX rescaled": [5.349, 3.657],
        "Ridge": [4.947, 3.691],
    }, index=["Walk-forward 2019-2024 (6 yearly tests)", "Final sealed test 2025-2026"]))

    st.subheader("Next-day direction: ROC-AUC (0.5 = guessing)")
    st.table(pd.DataFrame({
        "Best model AUC": ["0.535", "0.529 (mean of 6)", "0.565"],
        "95% range": ["0.486-0.589", "n/a", "0.510-0.621"],
    }, index=["Validation 2023-24", "Walk-forward 2019-2024", "Final test 2025-2026"]))
    st.caption("Direction skill is weak and unproven, so the app does not issue buy or sell signals.")

    st.subheader("Volatility-targeting back-test (5 bps costs, 12% target)")
    st.table(pd.DataFrame({
        "Strategy": ["Buy and hold", "Ridge targeting", "Buy and hold", "Ridge targeting"],
        "CAGR %": [14.31, 12.51, 4.57, 2.99],
        "Volatility %": [18.36, 13.21, 13.43, 12.35],
        "Sharpe": [0.82, 0.96, 0.40, 0.30],
        "Max drawdown %": [-38.44, -18.20, -15.56, -16.16],
    }, index=["2019-2024", "2019-2024", "2025-2026", "2025-2026"]))
    st.caption("It halved the worst drawdown in 2019-2024 (mostly the 2020 crash) but lagged in the calmer "
               "2025-2026 period: it works like insurance, not a return booster.")

# ---------------------------------------------------------------- tab 5
with tab_about:
    st.markdown("""
**Pipeline:** daily Nifty 50 and India VIX → validation and cleaning → leak-free features (tested) →
date-based split → baselines → Ridge, Random Forest and XGBoost → walk-forward testing → tuning →
sealed final test → SHAP → K-Means regimes → this dashboard.

**Key findings**
- VIX multiplied by a stable factor is a strong 5-day volatility forecaster; ML adds little.
- Next-day direction is weakly predictable at best.
- Regimes separate future risk and persist for weeks.

**Limitations:** daily data only, one market, one major crash in the sample, assumed costs,
the stress regime is lightly tested.

**Future scope:** official Angel One and Upstox data (loader already built), intraday candles,
option-chain features, HAR/GARCH and deep-learning (LSTM/GRU) challengers, rule-based strategies
with paper trading, MLflow tracking, scheduled retraining and drift monitoring.
""")