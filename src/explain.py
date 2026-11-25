import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import PROCESSED_DIR, BASE_DIR

FIG_DIR = BASE_DIR / "reports" / "figures"


def save_current_figure(name):
    plt.gcf().savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    pd.set_option("display.width", 200)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(PROCESSED_DIR / "selected_features.json") as f:
        features = json.load(f)["features"]
    with open(PROCESSED_DIR / "best_params_vol.json") as f:
        alpha = json.load(f)["Ridge"]["alpha"]
    df = pd.read_csv(PROCESSED_DIR / "model_dataset.csv", parse_dates=["date"])

    # ================= Volatility model (Ridge) =================
    vol = df.dropna(subset=["target_vol_5d"])
    X = vol[features]
    y = np.log(vol["target_vol_5d"] / vol["vix_close"])

    scaler = StandardScaler().fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=features, index=X.index)
    ridge = Ridge(alpha=alpha).fit(X_scaled, y)

    explainer = shap.LinearExplainer(ridge, X_scaled)
    sv = explainer(X_scaled)
    expl = shap.Explanation(values=sv.values, base_values=sv.base_values,
                            data=X.values, feature_names=features)

    mean_abs = pd.Series(np.abs(expl.values).mean(axis=0), index=features).sort_values(ascending=False)
    print("Top 10 features by average |SHAP| (volatility model):")
    print(mean_abs.head(10).round(4))

    coefs = pd.Series(ridge.coef_, index=features)
    print("\nLargest Ridge coefficients (on standardised features):")
    print(coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(10).round(4))

    shap.plots.beeswarm(expl, max_display=15, show=False)
    save_current_figure("14_shap_volatility_beeswarm")
    shap.plots.bar(expl, max_display=15, show=False)
    save_current_figure("15_shap_volatility_bar")

    # ----- Explain the most recent day (what the app will show) -----
    last = df[features].iloc[[-1]]
    last_scaled = pd.DataFrame(scaler.transform(last), columns=features, index=last.index)
    sv_last = explainer(last_scaled)
    expl_last = shap.Explanation(values=sv_last.values, base_values=sv_last.base_values,
                                 data=last.values, feature_names=features)
    shap.plots.waterfall(expl_last[0], max_display=12, show=False)
    save_current_figure("16_shap_latest_day")

    ratio = float(np.exp(ridge.predict(last_scaled))[0])
    vix_now = float(df["vix_close"].iloc[-1])
    print(f"\nLatest day {df['date'].iloc[-1].date()}: VIX {vix_now:.2f} -> "
          f"predicted ratio {ratio:.2f} -> 5-day volatility forecast {vix_now * ratio:.2f}%")

    # ================= Direction model (XGBoost), for completeness =================
    dr = df.dropna(subset=["target_direction"])
    xgb = XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.02, min_child_weight=20,
                        subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb.fit(dr[features], dr["target_direction"].astype(int))
    sv_dir = shap.TreeExplainer(xgb)(dr[features])
    shap.plots.beeswarm(sv_dir, max_display=12, show=False)
    save_current_figure("17_shap_direction_beeswarm")

    print("\nSaved 4 charts (14-17) to reports/figures/")


if __name__ == "__main__":
    main()