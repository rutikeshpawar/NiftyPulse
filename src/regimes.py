import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_DIR, BASE_DIR
from src.split import TRAIN_END

REGIME_FEATURES = ["vix_close", "vol_21d", "drawdown", "dist_sma_200", "ret_63d"]   # vix_close MUST stay first
FIG_DIR = BASE_DIR / "reports" / "figures"
MODEL_DIR = BASE_DIR / "models"


def choose_k(X_train, ks=(3, 4, 5)):
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X_train)
        rows.append({"k": k,
                     "silhouette": silhouette_score(X_train, km.labels_),
                     "davies_bouldin": davies_bouldin_score(X_train, km.labels_)})
    table = pd.DataFrame(rows).set_index("k").round(3)
    return int(table["silhouette"].idxmax()), table


def profile(d):
    g = d.groupby("regime")
    return pd.DataFrame({
        "days_%": g.size() / len(d) * 100,
        "avg_VIX": g["vix_close"].mean(),
        "avg_vol21d": g["vol_21d"].mean(),
        "avg_drawdown_%": g["drawdown"].mean() * 100,
        "avg_next_day_ret_%": g["target_next_ret"].mean() * 100,
        "up_day_share_%": g["target_direction"].mean() * 100,
        "avg_future_vol5d": g["target_vol_5d"].mean(),
    }).round(2)


def avg_spell_length(regime):
    """Average number of consecutive days a regime lasts before switching."""
    spells = (regime != regime.shift()).cumsum()
    runs = regime.groupby(spells).agg(["first", "size"])
    return runs.groupby("first")["size"].mean().round(1)


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR / "model_dataset.csv", parse_dates=["date"])
    train = df[df["date"] <= TRAIN_END]

    scaler = StandardScaler().fit(train[REGIME_FEATURES])          # fit on training years only
    X_train = scaler.transform(train[REGIME_FEATURES])

    k, scores = choose_k(X_train)
    print("Choosing the number of regimes (training period):")
    print(scores)
    print(f"-> chosen k = {k} (highest silhouette)\n")

    km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X_train)

    # Number the regimes from calmest (1) to most stressed (k) by their average VIX
    vix_centroid = scaler.inverse_transform(km.cluster_centers_)[:, 0]
    rank = {int(c): i + 1 for i, c in enumerate(np.argsort(vix_centroid))}
    df["regime"] = [rank[int(c)] for c in km.predict(scaler.transform(df[REGIME_FEATURES]))]

    after = df[df["date"] > TRAIN_END]
    print("REGIME PROFILE: fitting period (to 2022-12)")
    print(profile(df[df["date"] <= TRAIN_END]))
    print("\nREGIME PROFILE: after the fitting period (out-of-sample, 2023 onward)")
    print(profile(after))
    print("\nAverage spell length in days (all data):")
    print(avg_spell_length(df["regime"]))

    # Chart: Nifty coloured by regime
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, k))
    plt.figure(figsize=(12, 5))
    for r in range(1, k + 1):
        sub = df[df["regime"] == r]
        plt.scatter(sub["date"], sub["nifty_close"], s=6, color=colors[r - 1],
                    label=f"Regime {r}" + (" (calmest)" if r == 1 else " (most stressed)" if r == k else ""))
    plt.axvline(pd.Timestamp(TRAIN_END), color="black", linestyle="--", linewidth=1,
                label="end of fitting period")
    plt.title("Nifty 50 coloured by market regime")
    plt.ylabel("Nifty 50 close")
    plt.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "18_regimes.png", dpi=150)
    plt.close()

    # Save labels and the model (the app will reuse them)
    df[["date", "regime"]].to_csv(PROCESSED_DIR / "regimes.csv", index=False)
    joblib.dump({"scaler": scaler, "kmeans": km, "rank": rank, "features": REGIME_FEATURES},
                MODEL_DIR / "regime_model.joblib")
    latest = df.iloc[-1]
    print(f"\nLatest day {latest['date'].date()}: Regime {int(latest['regime'])} of {k}")
    print("Saved chart 18, regimes.csv and models/regime_model.joblib")


if __name__ == "__main__":
    main()