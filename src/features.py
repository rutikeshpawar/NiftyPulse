import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.targets import add_targets

TRADING_DAYS = 252

# The ONLY columns a model is allowed to use as inputs
FEATURE_COLUMNS = [
    # returns and momentum
    "ret_1d", "abs_ret_1d", "ret_lag_1", "ret_lag_2",
    "ret_5d", "ret_10d", "ret_21d", "ret_63d",
    # candle shape
    "gap_pct", "range_pct", "body_pct", "close_pos",
    # trend
    "dist_sma_20", "dist_sma_50", "dist_sma_200", "sma50_vs_sma200",
    "drawdown", "dist_126d_high",
    # volatility
    "vol_5d", "vol_10d", "vol_21d", "vol_ratio_5_21", "atr_14_pct",
    # momentum oscillator
    "rsi_14",
    # VIX
    "vix_close", "vix_chg_1d", "vix_chg_5d", "vix_vs_ma20",
    "vix_z_63", "vix_range_pct", "vix_minus_vol21",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features. Every value on row t uses data from day t and earlier only."""
    df = df.copy()
    close, high, low, open_ = df["nifty_close"], df["nifty_high"], df["nifty_low"], df["nifty_open"]
    prev_close = close.shift(1)
    ret = close.pct_change()

    # ---- Returns and momentum ----
    df["ret_1d"] = ret
    df["abs_ret_1d"] = ret.abs()
    df["ret_lag_1"] = ret.shift(1)
    df["ret_lag_2"] = ret.shift(2)
    for n in (5, 10, 21, 63):
        df[f"ret_{n}d"] = close / close.shift(n) - 1

    # ---- Candle shape ----
    df["gap_pct"] = open_ / prev_close - 1
    df["range_pct"] = (high - low) / close
    df["body_pct"] = (close - open_) / open_
    day_range = (high - low).replace(0, np.nan)
    df["close_pos"] = ((close - low) / day_range).fillna(0.5)

    # ---- Trend ----
    for n in (20, 50, 200):
        df[f"dist_sma_{n}"] = close / close.rolling(n).mean() - 1
    df["sma50_vs_sma200"] = close.rolling(50).mean() / close.rolling(200).mean() - 1
    df["drawdown"] = close / close.cummax() - 1
    df["dist_126d_high"] = close / close.rolling(126).max() - 1

    # ---- Volatility ----
    for n in (5, 10, 21):
        df[f"vol_{n}d"] = ret.rolling(n).std() * np.sqrt(TRADING_DAYS) * 100
    df["vol_ratio_5_21"] = df["vol_5d"] / df["vol_21d"]

    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    df["atr_14_pct"] = true_range.rolling(14).mean() / close * 100

    # ---- RSI (Wilder's smoothing) ----
    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    df["rsi_14"] = 100 - 100 / (1 + avg_gain / avg_loss)

    # ---- VIX ----
    vix = df["vix_close"]
    df["vix_chg_1d"] = vix.pct_change()
    df["vix_chg_5d"] = vix.pct_change(5)
    df["vix_vs_ma20"] = vix / vix.rolling(20).mean() - 1
    df["vix_z_63"] = (vix - vix.rolling(63).mean()) / vix.rolling(63).std()
    df["vix_range_pct"] = (df["vix_high"] - df["vix_low"]) / vix
    df["vix_minus_vol21"] = vix - df["vol_21d"]

    return df


def build_dataset() -> pd.DataFrame:
    """Clean data -> targets -> features -> drop warm-up rows."""
    df = pd.read_csv(PROCESSED_DIR / "market_clean.csv", parse_dates=["date"])
    df = add_targets(df)
    df = add_features(df)

    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    print(f"Warm-up rows dropped (not enough history yet): {before - len(df)}")
    return df


def main():
    df = build_dataset()

    # ---- Automatic checks ----
    assert not df[FEATURE_COLUMNS].isna().any().any(), "Missing values in features"
    assert not np.isinf(df[FEATURE_COLUMNS]).any().any(), "Infinite values in features"
    assert not any(c.startswith("target_") for c in FEATURE_COLUMNS), "A target is used as a feature"

    out_path = PROCESSED_DIR / "model_dataset.csv"
    df.to_csv(out_path, index=False)

    print(f"Rows: {len(df)} | Features: {len(FEATURE_COLUMNS)} | "
          f"{df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Saved -> {out_path}\n")
    print(df[FEATURE_COLUMNS].describe().T[["mean", "std", "min", "max"]].round(3))


if __name__ == "__main__":
    main()