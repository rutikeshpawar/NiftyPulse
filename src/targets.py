import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR

TRADING_DAYS = 252
VOL_WINDOW = 5      # future volatility is measured over the next 5 trading days


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add prediction targets. Each target describes the FUTURE of the row's date."""
    df = df.copy()
    ret = df["nifty_close"].pct_change()          # today's return

    # Task A: next-day direction (1 = up, 0 = down or flat)
    next_ret = ret.shift(-1)                      # tomorrow's return, moved onto today's row
    df["target_next_ret"] = next_ret              # for back-testing ONLY, never a feature
    df["target_direction"] = (next_ret > 0).astype(float)
    df.loc[next_ret.isna(), "target_direction"] = np.nan   # last row: tomorrow unknown

    # Task B: realised volatility over the NEXT 5 days, annualised in %
    df["target_vol_5d"] = (
        ret.rolling(VOL_WINDOW).std().shift(-VOL_WINDOW) * np.sqrt(TRADING_DAYS) * 100
    )
    return df


def main():
    df = pd.read_csv(PROCESSED_DIR / "market_clean.csv", parse_dates=["date"])
    df = add_targets(df)

    # --- Automatic check: does the label really match tomorrow's move? ---
    manual = (df["nifty_close"].shift(-1) > df["nifty_close"]).astype(float)
    valid = df["target_direction"].notna()
    assert (manual[valid] == df.loc[valid, "target_direction"]).all(), "Direction target is misaligned"

    print("Rows without a direction target :", int(df["target_direction"].isna().sum()))
    print("Rows without a volatility target:", int(df["target_vol_5d"].isna().sum()))
    print(f"Share of 'up' days (class balance): {df['target_direction'].mean():.1%}")
    print("\nFuture 5-day volatility (annualised %):")
    print(df["target_vol_5d"].describe().round(2))
    print("\nFirst rows, to eyeball the alignment:")
    print(df[["date", "nifty_close", "target_next_ret", "target_direction"]].head(6))


if __name__ == "__main__":
    main()