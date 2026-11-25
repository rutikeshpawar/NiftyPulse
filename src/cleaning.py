import pandas as pd

from src.config import RAW_DIR, PROCESSED_DIR

MAX_VIX_FILL_DAYS = 3   # forward-fill VIX for at most this many consecutive missing days


def clean_market_data() -> pd.DataFrame:
    """Merge Nifty and VIX raw files into one clean daily table."""
    nifty = pd.read_csv(RAW_DIR / "nifty50_daily.csv", parse_dates=["date"])
    vix = pd.read_csv(RAW_DIR / "indiavix_daily.csv", parse_dates=["date"])

    # 1. Keep only useful columns, with clear prefixes
    #    (volume dropped: unreliable for indices; VIX open dropped: just yesterday's close)
    nifty = nifty[["date", "open", "high", "low", "close"]]
    nifty.columns = ["date", "nifty_open", "nifty_high", "nifty_low", "nifty_close"]

    vix = vix[["date", "high", "low", "close"]]
    vix.columns = ["date", "vix_high", "vix_low", "vix_close"]

    # 2. Remove duplicate dates and sort by date
    nifty = nifty.drop_duplicates("date").sort_values("date")
    vix = vix.drop_duplicates("date").sort_values("date")

    # 3. Nifty is the master calendar; attach VIX to it
    df = nifty.merge(vix, on="date", how="left")

    # 4. Fill short VIX gaps using PAST values only, and flag them
    df["vix_filled"] = df["vix_close"].isna()
    df[["vix_high", "vix_low", "vix_close"]] = (
        df[["vix_high", "vix_low", "vix_close"]].ffill(limit=MAX_VIX_FILL_DAYS)
    )

    # 5. Drop any rows still incomplete (e.g. before VIX starts)
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows that were still incomplete")

    return df


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = clean_market_data()

    # Sanity checks
    assert df["date"].is_monotonic_increasing, "Dates are not sorted"
    assert not df["date"].duplicated().any(), "Duplicate dates found"
    assert not df.isna().any().any(), "Missing values remain"

    out_path = PROCESSED_DIR / "market_clean.csv"
    df.to_csv(out_path, index=False)

    print(f"Rows: {len(df)} | {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"VIX values filled from a previous day: {int(df['vix_filled'].sum())}")
    print(f"Saved -> {out_path}")
    print(df.tail(3))


if __name__ == "__main__":
    main()