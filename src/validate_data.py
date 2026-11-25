import pandas as pd

from src.config import RAW_DIR

EXPECTED_COLUMNS = ["date", "open", "high", "low", "close", "volume", "source"]
PRICE_COLS = ["open", "high", "low", "close"]

# Daily move size that is "suspicious" for each series
MOVE_LIMIT = {"NIFTY50": 0.07, "INDIAVIX": 0.40}

counts = {"PASS": 0, "WARN": 0, "FAIL": 0}


def record(status: str, message: str):
    counts[status] += 1
    print(f"{status:<5} | {message}")


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / f"{name.lower()}_daily.csv", parse_dates=["date"])


def validate(name: str, df: pd.DataFrame):
    print(f"\n===== {name} =====")
    print(f"Rows: {len(df)} | {df['date'].min().date()} to {df['date'].max().date()}")

    # 1. Columns
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        record("FAIL", f"Missing columns: {missing_cols}")
        return
    record("PASS", "All expected columns are present")

    # 2. Missing values
    nulls = int(df[PRICE_COLS].isna().sum().sum())
    record("PASS" if nulls == 0 else "FAIL", f"Missing price values: {nulls}")

    # 3. Duplicate dates
    dups = int(df["date"].duplicated().sum())
    record("PASS" if dups == 0 else "FAIL", f"Duplicate dates: {dups}")

    # 4. Sorted dates
    record("PASS" if df["date"].is_monotonic_increasing else "FAIL",
           "Dates are in increasing order")

    # 5. Prices above zero
    bad_price = int((df[PRICE_COLS] <= 0).any(axis=1).sum())
    record("PASS" if bad_price == 0 else "FAIL", f"Rows with price <= 0: {bad_price}")

    # 6. OHLC logic
    bad_ohlc = df[
        (df["high"] < df["low"])
        | (df["close"] > df["high"]) | (df["close"] < df["low"])
        | (df["open"] > df["high"]) | (df["open"] < df["low"])
    ]
    if len(bad_ohlc) == 0:
        record("PASS", "OHLC logic is consistent")
    else:
        record("WARN", f"Rows breaking OHLC logic: {len(bad_ohlc)}")
        print(bad_ohlc.head())

    # 7. Large date gaps
    gaps = df["date"].diff().dt.days
    big_gaps = df.loc[gaps > 5, "date"]
    if len(big_gaps) == 0:
        record("PASS", "No date gaps longer than 5 days")
    else:
        record("WARN", f"{len(big_gaps)} gaps longer than 5 days, e.g. before: "
                       f"{[str(d.date()) for d in big_gaps.head(3)]}")

    # 8. Extreme daily moves
    limit = MOVE_LIMIT[name]
    moves = df["close"].pct_change()
    extreme = df.loc[moves.abs() > limit, ["date", "close"]].assign(change=moves)
    if len(extreme) == 0:
        record("PASS", f"No daily move above {limit:.0%}")
    else:
        record("WARN", f"{len(extreme)} daily moves above {limit:.0%} (check they are real events)")
        print(extreme.to_string(index=False))

    # 9. Zero volume (information only)
    zero_vol = int((df["volume"] == 0).sum())
    print(f"INFO  | Rows with volume = 0: {zero_vol} ({zero_vol / len(df):.1%}), "
          "so we won't rely on volume")


def check_alignment(nifty: pd.DataFrame, vix: pd.DataFrame):
    print("\n===== NIFTY50 vs INDIAVIX alignment =====")
    n_dates, v_dates = set(nifty["date"]), set(vix["date"])
    common = len(n_dates & v_dates)
    only_n, only_v = len(n_dates - v_dates), len(v_dates - n_dates)
    print(f"Common dates: {common} | only in Nifty: {only_n} | only in VIX: {only_v}")
    status = "PASS" if common / max(len(n_dates), len(v_dates)) > 0.95 else "WARN"
    record(status, "Most dates match, so an inner join on date is safe")


def main():
    nifty, vix = load("NIFTY50"), load("INDIAVIX")
    validate("NIFTY50", nifty)
    validate("INDIAVIX", vix)
    check_alignment(nifty, vix)
    print(f"\nSUMMARY -> PASS: {counts['PASS']} | WARN: {counts['WARN']} | FAIL: {counts['FAIL']}")


if __name__ == "__main__":
    main()