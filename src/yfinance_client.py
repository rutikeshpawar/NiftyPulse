import pandas as pd
import yfinance as yf

SOURCE_NAME = "yfinance"

# Our friendly names mapped to Yahoo Finance ticker symbols
SYMBOLS = {
    "NIFTY50": "^NSEI",
    "INDIAVIX": "^INDIAVIX",
}

# The common columns every client must return
COLUMNS = ["date", "open", "high", "low", "close", "volume", "source"]


def fetch_history(name: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download daily OHLCV data for 'NIFTY50' or 'INDIAVIX'."""
    if name not in SYMBOLS:
        raise ValueError(f"Unknown symbol '{name}'. Choose from {list(SYMBOLS)}")

    ticker = SYMBOLS[name]
    raw = yf.Ticker(ticker).history(
        start=start, end=end, interval="1d", auto_adjust=False
    )

    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {name} ({ticker})")

    df = raw.reset_index()
    df = df.rename(columns={
        "Date": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })

    # Remove timezone so dates are plain calendar days
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["source"] = SOURCE_NAME

    return df[COLUMNS]
