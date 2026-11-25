import importlib
import logging
import socket

import pandas as pd

# Don't let a blocked network hang forever: give up on a connection after 15 s
socket.setdefaulttimeout(15)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("data_loader")

# Priority order: first = most preferred
DEFAULT_SOURCES = ["angelone", "upstox", "yfinance"]

# Source name -> module that contains its fetch_history()
MODULES = {
    "angelone": "src.angelone_client",
    "upstox": "src.upstox_client",
    "yfinance": "src.yfinance_client",
}


def get_history(name: str, start: str, end: str | None = None,
                sources: list[str] | None = None) -> pd.DataFrame:
    """Return daily OHLCV for 'NIFTY50' or 'INDIAVIX', trying sources in order."""
    sources = sources or DEFAULT_SOURCES
    errors = {}

    for source in sources:
        try:
            # Import only when needed, so a missing library can't break other sources
            module = importlib.import_module(MODULES[source])
            df = module.fetch_history(name, start, end)
            df = _drop_incomplete_today(df)
            log.info(f"{name}: loaded {len(df)} rows from {source}")
            return df
        except Exception as exc:
            errors[source] = f"{type(exc).__name__}: {exc}"
            log.warning(f"{name}: {source} failed -> {errors[source]}")

    raise RuntimeError(f"All sources failed for {name}: {errors}")


def _drop_incomplete_today(df: pd.DataFrame) -> pd.DataFrame:
    """Today's candle is still forming, so remove it to avoid training on partial data."""
    today = pd.Timestamp.today().normalize()
    return df[df["date"] < today].reset_index(drop=True)