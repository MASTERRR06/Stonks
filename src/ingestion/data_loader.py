"""
data_loader.py — NSE stock data ingestion module.

Responsibilities:
- Fetch OHLCV data from Alpha Vantage (historical data)
- Load locally cached raw data
- Provide mock data for offline/testing scenarios
- Save raw data to disk

NOTE: No preprocessing or feature engineering here — that lives exclusively
      in notebooks/model_pipeline.ipynb.
"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv

# Load credentials from .env file
load_dotenv()

from src.utils.helpers import (
    get_config,
    setup_logger,
    validate_ohlcv_columns,
    ensure_dir,
)

logger = setup_logger(__name__)
cfg = get_config()

# ── Alpha Vantage base URL ──
AV_BASE_URL = "https://www.alphavantage.co/query"

# ── Interval mapping ──
# Alpha Vantage uses different function names for different intervals
AV_INTERVAL_MAP = {
    "1m": ("TIME_SERIES_INTRADAY", "1min"),
    "5m": ("TIME_SERIES_INTRADAY", "5min"),
    "15m": ("TIME_SERIES_INTRADAY", "15min"),
    "30m": ("TIME_SERIES_INTRADAY", "30min"),
    "1h": ("TIME_SERIES_INTRADAY", "60min"),
    "1d": ("TIME_SERIES_DAILY_ADJUSTED", None),
    "1wk": ("TIME_SERIES_WEEKLY_ADJUSTED", None),
    "1mo": ("TIME_SERIES_MONTHLY_ADJUSTED", None),
}


# ──────────────────────────────────────────────
# Alpha Vantage fetcher
# ──────────────────────────────────────────────


def _fetch_alphavantage(
    ticker: str,
    interval: str = "1d",
    outputsize: str = "full",
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Alpha Vantage.

    Args:
        ticker:     NSE ticker symbol (e.g. 'RELIANCE'). Alpha Vantage uses
                    BSE format for Indian stocks: 'RELIANCE.BSE'
        interval:   Data frequency ('1d', '1wk', '1mo', '1m', '5m', etc.)
        outputsize: 'full' (up to 20 years) or 'compact' (last 100 bars)

    Returns:
        DataFrame with DatetimeIndex and OHLCV columns.
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "Alpha Vantage API key missing. "
            "Please set ALPHAVANTAGE_API_KEY in your .env file. "
            "Get a free key at: https://www.alphavantage.co/support/#api-key"
        )

    # Alpha Vantage uses BSE suffix for Indian stocks
    av_ticker = ticker if "." in ticker else f"{ticker}.BSE"

    function, av_interval = AV_INTERVAL_MAP.get(
        interval, ("TIME_SERIES_DAILY_ADJUSTED", None)
    )

    params = {
        "function": function,
        "symbol": av_ticker,
        "apikey": api_key,
        "outputsize": outputsize,
        "datatype": "json",
    }
    if av_interval:
        params["interval"] = av_interval

    logger.info("Fetching %s from Alpha Vantage | function=%s", av_ticker, function)

    response = requests.get(AV_BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    # Check for API errors
    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
    if "Note" in data:
        raise ValueError(
            "Alpha Vantage API limit reached (25 requests/day on free tier). "
            "Wait or upgrade your plan."
        )
    if "Information" in data:
        raise ValueError(f"Alpha Vantage: {data['Information']}")

    # Parse the time series key from response
    ts_key = [k for k in data.keys() if "Time Series" in k]
    if not ts_key:
        raise ValueError(f"No time series data found in response for '{av_ticker}'.")
    ts_key = ts_key[0]
    ts_data = data[ts_key]

    # Build DataFrame
    df = pd.DataFrame.from_dict(ts_data, orient="index")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Rename columns — Alpha Vantage prefixes with numbers
    col_map = {}
    for col in df.columns:
        if "open" in col.lower():
            col_map[col] = "Open"
        elif "high" in col.lower():
            col_map[col] = "High"
        elif "low" in col.lower():
            col_map[col] = "Low"
        elif "adjusted close" in col.lower() or "adjusted_close" in col.lower():
            col_map[col] = "Adj Close"
        elif "close" in col.lower():
            col_map[col] = "Close"
        elif "volume" in col.lower():
            col_map[col] = "Volume"
    df.rename(columns=col_map, inplace=True)

    # Ensure Adj Close exists
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    # Keep only OHLCV columns
    keep = [
        c
        for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        if c in df.columns
    ]
    df = df[keep].astype(float)
    df.index.name = "Date"

    return df


def _filter_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Filter a full DataFrame down to the requested period."""
    period_map = {
        "1d": timedelta(days=1),
        "5d": timedelta(days=5),
        "1mo": timedelta(days=30),
        "3mo": timedelta(days=90),
        "6mo": timedelta(days=180),
        "1y": timedelta(days=365),
        "2y": timedelta(days=730),
        "5y": timedelta(days=1825),
    }
    delta = period_map.get(period, timedelta(days=730))
    cutoff = pd.Timestamp.now() - delta
    return df[df.index >= cutoff]


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────


def fetch_nse_data(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    save: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for an NSE-listed stock via Alpha Vantage.

    Args:
        ticker:   NSE ticker symbol (e.g. 'RELIANCE', 'TCS').
        period:   Period string ('1d', '5d', '1mo', '3mo',
                  '6mo', '1y', '2y', '5y').
        interval: Data frequency ('1d', '1wk', '1mo',
                  '1m', '5m', '15m', '1h').
        save:     If True, persist raw data to data/raw/.

    Returns:
        DataFrame with DatetimeIndex and columns
        [Open, High, Low, Close, Adj Close, Volume].

    Raises:
        ValueError: If API key is missing or data is empty.
    """
    logger.info("Fetching %s | period=%s | interval=%s", ticker, period, interval)

    df = _fetch_alphavantage(ticker, interval=interval, outputsize="full")
    df = _filter_by_period(df, period)

    if df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. "
            "Check the symbol or try a different period."
        )

    if save:
        _save_raw(df, ticker)

    logger.info("Fetched %d rows for %s", len(df), ticker)
    return df


def load_raw_data(ticker: str) -> pd.DataFrame:
    """Load previously saved raw OHLCV data from disk."""
    raw_dir = Path(cfg["data"]["raw_dir"])
    path = raw_dir / f"{ticker}_raw.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No cached data for '{ticker}' at {path}. " "Call fetch_nse_data() first."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def get_mock_data(
    ticker: str = "MOCK",
    n_rows: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for offline testing."""
    np.random.seed(seed)
    dates = pd.bdate_range(end=datetime.today(), periods=n_rows, freq="B")
    close = 1000 + np.cumsum(np.random.randn(n_rows) * 15)
    close = np.maximum(close, 50)

    df = pd.DataFrame(
        {
            "Open": close * (1 + np.random.uniform(-0.01, 0.01, n_rows)),
            "High": close * (1 + np.random.uniform(0.00, 0.02, n_rows)),
            "Low": close * (1 - np.random.uniform(0.00, 0.02, n_rows)),
            "Close": close,
            "Adj Close": close,
            "Volume": np.random.randint(500_000, 5_000_000, n_rows).astype(float),
        },
        index=dates,
    )
    df.index.name = "Date"
    logger.info("Generated %d rows of mock data for '%s'", n_rows, ticker)
    return df


def load_processed_data() -> pd.DataFrame:
    """Load the feature-engineered dataset produced by the Jupyter notebook."""
    path = Path(cfg["model"]["processed_data_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. "
            "Please run notebooks/model_pipeline.ipynb first."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    logger.info("Loaded processed dataset: %d rows × %d cols", *df.shape)
    return df


def load_news_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load financial news data for sentiment analysis."""
    if filepath:
        df = pd.read_csv(filepath)
        if "headline" not in df.columns:
            raise ValueError("News CSV must contain a 'headline' column.")
        return df

    sample = {
        "headline": [
            "Reliance Industries reports record quarterly profit",
            "HDFC Bank faces regulatory scrutiny over lending practices",
            "Infosys wins $1.5 billion deal with European client",
            "Nifty50 crashes 3% amid global sell-off concerns",
            "TCS Q3 results beat analyst expectations on margin expansion",
            "Adani Group stocks fall sharply after short-seller report",
            "RBI keeps repo rate unchanged, signals accommodative stance",
            "Wipro announces major share buyback programme",
        ],
        "date": pd.date_range(end=datetime.today(), periods=8, freq="D").strftime(
            "%Y-%m-%d"
        ),
    }
    return pd.DataFrame(sample)


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────


def _save_raw(df: pd.DataFrame, ticker: str) -> None:
    """Persist a raw DataFrame to CSV under data/raw/."""
    raw_dir = ensure_dir(cfg["data"]["raw_dir"])
    path = raw_dir / f"{ticker}_raw.csv"
    df.to_csv(path)
    logger.info("Raw data saved → %s", path)
