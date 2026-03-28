"""
data_loader.py — NSE stock data ingestion module.

Responsibilities:
- Fetch OHLCV data from Yahoo Finance (NSE tickers use the '.NS' suffix)
- Load locally cached raw data
- Provide mock data for offline/testing scenarios
- Save raw data to disk

NOTE: No preprocessing or feature engineering here — that lives exclusively
      in notebooks/model_pipeline.ipynb.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from src.utils.helpers import get_config, setup_logger, validate_ohlcv_columns, ensure_dir

logger = setup_logger(__name__)
cfg = get_config()


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
    Fetch OHLCV data for an NSE-listed stock via Yahoo Finance.

    Args:
        ticker:   NSE ticker without suffix (e.g. 'RELIANCE'). The '.NS'
                  suffix is appended automatically.
        period:   yfinance period string ('1d', '5d', '1mo', '3mo',
                  '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max').
        interval: Data frequency ('1d', '1wk', '1mo').
        save:     If True, persist raw data to data/raw/.

    Returns:
        DataFrame with DatetimeIndex and columns [Open, High, Low, Close,
        Adj Close, Volume].

    Raises:
        ImportError: If yfinance is not installed.
        ValueError:  If the downloaded DataFrame is empty.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required: pip install yfinance") from exc

    yf_ticker = ticker if ticker.endswith(".NS") else f"{ticker}.NS"
    logger.info("Fetching %s | period=%s | interval=%s", yf_ticker, period, interval)

    df = yf.download(yf_ticker, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{yf_ticker}'. "
                         "Check the symbol or try a different period.")

    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    if save:
        _save_raw(df, ticker)

    logger.info("Fetched %d rows for %s", len(df), yf_ticker)
    return df


def load_raw_data(ticker: str) -> pd.DataFrame:
    """
    Load previously saved raw OHLCV data from disk.

    Args:
        ticker: NSE ticker symbol (without '.NS').

    Returns:
        DataFrame with DatetimeIndex.

    Raises:
        FileNotFoundError: If no cached file exists for the ticker.
    """
    raw_dir = Path(cfg["data"]["raw_dir"])
    path = raw_dir / f"{ticker}_raw.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No cached data for '{ticker}' at {path}. "
            "Call fetch_nse_data() first."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def get_mock_data(
    ticker: str = "MOCK",
    n_rows: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for offline testing.

    The price series follows a random-walk with realistic volatility.

    Args:
        ticker:  Label used as the ticker name.
        n_rows:  Number of trading-day rows to generate.
        seed:    Random seed for reproducibility.

    Returns:
        DataFrame with DatetimeIndex and OHLCV columns.
    """
    np.random.seed(seed)
    dates = pd.bdate_range(end=datetime.today(), periods=n_rows, freq="B")
    close = 1000 + np.cumsum(np.random.randn(n_rows) * 15)
    close = np.maximum(close, 50)  # prevent negative prices

    df = pd.DataFrame(
        {
            "Open":   close * (1 + np.random.uniform(-0.01, 0.01, n_rows)),
            "High":   close * (1 + np.random.uniform(0.00, 0.02, n_rows)),
            "Low":    close * (1 - np.random.uniform(0.00, 0.02, n_rows)),
            "Close":  close,
            "Adj Close": close,
            "Volume": np.random.randint(500_000, 5_000_000, n_rows).astype(float),
        },
        index=dates,
    )
    df.index.name = "Date"
    logger.info("Generated %d rows of mock data for '%s'", n_rows, ticker)
    return df


def load_processed_data() -> pd.DataFrame:
    """
    Load the feature-engineered dataset produced by the Jupyter notebook.

    Returns:
        DataFrame with technical indicators and lag features.

    Raises:
        FileNotFoundError: If the notebook has not been run yet.
    """
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
    """
    Load financial news/filings text data for sentiment analysis.

    Args:
        filepath: Optional path to a CSV with columns ['headline', 'date'].
                  If None, returns a small built-in sample.

    Returns:
        DataFrame with at least a 'headline' column.
    """
    if filepath:
        df = pd.read_csv(filepath)
        if "headline" not in df.columns:
            raise ValueError("News CSV must contain a 'headline' column.")
        return df

    # Built-in sample — useful for demos and API smoke-tests
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
