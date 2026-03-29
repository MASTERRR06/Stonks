"""
filings.py — Corporate filings, insider trades, and bulk/block deal scraper.

Data sources (all free, no API key required):
- NSE bulk/block deals: https://www.nseindia.com
- BSE corporate announcements: https://www.bseindia.com
- NSE insider trading filings: https://www.nseindia.com

Responsibilities:
- Fetch bulk and block deals from NSE
- Fetch insider trading disclosures
- Fetch corporate announcements (results, dividends, splits)
- Parse and return structured DataFrames
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from src.utils.helpers import get_config, setup_logger

logger = setup_logger(__name__)
cfg = get_config()

# ── Common headers to bypass NSE/BSE blocks ──
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

BSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.bseindia.com/",
}


def _get_nse_session() -> requests.Session:
    """
    Create a requests session with NSE cookies.
    NSE requires a cookie from the homepage before API calls work.
    """
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        # Visit homepage first to get cookies
        session.get("https://www.nseindia.com", timeout=10)
    except Exception as exc:
        logger.warning("Could not initialise NSE session: %s", exc)
    return session


# ──────────────────────────────────────────────
# Bulk & Block Deals
# ──────────────────────────────────────────────

def fetch_bulk_deals(date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch bulk deals from NSE for a given date.

    Bulk deals are trades where quantity > 0.5% of total shares.

    Args:
        date: Date string in 'YYYY-MM-DD' format. Defaults to today.

    Returns:
        DataFrame with columns:
          - 'date', 'symbol', 'name', 'client', 'buy_sell',
            'quantity', 'price'
    """
    if date is None:
        date = datetime.today().strftime("%d-%m-%Y")
    else:
        date = datetime.strptime(date, "%Y-%m-%d").strftime("%d-%m-%Y")

    url = f"https://www.nseindia.com/api/bulk-deal-data?date={date}"
    session = _get_nse_session()

    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        if not data or "data" not in data:
            logger.info("No bulk deals found for %s", date)
            return _empty_bulk_df()

        df = pd.DataFrame(data["data"])
        df = df.rename(columns={
            "BD_DT_DATE":     "date",
            "BD_SYMBOL":      "symbol",
            "BD_SCRIP_NAME":  "name",
            "BD_CLIENT_NAME": "client",
            "BD_BUY_SELL":    "buy_sell",
            "BD_QTY_TRD":     "quantity",
            "BD_TP_WATP":     "price",
        })
        keep = ["date", "symbol", "name", "client", "buy_sell", "quantity", "price"]
        df = df[[c for c in keep if c in df.columns]]
        logger.info("Fetched %d bulk deals for %s", len(df), date)
        return df

    except Exception as exc:
        logger.warning("Bulk deals fetch failed: %s", exc)
        return _empty_bulk_df()


def fetch_block_deals(date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch block deals from NSE for a given date.

    Block deals are large trades executed in the block deal window (8:45-9:00 AM).

    Args:
        date: Date string in 'YYYY-MM-DD' format. Defaults to today.

    Returns:
        DataFrame with columns:
          - 'date', 'symbol', 'name', 'client', 'buy_sell',
            'quantity', 'price'
    """
    if date is None:
        date = datetime.today().strftime("%d-%m-%Y")
    else:
        date = datetime.strptime(date, "%Y-%m-%d").strftime("%d-%m-%Y")

    url = f"https://www.nseindia.com/api/block-deal-data?date={date}"
    session = _get_nse_session()

    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        if not data or "data" not in data:
            logger.info("No block deals found for %s", date)
            return _empty_bulk_df()

        df = pd.DataFrame(data["data"])
        df = df.rename(columns={
            "BD_DT_DATE":     "date",
            "BD_SYMBOL":      "symbol",
            "BD_SCRIP_NAME":  "name",
            "BD_CLIENT_NAME": "client",
            "BD_BUY_SELL":    "buy_sell",
            "BD_QTY_TRD":     "quantity",
            "BD_TP_WATP":     "price",
        })
        keep = ["date", "symbol", "name", "client", "buy_sell", "quantity", "price"]
        df = df[[c for c in keep if c in df.columns]]
        logger.info("Fetched %d block deals for %s", len(df), date)
        return df

    except Exception as exc:
        logger.warning("Block deals fetch failed: %s", exc)
        return _empty_bulk_df()


def _empty_bulk_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["date", "symbol", "name", "client", "buy_sell", "quantity", "price"]
    )


# ──────────────────────────────────────────────
# Insider Trading
# ──────────────────────────────────────────────

def fetch_insider_trades(
    ticker: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch insider trading disclosures from NSE (SAST/PIT regulations).

    Args:
        ticker:    Optional NSE ticker to filter by (e.g. 'RELIANCE').
        from_date: Start date 'YYYY-MM-DD'. Defaults to 30 days ago.
        to_date:   End date 'YYYY-MM-DD'. Defaults to today.

    Returns:
        DataFrame with columns:
          - 'date', 'symbol', 'name', 'person', 'transaction_type',
            'quantity', 'price', 'post_holding_pct'
    """
    if from_date is None:
        from_date = (datetime.today() - timedelta(days=30)).strftime("%d-%m-%Y")
    else:
        from_date = datetime.strptime(from_date, "%Y-%m-%d").strftime("%d-%m-%Y")

    if to_date is None:
        to_date = datetime.today().strftime("%d-%m-%Y")
    else:
        to_date = datetime.strptime(to_date, "%Y-%m-%d").strftime("%d-%m-%Y")

    url = (
        f"https://www.nseindia.com/api/corporates-pit?"
        f"index=equities&from_date={from_date}&to_date={to_date}"
    )
    if ticker:
        url += f"&symbol={ticker.upper()}"

    session = _get_nse_session()

    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        if not data or "data" not in data:
            logger.info("No insider trades found.")
            return _empty_insider_df()

        df = pd.DataFrame(data["data"])
        df = df.rename(columns={
            "date":             "date",
            "symbol":           "symbol",
            "company":          "name",
            "acqName":          "person",
            "tdpTransactionType": "transaction_type",
            "tdpQtyTraded":     "quantity",
            "tdpPrice":         "price",
            "tdpPostHoldingPer": "post_holding_pct",
        })
        keep = ["date", "symbol", "name", "person",
                "transaction_type", "quantity", "price", "post_holding_pct"]
        df = df[[c for c in keep if c in df.columns]]
        logger.info("Fetched %d insider trades.", len(df))
        return df

    except Exception as exc:
        logger.warning("Insider trades fetch failed: %s", exc)
        return _empty_insider_df()


def _empty_insider_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["date", "symbol", "name", "person",
                 "transaction_type", "quantity", "price", "post_holding_pct"]
    )


# ──────────────────────────────────────────────
# Corporate Announcements (BSE)
# ──────────────────────────────────────────────

def fetch_corporate_announcements(
    ticker: Optional[str] = None,
    days: int = 7,
) -> pd.DataFrame:
    """
    Fetch recent corporate announcements from BSE.

    Covers: quarterly results, dividends, bonus, splits, AGM, board meetings.

    Args:
        ticker: Optional BSE ticker to filter (e.g. 'RELIANCE').
        days:   Number of past days to fetch announcements for.

    Returns:
        DataFrame with columns:
          - 'date', 'symbol', 'category', 'headline', 'description'
    """
    to_date = datetime.today().strftime("%Y%m%d")
    from_date = (datetime.today() - timedelta(days=days)).strftime("%Y%m%d")

    url = (
        f"https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w?"
        f"strCat=-1&strPrevDate={from_date}&strScrip=&strSearch=P"
        f"&strToDate={to_date}&strType=C&subcategory=-1"
    )

    try:
        response = requests.get(url, headers=BSE_HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()

        if not data or "Table" not in data:
            logger.info("No announcements found.")
            return _empty_announcement_df()

        df = pd.DataFrame(data["Table"])
        df = df.rename(columns={
            "News_submission_dt": "date",
            "SCRIP_CD":           "symbol",
            "CATEGORYNAME":       "category",
            "HEADLINE":           "headline",
            "SLONGNAME":          "description",
        })
        keep = ["date", "symbol", "category", "headline", "description"]
        df = df[[c for c in keep if c in df.columns]]

        # Filter by ticker if provided
        if ticker and "symbol" in df.columns:
            df = df[df["symbol"].astype(str).str.contains(ticker, case=False)]

        logger.info("Fetched %d corporate announcements.", len(df))
        return df

    except Exception as exc:
        logger.warning("Corporate announcements fetch failed: %s", exc)
        return _empty_announcement_df()


def _empty_announcement_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["date", "symbol", "category", "headline", "description"]
    )


# ──────────────────────────────────────────────
# Signal generation from filings
# ──────────────────────────────────────────────

def generate_filing_signals(ticker: str) -> List[Dict]:
    """
    Generate trading signals from corporate filings for a given ticker.

    Combines bulk deals, insider trades, and announcements into
    actionable signals with sentiment scores.

    Args:
        ticker: NSE ticker symbol (e.g. 'RELIANCE').

    Returns:
        List of signal dicts with 'type', 'description', 'score', 'date'.
    """
    signals = []

    # ── Bulk/Block deals ──
    bulk = fetch_bulk_deals()
    block = fetch_block_deals()
    deals = pd.concat([bulk, block], ignore_index=True)

    if not deals.empty and "symbol" in deals.columns:
        ticker_deals = deals[deals["symbol"].str.upper() == ticker.upper()]
        for _, row in ticker_deals.iterrows():
            is_buy = str(row.get("buy_sell", "")).upper() in ("B", "BUY")
            signals.append({
                "type":        "bulk_block_deal",
                "description": (
                    f"{'BUY' if is_buy else 'SELL'} deal by {row.get('client', 'Unknown')} "
                    f"— {row.get('quantity', 0):,} shares @ ₹{row.get('price', 0)}"
                ),
                "score":  0.3 if is_buy else -0.3,
                "date":   str(row.get("date", "")),
            })

    # ── Insider trades ──
    insiders = fetch_insider_trades(ticker=ticker, from_date=None)
    if not insiders.empty:
        for _, row in insiders.iterrows():
            tx = str(row.get("transaction_type", "")).upper()
            is_buy = "BUY" in tx or "ACQUI" in tx
            signals.append({
                "type":        "insider_trade",
                "description": (
                    f"Insider {row.get('person', 'Unknown')} "
                    f"{'bought' if is_buy else 'sold'} "
                    f"{row.get('quantity', 0):,} shares"
                ),
                "score":  0.4 if is_buy else -0.4,
                "date":   str(row.get("date", "")),
            })

    # ── Corporate announcements ──
    announcements = fetch_corporate_announcements(ticker=ticker)
    if not announcements.empty:
        for _, row in announcements.iterrows():
            headline = str(row.get("headline", "")).lower()
            category = str(row.get("category", "")).lower()

            # Score based on announcement type
            if any(w in headline for w in ["dividend", "bonus", "buyback", "profit"]):
                score = 0.3
            elif any(w in headline for w in ["loss", "penalty", "fraud", "default"]):
                score = -0.4
            elif "result" in category or "financial" in category:
                score = 0.1  # neutral, need to check actual numbers
            else:
                score = 0.0

            signals.append({
                "type":        "corporate_announcement",
                "description": row.get("headline", ""),
                "score":       score,
                "date":        str(row.get("date", "")),
            })

    logger.info(
        "Generated %d filing signals for %s", len(signals), ticker
    )
    return signals
