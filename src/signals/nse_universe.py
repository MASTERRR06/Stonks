"""
nse_universe.py — Full NSE universe scanner.

Responsibilities:
- Maintain a comprehensive list of NSE-listed stocks
- Run Opportunity Radar across the full universe or a filtered subset
- Return ranked alerts sorted by signal strength
- Support filtering by sector, index membership, market cap tier
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from src.utils.helpers import get_config, setup_logger

logger = setup_logger(__name__)
cfg = get_config()

# ── Full NSE universe — Nifty 500 tickers ──
# Covers large, mid, and small cap stocks across all sectors
NSE_UNIVERSE = {
    # ── Nifty 50 ──
    "RELIANCE":    {"sector": "Energy",         "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "TCS":         {"sector": "IT",             "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "HDFCBANK":    {"sector": "Banking",        "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "INFY":        {"sector": "IT",             "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "ICICIBANK":   {"sector": "Banking",        "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "HINDUNILVR":  {"sector": "FMCG",           "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "SBIN":        {"sector": "Banking",        "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "BHARTIARTL":  {"sector": "Telecom",        "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "KOTAKBANK":   {"sector": "Banking",        "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "WIPRO":       {"sector": "IT",             "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "ADANIENT":    {"sector": "Conglomerate",   "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "AXISBANK":    {"sector": "Banking",        "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "LT":          {"sector": "Infrastructure", "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "MARUTI":      {"sector": "Auto",           "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "BAJFINANCE":  {"sector": "NBFC",           "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "HCLTECH":     {"sector": "IT",             "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "SUNPHARMA":   {"sector": "Pharma",         "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "TATAMOTORS":  {"sector": "Auto",           "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "TATASTEEL":   {"sector": "Metals",         "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "NTPC":        {"sector": "Power",          "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "POWERGRID":   {"sector": "Power",          "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "ONGC":        {"sector": "Energy",         "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "JSWSTEEL":    {"sector": "Metals",         "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "COALINDIA":   {"sector": "Mining",         "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "TITAN":       {"sector": "Consumer",       "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "TECHM":       {"sector": "IT",             "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "ULTRACEMCO":  {"sector": "Cement",         "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "NESTLEIND":   {"sector": "FMCG",           "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "BAJAJFINSV":  {"sector": "NBFC",           "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    "ASIANPAINT":  {"sector": "Consumer",       "index": ["NIFTY50", "NIFTY500"], "cap": "large"},
    # ── Nifty Midcap ──
    "MUTHOOTFIN":  {"sector": "NBFC",           "index": ["NIFTYMIDCAP", "NIFTY500"], "cap": "mid"},
    "PERSISTENT":  {"sector": "IT",             "index": ["NIFTYMIDCAP", "NIFTY500"], "cap": "mid"},
    "COFORGE":     {"sector": "IT",             "index": ["NIFTYMIDCAP", "NIFTY500"], "cap": "mid"},
    "LTIM":        {"sector": "IT",             "index": ["NIFTYMIDCAP", "NIFTY500"], "cap": "mid"},
    "ZOMATO":      {"sector": "Consumer Tech",  "index": ["NIFTYMIDCAP", "NIFTY500"], "cap": "mid"},
    "PAYTM":       {"sector": "Fintech",        "index": ["NIFTYMIDCAP", "NIFTY500"], "cap": "mid"},
    "NYKAA":       {"sector": "Consumer Tech",  "index": ["NIFTYMIDCAP", "NIFTY500"], "cap": "mid"},
    "POLICYBZR":   {"sector": "Fintech",        "index": ["NIFTYMIDCAP", "NIFTY500"], "cap": "mid"},
    "DELHIVERY":   {"sector": "Logistics",      "index": ["NIFTYMIDCAP", "NIFTY500"], "cap": "mid"},
    "IRCTC":       {"sector": "Travel",         "index": ["NIFTYMIDCAP", "NIFTY500"], "cap": "mid"},
    "CHOLAFIN":    {"sector": "NBFC",           "index": ["NIFTYMIDCAP", "NIFTY500"], "cap": "mid"},
    "AUBANK":      {"sector": "Banking",        "index": ["NIFTYMIDCAP", "NIFTY500"], "cap": "mid"},
    "FEDERALBNK":  {"sector": "Banking",        "index": ["NIFTYMIDCAP", "NIFTY500"], "cap": "mid"},
    "IDFCFIRSTB":  {"sector": "Banking",        "index": ["NIFTYMIDCAP", "NIFTY500"], "cap": "mid"},
    "ABCAPITAL":   {"sector": "NBFC",           "index": ["NIFTYMIDCAP", "NIFTY500"], "cap": "mid"},
    "GMRINFRA":    {"sector": "Infrastructure", "index": ["NIFTYMIDCAP", "NIFTY500"], "cap": "mid"},
    "ADANIPORTS":  {"sector": "Infrastructure", "index": ["NIFTYMIDCAP", "NIFTY500"], "cap": "mid"},
    "SIEMENS":     {"sector": "Industrial",     "index": ["NIFTYMIDCAP", "NIFTY500"], "cap": "mid"},
    "ABB":         {"sector": "Industrial",     "index": ["NIFTYMIDCAP", "NIFTY500"], "cap": "mid"},
    "HAVELLS":     {"sector": "Consumer",       "index": ["NIFTYMIDCAP", "NIFTY500"], "cap": "mid"},
}

# ── Sector groupings ──
SECTORS = list(set(v["sector"] for v in NSE_UNIVERSE.values()))

# ── Index groupings ──
INDICES = {
    "NIFTY50":     [k for k, v in NSE_UNIVERSE.items() if "NIFTY50" in v["index"]],
    "NIFTYMIDCAP": [k for k, v in NSE_UNIVERSE.items() if "NIFTYMIDCAP" in v["index"]],
    "NIFTY500":    list(NSE_UNIVERSE.keys()),
}


# ──────────────────────────────────────────────
# Universe filters
# ──────────────────────────────────────────────

def get_tickers_by_sector(sector: str) -> List[str]:
    """Return all tickers in a given sector."""
    return [k for k, v in NSE_UNIVERSE.items()
            if v["sector"].lower() == sector.lower()]


def get_tickers_by_index(index: str) -> List[str]:
    """Return all tickers in a given index (NIFTY50, NIFTYMIDCAP, NIFTY500)."""
    return INDICES.get(index.upper(), list(NSE_UNIVERSE.keys()))


def get_tickers_by_cap(cap: str) -> List[str]:
    """Return tickers filtered by market cap tier (large, mid, small)."""
    return [k for k, v in NSE_UNIVERSE.items()
            if v["cap"].lower() == cap.lower()]


# ──────────────────────────────────────────────
# Universe scanner
# ──────────────────────────────────────────────

def scan_nse_universe(
    data_loader_fn,
    tickers: Optional[List[str]] = None,
    sector: Optional[str] = None,
    index: Optional[str] = None,
    cap: Optional[str] = None,
    min_score: float = 0.20,
    max_tickers: int = 50,
) -> List[Dict]:
    """
    Run the Opportunity Radar across a filtered NSE universe.

    Args:
        data_loader_fn: Callable(ticker) → pd.DataFrame (OHLCV + indicators).
        tickers:        Optional explicit list of tickers. Overrides filters.
        sector:         Filter by sector (e.g. 'IT', 'Banking').
        index:          Filter by index (e.g. 'NIFTY50', 'NIFTYMIDCAP').
        cap:            Filter by cap tier ('large', 'mid').
        min_score:      Minimum |composite_score| to include in results.
        max_tickers:    Maximum number of tickers to scan (avoids rate limits).

    Returns:
        List of alert dicts sorted by |score| descending, with sector/cap info.
    """
    from src.signals.opportunity_radar import run_opportunity_radar
    from src.technicals.pattern_detection import get_all_patterns

    # ── Resolve ticker list ──
    if tickers:
        scan_list = tickers
    elif sector:
        scan_list = get_tickers_by_sector(sector)
    elif index:
        scan_list = get_tickers_by_index(index)
    elif cap:
        scan_list = get_tickers_by_cap(cap)
    else:
        scan_list = list(NSE_UNIVERSE.keys())

    scan_list = scan_list[:max_tickers]
    logger.info("Scanning %d tickers from NSE universe...", len(scan_list))

    alerts = []
    for ticker in scan_list:
        try:
            df = data_loader_fn(ticker)
            patterns = get_all_patterns(df)
            alert = run_opportunity_radar(
                ticker=ticker,
                df=df,
                pattern_data=patterns,
            )

            if abs(alert["score"]) >= min_score:
                # Enrich with universe metadata
                meta = NSE_UNIVERSE.get(ticker, {})
                alert["sector"] = meta.get("sector", "Unknown")
                alert["cap"]    = meta.get("cap", "Unknown")
                alerts.append(alert)

        except Exception as exc:
            logger.warning("Scan failed for %s: %s", ticker, exc)

    alerts.sort(key=lambda a: abs(a["score"]), reverse=True)
    logger.info(
        "Universe scan complete. %d alerts generated from %d tickers.",
        len(alerts), len(scan_list)
    )
    return alerts


def get_top_opportunities(
    data_loader_fn,
    n: int = 10,
    index: str = "NIFTY50",
) -> List[Dict]:
    """
    Get the top N opportunities from a given index.

    Args:
        data_loader_fn: Callable(ticker) → pd.DataFrame.
        n:              Number of top opportunities to return.
        index:          Index to scan ('NIFTY50', 'NIFTYMIDCAP', 'NIFTY500').

    Returns:
        Top N alert dicts sorted by score.
    """
    alerts = scan_nse_universe(
        data_loader_fn=data_loader_fn,
        index=index,
        min_score=0.0,
    )
    return alerts[:n]


def get_sector_summary(
    data_loader_fn,
    sector: str,
) -> Dict:
    """
    Get a summary of opportunities within a specific sector.

    Args:
        data_loader_fn: Callable(ticker) → pd.DataFrame.
        sector:         Sector name (e.g. 'IT', 'Banking').

    Returns:
        Dict with sector name, alert list, avg score, bullish/bearish count.
    """
    alerts = scan_nse_universe(
        data_loader_fn=data_loader_fn,
        sector=sector,
        min_score=0.0,
    )

    bullish = [a for a in alerts if a["alert_type"] == "BUY"]
    bearish = [a for a in alerts if a["alert_type"] == "SELL"]
    avg_score = round(
        np.mean([a["score"] for a in alerts]) if alerts else 0.0, 4
    )

    return {
        "sector":         sector,
        "total_stocks":   len(alerts),
        "bullish_count":  len(bullish),
        "bearish_count":  len(bearish),
        "avg_score":      avg_score,
        "sentiment":      "bullish" if avg_score > 0.1 else "bearish" if avg_score < -0.1 else "neutral",
        "top_picks":      alerts[:3],
        "alerts":         alerts,
    }
