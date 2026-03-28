"""
pattern_detection.py — Technical chart pattern detection for NSE stocks.

Detects:
- Support & Resistance levels
- Breakouts (price above resistance / below support)
- Trend reversals (RSI-based + price action)
- Head & Shoulders (simplified)

All logic uses OHLCV data enriched with technical indicators from the
processed dataset. No feature engineering is performed here.

FIXES:
- Date extraction now uses pd.Timestamp() wrapper to safely handle both
  tz-aware and tz-naive DatetimeIndex values, preventing AttributeError.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

from src.utils.helpers import get_config, setup_logger

logger = setup_logger(__name__)
cfg = get_config()
_pat_cfg = cfg.get("patterns", {})


# ──────────────────────────────────────────────
# Internal helper: safe date string
# ──────────────────────────────────────────────

def _date_str(ts) -> str:
    """Return a YYYY-MM-DD string from any timestamp-like value."""
    return str(pd.Timestamp(ts).date())


# ──────────────────────────────────────────────
# Support & Resistance
# ──────────────────────────────────────────────

def find_support_resistance(
    df: pd.DataFrame,
    window: Optional[int] = None,
    n_levels: int = 3,
) -> Dict[str, List[float]]:
    """
    Identify key support and resistance price levels using rolling extrema.

    Args:
        df:       DataFrame with at least 'High' and 'Low' columns.
        window:   Rolling window for local min/max search.
        n_levels: Number of support and resistance levels to return.

    Returns:
        Dict with keys:
          - 'support'    — list of support prices (ascending)
          - 'resistance' — list of resistance prices (ascending)
    """
    window = window or _pat_cfg.get("support_resistance_window", 20)

    local_highs = df["High"][
        (df["High"] == df["High"].rolling(window, center=True).max())
    ].dropna()
    local_lows = df["Low"][
        (df["Low"] == df["Low"].rolling(window, center=True).min())
    ].dropna()

    resistance = _cluster_levels(local_highs.values, n_levels)
    support = _cluster_levels(local_lows.values, n_levels)

    logger.debug("Support: %s | Resistance: %s", support, resistance)
    return {"support": support, "resistance": resistance}


def _cluster_levels(prices: np.ndarray, n: int, tol: float = 0.01) -> List[float]:
    """
    Greedily cluster nearby price levels, returning the n most significant.

    Args:
        prices: Array of local extrema prices.
        n:      Maximum number of clusters.
        tol:    Fractional tolerance for clustering (default 1%).

    Returns:
        Sorted list of cluster centroids.
    """
    if len(prices) == 0:
        return []
    sorted_prices = np.sort(prices)
    clusters: List[List[float]] = [[sorted_prices[0]]]

    for p in sorted_prices[1:]:
        centroid = np.mean(clusters[-1])
        if abs(p - centroid) / centroid < tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])

    centroids = sorted([round(np.mean(c), 2) for c in clusters])
    touched = sorted(
        centroids,
        key=lambda c: sum(1 for p in prices if abs(p - c) / c < tol),
        reverse=True,
    )
    return sorted(touched[:n])


# ──────────────────────────────────────────────
# Breakout Detection
# ──────────────────────────────────────────────

def detect_breakouts(
    df: pd.DataFrame,
    levels: Optional[Dict[str, List[float]]] = None,
    breakout_pct: Optional[float] = None,
    confirm_bars: Optional[int] = None,
) -> List[Dict]:
    """
    Identify bullish (upside) and bearish (downside) breakout events.

    A bullish breakout occurs when Close exceeds a resistance level by at
    least breakout_pct percent for confirm_bars consecutive bars.

    Args:
        df:           OHLCV DataFrame (sorted ascending by date).
        levels:       Pre-computed support/resistance dict. If None, computed
                      from df.
        breakout_pct: Minimum % penetration to count as a breakout.
        confirm_bars: Number of consecutive closes required for confirmation.

    Returns:
        List of dicts, each containing:
          - 'date'        (str)
          - 'type'        ('bullish_breakout' | 'bearish_breakout')
          - 'price'       (float) — close at breakout bar
          - 'level'       (float) — the breached support/resistance level
          - 'pct_through' (float) — % penetration beyond the level
    """
    breakout_pct = breakout_pct or cfg["signals"].get("price_breakout_pct", 0.03)
    confirm_bars = confirm_bars or _pat_cfg.get("breakout_confirmation_bars", 2)

    if levels is None:
        levels = find_support_resistance(df)

    events = []
    closes = df["Close"].values
    dates = df.index

    for lvl in levels["resistance"]:
        for i in range(confirm_bars, len(closes)):
            window = closes[i - confirm_bars : i + 1]
            if all(c > lvl * (1 + breakout_pct) for c in window):
                pct = (closes[i] - lvl) / lvl
                events.append(
                    {
                        "date": _date_str(dates[i]),
                        "type": "bullish_breakout",
                        "price": round(float(closes[i]), 2),
                        "level": round(lvl, 2),
                        "pct_through": round(pct * 100, 2),
                    }
                )
                break  # one event per level

    for lvl in levels["support"]:
        for i in range(confirm_bars, len(closes)):
            window = closes[i - confirm_bars : i + 1]
            if all(c < lvl * (1 - breakout_pct) for c in window):
                pct = (lvl - closes[i]) / lvl
                events.append(
                    {
                        "date": _date_str(dates[i]),
                        "type": "bearish_breakout",
                        "price": round(float(closes[i]), 2),
                        "level": round(lvl, 2),
                        "pct_through": round(pct * 100, 2),
                    }
                )
                break

    logger.info("Detected %d breakout events.", len(events))
    return events


# ──────────────────────────────────────────────
# Trend Reversal Detection
# ──────────────────────────────────────────────

def detect_trend_reversals(df: pd.DataFrame) -> List[Dict]:
    """
    Detect potential trend reversal signals using RSI extremes and price action.

    Requires columns: 'Close', 'RSI' (pre-computed in the notebook).
    A bullish reversal candidate: RSI crosses from below 30 upward.
    A bearish reversal candidate: RSI crosses from above 70 downward.

    Args:
        df: Processed DataFrame with 'Close' and 'RSI' columns.

    Returns:
        List of reversal event dicts:
          - 'date'  (str)
          - 'type'  ('bullish_reversal' | 'bearish_reversal')
          - 'price' (float)
          - 'rsi'   (float)
    """
    if "RSI" not in df.columns:
        logger.warning("'RSI' column missing — trend reversal detection skipped.")
        return []

    ob = _pat_cfg.get("trend_reversal_rsi_overbought", 70)
    os_ = _pat_cfg.get("trend_reversal_rsi_oversold", 30)

    events = []
    rsi = df["RSI"].values
    closes = df["Close"].values
    dates = df.index

    for i in range(1, len(rsi)):
        if rsi[i - 1] < os_ and rsi[i] >= os_:
            events.append(
                {
                    "date": _date_str(dates[i]),
                    "type": "bullish_reversal",
                    "price": round(float(closes[i]), 2),
                    "rsi": round(float(rsi[i]), 2),
                }
            )
        elif rsi[i - 1] > ob and rsi[i] <= ob:
            events.append(
                {
                    "date": _date_str(dates[i]),
                    "type": "bearish_reversal",
                    "price": round(float(closes[i]), 2),
                    "rsi": round(float(rsi[i]), 2),
                }
            )

    logger.info("Detected %d trend reversal events.", len(events))
    return events


# ──────────────────────────────────────────────
# Head & Shoulders (simplified pivot detection)
# ──────────────────────────────────────────────

def detect_head_and_shoulders(
    df: pd.DataFrame,
    pivot_window: int = 10,
    shoulder_tolerance: float = 0.03,
) -> List[Dict]:
    """
    Simplified Head & Shoulders pattern detection using pivot highs.

    Identifies sequences: left shoulder → head → right shoulder where
    - head > both shoulders
    - both shoulders are within shoulder_tolerance of each other

    Args:
        df:                OHLCV DataFrame.
        pivot_window:      Bars on each side to confirm a local high.
        shoulder_tolerance: Max fractional difference between shoulders.

    Returns:
        List of pattern dicts with 'date', 'head_price', 'left_shoulder',
        'right_shoulder', 'neckline'.
    """
    highs = df["High"].values
    dates = df.index
    pivots: List[Tuple[int, float]] = []

    for i in range(pivot_window, len(highs) - pivot_window):
        if highs[i] == max(highs[i - pivot_window : i + pivot_window + 1]):
            pivots.append((i, highs[i]))

    patterns = []
    for j in range(1, len(pivots) - 1):
        ls_idx, ls = pivots[j - 1]
        h_idx, head = pivots[j]
        rs_idx, rs = pivots[j + 1]

        if head <= ls or head <= rs:
            continue
        if abs(ls - rs) / ((ls + rs) / 2) > shoulder_tolerance:
            continue

        neckline = round(
            (df["Low"].iloc[ls_idx] + df["Low"].iloc[rs_idx]) / 2, 2
        )
        patterns.append(
            {
                "date": _date_str(dates[rs_idx]),
                "pattern": "head_and_shoulders",
                "head_price": round(float(head), 2),
                "left_shoulder": round(float(ls), 2),
                "right_shoulder": round(float(rs), 2),
                "neckline": neckline,
            }
        )

    logger.info("Detected %d head-and-shoulders patterns.", len(patterns))
    return patterns


# ──────────────────────────────────────────────
# Combined pattern summary
# ──────────────────────────────────────────────

def get_all_patterns(df: pd.DataFrame) -> Dict:
    """
    Run all pattern detectors and return a consolidated summary dict.

    Args:
        df: Processed OHLCV + indicator DataFrame.

    Returns:
        Dict with keys: 'support_resistance', 'breakouts',
        'trend_reversals', 'head_and_shoulders'.
    """
    sr = find_support_resistance(df)
    return {
        "support_resistance": sr,
        "breakouts": detect_breakouts(df, levels=sr),
        "trend_reversals": detect_trend_reversals(df),
        "head_and_shoulders": detect_head_and_shoulders(df),
    }
