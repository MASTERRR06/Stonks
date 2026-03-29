"""
backtester.py — Pattern back-testing engine.

Responsibilities:
- Find historical occurrences of each pattern on a given stock
- Check what happened to price in the N days after each pattern
- Calculate win rate, avg return, and risk/reward for each pattern
- Return back-tested success rates per pattern per stock

This gives Chart Pattern Intelligence a quantitative edge — not just
detecting patterns but telling the user how reliable they've been
historically on that specific stock.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from src.utils.helpers import get_config, setup_logger
from src.technicals.pattern_detection import (
    find_support_resistance,
    detect_breakouts,
    detect_trend_reversals,
    detect_head_and_shoulders,
)

logger = setup_logger(__name__)
cfg = get_config()


# ──────────────────────────────────────────────
# Forward return calculator
# ──────────────────────────────────────────────

def _forward_return(
    df: pd.DataFrame,
    signal_date: str,
    holding_days: int = 10,
) -> Optional[float]:
    """
    Calculate the forward return N days after a signal date.

    Args:
        df:           OHLCV DataFrame sorted by date.
        signal_date:  Date string of the signal (YYYY-MM-DD).
        holding_days: Number of trading days to hold.

    Returns:
        Percentage return (float), or None if not enough future data.
    """
    try:
        idx = df.index.get_loc(signal_date)
    except KeyError:
        # Try nearest date
        try:
            pos = df.index.searchsorted(pd.Timestamp(signal_date))
            if pos >= len(df):
                return None
            idx = pos
        except Exception:
            return None

    future_idx = idx + holding_days
    if future_idx >= len(df):
        return None

    entry_price = df["Close"].iloc[idx]
    exit_price  = df["Close"].iloc[future_idx]

    if entry_price == 0:
        return None

    return round((exit_price - entry_price) / entry_price * 100, 4)


# ──────────────────────────────────────────────
# Individual pattern back-testers
# ──────────────────────────────────────────────

def backtest_breakouts(
    df: pd.DataFrame,
    holding_days: int = 10,
) -> Dict:
    """
    Back-test bullish breakout patterns on historical data.

    Args:
        df:           Full historical OHLCV DataFrame.
        holding_days: Days to hold after breakout signal.

    Returns:
        Dict with win_rate, avg_return, total_signals, avg_win, avg_loss.
    """
    # Use rolling windows to find breakouts at each point in history
    returns = []
    window = 60  # minimum history needed

    for i in range(window, len(df) - holding_days):
        chunk = df.iloc[:i]
        try:
            sr = find_support_resistance(chunk)
            breakouts = detect_breakouts(chunk, levels=sr)
            bullish = [b for b in breakouts if b["type"] == "bullish_breakout"]

            if bullish:
                latest = bullish[-1]
                fwd = _forward_return(df, latest["date"], holding_days)
                if fwd is not None:
                    returns.append(fwd)
        except Exception:
            continue

    return _compute_stats(returns, "bullish_breakout")


def backtest_reversals(
    df: pd.DataFrame,
    holding_days: int = 10,
) -> Dict:
    """
    Back-test bullish reversal patterns on historical data.

    Args:
        df:           Full historical OHLCV DataFrame.
        holding_days: Days to hold after reversal signal.

    Returns:
        Dict with win_rate, avg_return, total_signals, avg_win, avg_loss.
    """
    returns = []
    window = 60

    for i in range(window, len(df) - holding_days):
        chunk = df.iloc[:i]
        try:
            reversals = detect_trend_reversals(chunk)
            bullish = [r for r in reversals if r["type"] == "bullish_reversal"]

            if bullish:
                latest = bullish[-1]
                fwd = _forward_return(df, latest["date"], holding_days)
                if fwd is not None:
                    returns.append(fwd)
        except Exception:
            continue

    return _compute_stats(returns, "bullish_reversal")


def backtest_head_and_shoulders(
    df: pd.DataFrame,
    holding_days: int = 15,
) -> Dict:
    """
    Back-test Head & Shoulders patterns (bearish signal).

    Args:
        df:           Full historical OHLCV DataFrame.
        holding_days: Days to hold short after pattern.

    Returns:
        Dict with win_rate, avg_return, total_signals, avg_win, avg_loss.
    """
    returns = []
    window = 60

    for i in range(window, len(df) - holding_days):
        chunk = df.iloc[:i]
        try:
            patterns = detect_head_and_shoulders(chunk)
            if patterns:
                latest = patterns[-1]
                # H&S is bearish — invert the return
                fwd = _forward_return(df, latest["date"], holding_days)
                if fwd is not None:
                    returns.append(-fwd)  # negative because bearish
        except Exception:
            continue

    return _compute_stats(returns, "head_and_shoulders")


# ──────────────────────────────────────────────
# Stats calculator
# ──────────────────────────────────────────────

def _compute_stats(returns: List[float], pattern_name: str) -> Dict:
    """
    Compute back-test statistics from a list of forward returns.

    Args:
        returns:      List of percentage returns after each signal.
        pattern_name: Name of the pattern being back-tested.

    Returns:
        Dict with win_rate, avg_return, avg_win, avg_loss, total_signals,
        best_trade, worst_trade, pattern.
    """
    if not returns:
        return {
            "pattern":       pattern_name,
            "total_signals": 0,
            "win_rate":      0.0,
            "avg_return":    0.0,
            "avg_win":       0.0,
            "avg_loss":      0.0,
            "best_trade":    0.0,
            "worst_trade":   0.0,
            "verdict":       "Insufficient data",
        }

    returns_arr = np.array(returns)
    wins  = returns_arr[returns_arr > 0]
    losses = returns_arr[returns_arr <= 0]

    win_rate   = round(len(wins) / len(returns_arr) * 100, 1)
    avg_return = round(float(np.mean(returns_arr)), 2)
    avg_win    = round(float(np.mean(wins)), 2)   if len(wins)   > 0 else 0.0
    avg_loss   = round(float(np.mean(losses)), 2) if len(losses) > 0 else 0.0

    if win_rate >= 60:
        verdict = "Strong signal — historically reliable"
    elif win_rate >= 50:
        verdict = "Moderate signal — use with confirmation"
    else:
        verdict = "Weak signal — treat with caution"

    return {
        "pattern":       pattern_name,
        "total_signals": len(returns),
        "win_rate":      win_rate,
        "avg_return":    avg_return,
        "avg_win":       avg_win,
        "avg_loss":      avg_loss,
        "best_trade":    round(float(np.max(returns_arr)), 2),
        "worst_trade":   round(float(np.min(returns_arr)), 2),
        "verdict":       verdict,
    }


# ──────────────────────────────────────────────
# Full back-test runner
# ──────────────────────────────────────────────

def run_full_backtest(
    df: pd.DataFrame,
    ticker: str,
    holding_days: int = 10,
) -> Dict:
    """
    Run back-tests for all patterns on a given stock's historical data.

    Args:
        df:           Full historical OHLCV + indicator DataFrame.
        ticker:       Stock ticker (for labelling).
        holding_days: Days to hold after each signal.

    Returns:
        Dict with back-test results for each pattern and a summary.
    """
    logger.info("Running full backtest for %s | holding_days=%d", ticker, holding_days)

    breakout_stats  = backtest_breakouts(df, holding_days)
    reversal_stats  = backtest_reversals(df, holding_days)
    hs_stats        = backtest_head_and_shoulders(df, holding_days)

    results = {
        "ticker":       ticker,
        "holding_days": holding_days,
        "patterns": {
            "bullish_breakout":    breakout_stats,
            "bullish_reversal":    reversal_stats,
            "head_and_shoulders":  hs_stats,
        },
        "best_pattern": _find_best_pattern([
            breakout_stats, reversal_stats, hs_stats
        ]),
    }

    logger.info("Backtest complete for %s", ticker)
    return results


def _find_best_pattern(stats_list: List[Dict]) -> str:
    """Return the pattern name with the highest win rate."""
    valid = [s for s in stats_list if s["total_signals"] > 0]
    if not valid:
        return "No patterns with sufficient data"
    best = max(valid, key=lambda s: s["win_rate"])
    return f"{best['pattern']} (win rate: {best['win_rate']}%)"


# ──────────────────────────────────────────────
# Quick summary for API
# ──────────────────────────────────────────────

def get_pattern_success_rates(
    df: pd.DataFrame,
    ticker: str,
) -> List[Dict]:
    """
    Get back-tested success rates for all patterns — formatted for API response.

    Args:
        df:     Historical OHLCV + indicator DataFrame.
        ticker: Stock ticker symbol.

    Returns:
        List of dicts with pattern name, win rate, avg return, verdict.
    """
    results = run_full_backtest(df, ticker)
    output = []
    for pattern_name, stats in results["patterns"].items():
        output.append({
            "pattern":       pattern_name,
            "win_rate_pct":  stats["win_rate"],
            "avg_return_pct": stats["avg_return"],
            "total_signals": stats["total_signals"],
            "verdict":       stats["verdict"],
        })
    return output
