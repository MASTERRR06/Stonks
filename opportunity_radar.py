"""
opportunity_radar.py — Opportunity signal aggregation engine.

Combines:
- FinBERT sentiment scores (from nlp module)
- Price movement signals (from predictor)
- Volume spike detection
- Technical pattern signals (from pattern_detection)

Outputs actionable BUY / SELL / WATCH alerts with composite scores.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime

from src.utils.helpers import get_config, setup_logger

logger = setup_logger(__name__)
cfg = get_config()
_sig_cfg = cfg.get("signals", {})


# ──────────────────────────────────────────────
# Volume spike detector
# ──────────────────────────────────────────────

def detect_volume_spikes(
    df: pd.DataFrame,
    window: int = 20,
    threshold: Optional[float] = None,
) -> pd.Series:
    """
    Flag bars where volume is significantly above the rolling average.

    Args:
        df:        OHLCV DataFrame with a 'Volume' column.
        window:    Rolling average window in bars.
        threshold: Multiplier above rolling mean to flag as a spike.

    Returns:
        Boolean Series aligned with df.index — True where a spike is detected.
    """
    threshold = threshold or _sig_cfg.get("volume_spike_threshold", 2.0)
    rolling_avg = df["Volume"].rolling(window=window, min_periods=1).mean()
    spikes = df["Volume"] > (rolling_avg * threshold)
    logger.debug("Volume spikes detected: %d", spikes.sum())
    return spikes


# ──────────────────────────────────────────────
# Price momentum signal
# ──────────────────────────────────────────────

def compute_price_momentum(df: pd.DataFrame, lookback: int = 5) -> float:
    """
    Calculate recent price momentum as a percentage return.

    Args:
        df:       OHLCV DataFrame.
        lookback: Number of bars for the momentum calculation.

    Returns:
        Float — percentage return over the lookback period (positive = up).
    """
    if len(df) < lookback + 1:
        return 0.0
    start_price = df["Close"].iloc[-(lookback + 1)]
    end_price = df["Close"].iloc[-1]
    return round((end_price - start_price) / start_price * 100, 4)


# ──────────────────────────────────────────────
# Composite signal scoring
# ──────────────────────────────────────────────

def compute_opportunity_score(
    sentiment_score: float,
    price_momentum_pct: float,
    has_volume_spike: bool,
    has_breakout: bool,
    has_reversal: bool,
) -> Dict:
    """
    Compute a composite opportunity score from multiple signal sources.

    Scoring breakdown:
    - Sentiment      : ±0.30 weight
    - Price momentum : ±0.25 weight (normalised to ±10% range)
    - Volume spike   : +0.15 (binary)
    - Breakout       : ±0.20 (bullish / bearish)
    - Reversal       : ±0.10 (bullish / bearish)

    Args:
        sentiment_score:    Float in [-1, 1] from FinBERT aggregation.
        price_momentum_pct: Recent % return (e.g. +3.5 or -1.2).
        has_volume_spike:   Whether the latest bar has a volume spike.
        has_breakout:       Whether a bullish breakout was detected recently.
        has_reversal:       Whether a bullish reversal was detected recently.

    Returns:
        Dict with:
          - 'composite_score'  (float in [-1, 1])
          - 'action'           ('BUY' | 'SELL' | 'WATCH')
          - 'components'       (dict of individual sub-scores)
    """
    # Normalise momentum: ±10% → ±1.0
    norm_momentum = max(-1.0, min(1.0, price_momentum_pct / 10.0))

    components = {
        "sentiment":   round(sentiment_score * 0.30, 4),
        "momentum":    round(norm_momentum * 0.25, 4),
        "volume_spike": 0.15 if has_volume_spike else 0.0,
        "breakout":    0.20 if has_breakout else 0.0,
        "reversal":    0.10 if has_reversal else 0.0,
    }

    composite = round(sum(components.values()), 4)
    composite = max(-1.0, min(1.0, composite))  # clamp

    if composite >= 0.30:
        action = "BUY"
    elif composite <= -0.20:
        action = "SELL"
    else:
        action = "WATCH"

    return {
        "composite_score": composite,
        "action": action,
        "components": components,
    }


# ──────────────────────────────────────────────
# Main radar pipeline
# ──────────────────────────────────────────────

def run_opportunity_radar(
    ticker: str,
    df: pd.DataFrame,
    sentiment_results: Optional[List[Dict]] = None,
    pattern_data: Optional[Dict] = None,
    predicted_price: Optional[float] = None,
) -> Dict:
    """
    Run the full Opportunity Radar pipeline for a single ticker.

    Args:
        ticker:           NSE ticker symbol (e.g. 'RELIANCE').
        df:               Processed OHLCV + indicator DataFrame.
        sentiment_results: Output from finbert_sentiment.predict_sentiment().
                           If None, sentiment score defaults to 0.
        pattern_data:     Output from pattern_detection.get_all_patterns().
                          If None, pattern signals default to False.
        predicted_price:  LSTM predicted next close. Used in the alert detail.

    Returns:
        Radar alert dict containing:
          - 'ticker'
          - 'timestamp'
          - 'current_price'
          - 'predicted_price'
          - 'alert_type'    ('BUY' | 'SELL' | 'WATCH')
          - 'score'
          - 'components'
          - 'details'       (human-readable summary)
          - 'signals'       (sub-signal list)
    """
    current_price = round(float(df["Close"].iloc[-1]), 2)

    # ── Sentiment ──
    if sentiment_results:
        from src.nlp.finbert_sentiment import sentiment_to_signal
        sentiment_score = sentiment_to_signal(sentiment_results)
    else:
        sentiment_score = 0.0

    # ── Volume spike ──
    spikes = detect_volume_spikes(df)
    has_volume_spike = bool(spikes.iloc[-1]) if len(spikes) > 0 else False

    # ── Price momentum ──
    momentum = compute_price_momentum(df)

    # ── Pattern signals ──
    has_bullish_breakout = False
    has_bullish_reversal = False
    if pattern_data:
        breakouts = pattern_data.get("breakouts", [])
        reversals = pattern_data.get("trend_reversals", [])
        has_bullish_breakout = any(b["type"] == "bullish_breakout" for b in breakouts[-3:])
        has_bullish_reversal = any(r["type"] == "bullish_reversal" for r in reversals[-3:])

    # ── Composite score ──
    scoring = compute_opportunity_score(
        sentiment_score=sentiment_score,
        price_momentum_pct=momentum,
        has_volume_spike=has_volume_spike,
        has_breakout=has_bullish_breakout,
        has_reversal=has_bullish_reversal,
    )

    # ── Human-readable signals ──
    active_signals = []
    if abs(sentiment_score) > 0.3:
        label = "Positive" if sentiment_score > 0 else "Negative"
        active_signals.append(f"{label} news sentiment ({sentiment_score:+.2f})")
    if has_volume_spike:
        active_signals.append("Volume spike detected")
    if abs(momentum) > 1.5:
        direction = "upward" if momentum > 0 else "downward"
        active_signals.append(f"Strong {direction} price momentum ({momentum:+.1f}%)")
    if has_bullish_breakout:
        active_signals.append("Bullish resistance breakout confirmed")
    if has_bullish_reversal:
        active_signals.append("Bullish trend reversal signal (RSI)")

    detail = f"Score={scoring['composite_score']:+.2f} | " + (
        "; ".join(active_signals) if active_signals else "No strong signals"
    )

    alert = {
        "ticker": ticker,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "current_price": current_price,
        "predicted_price": predicted_price,
        "alert_type": scoring["action"],
        "score": scoring["composite_score"],
        "components": scoring["components"],
        "details": detail,
        "signals": active_signals,
    }

    logger.info(
        "Radar | %s | %s | score=%.2f",
        ticker, scoring["action"], scoring["composite_score"],
    )
    return alert


# ──────────────────────────────────────────────
# Batch radar across multiple tickers
# ──────────────────────────────────────────────

def scan_watchlist(
    tickers: List[str],
    data_loader_fn,
    min_score: float = 0.20,
) -> List[Dict]:
    """
    Run the Opportunity Radar across a watchlist and return sorted alerts.

    Args:
        tickers:        List of NSE ticker symbols.
        data_loader_fn: Callable(ticker) → pd.DataFrame (OHLCV + features).
        min_score:      Minimum |composite_score| to include in results.

    Returns:
        List of alert dicts sorted by |score| descending.
    """
    alerts = []
    for ticker in tickers:
        try:
            df = data_loader_fn(ticker)
            alert = run_opportunity_radar(ticker, df)
            if abs(alert["score"]) >= min_score:
                alerts.append(alert)
        except Exception as exc:
            logger.warning("Radar failed for %s: %s", ticker, exc)

    alerts.sort(key=lambda a: abs(a["score"]), reverse=True)
    logger.info("Watchlist scan complete. %d alerts generated.", len(alerts))
    return alerts
