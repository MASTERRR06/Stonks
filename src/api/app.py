"""
app.py — FastAPI REST API for the NSE AI Stock Analysis System.

Endpoints:
  POST /predict              — LSTM next-price prediction
  POST /sentiment            — FinBERT sentiment analysis
  GET  /signals/{ticker}     — Opportunity Radar composite alert
  GET  /patterns/{ticker}    — Technical chart pattern detection
  GET  /filings/{ticker}     — Corporate filings & insider trades
  GET  /backtest/{ticker}    — Pattern back-test success rates
  GET  /universe/scan        — Full NSE universe opportunity scan
  GET  /universe/sector/{s}  — Sector-level opportunity summary
  GET  /health               — Liveness check
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd

from src.utils.helpers import get_config, setup_logger
from src.ingestion.data_loader import fetch_nse_data, get_mock_data, load_news_data
from src.inference.predictor import predict_next_price, get_model_info
from src.nlp.finbert_sentiment import predict_sentiment, sentiment_to_signal
from src.technicals.pattern_detection import get_all_patterns
from src.signals.opportunity_radar import run_opportunity_radar
from src.filings.filings import generate_filing_signals
from src.backtesting.backtester import get_pattern_success_rates
from src.signals.nse_universe import scan_nse_universe, get_sector_summary

cfg = get_config()
logger = setup_logger(__name__)

app = FastAPI(
    title=cfg["api"]["title"],
    version=cfg["api"]["version"],
    description=(
        "Production-grade AI system for NSE stock analysis. "
        "Combines LSTM forecasting, FinBERT sentiment, chart pattern detection, "
        "corporate filings intelligence, pattern back-testing, "
        "and full NSE universe opportunity scanning."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────


class PredictRequest(BaseModel):
    ticker: str = Field(default="RELIANCE", examples=["RELIANCE"])
    period: str = Field(default="6mo", examples=["6mo"])
    use_mock: bool = Field(default=False)


class PredictResponse(BaseModel):
    ticker: str
    current_price: float
    predicted_price: float
    scaled_output: float
    window_size: int
    model_info: Dict[str, Any]


class SentimentRequest(BaseModel):
    texts: List[str] = Field(
        examples=[["Reliance Industries reports record quarterly profit"]]
    )


class SentimentResult(BaseModel):
    text: str
    label: str
    score: float
    confidence: float
    probabilities: Dict[str, float]


class SentimentResponse(BaseModel):
    results: List[SentimentResult]
    composite_signal: float
    signal_label: str


class SignalResponse(BaseModel):
    ticker: str
    timestamp: str
    current_price: float
    predicted_price: Optional[float]
    alert_type: str
    score: float
    components: Dict[str, float]
    details: str
    signals: List[str]


class PatternResponse(BaseModel):
    ticker: str
    support: List[float]
    resistance: List[float]
    breakouts: List[Dict]
    trend_reversals: List[Dict]
    head_and_shoulders: List[Dict]
    success_rates: Optional[List[Dict]] = None


class FilingSignal(BaseModel):
    type: str
    description: str
    score: float
    date: str


class FilingsResponse(BaseModel):
    ticker: str
    total_signals: int
    composite_score: float
    signals: List[FilingSignal]


class BacktestResult(BaseModel):
    pattern: str
    win_rate_pct: float
    avg_return_pct: float
    total_signals: int
    verdict: str


class BacktestResponse(BaseModel):
    ticker: str
    holding_days: int
    results: List[BacktestResult]


class UniverseScanResponse(BaseModel):
    total_alerts: int
    tickers_scanned: int
    alerts: List[Dict]


class SectorSummaryResponse(BaseModel):
    sector: str
    total_stocks: int
    bullish_count: int
    bearish_count: int
    avg_score: float
    sentiment: str
    top_picks: List[Dict]


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _get_df(ticker: str, period: str = "1y", use_mock: bool = False) -> pd.DataFrame:
    if use_mock:
        return get_mock_data(ticker=ticker)
    try:
        return fetch_nse_data(ticker=ticker, period=period, save=False)
    except Exception as exc:
        logger.warning("Data fetch failed (%s) — falling back to mock data", exc)
        return get_mock_data(ticker=ticker)


def _enrich_with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    try:
        import ta

        df = df.copy()
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(df["Close"])
        df["BB_Upper"] = bb.bollinger_hband()
        df["BB_Lower"] = bb.bollinger_lband()
        df.dropna(inplace=True)
        return df
    except ImportError:
        logger.warning("'ta' package not found — indicators skipped.")
        return df


# ──────────────────────────────────────────────
# Existing routes
# ──────────────────────────────────────────────


@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "version": cfg["api"]["version"]}


@app.post("/predict", response_model=PredictResponse, tags=["Forecasting"])
def predict(request: PredictRequest):
    df = _get_df(request.ticker, period=request.period, use_mock=request.use_mock)
    df = _enrich_with_indicators(df)
    feature_cols = [
        "Close",
        "Open",
        "High",
        "Low",
        "Volume",
        "RSI",
        "MACD",
        "MACD_Signal",
        "BB_Upper",
        "BB_Lower",
    ]
    available = [c for c in feature_cols if c in df.columns]
    df_feat = df[available].dropna()
    try:
        result = predict_next_price(df_feat)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503, detail=f"Model not found. Run the notebook first. ({exc})"
        )
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(exc))
    return PredictResponse(
        ticker=request.ticker,
        current_price=round(float(df["Close"].iloc[-1]), 2),
        predicted_price=result["predicted_price"],
        scaled_output=result["scaled_output"],
        window_size=result["window_size"],
        model_info=get_model_info(),
    )


@app.post("/sentiment", response_model=SentimentResponse, tags=["NLP"])
def sentiment(request: SentimentRequest):
    if not request.texts:
        raise HTTPException(status_code=400, detail="'texts' list cannot be empty.")
    try:
        results = predict_sentiment(request.texts)
    except Exception as exc:
        logger.exception("Sentiment error")
        raise HTTPException(status_code=500, detail=str(exc))
    composite = sentiment_to_signal(results)
    label = (
        "bullish" if composite >= 0.3 else "bearish" if composite <= -0.3 else "neutral"
    )
    return SentimentResponse(
        results=[SentimentResult(**r) for r in results],
        composite_signal=composite,
        signal_label=label,
    )


@app.get("/signals/{ticker}", response_model=SignalResponse, tags=["Signals"])
def signals(
    ticker: str,
    period: str = Query(default="1y"),
    use_mock: bool = Query(default=False),
    include_news: bool = Query(default=True),
):
    df = _get_df(ticker, period=period, use_mock=use_mock)
    df = _enrich_with_indicators(df)
    sentiment_results = None
    if include_news:
        try:
            news_df = load_news_data()
            sentiment_results = predict_sentiment(news_df["headline"].tolist())
        except Exception as exc:
            logger.warning("News sentiment skipped: %s", exc)
    patterns = get_all_patterns(df)
    try:
        feature_cols = [
            "Close",
            "Open",
            "High",
            "Low",
            "Volume",
            "RSI",
            "MACD",
            "MACD_Signal",
            "BB_Upper",
            "BB_Lower",
        ]
        available = [c for c in feature_cols if c in df.columns]
        pred = predict_next_price(df[available].dropna())
        predicted_price = pred["predicted_price"]
    except Exception:
        predicted_price = None
    alert = run_opportunity_radar(
        ticker=ticker,
        df=df,
        sentiment_results=sentiment_results,
        pattern_data=patterns,
        predicted_price=predicted_price,
    )
    return SignalResponse(**alert)


@app.get("/patterns/{ticker}", response_model=PatternResponse, tags=["Technicals"])
def patterns(
    ticker: str,
    period: str = Query(default="1y"),
    use_mock: bool = Query(default=False),
    include_backtest: bool = Query(
        default=True, description="Include back-tested success rates for each pattern."
    ),
):
    """
    Detect chart patterns and optionally include historical success rates.

    **Sample request:**
    ```
    GET /patterns/TCS?period=1y&include_backtest=true
    ```
    """
    df = _get_df(ticker, period=period, use_mock=use_mock)
    df = _enrich_with_indicators(df)
    result = get_all_patterns(df)

    success_rates = None
    if include_backtest:
        try:
            success_rates = get_pattern_success_rates(df, ticker)
        except Exception as exc:
            logger.warning("Backtest skipped: %s", exc)

    return PatternResponse(
        ticker=ticker,
        support=result["support_resistance"]["support"],
        resistance=result["support_resistance"]["resistance"],
        breakouts=result["breakouts"],
        trend_reversals=result["trend_reversals"],
        head_and_shoulders=result["head_and_shoulders"],
        success_rates=success_rates,
    )


# ──────────────────────────────────────────────
# New routes
# ──────────────────────────────────────────────


@app.get("/filings/{ticker}", response_model=FilingsResponse, tags=["Filings"])
def filings(ticker: str):
    """
    Fetch corporate filings signals for a ticker.

    Combines bulk/block deals, insider trades, and corporate announcements
    into actionable signals with composite sentiment score.

    **Sample request:**
    ```
    GET /filings/RELIANCE
    ```
    **Sample response:**
    ```json
    {
      "ticker": "RELIANCE",
      "total_signals": 3,
      "composite_score": 0.35,
      "signals": [
        {"type": "insider_trade", "description": "Insider bought 50,000 shares",
         "score": 0.4, "date": "2024-11-10"},
        ...
      ]
    }
    ```
    """
    try:
        signals = generate_filing_signals(ticker)
    except Exception as exc:
        logger.exception("Filings error")
        raise HTTPException(status_code=500, detail=str(exc))

    composite = round(
        sum(s["score"] for s in signals) / len(signals) if signals else 0.0, 4
    )
    return FilingsResponse(
        ticker=ticker,
        total_signals=len(signals),
        composite_score=composite,
        signals=[FilingSignal(**s) for s in signals],
    )


@app.get("/backtest/{ticker}", response_model=BacktestResponse, tags=["Backtesting"])
def backtest(
    ticker: str,
    period: str = Query(
        default="2y", description="Historical period for back-testing. Use at least 1y."
    ),
    holding_days: int = Query(
        default=10, description="Days to hold after each pattern signal."
    ),
    use_mock: bool = Query(default=False),
):
    """
    Back-test chart patterns on a stock's historical data.

    Returns win rate, average return, and a plain-English verdict
    for each pattern type on this specific stock.

    **Sample request:**
    ```
    GET /backtest/RELIANCE?period=2y&holding_days=10
    ```
    **Sample response:**
    ```json
    {
      "ticker": "RELIANCE",
      "holding_days": 10,
      "results": [
        {
          "pattern": "bullish_breakout",
          "win_rate_pct": 64.3,
          "avg_return_pct": 2.8,
          "total_signals": 14,
          "verdict": "Strong signal — historically reliable"
        },
        ...
      ]
    }
    ```
    """
    df = _get_df(ticker, period=period, use_mock=use_mock)
    df = _enrich_with_indicators(df)

    try:
        results = get_pattern_success_rates(df, ticker)
    except Exception as exc:
        logger.exception("Backtest error")
        raise HTTPException(status_code=500, detail=str(exc))

    return BacktestResponse(
        ticker=ticker,
        holding_days=holding_days,
        results=[BacktestResult(**r) for r in results],
    )


@app.get("/universe/scan", response_model=UniverseScanResponse, tags=["Universe"])
def universe_scan(
    index: str = Query(
        default="NIFTY50", description="Index to scan: NIFTY50, NIFTYMIDCAP, NIFTY500"
    ),
    sector: Optional[str] = Query(
        default=None, description="Filter by sector e.g. IT, Banking, Pharma"
    ),
    cap: Optional[str] = Query(default=None, description="Filter by cap: large, mid"),
    min_score: float = Query(
        default=0.20, description="Minimum signal score to include."
    ),
    use_mock: bool = Query(default=False),
):
    """
    Scan the full NSE universe for opportunities.

    Runs the Opportunity Radar across multiple stocks and returns
    ranked alerts sorted by signal strength.

    **Sample request:**
    ```
    GET /universe/scan?index=NIFTY50&min_score=0.3
    ```
    """

    def loader(ticker):
        df = _get_df(ticker, period="1y", use_mock=use_mock)
        return _enrich_with_indicators(df)

    try:
        alerts = scan_nse_universe(
            data_loader_fn=loader,
            index=index if not sector and not cap else None,
            sector=sector,
            cap=cap,
            min_score=min_score,
        )
    except Exception as exc:
        logger.exception("Universe scan error")
        raise HTTPException(status_code=500, detail=str(exc))

    return UniverseScanResponse(
        total_alerts=len(alerts),
        tickers_scanned=len(alerts),
        alerts=alerts,
    )


@app.get(
    "/universe/sector/{sector}", response_model=SectorSummaryResponse, tags=["Universe"]
)
def sector_summary(
    sector: str,
    use_mock: bool = Query(default=False),
):
    """
    Get a sector-level opportunity summary.

    Returns bullish/bearish count, average score, and top picks
    within a given sector.

    **Sample request:**
    ```
    GET /universe/sector/IT
    ```
    **Sample response:**
    ```json
    {
      "sector": "IT",
      "total_stocks": 6,
      "bullish_count": 4,
      "bearish_count": 1,
      "avg_score": 0.22,
      "sentiment": "bullish",
      "top_picks": [...]
    }
    ```
    """

    def loader(ticker):
        df = _get_df(ticker, period="1y", use_mock=use_mock)
        return _enrich_with_indicators(df)

    try:
        summary = get_sector_summary(data_loader_fn=loader, sector=sector)
    except Exception as exc:
        logger.exception("Sector summary error")
        raise HTTPException(status_code=500, detail=str(exc))

    return SectorSummaryResponse(
        sector=summary["sector"],
        total_stocks=summary["total_stocks"],
        bullish_count=summary["bullish_count"],
        bearish_count=summary["bearish_count"],
        avg_score=summary["avg_score"],
        sentiment=summary["sentiment"],
        top_picks=summary["top_picks"],
    )


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host=cfg["api"]["host"],
        port=cfg["api"]["port"],
        reload=cfg["api"]["reload"],
    )
