"""
app.py — FastAPI REST API for the NSE AI Stock Analysis System.

Endpoints:
  POST /predict        — LSTM next-price prediction
  POST /sentiment      — FinBERT sentiment analysis
  GET  /signals/{ticker} — Opportunity Radar composite alert
  GET  /patterns/{ticker} — Technical chart pattern detection
  GET  /health         — Liveness check

Sample requests are included in the docstrings below and in README.md.
"""

import os
import sys
from pathlib import Path

# ── Ensure project root is on sys.path when running directly ──
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

cfg = get_config()
logger = setup_logger(__name__)

# ──────────────────────────────────────────────
# App initialisation
# ──────────────────────────────────────────────

app = FastAPI(
    title=cfg["api"]["title"],
    version=cfg["api"]["version"],
    description=(
        "Production-grade AI system for NSE stock analysis. "
        "Combines LSTM forecasting, FinBERT sentiment, chart pattern detection, "
        "and opportunity signal generation."
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
    ticker: str = Field(
        default="RELIANCE",
        description="NSE ticker symbol without suffix (e.g. 'RELIANCE', 'TCS').",
        example="RELIANCE",
    )
    period: str = Field(
        default="6mo",
        description="Historical data period for context window.",
        example="6mo",
    )
    use_mock: bool = Field(
        default=False,
        description="Use synthetic data (no network required).",
    )


class PredictResponse(BaseModel):
    ticker: str
    current_price: float
    predicted_price: float
    scaled_output: float
    window_size: int
    model_info: Dict[str, Any]


class SentimentRequest(BaseModel):
    texts: List[str] = Field(
        description="List of financial headlines or news snippets.",
        example=[
            "Reliance Industries reports record quarterly profit",
            "Nifty50 crashes amid global sell-off",
        ],
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


# ──────────────────────────────────────────────
# Helper — fetch or mock data
# ──────────────────────────────────────────────

def _get_df(ticker: str, period: str = "6mo", use_mock: bool = False) -> pd.DataFrame:
    """Internal helper to fetch real or mock OHLCV data."""
    if use_mock:
        return get_mock_data(ticker=ticker)
    try:
        return fetch_nse_data(ticker=ticker, period=period, save=False)
    except Exception as exc:
        logger.warning("yfinance fetch failed (%s) — falling back to mock data", exc)
        return get_mock_data(ticker=ticker)


def _enrich_with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators inline (fallback when notebook artefacts aren't present).
    Used ONLY for demo purposes within the API; real indicator engineering is in the notebook.
    """
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
# Routes
# ──────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """
    Liveness probe endpoint.

    **Sample request:**
    ```
    GET /health
    ```
    **Sample response:**
    ```json
    {"status": "ok", "version": "1.0.0"}
    ```
    """
    return {"status": "ok", "version": cfg["api"]["version"]}


@app.post("/predict", response_model=PredictResponse, tags=["Forecasting"])
def predict(request: PredictRequest):
    """
    Predict the next closing price using the trained LSTM model.

    The endpoint loads the trained model from `models/saved_models/lstm_model.pt`
    (produced by running `notebooks/model_pipeline.ipynb`).

    **Sample request:**
    ```json
    {
      "ticker": "RELIANCE",
      "period": "6mo",
      "use_mock": false
    }
    ```
    **Sample response:**
    ```json
    {
      "ticker": "RELIANCE",
      "current_price": 2854.50,
      "predicted_price": 2871.22,
      "scaled_output": 0.623419,
      "window_size": 60,
      "model_info": {...}
    }
    ```
    """
    df = _get_df(request.ticker, period=request.period, use_mock=request.use_mock)
    df = _enrich_with_indicators(df)

    feature_cols = ["Close", "Open", "High", "Low", "Volume",
                    "RSI", "MACD", "MACD_Signal", "BB_Upper", "BB_Lower"]
    available = [c for c in feature_cols if c in df.columns]
    df_feat = df[available].dropna()

    try:
        result = predict_next_price(df_feat)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model not found. Run the notebook first. ({exc})",
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
    """
    Run FinBERT sentiment analysis on a list of financial texts.

    **Sample request:**
    ```json
    {
      "texts": [
        "Reliance Industries posts record profits",
        "Nifty50 drops 3% on global recession fears"
      ]
    }
    ```
    **Sample response:**
    ```json
    {
      "results": [
        {"text": "Reliance...", "label": "positive", "score": 1.0, "confidence": 0.94, ...},
        {"text": "Nifty50...", "label": "negative", "score": -1.0, "confidence": 0.88, ...}
      ],
      "composite_signal": 0.03,
      "signal_label": "neutral"
    }
    ```
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="'texts' list cannot be empty.")

    try:
        results = predict_sentiment(request.texts)
    except Exception as exc:
        logger.exception("Sentiment error")
        raise HTTPException(status_code=500, detail=str(exc))

    composite = sentiment_to_signal(results)
    if composite >= 0.3:
        label = "bullish"
    elif composite <= -0.3:
        label = "bearish"
    else:
        label = "neutral"

    return SentimentResponse(
        results=[SentimentResult(**r) for r in results],
        composite_signal=composite,
        signal_label=label,
    )


@app.get("/signals/{ticker}", response_model=SignalResponse, tags=["Signals"])
def signals(
    ticker: str,
    period: str = Query(default="6mo", description="Data period for analysis."),
    use_mock: bool = Query(default=False, description="Use synthetic data."),
    include_news: bool = Query(default=True, description="Include FinBERT sentiment."),
):
    """
    Generate a composite Opportunity Radar alert for a ticker.

    Combines price, volume, pattern, and sentiment signals.

    **Sample request:**
    ```
    GET /signals/RELIANCE?period=6mo&include_news=true
    ```
    **Sample response:**
    ```json
    {
      "ticker": "RELIANCE",
      "alert_type": "BUY",
      "score": 0.48,
      "details": "Score=+0.48 | Positive news sentiment; Volume spike detected",
      ...
    }
    ```
    """
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
        feature_cols = ["Close", "Open", "High", "Low", "Volume",
                        "RSI", "MACD", "MACD_Signal", "BB_Upper", "BB_Lower"]
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
    period: str = Query(default="1y", description="Data period for pattern detection."),
    use_mock: bool = Query(default=False, description="Use synthetic data."),
):
    """
    Detect chart patterns for a given NSE ticker.

    **Sample request:**
    ```
    GET /patterns/TCS?period=1y
    ```
    **Sample response:**
    ```json
    {
      "ticker": "TCS",
      "support": [3200.0, 3350.0],
      "resistance": [3600.0, 3750.0],
      "breakouts": [...],
      "trend_reversals": [...],
      "head_and_shoulders": [...]
    }
    ```
    """
    df = _get_df(ticker, period=period, use_mock=use_mock)
    df = _enrich_with_indicators(df)
    result = get_all_patterns(df)

    return PatternResponse(
        ticker=ticker,
        support=result["support_resistance"]["support"],
        resistance=result["support_resistance"]["resistance"],
        breakouts=result["breakouts"],
        trend_reversals=result["trend_reversals"],
        head_and_shoulders=result["head_and_shoulders"],
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
