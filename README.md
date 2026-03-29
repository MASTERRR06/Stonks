# 🇮🇳 STONKS — NSE AI Stock Analysis & Forecasting System

A production-grade AI system for Indian equity markets (NSE), combining LSTM price forecasting, FinBERT sentiment analysis, technical pattern detection, corporate filings intelligence, pattern back-testing, and full NSE universe scanning — all exposed via a FastAPI REST API.

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────┐
│              notebooks/model_pipeline.ipynb             │
│  • Data preprocessing  • Feature engineering           │
│  • LSTM training       • FinBERT sentiment pipeline     │
│  • Artefact saving (model.pt, scaler.pkl, CSVs)         │
└───────────────────────┬─────────────────────────────────┘
                        │ artefacts
        ┌───────────────▼───────────────┐
        │     models/saved_models/      │
        │  lstm_model.pt  scaler.pkl    │
        └───────────────┬───────────────┘
                        │ load
┌───────────────────────▼──────────────────────────────────┐
│                    src/ modules                          │
│                                                          │
│  ingestion/data_loader.py      ← Alpha Vantage OHLCV    │
│  features/feature_utils.py     ← scaler + windowing     │
│  inference/predictor.py        ← LSTM forward pass      │
│  nlp/finbert_sentiment.py      ← FinBERT inference      │
│  technicals/pattern_detection.py ← S/R, breakouts       │
│  signals/opportunity_radar.py  ← composite alerts       │
│  signals/nse_universe.py       ← full NSE universe scan │
│  filings/filings.py            ← insider trades, deals  │
│  backtesting/backtester.py     ← pattern success rates  │
│  api/app.py                    ← FastAPI REST (8 endpoints) │
└──────────────────────────────────────────────────────────┘
```

**Strict separation of concerns:**
| Concern | Location |
|---|---|
| Preprocessing, feature engineering, training | `notebooks/model_pipeline.ipynb` ONLY |
| Inference, API, signal logic | `src/` modules ONLY |

---

## 📁 Project Structure

```
Stonks/
├── .env                              ← API credentials (never commit)
├── notebooks/
│   └── model_pipeline.ipynb         ← Training pipeline
├── models/
│   └── saved_models/
│       ├── lstm_model.pt            ← Trained LSTM weights
│       └── scaler.pkl               ← Fitted MinMaxScaler
├── data/
│   ├── raw/                         ← Raw OHLCV CSVs
│   └── processed/
│       ├── processed_stock_data.csv ← Feature-engineered data
│       └── sentiment_output.csv     ← FinBERT batch output
├── src/
│   ├── ingestion/data_loader.py     ← Alpha Vantage data fetcher
│   ├── features/feature_utils.py
│   ├── inference/predictor.py
│   ├── nlp/finbert_sentiment.py
│   ├── technicals/pattern_detection.py
│   ├── signals/
│   │   ├── opportunity_radar.py
│   │   └── nse_universe.py          ← NSE universe scanner
│   ├── filings/filings.py           ← Corporate filings & insider trades
│   ├── backtesting/backtester.py    ← Pattern back-testing engine
│   ├── api/app.py
│   └── utils/helpers.py
├── config/config.yaml
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & install

```bash
git clone <repo-url> && cd Stonks
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up credentials

Copy `.env.template` to `.env` and fill in your API keys:

```env
# Alpha Vantage (Historical Data)
# Free key at: https://www.alphavantage.co/support/#api-key
ALPHAVANTAGE_API_KEY=your_api_key_here

# Fyers (Real-time Data)
FYERS_APP_ID=your_app_id_here
FYERS_SECRET_KEY=your_secret_key_here
FYERS_ACCESS_TOKEN=your_access_token_here
```

### 3. Run the notebook (trains & saves all artefacts)

```bash
jupyter notebook notebooks/model_pipeline.ipynb
```

Execute all cells top to bottom. This will:
- Fetch historical OHLCV data via Alpha Vantage
- Engineer 10+ features (RSI, MACD, Bollinger Bands, ATR, lags)
- Train a 2-layer LSTM for 50 epochs with early stopping
- Run FinBERT on sample financial headlines
- Save: `lstm_model.pt`, `scaler.pkl`, `processed_stock_data.csv`, `sentiment_output.csv`

### 4. Start the API

```bash
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs: **http://localhost:8000/docs**

---

## 🌐 API Reference

### `GET /health`
Liveness check.

```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "version": "1.0.0"}
```

---

### `POST /predict`
LSTM next-close-price prediction.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "RELIANCE", "period": "1y", "use_mock": false}'
```
```json
{
  "ticker": "RELIANCE",
  "current_price": 2854.50,
  "predicted_price": 2871.22,
  "scaled_output": 0.623419,
  "window_size": 60,
  "model_info": {
    "architecture": "Multi-layer LSTM",
    "input_size": 10,
    "hidden_size": 128,
    "num_layers": 2,
    "framework": "PyTorch"
  }
}
```

---

### `POST /sentiment`
FinBERT sentiment classification on financial text.

```bash
curl -X POST http://localhost:8000/sentiment \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Reliance Industries reports record quarterly profit",
      "Nifty50 crashes 3% amid global sell-off"
    ]
  }'
```
```json
{
  "results": [
    {
      "text": "Reliance Industries reports record quarterly profit",
      "label": "positive",
      "score": 1.0,
      "confidence": 0.9421,
      "probabilities": {"positive": 0.9421, "negative": 0.0312, "neutral": 0.0267}
    }
  ],
  "composite_signal": 0.027,
  "signal_label": "neutral"
}
```

---

### `GET /signals/{ticker}`
Composite Opportunity Radar alert — combines price, volume, sentiment, and pattern signals.

```bash
curl "http://localhost:8000/signals/RELIANCE?period=1y&include_news=true"
```
```json
{
  "ticker": "RELIANCE",
  "timestamp": "2024-11-15T09:30:00Z",
  "current_price": 2854.50,
  "predicted_price": 2871.22,
  "alert_type": "BUY",
  "score": 0.48,
  "components": {
    "sentiment": 0.24,
    "momentum": 0.15,
    "volume_spike": 0.15,
    "breakout": 0.0,
    "reversal": 0.0
  },
  "details": "Score=+0.48 | Positive news sentiment (+0.80); Volume spike detected",
  "signals": [
    "Positive news sentiment (+0.80)",
    "Volume spike detected",
    "Strong upward price momentum (+6.0%)"
  ]
}
```

---

### `GET /patterns/{ticker}`
Chart pattern detection with back-tested success rates.

```bash
curl "http://localhost:8000/patterns/TCS?period=1y&include_backtest=true"
```
```json
{
  "ticker": "TCS",
  "support": [3200.0, 3350.0, 3480.0],
  "resistance": [3600.0, 3720.0, 3800.0],
  "breakouts": [
    {"date": "2024-10-15", "type": "bullish_breakout", "price": 3742.0, "level": 3600.0, "pct_through": 3.94}
  ],
  "trend_reversals": [
    {"date": "2024-09-20", "type": "bullish_reversal", "price": 3312.0, "rsi": 31.4}
  ],
  "head_and_shoulders": [],
  "success_rates": [
    {"pattern": "bullish_breakout", "win_rate_pct": 64.3, "avg_return_pct": 2.8,
     "total_signals": 14, "verdict": "Strong signal — historically reliable"}
  ]
}
```

---

### `GET /filings/{ticker}`
Corporate filings intelligence — bulk/block deals, insider trades, announcements.

```bash
curl "http://localhost:8000/filings/RELIANCE"
```
```json
{
  "ticker": "RELIANCE",
  "total_signals": 3,
  "composite_score": 0.35,
  "signals": [
    {"type": "insider_trade", "description": "Insider bought 50,000 shares",
     "score": 0.4, "date": "2024-11-10"},
    {"type": "bulk_block_deal", "description": "BUY deal by XYZ Fund — 1,000,000 shares @ ₹2850",
     "score": 0.3, "date": "2024-11-12"},
    {"type": "corporate_announcement", "description": "Reliance declares interim dividend",
     "score": 0.3, "date": "2024-11-14"}
  ]
}
```

---

### `GET /backtest/{ticker}`
Pattern back-test — historical win rate and avg return per pattern on a specific stock.

```bash
curl "http://localhost:8000/backtest/RELIANCE?period=2y&holding_days=10"
```
```json
{
  "ticker": "RELIANCE",
  "holding_days": 10,
  "results": [
    {"pattern": "bullish_breakout", "win_rate_pct": 64.3, "avg_return_pct": 2.8,
     "total_signals": 14, "verdict": "Strong signal — historically reliable"},
    {"pattern": "bullish_reversal", "win_rate_pct": 55.0, "avg_return_pct": 1.2,
     "total_signals": 20, "verdict": "Moderate signal — use with confirmation"},
    {"pattern": "head_and_shoulders", "win_rate_pct": 42.0, "avg_return_pct": -1.5,
     "total_signals": 7, "verdict": "Weak signal — treat with caution"}
  ]
}
```

---

### `GET /universe/scan`
Scan the full NSE universe for opportunities — ranked by signal strength.

```bash
curl "http://localhost:8000/universe/scan?index=NIFTY50&min_score=0.3"
```
```json
{
  "total_alerts": 8,
  "tickers_scanned": 30,
  "alerts": [
    {"ticker": "TCS", "alert_type": "BUY", "score": 0.52, "sector": "IT", ...},
    {"ticker": "HDFCBANK", "alert_type": "BUY", "score": 0.41, "sector": "Banking", ...}
  ]
}
```

---

### `GET /universe/sector/{sector}`
Sector-level opportunity summary — bullish/bearish breakdown with top picks.

```bash
curl "http://localhost:8000/universe/sector/IT"
```
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

---

## ⚙️ Configuration

Edit `config/config.yaml` to adjust:

| Key | Description | Default |
|---|---|---|
| `data.window_size` | LSTM look-back window | 60 |
| `model.hidden_size` | LSTM hidden units | 128 |
| `model.num_layers` | LSTM depth | 2 |
| `signals.volume_spike_threshold` | Volume spike multiplier | 2.0× |
| `signals.price_breakout_pct` | Breakout penetration | 3% |
| `api.port` | API port | 8000 |

---

## 🧪 Testing with mock data

All endpoints accept `?use_mock=true` to bypass live data fetching — useful for CI/CD or demos without API keys.

```bash
curl "http://localhost:8000/signals/INFY?use_mock=true"
curl "http://localhost:8000/patterns/TCS?use_mock=true&include_backtest=true"
curl "http://localhost:8000/universe/scan?use_mock=true"
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "TCS", "use_mock": true}'
```

---

## 🤖 Module Summary

| Module | Purpose |
|---|---|
| `data_loader.py` | Alpha Vantage historical OHLCV fetcher + mock data |
| `feature_utils.py` | Apply saved scaler + sliding-window builder |
| `predictor.py` | Load & run LSTM inference |
| `finbert_sentiment.py` | FinBERT classification + score aggregation |
| `pattern_detection.py` | S/R levels, breakouts, reversals, H&S detection |
| `opportunity_radar.py` | Composite BUY/SELL/WATCH signal engine |
| `nse_universe.py` | Full NSE universe scanner (Nifty50/Midcap/500) |
| `filings.py` | Bulk/block deals, insider trades, BSE announcements |
| `backtester.py` | Historical pattern win rates per stock |
| `app.py` | FastAPI REST API (8 endpoints) |
| `helpers.py` | Config, logging, path utilities |

---

## 📦 Key Dependencies

- **PyTorch 2.3** — LSTM training & inference
- **HuggingFace Transformers** — ProsusAI/finbert
- **Alpha Vantage** — NSE/BSE historical OHLCV data
- **Fyers API v3** — Real-time market feed (WebSocket)
- **ta** — Technical indicators (RSI, MACD, Bollinger Bands, ATR)
- **FastAPI + uvicorn** — REST API
- **scikit-learn** — MinMaxScaler
- **pandas, numpy** — Data manipulation
- **python-dotenv** — Secure credential management

---

## 📜 Licence

MIT — free to use, modify, and distribute.
