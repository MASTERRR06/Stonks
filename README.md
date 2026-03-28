# 🇮🇳 NSE AI Stock Analysis & Forecasting System

A production-grade AI system for Indian equity markets (NSE), combining LSTM price forecasting, FinBERT sentiment analysis, technical pattern detection, and composite opportunity signal generation — all exposed via a FastAPI REST API.

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
│  ingestion/data_loader.py   ←  fetch / load OHLCV data  │
│  features/feature_utils.py  ←  apply scaler + windowing │
│  inference/predictor.py     ←  LSTM forward pass        │
│  nlp/finbert_sentiment.py   ←  FinBERT inference        │
│  technicals/pattern_detection.py  ←  S/R, breakouts     │
│  signals/opportunity_radar.py     ←  composite alerts   │
│  api/app.py                 ←  FastAPI REST endpoints    │
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
nse-ai/
├── notebooks/
│   └── model_pipeline.ipynb          ← Training pipeline
├── models/
│   └── saved_models/
│       ├── lstm_model.pt             ← Trained LSTM weights
│       └── scaler.pkl                ← Fitted MinMaxScaler
├── data/
│   ├── raw/                          ← Raw OHLCV CSVs
│   └── processed/
│       ├── processed_stock_data.csv  ← Feature-engineered data
│       └── sentiment_output.csv      ← FinBERT batch output
├── src/
│   ├── ingestion/data_loader.py
│   ├── features/feature_utils.py
│   ├── inference/predictor.py
│   ├── signals/opportunity_radar.py
│   ├── technicals/pattern_detection.py
│   ├── nlp/finbert_sentiment.py
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
git clone <repo-url> && cd nse-ai
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the notebook (trains & saves all artefacts)

```bash
jupyter notebook notebooks/model_pipeline.ipynb
```

Execute all cells top to bottom. This will:
- Fetch 2 years of RELIANCE.NS data (or use mock data offline)
- Engineer 10+ features (RSI, MACD, Bollinger Bands, lags)
- Train a 2-layer LSTM for 50 epochs with early stopping
- Run FinBERT on sample financial headlines
- Save: `lstm_model.pt`, `scaler.pkl`, `processed_stock_data.csv`, `sentiment_output.csv`

### 3. Start the API

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
  -d '{"ticker": "RELIANCE", "period": "6mo", "use_mock": false}'
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
FinBERT sentiment classification.

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
    },
    {
      "text": "Nifty50 crashes 3% amid global sell-off",
      "label": "negative",
      "score": -1.0,
      "confidence": 0.8876,
      "probabilities": {"positive": 0.0512, "negative": 0.8876, "neutral": 0.0612}
    }
  ],
  "composite_signal": 0.027,
  "signal_label": "neutral"
}
```

---

### `GET /signals/{ticker}`
Composite Opportunity Radar alert.

```bash
curl "http://localhost:8000/signals/RELIANCE?period=6mo&include_news=true"
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
Chart pattern detection.

```bash
curl "http://localhost:8000/patterns/TCS?period=1y"
```
```json
{
  "ticker": "TCS",
  "support": [3200.0, 3350.0, 3480.0],
  "resistance": [3600.0, 3720.0, 3800.0],
  "breakouts": [
    {
      "date": "2024-10-15",
      "type": "bullish_breakout",
      "price": 3742.0,
      "level": 3600.0,
      "pct_through": 3.94
    }
  ],
  "trend_reversals": [
    {
      "date": "2024-09-20",
      "type": "bullish_reversal",
      "price": 3312.0,
      "rsi": 31.4
    }
  ],
  "head_and_shoulders": []
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

All endpoints accept `?use_mock=true` (or `"use_mock": true` in the JSON body) to bypass live data fetching — useful for CI/CD or demos.

```bash
curl "http://localhost:8000/signals/INFY?use_mock=true"
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "TCS", "use_mock": true}'
```

---

## 🤖 Module Summary

| Module | Purpose |
|---|---|
| `data_loader.py` | NSE OHLCV ingestion (yfinance + mock) |
| `feature_utils.py` | Apply saved scaler + sliding-window builder |
| `predictor.py` | Load & run LSTM inference |
| `finbert_sentiment.py` | FinBERT classification + score aggregation |
| `pattern_detection.py` | S/R, breakouts, reversals, H&S detection |
| `opportunity_radar.py` | Composite BUY/SELL/WATCH signal engine |
| `app.py` | FastAPI REST API (4 endpoints) |
| `helpers.py` | Config, logging, path utilities |

---

## 📦 Key Dependencies

- **PyTorch 2.3** — LSTM training & inference
- **HuggingFace Transformers** — ProsusAI/finbert
- **yfinance** — NSE historical data
- **ta** — Technical indicators (RSI, MACD, Bollinger Bands, ATR)
- **FastAPI + uvicorn** — REST API
- **scikit-learn** — MinMaxScaler
- **pandas, numpy** — Data manipulation

---

## 📜 Licence

MIT — free to use, modify, and distribute.
