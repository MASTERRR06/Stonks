"""
finbert_sentiment.py — FinBERT-based financial sentiment analysis.

Responsibilities:
- Load the pretrained ProsusAI/finbert model via HuggingFace Transformers
- Classify financial text as positive / negative / neutral
- Return numerical sentiment scores suitable for downstream signal generation

This module performs ONLY inference. Model fine-tuning (if any) is done
exclusively in notebooks/model_pipeline.ipynb.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Union, Optional
from transformers import BertTokenizer, BertForSequenceClassification

from src.utils.helpers import get_config, setup_logger

logger = setup_logger(__name__)
cfg = get_config()

# FinBERT label mapping (ProsusAI/finbert outputs)
LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}
SCORE_MAP = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


# ──────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────

_tokenizer_cache: Optional[BertTokenizer] = None
_model_cache: Optional[BertForSequenceClassification] = None


def load_finbert(model_name: Optional[str] = None):
    """
    Load FinBERT tokenizer and classification model (cached after first call).

    Args:
        model_name: HuggingFace model identifier. Defaults to
                    'ProsusAI/finbert' from config.

    Returns:
        Tuple (tokenizer, model) — model is in eval mode.
    """
    global _tokenizer_cache, _model_cache

    if _tokenizer_cache is not None and _model_cache is not None:
        return _tokenizer_cache, _model_cache

    name = model_name or cfg["model"]["finbert_model_name"]
    logger.info("Loading FinBERT from '%s' …", name)

    _tokenizer_cache = BertTokenizer.from_pretrained(name)
    _model_cache = BertForSequenceClassification.from_pretrained(name)
    _model_cache.eval()

    logger.info("FinBERT loaded successfully.")
    return _tokenizer_cache, _model_cache


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────

def predict_sentiment(texts: Union[str, List[str]]) -> List[dict]:
    """
    Run FinBERT sentiment classification on one or more financial texts.

    Args:
        texts: A single headline string, or a list of headline strings.

    Returns:
        List of dicts, one per input text:
          - 'text'       (str)   — original input
          - 'label'      (str)   — 'positive' | 'negative' | 'neutral'
          - 'score'      (float) — numerical mapping: +1, -1, 0
          - 'confidence' (float) — softmax probability of predicted label
          - 'probabilities' (dict) — full {label: prob} mapping
    """
    if isinstance(texts, str):
        texts = [texts]

    tokenizer, model = load_finbert()
    results = []

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=1).squeeze().tolist()
        pred_idx = int(torch.argmax(logits, dim=1).item())
        label = LABEL_MAP[pred_idx]

        prob_dict = {LABEL_MAP[i]: round(probs[i], 4) for i in range(len(probs))}

        results.append(
            {
                "text": text,
                "label": label,
                "score": SCORE_MAP[label],
                "confidence": round(probs[pred_idx], 4),
                "probabilities": prob_dict,
            }
        )

    logger.info("Sentiment analysis complete for %d texts.", len(texts))
    return results


def sentiment_to_signal(sentiment_results: List[dict]) -> float:
    """
    Aggregate a batch of sentiment predictions into a single composite score.

    The composite score is the confidence-weighted average of numerical scores,
    clamped to [-1, 1].

    Args:
        sentiment_results: Output from predict_sentiment().

    Returns:
        Float in [-1, 1]: positive → bullish signal, negative → bearish.
    """
    if not sentiment_results:
        return 0.0

    total_weight = sum(r["confidence"] for r in sentiment_results)
    if total_weight == 0:
        return 0.0

    weighted_score = sum(
        r["score"] * r["confidence"] for r in sentiment_results
    )
    return round(weighted_score / total_weight, 4)


def analyse_news_dataframe(df: pd.DataFrame, text_col: str = "headline") -> pd.DataFrame:
    """
    Run FinBERT on every row of a news DataFrame and append sentiment columns.

    Args:
        df:       DataFrame with a text column.
        text_col: Name of the column containing headline/body text.

    Returns:
        Original DataFrame augmented with:
          - 'sentiment_label'   (str)
          - 'sentiment_score'   (float) — +1 / -1 / 0
          - 'sentiment_confidence' (float)
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame.")

    texts = df[text_col].fillna("").tolist()
    results = predict_sentiment(texts)

    df = df.copy()
    df["sentiment_label"] = [r["label"] for r in results]
    df["sentiment_score"] = [r["score"] for r in results]
    df["sentiment_confidence"] = [r["confidence"] for r in results]
    return df


def load_sentiment_output() -> pd.DataFrame:
    """
    Load the pre-computed sentiment output saved by the Jupyter notebook.

    Returns:
        DataFrame with sentiment scores indexed by date.

    Raises:
        FileNotFoundError: If the notebook artefact is missing.
    """
    path = Path(cfg["model"]["sentiment_output_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Sentiment output not found at {path}. "
            "Run notebooks/model_pipeline.ipynb first."
        )
    return pd.read_csv(path, index_col=0, parse_dates=True)
