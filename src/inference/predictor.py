"""
predictor.py — LSTM model loading and inference module.

Responsibilities:
- Load the serialised LSTM model trained in the Jupyter notebook
- Run forward-pass predictions on prepared feature sequences
- Return inverse-scaled price predictions

This module performs ONLY inference. Training lives exclusively in
notebooks/model_pipeline.ipynb.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union

from src.utils.helpers import get_config, setup_logger
from src.features.feature_utils import (
    load_scaler,
    prepare_latest_sequence,
    inverse_scale_price,
)

logger = setup_logger(__name__)
cfg = get_config()


# ──────────────────────────────────────────────
# Model architecture (must mirror notebook definition)
# ──────────────────────────────────────────────

class LSTMModel(nn.Module):
    """
    Multi-layer LSTM for univariate/multivariate stock price prediction.

    Architecture mirrors the definition inside model_pipeline.ipynb so that
    saved weights can be loaded without retraining.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, seq_len, input_size).

        Returns:
            Tensor of shape (batch, output_size).
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


# ──────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────

_model_cache: Optional[LSTMModel] = None
_scaler_cache = None


def load_model(model_path: Optional[str] = None) -> LSTMModel:
    """
    Load the trained LSTM from disk and cache it for subsequent calls.

    Args:
        model_path: Optional override for the model file path.

    Returns:
        LSTMModel in eval mode.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    path = Path(model_path or cfg["model"]["lstm_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. "
            "Run notebooks/model_pipeline.ipynb first."
        )

    model = LSTMModel(
        input_size=cfg["model"]["input_size"],
        hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"],
        output_size=cfg["model"]["output_size"],
        dropout=cfg["model"]["dropout"],
    )
    # FIX: weights_only=True suppresses the FutureWarning in PyTorch >= 2.0
    model.load_state_dict(
        torch.load(path, map_location=torch.device("cpu"), weights_only=True)
    )
    model.eval()
    _model_cache = model
    logger.info("LSTM model loaded from %s", path)
    return model


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────

def predict_next_price(df_features) -> dict:
    """
    Predict the next closing price for a given feature DataFrame.

    Args:
        df_features: pandas DataFrame with the same feature columns produced
                     by the notebook (Close, RSI, MACD, etc.).

    Returns:
        Dict containing:
          - 'predicted_price'  (float) — inverse-scaled next close price
          - 'scaled_output'    (float) — raw model output in [0, 1]
          - 'window_size'      (int)   — look-back window used
    """
    global _scaler_cache
    if _scaler_cache is None:
        _scaler_cache = load_scaler()

    model = load_model()
    window_size = cfg["data"]["window_size"]

    sequence = prepare_latest_sequence(
        df_features, scaler=_scaler_cache, window_size=window_size
    )
    tensor_input = torch.tensor(sequence, dtype=torch.float32)

    with torch.no_grad():
        scaled_output = model(tensor_input).item()

    price = inverse_scale_price(scaled_output, _scaler_cache)

    return {
        "predicted_price": round(price, 2),
        "scaled_output": round(scaled_output, 6),
        "window_size": window_size,
    }


def predict_sequence(scaled_X: np.ndarray) -> np.ndarray:
    """
    Run inference on a batch of pre-scaled sequences.

    Useful for backtesting across a full test set.

    Args:
        scaled_X: numpy array of shape (n, window_size, n_features) — already
                  scaled by the notebook scaler.

    Returns:
        1-D numpy array of raw (scaled) predictions, shape (n,).
    """
    model = load_model()
    tensor_input = torch.tensor(scaled_X, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        preds = model(tensor_input).cpu().numpy().flatten()

    return preds


def get_model_info() -> dict:
    """
    Return metadata about the loaded LSTM architecture.

    Returns:
        Dict with model config fields.
    """
    return {
        "architecture": "Multi-layer LSTM",
        "input_size": cfg["model"]["input_size"],
        "hidden_size": cfg["model"]["hidden_size"],
        "num_layers": cfg["model"]["num_layers"],
        "output_size": cfg["model"]["output_size"],
        "dropout": cfg["model"]["dropout"],
        "framework": "PyTorch",
    }
