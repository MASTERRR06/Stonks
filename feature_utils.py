"""
feature_utils.py — Feature transformation utilities for inference-time use.

Responsibilities:
- Apply the SAME feature transformations as in the notebook to NEW incoming data
- Load and apply the saved MinMaxScaler
- Prepare sliding-window sequences for LSTM input

IMPORTANT: This module does NOT define features — features are defined and
           engineered in notebooks/model_pipeline.ipynb. This module only
           APPLIES those transformations to fresh data at inference time.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple, Optional

from src.utils.helpers import get_config, setup_logger

logger = setup_logger(__name__)
cfg = get_config()


# ──────────────────────────────────────────────
# Scaler utilities
# ──────────────────────────────────────────────

def load_scaler():
    """
    Load the MinMaxScaler saved by the Jupyter notebook.

    Returns:
        Fitted sklearn MinMaxScaler instance.

    Raises:
        FileNotFoundError: If the scaler artifact does not exist.
    """
    path = Path(cfg["model"]["scaler_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {path}. Run notebooks/model_pipeline.ipynb first."
        )
    scaler = joblib.load(path)
    logger.info("Scaler loaded from %s", path)
    return scaler


def scale_features(df: pd.DataFrame, scaler=None) -> np.ndarray:
    """
    Apply the notebook-fitted scaler to a feature DataFrame.

    Args:
        df:      DataFrame of raw features (same columns as notebook output).
        scaler:  Optional pre-loaded scaler. If None, loads from disk.

    Returns:
        2-D numpy array of scaled features, shape (n_samples, n_features).
    """
    if scaler is None:
        scaler = load_scaler()
    scaled = scaler.transform(df.values)
    return scaled


# ──────────────────────────────────────────────
# Sequence preparation
# ──────────────────────────────────────────────

def make_sequences(
    scaled_data: np.ndarray,
    window_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a scaled feature array into (X, y) sliding-window sequences.

    Matches exactly the windowing scheme used during training in the notebook.

    Args:
        scaled_data:  2-D array, shape (n_samples, n_features).
        window_size:  Look-back window length. Defaults to config value.

    Returns:
        Tuple of:
          X — shape (n_sequences, window_size, n_features)
          y — shape (n_sequences,), the target Close price at each step
    """
    window_size = window_size or cfg["data"]["window_size"]
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size : i])
        y.append(scaled_data[i, 0])   # index 0 = 'Close' after notebook ordering
    return np.array(X), np.array(y)


def prepare_latest_sequence(
    df: pd.DataFrame,
    scaler=None,
    window_size: Optional[int] = None,
) -> np.ndarray:
    """
    Prepare the most recent window for a single-step forward prediction.

    Args:
        df:          DataFrame with the same feature columns as training data.
        scaler:      Optional pre-loaded scaler.
        window_size: Look-back window. Defaults to config value.

    Returns:
        3-D numpy array of shape (1, window_size, n_features) — ready for
        LSTM inference.

    Raises:
        ValueError: If df has fewer rows than window_size.
    """
    window_size = window_size or cfg["data"]["window_size"]
    if len(df) < window_size:
        raise ValueError(
            f"DataFrame has {len(df)} rows but window_size={window_size}. "
            "Provide at least window_size rows."
        )
    scaled = scale_features(df, scaler)
    last_window = scaled[-window_size:].reshape(1, window_size, scaled.shape[1])
    return last_window


# ──────────────────────────────────────────────
# Feature column validation
# ──────────────────────────────────────────────

EXPECTED_FEATURE_COLUMNS = [
    "Close", "Open", "High", "Low", "Volume",
    "RSI", "MACD", "MACD_Signal", "BB_Upper", "BB_Lower",
]


def validate_feature_columns(df: pd.DataFrame) -> bool:
    """
    Verify that an inference DataFrame contains all expected feature columns.

    Args:
        df: DataFrame to validate.

    Returns:
        True if all required columns are present.

    Raises:
        ValueError: Listing any missing columns.
    """
    missing = set(EXPECTED_FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns for inference: {sorted(missing)}")
    return True


def inverse_scale_price(scaled_value: np.ndarray, scaler) -> float:
    """
    Convert a single scaled 'Close' value back to the original price scale.

    Args:
        scaled_value: 1-D array or scalar from model output.
        scaler:       Fitted MinMaxScaler (must have been fit on all features).

    Returns:
        Original-scale price (float).
    """
    n_features = scaler.scale_.shape[0]
    dummy = np.zeros((1, n_features))
    dummy[0, 0] = float(scaled_value)
    original = scaler.inverse_transform(dummy)
    return float(original[0, 0])
