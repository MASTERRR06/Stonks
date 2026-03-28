"""
helpers.py — General-purpose utility functions for the NSE AI system.

Responsibilities:
- Configuration loading
- Logging setup
- Common path resolution
- Date/time helpers
- Data validation utilities
"""

import logging
import os
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, Optional


# ──────────────────────────────────────────────
# Path resolution
# ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return PROJECT_ROOT


def resolve_path(relative: str) -> Path:
    """
    Resolve a path string relative to the project root.

    Args:
        relative: Relative path string (e.g. 'data/raw').

    Returns:
        Absolute Path object.
    """
    return PROJECT_ROOT / relative


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load and return the YAML configuration file.

    Args:
        config_path: Optional override for config file path.

    Returns:
        Dictionary of configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = config_path or CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


_config_cache: Optional[Dict[str, Any]] = None


def get_config() -> Dict[str, Any]:
    """Return a cached copy of the configuration (loads once per process)."""
    global _config_cache
    if _config_cache is None:
        _config_cache = load_config()
    return _config_cache


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Create and configure a named logger.

    Args:
        name: Logger name (typically __name__ of calling module).
        level: Optional log level override ('DEBUG', 'INFO', etc.).

    Returns:
        Configured Logger instance.
    """
    cfg = get_config()
    log_level = level or cfg.get("logging", {}).get("level", "INFO")
    log_format = cfg.get("logging", {}).get(
        "format", "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    return logger


# ──────────────────────────────────────────────
# Date / time helpers
# ──────────────────────────────────────────────

def get_date_range(period: str = "1y") -> Dict[str, str]:
    """
    Convert a yfinance-style period string to explicit start/end dates.

    Args:
        period: String like '1y', '6mo', '2y', '3mo', '5d'.

    Returns:
        Dict with 'start' and 'end' date strings in YYYY-MM-DD format.

    Raises:
        ValueError: If the period string format is unrecognised.
    """
    end = datetime.today()
    unit_map = {"d": 1, "mo": 30, "y": 365}
    for suffix, days_per_unit in unit_map.items():
        if period.endswith(suffix):
            try:
                qty = int(period[: -len(suffix)])
                start = end - timedelta(days=qty * days_per_unit)
                return {
                    "start": start.strftime("%Y-%m-%d"),
                    "end": end.strftime("%Y-%m-%d"),
                }
            except ValueError:
                pass
    raise ValueError(f"Unrecognised period format: '{period}'. Use e.g. '1y', '6mo', '30d'.")


def fmt_date(dt: datetime) -> str:
    """Return a datetime formatted as YYYY-MM-DD."""
    return dt.strftime("%Y-%m-%d")


# ──────────────────────────────────────────────
# Data validation
# ──────────────────────────────────────────────

def validate_ohlcv_columns(df_columns) -> bool:
    """
    Check that a DataFrame contains the required OHLCV columns.

    Args:
        df_columns: Column index of the DataFrame.

    Returns:
        True if all required columns are present, False otherwise.
    """
    required = {"Open", "High", "Low", "Close", "Volume"}
    return required.issubset(set(df_columns))


def ensure_dir(path: str) -> Path:
    """
    Create a directory (and parents) if it does not already exist.

    Args:
        path: Directory path string.

    Returns:
        Path object of the created/existing directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ──────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────

def format_signal(signal: Dict[str, Any]) -> str:
    """
    Render an opportunity signal dictionary as a human-readable string.

    Args:
        signal: Dict with keys 'ticker', 'alert_type', 'score', 'details'.

    Returns:
        Formatted multi-line string.
    """
    lines = [
        f"{'='*50}",
        f"  SIGNAL → {signal.get('ticker', 'N/A')}",
        f"  Type   : {signal.get('alert_type', 'N/A')}",
        f"  Score  : {signal.get('score', 0):.2f}",
        f"  Detail : {signal.get('details', '')}",
        f"{'='*50}",
    ]
    return "\n".join(lines)
