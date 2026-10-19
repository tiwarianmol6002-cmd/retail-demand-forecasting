"""Forecast accuracy metrics (all take plain array-likes)."""
from __future__ import annotations

import numpy as np


def _arrays(y_true, y_pred):
    return np.asarray(y_true, dtype="float64"), np.asarray(y_pred, dtype="float64")


def mae(y_true, y_pred) -> float:
    y, p = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(y - p)))


def rmse(y_true, y_pred) -> float:
    y, p = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error = sum|e| / sum|y| (business-friendly, safe with zeros)."""
    y, p = _arrays(y_true, y_pred)
    denom = np.sum(np.abs(y))
    return float(np.sum(np.abs(y - p)) / denom) if denom > 0 else float("nan")


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in [0, 2]; a day where both actual and forecast are 0 counts as perfect."""
    y, p = _arrays(y_true, y_pred)
    denom = np.abs(y) + np.abs(p)
    ratio = np.divide(2 * np.abs(y - p), denom, out=np.zeros_like(denom), where=denom > 0)
    return float(np.mean(ratio))


def rmsle(y_true, y_pred) -> float:
    """Root mean squared log error (the Kaggle competition metric); negatives are clipped to 0."""
    y, p = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((np.log1p(np.clip(p, 0, None)) - np.log1p(np.clip(y, 0, None))) ** 2)))


def evaluate(y_true, y_pred, prefix: str = "") -> dict:
    return {
        f"{prefix}mae": mae(y_true, y_pred),
        f"{prefix}rmse": rmse(y_true, y_pred),
        f"{prefix}wape": wape(y_true, y_pred),
        f"{prefix}smape": smape(y_true, y_pred),
        f"{prefix}rmsle": rmsle(y_true, y_pred),
    }
