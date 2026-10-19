"""Naive baselines every ML model has to beat."""
from __future__ import annotations

import numpy as np
import pandas as pd

BASELINES = {
    "naive_yesterday": "lag_1",
    "naive_same_weekday_last_week": "lag_7",
    "rolling_mean_7": "rolling_mean_7",
}


def baseline_predictions(df: pd.DataFrame, name: str) -> np.ndarray:
    """Baseline forecast; if its source feature is missing fall back to the 28-day average, then 0."""
    col = BASELINES[name]
    pred = df[col].astype("float64")
    pred = pred.fillna(df["rolling_mean_28"].astype("float64")).fillna(0.0)
    return pred.to_numpy()
