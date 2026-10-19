"""Pandas feature engineering (used for local runs, recursive forecasting and the API).

The Spark implementation in ``spark_etl.py`` mirrors this logic; ``tests/test_spark_parity.py``
asserts both produce the same numbers so training and inference cannot silently drift apart.

Leakage rule: every sales-derived feature for day d only uses sales up to day d-1, and
"past covariates" (transactions, oil) are lagged by HORIZON days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    CALENDAR_FEATURES, ENGINEERED_FEATURES, LAGS, ROLL_MEAN_WINDOWS, ROLL_STD_WINDOWS,
    SERIES_KEYS, HORIZON,
)


def min_periods(window: int) -> int:
    """Minimum number of observed days required inside a rolling window."""
    return max(2, window // 2)


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    d = pd.to_datetime(out[date_col])
    out["day_of_week"] = d.dt.dayofweek.astype("int32")          # Monday=0
    out["day_of_month"] = d.dt.day.astype("int32")
    out["week_of_year"] = d.dt.isocalendar().week.astype("int32")
    out["month"] = d.dt.month.astype("int32")
    out["quarter"] = d.dt.quarter.astype("int32")
    out["year"] = d.dt.year.astype("int32")
    out["is_weekend"] = (d.dt.dayofweek >= 5).astype("int32")
    out["is_month_start"] = d.dt.is_month_start.astype("int32")
    out["is_month_end"] = d.dt.is_month_end.astype("int32")
    # Ecuadorian public-sector wages are paid on the 15th and the last day of the month
    out["is_payday"] = ((d.dt.day == 15) | d.dt.is_month_end).astype("int32")
    return out


def add_series_features(pdf: pd.DataFrame) -> pd.DataFrame:
    """Add lag / rolling / past-covariate / calendar features to ONE (store, family) series.

    ``pdf`` needs: date, sales (NaN for future rows), transactions, oil_price.
    Calendar days that have no row (e.g. 25 December) are treated as missing, so a lag of
    7 always means exactly 7 calendar days.
    """
    pdf = pdf.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(pdf["date"])
    full = pd.date_range(dates.min(), dates.max(), freq="D")

    def daily(col: str) -> pd.Series:
        s = pd.Series(pdf[col].astype("float64").to_numpy(), index=dates.to_numpy())
        return s.reindex(full)

    sales, trans, oil = daily("sales"), daily("transactions"), daily("oil_price")
    feats = pd.DataFrame(index=full)
    for k in LAGS:
        feats[f"lag_{k}"] = sales.shift(k)
    past = sales.shift(1)
    for w in ROLL_MEAN_WINDOWS:
        feats[f"rolling_mean_{w}"] = past.rolling(w, min_periods=min_periods(w)).mean()
    for w in ROLL_STD_WINDOWS:
        feats[f"rolling_std_{w}"] = past.rolling(w, min_periods=min_periods(w)).std()
    feats["transactions_lag_7"] = trans.shift(HORIZON)
    feats["transactions_roll_mean_28"] = trans.shift(HORIZON).rolling(28, min_periods=14).mean()
    feats["oil_price_lag_7"] = oil.shift(HORIZON)

    feats = feats.loc[dates.to_numpy()].reset_index(drop=True)
    out = pd.concat([pdf, feats], axis=1)
    return add_calendar_features(out)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ``add_series_features`` to every (store, family) series."""
    parts = [add_series_features(g) for _, g in df.groupby(SERIES_KEYS, sort=True)]
    if not parts:
        return df.copy()
    return pd.concat(parts, ignore_index=True)


def feature_columns_present(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in ENGINEERED_FEATURES) and all(c in df.columns for c in CALENDAR_FEATURES)
