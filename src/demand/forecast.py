"""Recursive multi-step forecasting: predict day 1, feed it back as history, predict day 2, ..."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .store_calendar import FLAG_COLUMNS, build_calendar
from .config import HORIZON, SERIES_KEYS, SILVER_COLUMNS
from .features import build_features
from .models import predict

STATIC_COLUMNS = ["city", "state", "store_type", "cluster"]


def future_dates(last_date, horizon: int = HORIZON) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(last_date) + pd.Timedelta(days=1), periods=horizon, freq="D")


def build_future_frame(history: pd.DataFrame, dates, test_promo: pd.DataFrame | None,
                       holidays: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """One row per series per future date with everything that is KNOWN in advance:
    store attributes, promotions (from the competition's test.csv), holiday calendar.
    Sales, transactions and oil are unknown (NaN)."""
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"])
    series = history.sort_values("date").groupby(SERIES_KEYS, as_index=False).last()[SERIES_KEYS + STATIC_COLUMNS]
    dates = pd.DatetimeIndex(dates)
    fut = series.merge(pd.DataFrame({"date": dates}), how="cross")

    if test_promo is not None and len(test_promo):
        tp = test_promo[["date", "store_nbr", "family", "onpromotion"]].copy()
        tp["date"] = pd.to_datetime(tp["date"])
        fut = fut.merge(tp, on=["date", "store_nbr", "family"], how="left")
    else:
        fut["onpromotion"] = np.nan
    # promotions unknown for a series -> assume its average over the last 28 days of history
    recent = history[history["date"] > history["date"].max() - pd.Timedelta(days=28)]
    fallback = recent.groupby(SERIES_KEYS)["onpromotion"].mean().rename("_promo_fallback").reset_index()
    fut = fut.merge(fallback, on=SERIES_KEYS, how="left")
    fut["onpromotion"] = fut["onpromotion"].fillna(fut["_promo_fallback"]).fillna(0).round().astype("int32")
    fut = fut.drop(columns="_promo_fallback")

    cal = build_calendar(holidays, stores, dates.min(), dates.max())
    fut = fut.merge(cal, on=["date", "store_nbr"], how="left")
    for c in FLAG_COLUMNS:
        fut[c] = fut[c].fillna(0).astype("int32")
    fut["sales"] = np.nan
    fut["transactions"] = np.nan
    fut["oil_price"] = np.nan
    return fut[SILVER_COLUMNS]


def recursive_forecast(model, history: pd.DataFrame, future: pd.DataFrame, lookback_days: int = 70) -> pd.DataFrame:
    """Forecast every future date in order; each day's prediction becomes 'sales' history for the next day."""
    hist = history[SILVER_COLUMNS].copy()
    hist["date"] = pd.to_datetime(hist["date"])
    future = future[SILVER_COLUMNS].copy()
    future["date"] = pd.to_datetime(future["date"])
    dates = sorted(future["date"].unique())
    if len(dates) > HORIZON:
        raise ValueError(f"Horizon {len(dates)} > {HORIZON}: past-covariate features are only valid up to {HORIZON} days.")
    if lookback_days < 40:
        raise ValueError("lookback_days must be >= 40 to compute 28-day rolling features and 7-day-lagged covariates.")
    if hist["date"].max() >= min(dates):
        raise ValueError("Future dates must be after the last history date.")

    cutoff = hist["date"].max() - pd.Timedelta(days=lookback_days)
    frame = hist[hist["date"] > cutoff].reset_index(drop=True)
    out = []
    for step, d in enumerate(dates, start=1):
        frame = pd.concat([frame, future[future["date"] == d]], ignore_index=True)
        feats = build_features(frame)
        x = feats[feats["date"] == d].reset_index(drop=True)
        pred = predict(model, x)
        preds = x[SERIES_KEYS].assign(_pred=pred)
        mask = frame["date"] == d
        frame.loc[mask, "sales"] = frame.loc[mask, SERIES_KEYS].merge(preds, on=SERIES_KEYS, how="left")["_pred"].to_numpy()
        out.append(preds.assign(forecast_date=pd.Timestamp(d), horizon_day=step))
    res = pd.concat(out, ignore_index=True).rename(columns={"_pred": "predicted_sales"})
    return res[["forecast_date", "store_nbr", "family", "predicted_sales", "horizon_day"]]
