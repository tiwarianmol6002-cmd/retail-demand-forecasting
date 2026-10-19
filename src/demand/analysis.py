"""Error analysis helpers: where does the model do well, and where does it fail?"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_eval_frame(df: pd.DataFrame, predicted) -> pd.DataFrame:
    cols = ["date", "store_nbr", "family", "city", "store_type", "onpromotion", "is_holiday"]
    out = df[cols].copy()
    out["actual"] = df["sales"].to_numpy(dtype="float64")
    out["predicted"] = np.asarray(predicted, dtype="float64")
    out["error"] = out["predicted"] - out["actual"]
    out["abs_error"] = out["error"].abs()
    out["period_type"] = np.where(out["is_holiday"] == 1, "holiday",
                                  np.where(out["onpromotion"] > 0, "promotion", "normal"))
    return out.reset_index(drop=True)


def slice_metrics(eval_df: pd.DataFrame, by) -> pd.DataFrame:
    """MAE / RMSE / WAPE per slice (store, family, period type, ...), hardest slices first."""
    by = [by] if isinstance(by, str) else list(by)
    g = eval_df.assign(sq_error=eval_df["error"] ** 2).groupby(by, dropna=False)
    out = g.agg(n_rows=("actual", "size"), total_sales=("actual", "sum"),
                mae=("abs_error", "mean"), mse=("sq_error", "mean"), abs_error_sum=("abs_error", "sum")).reset_index()
    out["rmse"] = np.sqrt(out.pop("mse"))
    out["wape"] = np.where(out["total_sales"] > 0, out["abs_error_sum"] / out["total_sales"], np.nan)
    out = out.drop(columns="abs_error_sum")
    return out.sort_values("wape", ascending=False, na_position="last").reset_index(drop=True)


def worst_errors(eval_df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    return eval_df.sort_values("abs_error", ascending=False).head(n).reset_index(drop=True)
