"""Pandas version of the Bronze -> Silver step (used for local runs and as the reference in tests)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .store_calendar import FLAG_COLUMNS, build_calendar
from .config import SILVER_COLUMNS

RAW_FILES = {
    "train": "train.csv", "test": "test.csv", "stores": "stores.csv", "oil": "oil.csv",
    "holidays_events": "holidays_events.csv", "transactions": "transactions.csv",
}


def read_raw(raw_dir) -> dict[str, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    missing = [f for f in RAW_FILES.values() if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw files in {raw_dir}: {missing}")
    return {name: pd.read_csv(raw_dir / fname) for name, fname in RAW_FILES.items()}


def build_oil_daily(oil: pd.DataFrame) -> pd.DataFrame:
    """Oil is published on business days only: expand to every calendar day, forward-fill, then back-fill."""
    o = oil.copy()
    o["date"] = pd.to_datetime(o["date"])
    o = o.set_index("date")["dcoilwtico"].astype("float64")
    full = pd.date_range(o.index.min(), o.index.max(), freq="D")
    daily = o.reindex(full).ffill().bfill()
    return daily.rename("oil_price").rename_axis("date").reset_index()


def build_silver_pandas(raw: dict[str, pd.DataFrame], calendar_end=None) -> pd.DataFrame:
    sales = raw["train"].copy()
    sales["date"] = pd.to_datetime(sales["date"])
    stores = raw["stores"].rename(columns={"type": "store_type"})
    tx = raw["transactions"].copy()
    tx["date"] = pd.to_datetime(tx["date"])
    oil = build_oil_daily(raw["oil"])

    end = pd.Timestamp(calendar_end) if calendar_end is not None else sales["date"].max()
    cal = build_calendar(raw["holidays_events"], raw["stores"], sales["date"].min(), end)

    df = (sales.merge(stores[["store_nbr", "city", "state", "store_type", "cluster"]], on="store_nbr", how="left")
               .merge(tx, on=["date", "store_nbr"], how="left")
               .merge(oil, on="date", how="left")
               .merge(cal, on=["date", "store_nbr"], how="left"))
    for c in FLAG_COLUMNS:
        df[c] = df[c].fillna(0).astype("int32")
    df = df.drop_duplicates(["date", "store_nbr", "family"])
    df["sales"] = df["sales"].astype("float64")
    df["onpromotion"] = df["onpromotion"].astype("int32")
    return df[SILVER_COLUMNS].sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
