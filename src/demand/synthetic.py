"""Synthetic data in the exact schema of the Kaggle 'Store Sales - Time Series Forecasting' files.

Used for unit tests, CI and a quick local demo. It is NOT used for any reported result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .store_calendar import build_calendar

FAMILIES = ["BEVERAGES", "PRODUCE", "DAIRY", "BREAD/BAKERY", "BOOKS"]
LOCATIONS = [("Quito", "Pichincha"), ("Guayaquil", "Guayas"), ("Cuenca", "Azuay")]


def _holidays() -> pd.DataFrame:
    rows = []
    for year in range(2015, 2018):
        rows += [
            (f"{year}-01-01", "Holiday", "National", "Ecuador", "Primer dia del ano", False),
            (f"{year}-05-01", "Holiday", "National", "Ecuador", "Dia del Trabajo", False),
            (f"{year}-12-06", "Holiday", "Local", "Quito", "Fundacion de Quito", False),
            (f"{year}-07-25", "Holiday", "Regional", "Guayas", "Fundacion de Guayaquil", False),
            (f"{year}-05-14", "Event", "National", "Ecuador", "Dia de la Madre", False),
        ]
    rows += [
        ("2016-10-09", "Holiday", "National", "Ecuador", "Independencia de Guayaquil", True),   # transferred away
        ("2016-10-12", "Transfer", "National", "Ecuador", "Traslado Independencia de Guayaquil", False),
        ("2016-11-12", "Work Day", "National", "Ecuador", "Recupero puente", False),
        ("2017-08-18", "Holiday", "National", "Ecuador", "Feriado de prueba", False),
    ]
    return pd.DataFrame(rows, columns=["date", "type", "locale", "locale_name", "description", "transferred"])


def generate_raw(n_stores: int = 6, families=FAMILIES, start="2015-09-01", end="2017-08-15",
                 horizon_days: int = 16, seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    stores = pd.DataFrame({
        "store_nbr": np.arange(1, n_stores + 1),
        "city": [LOCATIONS[i % len(LOCATIONS)][0] for i in range(n_stores)],
        "state": [LOCATIONS[i % len(LOCATIONS)][1] for i in range(n_stores)],
        "type": [("A", "B", "C", "D")[i % 4] for i in range(n_stores)],
        "cluster": [(i * 3) % 17 + 1 for i in range(n_stores)],
    })
    holidays = _holidays()

    all_dates = pd.date_range(start, end, freq="D")
    all_dates = all_dates[~((all_dates.month == 12) & (all_dates.day == 25))]  # stores closed on 25 Dec
    future_dates = pd.date_range(pd.Timestamp(end) + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    cal = build_calendar(holidays, stores, all_dates.min(), future_dates.max())
    cal = cal.set_index(["date", "store_nbr"])

    level = {(s, f): rng.uniform(2, 60) * (0.05 if f == "BOOKS" else 1.0)
             for s in stores["store_nbr"] for f in families}
    dow_mult = np.array([0.85, 0.85, 0.9, 0.95, 1.1, 1.35, 1.25])

    def make_rows(dates, with_sales: bool):
        recs = []
        for s in stores["store_nbr"]:
            for f in families:
                promo = rng.poisson(1.5 * (1 + (sum(map(ord, f)) % 3)), len(dates)) * (rng.random(len(dates)) < 0.35)
                for i, d in enumerate(dates):
                    flags = cal.loc[(d, s)]
                    sales = np.nan
                    if with_sales:
                        mu = level[(s, f)] * dow_mult[d.dayofweek] * (1 + 0.012 * min(promo[i], 30))
                        mu *= 1.12 if (d.day == 15 or d.is_month_end) else 1.0
                        mu *= 1.2 if flags["is_holiday"] else 1.0
                        mu *= 1 + 0.0004 * (d - dates[0]).days  # mild trend
                        sales = float(rng.poisson(mu))
                    recs.append((s, f, d, sales, int(promo[i])))
        return pd.DataFrame(recs, columns=["store_nbr", "family", "date", "sales", "onpromotion"])

    train = make_rows(all_dates, True).sort_values(["date", "store_nbr", "family"]).reset_index(drop=True)
    train.insert(0, "id", np.arange(len(train)))
    train = train[["id", "date", "store_nbr", "family", "sales", "onpromotion"]]           # Kaggle column order
    test = make_rows(future_dates, False).sort_values(["date", "store_nbr", "family"]).reset_index(drop=True)
    test = test.drop(columns="sales")
    test.insert(0, "id", np.arange(len(train), len(train) + len(test)))
    test = test[["id", "date", "store_nbr", "family", "onpromotion"]]

    base_tx = {s: rng.uniform(800, 2500) for s in stores["store_nbr"]}
    tx = pd.DataFrame([(d, s, float(rng.normal(base_tx[s] * dow_mult[d.dayofweek], 60)))
                       for d in all_dates for s in stores["store_nbr"]], columns=["date", "store_nbr", "transactions"])
    tx = tx[rng.random(len(tx)) > 0.02].copy()  # a few missing days, like the real file
    tx["transactions"] = tx["transactions"].round(0)

    bdays = pd.bdate_range(all_dates.min(), future_dates.max())
    oil_vals = 50 + np.cumsum(rng.normal(0, 0.6, len(bdays)))
    oil = pd.DataFrame({"date": bdays, "dcoilwtico": np.round(oil_vals, 2)})
    oil.loc[rng.random(len(oil)) < 0.03, "dcoilwtico"] = np.nan
    oil.loc[0, "dcoilwtico"] = np.nan  # leading gap to exercise back-fill

    fmt = lambda df: df.assign(date=pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"))  # noqa: E731
    return {
        "train": fmt(train), "test": fmt(test), "stores": stores, "oil": fmt(oil),
        "holidays_events": holidays, "transactions": fmt(tx),
    }


def write_raw(raw: dict[str, pd.DataFrame], out_dir) -> None:
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in raw.items():
        df.to_csv(out / f"{name}.csv", index=False)
