"""Holiday / event calendar per store, built from Kaggle's holidays_events.csv.

Rules (from the dataset description):
* a holiday with ``transferred == True`` is really a normal day; the celebration is the
  matching row of type ``Transfer`` -> transferred rows are dropped, Transfer rows are kept;
* locale matters: National applies to every store, Regional to stores whose *state* matches
  ``locale_name``, Local to stores whose *city* matches ``locale_name``;
* type ``Work Day`` is an extra working day (flag only), type ``Event`` is not a day off.
"""
from __future__ import annotations

import pandas as pd

HOLIDAY_TYPES = ["Holiday", "Transfer", "Additional", "Bridge"]
FLAG_COLUMNS = [
    "is_national_holiday", "is_regional_holiday", "is_local_holiday", "is_holiday", "is_event", "is_work_day",
]


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("true")


def build_calendar(holidays: pd.DataFrame, stores: pd.DataFrame, start, end) -> pd.DataFrame:
    h = holidays.copy()
    h["date"] = pd.to_datetime(h["date"])
    h["transferred"] = _as_bool(h["transferred"])
    stores = stores[["store_nbr", "city", "state"]].drop_duplicates("store_nbr")

    dates = pd.date_range(start, end, freq="D")
    cal = pd.MultiIndex.from_product([dates, stores["store_nbr"].to_numpy()], names=["date", "store_nbr"]).to_frame(index=False)
    cal = cal.merge(stores, on="store_nbr", how="left")

    real = h[(~h["transferred"]) & h["type"].isin(HOLIDAY_TYPES)]
    events = h[h["type"] == "Event"]
    workdays = h[h["type"] == "Work Day"]

    def scope_flag(rows: pd.DataFrame, name: str) -> pd.Series:
        """1 where a row of ``rows`` applies to the (date, store) pair, taking locale into account."""
        flag = pd.Series(0, index=cal.index, dtype="int32")
        nat = rows[rows["locale"] == "National"]
        flag |= cal["date"].isin(nat["date"]).astype("int32")
        for locale, col in (("Regional", "state"), ("Local", "city")):
            sub = rows[rows["locale"] == locale][["date", "locale_name"]].drop_duplicates()
            sub = sub.rename(columns={"locale_name": col}).assign(_hit=1)
            merged = cal[["date", col]].merge(sub, on=["date", col], how="left")["_hit"].fillna(0).astype("int32")
            flag |= merged.to_numpy()
        return flag.rename(name)

    nat_only = real[real["locale"] == "National"]
    reg_only = real[real["locale"] == "Regional"]
    loc_only = real[real["locale"] == "Local"]
    cal["is_national_holiday"] = scope_flag(nat_only, "n").to_numpy()
    cal["is_regional_holiday"] = scope_flag(reg_only, "r").to_numpy()
    cal["is_local_holiday"] = scope_flag(loc_only, "l").to_numpy()
    cal["is_holiday"] = cal[["is_national_holiday", "is_regional_holiday", "is_local_holiday"]].max(axis=1).astype("int32")
    cal["is_event"] = scope_flag(events, "e").to_numpy()
    cal["is_work_day"] = scope_flag(workdays, "w").to_numpy()
    return cal[["date", "store_nbr"] + FLAG_COLUMNS]
