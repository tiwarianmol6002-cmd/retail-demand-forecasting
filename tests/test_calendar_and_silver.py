import pandas as pd

from demand.config import SILVER_COLUMNS
from demand.etl import build_oil_daily
from demand.store_calendar import build_calendar


def _holidays():
    return pd.DataFrame([
        ("2016-10-09", "Holiday", "National", "Ecuador", "x", "True"),     # transferred away -> normal day
        ("2016-10-12", "Transfer", "National", "Ecuador", "x", "False"),   # the real celebration
        ("2016-12-06", "Holiday", "Local", "Quito", "x", "False"),
        ("2016-07-25", "Holiday", "Regional", "Guayas", "x", "False"),
        ("2016-05-08", "Event", "National", "Ecuador", "x", "False"),
        ("2016-11-12", "Work Day", "National", "Ecuador", "x", "False"),
    ], columns=["date", "type", "locale", "locale_name", "description", "transferred"])


def _stores():
    return pd.DataFrame({"store_nbr": [1, 2], "city": ["Quito", "Guayaquil"], "state": ["Pichincha", "Guayas"],
                         "type": ["A", "B"], "cluster": [1, 2]})


def _flag(cal, date, store, col):
    return int(cal[(cal["date"] == pd.Timestamp(date)) & (cal["store_nbr"] == store)][col].iloc[0])


def test_transferred_holiday_is_a_normal_day_and_transfer_row_is_the_holiday():
    cal = build_calendar(_holidays(), _stores(), "2016-01-01", "2016-12-31")
    assert _flag(cal, "2016-10-09", 1, "is_holiday") == 0
    assert _flag(cal, "2016-10-12", 1, "is_holiday") == 1 and _flag(cal, "2016-10-12", 2, "is_holiday") == 1


def test_local_and_regional_holidays_only_apply_to_matching_stores():
    cal = build_calendar(_holidays(), _stores(), "2016-01-01", "2016-12-31")
    assert _flag(cal, "2016-12-06", 1, "is_local_holiday") == 1       # Quito store
    assert _flag(cal, "2016-12-06", 2, "is_local_holiday") == 0       # Guayaquil store
    assert _flag(cal, "2016-07-25", 2, "is_regional_holiday") == 1    # Guayas
    assert _flag(cal, "2016-07-25", 1, "is_regional_holiday") == 0


def test_events_and_work_days_are_not_holidays():
    cal = build_calendar(_holidays(), _stores(), "2016-01-01", "2016-12-31")
    assert _flag(cal, "2016-05-08", 1, "is_event") == 1 and _flag(cal, "2016-05-08", 1, "is_holiday") == 0
    assert _flag(cal, "2016-11-12", 1, "is_work_day") == 1 and _flag(cal, "2016-11-12", 1, "is_holiday") == 0


def test_oil_expanded_to_calendar_days_and_filled():
    oil = pd.DataFrame({"date": ["2016-01-01", "2016-01-04", "2016-01-05"], "dcoilwtico": [None, 30.0, 31.0]})
    daily = build_oil_daily(oil)
    assert len(daily) == 5 and daily["oil_price"].notna().all()
    assert daily.loc[0, "oil_price"] == 30.0                         # leading gap back-filled
    assert daily.loc[1, "oil_price"] == 30.0                         # weekend carried forward


def test_silver_schema_and_keys(silver):
    assert list(silver.columns) == SILVER_COLUMNS
    assert not silver.duplicated(["date", "store_nbr", "family"]).any()
    assert silver["oil_price"].notna().all()
    assert (silver["sales"] >= 0).all()
