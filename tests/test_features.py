import numpy as np
import pandas as pd

from demand.config import ENGINEERED_FEATURES, SERIES_KEYS
from demand.features import add_calendar_features, add_series_features, build_features


def _series(n=60, gap_date=None):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    if gap_date is not None:
        dates = dates[dates != pd.Timestamp(gap_date)]
    return pd.DataFrame({"date": dates, "store_nbr": 1, "family": "A", "sales": np.arange(len(dates), dtype=float) + 10,
                         "transactions": 1000.0, "oil_price": 50.0})


def test_calendar_features():
    df = add_calendar_features(pd.DataFrame({"date": pd.to_datetime(["2017-08-15", "2017-08-31", "2017-08-19"])}))
    assert df["is_payday"].tolist() == [1, 1, 0]
    assert df["is_weekend"].tolist() == [0, 0, 1]
    assert df["day_of_week"].tolist() == [1, 3, 5]          # Tue, Thu, Sat
    assert df["is_month_end"].tolist() == [0, 1, 0]


def test_lags_use_only_the_past():
    out = add_series_features(_series())
    row = out[out["date"] == pd.Timestamp("2024-02-10")].iloc[0]       # day index 40 -> sales 50
    assert row["lag_1"] == 49 and row["lag_7"] == 43 and row["lag_28"] == 22


def test_rolling_mean_excludes_today():
    s = _series()
    out = add_series_features(s)
    i = 40
    expected = s["sales"].iloc[i - 7:i].mean()                       # days i-7 .. i-1
    assert np.isclose(out.loc[i, "rolling_mean_7"], expected)


def test_no_leakage_changing_todays_sales_does_not_change_todays_features():
    s = _series()
    a = add_series_features(s)
    s2 = s.copy()
    s2.loc[40, "sales"] = 99999.0
    b = add_series_features(s2)
    cols = [c for c in ENGINEERED_FEATURES]
    pd.testing.assert_series_equal(a.loc[40, cols], b.loc[40, cols], check_names=False)
    assert not np.isclose(a.loc[41, "lag_1"], b.loc[41, "lag_1"])      # but tomorrow's lag_1 does change


def test_missing_calendar_day_is_a_gap_not_a_shift():
    out = add_series_features(_series(gap_date="2024-02-03"))
    row = out[out["date"] == pd.Timestamp("2024-02-10")].iloc[0]      # lag_7 -> 2024-02-03 which has no row
    assert np.isnan(row["lag_7"])
    assert not np.isnan(row["lag_1"])


def test_past_covariates_are_lagged_by_the_horizon():
    s = _series()
    s["transactions"] = np.arange(len(s), dtype=float)
    out = add_series_features(s)
    assert out.loc[30, "transactions_lag_7"] == 23


def test_build_features_handles_many_series(silver):
    out = build_features(silver)
    assert len(out) == len(silver)
    assert out.groupby(SERIES_KEYS).ngroups == silver.groupby(SERIES_KEYS).ngroups
    assert all(c in out.columns for c in ENGINEERED_FEATURES)
