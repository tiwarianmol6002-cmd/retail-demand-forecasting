import numpy as np
import pandas as pd

from demand import metrics
from demand.baselines import BASELINES, baseline_predictions


def test_metrics_known_values():
    y, p = [10, 20, 0], [12, 18, 0]
    assert metrics.mae(y, p) == (2 + 2 + 0) / 3
    assert np.isclose(metrics.rmse(y, p), np.sqrt((4 + 4 + 0) / 3))
    assert np.isclose(metrics.wape(y, p), 4 / 30)
    assert np.isclose(metrics.smape([0, 10], [0, 10]), 0.0)          # both zero counts as perfect


def test_wape_is_nan_when_everything_is_zero():
    assert np.isnan(metrics.wape([0, 0], [1, 1]))


def test_baselines_fall_back_when_source_missing():
    df = pd.DataFrame({"lag_1": [5.0, np.nan], "lag_7": [np.nan, np.nan], "rolling_mean_7": [4.0, np.nan],
                       "rolling_mean_28": [3.0, np.nan]})
    assert baseline_predictions(df, "naive_yesterday").tolist() == [5.0, 0.0]
    assert baseline_predictions(df, "naive_same_weekday_last_week").tolist() == [3.0, 0.0]
    assert set(BASELINES) == {"naive_yesterday", "naive_same_weekday_last_week", "rolling_mean_7"}
