import numpy as np
import pandas as pd
import pytest

from demand.config import HORIZON, MODEL_FEATURES, SERIES_KEYS
from demand.forecast import build_future_frame, future_dates, recursive_forecast
from demand.models import fast_candidates, make_model, prepare_X
from demand.training import make_splits, prepare_training_frame, run_experiments, save_local_model, split_frames


def test_splits_are_chronological_and_do_not_overlap(features):
    df = prepare_training_frame(features)
    s = make_splits(df["date"].max())
    parts = split_frames(df, s)
    assert parts["train"]["date"].max() < parts["valid"]["date"].min()
    assert parts["valid"]["date"].max() < parts["test"]["date"].min()
    assert parts["test"]["date"].max() == df["date"].max()
    assert sum(len(p) for p in parts.values()) <= len(df)


def test_training_frame_drops_rows_without_history_or_target(features):
    df = prepare_training_frame(features)
    assert df["lag_28"].notna().all() and df["sales"].notna().all()


@pytest.fixture(scope="module")
def experiment(features, tmp_path_factory):
    d = tmp_path_factory.mktemp("mlflow")
    import os
    old = os.getcwd()
    os.chdir(d)
    try:
        df = prepare_training_frame(features)
        res = run_experiments(df, "pytest-exp", fast_candidates(), registered_model_name="pytest_model",
                              tracking_uri=f"sqlite:///{d}/mlflow.db")
        yield res, d
    finally:
        os.chdir(old)


def test_experiment_compares_baselines_and_models(experiment):
    res, _ = experiment
    r = res.results
    assert {"naive_yesterday", "ridge", "random_forest", "xgboost_small", "lightgbm_small"} <= set(r["model"])
    assert r["val_wape"].is_monotonic_increasing                       # sorted by the selection metric
    champion_row = r[r["model"] == res.champion_name].iloc[0]
    ml = r[r["model_type"] != "baseline"]
    assert champion_row["val_wape"] == ml["val_wape"].min()           # chosen on validation, not on test


def test_champion_beats_best_baseline_on_structured_data(experiment):
    res, _ = experiment
    r = res.results
    best_baseline = r[r["model_type"] == "baseline"]["val_wape"].min()
    assert r[r["model"] == res.champion_name]["val_wape"].iloc[0] < best_baseline


def test_mlflow_logged_runs_and_registered_model(experiment):
    import mlflow
    res, d = experiment
    mlflow.set_tracking_uri(f"sqlite:///{d}/mlflow.db")
    exp = mlflow.get_experiment_by_name("pytest-exp")
    runs = mlflow.search_runs([exp.experiment_id])
    assert len(runs) >= 1 + 3 + 4 + 1                                  # parent + baselines + candidates + champion
    assert "metrics.val_wape" in runs.columns and "params.model_type" in runs.columns
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias("pytest_model", "champion")
    assert int(mv.version) >= 1
    loaded = mlflow.sklearn.load_model("models:/pytest_model@champion")
    assert hasattr(loaded, "predict")


def test_eval_frame_and_importance(experiment):
    res, _ = experiment
    assert {"actual", "predicted", "abs_error", "period_type"} <= set(res.eval_frame.columns)
    assert (res.eval_frame["predicted"] >= 0).all()
    if res.importance is not None:
        assert set(res.importance["feature"]) <= set(MODEL_FEATURES)


class _RollingMeanModel:
    """Stub 'model': predicts the rolling 7-day mean (lets us verify the recursion without training)."""
    def predict(self, X):
        return X["rolling_mean_7"].fillna(0).to_numpy()


def test_recursive_forecast_shape_and_feedback(silver, raw):
    dates = future_dates(silver["date"].max(), HORIZON)
    fut = build_future_frame(silver, dates, raw["test"], raw["holidays_events"], raw["stores"])
    n_series = silver.groupby(SERIES_KEYS).ngroups
    assert len(fut) == n_series * HORIZON and fut["sales"].isna().all()
    fc = recursive_forecast(_RollingMeanModel(), silver, fut)
    assert len(fc) == n_series * HORIZON
    assert (fc["predicted_sales"] >= 0).all() and set(fc["horizon_day"]) == set(range(1, HORIZON + 1))
    # feedback check: day-2 forecast must depend on day-1's *prediction* (rolling mean moves toward it)
    one = fc[(fc["store_nbr"] == 1) & (fc["family"] == "BEVERAGES")].sort_values("horizon_day")["predicted_sales"].to_numpy()
    h = silver[(silver["store_nbr"] == 1) & (silver["family"] == "BEVERAGES")].sort_values("date")["sales"].to_numpy()
    expected_day2 = (h[-6:].sum() + one[0]) / 7
    assert np.isclose(one[1], expected_day2)


def test_recursive_forecast_rejects_long_horizon(silver, raw):
    dates = future_dates(silver["date"].max(), HORIZON + 1)
    fut = build_future_frame(silver, dates, raw["test"], raw["holidays_events"], raw["stores"])
    with pytest.raises(ValueError):
        recursive_forecast(_RollingMeanModel(), silver, fut)


def test_future_promos_come_from_test_file_and_calendar_is_known(silver, raw):
    dates = future_dates(silver["date"].max(), HORIZON)       # 2017-08-16 .. 2017-08-22
    fut = build_future_frame(silver, dates, raw["test"], raw["holidays_events"], raw["stores"])
    row = fut[(fut["store_nbr"] == 1) & (fut["family"] == "BEVERAGES") & (fut["date"] == "2017-08-17")].iloc[0]
    expected = raw["test"][(raw["test"]["store_nbr"] == 1) & (raw["test"]["family"] == "BEVERAGES")
                           & (raw["test"]["date"] == "2017-08-17")]["onpromotion"].iloc[0]
    assert row["onpromotion"] == expected
    assert fut[fut["date"] == "2017-08-18"]["is_national_holiday"].eq(1).all()   # synthetic national holiday


def test_api_predict_and_health(experiment, tmp_path, monkeypatch):
    res, _ = experiment
    from fastapi.testclient import TestClient
    model_path = tmp_path / "demand_model.joblib"
    save_local_model(res.final_model, model_path, {"champion": res.champion_name})
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    import importlib
    import api.main as api_main
    importlib.reload(api_main)
    payload = {"date": "2017-08-16", "store_nbr": 1, "family": "BEVERAGES", "lag_1": 30, "lag_7": 31, "lag_14": 29,
               "lag_28": 32, "rolling_mean_7": 30, "rolling_mean_14": 30, "rolling_mean_28": 30,
               "onpromotion": 3, "city": "Quito", "state": "Pichincha", "store_type": "A", "cluster": 1}
    with TestClient(api_main.app) as client:
        h = client.get("/health").json()
        assert h["status"] == "ok" and h["model_loaded"] is True
        r = client.post("/predict", json=payload)
        assert r.status_code == 200 and r.json()["predicted_sales"] >= 0
        bad = client.post("/predict", json={**payload, "lag_1": -5})
        assert bad.status_code == 422
        assert client.get("/forecasts").status_code == 404           # no forecast file in tmp


def test_api_without_model_returns_503(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    import importlib
    import api.main as api_main
    importlib.reload(api_main)
    with TestClient(api_main.app) as client:
        assert client.get("/health").json()["model_loaded"] is False
        r = client.post("/predict", json={"date": "2017-08-16", "store_nbr": 1, "family": "X", "lag_1": 1, "lag_7": 1,
                                          "lag_14": 1, "lag_28": 1, "rolling_mean_7": 1, "rolling_mean_14": 1,
                                          "rolling_mean_28": 1})
        assert r.status_code == 503
