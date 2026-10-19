"""Run the whole project locally with pandas (no Databricks / Spark needed).

    python -m demand.pipeline --raw data/raw --out outputs --train-start 2016-01-01
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

from . import analysis
from .config import HORIZON
from .etl import build_silver_pandas, read_raw
from .features import build_features
from .forecast import build_future_frame, future_dates, recursive_forecast
from .models import DEFAULT_CANDIDATES, Candidate
from .training import prepare_training_frame, run_experiments, save_local_model


def run_local(raw_dir, out_dir="outputs", models_dir="models", train_start=None,
              candidates: list[Candidate] | None = None, experiment_name="retail-demand-forecasting",
              tracking_uri="sqlite:///mlflow.db", horizon: int = HORIZON) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    out, models = Path(out_dir), Path(models_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("1/5 reading raw files and building Silver ...")
    raw = read_raw(raw_dir)
    silver = build_silver_pandas(raw)
    print(f"    silver rows: {len(silver):,}")

    print("2/5 engineering features ...")
    features = build_features(silver)
    train_df = prepare_training_frame(features, train_start)
    print(f"    training frame: {len(train_df):,} rows, {train_df['date'].min().date()} -> {train_df['date'].max().date()}")

    print("3/5 training and comparing models (tracked in MLflow) ...")
    result = run_experiments(train_df, experiment_name, candidates or DEFAULT_CANDIDATES, tracking_uri=tracking_uri)
    result.results.to_csv(out / "model_comparison.csv", index=False)
    print(result.results[["model", "val_wape", "test_wape", "test_mae"]].round(4).to_string(index=False))
    print(f"    champion: {result.champion_name}")

    print("4/5 error analysis ...")
    ev = result.eval_frame
    ev.to_csv(out / "forecast_evaluation.csv", index=False)
    for name, by in {"error_by_family": "family", "error_by_store": "store_nbr", "error_by_period": "period_type"}.items():
        analysis.slice_metrics(ev, by).to_csv(out / f"{name}.csv", index=False)
    analysis.worst_errors(ev).to_csv(out / "worst_errors.csv", index=False)
    if result.importance is not None:
        result.importance.to_csv(out / "feature_importance.csv", index=False)

    print(f"5/5 {horizon}-day recursive forecast ...")
    dates = future_dates(silver["date"].max(), horizon)
    future = build_future_frame(silver, dates, raw["test"], raw["holidays_events"], raw["stores"])
    fc = recursive_forecast(result.final_model, silver, future)
    fc["model_name"] = result.champion_name
    fc["generated_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
    fc.to_csv(models / "forecasts.csv", index=False)
    save_local_model(result.final_model, models / "demand_model.joblib",
                     {"champion": result.champion_name, "trained_through": str(silver["date"].max().date())})
    print(f"done. outputs in {out}/, model + forecasts in {models}/")
    return {"result": result, "forecasts": fc, "silver": silver}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--models", default="models")
    ap.add_argument("--train-start", default=None, help="e.g. 2016-01-01 (smaller = faster)")
    ap.add_argument("--fast", action="store_true", help="tiny model settings for a quick demo")
    a = ap.parse_args()
    cands = None
    if a.fast:
        from .models import fast_candidates
        cands = fast_candidates()
    run_local(a.raw, a.out, a.models, a.train_start, cands)


if __name__ == "__main__":
    main()
