"""Chronological splitting, experiment runs tracked in MLflow, model selection and (re)fit of the champion."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import baselines, config
from .metrics import evaluate
from .models import DEFAULT_CANDIDATES, Candidate, feature_importance, make_model, predict, prepare_X

SELECTION_METRIC = "val_wape"


@dataclass
class Splits:
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def as_params(self) -> dict:
        return {k: str(v.date()) for k, v in asdict(self).items()}


def make_splits(max_date, valid_days: int = config.VALID_DAYS, test_days: int = config.TEST_DAYS) -> Splits:
    """train | validation | test, strictly in time order. The test block is the last ``test_days`` days."""
    max_date = pd.Timestamp(max_date).normalize()
    test_start = max_date - pd.Timedelta(days=test_days - 1)
    valid_end = test_start - pd.Timedelta(days=1)
    valid_start = valid_end - pd.Timedelta(days=valid_days - 1)
    return Splits(valid_start - pd.Timedelta(days=1), valid_start, valid_end, test_start, max_date)


def prepare_training_frame(features: pd.DataFrame, train_start=None) -> pd.DataFrame:
    """Keep rows that have a target and a full 28-day history; optionally start later than the first date."""
    df = features.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["sales"].notna() & df["lag_28"].notna()]
    if train_start is not None:
        df = df[df["date"] >= pd.Timestamp(train_start)]
    return df.sort_values(["date", "store_nbr", "family"]).reset_index(drop=True)


def split_frames(df: pd.DataFrame, s: Splits) -> dict[str, pd.DataFrame]:
    d = df["date"]
    return {
        "train": df[d <= s.train_end],
        "valid": df[(d >= s.valid_start) & (d <= s.valid_end)],
        "test": df[(d >= s.test_start) & (d <= s.test_end)],
    }


def _fit(candidate: Candidate, frame: pd.DataFrame):
    if candidate.max_train_rows and len(frame) > candidate.max_train_rows:
        frame = frame.sample(candidate.max_train_rows, random_state=42)
    model = make_model(candidate.model_type, candidate.params)
    model.fit(prepare_X(frame), frame["sales"].to_numpy(dtype="float64"))
    return model, len(frame)


@dataclass
class ExperimentResult:
    results: pd.DataFrame
    champion_name: str
    champion_candidate: Candidate
    final_model: object
    eval_frame: pd.DataFrame
    importance: pd.DataFrame | None
    splits: Splits
    parent_run_id: str


def run_experiments(df: pd.DataFrame, experiment_name: str, candidates: list[Candidate] | None = None,
                    splits: Splits | None = None, registered_model_name: str | None = None,
                    refit_final: bool = True, tracking_uri: str | None = None) -> ExperimentResult:
    """Train every candidate on the training block, compare on the validation block, report the test block.

    * Selection uses validation WAPE only. Test metrics are logged for transparency but never used to choose.
    * Baselines (naive / same weekday last week / 7-day mean) are logged next to the ML models.
    * The champion is refit on ALL available history before it is logged / registered for serving.
    """
    import mlflow
    from mlflow.models import infer_signature

    candidates = candidates or DEFAULT_CANDIDATES
    splits = splits or make_splits(df["date"].max())
    parts = split_frames(df, splits)
    train, valid, test = parts["train"], parts["valid"], parts["test"]
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError("Empty train/validation/test block - check the date range and split sizes.")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    base_params = {"feature_version": config.FEATURE_VERSION, "target_transform": "log1p",
                   "n_features": len(config.MODEL_FEATURES), **splits.as_params(),
                   "train_rows": len(train), "valid_rows": len(valid), "test_rows": len(test)}
    rows = []
    best = {"score": np.inf, "cand": None, "model": None, "val_pred": None, "test_pred": None}

    with mlflow.start_run(run_name="model_comparison") as parent:
        for name in baselines.BASELINES:
            with mlflow.start_run(run_name=name, nested=True):
                v_pred, t_pred = (baselines.baseline_predictions(f, name) for f in (valid, test))
                m = {**evaluate(valid["sales"], v_pred, "val_"), **evaluate(test["sales"], t_pred, "test_")}
                mlflow.log_params({"model_type": "baseline", "baseline": name, **base_params})
                mlflow.log_metrics(m)
                rows.append({"model": name, "model_type": "baseline", **m})

        for cand in candidates:
            with mlflow.start_run(run_name=cand.name, nested=True):
                t0 = time.time()
                model, n_fit = _fit(cand, train)
                fit_seconds = time.time() - t0
                v_pred, t_pred = predict(model, valid), predict(model, test)
                m = {**evaluate(valid["sales"], v_pred, "val_"), **evaluate(test["sales"], t_pred, "test_")}
                mlflow.log_params({"model_type": cand.model_type, **cand.params, **base_params, "fit_rows": n_fit})
                mlflow.log_metrics({**m, "fit_seconds": fit_seconds})
                imp = feature_importance(model)
                if imp is not None:
                    mlflow.log_text(imp.to_csv(index=False), "feature_importance.csv")
                rows.append({"model": cand.name, "model_type": cand.model_type, **m, "fit_seconds": fit_seconds})
                if m[SELECTION_METRIC] < best["score"]:
                    best = {"score": m[SELECTION_METRIC], "cand": cand, "model": model, "val_pred": v_pred, "test_pred": t_pred}

        results = pd.DataFrame(rows).sort_values(SELECTION_METRIC).reset_index(drop=True)
        champion: Candidate = best["cand"]
        baseline_best = results[results["model_type"] == "baseline"][SELECTION_METRIC].min()
        improvement = 1 - best["score"] / baseline_best if baseline_best else float("nan")
        mlflow.set_tags({"champion": champion.name, "selection_metric": SELECTION_METRIC})
        mlflow.log_metrics({"champion_val_wape": best["score"], "best_baseline_val_wape": float(baseline_best),
                            "improvement_vs_best_baseline": float(improvement)})

        eval_frame = _eval_frame(test, best["test_pred"])
        importance = feature_importance(best["model"])

        final_model = best["model"]
        with mlflow.start_run(run_name=f"champion_{champion.name}", nested=True):
            if refit_final:
                final_model, n_fit = _fit(champion, pd.concat([train, valid, test]))
            else:
                n_fit = len(train)
            X_sig = prepare_X(test.head(200))
            signature = infer_signature(X_sig, predict(final_model, test.head(200)))
            mlflow.log_params({"model_type": champion.model_type, **champion.params, **base_params,
                               "refit_on_all_history": refit_final, "fit_rows": n_fit})
            mlflow.set_tags({"role": "champion", "selected_by": SELECTION_METRIC,
                             "test_wape_before_refit": f"{results.iloc[0]['test_wape'] if results.iloc[0]['model'] == champion.name else float('nan'):.5f}"})
            info = _log_sklearn_model(final_model, signature, X_sig.head(3))
            if registered_model_name:
                _register(info.model_uri, registered_model_name)
        parent_id = parent.info.run_id

    return ExperimentResult(results, champion.name, champion, final_model, eval_frame, importance, splits, parent_id)


def _log_sklearn_model(model, signature, input_example):
    """Log the pipeline. cloudpickle keeps tree models loadable (MLflow 3 defaults to skops, which rejects
    scikit-learn trees); ``name`` is the MLflow 3 argument, ``artifact_path`` the MLflow 2 one."""
    import mlflow
    kwargs = dict(signature=signature, input_example=input_example, serialization_format="cloudpickle")
    try:
        return mlflow.sklearn.log_model(model, name="model", **kwargs)
    except TypeError:
        return mlflow.sklearn.log_model(model, artifact_path="model", **kwargs)


def _register(model_uri: str, name: str) -> None:
    """Register the logged model and point the 'champion' alias at the newest version."""
    import mlflow
    from mlflow import MlflowClient
    try:
        mv = mlflow.register_model(model_uri, name)
        MlflowClient().set_registered_model_alias(name, "champion", str(mv.version))
        print(f"Registered {name} version {mv.version} with alias 'champion'.")
    except Exception as exc:  # registry unavailable (e.g. restricted workspace) - the logged model is still usable
        print(f"WARNING: model registration failed ({type(exc).__name__}: {exc}). "
              f"The model is still logged in the run at {model_uri}.")


def _eval_frame(test: pd.DataFrame, pred) -> pd.DataFrame:
    from .analysis import build_eval_frame
    return build_eval_frame(test, pred)


def save_local_model(model, path, metadata: dict | None = None) -> None:
    """Pickle the pipeline for the API and write a JSON file with the library versions used to train it."""
    import sklearn
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    meta = {"feature_version": config.FEATURE_VERSION, "features": config.MODEL_FEATURES,
            "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}
    try:
        import xgboost, lightgbm  # noqa: E401
        meta.update(xgboost=xgboost.__version__, lightgbm=lightgbm.__version__)
    except ImportError:
        pass
    meta.update(metadata or {})
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
