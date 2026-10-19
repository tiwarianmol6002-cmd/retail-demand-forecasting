"""Model factory: every model is a scikit-learn pipeline that goes from the raw feature frame to a
sales forecast (preprocessing + estimator + log1p target transform) so it can be pickled / logged as one object."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .config import MODEL_FEATURES, NUMERIC_FEATURES, STATIC_CAT


@dataclass
class Candidate:
    name: str
    model_type: str                 # ridge | random_forest | xgboost | lightgbm
    params: dict = field(default_factory=dict)
    max_train_rows: int | None = None   # cap for slow models (random sample, fixed seed)


def prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    """Select and type the model inputs. Categorical columns become strings so that unseen values are handled."""
    X = df[MODEL_FEATURES].copy()
    for c in STATIC_CAT:
        X[c] = X[c].astype(str)
    for c in NUMERIC_FEATURES:
        X[c] = X[c].astype("float64")
    return X


def _preprocessor(model_type: str) -> ColumnTransformer:
    if model_type == "ridge":
        num = Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)), ("sc", StandardScaler())])
        cat = OneHotEncoder(handle_unknown="ignore")
    else:
        num = (SimpleImputer(strategy="median", keep_empty_features=True) if model_type == "random_forest"
               else "passthrough")  # XGBoost / LightGBM handle NaN natively
        cat = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    return ColumnTransformer(
        [("cat", cat, STATIC_CAT), ("num", num, NUMERIC_FEATURES)],
        verbose_feature_names_out=False,
    )


def _estimator(model_type: str, params: dict):
    if model_type == "ridge":
        return Ridge(**params)
    if model_type == "random_forest":
        return RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    if model_type == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(objective="reg:squarederror", tree_method="hist", n_jobs=-1, random_state=42, **params)
    if model_type == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1, subsample_freq=1, **params)
    raise ValueError(f"Unknown model_type: {model_type}")


def make_model(model_type: str, params: dict | None = None) -> TransformedTargetRegressor:
    pipe = Pipeline([("pre", _preprocessor(model_type)), ("model", _estimator(model_type, params or {}))])
    # demand is right-skewed and spans several orders of magnitude -> model log1p(sales)
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1, check_inverse=False)


def predict(model, df: pd.DataFrame) -> np.ndarray:
    return np.clip(np.asarray(model.predict(prepare_X(df)), dtype="float64"), 0.0, None)


def feature_importance(model) -> pd.DataFrame | None:
    """Importance table for tree models (gain-style importances) and |coef| for ridge."""
    pipe = model.regressor_
    est = pipe.named_steps["model"]
    names = pipe.named_steps["pre"].get_feature_names_out()
    if hasattr(est, "feature_importances_"):
        values = np.asarray(est.feature_importances_, dtype="float64")
    elif hasattr(est, "coef_"):
        values = np.abs(np.ravel(est.coef_))
    else:
        return None
    if len(values) != len(names):
        return None
    out = pd.DataFrame({"feature": names, "importance": values})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


DEFAULT_CANDIDATES = [
    Candidate("ridge", "ridge", {"alpha": 1.0}),
    Candidate("random_forest", "random_forest",
              {"n_estimators": 150, "max_depth": 16, "min_samples_leaf": 5, "max_features": 0.5},
              max_train_rows=300_000),
    Candidate("xgboost_depth6", "xgboost",
              {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8}),
    Candidate("xgboost_depth8", "xgboost",
              {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 8, "subsample": 0.8, "colsample_bytree": 0.8}),
    Candidate("lightgbm_leaves31", "lightgbm",
              {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.8}),
    Candidate("lightgbm_leaves63", "lightgbm",
              {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 63, "subsample": 0.8, "colsample_bytree": 0.8}),
]


def fast_candidates() -> list[Candidate]:
    """Tiny configurations for unit tests / CI."""
    return [
        Candidate("ridge", "ridge", {"alpha": 1.0}),
        Candidate("random_forest", "random_forest", {"n_estimators": 20, "max_depth": 8, "min_samples_leaf": 5}, max_train_rows=5000),
        Candidate("xgboost_small", "xgboost", {"n_estimators": 40, "learning_rate": 0.1, "max_depth": 4}),
        Candidate("lightgbm_small", "lightgbm", {"n_estimators": 40, "learning_rate": 0.1, "num_leaves": 15}),
    ]
