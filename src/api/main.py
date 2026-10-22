"""FastAPI service for the demand model.

    uvicorn src.api.main:app --reload        ->  http://127.0.0.1:8000/docs

Environment variables
    MODEL_PATH     local pickle (default models/demand_model.joblib)
    MODEL_URI      optional MLflow URI instead, e.g. models:/retail_forecasting.gold.demand_forecaster@champion
    FORECAST_PATH  CSV with stored 7-day forecasts (default models/forecasts.csv)
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demand.config import MODEL_FEATURES  # noqa: E402
from demand.features import add_calendar_features  # noqa: E402
from demand.models import predict  # noqa: E402

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/demand_model.joblib"))
FORECAST_PATH = Path(os.getenv("FORECAST_PATH", "models/forecasts.csv"))
MODEL_URI = os.getenv("MODEL_URI")


def load_model():
    """Return (model, metadata). Either from an MLflow URI or from the local pickle."""
    if MODEL_URI:
        import mlflow.sklearn
        return mlflow.sklearn.load_model(MODEL_URI), {"source": MODEL_URI}
    if MODEL_PATH.exists():
        meta_path = MODEL_PATH.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return joblib.load(MODEL_PATH), {"source": str(MODEL_PATH), **meta}
    return None, {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.meta = load_model()
    yield


app = FastAPI(title="Retail Demand Forecasting API", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "date": "2017-08-16", "store_nbr": 1, "family": "BEVERAGES", "lag_1": 1942, "lag_7": 2311, "lag_14": 2645,
        "lag_28": 2369, "rolling_mean_7": 1757.0, "rolling_mean_14": 1930.36, "rolling_mean_28": 2041.18,
        "rolling_std_7": 633.3, "rolling_std_28": 550.5, "onpromotion": 20,
        "transactions_lag_7": 1766, "transactions_roll_mean_28": 1540.43, "oil_price_lag_7": 49.59,
        "city": "Quito", "state": "Pichincha", "store_type": "D", "cluster": 13}})

    date: date
    store_nbr: int = Field(ge=1)
    family: str
    lag_1: float = Field(ge=0)
    lag_7: float = Field(ge=0)
    lag_14: float = Field(ge=0)
    lag_28: float = Field(ge=0)
    rolling_mean_7: float = Field(ge=0)
    rolling_mean_14: float = Field(ge=0)
    rolling_mean_28: float = Field(ge=0)
    rolling_std_7: float | None = Field(default=None, ge=0)
    rolling_std_28: float | None = Field(default=None, ge=0)
    onpromotion: float = Field(default=0, ge=0)
    transactions_lag_7: float = Field(ge=0)
    transactions_roll_mean_28: float = Field(ge=0)
    oil_price_lag_7: float = Field(ge=0)
    is_holiday: int = Field(default=0, ge=0, le=1)
    is_national_holiday: int = Field(default=0, ge=0, le=1)
    is_regional_holiday: int = Field(default=0, ge=0, le=1)
    is_local_holiday: int = Field(default=0, ge=0, le=1)
    is_event: int = Field(default=0, ge=0, le=1)
    is_work_day: int = Field(default=0, ge=0, le=1)
    city: str | None = None
    state: str | None = None
    store_type: str | None = None
    cluster: int | None = None


class PredictResponse(BaseModel):
    predicted_sales: float
    model: str | None = None
    trained_through: str | None = None


def request_to_frame(req: PredictRequest) -> pd.DataFrame:
    row = req.model_dump()
    row["date"] = pd.Timestamp(row["date"])
    df = add_calendar_features(pd.DataFrame([row]))
    for col in MODEL_FEATURES:  # optional inputs left out -> NaN / "None"; the pipeline imputes or ignores them
        if col not in df.columns:
            df[col] = float("nan")
    return df


@app.get("/health")
def health():
    meta = getattr(app.state, "meta", {})
    return {"status": "ok", "model_loaded": app.state.model is not None, "model": meta.get("champion"),
            "source": meta.get("source")}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train one first (python -m demand.pipeline).")
    value = float(predict(app.state.model, request_to_frame(req))[0])
    meta = app.state.meta
    return PredictResponse(predicted_sales=round(value, 3), model=meta.get("champion"),
                           trained_through=meta.get("trained_through"))


@app.get("/forecasts")
def forecasts(store_nbr: int | None = None, family: str | None = None, limit: int = Query(200, ge=1, le=5000)):
    """Stored 7-day forecasts produced by the batch pipeline."""
    if not FORECAST_PATH.exists():
        raise HTTPException(status_code=404, detail="No forecast file found. Run the pipeline first.")
    df = pd.read_csv(FORECAST_PATH)
    if store_nbr is not None:
        df = df[df["store_nbr"] == store_nbr]
    if family is not None:
        df = df[df["family"].str.upper() == family.upper()]
    return df.head(limit).to_dict(orient="records")
