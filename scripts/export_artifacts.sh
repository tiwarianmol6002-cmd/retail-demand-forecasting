#!/usr/bin/env bash
# Download the champion model and the stored forecasts so the FastAPI service can use them.
set -euo pipefail
CATALOG="${CATALOG:-retail_forecasting}"
SRC="dbfs:/Volumes/$CATALOG/gold/artifacts"
mkdir -p models
for f in demand_model.joblib demand_model.json forecasts.csv; do
  databricks fs cp "$SRC/$f" "models/$f" --overwrite
done
ls -lh models
