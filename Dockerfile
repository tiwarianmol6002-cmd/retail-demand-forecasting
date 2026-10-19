FROM python:3.12-slim

# LightGBM / XGBoost need the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src ./src
# models/ (demand_model.joblib, demand_model.json, forecasts.csv) is mounted or copied in at run time
ENV MODEL_PATH=/app/models/demand_model.joblib \
    FORECAST_PATH=/app/models/forecasts.csv

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
