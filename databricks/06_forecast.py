# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Recursive 7-day forecast -> `gold.demand_forecasts`
# MAGIC Loads the registered `champion` model, forecasts day 1, feeds it back as history, forecasts day 2, ...
# MAGIC Future promotions come from Kaggle's `test.csv`; holidays from the holiday calendar; transactions/oil are only used lagged.

# COMMAND ----------

# MAGIC %pip install -q xgboost lightgbm

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "src")))

import joblib
import mlflow
import pandas as pd
from pyspark.sql import functions as F
from demand.config import HORIZON, SERIES_KEYS, SILVER_COLUMNS
from demand.forecast import build_future_frame, future_dates, recursive_forecast

dbutils.widgets.text("catalog", "retail_forecasting")
dbutils.widgets.text("horizon", str(HORIZON))
CATALOG = dbutils.widgets.get("catalog")
H = int(dbutils.widgets.get("horizon"))
MODEL_NAME = f"{CATALOG}.gold.demand_forecaster"
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------
# MAGIC %md ## Load the champion model

# COMMAND ----------

try:
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    model_version = mlflow.MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion").version
except Exception as exc:
    print(f"Registry load failed ({type(exc).__name__}); falling back to the exported pickle.")
    model = joblib.load(f"/Volumes/{CATALOG}/gold/artifacts/demand_model.joblib")
    model_version = "pickle"
print("model version:", model_version)

# COMMAND ----------
# MAGIC %md ## Build history + future inputs

# COMMAND ----------

last_date = spark.table(f"{CATALOG}.silver.sales").agg(F.max("date")).first()[0]
history = (spark.table(f"{CATALOG}.silver.sales").select(*SILVER_COLUMNS)
           .filter(F.col("date") > F.date_sub(F.lit(last_date), 80)).toPandas())
history["date"] = pd.to_datetime(history["date"])
test_promo = spark.table(f"{CATALOG}.bronze.test").select("date", "store_nbr", "family", "onpromotion").toPandas()
holidays = spark.table(f"{CATALOG}.bronze.holidays").toPandas()
stores = spark.table(f"{CATALOG}.bronze.stores").toPandas()

dates = future_dates(history["date"].max(), H)
future = build_future_frame(history, dates, test_promo, holidays, stores)

# COMMAND ----------
# MAGIC %md ## Forecast

# COMMAND ----------

fc = recursive_forecast(model, history, future)
n_series = history.groupby(SERIES_KEYS).ngroups
assert len(fc) == n_series * H, "unexpected number of forecast rows"
assert fc["predicted_sales"].notna().all() and (fc["predicted_sales"] >= 0).all(), "invalid forecasts"

fc["model_name"] = MODEL_NAME
fc["model_version"] = str(model_version)
fc["generated_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
out = fc.rename(columns={"forecast_date": "date"}).assign(date=lambda d: d["date"].dt.date)
spark.createDataFrame(out).write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{CATALOG}.gold.demand_forecasts")
fc.to_csv(f"/Volumes/{CATALOG}/gold/artifacts/forecasts.csv", index=False)
print(f"gold.demand_forecasts: {len(fc):,} rows ({n_series:,} series x {H} days), {dates.min().date()} -> {dates.max().date()}")
