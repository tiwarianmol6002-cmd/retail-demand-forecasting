# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Train, compare and register models (MLflow)
# MAGIC Baselines + Ridge + Random Forest + XGBoost + LightGBM, chronological train / validation / test split.
# MAGIC The champion is chosen on **validation WAPE**, refit on all history, logged and registered in Unity Catalog.

# COMMAND ----------
# MAGIC %pip install -q xgboost lightgbm

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "src")))

import mlflow
import pandas as pd
from pyspark.sql import functions as F
from demand import analysis
from demand.config import ENGINEERED_FEATURES, SILVER_COLUMNS
from demand.models import DEFAULT_CANDIDATES, Candidate
from demand.training import prepare_training_frame, run_experiments, save_local_model

dbutils.widgets.text("catalog", "retail_forecasting")
dbutils.widgets.text("train_start", "2016-01-01")
dbutils.widgets.text("rf_max_rows", "300000")
CATALOG = dbutils.widgets.get("catalog")
TRAIN_START = dbutils.widgets.get("train_start")
RF_MAX_ROWS = int(dbutils.widgets.get("rf_max_rows"))

user = spark.sql("SELECT current_user()").first()[0]
EXPERIMENT = f"/Users/{user}/retail-demand-forecasting"
MODEL_NAME = f"{CATALOG}.gold.demand_forecaster"
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------
# MAGIC %md ## Load the feature table (training window only)

# COMMAND ----------

cols = SILVER_COLUMNS + ENGINEERED_FEATURES
sdf = (spark.table(f"{CATALOG}.gold.demand_features").select(*cols)
       .filter((F.col("date") >= TRAIN_START) & F.col("sales").isNotNull() & F.col("lag_28").isNotNull()))
pdf = prepare_training_frame(sdf.toPandas())
print(f"training frame: {len(pdf):,} rows, {pdf['date'].min().date()} -> {pdf['date'].max().date()}")

candidates = [Candidate(c.name, c.model_type, c.params, RF_MAX_ROWS if c.model_type == "random_forest" else c.max_train_rows)
              for c in DEFAULT_CANDIDATES]

# COMMAND ----------
# MAGIC %md ## Run the experiments

# COMMAND ----------

result = run_experiments(pdf, EXPERIMENT, candidates, registered_model_name=MODEL_NAME)
display(result.results)
print("champion:", result.champion_name)

# COMMAND ----------
# MAGIC %md ## Save comparison + error analysis as Delta tables

# COMMAND ----------

def save(pdf_, table):
    out = pdf_.copy()
    for c in out.columns:
        if str(out[c].dtype).startswith("datetime"):
            out[c] = out[c].dt.date
    spark.createDataFrame(out).write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
        .saveAsTable(f"{CATALOG}.gold.{table}")

comparison = result.results.assign(is_champion=lambda d: d["model"] == result.champion_name)
save(comparison, "model_comparison")
ev = result.eval_frame
save(ev, "forecast_evaluation")
save(analysis.slice_metrics(ev, "family"), "error_by_family")
save(analysis.slice_metrics(ev, "store_nbr"), "error_by_store")
save(analysis.slice_metrics(ev, "period_type"), "error_by_period")
save(analysis.worst_errors(ev), "worst_errors")
if result.importance is not None:
    save(result.importance, "feature_importance")

# COMMAND ----------
# MAGIC %md ## Export the champion for the FastAPI service

# COMMAND ----------

ART = f"/Volumes/{CATALOG}/gold/artifacts"
save_local_model(result.final_model, f"{ART}/demand_model.joblib",
                 {"champion": result.champion_name, "trained_through": str(pdf["date"].max().date())})
print("exported to", ART)
