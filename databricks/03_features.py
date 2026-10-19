# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Feature engineering -> `gold.demand_features`
# MAGIC Lags, rolling statistics, calendar effects and horizon-lagged covariates, computed with calendar-day window functions.
# MAGIC The same logic exists in pandas (`src/demand/features.py`) and a unit test proves both agree.

# COMMAND ----------

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "src")))

from demand import spark_etl as se

dbutils.widgets.text("catalog", "retail_forecasting")
CATALOG = dbutils.widgets.get("catalog")

# COMMAND ----------

features = se.build_feature_table(spark.table(f"{CATALOG}.silver.sales"))
features.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.gold.demand_features")
n = spark.table(f"{CATALOG}.gold.demand_features").count()
print(f"gold.demand_features: {n:,} rows, {len(features.columns)} columns")
