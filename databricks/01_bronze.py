# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Bronze: raw CSV files -> Delta (no business logic)
# MAGIC Adds `source_file` and `bronze_ingested_at` so every row can be traced back to its file.

# COMMAND ----------

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "src")))

from pyspark.sql import functions as F
from demand import spark_etl as se

dbutils.widgets.text("catalog", "retail_forecasting")
dbutils.widgets.text("raw_base", "/Volumes/retail_forecasting/bronze/raw_data")
CATALOG = dbutils.widgets.get("catalog")
RAW_BASE = dbutils.widgets.get("raw_base")

TABLES = {"train": "sales", "test": "test", "stores": "stores", "oil": "oil",
          "holidays_events": "holidays", "transactions": "transactions"}

# COMMAND ----------

for raw_name, table in TABLES.items():
    df = (se.read_raw_csv(spark, RAW_BASE, raw_name)
          .withColumn("source_file", F.col("_metadata.file_path"))
          .withColumn("bronze_ingested_at", F.current_timestamp()))
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.bronze.{table}")
    print(f"bronze.{table}: {spark.table(f'{CATALOG}.bronze.{table}').count():,} rows")
