# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - Setup
# MAGIC Creates the catalog, the three schemas and two volumes (raw CSV files, exported model artifacts).

# COMMAND ----------

dbutils.widgets.text("catalog", "retail_forecasting")
CATALOG = dbutils.widgets.get("catalog")

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
for schema in ("bronze", "silver", "gold"):
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.bronze.raw_data")        # the Kaggle CSV files go here
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.gold.artifacts")         # exported model + forecasts
print(f"Ready: {CATALOG}.[bronze|silver|gold], volumes raw_data and artifacts")
