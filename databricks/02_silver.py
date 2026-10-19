# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Silver: clean + join
# MAGIC sales + stores + transactions + daily oil price + per-store holiday calendar -> `silver.sales`.

# COMMAND ----------

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "src")))

import pandas as pd
from pyspark.sql import functions as F
from demand import spark_etl as se
from demand.store_calendar import build_calendar

dbutils.widgets.text("catalog", "retail_forecasting")
CATALOG = dbutils.widgets.get("catalog")
bronze = lambda t: spark.table(f"{CATALOG}.bronze.{t}")

def save(df, table):
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.silver.{table}")
    print(f"silver.{table}: {spark.table(f'{CATALOG}.silver.{table}').count():,} rows")

# COMMAND ----------
# MAGIC %md ## Oil: business days -> every calendar day

# COMMAND ----------

oil_daily = se.build_oil_daily(spark, bronze("oil"))
save(oil_daily, "oil_daily")

# COMMAND ----------
# MAGIC %md ## Holiday calendar per store (national / regional / local, transferred holidays handled)

# COMMAND ----------

sales_dates = bronze("sales").agg(F.min(F.to_date("date")), F.max(F.to_date("date"))).first()
test_max = bronze("test").agg(F.max(F.to_date("date"))).first()[0]
cal_end = max(pd.Timestamp(sales_dates[1]), pd.Timestamp(test_max))       # covers the forecast horizon
calendar_pdf = build_calendar(bronze("holidays").toPandas(), bronze("stores").toPandas(),
                              pd.Timestamp(sales_dates[0]), cal_end)
calendar = se.calendar_to_spark(spark, calendar_pdf)
save(calendar, "calendar")

# COMMAND ----------
# MAGIC %md ## Join everything

# COMMAND ----------

silver = se.build_silver(bronze("sales"), bronze("stores"), bronze("transactions"), oil_daily, calendar)
save(silver, "sales")
