# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Data-quality gate
# MAGIC Every query returns the number of offending rows and must be 0, otherwise the job fails before any model is trained.

# COMMAND ----------

dbutils.widgets.text("catalog", "retail_forecasting")
C = dbutils.widgets.get("catalog")
S, F_ = f"{C}.silver.sales", f"{C}.gold.demand_features"

CHECKS = {
    "silver.sales: duplicate (date, store, family)":
        f"SELECT COUNT(*) FROM (SELECT 1 FROM {S} GROUP BY date, store_nbr, family HAVING COUNT(*) > 1)",
    "silver.sales: negative sales": f"SELECT COUNT(*) FROM {S} WHERE sales < 0",
    "silver.sales: null key or target": f"SELECT COUNT(*) FROM {S} WHERE date IS NULL OR store_nbr IS NULL OR family IS NULL OR sales IS NULL",
    "silver.sales: stores missing from stores table": f"SELECT COUNT(*) FROM {S} WHERE city IS NULL",
    "silver.sales: null oil price": f"SELECT COUNT(*) FROM {S} WHERE oil_price IS NULL",
    "silver.sales: holiday flag not 0/1": f"SELECT COUNT(*) FROM {S} WHERE is_holiday NOT IN (0, 1)",
    "silver.sales: fewer than 2 stores or 2 families":
        f"SELECT CASE WHEN COUNT(DISTINCT store_nbr) < 2 OR COUNT(DISTINCT family) < 2 THEN 1 ELSE 0 END FROM {S}",
    "features: row count differs from silver":
        f"SELECT ABS((SELECT COUNT(*) FROM {S}) - (SELECT COUNT(*) FROM {F_}))",
    "features: negative rolling std": f"SELECT COUNT(*) FROM {F_} WHERE rolling_std_7 < 0 OR rolling_std_28 < 0",
    # leakage guard: lag_1 must equal the previous calendar day's sales of the same series
    "features: lag_1 != sales of previous day (last 120 days)":
        f"""SELECT COUNT(*) FROM {F_} a JOIN {F_} b
            ON a.store_nbr = b.store_nbr AND a.family = b.family AND a.date = date_add(b.date, 1)
            WHERE a.date >= date_sub((SELECT MAX(date) FROM {F_}), 120) AND a.lag_1 != b.sales""",
    # leakage guard: the feature must NOT equal today's sales more often than chance allows
    "features: lag_1 identical to same-day sales for >50% of non-zero rows":
        f"""SELECT CASE WHEN AVG(CASE WHEN lag_1 = sales THEN 1.0 ELSE 0.0 END) > 0.5 THEN 1 ELSE 0 END
            FROM {F_} WHERE sales > 0 AND lag_1 IS NOT NULL""",
}

# COMMAND ----------

failed = []
for name, query in CHECKS.items():
    bad = spark.sql(query).first()[0]
    print(f"[{'PASS' if bad == 0 else 'FAIL'}] {name} -> {bad}")
    if bad != 0:
        failed.append(name)
if failed:
    raise Exception(f"{len(failed)} data-quality check(s) failed: {failed}")
print("All data-quality checks passed.")
