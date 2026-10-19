-- Run once in the Databricks SQL editor BEFORE uploading the CSV files
-- (the bundle's 00_setup notebook does the same, but the volume must exist before the first upload).
CREATE CATALOG IF NOT EXISTS retail_forecasting;
CREATE SCHEMA IF NOT EXISTS retail_forecasting.bronze;
CREATE SCHEMA IF NOT EXISTS retail_forecasting.silver;
CREATE SCHEMA IF NOT EXISTS retail_forecasting.gold;
CREATE VOLUME IF NOT EXISTS retail_forecasting.bronze.raw_data;
CREATE VOLUME IF NOT EXISTS retail_forecasting.gold.artifacts;
SHOW SCHEMAS IN retail_forecasting;
