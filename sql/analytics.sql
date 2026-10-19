-- ===== Exploration (Silver) =====

-- 1. Average daily sales by product family
SELECT family, ROUND(AVG(sales), 1) AS avg_daily_sales, ROUND(SUM(sales)) AS total_sales
FROM retail_forecasting.silver.sales
GROUP BY family ORDER BY avg_daily_sales DESC;

-- 2. Sales by store
SELECT store_nbr, city, store_type, ROUND(SUM(sales)) AS total_sales
FROM retail_forecasting.silver.sales
GROUP BY store_nbr, city, store_type ORDER BY total_sales DESC;

-- 3. Promotion effect (average sales with / without promotion, per family)
SELECT family,
       ROUND(AVG(CASE WHEN onpromotion > 0 THEN sales END), 1) AS avg_sales_on_promo,
       ROUND(AVG(CASE WHEN onpromotion = 0 THEN sales END), 1) AS avg_sales_no_promo
FROM retail_forecasting.silver.sales
GROUP BY family ORDER BY avg_sales_on_promo - avg_sales_no_promo DESC;

-- 4. Weekday seasonality (0 = Monday)
SELECT pmod(dayofweek(date) - 2, 7) AS day_of_week, ROUND(AVG(sales), 1) AS avg_sales
FROM retail_forecasting.silver.sales GROUP BY 1 ORDER BY 1;

-- 5. Monthly seasonality
SELECT month(date) AS month, ROUND(AVG(sales), 1) AS avg_sales
FROM retail_forecasting.silver.sales GROUP BY 1 ORDER BY 1;

-- 6. Holiday vs normal days
SELECT is_holiday, ROUND(AVG(sales), 1) AS avg_sales, COUNT(*) AS rows
FROM retail_forecasting.silver.sales GROUP BY is_holiday;

-- 7. Payday effect (15th and last day of the month)
SELECT (day(date) = 15 OR date = last_day(date)) AS is_payday, ROUND(AVG(sales), 1) AS avg_sales
FROM retail_forecasting.silver.sales GROUP BY 1;

-- ===== Model results (Gold) =====

-- 8. Model comparison (validation drives selection; test is the final hold-out)
SELECT model, model_type, ROUND(val_wape, 4) AS val_wape, ROUND(test_wape, 4) AS test_wape,
       ROUND(test_mae, 2) AS test_mae, ROUND(test_rmse, 2) AS test_rmse, is_champion
FROM retail_forecasting.gold.model_comparison ORDER BY val_wape;

-- 9. Hardest product families to forecast
SELECT family, ROUND(wape, 4) AS wape, ROUND(mae, 2) AS mae, total_sales
FROM retail_forecasting.gold.error_by_family WHERE total_sales > 0 ORDER BY wape DESC;

-- 10. Error on holidays / promotions / normal days
SELECT period_type, n_rows, ROUND(wape, 4) AS wape, ROUND(mae, 2) AS mae
FROM retail_forecasting.gold.error_by_period ORDER BY wape DESC;

-- 11. Mean absolute error per family (from the raw evaluation table)
SELECT family, AVG(ABS(actual - predicted)) AS mae
FROM retail_forecasting.gold.forecast_evaluation GROUP BY family ORDER BY mae;

-- 12. Biggest forecast misses
SELECT date, store_nbr, family, actual, ROUND(predicted, 1) AS predicted, ROUND(abs_error, 1) AS abs_error, period_type
FROM retail_forecasting.gold.worst_errors;

-- 13. Most important features of the champion
SELECT feature, ROUND(importance, 4) AS importance
FROM retail_forecasting.gold.feature_importance ORDER BY importance DESC LIMIT 15;

-- ===== Forecasts =====

-- 14. Next 7 days for one store / family
SELECT date, horizon_day, ROUND(predicted_sales, 1) AS predicted_sales, model_version
FROM retail_forecasting.gold.demand_forecasts
WHERE store_nbr = 1 AND family = 'BEVERAGES' ORDER BY date;

-- 15. Total expected demand per family over the horizon
SELECT family, ROUND(SUM(predicted_sales)) AS expected_units_7d
FROM retail_forecasting.gold.demand_forecasts GROUP BY family ORDER BY expected_units_7d DESC;
