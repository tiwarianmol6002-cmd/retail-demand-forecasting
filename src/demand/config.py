"""Single source of truth for column names, feature lists and modelling constants."""

SERIES_KEYS = ["store_nbr", "family"]
TARGET = "sales"
HORIZON = 7  # forecast horizon in days; "past covariates" are lagged by this much

# ---- sales-derived features (computed on calendar days, never using sales of day d itself) ----
LAGS = [1, 7, 14, 28]
ROLL_MEAN_WINDOWS = [7, 14, 28]
ROLL_STD_WINDOWS = [7, 28]

# ---- columns coming from the Silver table ----
STATIC_CAT = ["store_nbr", "family", "city", "state", "store_type", "cluster"]
KNOWN_FUTURE = [  # known in advance for the day being predicted
    "onpromotion", "is_holiday", "is_national_holiday", "is_regional_holiday",
    "is_local_holiday", "is_event", "is_work_day",
]
SILVER_COLUMNS = (
    ["date", "store_nbr", "family", "sales", "onpromotion", "city", "state", "store_type", "cluster",
     "transactions", "oil_price", "is_national_holiday", "is_regional_holiday", "is_local_holiday",
     "is_holiday", "is_event", "is_work_day"]
)

CALENDAR_FEATURES = [
    "day_of_week", "day_of_month", "week_of_year", "month", "quarter", "year",
    "is_weekend", "is_month_start", "is_month_end", "is_payday",
]
# "Past covariates" (not known in the future) are lagged by HORIZON days so that the
# feature value is identical at training time and at inference time for every horizon day.
PAST_COVARIATE_FEATURES = ["transactions_lag_7", "transactions_roll_mean_28", "oil_price_lag_7"]

SALES_FEATURES = (
    [f"lag_{k}" for k in LAGS]
    + [f"rolling_mean_{w}" for w in ROLL_MEAN_WINDOWS]
    + [f"rolling_std_{w}" for w in ROLL_STD_WINDOWS]
)

ENGINEERED_FEATURES = SALES_FEATURES + PAST_COVARIATE_FEATURES + CALENDAR_FEATURES
NUMERIC_FEATURES = KNOWN_FUTURE + ENGINEERED_FEATURES
MODEL_FEATURES = STATIC_CAT + NUMERIC_FEATURES

INT_FEATURES = set(CALENDAR_FEATURES)
FEATURE_VERSION = "v1"

# ---- chronological split sizes (days counted back from the last date in the data) ----
TEST_DAYS = 31
VALID_DAYS = 45
