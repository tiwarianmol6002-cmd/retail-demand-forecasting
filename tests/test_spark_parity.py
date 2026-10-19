"""The Spark (Databricks) pipeline and the pandas pipeline must produce the same numbers."""
import shutil

import numpy as np
import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")
if shutil.which("java") is None:
    pytest.skip("Java is required for local Spark", allow_module_level=True)

from demand import spark_etl as se  # noqa: E402
from demand.config import SILVER_COLUMNS  # noqa: E402
from demand.etl import build_oil_daily as pandas_oil  # noqa: E402
from demand.store_calendar import build_calendar  # noqa: E402
from demand.synthetic import write_raw  # noqa: E402

KEY = ["store_nbr", "family", "date"]
TEXT = {"family", "city", "state", "store_type"}


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession
    s = (SparkSession.builder.master("local[1]").appName("parity")
         .config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "2")
         .config("spark.sql.session.timeZone", "UTC").getOrCreate())
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()


@pytest.fixture(scope="module")
def spark_tables(spark, raw, tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    write_raw(raw, d)
    tables = {n: se.read_raw_csv(spark, str(d), n) for n in se.RAW_SCHEMAS}
    dates = pd.to_datetime(raw["train"]["date"])
    cal_pdf = build_calendar(raw["holidays_events"], raw["stores"], dates.min(), dates.max())
    oil = se.build_oil_daily(spark, tables["oil"])
    silver = se.build_silver(tables["train"], tables["stores"], tables["transactions"], oil,
                             se.calendar_to_spark(spark, cal_pdf))
    return {"oil": oil, "silver": silver}


def _assert_frames_equal(spark_pdf, pandas_pdf, cols):
    a = spark_pdf.assign(date=pd.to_datetime(spark_pdf["date"])).sort_values(KEY).reset_index(drop=True)
    b = pandas_pdf.assign(date=pd.to_datetime(pandas_pdf["date"])).sort_values(KEY).reset_index(drop=True)
    assert len(a) == len(b)
    for c in cols:
        if c == "date":
            assert (a[c] == b[c]).all()
        elif c in TEXT:
            assert (a[c].astype(str) == b[c].astype(str)).all(), c
        else:
            np.testing.assert_allclose(a[c].astype(float), b[c].astype(float), rtol=1e-6, atol=1e-9,
                                       equal_nan=True, err_msg=c)


def test_oil_daily_matches_pandas(spark_tables, raw):
    got = spark_tables["oil"].toPandas().sort_values("date").reset_index(drop=True)
    exp = pandas_oil(raw["oil"])
    np.testing.assert_allclose(got["oil_price"].to_numpy(), exp["oil_price"].to_numpy())


def test_silver_matches_pandas(spark_tables, silver):
    _assert_frames_equal(spark_tables["silver"].toPandas(), silver, SILVER_COLUMNS)


def test_feature_table_matches_pandas(spark_tables, features):
    got = se.build_feature_table(spark_tables["silver"]).toPandas()
    assert list(got.columns) == list(features.columns)
    _assert_frames_equal(got, features, list(features.columns))
