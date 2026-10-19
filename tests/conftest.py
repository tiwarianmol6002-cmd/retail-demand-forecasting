import os
import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from demand.etl import build_silver_pandas  # noqa: E402
from demand.features import build_features  # noqa: E402
from demand.synthetic import generate_raw  # noqa: E402


@pytest.fixture(scope="session")
def raw():
    return generate_raw(n_stores=3, start="2016-01-01")


@pytest.fixture(scope="session")
def silver(raw):
    return build_silver_pandas(raw)


@pytest.fixture(scope="session")
def features(silver):
    return build_features(silver)


@pytest.fixture()
def in_tmp(tmp_path, monkeypatch):
    """Run in a temp dir so MLflow artifacts never pollute the repo."""
    monkeypatch.chdir(tmp_path)
    return tmp_path
