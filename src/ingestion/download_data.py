"""Get the raw CSV files into data/raw/.

    python src/ingestion/download_data.py --source synthetic     # tiny demo data (no account needed)
    python src/ingestion/download_data.py --source zip --zip ~/Downloads/store-sales-time-series-forecasting.zip
    python src/ingestion/download_data.py --source kaggle        # needs a configured Kaggle API token + accepted rules
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demand.etl import RAW_FILES  # noqa: E402
from demand.synthetic import generate_raw, write_raw  # noqa: E402

COMPETITION = "store-sales-time-series-forecasting"


def check(out: Path) -> None:
    missing = [f for f in RAW_FILES.values() if not (out / f).exists()]
    if missing:
        raise SystemExit(f"Still missing in {out}: {missing}")
    for f in RAW_FILES.values():
        print(f"  {f:<24}{(out / f).stat().st_size / 1e6:8.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "zip", "kaggle"], required=True)
    ap.add_argument("--zip", help="path to the downloaded competition zip (for --source zip)")
    ap.add_argument("--out", default="data/raw")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.source == "synthetic":
        write_raw(generate_raw(), out)
    else:
        if a.source == "kaggle":
            subprocess.run(["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(out)], check=True)
            zip_path = out / f"{COMPETITION}.zip"
        else:
            if not a.zip:
                raise SystemExit("--zip is required with --source zip")
            zip_path = Path(a.zip).expanduser()
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out)
    check(out)


if __name__ == "__main__":
    main()
