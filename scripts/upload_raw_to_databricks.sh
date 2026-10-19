#!/usr/bin/env bash
# Copy the raw CSVs into the Unity Catalog volume the Bronze notebook reads from.
# Run `databricks bundle run` once first (or the 00_setup notebook) so the volume exists.
set -euo pipefail
CATALOG="${CATALOG:-retail_forecasting}"
DEST="dbfs:/Volumes/$CATALOG/bronze/raw_data"
for f in train test stores oil holidays_events transactions; do
  databricks fs cp "data/raw/$f.csv" "$DEST/$f.csv" --overwrite
done
databricks fs ls "$DEST"
