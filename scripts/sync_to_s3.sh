#!/usr/bin/env bash
# Archive the raw Kaggle CSVs in S3.  Usage: BUCKET=your-bucket ./scripts/sync_to_s3.sh
set -euo pipefail
BUCKET="${BUCKET:?Set BUCKET=your-bucket-name}"
PREFIX="${PREFIX:-retail_demand/raw}"
aws s3 sync data/raw/ "s3://$BUCKET/$PREFIX/" --exclude "*" --include "*.csv"
aws s3 ls "s3://$BUCKET/$PREFIX/" --human-readable
