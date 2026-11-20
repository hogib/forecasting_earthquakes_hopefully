#!/usr/bin/env bash
# The small events around each region from EMSC, for the count rival (rivals.py).  usage: scripts/fetch_small_events.sh OUT_DIR
# Centre and radius cover every station of the region with 50 km to spare; the periods cover the recordings
# and the 28 days before them.
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=$1; status=0
get() { uv run python -u -m forecasting_earthquakes_hopefully.small_events "$OUT/$1.csv" --lat "$2" --lon "$3" --radius-km 150 --start "$4" --end "$5" || status=1; }
get van_2011       38.90 43.40 2011-06-20 2011-11-12
get van_2026       38.90 43.40 2025-12-01 2026-04-07
get balikesir      39.30 28.20 2024-08-25 2026-08-12
get elazig         38.40 39.10 2014-12-20 2021-06-30
echo "[$(date '+%F %T')] SMALL EVENTS DONE status=$status"
exit $status
