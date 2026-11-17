#!/usr/bin/env bash
# Band envelopes (envelopes.py) of several stations, one after the other.
# usage: scripts/build_envelopes.sh RAW_ROOT OUT_DIR COMPONENT JOBS STATION [STATION ...]
#        (RAW_ROOT/<STATION>/ holds the files; one OUT_DIR per component)
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$1; OUT=$2; COMP=$3; JOBS=$4; shift 4; status=0
for s in "$@"; do
  echo "[$(date '+%F %T')] $s"
  uv run python -u -m forecasting_earthquakes_hopefully.envelopes "$OUT" "$ROOT" --station "$s" --component "$COMP" --jobs "$JOBS" || status=1
done
echo "[$(date '+%F %T')] ENVELOPES DONE status=$status"
exit $status
