#!/usr/bin/env bash
# Three-component band envelopes (envelopes.py --component ZNE) of every station that is not held out.
# usage: scripts/build_envelopes_zne.sh OUT_DIR JOBS LOCAL DRIVE
#        LOCAL: the folder with stations/ELZG and stations_raw/{sivrice_raw,sivrice_fdsn_raw}   (~/data)
#        DRIVE: the folder with data/stations and tdvms                                          (/mnt/port)
# The day files of FRT, KOVA, MAYA, NARI and SVRC hold the north component alone and are not read.
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=$1; JOBS=$2; LOCAL=$3; DRIVE=$4; status=0
run() {   # STATION ROOT [ROOT ...]
  s=$1; shift
  echo "[$(date '+%F %T')] $s"
  uv run python -u -m forecasting_earthquakes_hopefully.envelopes "$OUT" "$@" --station "$s" --component ZNE --jobs "$JOBS" || status=1
}
run ELZG "$LOCAL/stations"
run SVRC "$LOCAL/stations_raw/sivrice_fdsn_raw" "$LOCAL/stations_raw/sivrice_raw" "$DRIVE/data/stations"
run FRT "$LOCAL/stations_raw/sivrice_raw" "$DRIVE/data/stations"
for s in KOVA MAYA NARI; do run $s "$LOCAL/stations_raw/sivrice_raw"; done
for s in CNG PURS STEP BKES DURS AKHS DEMI TVAN VMUR GEVA DYDN; do run $s "$DRIVE/data/stations"; done
echo "[$(date '+%F %T')] ENVELOPES DONE status=$status"
[ $status -eq 0 ] && touch "$OUT/ALL_DONE"
exit $status
