#!/usr/bin/env bash
# Three-component band envelopes (envelopes.py --component ZNE) of every station that is not held out.
# usage: setsid nohup nice -n 19 scripts/build_envelopes_zne.sh OUT_DIR JOBS LOCAL DRIVE > LOG 2>&1 < /dev/null &
#        A stopped build goes on where it was when the same command is given again (scripts/build_control.sh).
#        LOCAL: the folder with stations/ELZG and stations_raw/{sivrice_raw,sivrice_fdsn_raw}   (~/data)
#        DRIVE: the folder with data/stations and tdvms                                          (/mnt/port)
# The day files of FRT, KOVA, MAYA, NARI and SVRC hold the north component alone and are not read.
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=$1; JOBS=$2; LOCAL=$3; DRIVE=$4; status=0
mkdir -p "$OUT"; echo $$ > "$OUT/build.pgid"      # started with setsid this is the process group: scripts/build_control.sh
# One thread per worker: the filtering gains nothing from more (measured), the threads only compete.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
run() {   # STATION ROOT [ROOT ...]
  s=$1; shift
  echo "[$(date '+%F %T')] $s"
  uv run python -u -m forecasting_earthquakes_hopefully.envelopes "$OUT" "$@" --station "$s" --component ZNE --jobs "$JOBS" || status=1
}
# The stations read from the drive go first, so that the drive is free as early as possible.
for s in CNG PURS STEP BKES DURS AKHS DEMI TVAN VMUR GEVA DYDN; do run $s "$DRIVE/data/stations"; done
run SVRC "$LOCAL/stations_raw/sivrice_fdsn_raw" "$LOCAL/stations_raw/sivrice_raw" "$DRIVE/data/stations"
run FRT "$LOCAL/stations_raw/sivrice_raw" "$DRIVE/data/stations"
echo "[$(date '+%F %T')] THE DRIVE IS NO LONGER READ"
for s in KOVA MAYA NARI; do run $s "$LOCAL/stations_raw/sivrice_raw"; done
run ELZG "$LOCAL/stations"
echo "[$(date '+%F %T')] ENVELOPES DONE status=$status"
[ $status -eq 0 ] && touch "$OUT/ALL_DONE"
exit $status
