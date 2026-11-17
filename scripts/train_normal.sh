#!/bin/sh
# The normal-days model, one fold after the other.  scripts/train_normal.sh ENVELOPES_DIR OUT_DIR REGION FOLDS
set -eu
for f in $(seq 0 $(($4 - 1))); do
    uv run python -u -m forecasting_earthquakes_hopefully.normal "$1" "$2" --region "$3" --fold "$f" --folds "$4" \
        > "logs/normal_$3_f$f.log" 2>&1
done
touch "$2/ALL_DONE"
