#!/usr/bin/env bash
# Train and score the classifier for every variant of the question and several seeds.
# usage: scripts/sweep_question.sh LABELS_GRID_DIR OUT_DIR [SEEDS=5] [PARALLEL=4]
#   LABELS_GRID_DIR/<cell>/labels.csv   one label table per variant (labels.py with --min-magnitude, --radius-km,
#                                       --lead-days); the name of the directory is the name of the cell
#   OUT_DIR/<cell>/seed<k>/             the run of train.py with --seed k; logs in logs/<name of OUT_DIR>/
# The model and the training are those of train.py without any flag: only the labels and the seed change.
# A run that has its DONE file is skipped, so the same command goes on after a stop and repeats the runs
# that failed; ALL_DONE is written only when every run has its DONE file. Seeds go first to last,
# so every cell has seed 0 before any has seed 1. Start it with setsid; it writes OUT_DIR/build.pgid for
# scripts/build_control.sh OUT_DIR pause|resume|stop|status.
set -u
GRID=$1; OUT=$2; SEEDS=${3:-5}; PAR=${4:-4}
LOGS=logs/$(basename "$OUT")
mkdir -p "$OUT" "$LOGS"
ps -o pgid= -p $$ | tr -d ' ' > "$OUT/build.pgid"
one() {
  cell=$1; seed=$2; out=$OUT/$cell/seed$seed
  [ -f "$out/DONE" ] && return 0
  echo "$(date +%F\ %T) start $cell seed $seed"
  nice -n 19 uv run python -u -m forecasting_earthquakes_hopefully.train data/minutes "$GRID/$cell/labels.csv" "$out" \
    --seed "$seed" > "$LOGS/${cell}_seed$seed.log" 2>&1
  rc=$?
  echo "$(date +%F\ %T) end   $cell seed $seed exit $rc"
}
export -f one; export GRID OUT LOGS
for seed in $(seq 0 $((SEEDS - 1))); do
  for d in "$GRID"/*/; do echo "$(basename "$d") $seed"; done
done | xargs -P "$PAR" -L 1 bash -c 'one "$0" "$1"'
n=$(find "$OUT" -name DONE | wc -l); want=$(( SEEDS * $(ls -d "$GRID"/*/ | wc -l) ))
echo "$(date +%F\ %T) finished: $n of $want runs done"
[ "$n" -eq "$want" ] || exit 1
touch "$OUT/ALL_DONE"
