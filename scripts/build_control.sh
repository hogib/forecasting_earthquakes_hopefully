#!/usr/bin/env bash
# Pause, resume or stop a run of this repository that was started with setsid and wrote its process group
# to OUT_DIR/build.pgid: the envelope build of scripts/build_envelopes_zne.sh or a training run.
# usage: scripts/build_control.sh OUT_DIR pause|resume|stop|status
#   pause   freezes every process of the build: the processor is free at once, the memory stays taken
#   resume  lets a paused build go on
#   stop    ends the build and frees the memory; the files in work are lost (minutes), everything else is kept,
#           and starting the build again with the same command goes on from there
set -u
G=$(cat "$1/build.pgid") || exit 1
if ! ps -o args= -g "$G" | grep -q -E "build_envelopes_zne|forecasting_earthquakes_hopefully"; then echo "no build is running in group $G"; exit 1; fi
case $2 in
  pause)  kill -STOP -- "-$G" && echo "paused" ;;
  resume) kill -CONT -- "-$G" && echo "resumed" ;;
  stop)   kill -CONT -- "-$G"; kill -TERM -- "-$G" && echo "stopped" ;;
  status) ps -o pid,stat,etime,rss,args -g "$G" | cut -c1-110 ;;
  *) echo "usage: $0 OUT_DIR pause|resume|stop|status"; exit 2 ;;
esac
