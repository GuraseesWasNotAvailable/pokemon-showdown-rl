#!/usr/bin/env bash
# On the box: today's mix against mix-lines (reading ahead), the same player playing both arms,
# balanced within each side, against two opponents at once on two local servers: Foul Play at fair
# compute (port 10500) and the network alone (port 10501). No game is started after MIN minutes.
#     bash lines_gauntlet.sh [MIN] [PER_ARM]
# The scripts of bench/counter_gauntlet (phase.sh, server.sh) with this run's folder, code and arms.
set -u
MIN=${1:-35}
PER=${2:-400}
D=$HOME/dev/lines-gauntlet
CODE=$HOME/dev/decide
PY=$HOME/venv/bin/python
CKPT=$HOME/dev/ab-ladder-ckpt/learner_update_024381.pt
ARMS=mix,mix-lines
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$D/g"
cd "$CODE" || exit 1

server() {
  local PORT=$1
  if ss -ltn | grep -q ":$PORT "; then echo "port $PORT in use"; return 1; fi
  mkdir -p "$D/srv$PORT"
  setsid nice -n 15 "$PY" start_servers.py --dir "$HOME/pokemon-showdown" --ports "$PORT" --log-dir "$D/srv$PORT" > "$D/srv$PORT.out" 2>&1 < /dev/null &
  echo "$!" > "$D/srv$PORT.pgid"
  for _ in $(seq 1 90); do ss -ltn | grep -q ":$PORT " && break; sleep 1; done
  ss -ltn | grep -q ":$PORT " && echo "server up on $PORT" || { echo "server on $PORT did not start"; return 1; }
}

phase() {
  local NAME=$1 KIND=$2 PAR=$3 HELP=$4 PORT=$5 FPMS=$6 FPH=$7 SEED=$8
  local OUT=$D/g/$NAME
  if [ -e "$OUT/results.jsonl" ]; then echo "$OUT already has results; not starting"; return 1; fi
  mkdir -p "$OUT"
  local OURS="lin${NAME}x" ACC="zacc${NAME}" CHA="zcha${NAME}"
  echo "$(date -u +%FT%TZ) phase $NAME: $KIND, arms $ARMS, parallel $PAR, helpers $HELP, $MIN min, $PER per arm, port $PORT, fp $FPMS ms x $FPH, seed $SEED" | tee -a "$OUT/phase.log"
  setsid nice -n 15 "$PY" bench/gauntlet.py opp --port "$PORT" --kind "$KIND" --pairs "$ACC:$CHA:$OURS" --fp-ms "$FPMS" --fp-helpers "$FPH" \
    --checkpoint "$CKPT" --out "$OUT/opp.jsonl" --stop-file "$OUT/opp.stop" > "$OUT/opp.out" 2>&1 < /dev/null &
  local OPP=$!
  for _ in $(seq 1 240); do grep -q "ready" "$OUT/opp.out" 2>/dev/null && break; kill -0 $OPP 2>/dev/null || break; sleep 1; done
  if ! grep -q "ready" "$OUT/opp.out"; then echo "opponent did not get ready" | tee -a "$OUT/phase.log"; kill -- -$OPP 2>/dev/null; return 1; fi
  echo "$(date -u +%FT%TZ) opponent ready" | tee -a "$OUT/phase.log"
  setsid nice -n 15 "$PY" bench/gauntlet.py ours --port "$PORT" --user "$OURS" --arms "$ARMS" --per-arm "$PER" --parallel "$PAR" --helpers "$HELP" \
    --search-ms 500 --deadline 2.5 --mix-rule veto --checkpoint "$CKPT" --opp-accept "$ACC" --opp-challenge "$CHA" \
    --out "$OUT/results.jsonl" --stop-file "$OUT/stop" --minutes "$MIN" --seed "$SEED" --phase "$NAME" > "$OUT/ours.out" 2>&1 < /dev/null &
  local OURSPID=$!
  wait $OURSPID
  echo "$(date -u +%FT%TZ) ours exit $?" | tee -a "$OUT/phase.log"
  touch "$OUT/opp.stop"
  for _ in $(seq 1 90); do kill -0 $OPP 2>/dev/null || break; sleep 1; done
  kill -0 $OPP 2>/dev/null && { kill -- -$OPP 2>/dev/null; echo "opponent group killed" | tee -a "$OUT/phase.log"; }
  sleep 2
  kill -0 -- -$OURSPID 2>/dev/null && { kill -- -$OURSPID 2>/dev/null; echo "leftover of our group killed" | tee -a "$OUT/phase.log"; }
  echo "$(date -u +%FT%TZ) phase $NAME done" | tee -a "$OUT/phase.log"
}

echo "$(date -u +%FT%TZ) start: $MIN minutes, $PER per arm" >> "$D/master.log"
server 10500 >> "$D/master.log" 2>&1 || exit 1
server 10501 >> "$D/master.log" 2>&1 || exit 1
phase fp foulplay 5 6 10500 450 3 51 >> "$D/master.log" 2>&1 &
P1=$!
phase nt net 5 6 10501 450 2 52 >> "$D/master.log" 2>&1 &
P2=$!
wait $P1 $P2          # (the two phases only: the servers are children of this shell too, and never end by themselves)
for PORT in 10500 10501; do kill -- -"$(cat "$D/srv$PORT.pgid")" 2>/dev/null; done
sleep 3
"$PY" bench/gauntlet.py report "$D/g/fp" "$D/g/nt" --json "$D/report.json" > "$D/report.txt" 2>&1
echo "$(date -u +%FT%TZ) all done" >> "$D/master.log"
