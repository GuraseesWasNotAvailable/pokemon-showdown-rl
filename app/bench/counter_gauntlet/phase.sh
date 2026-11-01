#!/usr/bin/env bash
# One measured phase of the counter gauntlet: today's mix against mix-counter (round 3's two rules), one
# opponent process (gauntlet.py opp) and ours (gauntlet.py ours) on PORT. The same player plays both arms,
# so both share the search time, the deadline and the helpers; the arm is balanced within each side.
#   phase.sh NAME KIND PARALLEL HELPERS MINUTES PER_ARM PORT FP_MS FP_HELPERS SEED
set -u
NAME=${1:?name}; KIND=${2:?kind}; PAR=${3:?parallel}; HELP=${4:?helpers}; MIN=${5:?minutes}; PER=${6:?per arm}
PORT=${7:?port}; FPMS=${8:-450}; FPH=${9:-2}; SEED=${10:-41}
ARMS=mix,mix-counter
D=$HOME/dev/counter-gauntlet
CODE=$HOME/dev/counter-play
PY=$HOME/venv/bin/python
CKPT=$HOME/dev/setup-prove/learner_update_011850.pt
OUT=$D/g/$NAME
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
if [ -e "$OUT/results.jsonl" ]; then echo "$OUT already has results; not starting"; exit 1; fi
mkdir -p "$OUT"
cd "$CODE"
OURS="spc${NAME}x"; ACC="zacc${NAME}"; CHA="zcha${NAME}"
echo "$(date -u +%FT%TZ) phase $NAME: $KIND, arms $ARMS, parallel $PAR, helpers $HELP, $MIN min, $PER per arm, port $PORT, fp $FPMS ms x $FPH, seed $SEED" | tee -a "$OUT/phase.log"
setsid nice -n 15 "$PY" bench/gauntlet.py opp --port $PORT --kind "$KIND" --pairs "$ACC:$CHA:$OURS" --fp-ms "$FPMS" --fp-helpers "$FPH" \
  --checkpoint "$CKPT" --out "$OUT/opp.jsonl" --stop-file "$OUT/opp.stop" > "$OUT/opp.out" 2>&1 < /dev/null &
OPP=$!
echo "opp $OPP" >> "$OUT/pgids"
for _ in $(seq 1 240); do grep -q "ready" "$OUT/opp.out" 2>/dev/null && break; kill -0 $OPP 2>/dev/null || break; sleep 1; done
if ! grep -q "ready" "$OUT/opp.out"; then echo "opponent did not get ready" | tee -a "$OUT/phase.log"; kill -- -$OPP 2>/dev/null; exit 1; fi
echo "$(date -u +%FT%TZ) opponent ready" | tee -a "$OUT/phase.log"
setsid nice -n 15 "$PY" bench/gauntlet.py ours --port $PORT --user "$OURS" --arms "$ARMS" --per-arm "$PER" --parallel "$PAR" --helpers "$HELP" \
  --search-ms 500 --deadline 2.5 --mix-rule veto --checkpoint "$CKPT" --opp-accept "$ACC" --opp-challenge "$CHA" \
  --out "$OUT/results.jsonl" --stop-file "$OUT/stop" --minutes "$MIN" --seed "$SEED" --phase "$NAME" > "$OUT/ours.out" 2>&1 < /dev/null &
OURSPID=$!
echo "ours $OURSPID" >> "$OUT/pgids"
wait $OURSPID
echo "$(date -u +%FT%TZ) ours exit $?" | tee -a "$OUT/phase.log"
touch "$OUT/opp.stop"
for _ in $(seq 1 90); do kill -0 $OPP 2>/dev/null || break; sleep 1; done
if kill -0 $OPP 2>/dev/null; then kill -- -$OPP 2>/dev/null; echo "opponent group killed" | tee -a "$OUT/phase.log"; fi
sleep 2
kill -0 -- -$OURSPID 2>/dev/null && { kill -- -$OURSPID 2>/dev/null; echo "leftover of our group killed" | tee -a "$OUT/phase.log"; }
echo "$(date -u +%FT%TZ) phase $NAME done" | tee -a "$OUT/phase.log"
