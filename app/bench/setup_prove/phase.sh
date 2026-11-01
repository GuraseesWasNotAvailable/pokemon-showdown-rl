#!/usr/bin/env bash
# One measured phase of the set-up proof: an opponent process (gauntlet.py opp) and ours (gauntlet.py ours) on PORT.
#   phase.sh NAME KIND ARMS PARALLEL HELPERS MINUTES PORT FP_MS FP_HELPERS SEED THINK_MS
set -u
NAME=${1:?name}; KIND=${2:?kind}; ARMS=${3:?arms}; PAR=${4:?parallel}; HELP=${5:?helpers}; MIN=${6:?minutes}
PORT=${7:?port}; FPMS=${8:-450}; FPH=${9:-2}; SEED=${10:-21}; THINK=${11:-0}
D=$HOME/dev/setup-prove
CODE=$HOME/dev/setup-prove-play
PY=$HOME/venv/bin/python
CKPT=$D/learner_update_011850.pt
OUT=$D/g/$NAME
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
if [ -e "$OUT/results.jsonl" ]; then echo "$OUT already has results; not starting"; exit 1; fi
mkdir -p "$OUT"
cd "$CODE"
OURS="sp${NAME}x"; ACC="zacc${NAME}"; CHA="zcha${NAME}"
echo "$(date -u +%FT%TZ) phase $NAME: $KIND, arms $ARMS, parallel $PAR, helpers $HELP, $MIN min, port $PORT, fp $FPMS ms x $FPH, seed $SEED, think $THINK" | tee -a "$OUT/phase.log"
setsid nice -n 15 "$PY" bench/gauntlet.py opp --port $PORT --kind "$KIND" --pairs "$ACC:$CHA:$OURS" --fp-ms "$FPMS" --fp-helpers "$FPH" \
  --checkpoint "$CKPT" --out "$OUT/opp.jsonl" --stop-file "$OUT/opp.stop" > "$OUT/opp.out" 2>&1 < /dev/null &
OPP=$!
echo "opp $OPP" >> "$OUT/pgids"
for _ in $(seq 1 240); do grep -q "ready" "$OUT/opp.out" 2>/dev/null && break; kill -0 $OPP 2>/dev/null || break; sleep 1; done
if ! grep -q "ready" "$OUT/opp.out"; then echo "opponent did not get ready" | tee -a "$OUT/phase.log"; kill -- -$OPP 2>/dev/null; exit 1; fi
setsid nice -n 15 "$PY" bench/gauntlet.py ours --port $PORT --user "$OURS" --arms "$ARMS" --per-arm 400 --parallel "$PAR" --helpers "$HELP" \
  --search-ms 500 --deadline 2.5 --mix-rule veto --checkpoint "$CKPT" --opp-accept "$ACC" --opp-challenge "$CHA" \
  --out "$OUT/results.jsonl" --stop-file "$OUT/stop" --minutes "$MIN" --seed "$SEED" --phase "$NAME" --think-longer "$THINK" > "$OUT/ours.out" 2>&1 < /dev/null &
OURSPID=$!
echo "ours $OURSPID" >> "$OUT/pgids"
wait $OURSPID
echo "$(date -u +%FT%TZ) ours exit $?" | tee -a "$OUT/phase.log"
touch "$OUT/opp.stop"
for _ in $(seq 1 90); do kill -0 $OPP 2>/dev/null || break; sleep 1; done
if kill -0 $OPP 2>/dev/null; then kill -- -$OPP 2>/dev/null; echo "opponent group killed" | tee -a "$OUT/phase.log"; fi
sleep 2
echo "$(date -u +%FT%TZ) phase $NAME done" | tee -a "$OUT/phase.log"
