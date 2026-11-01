#!/bin/bash
# Run T's matched evaluations (bench/notes/TOPSTART.md section 5): one checkpoint against Foul Play at a fixed 88,000
# iterations (400 games, 4 processes), then simple and psbot (200 games each), with evaluate.py from the topstart
# release, into ~/dev/topstart/eval-NAME (its own eval/results.jsonl and REPORT.md). Never a run directory: S's and
# T's checkpoints are only read.
#
#   bash ~/rel/topstart/cloud/v2/topstart_eval.sh NAME CHECKPOINT PORT [GPU]
#   e.g. bash ~/rel/topstart/cloud/v2/topstart_eval.sh T0 ~/dev/topstart/start/T0.pt 10791 3
#
# Ports: T's servers use 10700-10735 and its evaluator 10790; these evaluations take 10791-10799. Each one needs
# its own port while it runs. FP_GAMES / OTHER_GAMES / FP_PARALLEL override the counts (the rehearsal's smoke).
# It runs in the foreground at nice 10; start it with setsid nohup ... & to leave it running.
set -u
NAME=${1:?NAME}; CK=${2:?CHECKPOINT}; PORT=${3:?PORT}; GPU=${4:-3}
REL=${REL:-$HOME/rel/topstart}; OUT=$HOME/dev/topstart/eval-$NAME; PY=$HOME/venv/bin/python
FP_GAMES=${FP_GAMES:-400}; OTHER_GAMES=${OTHER_GAMES:-200}; FP_PARALLEL=${FP_PARALLEL:-4}
case "$PORT" in 1079[1-9]) ;; *) [ -n "${ALLOW_PORT:-}" ] || { echo "port $PORT: use 10791-10799 (ALLOW_PORT=1 to override)"; exit 2; };; esac
[ -f "$CK" ] || { echo "no checkpoint at $CK"; exit 2; }
case "$(readlink -f "$OUT")" in "$HOME"/runs/*) echo "refused: $OUT is inside ~/runs"; exit 2;; esac
mkdir -p "$OUT"
cd "$REL" || exit 2
export CUDA_VISIBLE_DEVICES=$GPU FOULPLAY_DIR=$HOME/foul-play PYTHONUTF8=1 PYTHONUNBUFFERED=1 PSPPO_GPU_SPLIT=1
echo "$(date -u +%FT%TZ) $NAME: $CK ($(sha256sum "$CK" | cut -c1-16)...) port $PORT GPU $GPU" | tee -a "$OUT/runs.log"
nice -n 10 $PY evaluate.py --run-dir "$OUT" --checkpoint "$CK" --opponents foulplay --games "$FP_GAMES" \
  --foulplay-iterations 88000 --foulplay-parallel "$FP_PARALLEL" --port "$PORT" --server-dir "$HOME/pokemon-showdown" >> "$OUT/foulplay.log" 2>&1
echo "$(date -u +%FT%TZ) $NAME: foulplay exit $?" | tee -a "$OUT/runs.log"
nice -n 10 $PY evaluate.py --run-dir "$OUT" --checkpoint "$CK" --opponents simple,psbot --games "$OTHER_GAMES" \
  --port "$PORT" --server-dir "$HOME/pokemon-showdown" >> "$OUT/others.log" 2>&1
echo "$(date -u +%FT%TZ) $NAME: simple,psbot exit $?" | tee -a "$OUT/runs.log"
python3 - "$OUT/eval/results.jsonl" <<'P' | tee -a "$OUT/runs.log"
import json, sys
for line in open(sys.argv[1]):
    r = json.loads(line)
    print(f"  {r['opponent']:<9} {r['wins']:>4}-{r['losses']:<4} of {r['games']} ({r['errors']} errors)  {100 * r['winrate']:.1f}%  "
          f"95% {100 * r['ci_low']:.1f} to {100 * r['ci_high']:.1f}  {r['checkpoint']}")
P
