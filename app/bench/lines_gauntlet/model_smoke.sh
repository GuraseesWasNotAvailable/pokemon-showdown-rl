#!/usr/bin/env bash
# On the box: real games of the model arms with reading ahead (sol-lines and luna-lines, one per
# game) against the network alone, on a local Showdown server, the models played by the fake API
# of tests/fake_decisions.py (answers 0.7 to 1.8 s late, 8% of them errors). No key of OpenAI's:
# the fake has a key of its own. ladder.py is run as the window app runs it (--app-events).
#     bash model_smoke.sh [GAMES]
set -u
GAMES=${1:-10}
D=$HOME/dev/model-smoke
CODE=$HOME/dev/decide
PY=$HOME/venv/bin/python
CKPT=$HOME/dev/ab-ladder-ckpt/learner_update_024381.pt
PORT=10502
API=11460
KEY=sk-fake-smoke-key-000000000001
mkdir -p "$D"
cd "$CODE" || exit 1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PSBOT_LOCK_DIR=$D/locks
if ss -ltn | grep -q ":$PORT "; then echo "port $PORT in use"; exit 1; fi
setsid "$PY" start_servers.py --dir "$HOME/pokemon-showdown" --ports "$PORT" --log-dir "$D/srv" > "$D/srv.out" 2>&1 < /dev/null &
SRV=$!
for _ in $(seq 1 90); do ss -ltn | grep -q ":$PORT " && break; sleep 1; done
FAKE_LUNA_KEY=$KEY setsid "$PY" tests/fake_decisions.py --port "$API" --mode mix --delay 0.7 1.8 --error-rate 0.08 --usage --log "$D/fake.jsonl" > "$D/fake.out" 2>&1 < /dev/null &
FAKE=$!
sleep 2
"$PY" ladder.py --server "local:$PORT" --user netsmoke1 --player net --games "$GAMES" --parallel 3 --checkpoint "$CKPT" \
  --out "$D/net/results.jsonl" > "$D/net.out" 2>&1 < /dev/null &
NET=$!
printf 'unused-password\n%s\n' "$KEY" | "$PY" ladder.py --server "local:$PORT" --user solsmoke1 --arms sol-lines,luna-lines --games "$GAMES" \
  --parallel 3 --helpers 4 --search-ms 500 --deadline 2.5 --checkpoint "$CKPT" --out "$D/model/results.jsonl" \
  --sol-url "http://127.0.0.1:$API/v1/responses" --luna-url "http://127.0.0.1:$API/v1/decisions" \
  --app-events --password-stdin --key-stdin > "$D/model.out" 2>&1
echo "model exit $?" >> "$D/model.out"
for _ in $(seq 1 120); do kill -0 $NET 2>/dev/null || break; sleep 1; done
kill $NET 2>/dev/null
kill -- -$FAKE 2>/dev/null
kill -- -$SRV 2>/dev/null
echo done > "$D/done"
