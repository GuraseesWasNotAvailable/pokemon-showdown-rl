#!/usr/bin/env bash
# The counter gauntlet's own local Showdown server on PORT (10500-10599 only), in its own process group.  server.sh PORT
set -u
PORT=${1:?port}
D=$HOME/dev/counter-gauntlet
cd "$HOME/dev/counter-play"
if ss -ltn | grep -q ":$PORT "; then echo "port $PORT in use"; exit 1; fi
mkdir -p "$D/srv$PORT"
setsid nice -n 15 "$HOME/venv/bin/python" start_servers.py --dir "$HOME/pokemon-showdown" --ports "$PORT" --log-dir "$D/srv$PORT" > "$D/srv$PORT.out" 2>&1 < /dev/null &
echo "srv $!" > "$D/srv$PORT.pgid"
for _ in $(seq 1 90); do ss -ltn | grep -q ":$PORT " && break; sleep 1; done
ss -ltn | grep -q ":$PORT " && echo "server up on $PORT (pgid $(cat $D/srv$PORT.pgid))" || echo "server did not start"
