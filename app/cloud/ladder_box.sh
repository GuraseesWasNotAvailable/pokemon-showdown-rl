#!/usr/bin/env bash
# The lookahead ladder bot on the training box, in a tmux session. The Desktop's Net Play, Net Stop
# and Net Elo call this through cloud/vm.sh (vm.sh tty for play, vm.sh ssh for the others).
#
#   bash ~/ladder/cloud/ladder_box.sh play [test]     attach to the bot's tmux session, starting the bot
#                                                     in it first when there is no session
#   bash ~/ladder/cloud/ladder_box.sh stop [test]     no new games; the ones in progress are finished
#   bash ~/ladder/cloud/ladder_box.sh elo [test]      write the rating graph; the page (HTML) goes to stdout
#   bash ~/ladder/cloud/ladder_box.sh status [test]   running or not, and the last lines it printed
#   bash ~/ladder/cloud/ladder_box.sh run [test]      what runs inside the session (play starts it)
#
# Live: play.pokemonshowdown.com, the code of this tree (~/ladder), the network ckpt/current.pt (a
# link to the checkpoint file), results in runs/ladder/ (results.jsonl, elo.html, battles/, bot.log).
# The player is "mix" (the network and Foul Play's search together, rule veto) in every game:
# search 500 ms per decision on 4 helpers, 3 games at a time, battle timer on, the network on the
# CPU (the GPU is the trainer's), at nice 5 so that the trainer comes first.
#
# test: the same code and the same settings against a LOCAL server (port $LADDER_TEST_PORT, default
# 9300, from the Showdown copy in ~/dev/ladder/ps) with SimpleHeuristics opponents laddering on it,
# everything in ~/dev/ladder/live-test/, tmux socket ladder-test. Nothing is sent to the public
# server or to its login server: the password is asked for at the prompt and dropped.
#
# What the owner types, in the tmux window: the Showdown username, the number of games and the
# password. The password is read by ladder.py itself (getpass): it is not shown, not stored, not on
# any command line and not in any file. PS_PASSWORD is removed from the environment first.
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
CMD=${1:-play}
MODE=${2:-live}
case "$MODE" in live|test) ;; *) echo "the second word is 'test' or nothing" >&2; exit 2 ;; esac
if [ "$MODE" = test ]; then
  SOCK=ladder-test
  DIR=$HOME/dev/ladder/live-test
  PORT=${LADDER_TEST_PORT:-9300}
  SERVER="local:$PORT"
  NICE=15
else
  SOCK=ladder
  DIR=$HERE/runs/ladder
  PORT=
  SERVER=public
  NICE=5
fi
SESSION=$SOCK
RESULTS=$DIR/results.jsonl
STOP=$DIR/ladder.stop
mkdir -p "$DIR"
T=(tmux -u -L "$SOCK")
PY=$HOME/venv/bin/python

running() {   # the PID of this mode's ladder.py, if it runs
  pgrep -u "$(id -u)" -f "ladder.py --server $SERVER .*--out $RESULTS" | head -1
}

case "$CMD" in
  play)
    if ! "${T[@]}" has-session -t "=$SESSION" 2>/dev/null; then
      "${T[@]}" new-session -d -s "$SESSION" -x 160 -y 48 "bash '$HERE/cloud/ladder_box.sh' run $MODE" || exit 1
      "${T[@]}" set-option -t "$SESSION" history-limit 100000 >/dev/null
      "${T[@]}" set-option -t "$SESSION" status-right " Ctrl+B then D: leave, the bot plays on " >/dev/null
      "${T[@]}" set-option -t "$SESSION" status-right-length 60 >/dev/null
    fi
    if ! infocmp "${TERM:-dumb}" >/dev/null 2>&1; then export TERM=xterm-256color; fi
    exec "${T[@]}" attach -t "=$SESSION"
    ;;

  run)
    cd "$HERE" || exit 1
    unset PS_PASSWORD
    export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    export PYTHONWARNINGS="ignore:flex_attention called without torch.compile"   # harmless on the CPU, and a screenful
    finish() {
      echo
      echo "The bot is not running now. Press Enter to close this window (Net Play then starts a new run)."
      read -r -t 3600 _ || true
      exit "${1:-0}"
    }
    CKPT=$(readlink -f "$HERE/ckpt/current.pt" 2>/dev/null)
    if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then echo "no network: $HERE/ckpt/current.pt is missing"; finish 1; fi
    if [ -n "$(running)" ]; then echo "the bot already runs (pid $(running))"; finish 1; fi
    echo "PSBot network + look-ahead on the Showdown ladder ($([ "$MODE" = test ] && echo "TEST: local server $SERVER, SimpleHeuristics opponents" || echo "play.pokemonshowdown.com"))"
    echo "network $(basename "$CKPT"), player mix (rule veto), search 500 ms on 4 helpers, 3 games at a time, timer on"
    echo "Leave this window with Ctrl+B then D, or just close it: the bot keeps playing. Net Play comes back here."
    echo "To stop: Net Stop (the games in progress are finished first). Ctrl+C here abandons the games in progress."
    echo
    LAST=$(head -c 40 "$DIR/last_user" 2>/dev/null)
    read -r -p "Showdown username${LAST:+ (Enter for $LAST)}: " U
    U=${U:-$LAST}
    if [ -z "$U" ]; then echo "no username given"; finish 1; fi
    read -r -p "How many games (Enter for 20): " G
    G=${G:-20}
    case "$G" in ''|*[!0-9]*) echo "not a number: $G"; finish 1 ;; esac
    printf '%s\n' "$U" > "$DIR/last_user"
    if [ -e "$STOP" ]; then mv "$STOP" "$STOP.stale.$(date -u +%Y%m%dT%H%M%S)"; fi   # an old stop file must not end this run at once
    EXTRA=()
    if [ "$MODE" = test ]; then
      EXTRA=(--ask-password)
      if ss -ltn | grep -q ":$PORT "; then echo "port $PORT is in use: set LADDER_TEST_PORT"; finish 1; fi
      # The local server and the opponents, each in a process group of its own, stopped on the way out.
      setsid nice -n 15 "$PY" "$HERE/start_servers.py" --dir "$HOME/dev/ladder/ps" --ports "$PORT" --log-dir "$DIR/srv" \
        > "$DIR/srv.out" 2>&1 < /dev/null &
      SRV=$!
      trap 'kill -- -$SRV ${OPP:+-$OPP} 2>/dev/null' EXIT
      for _ in $(seq 1 90); do ss -ltn | grep -q ":$PORT " && break; sleep 1; done
      ss -ltn | grep -q ":$PORT " || { echo "the local server did not start: $DIR/srv.out"; finish 1; }
      setsid nice -n 15 "$PY" bench/ladder_opp.py --port "$PORT" --games $((G + 6)) --parallel 3 --prefix "heurt$(date +%H%M%S)" \
        > "$DIR/opp.out" 2>&1 < /dev/null &
      OPP=$!
      echo "test: local server pgid $SRV, SimpleHeuristics opponents pgid $OPP"
    fi
    echo "$(date -u +%FT%TZ) start: $U, $G games, $(basename "$CKPT"), $SERVER" >> "$DIR/bot.log"
    nice -n "$NICE" "$PY" ladder.py --server "$SERVER" --user "$U" --games "$G" --player mix --mix-rule veto \
      --search-ms 500 --helpers 4 --parallel 3 --deadline 2.5 --timer --device cpu --checkpoint "$CKPT" \
      --out "$RESULTS" --stop-file "$STOP" "${EXTRA[@]}" 2>&1 | tee -a "$DIR/bot.log"
    CODE=${PIPESTATUS[0]}
    echo "$(date -u +%FT%TZ) end: exit $CODE" >> "$DIR/bot.log"
    if [ "$MODE" = test ]; then kill -- -"$SRV" -"$OPP" 2>/dev/null; trap - EXIT; fi
    finish "$CODE"
    ;;

  stop)
    PID=$(running)
    if [ -z "$PID" ]; then
      echo "The bot is not running$("${T[@]}" has-session -t "=$SESSION" 2>/dev/null && echo " (its window is still open: Net Play shows it)")."
      exit 0
    fi
    touch "$STOP"
    echo "Stop file written: the bot starts no new games, finishes the ones in progress (a few minutes) and stops."
    ;;

  elo)
    cd "$HERE" || exit 1
    nice -n 10 "$PY" ladder.py --graph --once --server "$SERVER" --out "$RESULTS" >&2 || exit 1
    [ -f "$DIR/elo.html" ] || { echo "no graph: no games recorded yet" >&2; exit 1; }
    cat "$DIR/elo.html"
    ;;

  status)
    PID=$(running)
    echo "bot: $([ -n "$PID" ] && echo "running (pid $PID)" || echo "not running"); tmux session: $("${T[@]}" has-session -t "=$SESSION" 2>/dev/null && echo open || echo none)"
    tail -n 12 "$DIR/bot.log" 2>/dev/null
    ;;

  *)
    sed -n 2,12p "$0"; exit 2 ;;
esac
