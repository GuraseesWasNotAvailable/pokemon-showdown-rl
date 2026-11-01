#!/usr/bin/env bash
# The pre-registered three-arm ladder A/B on the cloud box: net, mix and mix-deep, one per game,
# experiment ab3-net-mix-deep-024381-box (the laptop's 30 games of ab3-net-mix-deep-024381 stay a
# pilot). The rules are bench/ab_report.py's: 500 games per arm, one interim look at 250, the
# cheapest arm within 1 point of the highest raw win rate, significant at |z| >= 2.
#
#   bash ~/dev/ab-ladder/run_box_ab.sh        (cloud/box_ladder.sh start runs it inside tmux)
#
# The compute per decision is the laptop's, measured in search iterations on the laptop's own
# ladder positions (bench/ab_box_calibrate.py: the laptop's 22 mix and 10 mix-deep games on 024381
# replayed here, each decision put next to the iterations the laptop logged for it; 2026-10-09):
#   mix       500 ms per world on 1x the worlds, one core, as on the laptop. Here / laptop on the
#             same decisions: 1.02 with 6 mix searches back to back; 1.05 and 1.04 at the ladder's
#             pace (the laptop's own gaps between turns) with 2 deep games beside it, and with 4;
#             0.94 with 3 deep games searching back to back (the worst case). So the laptop's
#             538,000 per decision, within 10%.
#   mix-deep  1425 ms per world on twice the worlds, the worlds searched 4 at a time (--deep-workers
#             4: Foul Play's own pool, forked from the helper, one core each). Here / laptop's
#             mix-deep (1250 ms, one after the other): 1.06 and 1.02 at the ladder's pace, 1.005
#             back to back; about 5 times mix (4.9 to 5.4 on those numbers). More than 1250 ms
#             because a freshly forked worker searches about 12% slower than the helper itself.
#             A decision takes 1.76 s on average, 1.82 s at the 95th, 1.85 s at the slowest (the
#             laptop: about 6 s), well inside the 9 s fallback and the 5 s deadline.
# 6 games at a time (Showdown's own cap per account: a search with more than 5 games in
# progress is refused), 7 helpers, the network on the GPU.
#
# The password: ladder.py asks for it itself (getpass: nothing shown, nothing stored, not on any
# command line, not in any file, environment variable or log). PS_PASSWORD is removed first.
# The login is told one step a line, and a failed one in one line with its reason (the login
# server's, "Wrong password." say; a web page in its place; Showdown's own answer). When ladder.py
# ends, one line says why (its --exit-file, nothing secret in it), on a login failure with "run
# box_ladder.sh start again", and the window stays open until Enter.
#
# For a dry run against a LOCAL server, with local names and no password (nothing is sent to the
# public server or its login server), the same settings:
#   AB_SERVER=local:10850 AB_USER=abdry1 AB_GAMES=8 AB_DIR=~/dev/ab-ladder-dry AB_TAG=ab3-dry bash run_box_ab.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE" || exit 1

# The settings (bench/ab_box_calibrate.py and bench/ab_deep_measure.py on this box, 2026-10-09).
PARALLEL=6
HELPERS=7
SEARCH_MS=500
DEADLINE=2.5
DEEP_MS=1425
DEEP_WORLDS=2
DEEP_WORKERS=4
DEEP_DEADLINE=5.0

TAG=${AB_TAG:-ab3-net-mix-deep-024381-box}
SERVER=${AB_SERVER:-public}
ACCOUNT=${AB_USER:-YourShowdownName}
GAMES=${AB_GAMES:-1500}
DIR=${AB_DIR:-$HERE/runs/ladder}
STOP=${AB_STOP:-$HERE/ladder.stop}
CKPT=$HOME/dev/ab-ladder-ckpt/learner_update_024381.pt
PY=$HOME/venv/bin/python
RESULTS=$DIR/results.jsonl
EXITF=$DIR/last_exit.txt

unset PS_PASSWORD
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONWARNINGS="ignore:flex_attention called without torch.compile"

finish() {
  echo
  echo "The A/B is not running now."
  read -r -t 86400 -p "Press Enter to close this window. " _ || true
  exit "${1:-0}"
}

# Why ladder.py ended, in one line: its --exit-file ("KIND: why", nothing secret), else its exit code.
# A login failure says to run box_ladder.sh start again (after Enter here: start attaches to this
# window while it is open).
exit_line() {
  local code=$1 note kind why
  note=$(head -n 1 "$EXITF" 2>/dev/null | head -c 600)
  kind=${note%%: *}
  why=${note#*: }
  why=${why%.}
  if [ -z "$note" ]; then
    kind=""
    case "$code" in
      0) why="it finished" ;;
      2) kind=login; why="the login did not go through (the reason is above)" ;;
      3) why="another bot is playing this account" ;;
      4) why="it stopped without a connection to the server (the reason is above)" ;;
      130) why="it was stopped with Ctrl+C" ;;
      *) why="it stopped with an error (exit $code, the reason is above)" ;;
    esac
  fi
  case "$kind" in
    password) echo "ladder.py ended (exit $code): $why. Press Enter, then run bash cloud/box_ladder.sh start again on the laptop and type the password carefully." ;;
    login)    echo "ladder.py ended (exit $code): $why. Press Enter, then run bash cloud/box_ladder.sh start again on the laptop." ;;
    *)        echo "ladder.py ended (exit $code): $why." ;;
  esac
}

mkdir -p "$DIR"
if [ ! -f "$CKPT" ]; then echo "no network: $CKPT is missing"; finish 1; fi
if pgrep -u "$(id -u)" -f "ladder.py .*--out $RESULTS" > /dev/null; then
  echo "a ladder.py with these results is already running (pid $(pgrep -u "$(id -u)" -f "ladder.py .*--out $RESULTS" | head -1))"
  finish 1
fi
# An old stop file must not end this run at once.
if [ -e "$STOP" ]; then mv "$STOP" "$STOP.stale.$(date -u +%Y%m%dT%H%M%S)"; fi

echo "PSBot ladder A/B on the box: net, mix and mix-deep, one per game, experiment $TAG"
echo "server $SERVER, account $ACCOUNT, up to $GAMES games, $PARALLEL at a time, network $(basename "$CKPT") on the GPU"
echo "mix $SEARCH_MS ms per world; mix-deep $DEEP_MS ms per world on ${DEEP_WORLDS}x the worlds, $DEEP_WORKERS at a time, answer within $DEEP_DEADLINE s"
echo "results: $RESULTS"
echo
if [ "$SERVER" = public ]; then
  echo "REMINDER: the laptop must NOT play $ACCOUNT while this runs: no Look Play, no PSBot Lookahead app,"
  echo "no ladder games in the browser on this account. Two bots on one account answer the same games."
  echo
fi
echo "Leave this window with Ctrl+B then D, or just close it: the A/B keeps playing."
echo "To stop: bash cloud/box_ladder.sh stop on the laptop (no new games; the ones in progress are finished)."
echo "Ctrl+C here abandons the games in progress."
echo
if [ "$SERVER" = public ]; then
  echo "In a few seconds ladder.py asks: Showdown password for $ACCOUNT. Type it only after that line."
  echo "Nothing you type is shown, before the prompt or at it (keys typed before it are thrown away)."
  echo
fi

# No echo from here on: keys typed before ladder.py's getpass prompt would otherwise be shown and
# stay in tmux's scrollback (getpass throws them away, but only after they were echoed; measured
# on the box). getpass itself works the same with echo off. Echo comes back when ladder.py ends.
TTY_ECHO_OFF=0
if [ -t 0 ] && stty -echo 2>/dev/null; then TTY_ECHO_OFF=1; fi

echo "$(date -u +%FT%TZ) start: $ACCOUNT, $GAMES games, $TAG, $SERVER" >> "$DIR/run_box_ab.log"
: > "$EXITF"     # this run's reason only
"$PY" ladder.py --server "$SERVER" --user "$ACCOUNT" --games "$GAMES" --exit-file "$EXITF" \
  --arms net,mix,mix-deep --experiment "$TAG" --predictions --mix-rule veto \
  --parallel "$PARALLEL" --helpers "$HELPERS" --search-ms "$SEARCH_MS" --deadline "$DEADLINE" \
  --deep-search-ms "$DEEP_MS" --deep-worlds "$DEEP_WORLDS" --deep-workers "$DEEP_WORKERS" --deep-deadline "$DEEP_DEADLINE" \
  --timer --device cuda --checkpoint "$CKPT" --out "$RESULTS" --stop-file "$STOP"
CODE=$?
if [ "$TTY_ECHO_OFF" = 1 ]; then stty echo 2>/dev/null; fi
LINE=$(exit_line "$CODE")
NOTE=$(head -n 1 "$EXITF" 2>/dev/null | head -c 600)
echo "$(date -u +%FT%TZ) end: exit $CODE${NOTE:+: $NOTE}" >> "$DIR/run_box_ab.log"
echo
echo "The A/B so far (all of this experiment's games on this box):"
"$PY" bench/ab_report.py --results "$RESULTS" --experiment "$TAG" --mix-iterations-ref 538000
echo
echo "$LINE"
finish "$CODE"
