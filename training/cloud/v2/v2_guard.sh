#!/bin/bash
# The v2 waves' supervisor on one box (unit psppo-v2-guard). Every 5 minutes:
#  1. CHAIN pairs "first:next": when arm `first` is finished (state.json status done and eval/final.done
#     written, or failed), it starts arm `next`'s trainer once, on the GPU `first` used (its unit says
#     which), and its evaluator 3.5 h later. So no GPU waits for a whole batch, and no person is needed.
#  2. Evaluators: an arm whose trainer has run 3.5 h and whose evaluator is not running gets it started
#     (this is what brings them back after a reboot or a spot interruption).
#  3. The box counts as idle when every arm in ARMS is done or failed AND no trainer, one-off evaluation,
#     gauntlet or test runs (the evaluators' --watch loops do not count). After IDLE_MIN idle minutes in
#     a row it shuts the box down (a stop: the disk and everything on it stay).
# Settings come from ~/runs/v2/guard.env (ARMS, CHAIN, IDLE_MIN). Touch ~/dev/KEEP_BOX to hold off the
# shutdown; touch ~/runs/v2/NO_CHAIN to keep new arms from starting.
IDLE_MIN=120
ARMS=""
CHAIN=""
[ -f ~/runs/v2/guard.env ] && . ~/runs/v2/guard.env
idle=0
log() { echo "$(date -u +%FT%TZ) $*"; }
status() { python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('status',''))" "$HOME/runs/v2/$1/state.json" 2>/dev/null; }
finished() { local st; st=$(status "$1"); [ "$st" = failed ] || { [ "$st" = done ] && [ -e "$HOME/runs/v2/$1/eval/final.done" ]; }; }
while true; do
  [ -f ~/runs/v2/guard.env ] && . ~/runs/v2/guard.env
  for pair in $CHAIN; do
    first=${pair%%:*}; next=${pair##*:}
    if finished "$first" && [ ! -e ~/runs/v2/$next.started ] && [ ! -e ~/runs/v2/NO_CHAIN ]; then
      log "$first is done: starting $next"
      sudo systemctl enable --now psppo-v2-$next-train
      date -u +%FT%TZ > ~/runs/v2/$next.started
    fi
  done
  for a in $ARMS; do
    first_log=$(ls ~/runs/v2/$a/logs/trainer_*.log 2>/dev/null | head -1)
    if [ -n "$first_log" ] && [ $(( $(date +%s) - $(stat -c %Y "$first_log") )) -ge 12600 ] \
       && ! systemctl is-active --quiet psppo-v2-$a-eval; then
      log "starting the evaluator of $a"
      sudo systemctl start psppo-v2-$a-eval
    fi
  done
  busy=""
  for a in $ARMS; do
    st=$(status $a)
    case "$st" in done|failed) ;; *) busy="$busy $a:${st:-none}";; esac
  done
  if pgrep -f "[t]rain.py|[g]auntlet.py|[p]ytest" >/dev/null; then busy="$busy procs"; fi
  if pgrep -af "[e]valuate.py" | grep -v -- "--watch" >/dev/null; then busy="$busy oneoff-eval"; fi
  if [ -e ~/dev/KEEP_BOX ]; then busy="$busy keep-box"; fi
  if [ -n "$busy" ]; then
    [ $idle -gt 0 ] && log "busy again:$busy"
    idle=0
  else
    idle=$((idle + 5))
    log "idle for $idle of $IDLE_MIN minutes"
    if [ $idle -ge $IDLE_MIN ]; then
      log "shutting down"
      sudo shutdown -h now
      exit 0
    fi
  fi
  sleep 300
done
