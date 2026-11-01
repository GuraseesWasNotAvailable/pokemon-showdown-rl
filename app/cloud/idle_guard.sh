#!/usr/bin/env bash
# The cloud box stops itself when nothing uses it. Look Play Remote (cloud/remote_search.sh) starts
# this on the box right before ladder.py, in the background:
#
#     bash cloud/idle_guard.sh [minutes]          (default 30)
#
# The launcher asks at the end whether to stop the box. When that question never comes (the window
# closed, the laptop's lid shut, the laptop off), this stops it instead: once a minute it looks for
# anything using the box, and after <minutes> in a row with nothing it runs `sudo shutdown -h now`.
# The instance's shutdown behaviour is "stop": the box is stopped, not terminated, and its disk is kept.
#
# In use: a search helper of a laptop's ladder.py (--tag fpr.), a ladder.py, psppo-train active or
# starting, a train.py, anyone logged in with a terminal (who), or a 1-minute load above 2 (any other
# work). One guard at a time (~/.cache/psbot-idle-guard.pid, its lines in ~/.cache/psbot-idle-guard.log).
# The launcher stops it (kill <pid>) when the owner keeps the box running.
# IDLE_GUARD_DRY=1 writes "would stop the box" instead; IDLE_GUARD_STEP_S sets the look interval (60 s).
set -u
MIN=${1:-30}
STEP=${IDLE_GUARD_STEP_S:-60}
PIDFILE=$HOME/.cache/psbot-idle-guard.pid
mkdir -p "$HOME/.cache"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
  echo "$(date -u +%FT%TZ) a guard runs already (pid $(cat "$PIDFILE"))"
  exit 0
fi
echo $$ > "$PIDFILE"

busy() {
  pgrep -f -- '[-]-tag fpr[.]' >/dev/null && { echo "search helpers"; return 0; }
  pgrep -f '[l]adder[.]py' >/dev/null && { echo "ladder.py"; return 0; }
  pgrep -f '[t]rain[.]py' >/dev/null && { echo "train.py"; return 0; }
  case "$(systemctl is-active psppo-train 2>/dev/null)" in active|activating|reloading) echo "psppo-train"; return 0 ;; esac
  [ "$(who | grep -c .)" -gt 0 ] && { echo "a login"; return 0; }
  awk '{ exit !($1 > 2.0) }' /proc/loadavg && { echo "load $(cut -d' ' -f1 /proc/loadavg)"; return 0; }
  return 1
}

echo "$(date -u +%FT%TZ) guard $$: the box stops after $MIN minutes in a row with nothing running on it"
idle=0
while :; do
  if why=$(busy); then
    [ "$idle" -gt 0 ] && echo "$(date -u +%FT%TZ) in use again ($why)"
    idle=0
  else
    idle=$((idle + 1))
  fi
  if [ "$idle" -ge "$MIN" ]; then
    if [ -n "${IDLE_GUARD_DRY:-}" ]; then
      echo "$(date -u +%FT%TZ) idle for $MIN looks in a row: would stop the box (dry run)"
      exit 0
    fi
    echo "$(date -u +%FT%TZ) idle for $MIN minutes: stopping the box"
    sudo -n shutdown -h now
    exit 0
  fi
  sleep "$STEP"
done
