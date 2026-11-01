#!/usr/bin/env bash
# From the laptop (Git Bash): the ladder A/B on the cloud box, ab3-net-mix-deep-024381-box.
#
#   bash cloud/box_ladder.sh start    into the box's tmux session "ladder": the first time it starts
#                                     ~/dev/ab-ladder/run_box_ab.sh there, which asks for the Showdown
#                                     password (ladder.py's own prompt); later it attaches again.
#                                     Leave with Ctrl+B then D, or close the window: the A/B plays on.
#   bash cloud/box_ladder.sh status   running or not, the last lines it printed, and bench/ab_report.py
#                                     on the box's results
#   bash cloud/box_ladder.sh stop     the stop file: no new games, the ones in progress are finished
#   bash cloud/box_ladder.sh pull     results.jsonl, the battles and the console logs to
#                                     runs/ladder-box/ next to this script's repository (nothing is deleted)
#
# The box's address and SSH options are vm.sh's (C:/Users/you/ps-ppo/cloud/vm.sh, its current
# target): the address is looked up from AWS every time. When SSH stops answering after a network
# change: bash C:/Users/you/ps-ppo/cloud/vm.sh allow-ip.
# Never run start while the laptop plays the same account (Look Play, the PSBot Lookahead app): start
# refuses while one of them runs on this laptop, and status warns when one appears during the run.
#
# A dry run against a LOCAL server on the box (local names, no password): AB_SERVER, AB_USER, AB_GAMES
# and AB_TAG set here are handed to run_box_ab.sh (start), and AB_TAG to the report (status).
# BOX_LADDER_TTY=-tt forces a terminal for start when it is run from a script.
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
VMDIR=${PSPPO_VMDIR:-C:/Users/you/ps-ppo/cloud}
VM="$VMDIR/vm.sh"
TARGET=${PSPPO_TARGET:-$(cat "$VMDIR/.state/current" 2>/dev/null || echo a)}
if [ -f "$VMDIR/.state/targets/$TARGET.env" ]; then . "$VMDIR/.state/targets/$TARGET.env"; fi
KEY=${PSPPO_KEY:-$HOME/.ssh/psppo-key.pem}
SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$HOME/.ssh/psppo_known_hosts"
          -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -o ConnectTimeout=15 -o LogLevel=ERROR)

REMOTE=dev/ab-ladder                      # under the box's home
TAG=${AB_TAG:-ab3-net-mix-deep-024381-box}
SESSION=ladder
LOCAL=$HERE/runs/ladder-box

address() { bash "$VM" ip; }

# The bots that can play the account from THIS laptop: a ladder.py that plays (Look Play, Look Play
# AB, the PSBot Lookahead app's child; not --graph or --history, not a local server) and the PSBot
# Lookahead app itself. Their account lock is per computer and cannot see the box, so start
# refuses while one runs here and status warns. Prints "pid name; pid name", or nothing. When the
# question itself fails (no PowerShell), prints "?".
laptop_bots() {
  local q="@(Get-CimInstance Win32_Process | Where-Object { (\$_.Name -like 'python*' -and \$_.CommandLine -match 'ladder\.py' -and \$_.CommandLine -notmatch '--graph|--history|--server[ =]+local') -or \$_.Name -like 'PSBot Lookahead*' } | ForEach-Object { '{0} {1}' -f \$_.ProcessId, \$_.Name }) -join '; '"
  local out
  out=$(MSYS_NO_PATHCONV=1 powershell.exe -NoProfile -NonInteractive -Command "$q" 2>/dev/null) || { echo "?"; return; }
  printf '%s' "$out" | tr -d '\r\n'
}

cmd=${1:-}
case "$cmd" in
  start)
    if [ "${AB_SERVER:-public}" = public ]; then
      BOTS=$(laptop_bots)
      if [ "$BOTS" = "?" ]; then
        echo "(could not look for bots on this laptop; make sure no Look Play window and no PSBot Lookahead app is open)"
      elif [ -n "$BOTS" ]; then
        echo "A bot is running on this laptop: $BOTS"
        echo "Stop it first (Look Stop, or Stop in the PSBot Lookahead app, then close the app), then start again."
        echo "Two bots on one account answer the same games, and the box's A/B would get the laptop's moves."
        exit 1
      fi
      echo "While the box plays: no Look Play, no PSBot Lookahead app and no PSBot in the browser on this account."
    fi
    A=$(address) || exit 1
    ENVS=""
    for v in AB_SERVER AB_USER AB_GAMES AB_TAG; do
      val=${!v:-}
      case "$val" in *[!A-Za-z0-9:._-]*) echo "$v: letters, digits and :._- only"; exit 2 ;; esac
      if [ -n "$val" ]; then ENVS="$ENVS $v=$val"; fi
    done
    # tmux new -A: attach when the session is there, else make it with the A/B in it.
    exec ssh "${BOX_LADDER_TTY:--t}" "${SSH_OPTS[@]}" "ubuntu@$A" \
      "if ! infocmp \"\${TERM:-dumb}\" >/dev/null 2>&1; then export TERM=xterm-256color; fi;
       cd ~/$REMOTE || exit 1;
       exec tmux -u new -A -s $SESSION -x 180 -y 50 'env$ENVS bash ~/$REMOTE/run_box_ab.sh'"
    ;;
  status)
    BOTS=$(laptop_bots)
    if [ -n "$BOTS" ] && [ "$BOTS" != "?" ]; then
      echo "WARNING: a bot is running on this laptop too: $BOTS"
      echo "If the box is playing (below), stop the laptop's bot now: Look Stop, or Stop in the PSBot Lookahead app."
    fi
    A=$(address) || exit 1
    ssh "${SSH_OPTS[@]}" "ubuntu@$A" "cd ~/$REMOTE || exit 1
      PID=\$(pgrep -u \$(id -u) -f \"ladder.py .*--out \$HOME/$REMOTE/runs/ladder/results.jsonl\" | head -1)
      echo \"A/B $TAG: \$([ -n \"\$PID\" ] && echo \"running (pid \$PID)\" || echo 'not running'); tmux session $SESSION: \$(tmux has-session -t =$SESSION 2>/dev/null && echo open || echo none)\"
      LOG=\$(ls -t runs/ladder/console-*.log 2>/dev/null | head -1)
      if [ -n \"\$LOG\" ]; then echo \"--- the last lines of \$LOG\"; tail -n 15 \"\$LOG\"; fi
      echo '---'
      if [ -f runs/ladder/results.jsonl ]; then
        ~/venv/bin/python bench/ab_report.py --results runs/ladder/results.jsonl --experiment $TAG --mix-iterations-ref 538000
      else
        echo 'no games recorded yet'
      fi"
    ;;
  stop)
    A=$(address) || exit 1
    ssh "${SSH_OPTS[@]}" "ubuntu@$A" "touch ~/$REMOTE/ladder.stop && echo 'stop file written: no new games; the games in progress are finished, then it stops (a few minutes)'"
    ;;
  pull)
    A=$(address) || exit 1
    mkdir -p "$LOCAL" || exit 1
    ssh "${SSH_OPTS[@]}" "ubuntu@$A" "cd ~/$REMOTE/runs/ladder 2>/dev/null || exit 3
      tar -czf - \$(ls -d results.jsonl battles console-*.log run_box_ab.log 2>/dev/null)" | tar -xzf - -C "$LOCAL"
    CODES=("${PIPESTATUS[@]}")
    if [ "${CODES[0]}" = 3 ]; then echo "no results on the box yet"; exit 1; fi
    if [ "${CODES[0]}" != 0 ] || [ "${CODES[1]}" != 0 ]; then echo "the copy failed (ssh ${CODES[0]}, tar ${CODES[1]})"; exit 1; fi
    echo "copied to $LOCAL: $(wc -l < "$LOCAL/results.jsonl" 2>/dev/null || echo 0) result lines, $(ls "$LOCAL/battles" 2>/dev/null | wc -l) battle files"
    ;;
  *)
    sed -n 2,22p "$0"; exit 2 ;;
esac
