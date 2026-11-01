#!/usr/bin/env bash
# The ladder A/B with this laptop playing and the cloud box searching (option B).
#
#   bash cloud/remote_search.sh        (launchers/laptop/Look Play Remote.cmd runs it in C:\Users\you\ps-ppo-predict)
#
# ladder.py runs HERE: Showdown sees this laptop's own home connection (a datacenter address is
# locked by Showdown as a proxy, so the box never connects to Showdown), and the network runs on
# this laptop's GPU. Only Foul Play's search helpers run on the box, each one started over ssh by
# ladder.py itself (--search-remote, foulplay_bridge.RemoteTarget). Nothing is routed through the
# box: it only answers "what would the search play here". The laptop holds the connection and the
# network; all the search CPU is the box's. 6 games at a time (Showdown's cap per account), 7
# helpers, the box's calibrated settings: mix 500 ms per world, mix-deep 1425 ms per world on twice
# the worlds, 4 worlds at a time, answer within 5 s (run_box_ab.sh says how they were measured).
#
# In order: no other bot on this laptop or on the box (one bot per account), a memory check, the
# box started when it is stopped (no capacity: tried again every minute for up to 15 minutes, then
# play on the laptop with its own WSL helpers as before, or quit), SSH waited for (vm.sh allow-ip
# when this laptop's address is not let in), no training on the box (psppo-train resumes an
# unfinished run by itself at boot; asked, default no), this repository copied to ~/dev/ab-ladder on the box
# when the box's copy is not this commit, the round trip measured and one helper tried there
# (foulplay_bridge.py remote-check), ladder.py (it asks for the Showdown password itself, as
# always), bench/ab_report.py for the experiment, and at the end the question whether to stop the
# box (it costs about $7.58 an hour while it runs): stopped unless n is typed within 60 s.
#
# Optional settings: PSPPO_VMDIR (cloud/ with its .state; default C:/Users/you/ps-ppo/cloud),
# PSPPO_PY (the laptop's python), AB_CKPT (the network), AB_GAMES (skips the question),
# REMOTE_SEARCH_DRY=1 (every step up to ladder.py, whose command line is printed instead of run;
# no username is asked, and the box is left as it is).
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VMDIR=${PSPPO_VMDIR:-C:/Users/you/ps-ppo/cloud}
VM="$VMDIR/vm.sh"
AWS2="$VMDIR/.state/aws2.sh"
PY=${PSPPO_PY:-C:/Users/you/ps-ppo/.venv/Scripts/python.exe}
CKPT=${AB_CKPT:-C:/Users/you/ps-ppo/runs/vm/checkpoints/learner_update_024381.pt}
TAG=ab3-net-mix-deep-024381
REMOTE_DIR="~/dev/ab-ladder"          # given to ladder.py as it is: expanded on the box
DRY=${REMOTE_SEARCH_DRY:-0}
CAPACITY_WAIT_S=${REMOTE_SEARCH_CAPACITY_WAIT_S:-900}   # InsufficientInstanceCapacity: tried again for this long,
RETRY_S=${REMOTE_SEARCH_RETRY_S:-60}                    # this often (the tests give shorter ones)
export PYTHONUTF8=1
# Git Bash rewrites a word that looks like a /path for a Windows program: the python calls below say
# MSYS_NO_PATHCONV=1 (their arguments are Windows paths and ~/... already), git and tar keep it.

say() { echo "$(date +%H:%M:%S) $*"; }
aws2() { MSYS_NO_PATHCONV=1 bash "$AWS2" "$@"; }

finish() {
  echo
  read -r -p "Press Enter to close this window. " _ || true
  exit "${1:-0}"
}

# The bots that can play the account from THIS laptop (as cloud/box_ladder.sh looks for them): a
# ladder.py that plays (not --graph or --history, not a local server) and the PSBot Lookahead app.
laptop_bots() {
  local q="@(Get-CimInstance Win32_Process | Where-Object { (\$_.Name -like 'python*' -and \$_.CommandLine -match 'ladder\.py' -and \$_.CommandLine -notmatch '--graph|--history|--server[ =]+local') -or \$_.Name -like 'PSBot Lookahead*' } | ForEach-Object { '{0} {1}' -f \$_.ProcessId, \$_.Name }) -join '; '"
  local out
  out=$(powershell.exe -NoProfile -NonInteractive -Command "$q" 2>/dev/null) || { echo "?"; return; }
  printf '%s' "$out" | tr -d '\r\n'
}

free_gb() {
  powershell.exe -NoProfile -NonInteractive -Command "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" 2>/dev/null | tr -d '\r\n '
}

# The box vm.sh talks to: "id type state address zone".
box_line() { bash "$VM" info 2>/dev/null | head -n 1; }

start_box() {
  # 0: started (pending or running); 1: no capacity for CAPACITY_WAIT_S; 2: another error.
  local until=$((SECONDS + CAPACITY_WAIT_S)) out
  while :; do
    out=$(aws2 ec2 start-instances --region "$REGION" --instance-ids "$ID" --query "StartingInstances[0].CurrentState.Name" --output text 2>&1 | tail -n 1 | tr -d '\r')
    case "$out" in
      pending|running) STARTED_HERE=1; say "the box is starting ($out)"; return 0 ;;
      *InsufficientInstanceCapacity*) ;;
      *) say "the box did not start: $out"; return 2 ;;
    esac
    if [ "$SECONDS" -ge "$until" ]; then return 1; fi
    say "AWS has no capacity for the box right now (InsufficientInstanceCapacity); trying again in $RETRY_S s (for up to $((CAPACITY_WAIT_S / 60)) minutes in all)"
    sleep "$RETRY_S"
    [ "$INTERRUPTED" = 1 ] && return 1
  done
}

stop_box() {
  local out
  out=$(aws2 ec2 stop-instances --region "$REGION" --instance-ids "$ID" --query "StoppingInstances[0].CurrentState.Name" --output text 2>&1 | tail -n 1 | tr -d '\r')
  case "$out" in
    stopping|stopped) say "the box is stopping ($out): nothing more is paid for it but its disk" ;;
    *) say "the box did NOT stop: $out"; echo "Stop it by hand: bash $AWS2 ec2 stop-instances --region $REGION --instance-ids $ID" ;;
  esac
}

# ssh to the box with vm.sh's own settings (its key and known hosts, read once from `vm.sh ssh -G`:
# ssh's resolved settings, file names only), without asking AWS for the address every time.
bssh() {
  ssh -i "$KEY" -o IdentitiesOnly=yes -o UserKnownHostsFile="$KH" -o StrictHostKeyChecking=accept-new -o BatchMode=yes \
      -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o LogLevel=ERROR "$HOST" "$@"
}

# The box stops itself after IDLE_GUARD_MIN (30) minutes with nothing running on it (cloud/idle_guard.sh),
# for when the question at the end never comes: this window closed early, the laptop's lid shut.
# Stopped again when the owner keeps the box running.
start_guard() {
  if bssh "mkdir -p ~/.cache && setsid nohup bash ~/dev/ab-ladder/cloud/idle_guard.sh ${IDLE_GUARD_MIN:-30} >> ~/.cache/psbot-idle-guard.log 2>&1 < /dev/null &" >/dev/null 2>&1; then
    say "the box stops itself after ${IDLE_GUARD_MIN:-30} minutes with nothing running on it, should this window close early"
  else
    echo "could not start the box's idle guard: keep this window open until the last question"
  fi
}
stop_guard() { bssh 'kill $(cat ~/.cache/psbot-idle-guard.pid 2>/dev/null) 2>/dev/null' >/dev/null 2>&1 || true; }

# A ladder.py playing on the box itself (cloud/box_ladder.sh's A/B): two bots on one account.
remote_ladders() { bssh "pgrep -af '[l]adder[.]py' | grep -v -e '--server local' -e '--server=local' | grep -c ." 2>/dev/null | tr -d '\r\n '; }

# Training on the box: psppo-train (enabled at boot: an unfinished run resumes by itself when the box
# starts; "activating" while it starts or waits to restart) or a train.py. Empty when none.
box_training() {
  bssh "case \$(systemctl is-active psppo-train 2>/dev/null) in active|activating|reloading) echo psppo-train;; esac; pgrep -f '[t]rain[.]py --mode' >/dev/null && echo train.py; true" 2>/dev/null | tr '\r\n' '  ' | sed 's/ *$//'
}

report() {
  echo
  echo "The A/B so far (all of this experiment's games):"
  MSYS_NO_PATHCONV=1 "$PY" bench/ab_report.py --experiment "$TAG" --mix-iterations-ref 538000
}

ask_account() {
  if [ "$DRY" = 1 ]; then PS_USERNAME=${PS_USERNAME:-dry-run}; GAMES=${AB_GAMES:-30}; return; fi
  read -r -p "Showdown username: " PS_USERNAME
  export PS_USERNAME
  GAMES=${AB_GAMES:-}
  if [ -z "$GAMES" ]; then read -r -p "How many games (Enter for 30): " GAMES; fi
  GAMES=${GAMES:-30}
  case "$GAMES" in *[!0-9]*|"") echo "not a number of games: $GAMES"; finish 2 ;; esac
}

# The way it was before: everything on this laptop, the search in its own WSL (Look Play AB.cmd's settings).
play_on_laptop() {
  echo
  echo "Playing on this laptop with its own WSL helpers, as before: 3 games at a time on 4 helpers,"
  echo "mix-deep 1250 ms per world within 7.5 s. The laptop works hard for the whole run (about 8 GB, 4 cores)."
  ask_account
  unset PS_PASSWORD
  cd "$REPO" || finish 1
  if [ "$DRY" = 1 ]; then
    echo "dry run: \"$PY\" ladder.py --games $GAMES --arms net,mix,mix-deep --experiment $TAG --predictions --parallel 3 --helpers 4 --search-ms 500 --deadline 2.5 --deep-search-ms 1250 --deep-worlds 2 --deep-deadline 7.5 --checkpoint $CKPT"
    finish 0
  fi
  MSYS_NO_PATHCONV=1 "$PY" ladder.py --games "$GAMES" --arms net,mix,mix-deep --experiment "$TAG" --predictions --parallel 3 --helpers 4 --search-ms 500 --deadline 2.5 --deep-search-ms 1250 --deep-worlds 2 --deep-deadline 7.5 --checkpoint "$CKPT"
  echo "ladder.py ended (exit $?)."
  report
  finish 0
}

# Ctrl+C: ladder.py stops by itself (a game in progress is lost on time; Look Stop finishes them
# instead), and this script goes on to the report and the question about the box. While the box
# is being started or waited for, Ctrl+C ends this script (the box question is still asked).
INTERRUPTED=0
trap 'INTERRUPTED=1; echo' INT

# ---- the steps ------------------------------------------------------------------------------
cd "$REPO" || finish 1
echo "PSBot ladder A/B, experiment $TAG: this laptop plays, the cloud box searches"
echo "network: $(basename "$CKPT") on this laptop's GPU; search: Foul Play on the box, over ssh"
echo "Keep this window open to the end: it asks whether to stop the box (closing it earlier leaves the box"
echo "running and paid for). Look Stop ends the run cleanly: no new games, the ones in progress are finished."
echo
if [ ! -f "$PY" ]; then echo "no python at $PY (set PSPPO_PY)"; finish 1; fi
if [ ! -f "$CKPT" ]; then echo "no network at $CKPT (set AB_CKPT)"; finish 1; fi
if [ ! -f "$VM" ] || [ ! -f "$AWS2" ]; then echo "the box's settings are not at $VMDIR (vm.sh and .state/aws2.sh; set PSPPO_VMDIR)"; finish 1; fi

# 1. One bot per account: nothing else plays it from this laptop (ladder.py's own lock checks again).
BOTS=$(laptop_bots)
if [ "$BOTS" = "?" ]; then
  echo "(could not look for other bots on this laptop: make sure no Look Play window and no PSBot Lookahead app is open)"
elif [ -n "$BOTS" ]; then
  echo "A bot is running on this laptop: $BOTS"
  echo "Stop it first (Look Stop, or Stop in the PSBot Lookahead app), then start this again."
  echo "Two bots on one account answer the same games, and the A/B would mix two programs' games."
  finish 1
fi

# 2. Memory: the laptop holds the connection and the network only (no Foul Play here), so 4 GB is enough.
FREEGB=$(free_gb)
if [ -n "$FREEGB" ] && [ "$FREEGB" -lt 4 ] 2>/dev/null; then
  echo "Only $FREEGB GB of memory is free (FreePhysicalMemory). The run wants 4 GB: close what you can."
  read -r -p "Press Enter to go on anyway. " _ || true
elif [ -n "$FREEGB" ]; then
  echo "$FREEGB GB of memory free."
fi

# 3. The box: started when it is stopped.
LINE=$(box_line)
ID=$(echo "$LINE" | awk '{print $1}')
STATE=$(echo "$LINE" | awk '{print $3}')
ZONE=$(echo "$LINE" | awk '{print $5}')
REGION=${ZONE%?}
STARTED_HERE=0
if [ -z "$ID" ] || [ -z "$REGION" ]; then
  echo "vm.sh does not find the box ($LINE). Is $VMDIR/.state/current the right one?"
  finish 1
fi
say "the box $ID is $STATE"
if [ "$STATE" = stopping ]; then
  say "waiting for it to finish stopping before it is started again"
  for _ in $(seq 1 60); do
    sleep 10
    STATE=$(box_line | awk '{print $3}')
    [ "$STATE" = stopping ] || break
    [ "$INTERRUPTED" = 1 ] && finish 130
  done
fi
case "$STATE" in
  running|pending) ;;
  stopped)
    start_box
    case $? in
      0) ;;
      1)
        echo
        echo "AWS had no capacity for the box for $((CAPACITY_WAIT_S / 60)) minutes."
        echo "  W  play on this laptop with its own WSL helpers, as before (the laptop works hard)"
        echo "  Q  quit (start this again later)"
        read -r -p "W or Q: " CHOICE || CHOICE=q
        case "$CHOICE" in w|W) play_on_laptop ;; *) finish 1 ;; esac
        ;;
      *) finish 1 ;;
    esac
    ;;
  *) echo "the box is $STATE: cannot use it now"; finish 1 ;;
esac

# Its address (new at every start), then ssh's settings for it as vm.sh has them.
A=""
for _ in $(seq 1 60); do
  A=$(bash "$VM" ip 2>/dev/null | tr -d '\r') && [ -n "$A" ] && break
  [ "$INTERRUPTED" = 1 ] && break
  sleep 5
done
if [ -z "$A" ]; then echo "the box has no address after 5 minutes"; STEP_FAILED=1; fi
if [ -z "${STEP_FAILED:-}" ]; then
  CONF=$(bash "$VM" ssh -G 2>/dev/null | tr -d '\r')
  HOST="$(echo "$CONF" | awk '$1=="user"{print $2; exit}')@$(echo "$CONF" | awk '$1=="hostname"{print $2; exit}')"
  KEY=$(echo "$CONF" | awk '$1=="identityfile"{print $2; exit}')
  KH=$(echo "$CONF" | awk '$1=="userknownhostsfile"{print $2; exit}')
  KEY=$(cygpath -m "${KEY/#\~/$HOME}")          # C:/Users/... : the same file for this bash's ssh and for ladder.py's
  KH=$(cygpath -m "${KH/#\~/$HOME}")
  SSH=$(cygpath -m "$(command -v ssh)")         # Git's ssh, the one vm.sh uses
  say "the box is at $HOST"

  # 4. SSH: up within a minute or two of the start; when this laptop's address is not let in, vm.sh allow-ip.
  ALLOWED=0
  for i in $(seq 1 40); do
    if bssh true >/dev/null 2>&1; then break; fi
    [ "$INTERRUPTED" = 1 ] && break
    if [ "$i" = 5 ] && [ "$ALLOWED" = 0 ]; then
      say "no SSH yet: letting this laptop's address in (vm.sh allow-ip)"
      bash "$VM" allow-ip 2>&1 | tail -n 1
      ALLOWED=1
    fi
    sleep 5
  done
  if ! bssh true >/dev/null 2>&1; then echo "no SSH to the box after about 4 minutes"; STEP_FAILED=1; fi
fi

if [ -z "${STEP_FAILED:-}" ]; then
  # 5. One bot per account: no ladder.py playing on the box itself (cloud/box_ladder.sh's run).
  ON_BOX=$(remote_ladders)
  if [ -n "$ON_BOX" ] && [ "$ON_BOX" != 0 ]; then
    echo "A ladder.py is playing ON the box ($ON_BOX): stop it first (bash cloud/box_ladder.sh stop), then start this again."
    STEP_FAILED=1
  fi
fi

if [ -z "${STEP_FAILED:-}" ]; then
  # The A/B's compute: the box settings were measured on a box that was not training. A trainer
  # takes the cores the search needs, and mix and mix-deep would search less than calibrated.
  TRAINING=$(box_training)
  if [ -n "$TRAINING" ]; then
    echo "The box is training ($TRAINING): the search would share its cores, so the A/B's games would not"
    echo "have the calibrated compute (ab_report's COMPUTE warning). Stop the training first, or play anyway."
    ANSWER=""
    read -r -p "Play anyway? [y/N] " ANSWER || true
    case "$ANSWER" in y|Y|yes) ;; *) STEP_FAILED=1 ;; esac
  fi
fi

if [ -z "${STEP_FAILED:-}" ]; then
  # 6. The box's copy of this repository, when it is not this commit (files git knows; nothing is deleted there).
  HEAD=$(git -C "$REPO" rev-parse HEAD 2>/dev/null)
  DIRTY=$(git -C "$REPO" status --porcelain | wc -l | tr -d ' ')
  FILES=$(git -C "$REPO" ls-files -co --exclude-standard | wc -l | tr -d ' ')
  MARK="$HEAD +$DIRTY"
  THERE=$(bssh "cat ~/dev/ab-ladder/.synced_from 2>/dev/null" | tr -d '\r')
  if [ -z "$HEAD" ] || [ "$FILES" -lt 50 ] 2>/dev/null; then
    echo "git does not read the repository at $REPO ($FILES files): nothing copied"
    STEP_FAILED=1
  elif [ "$DIRTY" = 0 ] && [ "$THERE" = "$MARK" ]; then
    say "the box's copy (~/dev/ab-ladder) is this commit, ${HEAD:0:7}"
  else
    say "copying this repository (${HEAD:0:7}, $FILES files, $DIRTY changed) to ~/dev/ab-ladder on the box"
    # pipefail: a failed git or tar here fails the copy (and leaves the box's mark as it was).
    if ! (set -o pipefail; git -C "$REPO" ls-files -co --exclude-standard -z | (cd "$REPO" && tar --null -T - -czf -) \
        | bssh "mkdir -p ~/dev/ab-ladder && tar -xzf - -C ~/dev/ab-ladder && echo '$MARK' > ~/dev/ab-ladder/.synced_from"); then
      echo "the copy to the box failed"
      STEP_FAILED=1
    fi
  fi
fi

if [ -z "${STEP_FAILED:-}" ]; then
  # 7. The round trip, and one helper started there, probed and stopped (nothing left running).
  say "measuring the round trip and trying one helper on the box (about 15 s)"
  CHECK=$(MSYS_NO_PATHCONV=1 "$PY" foulplay_bridge.py remote-check --remote "$HOST" --key "$KEY" --known-hosts "$KH" --ssh "$SSH" --remote-dir "$REMOTE_DIR" --pings 10 2>&1)
  CHECK_CODE=$?
  echo "$CHECK" | grep -v '^rtt_ms=' | sed 's/^/  /'
  RTT=$(echo "$CHECK" | sed -n 's/^rtt_ms=\([0-9]*\).*/\1/p' | tail -n 1 | tr -d '\r')
  if [ "$CHECK_CODE" != 0 ] || [ -z "$RTT" ]; then
    echo "the search on the box is not usable (above)"
    STEP_FAILED=1
  elif [ "$RTT" -gt 400 ]; then
    echo "The round trip to the box is long ($RTT ms): every search has that much less of its time. ladder.py refuses"
    echo "when a deadline cannot hold it."
  fi
fi

if [ -n "${STEP_FAILED:-}" ] || [ "$INTERRUPTED" = 1 ]; then
  CODE=1
else
  # 8. ladder.py, here. It asks for the Showdown password itself (not shown, not stored).
  ask_account
  unset PS_PASSWORD
  echo
  echo "$GAMES games, 6 at a time; the search on $HOST (round trip $RTT ms). Look Stop finishes the games in progress and stops."
  if [ "$DRY" = 1 ]; then
    echo "dry run, not started: \"$PY\" ladder.py --games $GAMES --arms net,mix,mix-deep --experiment $TAG --predictions --parallel 6 --search-remote $HOST --search-key $KEY --search-known-hosts $KH --search-ssh \"$SSH\" --search-remote-dir $REMOTE_DIR --helpers 7 --search-ms 500 --deadline 2.5 --deep-search-ms 1425 --deep-worlds 2 --deep-workers 4 --deep-deadline 5.0 --checkpoint $CKPT"
    CODE=0
  else
    start_guard
    MSYS_NO_PATHCONV=1 "$PY" ladder.py --games "$GAMES" --arms net,mix,mix-deep --experiment "$TAG" --predictions --parallel 6 --search-remote "$HOST" --search-key "$KEY" --search-known-hosts "$KH" --search-ssh "$SSH" --search-remote-dir "$REMOTE_DIR" --helpers 7 --search-ms 500 --deadline 2.5 --deep-search-ms 1425 --deep-worlds 2 --deep-workers 4 --deep-deadline 5.0 --checkpoint "$CKPT"
    CODE=$?
    echo "ladder.py ended (exit $CODE)."
  fi
  # 9. The A/B so far.
  report
  # Whatever of this laptop's helpers is still running on the box (ladder.py stops them; this checks).
  LEFT=$(bssh "pgrep -fc '[f]p_bridge_server[.]py.*--tag fpr[.]'" 2>/dev/null | tr -d '\r\n ')
  if [ -n "$LEFT" ] && [ "$LEFT" != 0 ]; then
    echo "$LEFT search helper process(es) of a laptop run still on the box: they stop by themselves within 20 s, and with the box."
  fi
fi

# 10. The box: stopped unless the owner says no (it costs about $7.58 an hour while it runs).
if [ "$DRY" = 1 ]; then
  say "dry run: the box is left as it is ($(box_line | awk '{print $3}'))"
  finish "$CODE"
fi
STATE=$(box_line | awk '{print $3}')
if [ "$STATE" != running ] && [ "$STATE" != pending ]; then
  say "the box is $STATE"
  finish "$CODE"
fi
# Work of someone else's on the box: a trainer, or a ladder.py there. (The evaluator, psppo-eval,
# starts with the box and only waits for checkpoints while no trainer runs: not counted.)
BUSY=""
if [ -n "${HOST:-}" ] && [ -n "${KEY:-}" ]; then
  BUSY=$(box_training)
  if bssh "pgrep -f '[l]adder[.]py' >/dev/null" 2>/dev/null; then BUSY="${BUSY:+$BUSY }ladder.py"; fi
fi
echo
if [ -n "$BUSY" ]; then
  echo "The box is also running something else: $BUSY. Stopping it would stop that too."
  ANSWER=""
  read -r -t 60 -p "Stop the cloud box now anyway? [y/N] " ANSWER || echo
  case "$ANSWER" in y|Y|yes) stop_box ;; *) stop_guard; say "the box is left running (about \$7.58 an hour): bash $AWS2 ec2 stop-instances --region $REGION --instance-ids $ID stops it" ;; esac
else
  [ "$STARTED_HERE" = 1 ] || echo "(the box was already running when this started)"
  ANSWER=""
  read -r -t 60 -p "Stop the cloud box now? [Y/n] " ANSWER || echo
  case "$ANSWER" in
    n|N|no|No) stop_guard; say "the box is left running (about \$7.58 an hour): bash $AWS2 ec2 stop-instances --region $REGION --instance-ids $ID stops it" ;;
    *) stop_box ;;
  esac
fi
finish "$CODE"
