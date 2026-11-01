#!/usr/bin/env bash
# train.py --rollout-nice at full load on one box (written for box c: 16 vCPU, one L40S).
#
# NSERV Showdown servers (ports BASE..BASE+NSERV-1), the live trainer flags, INFER inference actors and the
# learner on GPU 0, resumed from CKPT. Over WINDOW seconds of steady state it measures the decisions
# inferred and learned per second, the CPU by process group, the run-queue waits and the learner's
# seconds per minibatch, and it records the nice of every thread of the learner, the inference actors,
# the rollout workers and the Showdown servers.
#
#   MODE=flag   train.py --rollout-nice NICE, and the servers started the way the run.json edit starts them
#               (servers.command = nice -n NICE python start_servers.py ...). Nothing is reniced by hand.
#   MODE=off    the control: no flag, the servers at nice 0 (Ray's own worker_niceness, 15)
#   MODE=hand   box a today: as off, then every thread of the learner and the inference actors put to nice 0 by
#               hand (sudo renice, as on box a at 09:49 UTC 2026-10-17) once the workers are up
#   HUMAN=on    the top-player term on, as in the earlier box c measurements (bench notes: 0.147 s per
#               minibatch at Ray's default, 0.124 with the hand renice); HUMAN=off as the live run.json is now
#   SEATS=...   the opponent seats (--opponent-seats), default the live setupnet:0.15. With psbot or foulplay
#               seats the workers start helper processes; their threads are listed as the workers' processes
#   EXTRA="..." more train.py arguments for this run (v2-fast: EXTRA="--fast-update --fast-ingest" beside MODE=flag
#               for every speed flag on); CODE may name any tree, 4cfd496 included (MODE=off or hand there)
#   KILL=1      after the window, one rollout worker is killed by its own pid (SIGKILL): what train.py and Ray
#               do next is logged, and then the trainer is started again the way the supervisor starts it
#               after a trainer exit, and the new workers' nice is recorded
#
#   CKPT=~/dev/fast/side/ckpt/learner_update_009760.pt MODE=flag HUMAN=on \
#       flock ~/dev/fast/bench.lock bash bench/rollout_nice_full.sh
#
# Run it under flock ~/dev/fast/bench.lock. It stops everything it started, by pid (no pkill, no ray stop),
# and deletes nothing.
set -u
CODE=${CODE:-$HOME/dev/fast-nice/code}
OUT=${OUT:-$HOME/dev/fast-nice/out}
SIDE=${SIDE:-$HOME/dev/fast/side}                   # cpushare.py and schedwait.py (read-only /proc readers)
CKPT=${CKPT:?CKPT names the learner checkpoint to resume from}
HSHARDS=${HSHARDS:-$HOME/dev/fast/prof/human_shards}
PY=${PY:-$HOME/venv/bin/python}
SHOWDOWN=${SHOWDOWN:-$HOME/pokemon-showdown}
NSERV=${NSERV:-12}; BASE=${BASE:-11460}; BATTLES=${BATTLES:-768}; INFER=${INFER:-2}
MODE=${MODE:-flag}; NICE=${NICE:-5}; HUMAN=${HUMAN:-on}; KILL=${KILL:-0}; WINDOW=${WINDOW:-150}
SEATS=${SEATS:-setupnet:0.15}
EXTRA=${EXTRA:-}
TAG=${TAG:-${MODE}_human${HUMAN}}
LAST=$((BASE + NSERV - 1))
O=$OUT/full_$TAG; mkdir -p $O/servers
CK=$O/ckpt; mkdir -p $CK
[ -f $CK/$(basename $CKPT) ] || cp -l $CKPT $CK/ 2>/dev/null || cp $CKPT $CK/
STOP=$O/stop.$(date +%s)

listening() { local n=0 p; for p in $(seq $BASE $LAST); do ss -Hltn "sport = :$p" | grep -q . && n=$((n + 1)); done; echo $n; }
tree() {  # tree ROOT: "pid comm" of ROOT and of every process under it, from one ps listing
  ps -eo pid=,ppid=,comm= | awk -v root=$1 '{ pid[NR] = $1; pp[NR] = $2; c[NR] = $3 }
    END { keep[root] = 1; more = 1
          while (more) { more = 0; for (i = 1; i <= NR; i++) if (!(pid[i] in keep) && (pp[i] in keep)) { keep[pid[i]] = 1; more = 1 } }
          for (i = 1; i <= NR; i++) if (pid[i] in keep) print pid[i], c[i] }'
}
desc() { tree $1 | cut -d' ' -f1; }
named() {  # named ROOT REGEX: the processes under ROOT whose whole comm matches REGEX (grep -E)
  tree $1 | grep -E "^[0-9]+ ($2)\$" | cut -d' ' -f1
}
thr() {  # thr LABEL PID...: how many threads of these processes are at each nice
  local label=$1; shift
  [ $# -gt 0 ] || { echo "$label: none"; return; }
  echo "$label ($# procs): $(ps -L -o ni= -p $(echo "$@" | tr ' ' ',') | sort -n | uniq -c | awk '{printf "%s threads at nice %s; ", $1, $2}')"
}

[ "$(listening)" = 0 ] || { echo "ports $BASE-$LAST are taken: not starting"; exit 1; }
case "$MODE" in flag) SPRE="nice -n $NICE"; FARGS="--rollout-nice $NICE";; off|hand) SPRE=""; FARGS="";; *) echo "MODE flag|off|hand"; exit 1;; esac
HARGS=""; [ "$HUMAN" = on ] && HARGS="--human-imitation-coef 0.012 --human-imitation-every 4 --human-imitation-data $HSHARDS"
echo "mode $MODE${EXTRA:+ + $EXTRA} (human $HUMAN, seats $SEATS): $NSERV servers $BASE-$LAST, $BATTLES battles, $INFER inference actors, resume $(basename $CKPT), code $CODE"

cd $CODE
# As the supervisor starts them: nice execs the launcher, so $! is start_servers.py's own pid.
$SPRE $PY start_servers.py --dir $SHOWDOWN --ports $BASE-$LAST --heap-mb 2048 --log-dir $O/servers > $O/servers.log 2>&1 &
SPID=$!
for i in $(seq 1 180); do n=$(listening); [ "$n" -ge $NSERV ] && break; sleep 1; done
echo "servers listening: $n; launcher $(ps -o pid=,ni=,args= -p $SPID | cut -c1-90)"

start_trainer() {  # start_trainer LOG
  env CUDA_VISIBLE_DEVICES=0 RAY_ADDRESS=local PYTHONUNBUFFERED=1 $PY train.py --mode ppo --ports $BASE-$LAST --battles $BATTLES \
    --rooms-per-pair 16 --teacher psbot --steps-per-update 36864 --minibatch 768 --seq-len 256 --grad-accum 1 --save-every 1000 \
    --ckpt-dir $CK --stop-file $STOP --minutes 20 \
    --obs-matchup --obs-movesets --infer-actors $INFER --infer-sync-s 1 --set-head-coef 0.1 --opp-action-aux-coef 0.1 \
    --read-bonus 0.3 --read-anneal 20000000:45000000 --opponent-seats $SEATS --boosted-row-weight 2 \
    --lr-hold-steps 479713 --lr-total-steps 1113566 $HARGS $FARGS $EXTRA > $1 2>&1 &
  TPID=$!
}
wait_up() {  # wait_up LOG UPDATES: the workers up, UPDATES updates and one telemetry line
  local log=$1 want=$2 u nw nt end=$(( $(date +%s) + 600 ))
  while [ $(date +%s) -lt $end ]; do
    u=$(grep -c "learner\] upd=" $log); nt=$(grep -c "\[Telemetry\]" $log)
    nw=$(named $TPID "ray::RolloutWor|ray::OpponentSe" | wc -l)
    [ "$u" -ge "$want" ] && [ "$nw" -ge $NSERV ] && [ "$nt" -ge 1 ] && break
    kill -0 $TPID 2>/dev/null || { echo "train.py left early"; break; }
    sleep 2
  done
  echo "workers up: $nw of $NSERV, updates $u, telemetry lines $nt at $(date -u +%T)"
}
nices() {  # nices FILE: every thread's nice per group, and the full ps -eLo listing
  L=$(named $TPID "ray::LearnerAct"); I=$(named $TPID "ray::InferenceA"); W=$(named $TPID "ray::RolloutWor|ray::OpponentSe")
  N=$(for p in $(desc $SPID); do [ $p = $SPID ] || echo $p; done)
  SIMS=$(for p in $N; do ps -o args= -p $p 2>/dev/null | grep -q room-battle.js && echo $p; done)
  C=$(for w in $W; do desc $w | grep -vx $w; done)
  {
    thr learner $L; thr inference $I; thr "rollout workers" $W; thr "processes the rollout workers started" $C
    thr "server launcher" $SPID; thr "node servers and sims" $N; thr "of which sims" $SIMS
    thr "other ray processes" $(named $TPID "ray::.*" | grep -vxE "$(echo $L $I $W | tr ' ' '|')")
  } | tee $1
  ps -eLo pid,tid,ni,comm > ${1%.txt}_ps_eLo.txt
}

start_trainer $O/train.log
echo "train pid $TPID"
wait_up $O/train.log 2
if [ "$MODE" = hand ] && kill -0 $TPID 2>/dev/null; then
  # as box a now: every thread of the learner and the inference actors at 0, by hand (lowering needs sudo)
  for p in $(named $TPID "ray::LearnerAct|ray::InferenceA"); do for t in /proc/$p/task/*; do sudo -n renice -n 0 -p ${t##*/} > /dev/null; done; done
  echo "hand renice of the learner and the inference actors to 0 at $(date -u +%T)"
fi
nices $O/nice_threads.txt
grep -E "CPU priorities|rollout nice" $O/train.log | sed -E 's/^.*\[(INFO|WARNING|ERROR|CRITICAL)\] //' | cut -c1-260
MEASURED=0
if kill -0 $TPID 2>/dev/null; then
MEASURED=1
# one more update before measuring (the same window as the earlier runs)
u0=$(grep -c "learner\] upd=" $O/train.log)
for i in $(seq 1 300); do [ "$(grep -c "learner\] upd=" $O/train.log)" -gt "$u0" ] && break; sleep 1; done
echo "measuring from $(date -u +%T), updates so far $(grep -c "learner\] upd=" $O/train.log)"
grep "Telemetry" $O/train.log | tail -1 > $O/telem_a.txt
nA=$(grep -c "learner\] upd=" $O/train.log)
python3 $SIDE/cpushare.py $WINDOW > $O/cpushare.txt
grep "Telemetry" $O/train.log | tail -1 > $O/telem_b.txt
nB=$(grep -c "learner\] upd=" $O/train.log)
grep "learner\] upd=" $O/train.log | sed -n "$((nA+1)),${nB}p" | grep -oE "n_mb=[0-9]+ sec=[0-9.]+" > $O/updates_in_window.txt
python3 $SIDE/schedwait.py 20 $(echo $L | cut -d' ' -f1):learner $(echo $I | cut -d' ' -f1):infer $(echo $W | cut -d' ' -f1):worker \
  $(echo $SIMS | cut -d' ' -f1):sim > $O/schedwait.txt 2>&1
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader > $O/gpu.txt
nices $O/nice_threads_end.txt > /dev/null
else
  echo "train.py is not running: nothing measured"
fi

TREES="$(desc $TPID | tr '\n' ' ')"
if [ "$KILL" = 1 ] && kill -0 $TPID 2>/dev/null; then
  W0=$(named $TPID "ray::RolloutWor|ray::OpponentSe" | tr '\n' ' ')
  V=$(echo $W0 | cut -d' ' -f1)
  echo "kill test: SIGKILL to rollout worker pid $V ($(ps -o comm=,ni= -p $V)) at $(date -u +%T)"
  kill -KILL $V
  SEEN=""
  for i in $(seq 1 120); do
    NEW=$(for p in $(named $TPID "ray::RolloutWor|ray::OpponentSe" 2>/dev/null); do case " $W0 " in *" $p "*) ;; *) echo $p;; esac; done | tr '\n' ' ')
    NOW="$NEW $(for p in $NEW; do ps -L -o ni= -p $p | sort -u | tr -d ' \n'; done)"
    [ -n "$NEW" ] && [ "$NOW" != "$SEEN" ] && { echo "  +${i}s: new worker process(es) $NEW: $(thr '' $NEW)"; SEEN=$NOW; }
    kill -0 $TPID 2>/dev/null || { echo "  +${i}s: train.py exited"; break; }
    sleep 1
  done
  grep -E "crashed|Shutting down|answer without|asked again|final checkpoint" $O/train.log | sed -E 's/^.*\[(INFO|WARNING|ERROR|CRITICAL)\] //' | cut -c1-200
  if kill -0 $TPID 2>/dev/null; then
    echo "  train.py still runs 120 s after the kill"
    nices $O/nice_threads_after_kill.txt
  else
    wait $TPID; echo "  train.py exit code $?"
    sleep 5
    for p in $TREES; do kill -0 $p 2>/dev/null && { echo "left over after the exit: $p $(ps -o args= -p $p | cut -c1-80)"; kill -TERM $p; }; done
    echo "restart: the trainer started again the way the supervisor does after a trainer exit"
    start_trainer $O/train_restart.log
    wait_up $O/train_restart.log 1
    nices $O/nice_threads_restart.txt
    grep -E "CPU priorities|rollout nice" $O/train_restart.log | sed -E 's/^.*\[(INFO|WARNING|ERROR|CRITICAL)\] //' | cut -c1-260
    TREES="$TREES $(desc $TPID | tr '\n' ' ')"
  fi
fi

touch $STOP
for i in $(seq 1 180); do kill -0 $TPID 2>/dev/null || break; sleep 1; done
if kill -0 $TPID 2>/dev/null; then echo "train.py did not stop on the stop file: SIGTERM"; kill -TERM $TPID; fi
wait $TPID
echo "train exit $?"
sleep 5
for p in $TREES; do kill -0 $p 2>/dev/null && { echo "left over: $p $(ps -o args= -p $p | cut -c1-80)"; kill -TERM $p; }; done
kill -INT $SPID
wait $SPID
echo "servers stopped at $(date -u +%T); still listening: $(listening)"
# Anything of this run still alive: a process working in this code tree (the trainer, Ray and its workers),
# one that runs a file of it (Foul Play's helpers work in ~/foul-play, in a session of their own) or one that
# names this run's ports or output directory.
sleep 3
# This script's own callers (an orchestrating script that works in $CODE, flock) are never stopped.
ANC=" "; q=$$
while [ -n "$q" ] && [ "$q" -gt 1 ] 2>/dev/null; do ANC="$ANC$q "; q=$(ps -o ppid= -p $q 2>/dev/null | tr -d ' '); done
for d in /proc/[0-9]*; do
  p=${d#/proc/}; case "$ANC" in *" $p "*) continue;; esac
  cwd=$(readlink $d/cwd 2>/dev/null); args=$(tr '\0' ' ' < $d/cmdline 2>/dev/null)
  case "$cwd" in $CODE|$CODE/*) ;; *) case "$args" in *"$BASE-$LAST"*|*"$O"*|*"$CODE/"*) ;; *) continue;; esac;; esac
  case "$args" in *rollout_nice_full.sh*|"") continue;; esac
  echo "left over at the end: $p $(echo "$args" | cut -c1-80)"; kill -TERM $p
done
[ "$MEASURED" = 1 ] && python3 - $O <<'EOF'
import datetime, re, sys
O = sys.argv[1]
def tel(p):
    s = open(p).read()
    t = re.search(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d+)", s)   # the date too: a window may cross midnight
    tt = datetime.datetime.strptime(t[1], "%Y-%m-%d %H:%M:%S").timestamp() + int(t[2]) / 1000
    return tt, int(re.search(r"total_requests.: (\d+)", s)[1]), int(re.search(r"total_steps.: (\d+)", s)[1]), int(re.search(r"episodes.: (\d+)", s)[1])
a, b = tel(O + "/telem_a.txt"), tel(O + "/telem_b.txt")
dt = b[0] - a[0]
ups = [tuple(map(float, re.findall(r"=([0-9.]+)", l))) for l in open(O + "/updates_in_window.txt")]
mb = sum(u[0] for u in ups); sec = sum(u[1] for u in ups)
print("RESULT window %.0f s: inferred %.0f/s, learned %.0f/s, learning episodes %.1f/s, updates %d, sec/mb %.4f" % (
    dt, (b[1]-a[1])/dt, (b[2]-a[2])/dt, (b[3]-a[3])/dt, len(ups), sec/mb if mb else float("nan")))
EOF
