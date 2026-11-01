# Run T (topstart): the launch plan

Run T trains the v2 network from random weights on the decisions of 2100+ players (offline, from shards, no PSBot
anywhere), then runs warmup and 600M PPO decisions with every v2 technique, as a one-variable A/B against the
from-scratch run S. This file is the exact launch: what the owner decides first, the commands in order, the checks,
what to watch in the first hours, the bars, and the undo. **Nothing of the real run has been started.** The
rehearsal ran on box a on 2026-10-18 07:14 to 07:27 UTC and was checked again from its files (section 8). Box a now
trains the owner's extension of S (`~/runs/v2/scratchx`, to 800M), so T launches on box a after scratchx is done.
The owner's session launches it.

| | |
|---|---|
| Release | box a `~/rel/topstart`: branch `topstart` as a `git archive` with LF line ends over a copy of `~/rel/v2-humanaux` (its `heuristic/cache`: 13 files, a set table of 509 species). It holds b79370b: its 315 tracked files match that commit's sha256 one for one. The commit with this file changes only this file; step 3.2 puts it on the box |
| Code | topstart-lr-decisions 8dfd55f (learning rate in decisions), v2-fast 8ecb1a2 (`--fast-update --fast-ingest --rollout-nice`), topstart-topstart 6777b8c (`human/topstart.py`, `cloud/v2/topstart_setup.py`). Every branch of origin it builds on is in it (checked after `git fetch` at 07:52) |
| Data | 781 read shards of the manifest `s3://<your-bucket>/human/imitation_data.json` (sha256 6244ae54..., the same in S3, on box a and on box c), linked by its recipe; the 65 skip shards are never linked. `index`: 169,241 train sides, 5,024,032 train decisions, 25,803 test_player sides, 8,878 test_game sides, pass; held-out set digest ho866789933298928784 (32,785 decisions, 226 players) on both boxes |
| Where | Offline start, control and gate evaluations: **box c, now** (`~/dev/topstart`, ports 11600 to 11699, GPU 0). The run: **box a after scratchx** (`~/runs/v2/topstart`, all 4 GPUs) |
| Units | `psppo-v2-topstart-train`, `psppo-v2-topstart-eval` (written by the setup in step 3.5, from S's template) |
| Ports on box a | servers 10700-10735, evaluator 10790 (scratchx: 10600-10635 and 10690; ladder-pick 11500-11599; the rehearsal 11700-11735) |
| Time | box c: start T0 1.6 to 2.9 h, control C0 25 min, gate evaluations about 1 h each. Box a: scratchx's trainer ends about 17:00 UTC (it ran 6,135 decisions/s at 08:12; the owner's estimate is about 19:00), then T: warmup about 5 minutes, PPO 27 to 30 h at 5,500 to 6,200 decisions/s (S took 36.4 h) |

`A=$(bash C:/Users/you/ps-ppo-v2/cloud/v2/boxssh.sh a)` and `C=$(bash C:/Users/you/ps-ppo-v2/cloud/v2/boxssh.sh c)`
give the ssh commands used below. A `~` inside single quotes is the box's home.

## 0. Two decisions for the owner before the launch

**D1, disk on box a (required).** Box a has 39 GB free now (`df -h`). scratchx saves every 50 updates and will
write about 103 more checkpoints of 229 MB (about 22 GB), so about 16 GB will be free when it ends. T needs about
18 GB (80 PPO checkpoints at save_every 200, plus warmup files) and the supervisor stops every run below 10 GB free.
Left as is, **T stops itself somewhere between 150M and 200M decisions**. Nothing may be deleted, so space comes
from moving files to the box's instance store `/opt/dlami/nvme` (3.3 TB free, ephemeral: wiped when the box
stops; S3 keeps the lasting copy).

| Option | What moves | Frees | Note |
|---|---|---|---|
| **A (recommended)** | S's checkpoints except updates 700, 2,700, 4,050, 8,100 and 16,041 (331 files) | about 70 GB | All 336 of S's checkpoints (71 GB) are in `s3://<your-bucket>/main-g6e12/runs/v2/scratch/checkpoints/` with equal sizes (checked 08:20). S is finished and fully measured. Can be done any time; `ionice -c3` keeps it off scratchx's way |
| B | scratchx's intermediate checkpoints (all but its last and 16,041), after its final evaluation | about 22 GB | Leaves about 20 GB free when T ends; possible only after scratchx ends |
| C | nothing | 0 | T stops between 150M and 200M decisions. Not viable |

The backup never deletes in S3 (no `--delete`, a versioned bucket, and the box's role cannot delete), so a move
leaves S3 as it is.

**D2, where step 1 runs.** Recommended: box c, now, while scratchx trains, so the gate is decided before scratchx
ends and T starts the moment box a is free. Box c is idle (load 0.1, GPU free at 08:09) and has everything in place
(section 1). The alternative is box a after scratchx (GPUs 1 to 3), which adds about 4 h of box a before PPO.

## 1. Step 1 on box c, now: the start T0, the gate's P0, the control C0

Ready on box c (prepared and checked 08:13 to 08:25, nothing started):

| Path | What |
|---|---|
| `~/dev/topstart/rel` | the release replica: this commit's `git archive` (LF) plus the release's `heuristic/cache` (sha256 equal to box a's); `RELEASE` names the commit |
| `~/dev/topstart/full/ALL` | the 781 read shards (3.8 GB from S3), `index.json`: pass, the same counts and held-out digest as box a |
| `~/dev/topstart/ckpt/S/` | S's P0 (`imitation_learner_update_000324.pt`, 47812333...), SFIN (`ppo_learner_update_016041.pt`, a1ad3087...), updates 700, 2,700, 4,050, 8,100, S's `S_metrics.jsonl` and `S_files.txt`; every checkpoint's sha256 equal to box a's |
| `~/dev/topstart/ckpt/P0_ms.pt` | P0 widened with the moveset block (`tools/widen_checkpoint.py --movesets`): the same weights as box a's `~/dev/topstart/P0_ms.pt`, only the widening's timestamp differs |
| `~/dev/topstart/measure/P0_ms.json` | `measure` on P0_ms: capped 0.4576 (bit for bit box a's 0.4576178130), all of test_player 0.4543, test_game 0.4540 |

Check that nothing else uses box c, then start T0 and P0's evaluation (they share GPU 0; the evaluation needs little
of it):

```
$C 'uptime; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; pgrep -af "pytest|train.py|evaluate.py|topstart.py" | grep -v pgrep; free -g | sed -n 2p'

# T0: random weights, every default of the plan (warmup 1,000, hold, plateau anneal from 60k, at most 150k steps, abort 40k:0.30)
$C 'mkdir -p ~/dev/topstart/start && cd ~/dev/topstart/rel && CUDA_VISIBLE_DEVICES=0 setsid nohup nice -n 10 ~/venv/bin/python human/topstart.py train --data ~/dev/topstart/full/ALL --work ~/dev/topstart/T0 --out ~/dev/topstart/start/T0.pt > ~/dev/topstart/T0.log 2>&1 < /dev/null &'

# P0 for gate G1: Foul Play 400 games at 88,000 iterations (4 at a time), then simple and psbot 200 each
$C 'cd ~ && REL=~/dev/topstart/rel ALLOW_PORT=1 setsid nohup bash ~/dev/topstart/rel/cloud/v2/topstart_eval.sh P0 ~/dev/topstart/ckpt/S/imitation_learner_update_000324.pt 11691 0 > /dev/null 2>&1 < /dev/null &'
```

- Progress: `$C 'grep -E "held-out|anneal|abort|wrote|nonfinite" ~/dev/topstart/T0.log | tail'` (a held-out line every
  2,000 steps). The rehearsal's curve from random weights: 0.231 at 1k, 0.235 at 2k, 0.336 at 3k, 0.389 at 5k, 0.445
  at 8k. Speed: 15 steps a second on box a's L40S, 12 to 13 in box c's 300-step smoke (the first group loading).
- Stop and resume: `touch ~/dev/topstart/T0/STOP` stops at the next group boundary with a resume file. Move the STOP
  file aside before running the same command again, or the rerun stops after one group. A rerun resumes from the
  newest `resume_*.pt` and reproduces the same batches.
- An evaluation's summary: `$C 'cat ~/dev/topstart/eval-P0/runs.log'`. Expect about 1 h for 400 Foul Play games
  (median game 38 to 46 s, 4 at a time). `topstart_eval.sh` refuses an output inside `~/runs`; `ALLOW_PORT=1` lets it
  take box c's ports.

When T0 has written `~/dev/topstart/start/T0.pt` (its `.sha256` and `.json` beside it):

```
# C0, the measured control: arm C's recipe from P0 (3 passes, about 20,300 steps, cosine from step 0); about 25 min
$C 'cd ~/dev/topstart/rel && CUDA_VISIBLE_DEVICES=0 setsid nohup nice -n 10 ~/venv/bin/python human/topstart.py train --data ~/dev/topstart/full/ALL --work ~/dev/topstart/C0 --out ~/dev/topstart/start/C0.pt --init ~/dev/topstart/ckpt/P0_ms.pt --warmup 100 --min-hold 0 --max-passes 3 --abort-at off --best-after 0 > ~/dev/topstart/C0.log 2>&1 < /dev/null &'

# T0's measures (about 3 minutes each) and its gate evaluation
$C 'cd ~/dev/topstart/rel && for x in T0:start/T0.pt SFIN:ckpt/S/ppo_learner_update_016041.pt; do CUDA_VISIBLE_DEVICES=0 nice -n 10 ~/venv/bin/python human/topstart.py measure --data ~/dev/topstart/full/ALL --checkpoint ~/dev/topstart/${x#*:} --out ~/dev/topstart/measure/${x%%:*}.json > ~/dev/topstart/measure/${x%%:*}.log 2>&1; done'
$C 'cd ~ && REL=~/dev/topstart/rel ALLOW_PORT=1 setsid nohup bash ~/dev/topstart/rel/cloud/v2/topstart_eval.sh T0 ~/dev/topstart/start/T0.pt 11692 0 > /dev/null 2>&1 < /dev/null &'
```

When C0 is written: the same `measure` for C0 and `topstart_eval.sh C0 ~/dev/topstart/start/C0.pt 11693 0`.
Then copy the outputs to S3 (counts only, no player hash; the shards and S's files are already there):

```
$C 'aws s3 cp ~/dev/topstart/start/ s3://<your-bucket>/topstart/start/ --recursive --only-show-errors && aws s3 sync ~/dev/topstart s3://<your-bucket>/topstart/work-boxc/ --exclude "full/*" --exclude "ckpt/*" --exclude "rel/*" --only-show-errors'
```

## 2. Gate G1 (fixed before any training)

Launch T from T0 only if **FP(T0) >= FP(P0)**, both 400 games on box c, **and T0's capped held-out top-1 >= 0.50**.
Otherwise do not launch: report P0's, T0's and C0's Foul Play numbers and measures, with the options C0 as the start,
a longer topstart, or no new run. B1's agreement bars are in section 7 (P0 0.4576 and SFIN 0.5151 on this set, so
T0 needs 0.5076 and 0.5351 there, and 0.55 overall). C0 does not hold up the launch; it is reported beside.

## 3. Steps on box a once scratchx is done

### 3.1 The box after scratchx

```
$A 'python3 -c "import json;s=json.load(open(\"/home/ubuntu/runs/v2/scratchx/state.json\"));print(s[\"status\"],s[\"phase\"],s[\"decisions_total\"])"; ls ~/runs/v2/scratchx/eval; uptime; free -g | sed -n 2p; df -h / | tail -1; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; ls ~/dev/KEEP_BOX; cat ~/runs/v2/guard.env; ss -ltn | grep -cE ":107[0-9][0-9]\b"; pgrep -af "topstart|evaluate.py" | grep -v pgrep'
```

- scratchx `done` (its supervisor then idles, holding no server, port or GPU). Its evaluator may still be running:
  the guard starts it 3.5 h after the mtime of scratchx's first trainer log, which stops at the 6-hourly restart
  (about 13:52), so about 17:25, and it writes `eval/final.done` after the final checkpoint's games (S's took 77
  minutes). To have it sooner: `$A 'sudo systemctl start psppo-v2-scratchx-eval'` once state.json says done. It may
  run beside T's start, as S's evaluator always ran beside S (2 Foul Play processes, a little of GPU 0).
- Nothing listening on 10700-10799, `KEEP_BOX` present (the guard then never stops the box).

### 3.2 The latest version of the release onto box a

The release holds b79370b; this commit changes only this file. Put it over the release before anything runs from
it, and check it:

```
cd C:/Users/you/ps-ppo-v2-topstart && S=$(git rev-parse --short topstart) && git -c core.autocrlf=false archive --format=tar topstart | $A "(test -e ~/dev/topstart/rel-before/RELEASE.b79370b || cp -p ~/rel/topstart/RELEASE ~/dev/topstart/rel-before/RELEASE.b79370b) && tar -x -C ~/rel/topstart && echo 'topstart $S $(date -u +%FT%TZ) (git archive, LF, over a copy of ~/rel/v2-humanaux; Python code as 433cb30, which the rehearsal ran)' > ~/rel/topstart/RELEASE && cat ~/rel/topstart/RELEASE"

# the check: every tracked file of the branch, sha256 in box a's release against box c's replica of the same commit
T=$(mktemp -d) && git ls-tree -r --name-only topstart > $T/files && $A 'cd ~/rel/topstart && tr "\n" "\0" | xargs -0 sha256sum' < $T/files > $T/a.sha && $C 'cd ~/dev/topstart/rel && tr "\n" "\0" | xargs -0 sha256sum' < $T/files > $T/c.sha && diff $T/a.sha $T/c.sha && echo "release == topstart, $(wc -l < $T/files) files"
```

The diff must be empty. (Box c's `~/dev/topstart/rel` is this commit's archive; on 2026-10-18 the same check matched
box a's b79370b file for file.)

### 3.3 Disk (D1)

Option A, after the owner's OK (lists in `~/dev/topstart`, nothing deleted, nothing overwritten):

```
$A 'set -e; D=/opt/dlami/nvme/aside/scratch-checkpoints; mkdir -p $D; cd ~/runs/v2/scratch/checkpoints; ls -l learner_update_*.pt | awk "{print \$9, \$5}" | sort > ~/dev/topstart/disk-local.txt; aws s3 ls s3://<your-bucket>/main-g6e12/runs/v2/scratch/checkpoints/ | awk "{print \$4, \$3}" | grep learner_update_ | sort > ~/dev/topstart/disk-s3.txt; diff ~/dev/topstart/disk-local.txt ~/dev/topstart/disk-s3.txt; for f in learner_update_*.pt; do case $f in learner_update_000700.pt|learner_update_002700.pt|learner_update_004050.pt|learner_update_008100.pt|learner_update_016041.pt) ;; *) test -e $D/$f || ionice -c3 mv $f $D/;; esac; done; ls $D | wc -l; df -h / | tail -1'
```

It stops at the `diff` if any file is not in S3 with the same size. Expect 331 files moved and 85 to 90 GB free.
Option B is the same loop over `~/runs/v2/scratchx/checkpoints` keeping its newest file and `learner_update_016041.pt`,
only after scratchx's `eval/final.done`, with its own S3 listing (`main-g6e12/runs/v2/scratchx/checkpoints/`).
**Do not go on with less than 30 GB free.**

### 3.4 The start onto box a

```
$A 'mkdir -p ~/dev/topstart/start && cd ~/dev/topstart/start && test ! -e T0.pt && aws s3 cp s3://<your-bucket>/topstart/start/ . --recursive --exclude "*" --include "T0.pt*" --only-show-errors && sha256sum -c T0.pt.sha256'
```

### 3.5 The run directory, the units, the checks

```
$A 'cd ~/rel/topstart && ~/venv/bin/python cloud/v2/topstart_setup.py --rel ~/rel/topstart --start ~/dev/topstart/start/T0.pt --fast --write --units'
$A '~/venv/bin/python ~/rel/topstart/cloud/check_restart.py --config ~/runs/v2/topstart/run.json --run-dir ~/runs/v2/topstart --phase warmup'
$A '~/venv/bin/python ~/rel/topstart/cloud/check_restart.py --config ~/runs/v2/topstart/run.json --run-dir ~/runs/v2/topstart --phase ppo --checkpoint ~/runs/v2/topstart/checkpoints/learner_update_000000.pt'
$A 'cd ~/rel/topstart && ~/venv/bin/python evaluate.py --run-dir ~/runs/v2/topstart --check-config ~/runs/v2/topstart/run.json'
$A '~/venv/bin/python ~/rel/topstart/pipeline.py --config ~/runs/v2/topstart/run.json --run-dir ~/runs/v2/topstart --dry-run | head -20; ~/venv/bin/python ~/rel/topstart/pipeline.py --config ~/runs/v2/topstart/run.json --run-dir ~/runs/v2/topstart --can-start; echo can-start $?'
```

Each must exit 0. The two check_restart runs end with "the checkpoint is of phase 'imitation' and the start is
'warmup'" (then 'ppo'): "the optimizer and the schedule start afresh". The PPO one prints "schedule in decisions of
this phase: full rate up to 357,000,000 decisions, then down to 1/27 at 600,000,000". The setup refuses (exit 2) an
existing state.json, a start without set_head.*, a start that is not mode imitation with all three observation
settings, a release without the decisions flags or the heuristic cache, and a v2-fast release without
`--fast`/`--no-fast`. The units are written but not enabled or started. The setup's base is S's run.json
(`~/runs/v2/scratch/run.json`, sha256 bde91ae2..., unchanged since S ended); scratchx's run.json is not used.

The run.json it writes (rebuilt on box c from a copy of S's run.json: identical to the rehearsal's real-size dry run
`dryT` but for repo_dir):

```
{"repo_dir":"/home/ubuntu/rel/topstart","python":"/home/ubuntu/venv/bin/python","showdown_dir":"/home/ubuntu/pokemon-showdown",
 "servers":{"count":36,"base_port":10700,"heap_mb":2048,"ready_timeout_seconds":420,
   "command":["nice","-n","5","{python}","start_servers.py","--dir","{showdown_dir}","--ports","{ports}","--heap-mb","{heap_mb}","--log-dir","{server_log_dir}"]},
 "trainer":{"stop_timeout_seconds":420,"failure_stop_timeout_seconds":120,"env":{"CUDA_VISIBLE_DEVICES":"0,1,2,3"}},
 "train_defaults":{"battles":2304,"rooms_per_pair":16,"minibatch":768,"seq_len":256,"grad_accum":1,"steps_per_update":36864,"teacher":"psbot","save_every":200,
   "extra_train_args":["--obs-matchup","--obs-movesets","--infer-actors","3","--infer-sync-s","1","--rollout-nice","5"]},
 "phases":{"imitation":{"skip":true,"min_decisions":0,"max_decisions":1},
   "warmup":{"skip":false,"min_decisions":500000,"max_decisions":1500000,
     "plateau":{"metric":"vloss","window_decisions":300000,"min_improvement_pct":2.0,"smooth_updates":4},"train":{"save_every":5},
     "extra_train_args":["--set-head-coef","0.1"]},
   "ppo":{"skip":false,"target_decisions":600000000,
     "extra_train_args":["--set-head-coef","0.1","--opp-action-aux-coef","0.1","--read-bonus","0.3","--read-anneal","20000000:45000000",
       "--opponent-seats","setupnet:0.15","--boosted-row-weight","2","--lr-hold-decisions","357000000","--lr-total-decisions","600000000",
       "--fast-update","--fast-ingest"]}},
 "watchdog":{"stall_minutes":10,"startup_grace_minutes":20,"update_stall_minutes":30,"max_restarts_per_hour":6,"backoff_seconds":[30,120,300,600,900],
   "max_fruitless_starts":5,"nan_updates_limit":3,"telemetry_error_minutes":5,"recycle_hours":6},
 "disk":{"min_free_gb":10},"retention":{"enabled":false,"keep_newest":4,"milestone_every":500,"keep_hours":0,"check_minutes":10},
 "on_done":"idle","on_failed":"idle",
 "eval":{"every_hours":4,"games":200,"opponents":["simple","psbot"],"port":10790,"foulplay":{"games":200,"every_hours":4,"parallel":2,"iterations":88000}}}
```

`teacher: psbot` is S's setting and does nothing here: PSBot's engine runs only in the imitation modes, and T skips
that phase. Warmup and PPO play the network.

## 4. Start it

Only when G1 passed, every check of 3.5 exited 0, free disk is at least 30 GB, and no offline job or one-off
evaluation runs on box a (`$A 'pgrep -af "topstart.py|evaluate.py" | grep -v -- --watch | grep -v pgrep'` prints
nothing; scratchx's `--watch` evaluator may run).

```
$A 'test -e ~/runs/v2/guard.env.pre-topstart || cp -p ~/runs/v2/guard.env ~/runs/v2/guard.env.pre-topstart; sed -i "s/^ARMS=.*/ARMS=\"scratchx topstart\"/" ~/runs/v2/guard.env; cat ~/runs/v2/guard.env'
$A 'sudo systemctl enable --now psppo-v2-topstart-train && systemctl is-active psppo-v2-topstart-train'
```

- guard.env becomes `ARMS="scratchx topstart"`, `CHAIN=""`, `IDLE_MIN=120`. The guard re-reads it every 5 minutes.
  With scratchx still in ARMS it keeps (and, after a reboot, restarts) scratchx's evaluator until its final games are
  played; it starts `psppo-v2-topstart-eval` 3.5 h after the mtime of T's first trainer log (the warmup's, frozen
  about 5 minutes in), as it did for S: 200 games against simple, psbot and Foul Play at 88k every 4 h.
- Once `~/runs/v2/scratchx/eval/final.done` exists, in this order (the guard would restart an evaluator of an arm in
  ARMS): `$A 'sed -i "s/^ARMS=.*/ARMS=\"topstart\"/" ~/runs/v2/guard.env'`, then
  `$A 'sudo systemctl stop psppo-v2-scratchx-eval; sudo systemctl disable --now psppo-v2-scratchx-train'` (the second
  only stops scratchx's idle supervisor so a reboot does not bring it back; it changes no file).
- `~/dev/KEEP_BOX` stays, so the guard never stops the box. Enabled, the train unit comes back after a spot
  interruption and resumes from the newest checkpoint. With `--fast` the learner and the inference actors start at
  nice 0 on every restart and every rollout worker goes to 5; no hand renice is needed.
- Backup: psppo-v2-backup already copies all of `~/runs/v2` (so `~/runs/v2/topstart`) to
  `s3://<your-bucket>/main-g6e12/runs/v2/` every 10 minutes, never deleting. The start is in
  `s3://<your-bucket>/topstart/start/` (step 1) and in `~/runs/v2/topstart/topstart/` with its sha256.

## 5. What to watch in the first hours

```
$A 'tail -3 ~/runs/v2/topstart/logs/pipeline.log'                               # phase, decisions, N/s over the last 10m
$A 'L=$(ls ~/runs/v2/topstart/logs/trainer_*.log | tail -1); grep -E "PHASE CHANGE|schedule in decisions|rate at the resume|fast_ingest on|rollout nice|CPU priorities: learner|fast_update=" $L | cut -c1-200; grep -ciE "nan|traceback" $L'
$A 'python3 -c "import json; [print(r[\"phase\"], r[\"update\"], r[\"total_steps\"], r[\"vloss\"], r[\"kl\"], r[\"ent\"], r[\"clip\"], r[\"n_mb\"], r[\"sec\"], r[\"lr\"], r.get(\"opp_acc\"), r.get(\"set_loss\")) for r in map(json.loads, list(open(\"/home/ubuntu/runs/v2/topstart/metrics.jsonl\"))[-5:])]"'
$A 'df -h / | tail -1; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'
```

| When | Expect | Act if |
|---|---|---|
| Start | servers up in about 36 s; trainer log: `PHASE CHANGE DETECTED (imitation -> warmup)`, `rollout nice 5: 36 of 36 workers at nice 5 (... 0 could not be)`, `CPU priorities: learner pid ... nice 0; inference pids [...] nice [0, 0, 0]` | a worker that "could not be" set, a REFUSED line, a restart loop |
| First 3 minutes of every trainer start | pipeline.log and metrics.jsonl say `decisions=0` / `total_steps 0` until the first count, about 2 min 45 s in (S and scratchx show the same; the learner's own counter is right all along) | no decisions after 20 minutes (the watchdog restarts then anyway) |
| Warmup | vloss 13.6 at update 1, under 3 by update 5, about 2.4 by updates 15 to 20 (section 8, finding 1). Only `v_head.*` changes. About 5.9 s an update. It ends on the plateau rule or at 1.5M; expect 0.9M to 1.5M decisions, because the first count comes after about 28 updates | vloss above 4 after update 10, or NaN |
| PPO start | `PHASE CHANGE DETECTED (warmup -> ppo)`, `learning-rate schedule in decisions of this phase: ... until 357,000,000 decisions, then down to 1/27 at 600,000,000 and after`, `learner speed: fast_update=True`, `fast_ingest on: ...` | any SPEED.md section 5 trouble line (`joins its observations on the CPU`, `weight push of version N failed`, ...) more than once |
| PPO first hour | lr 2.91e-05 at update 1, then +2.88e-05 an update (the 1,000-step ramp at 96 steps an update) to 3.00e-04 from update 11; entropy about 1.43 at update 1 (S: 1.02; a copied start is more random); KL 0.002 to 0.012 in the rehearsal's first 8 updates, so KL early stops (n_mb under 96) may come at the full rate; clip up to about 0.39; vloss 2.3 to 2.4; set_loss about 0.63; opp_acc 0.34 at update 1 and above 0.85 within 5 updates; boosted share 0.22 to 0.29 | NaN, KL above 0.05 for several updates, entropy under 0.3 within the first hour |
| Speed | learner `sec` 5.3 to 6.1 at 96 minibatches; 6.1 to 6.7 s an update between restarts, 5,500 to 6,200 decisions/s (scratchx on the same box and release type: 6.0 s, 6,135/s). 600M then takes 27 to 30 h. scratchx's evaluator, while it runs, takes a little | under 4,500/s over an hour: check the nice levels (SPEED.md section 5 command) |
| Every few hours | a checkpoint every 200 updates (about 7.4M decisions, about 21 minutes), 229 MB each; the 6-hourly planned restart costs about 2 minutes; T's evaluator from about 3.6 h | free disk under 15 GB |

When the rate starts to fall (357.0M, about 17 h into PPO) and at 600M, the lr column must follow
3e-4 x 1/(8p+1)^1.5 with p = (D - 357M) / 243M, reaching 1.11e-05 at 600M (the metrics show 3 significant
digits). After any restart, the trainer log's `policy rate at the resume` must equal check_restart's for that
checkpoint. A checkpoint saved at a stop counts the decisions taken in after the last update (the rehearsal's
update 16 saved D = 596,489, about 4,000 more than its last update used), so each restart moves the schedule
earlier by a few thousand decisions; at 600M that does not matter.

## 6. Milestones and the matched evaluations

All matched evaluations run on **box c**, so every Foul Play number compared with another comes from the same Foul
Play build (8f541f8, poke-engine 0.0.48 built on that box) and the same load. T's checkpoints reach S3 within 10
minutes of being written, so box a is never touched for them.

- T: its first checkpoint at or after the milestone (every 200 updates, so up to about 7.4M past it). Its decisions:
  its line in T's metrics.jsonl (`s3://.../main-g6e12/runs/v2/topstart/metrics.jsonl`).
- S: the checkpoint nearest to T's decisions (S has one every 50 updates, 1.87M apart). On box c:

```
$C 'python3 - 29.9e6 <<"P"
import json, os, sys
d = float(sys.argv[1]); S = os.path.expanduser("~/dev/topstart/ckpt/S")
u = {r["update"]: r["total_steps"] for r in map(json.loads, open(f"{S}/S_metrics.jsonl")) if r.get("phase") == "ppo" and r.get("total_steps", 0) > 0}
have = {int(n[15:21]) for n in open(f"{S}/S_files.txt").read().split()}
b = min((k for k in have if k in u), key=lambda k: abs(u[k] - d))
print(f"S learner_update_{b:06d}.pt at {u[b] / 1e6:.2f}M")
P'
```

  (the number is T's decisions; for example 25.0M gives update 700 at 25.83M, 104.7M gives 2,850 at 105.19M). Updates
  700, 2,700, 4,050 and 8,100 are already in `~/dev/topstart/ckpt/S/`; another one comes from
  `s3://<your-bucket>/main-g6e12/runs/v2/scratch/checkpoints/`.
- Both with `REL=~/dev/topstart/rel ALLOW_PORT=1 bash ~/dev/topstart/rel/cloud/v2/topstart_eval.sh NAME CKPT PORT 0`,
  NAME `T<M>` and `S<M>`, two at a time on ports 11694 to 11699 (Foul Play 400 games, simple and psbot 200).

**What T against S measures.** S ran its first 60.9M PPO decisions without the moveset block (widened at update
1,650) and had the PPO-time top-player imitation term from 356.9M to 374.9M. So B3 and B4 measure the start and the
moveset block together; B5 is the first that reads the start alone (`topstart_setup.py --milestones` prints this).

## 7. The success bars (fixed before any training)

FP = Foul Play at a fixed 88,000 iterations. A 400-game rate has a 95% interval of about +-4.9 points and a
difference of two such rates about +-6.9, so a gap under 7 points is a reading, not proof.

| Bar | When | Pass | Expected from the evidence |
|---|---|---|---|
| B1 | end of offline imitation, `measure` on T0 | capped top-1 >= 0.55; >= P0's + 5 points and >= SFIN's + 2 (on this set P0 0.4576 and SFIN 0.5151, so 0.5076 and 0.5351); test_game within 1.5 points of test_player; train top-1 at most 3 points above held-out. Abort: below 0.30 at step 40k | pass |
| B2 = G1 | 0 PPO decisions | FP(T0) >= FP(P0) and capped top-1 >= 0.50, else no launch | pass: P0 about 20 to 25%, T0 about 30 to 42% (arm C 41.1%) |
| B3 | 25M | FP(T) >= FP(S) + 5 (start plus moveset block) | pass |
| B4 | 100M | FP(T) >= FP(S) + 3; if not, tell the owner at once (the head start is gone, arm C's pattern); default: continue | doubtful |
| B5 | 150M | stop rule: FP(T) <= FP(S) - 7 means the start hurts: stop (section 9) and report | not fired |
| B6 | 300M | FP(T) >= FP(S), and ladder-pick B-T300 against M24381 >= 40.5% | doubtful |
| B7 | 600M | T better than S: A-T600 >= 61.0% (SFIN's A 56.0% + 5) and B-T600 against SFIN >= 55%. Caught up: B-T600 against M24381 >= 50% and A-T600 >= 60% (M24381's A 60.2%; SFIN's B 48.8%). Ladder only if B-T600-M24381 >= 50%, after the C cells with search | T600 about equal to SFIN, under M24381 |

The A and B cells run in the ladder-pick harness exactly as `~/dev/ladderpick/FINAL.md` (branch ladder-pick 96d576c,
`ladderpick.py snew --snew <name>`, `launch_snew.sh`), after `export_v2.py` puts the checkpoint in S3 `export/`; on
box d (`~/dev/ladderpick`, ports 11500-11599, leave 13 cores) or on box a once T has finished. C cells only if
A >= 58% or B-T-M24381 >= 48%. scratchx's final network (800M at the floor rate) is not a bar; a B cell against it
can be added beside B7 for information.

If B4 shows no lead at 100M, the remaining 500M most likely reproduce S, and the owner may prefer to free box a
then (the default is to continue). The next levers are elsewhere: PPO length and learning rate, the opponent pool,
value calibration (S lost 9 of 18 ladder games after its own win estimate passed 70%).

## 8. The rehearsal and its check (2026-10-18)

The rehearsal ran on box a with release 433cb30 (the Python of this commit), all of it stopped and left in place:
`~/dev/topstart/rehearsal` (offline) and `~/runs/v2/topstart-rehearsal` (the run), both in S3. At 08:00 to 08:30 every
number below was read again from the files: on box a by reading only, and on box c from S3 copies whose sha256
equal box a's files (12 of 12 checked).

**Offline imitation** (real shards, GPU 1): 8,000 steps (anneal from 6,000), a stop after 2 groups (1,695 steps)
and a resume. No non-finite step; the data wait 8.5 s twice (the first group of each run), then 0.0; 14.9 to 15.0
steps a second.

| Step | 1,000 | 2,000 | 3,000 | 4,000 | 5,000 | 6,000 | 7,000 | 8,000 |
|---|---|---|---|---|---|---|---|---|
| capped held-out top-1 (32,785 decisions, 226 players) | 0.231 | 0.235 | 0.336 | 0.352 | 0.389 | 0.407 | 0.430 | 0.445 |
| value loss | 2.48 | 2.03 | 1.88 | 1.74 | 1.54 | 1.53 | 1.41 | 1.29 |

`measure` on the start R0: capped 0.4447, all of test_player 0.4454 (784,253 decisions), test_game 0.4409; by set
0.449 (40k) and 0.442 (bulk); by period 0.442 (2025), 0.442 (2026-01..06), 0.449 (2026-07+). SetHead on test_player:
ability 0.717, item 0.722, moves 0.769, Tera 0.488. For reference on the capped set: P0 0.4576, SFIN 0.5151.

**The run at the real size** (36 servers, 2,304 battles, 3 GPU inference actors, every v2 technique, v2-fast on;
ports 11700-11735; warmup min 100k, max 200k; PPO to 745k decisions with hold 300k and total 600k; a stop and
restart at PPO update 16):

| | |
|---|---|
| Timeline | servers up 07:15:45; warmup updates 1 to 20 07:16:42 to 07:18:35; PPO update 1 at 07:20:17; stop at update 16 (07:21:37, saved in 26 s); check_restart exit 0; update 17 at 07:23:40; update 33 and done at 07:25:40; pipeline.py exit 0 |
| Handover start -> warmup | `PHASE CHANGE DETECTED (imitation -> warmup)`; the learner's counter from 0 (warmup update 5: 184,777 decisions, scheduler step 480 = 5 x 96); warmup final (update 20, 739,797 decisions, step 1,920): 104 of 106 tensors equal to the start, the 2 that differ are `v_head.*`; set_head and pi_head equal |
| Handover warmup -> PPO | `PHASE CHANGE DETECTED (warmup -> ppo)`; a fresh optimizer and scheduler (update 10: 368,935 decisions, scheduler step 960 = 10 x 96); three parameter groups at 3e-5, 3e-4 and 6e-4 (the zone multipliers kept); opp_action_head new in PPO; JEPA predictor unchanged; set_head drift 0.018 at update 10 and 0.022 at update 33 (its weights reach 0.10 to 0.22) |
| Learning rate in decisions | the rate of every update equals 3e-4 x warm(step) x share(D) to the metrics' 3 digits: 2.91e-05 at update 1, 2.31e-04 at update 8 (D about 295k, under H), 1.03e-04 at update 9 (D about 332k, the first update past H, inside [H, H + 36,864)), 6.03e-05 at update 10 (the checkpoint's own rate 6.0293e-05, equal to the formula), 1.11e-05 = 3e-4/27 from update 17 (the first with D past 600k) to 33 |
| Restart mid-anneal | check_restart: "policy rate at the resume: 1.129e-05 ... the checkpoint was written at 1.129e-05 ... OK to start", exit 0; the trainer printed the same rate at 596,489 decisions and optimizer step 1,536; the next update ran at the floor, as scheduled |
| Warmup | vloss 13.58, 12.10, 8.78, 4.93, 2.89 (update 5), 2.45 (10), 2.39 (19); learner 3.5 s an update; 5.9 s an update, 6,580 decisions/s (rollout-bound: the value-only learner waits for games) |
| PPO | vloss 2.32 to 2.42; KL 0.002 to 0.012; clip 0.14 to 0.39 at full rate, 0.08 to 0.12 at the floor; entropy 1.43 to 0.87; n_mb 96 every update; opp_acc 0.34 to 0.96; set_loss 0.62 to 0.64; boosted share 0.22 to 0.29; read bonus on (b 0.3) |
| Throughput | learner `sec` 5.3 to 6.1 a 96-minibatch update (S's last hours: 8.7 to 9.9); between restarts 6.1 to 6.7 s an update, 5,500 to 6,200 decisions/s (updates 10 to 16: 6,224/s; 20 to 33: 5,711/s); with each segment's start-up counted, 6.8 s and 7.2 s an update |
| Priorities | every start: `rollout nice 5: 36 of 36 workers at nice 5 (2052 threads, 2052 raised, 0 could not be)`, learner and inference actors at nice 0; servers behind `nice -n 5` |
| Faults | no NaN or non-finite value in any model, optimizer or target of the 10 files checked; no Traceback or error line in the 3 trainer logs; 5 refused choices ("can't switch to an active Pokémon" with Ditto in play), which S's logs show too (118 to 179 a 6-hour segment); the pipeline's one WARNING is the expected "no state.json but 1 checkpoints already exist" |
| Stopped | the unit stopped and disabled, no process, port or GPU memory left; outputs in `s3://<your-bucket>/topstart/rehearsal/` and the backup |

**check_restart against copies** (box c `~/dev/topstart-integ/chk`, the copies with repo_dir set to a box c replica
of the release): 12 of 12 cases gave the expected exit code.

| Case | Exit |
|---|---|
| real config (as written by the setup, R0 as the start) `--phase warmup`; `--phase ppo` from the start | 0, 0 ("phase 'imitation' and the start is 'warmup'/'ppo'") |
| rehearsal: PPO from the warmup's final checkpoint | 0 ("phase 'warmup' and the start is 'ppo'") |
| rehearsal resumes at update 10 (6.029e-05), 16 (1.129e-05), 33 (1.111e-05, "the anneal ended") | 0, 0, 0 |
| hold moved to 350k at update 10 (the rate x2.35) | 3; 0 with `--allow-rate-change` |
| hold moved at update 33 (same rate, another schedule) | 3 |
| S's step schedule against a decisions checkpoint (the rate x26.6) | 3 |
| a warmup resume at update 15; the run's own state.json ("done") | 0, 0 |

**The release check.** On box c the setup rebuilt the real run.json from a copy of S's run.json (sha256 bde91ae2...):
identical to the rehearsal's `dryT` on box a but for repo_dir. Box c's 300-step smoke of `topstart.py train` on the
full data and a smoke of `topstart_eval.sh` (Foul Play, simple, psbot) ran clean; `measure` on P0 there equals box a's
to the last digit.

**Tests on box c** (`~/dev/topstart-integ`, a clone of 433cb30 with the heuristic cache, the speed workflow's
references): 1,163 tests, 1,103 passed, 30 skipped, 29 failed, 1 error. Against v2-fast's own run, 10 outcomes differ:
7 skipped to passed (the heuristic cache is present), the real-Ray nice test skipped (the suite runs at nice 10; it
passed at nice 0), and `test_pipeline.py::test_e2e_the_state_is_refreshed_while_the_servers_take_their_time` failed
once under the full suite's load and passed alone 6 times of 6 (pipeline.py and that test are byte for byte
v2-fast's). The other 28 failures and the error are the known ones that also fail on 4cfd496 and v2-fast
(test_worker_* against release r1, test_tera's compiler error, test_speed_obs, test_obs_matchup, test_ppo_health's
collection). test_topstart (41), test_lr_decisions with test_topstart_release (30), test_fast_* (35),
test_human_imitation (31), test_extras (14), test_train_v2 (22): all pass. This commit changes no code.

### What the rehearsal found

1. **The offline value head predicts a different distribution from the one PPO trains.** The start's head was
   trained on the Monte Carlo return G = gamma^(T-1-t) z - 0.5 Phi, which has two modes (win or loss); the learner
   trains on GAE lambda-returns near the mean. So each head scores badly on the other's targets: the start's head
   13.6 at warmup update 1 on self-play; on the human targets S's final head 10.0, P0's 8.9, the warmup's after 20
   updates 7.7. Warmup fixes it as it is meant to (under 3 by update 5, 2.39 by update 19; S's warmup went 3.24 to
   2.28 in 14 updates). Only `v_head` trains in warmup, so the policy and the SetHead reach PPO untouched.
2. **The supervisor sees no decisions for about 2 min 45 s after each trainer start** (S's and scratchx's logs show
   the same). A short phase overshoots: the rehearsal's warmup ran 703k decisions against a 200k cap. For T this
   means a warmup of 0.9M to 1.5M decisions instead of S's 0.51M. It does not matter at 600M.
3. **After a resume, `topstart.py` writes the next resume file one group early** (resume_001695, then
   resume_002507). One extra 196 MB file per resume; harmless.
4. **A copied start is more random than PSBot's:** entropy 1.43 at PPO update 1 against S's 1.02, and no KL early
   stop in the first 33 updates (KL under the 0.02 target). 33 PPO updates cost 1.4 points of top-player agreement
   (capped 0.445 to 0.431).
5. **A stop's checkpoint counts decisions its last update did not use** (section 5, the last paragraph).

## 9. The undo

Nothing here deletes anything. Every step leaves the files where they are.

- **Before the start:** nothing to undo. Box c's `~/dev/topstart`, the run dir and the units (not enabled) can stay.
- **Stop T** (B5's stop rule, a failed gate after the fact, or the owner's call), guard first (the guard starts the
  evaluator of an arm in ARMS again within 5 minutes): take `topstart` out of ARMS
  (`$A 'sed -i "s/^ARMS=.*/ARMS=\"\"/" ~/runs/v2/guard.env'`, or `ARMS="scratchx"` while scratchx's final evaluation
  has not written final.done; `guard.env.pre-topstart` is the file as it was), then
  `$A 'sudo systemctl disable --now psppo-v2-topstart-train psppo-v2-topstart-eval'`. KEEP_BOX stays.
  The supervisor stops the trainer through its stop file, which saves a checkpoint first (26 s in the rehearsal). To
  resume later: `topstart` in ARMS again and enable the train unit; it continues from the newest checkpoint.
- **A wrong start file** (before PPO): stop as above, move `~/runs/v2/topstart` aside
  (`mv ~/runs/v2/topstart ~/runs/v2/topstart.aside-$(date -u +%m%d%H%M)`), and run 3.5 again with the right start.
  The setup refuses a directory with a state.json, so it cannot overwrite one.
- **Drop v2-fast** (one flag misbehaves, SPEED.md section 5 lists when): stop the train unit, keep the old file as
  `run.json.pre-nofast`, take the flag out of `~/runs/v2/topstart/run.json` (`--fast-update` or `--fast-ingest` from
  `phases.ppo.extra_train_args`; for `--rollout-nice`, also `servers.command` and the `--rollout-nice 5` pair in
  `train_defaults`), run check_restart without `--phase` (exit 0), start the unit. Without `--rollout-nice`, renice
  the learner and the inference actors by hand after every trainer start (cloud/v2/RECOVERY.md). Checkpoints are the
  same with or without the flags.
- **The learning-rate schedule:** never edit `--lr-hold-decisions` or `--lr-total-decisions` in a running run
  without check_restart (it exits 3 when the rate or the schedule would move; `--allow-rate-change` only on purpose).
- **The disk move:** while the box has not stopped, `cd /opt/dlami/nvme/aside/scratch-checkpoints && for f in *.pt; do
  test -e ~/runs/v2/scratch/checkpoints/$f || mv $f ~/runs/v2/scratch/checkpoints/; done` brings S's files back (it
  needs the 70 GB free again); after a stop the instance store is empty and they come from
  `s3://<your-bucket>/main-g6e12/runs/v2/scratch/checkpoints/`.
- **The release:** `~/dev/topstart/rel-before/RELEASE.b79370b` records what 3.2 replaced; the Python files do not
  change, only this note.
- **S, scratchx, the data and the rehearsal:** nothing of them is written by this plan except the moves of D1. The
  rehearsal's `~/runs/v2/topstart-rehearsal` (1.4 GB, unit stopped and disabled) and `~/dev/topstart/rehearsal`
  (1.1 GB) stay as the record; both are in S3.
