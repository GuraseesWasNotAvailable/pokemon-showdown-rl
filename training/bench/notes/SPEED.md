# v2-fast: the same training, a faster learner (branch v2-fast)

v2-fast is v2-humanaux 4cfd496 with three speed flags merged in, all off by default:
`--fast-update`, `--fast-ingest` and `--rollout-nice N`. With all three on, every update is the
same as 4cfd496's bit for bit (weights, fp32 masters, Adam moments, losses, KL stops, pushes and
checkpoints), and the learner's update cycle takes 24% less time at 96 minibatches and 35% less at
about 50, measured under the box-a emulation from the run's final checkpoint. On box a that is
about **5,200 to 5,400 learned decisions/s against the 4,000/s of the run's last hours (+30 to
+34%)**, still under the about 6,150/s box a's games can supply. Box c gets no job in the live run
from this round (section 6).

The run itself reached its 600M target at 02:14 UTC on 2026-10-18 (final checkpoint
`learner_update_016041.pt`, 600,026,902 decisions) and is idle. So this release is for the next
start of a PPO phase on box a: an extended target or the next run (section 5).

## 1. What changed

| flag | what it does | files | training math |
|---|---|---|---|
| `--fast-update` | ppo_update's minibatch loop without GPU syncs: the figures summed on the GPU, the KL early stop read behind a CUDA event, the all-zero action mask fixed up without reading the GPU, q/k/v by `unbind`, foreach copies of the fp32 masters | ppo_core.py, learner.py, train.py | none (same bits) |
| `--fast-ingest` | each worker message's observations go to the GPU as they arrive (a copy thread), so an update no longer joins about 200 CPU chunks (0.6 GB) and copies them again; the advantages of an update's messages computed at once (same float32 operations); the weight push on its own thread, its bytes to Ray out of band | fast_ingest.py (new), learner.py, train.py | none (same bits) |
| `--rollout-nice N` | Ray starts its workers at nice 0 instead of 15, so the learner and the inference actors run at 0; every rollout worker (and any process it starts) raises itself to N. The Showdown servers go to N by run.json (`servers.command` with `nice -n N`) | worker.py, train.py | none (scheduling only) |

None of them adds anything to RunConfig: a checkpoint's `run_cfg` is the same with or without them,
and every flag can be dropped at any restart. Without the flags the trainer is 4cfd496's (tested
below). What changes that you can see: the "Zero mask detected" warning becomes one count per
update (`--fast-update`); the inference actors get each new set of weights up to about 0.2 s
earlier or later, inside the 1 s sync interval (`--fast-ingest`); GPU 0 holds a 1.06 GB
observation buffer (`--fast-ingest`; the peak rises 0.35 GB because the join it replaces also
lived on the GPU); the trainer log has a few new lines (section 5).

## 2. The merge

| item | branch, final commit | review | merged |
|---|---|---|---|
| syncfree (`--fast-update`) | v2-fast-syncfree 4f57d36 | no blocker or major; 5 minor findings, all fixed in 4f57d36 | yes, 9ba7057 |
| nice (`--rollout-nice`) | v2-fast-nice 52cc4b3 | no blocker or major; 1 minor (helpers of PSBot and Foul Play seats left at nice 0), fixed in fb310c3 | yes, 279d72c |
| ingest (`--fast-ingest`) | v2-fast-ingest b7ed0dd | no correctness defect; 2 minor (the 50% outside-time target, the fixed-cost figures), fixed in b7ed0dd | yes, c855b74 |

All three passed, so none was left out. Two conflicts, both mechanical:
- train.py `parse_args`: both argument groups are added (learner speed, then CPU priorities).
- learner.py `LearnerActor.__init__`: both keywords (`fast_update`, `fast_ingest`) and both blocks.

After the merge, `ppo_core.py` is v2-fast-syncfree's, `worker.py` is v2-fast-nice's and
`fast_ingest.py` is v2-fast-ingest's, byte for byte. Commits on top:
- 3ecb5e8: `tests/test_fast_all.py` (the three flags together; both learner flags on one update
  against 4cfd496), and `--fast-update` in `bench/fast_ingest_bench.py` (check, digest, time).
- c98d834: `bench/rollout_nice_full.sh` takes `EXTRA` train.py arguments (every flag on for the A/B).
- b5eff62: `tests/test_human_imitation.py` held that `human_imitation` was the learner's last
  parameter, which the merge makes false. It now holds what that was for: `human_imitation` in its
  place with default None, and every keyword after it off by default. No runtime code changed.

## 3. Same training: the tests and checks (box c, 2026-10-18)

**Every test file, both trees, the same environment** (`~/dev/fast/int/tests2.sh`, junit XML
compared test by test with `junit_compare.py`):

| tree | tests | passed | skipped | failed | error |
|---|---|---|---|---|---|
| v2-fast b5eff62 | 1,092 | 1,026 | 37 | 28 | 1 |
| 4cfd496 | 1,008 | 942 | 37 | 28 | 1 |

- Of the 1,008 tests both trees have, **no outcome differs**. The 84 new tests (test_fast_update,
  test_fast_all, test_fast_ingest, test_rollout_nice) all pass.
- The 29 that fail or error fail the same way on 4cfd496, for reasons outside this work:
  test_worker_memory, test_worker_handover and test_worker_choice_storm compare against release r1
  (8227671), whose observation predates the matchup and moveset blocks; test_obs_matchup and
  test_speed_obs compare against older observation code; test_tera's inductor C++ build fails on
  box c's compiler; test_ppo_health imports `bench/check_history.py`, which this line does not have.
- Skips are the usual ones: old-code checkpoints not given (`PSPPO_OLD_CKPT`, the widen and
  moveset checkpoints), Windows-only paths, the heuristic cache that only the release has.
- References used: 4cfd496 for test_fast_update and test_fast_all, d6c572d for test_extras,
  dabca7b for test_train_v2, the commit before the human port for test_human_imitation, the live
  `pipeline.py` and `run.json` (md5 equal to box a's) for test_rollout_nice, the real human shards,
  recorded streams, and real Ray (`PSPPO_RAY_TESTS=1`). test_pipeline's servers on 11400-11401.

**The live-sized checks, from the final live checkpoint** (`learner_update_016041.pt`, md5 equal
to box a's), on 2,430 worker messages recorded from it by a flag-off train.py run on box c, cut to
box a's message size (184 rows), 5 queued at once, deterministic kernels:
- **No flag against both learner flags** (`fast_ingest_bench.py check --fast-update`, the
  learner's own loop, 3 updates of 96, 96 and 98 minibatches): every column handed to each update
  equal, the observations from the GPU buffer every time, stats, n_mb, update digests, final
  digests, the weight-store pushes in order, temperatures, counters, the leftover rows and the
  checkpoints written after: all equal. CHECK PASSED.
- **4cfd496's own tree against v2-fast with both learner flags** (`digest`): the same digests,
  stats and n_mb at all 3 updates (final weights a0a56d722a972295 in both), the same pushes; the
  checkpoints after 3 updates have no differing entry (the checkpoint directory in run_cfg left out).
- **Cross-loading**: 4cfd496 resuming v2-fast's checkpoint and v2-fast resuming 4cfd496's give the
  same next update (52c948938bf404d6). XLOAD PASSED.
- **The KL early stop** with both learner flags: test_fast_all holds a stop at the first minibatch
  of the second epoch, and test_fast_update every stop the update's own KLs allow (each item's
  live-sized forced stop is in its own report).

**cloud/check_restart.py from the v2-fast release** (a box c replica of `~/rel/v2-fast`, built
exactly as section 5 builds it: box a's `~/rel/v2-humanaux` copied, the v2-fast archive over it),
on copies of the live run.json, of state.json and of the final checkpoint 016041:
- the deploy edit (repo_dir, servers.command, `--fast-update --fast-ingest --rollout-nice 5`):
  exit 0, "OK to start: the config is valid, train.py takes the command line, and no learning
  rate moves at the resume." (policy rate 1.160e-05 at the resume, as written);
- the control (repo_dir only): exit 0, the same report but for the three arguments;
- the same edit checked by 4cfd496's release: exit 2, "DO NOT START" (it does not know the flags),
  so a run.json edit without the new release cannot slip through;
- the live supervisor (`~/rel/v2-movesets/pipeline.py`, copied) builds the server command as
  `nice -n 5 /home/ubuntu/venv/bin/python start_servers.py --dir /home/ubuntu/pokemon-showdown --ports 10600-10635 --heap-mb 2048 --log-dir /home/ubuntu/runs/v2/scratch/logs/servers`
  and the trainer command with the three flags at its end;
- with the run's real final state.json (phase done) the check says "the run is in phase 'done':
  nothing would be started", which is the run's state, not the release's (section 5).

## 4. Measured gains

### Learner seconds per update (the A/B that decides deployment)

`bench/fast_ingest_bench.py time`: the learner's own loop and `_perform_update` over the recorded
messages (184 rows each), the real Ray WeightStore, the live learner flags (top-player term off,
as the live run.json), the final live checkpoint, every round from the same state; each regime
3 updates x 2 reps. 4cfd496's own tree against v2-fast with `--fast-ingest --fast-update`. All runs
in one hold of the bench lock. Box-a emulation: this process on vCPUs 0-1,
11 memory, 11 interpreter and 2 bandwidth hogs on 2-15 (2 rounds, order rotated); quiet: nothing
else.

| box-a emulation (24 updates a side) | 4cfd496 | v2-fast, all flags | change |
|---|---|---|---|
| update, about 50 minibatches (KL stop at the start of epoch 2) | 5.37-5.56 s | 3.69 s | -31 to -34% |
| cycle (update plus taking the messages in), about 50 | 5.75-5.96 s | 3.78-3.79 s | -34 to -36% |
| update, 96 minibatches (no stop, the run's last hours) | 9.53 s | 7.32 s | -23% |
| cycle, 96 | 9.86 s | 7.44 s | -24.5% |
| ppo_update per minibatch | 90-98 ms | 70-76 ms | about -22% |
| fixed parts per update (join, copy, push) | 0.80 s | 0.15 s | -0.65 s |
| learner-thread time outside the update | 0.34 s | 0.13 s | -63% |
| peak GPU memory reserved | 4.14 GB | 4.49 GB | +0.35 GB |

| quiet box c (12 updates a side) | 4cfd496 | v2-fast, all flags | change |
|---|---|---|---|
| cycle, about 50 minibatches | 3.80-3.81 s | 2.94-2.95 s | -22.5% |
| cycle, 96 | 6.98 s | 5.64 s | -19% |
| fixed parts per update | 0.28 s | 0.08 s | -0.21 s |

No update was slow (more than twice its side's median fixed cost) in any run; no buffer fallback;
the push went out of band every time.

**On box a.** The run's last hours: 96-minibatch updates, 9.2-9.4 s a cycle, about 4,000 learned
decisions/s, learner bound. The emulation's 4cfd496 cycle at 96 (9.68 and 9.88 s in the two
rounds) is within 5% of that. Applying the emulation's ratios (0.745 and 0.769) gives a cycle of
about 7.0-7.2 s and **about 5,200-5,400 decisions/s (+30 to +34%)**; the quiet box's ratio (0.81)
is a floor of about 4,950/s (+24%). Per component: `--fast-update`'s box-a calibration (live
learner probe, 91 to 72-77 ms a minibatch) saves 1.35-1.8 s an update and `--fast-ingest` about
0.85 s, which lands at the same range. With a higher learning rate (a new run, KL stops near
minibatch 50) the cycle falls by about a third. Box a's games supply about 6,150 learnable
decisions/s, so the run stays learner bound after this, with about 12-15% of the learnable decisions
still thrown away instead of 35%.

### Box-c-sized rollout at full load

`bench/rollout_nice_full.sh` (with `EXTRA` for the two learner flags): 12 Showdown servers on
11410-11421, 768 battles, the learner and 2 inference actors on box c's one L40S, the live flags
(top-player term off), the final live checkpoint, a 300 s window once the workers are up and two
updates are done. Three arms: 4cfd496 as a plain restart starts it (Ray's default, nice 15),
4cfd496 with box a's hand renice (the learner and the inference actors at 0: box a today), and
v2-fast with every flag on (the servers behind `nice -n 5`). Three rounds, order rotated, each arm
under the bench lock. Two windows were disturbed by another session's unlocked jobs on box c (the
round-2 hand-renice arm lost 1.6 cores to them; in the round-3 Ray-default arm `topstart.py
measure` ran, a core each, during the window). They are left out below and kept on box c.

| arm (clean windows) | learner s per 96-minibatch update | s per minibatch | inferred decisions/s | learned decisions/s |
|---|---|---|---|---|
| 4cfd496, Ray default (a plain restart) | 12.1-12.3 | 0.126-0.128 | 2,396-2,519 | 2,261-2,333 |
| 4cfd496, hand renice (box a today) | 10.6 | 0.110-0.111 | 2,414-2,415 | 2,205-2,221 |
| v2-fast, every flag on | 7.7 | 0.080 | 2,292-2,409 | 2,116-2,221 |

- With real games competing for the CPU, the learner's update is **27% shorter than box a's present
  state and 37% shorter than a plain restart's**. Every update in every window had 96 minibatches.
- Box c at this size is rollout bound: the learner waits for data between updates (half a core in
  the v2-fast arm), so learned decisions/s follow the games here, not the learner. v2-fast infers
  0.3% (round 3, same round) to 1.4% (round 1) fewer decisions/s than the hand renice, about 2% on
  the mean with round 2's lower window (its box also had 0.7 cores unaccounted). That is the nice
  item's known cost: workers and servers both at nice 5 give the sims a little less CPU than nice 0
  did. It costs nothing on box a while its learner is the limit; it trims box a's spare game
  capacity from about 6,150 to roughly 6,000/s, still above the 5,200-5,400/s learner.
- KL 0.0023-0.0025, entropy 0.37-0.39, value loss 2.08-2.13 in every arm (the same math from the
  same checkpoint; only the games differ).
- Nice levels with the flags: all 54 learner and 94 inference threads at 0; all 600 worker
  threads, the server launcher and all 859 server and sim threads at 5; "rollout nice 5: 12 of 12
  workers at nice 5 (588 threads, 588 raised, 0 could not be)".
- GPU memory on the shared L40S: 7.7-7.8 GB for v2-fast against 7.4-8.1 GB for 4cfd496.

**The decision.** Deploy: the updates are 4cfd496's bit for bit, the learner's cycle is 19 to 36%
shorter in every setting measured (24 to 36% emulated, 19 to 22% quiet, 27% for the update under
real games against box a's present state), and the only cost
found, 0.3 to 2% of game throughput, falls on the part of box a that has room to spare.

Raw results on box c: `~/dev/fast/int/ab/` (`time/*.json`, `time/compare_*.json`,
`full/*.log` with `full/full_*/`, `check_all.json`, `xload/`), `~/dev/fast/int/out_ab.log` and
`out_ab_r3.log`.

## 5. Deploying on box a

**When.** The run is done and idle (02:14 UTC, phase done, `on_done: idle`), so nothing is lost by
switching now, but nothing will train either: the release takes effect at the next start of a
PPO phase. That is either (a) the owner extending this run (reopening the PPO phase in state.json
and choosing the learning rate: the schedule's `lr_total_steps` 1,113,566 ends about 20,000
optimizer steps after the final checkpoint's 1,093,463) or (b) the next run. Both are the owner's
calls and not part of this work. The steps below are the same in both cases; for a new run, make
the same three edits in its run.json. In a run that is training, the switch costs one restart
(the trainer saves a final checkpoint on stop; the servers and the compile take a few minutes).

```
A=$(bash C:/Users/you/ps-ppo-v2/cloud/v2/boxssh.sh a)      # the ssh command
# 1. the release: v2-humanaux's files (heuristic cache included) with v2-fast over them; nothing running is touched
$A 'test ! -e ~/rel/v2-fast && cp -a ~/rel/v2-humanaux ~/rel/v2-fast'
git -C C:/Users/you/ps-ppo-v2-fast archive --format=tar v2-fast | $A 'tar -x -C ~/rel/v2-fast && echo "v2-fast $(date -u +%FT%TZ)" > ~/rel/v2-fast/DEPLOYED'
$A 'cd ~/rel && diff -rq v2-humanaux v2-fast | grep -v __pycache__'
#    expect: learner.py, ppo_core.py, train.py, worker.py, tests/test_human_imitation.py differ; only in v2-fast:
#    DEPLOYED, fast_ingest.py, bench/{fast_ingest_bench.py,fast_update_digest.py,learner_sched_probe.py,rollout_nice_full.sh},
#    bench/notes/SPEED.md, tests/{test_fast_all,test_fast_ingest,test_fast_update,test_rollout_nice}.py
# 2. run.json: repo_dir, the server command behind nice 5, three arguments of the PPO phase; the old file kept
$A 'cd ~/runs/v2/scratch && cp -p run.json run.json.pre-fast && ~/venv/bin/python -c "
import json; p=\"run.json\"; r=json.load(open(p))
r[\"repo_dir\"]=\"/home/ubuntu/rel/v2-fast\"
r[\"servers\"][\"command\"]=[\"nice\",\"-n\",\"5\",\"{python}\",\"start_servers.py\",\"--dir\",\"{showdown_dir}\",\"--ports\",\"{ports}\",\"--heap-mb\",\"{heap_mb}\",\"--log-dir\",\"{server_log_dir}\"]
r[\"phases\"][\"ppo\"][\"extra_train_args\"]+=[\"--fast-update\",\"--fast-ingest\",\"--rollout-nice\",\"5\"]
json.dump(r,open(p,\"w\"),indent=1)"'
# 3. the checks: check_restart must end "OK to start" (on a run in a PPO phase; on the done run it says
#    "nothing would be started"), and the live supervisor must build the server command behind nice -n 5
$A '~/venv/bin/python ~/rel/v2-fast/cloud/check_restart.py --config ~/runs/v2/scratch/run.json --run-dir ~/runs/v2/scratch'
$A 'cd ~/rel/v2-movesets && ~/venv/bin/python -c "import pipeline as p; c=p.load_config(\"/home/ubuntu/runs/v2/scratch/run.json\"); print(\" \".join(p.build_server_command(c, p.Paths(\"/home/ubuntu/runs/v2/scratch\"), \"ppo\")))"'
#    expect: nice -n 5 /home/ubuntu/venv/bin/python start_servers.py --dir /home/ubuntu/pokemon-showdown --ports 10600-10635 ...
# 4. apply (the supervisor reads run.json at its start): stop, keep the start point, start
$A 'sudo systemctl stop psppo-v2-scratch-train && mkdir -p ~/runs/v2/scratch/pre_fast && cp -p ~/runs/v2/scratch/checkpoints/$(ls -t ~/runs/v2/scratch/checkpoints | head -1) ~/runs/v2/scratch/pre_fast/ && sudo systemctl start psppo-v2-scratch-train'
```

No box c unit is needed. The evaluator and the guard units are unchanged. The hand renice of the
learner and the inference actors is no longer needed after this restart (it does no harm if
someone still does it).

**The first hour.** In the newest `~/runs/v2/scratch/logs/trainer_*.log`, at the start:
- `learner speed: fast_update=True` and `learner ingest: fast_ingest=True`, then
  `[learner] fast_ingest on: observations to the GPU on arrival, ...`;
- `CPU priorities: Ray's workers start at nice 0 (the learner, the inference actors); every rollout
  worker puts its threads at nice 5. ...`, then `rollout nice 5: 36 of 36 workers at nice 5 (...,
  0 could not be)` and `CPU priorities: learner pid ... nice 0; inference pids [...] nice [0, 0, 0]`.

Then, every few minutes:
- The nice of every thread, per group (counts of threads at each nice):
  ```
  $A 'for g in LearnerActor InferenceActor RolloutWorker OpponentSeat; do echo "$g: $(for p in $(pgrep -f "^ray::$g"); do ps -L -o ni= -p $p; done | sort -n | uniq -c | tr -s " " | tr "\n" ";")"; done; echo "servers: $(for p in $(pgrep -f "^(/usr/bin/)?node |^[^ ]*python[^ ]* start_servers.py"); do ps -L -o ni= -p $p; done | sort -n | uniq -c | tr -s " " | tr "\n" ";")"'
  ```
  With the flag: only 0 for LearnerActor and InferenceActor, only 5 for RolloutWorker, OpponentSeat
  and the servers. (Without it, as before the switch, each Ray actor shows its Python threads at
  15 and Ray's own threads at 0, and the servers at 0.)
- The update lines (`[learner] upd=... n_mb=96 sec=`): sec about 7 instead of 8.7-9.4 at 96
  minibatches. pipeline.log's "N/s over the last 10m": about 5,200 or more instead of about 4,000
  (at the same learning rate and minibatch count).
- KL, entropy, value loss and clip in the band of the hour before the switch (the math is the same;
  only the data are fresher). `failed_updates=0`, no NaN.
- GPU 0 memory about 1.1 GB above the 11.6 GB before (the observation buffer).
- None of these trainer-log lines: `fast_ingest: update N joins its observations on the CPU`,
  `weight push of version N failed`, `could not take version N out of band`,
  `waited N s for M observation copies`, `the weight push before is still in flight`. One of them
  once is harmless (that update ran the old path, or the push fell back to sending the tensors);
  every update or every few is a reason to turn the flag off.

**Turn a flag off when** (each flag stands alone; drop the one that misbehaves, or all three):
- `--fast-ingest`: the lines above repeat, a CUDA out-of-memory on GPU 0, or the telemetry's
  `INFER: {... 'model_version': N}` stops following the learner's `upd=`.
- `--fast-update`: an error inside an update, a NaN loss, or KL stops at different minibatch counts
  than the hour before at the same rate (it should not happen: the stop is the same test on the
  same numbers).
- `--rollout-nice` (with servers.command): the start line says workers "could not be" set, the
  servers do not come up, or the inferred decisions/s fall more than 5% below the hour before
  while the learner keeps up.
- Any of them: learned decisions/s over an hour below the hour before the switch, or a trainer
  restart loop that did not happen before.

**The full undo** (the release before, from the newest checkpoint; checkpoints written with the
flags load in 4cfd496 and give the same next update, so none needs moving):

```
$A 'cp -p ~/runs/v2/scratch/run.json.pre-fast ~/runs/v2/scratch/run.json && sudo systemctl restart psppo-v2-scratch-train'
```

To drop one flag only, remove its argument from `phases.ppo.extra_train_args` (and for
`--rollout-nice`, also `servers.command`), run step 3's check and restart. `~/rel/v2-fast` can
stay where it is.

## 6. Box c

No item of this round gives box c a job in the live run, so there was no dress rehearsal of one.
Box c has no network path to box a (security group psppo-sg admits only TCP 22 from the laptop),
and the learner, which is what limits the run, is on box a. Box c's job this round was to build and
prove the items: the bit-exact replica of the live update, the box-a emulation, the full-load rig
and the release replica.

Its measured best use later is a Ray rollout node (+2,300 to 2,450 decisions/s, about +35%
rollout). That only pays once box a's learner is faster than box a's games, which this round
brings closer but does not reach (about 5,300 against 6,150/s). It needs the owner's approval of
one ingress rule on sg-<security-group-id> from the group itself, a timeout on train.py's telemetry
`ray.get` and a fall-back for spot loss first. Until then box c is idle, and it costs money while
idle: nothing in this release needs it after the deploy, so stopping it is the owner's call.

## 7. Not done, and worth knowing

- **Not measured on box a.** Every number here is box c's; the box-a figures are projections from
  the emulation, checked against box a's own last-hours cycle. The first hour after the switch
  measures it (section 5); `bench/learner_sched_probe.py` (read-only) measures the learner's CPU
  per minibatch before and after.
- **The run is over.** The switch waits for the owner's choice of an extended target or a new run.
- **Rollout side.** `--rollout-nice` keeps the learner at nice 0 across restarts; on box c it infers
  0.3 to 2% fewer decisions/s than the hand renice, which costs nothing while the run is learner
  bound and may cost a little once it is not. Servers at nice 0 with the workers at 5 might win that
  back; it was not measured.
- **Next learner step.** CUDA graphs or torch.compile of the update (about 9% more after this, but
  it changes float rounding, so it needs a training-curve check rather than a digest).
- Output on box c: `~/dev/fast/int/` (repo, base, rel, live, ab, out; tests2.sh, ab.sh,
  check_restart.sh, junit_compare.py, xload_compare.py). Nothing was started on box a and nothing
  was written there.
