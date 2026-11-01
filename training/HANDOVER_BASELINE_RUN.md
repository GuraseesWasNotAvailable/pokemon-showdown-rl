# Handover: train the new baseline on EC2

For an agent with AWS access (the `aws` CLI, key pair `psppo-key`, security group `psppo-sg`).
Written 2026-10-13.

## What to train

- **Code:** branch `search` at merge `73f524b` (this file sits on top of it). That is the
  collaborator's `search` (5b13fff) plus these fixes:
  - **9422caf**, four training bugs:
    - KV history no longer wiped on every weight sync.
    - `lr_total_steps` corrected to 1,000,000.
    - `ff_expansion` and `dropout` now reach the layers.
    - KL early stop now applies in every PPO mode.
  - **c213830, 16a975a, 2d23dec:** the network now sees its own Tera type (on by default), and
    STAB on the original types comes back after a Tera.
- **Not this run:** every `idea/*` branch (league, MMD, expert iteration, bigger model,
  backbone LR, set data, matchup shaping, test-time compute, top-player imitation),
  `jaxcalibur` and `psbot-hybrid`. Do not merge or cherry-pick them. They are tested later as
  arms *against* this baseline.
- **Run shape:** keep training the latest PPO checkpoint, now on the fixed code.
  - Start from the newest PPO checkpoint of the collaborator's run. The ladder bot plays
    `learner_update_011850.pt`, but the box may hold a newer one. Use the newest PPO checkpoint
    you can get.
  - Every idea guide (`bench/*.md` on the idea branches) starts its arms "from the same `ppo`
    checkpoint". This run's output becomes that checkpoint.
  - Own Tera type switches on at resume. The learner prints
    `trained with own_tera_type=False, continuing with own_tera_type=True`.
  - A from-scratch run (imitation → warmup → PPO) is option B at the end. Use it only if the
    owner asks.

## 0. Before anything costs money

```bash
git fetch origin && git checkout search && git pull --ff-only
git log --oneline -3          # HEAD is this handover commit, then 73f524b
git status --short            # must be empty: vm.sh copies untracked files too
ls heuristic/cache            # PSBot data. Missing -> run: node heuristic/setup.js
bash cloud/vm.sh info         # is there already a psppo-train box?
```

**Ask the owner before you:**
- launch an instance;
- stop or replace any trainer already running;
- run `cloud/teardown.sh`.

On an existing box, also leave the ladder bot alone: the tmux session `ladder`, code under
`~/ladder`.

## 1. Get a box

**If `vm.sh info` shows a box:**
1. Reuse it.
2. Run `bash cloud/vm.sh ssh 'systemctl is-active psppo-train psppo-eval; ls -t ~/runs/*/checkpoints/*.pt | head'`
   to see what runs there and where the newest checkpoint is.
3. If an old-code run is going, ask the owner whether to stop it. Stop it with
   `sudo systemctl stop psppo-train psppo-eval`, which saves a checkpoint.

**If there is no box,** launch one:

```bash
bash cloud/launch.sh                                   # spot, us-east-1: g6e.16xlarge, then g6e.8xlarge
# or on demand, in the account the owner names:
AWSCMD="aws --profile <profile>" bash cloud/launch_od.sh us-east-2 g6e.16xlarge
bash cloud/vm.sh allow-ip                              # if SSH times out
```

- **Instance type:** use g6e.16xlarge (64 vCPU, L40S 48 GB). `cloud/run_big.json` is sized for
  it.
- **Smaller box:** on g6e.8xlarge, halve `servers.count` and `battles` in step 3.
- **Second account:** set up `cloud/.state/targets/<name>.env` for it (see the header of
  `cloud/vm.sh`). It sets `PSPPO_REGION`, `AWSCMD` and `PSPPO_NAME`. Then put that name in
  `cloud/.state/current`.
- **Default quota:** new accounts have a GPU quota of zero for spot and for on-demand.

## 2. Put the code on the box

**New box:**

```bash
bash cloud/vm.sh deploy --head
bash cloud/vm.sh ssh 'bash ~/ps-ppo/cloud/bootstrap.sh'      # 15-30 min; idempotent
```

**Box with code already running:** never `deploy` over it. Release side by side, then switch
once the trainer and evaluator are stopped:

```bash
bash cloud/vm.sh release baseline-73f524b
bash cloud/vm.sh activate baseline-73f524b
bash cloud/vm.sh ssh 'cat ~/ps-ppo/DEPLOYED; bash ~/ps-ppo/cloud/bootstrap.sh'
```

**Then, on the box,** run the tests that cover the fixes:

```bash
cd ~/ps-ppo && ~/venv/bin/python -m pytest -q tests/test_tera.py tests/test_speed_kvpool.py tests/test_speed_update.py tests/test_pipeline.py
```

All must pass. Stop and report if not.

## 3. New run directory, seeded with the start checkpoint

Use a new directory, `~/runs/baseline`. Do not reuse `~/runs/main`: its `state.json`,
counters and evaluations belong to the old code.

1. Make the directory and copy the start checkpoint in:

   ```bash
   mkdir -p ~/runs/baseline/checkpoints
   cp <newest PPO checkpoint> ~/runs/baseline/checkpoints/
   ```

   - **Checkpoint on the box:** it is under `~/runs/main/checkpoints/`.
   - **Checkpoint only on the collaborator's laptop:** it is at
     `runs\vm\checkpoints\learner_update_011850.pt`. Copy it up with `scp`, using the key
     `vm.sh` uses.

2. Inspect it:

   ```bash
   ~/venv/bin/python - <<'PY'
   import glob, torch
   p = sorted(glob.glob('/home/ubuntu/runs/baseline/checkpoints/*.pt'))[-1]
   c = torch.load(p, map_location='cpu', weights_only=False)
   rc = c.get('run_cfg') or {}
   print(p)
   print('update', c['update_idx'], 'total_steps', c['total_steps'])
   print('optimizer step', (c.get('scheduler') or {}).get('last_epoch'))
   print('mode', (rc.get('learner') or {}).get('mode'), 'own_tera_type', (rc.get('obs') or {}).get('own_tera_type'))
   PY
   ```

   Check these before going on:
   - **mode** must be `ppo`. Anything else means a wrong checkpoint: stop.
   - **Optimizer step between 501,000 and 1,000,000:** the fixed LR schedule resumes about 8×
     higher than the old run was using. It was dropped 27× in one step at 501k. Report the
     number to the owner before starting. Past 1,000,000 nothing changes. Under 501,000 the
     schedule is the same as before.

3. Write the run config, outside the code so a new release cannot reset it:

   ```bash
   cp ~/ps-ppo/cloud/run_big.json ~/runs/baseline/run.json
   ```

   Edit `~/runs/baseline/run.json`:
   - **`phases.ppo.target_decisions`:** set it to `total_steps` from the checkpoint plus the
     budget. The count continues from the checkpoint, so a bare 600M could already be almost
     reached. The default budget is **+250,000,000 decisions**. That is about 17–20 h at
     3,500–4,000 decisions/s, roughly $130–150 on-demand at $7.58/h. Confirm the budget with
     the owner.
   - **Keep:** `imitation.skip` and `warmup.skip` stay `true`, `on_done` stays `"shutdown"`, and
     `extra_train_args` stay empty. Own Tera is the default and needs no flag.
   - **`_notes`:** update them to say what this run is (baseline, commit, start checkpoint).

4. Dry run:

   ```bash
   ~/venv/bin/python ~/ps-ppo/pipeline.py --config ~/runs/baseline/run.json --run-dir ~/runs/baseline --dry-run
   ```

   Check that the PPO command shows `--mode ppo`. It must have no `--no-own-tera-type`.

## 4. Start it under systemd

```bash
RUN_DIR=/home/ubuntu/runs/baseline CONFIG=/home/ubuntu/runs/baseline/run.json \
  bash ~/ps-ppo/cloud/install_services.sh --now
```

This points `psppo-train` and `psppo-eval` at the new run. Units start at every boot, so a spot
stop resumes on its own.

**First 30 minutes,** watch:

```bash
bash cloud/vm.sh ssh '~/venv/bin/python ~/ps-ppo/status.py --run-dir ~/runs/baseline'
bash cloud/vm.sh ssh 'journalctl -u psppo-train -n 100 --no-pager'
bash cloud/vm.sh ssh 'grep -h "own_tera_type\|upd=" ~/runs/baseline/logs/trainer_*.log | tail -20'
```

| Check | Expected |
|---|---|
| Resume line | `continuing with own_tera_type=True`; update and total_steps continue from the checkpoint |
| Learner | `[learner] upd=` count keeps rising; `ploss` / `vloss` finite |
| Throughput | 3,000–4,000 decisions/s after minute 5 (run_big measured 3,500) |
| `status.py` first line | `OK` |
| GPU / disk | `nvidia-smi` busy; ≥ 20 GB free (retention prunes) |

**If it does not reach that:**
1. Stop it: `sudo systemctl stop psppo-train psppo-eval`.
2. Read `~/runs/baseline/logs/`.
3. Report. Do not tune hyperparameters or turn on idea flags to get it running.

## 5. While it runs, and when it ends

- **Check-ins:** every few hours, run `status.py`. Also look at `~/runs/baseline/eval/REPORT.md`:
  every 3 h it runs 400 games each against `simple` and `psbot`, and 200 against Foul Play.
- **At the target:** the supervisor waits for the final evaluation and then halts the box. The
  disk stays.
- **Get results:** `vm.sh pull` only reads `~/runs/main`, so copy by hand:

  ```bash
  scp <ssh opts from vm.sh> ubuntu@<ip>:runs/baseline/checkpoints/<newest>.pt runs/baseline/
  scp <ssh opts from vm.sh> ubuntu@<ip>:runs/baseline/eval/REPORT.md runs/baseline/
  ```

- **Report to the owner:**
  - final checkpoint name and its `total_steps`;
  - win rates from `REPORT.md` with intervals;
  - decisions/s;
  - any restarts from `events.jsonl`.

  That checkpoint is the start point for every idea arm.
- **Known open issue:** KV-cache rows of workers that died are not reclaimed until the trainer
  restarts. The pool logs a warning as it grows. `watchdog.recycle_hours: 6` bounds it.
  Mention it only if the warning keeps growing between recycles.
- **Stop paying:** `sudo systemctl stop psppo-train psppo-eval` stops training. A halted box
  costs only its disk. `cloud/teardown.sh` deletes everything; it asks first, and the owner
  decides.

## Option B: from scratch (only if the owner asks)

Same steps, with these differences:
- Leave `~/runs/baseline/checkpoints/` empty.
- Use `cloud/run_vm.json` as the template for `run.json`, with phases imitation → warmup → PPO,
  target 250M. Take `servers`, `battles` and `save_every` from `run_big.json` for the 64-vCPU
  box.
- `teacher` stays `psbot`.

This costs days rather than hours, and its result is not comparable to the ladder bot's
checkpoint lineage.
