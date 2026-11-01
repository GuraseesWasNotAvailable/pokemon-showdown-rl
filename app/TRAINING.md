# Training the policy

Three runs of `train.py`, one after another, on local Pokemon Showdown servers: imitation, then
warmup, then PPO self-play. They can be started by hand (the next two sections) or left to the
supervisor (`pipeline.py`, "Unattended runs" below), which is how the cloud box runs them. Every
command below was run on the Windows laptop (RTX 4060 8 GB, Core Ultra 7 155H, 31.5 GB RAM) on
2026-10-05 and worked. What has not been done on the laptop is a long run.

## One-time setup

| Step | Command (PowerShell, from `C:\Users\you\ps-ppo`) | Why |
|---|---|---|
| Triton for Windows | `.venv\Scripts\python.exe -m pip install triton-windows` | The model's attention is compiled by PyTorch and needs Triton, which PyTorch does not ship on Windows. Without it every update fails while the run still looks alive. |
| Spawning plugin | `Copy-Item pokemon-showdown\server\chat-plugins\rlspawn.ts C:\Users\you\showdown-bot\server\server\chat-plugins\` then `node build` in that server folder | The workers ask the server to keep battles running with `/rlautospawn`. A stock server does not have the command. |

Both are already done on this machine.

## Every run, by hand

Terminal 1, the servers. Restart them before each run: the plugin keeps refilling battles for
pairs from the previous run for a long time.

```powershell
.venv\Scripts\python.exe start_servers.py --dir C:\Users\you\showdown-bot\server --ports 8000-8003
```

Each server's own output goes to `logs\multi\` in the Showdown folder, or to the folder given
with `--log-dir`.

Terminal 2, the trainer. Set this once per terminal (the learner prints characters the default
Windows console encoding cannot):

```powershell
$env:PYTHONUTF8 = "1"
$common = "--ports", "8000-8003", "--battles", "128", "--rooms-per-pair", "16", "--minibatch", "256", "--seq-len", "256", "--grad-accum", "3", "--steps-per-update", "16384", "--ckpt-dir", "checkpoints"
```

| Phase | Command | What trains | Move on when |
|---|---|---|---|
| 1. Imitation | `.venv\Scripts\python.exe train.py --mode imitation --teacher psbot @common` | The whole network copies PSBot playing both seats. | `ploss` in the `[learner] upd=` lines has stopped falling, and the policy beats SimpleHeuristics in the evaluation below about as often as its teacher allows. |
| 2. Warmup | `.venv\Scripts\python.exe train.py --mode warmup @common` | Only the value head, on the policy's own games. | `vloss` has flattened. The imitation phase leaves the value head badly trained (it never sees a real return), and PPO needs it. |
| 3. PPO | `.venv\Scripts\python.exe train.py --mode ppo @common` | Everything, by self-play. | This is the long one. Judge by the evaluation, not by the loss. |

Three ways to stop a run, all of which save a checkpoint on the way out:

| How | For |
|---|---|
| Ctrl+C | a run in your own terminal |
| `--minutes N` | a run that should end by itself |
| `--stop-file PATH` | a run started by another program: it stops as soon as that file exists. A parent process cannot send Ctrl+C reliably on Windows, so this is what `pipeline.py` uses. The file is looked for between two telemetry lines, so once every 5 seconds, or after the learner update in progress. |

The next run picks up the newest checkpoint in `--ckpt-dir`, and when the mode differs from the
checkpoint's it starts that phase fresh from those weights, with the update and decision
counters back at zero.

`--mode`, `--ports`, `--battles` and the rest override `config.py` for that run. With no
arguments `train.py` does what `config.py` says, which is sized for a 24 GB GPU and ten servers
and starts in `ppo_with_jepa`. Do not run it bare on this laptop.

## Unattended runs

`pipeline.py` is one long-lived process that takes a run directory from nothing to a finished
PPO run: it starts the servers and the trainer for each phase, reads the trainer's output, ends a
phase when its rule fires, restarts everything when something breaks, and resumes where it was
when it is started again. `status.py` shows the run on one screen. `evaluate.py --watch` measures
the checkpoints beside it (next section).

```powershell
.venv\Scripts\python.exe pipeline.py --config cloud\run_laptop_test.json --run-dir runs\laptop_test
.venv\Scripts\python.exe evaluate.py --run-dir runs\laptop_test --watch --config cloud\run_laptop_test.json
.venv\Scripts\python.exe status.py --run-dir runs\laptop_test
```

`cloud\run_laptop_test.json` is a few minutes of the real thing on ports 8110 and 8111 (the
evaluator on 8100). `cloud/run_vm.json` is the full run on the box, where the two long-lived
programs are systemd units (`cloud/install_services.sh`, then
`sudo systemctl start psppo-train psppo-eval`).

| Command | What it does |
|---|---|
| `pipeline.py --config CFG --run-dir RUN` | Starts the run, or resumes it: same phase, newest checkpoint. |
| `pipeline.py --config CFG --run-dir RUN --dry-run` | Prints the server and trainer command of every phase and the rules. Starts nothing. |
| `pipeline.py --run-dir RUN --stop` | Asks the running supervisor to stop and waits for it. The trainer saves a checkpoint first. Exit code 0 when everything is stopped, 1 when the supervisor is still there or something it started could not be stopped (it names the process ids and ports). On the box use `sudo systemctl stop psppo-train`, or systemd starts it again. |
| `pipeline.py --run-dir RUN --clear-failed` | Makes a run that was marked failed startable again, after the cause is fixed. |
| `status.py --run-dir RUN` | One screen. `--json` for a program, `--exit-code` to fail unless the first line is OK, STARTING or DONE. |

What is in the config, by section: `servers` (how many, from which port), `train_defaults` and
per-phase `train` (the `train.py` flags), `phases` (the rule that ends each phase), `watchdog`,
`disk`, `retention`, `on_done`, and `eval` (read by `evaluate.py`). A key that starts with `_` is
a comment. A key that is not known is an error in every section but `eval`, where the evaluator
logs it and carries on.

When a phase ends:

| Phase | Rule |
|---|---|
| imitation, warmup | At `max_decisions`. Or earlier, once `min_decisions` are in and the `plateau.metric` (`ploss` or `vloss`, mean of `smooth_updates` updates) improved by less than `min_improvement_pct` percent over the last `window_decisions`. |
| ppo | At `target_decisions`. |

At the end of a phase the trainer is stopped through its stop file, its final checkpoint is kept
under `RUN\phase_final\` as well, and the next phase starts from it.

What makes the supervisor stop the trainer and the servers and start both again, from the newest
checkpoint: `total_steps` not growing for `stall_minutes`, no learner update for
`update_stall_minutes` while decisions arrive (a failed update is logged and skipped by the
learner), the trainer exiting, its telemetry failing, a Showdown server going down, a loss that
is not a number, a checkpoint that does not load. It waits longer before each restart
(`backoff_seconds`) and marks the run failed, and leaves it alone, after more than
`max_restarts_per_hour` failures in an hour or `max_fruitless_starts` starts in a row without a
single decision. A second `pipeline.py` on the same run directory refuses to start (exit code 5).

The run directory:

| Path | What |
|---|---|
| `state.json` | The supervisor's state: phase, status, counters. Rewritten every few seconds. |
| `metrics.jsonl` | One line per learner update: losses, and the decision count at the last telemetry line before it. |
| `events.jsonl` | One line per supervisor event: starts, failures, restarts, phase changes. |
| `checkpoints\` | `learner_update_NNNNNN.pt`, written by the trainer. The number restarts at every change of phase, so a later phase writes over the names of an earlier one. |
| `phase_final\` | The last checkpoint of each finished phase, under `<phase>_learner_update_NNNNNN.pt`. Never written over. |
| `logs\` | `pipeline.log`, `trainer_NNNN_<phase>_<time>.log` (the trainer's output, byte for byte), `servers_*.log` (what `start_servers.py` printed), `servers\server_<port>.out` and `.err` (each Showdown server's own output). |
| `eval\` | Written by `evaluate.py`: `results.jsonl`, `REPORT.md`, `watch.json` (its heartbeat), `final.done`. |

How decisions are counted. The count of a phase is the trainer's own `total_steps`, which is
stored in every checkpoint. After a crash it falls back to the checkpoint the trainer resumed
from, and the difference is shown as "lost to restarts". The count of a finished or stopped phase
is the last one the trainer reported before it was asked to stop. Its final checkpoint is saved
a few seconds later and holds what arrived in between (900 to 3,900 decisions more in the laptop
runs), which is why `eval\REPORT.md`, which reads the checkpoint, shows a slightly higher number
than `status.py` for the same checkpoint. The trainer reports between learner updates only, so
a phase ends at the first report at or past its limit and overshoots by what arrived since the
report before: with the laptop's tiny updates warmup once went from 4,120 to 11,211 in one
report, against a limit of 8,192. With the box's limits in the millions and updates of 36,864
that should be a few percent at most. It has not been measured there.

### Handing over a run that was started by hand

A run directory needs no `state.json` to be picked up. Stop the hand-started trainer (Ctrl+C: it
saves a checkpoint) and its servers (the supervisor refuses ports that are in use), make sure its
checkpoints are in `RUN\checkpoints\`, and start `pipeline.py` on `RUN`. It begins with the first
phase that is not marked `"skip": true` in the config, and the trainer resumes from the newest
checkpoint: in the same mode its update and decision counts continue, and count towards that
phase's rule; in another mode they start at zero on those weights. The supervisor's totals start
at the handover: decisions made before it, in a phase that is now skipped, are not in them.

### Checkpoints and disk

A checkpoint is 219 MB (the warmup checkpoint of the laptop run was 65 MB: that phase trains the
value head only). Nothing is removed unless `retention.enabled` is true in the config, and it is
false unless the config says otherwise. With it on, the supervisor removes checkpoints every
`check_minutes` and keeps:

| Kept | Setting |
|---|---|
| the newest ones, by time | `keep_newest` (at least 2: the newest is what a restart resumes from, the one before is the fallback) |
| every N-th update number | `milestone_every` |
| everything younger than this | `keep_hours` |
| the last checkpoint of each finished phase | always |
| every checkpoint the evaluator names (evaluated, being evaluated, or next) | always |
| the checkpoint the running trainer resumed from | always |

`"dry_run": true` only logs what would go. The copies under `phase_final\` are never touched.
The supervisor also stops the run cleanly when less than `disk.min_free_gb` is free. On the box
(`run_vm.json`) retention is on. On the laptop it is off and nothing is ever deleted.

### What was tested, with the real trainer and real servers

All on the laptop, 2026-10-05, 2 servers, 32 battles, updates of 4,096 decisions.

| Test | Result |
|---|---|
| The whole run, `cloud\run_laptop_test.json`, with the evaluator beside it | imitation 12,376 decisions and 3 updates, warmup 8,239 and 2, PPO 12,340 and 3, then done. 6 minutes of work, after a 10-minute stall that the watchdog ended (see the next row). The evaluator played 6 evaluations of 2 opponents during it, two of them because the phase had changed, and wrote `eval\final.done` 44 seconds after the run was done. The name `learner_update_000002.pt` was written by all three phases and evaluated three times, as three checkpoints. |
| A trainer that never produces a decision | Happened by accident: `inference.py` was saved by another edit while the trainer was starting, and both rollout workers' loops died. The trainer stayed up with `total_steps` at 0. After `startup_grace_minutes` (10) the supervisor stopped it, restarted servers and trainer, and the run went on. |
| The whole run again, with the code as it is now | imitation 12,355 decisions and 3 updates, warmup 11,211 and 3, PPO 13,408 and 4, done after 5 minutes, no restart, 6 evaluations, `eval\final.done` 40 seconds after done. `pipeline.py --stop` and a Ctrl+Break to the evaluator ended both cleanly (the evaluator in 4.4 s, saying "stopped" in its heartbeat). 318 processes were seen under the two, none was alive afterwards. |
| `status.py` against the logs | Looked at 11 times during those two runs. Every number on the screen was recomputed from the trainer logs, `events.jsonl`, the checkpoint files and `eval\results.jsonl`, 27 to 74 numbers each time. Nine times all were equal. Twice a learner update landed in the second or two between `status.py` reading `state.json` and the check reading the log, and the 3 and 11 numbers that depend on it were that one update apart. The screen is up to 5 seconds behind the log (`reporting.state_seconds`). |
| The trainer's process tree killed in the middle of a phase | Noticed after 1.3 s. Servers and trainer were started again 7 s after the wait before a restart (10, 30 or 60 s here), the first telemetry line came 25 s later, new decisions 15 s after that. The count went from 33,029 back to the 30,336 the newest checkpoint held, with 2,693 shown as lost. In another such kill it was noticed after 6.1 s and the trainer needed 90 s, not 25, to its first line on a busy machine; that kill came just after a checkpoint, so nothing was lost. |
| One Showdown server killed | Its nine child processes left by themselves within a second and `start_servers.py` started it again within three. The supervisor saw that after 2.3 s, stopped the trainer cleanly (2.5 s, checkpoint saved, nothing lost), restarted everything, and decisions arrived again 84 s after the kill, 30 s of that being the wait before a second restart. That is the intended answer: the trainer's players on a restarted server are gone and do not come back. |
| `pipeline.py --stop` in the middle of a phase | Returned after 9.4 s and 7.9 s. The trainer saved through the stop file within 6 s. Nothing was left running: every process that had been seen under the supervisor (262 and 346) was gone and the ports were closed. |
| The same start command again | Resumed the same phase from that checkpoint: the trainer's first line carried the checkpoint's own count (24,882 decisions, update 5), and new decisions arrived 46 s after the start. |
| A second supervisor on the same run directory | Refused after 0.5 s with exit code 5. The first one was not disturbed. |
| A checkpoint written by the code of the morning (commit b5680bd, imitation, update 2, 32,845 decisions) handed to the supervisor, no `state.json` | Imitation continued from it: optimizer and scheduler state loaded, the first telemetry line said update 2 and 32,845 decisions, updates 3 and 4 followed, and the run went on through warmup and PPO to done. With imitation marked skipped the same checkpoint started warmup at zero and went on to PPO. No failure, nothing lost, nothing left running (245 and 163 processes seen, none alive afterwards). |
| The same checkpoint through the evaluator's network code of that commit and of today | 296 recorded decisions of 6 real battles (48 to 51 turns each, so not past the 64-turn history window): the outputs are equal bit for bit on the GPU. One battle at a time and all battles in one forward pass (what `evaluate.py` does) differ by at most 6e-7 and pick the same action in all 296. |
| A stop while the laptop was out of memory | Went wrong once, in the evaluation-cost measurement below: `taskkill` failed and three Showdown servers outlived `pipeline.py` and `evaluate.py`. Fixed with job objects, and tested by making `taskkill` fail and by letting a parent leave before its children. The out-of-memory condition itself was not reproduced afterwards. |

Not tested here: anything on Linux beyond the parser (which is tested on output captured from
the box, `tests/fixtures/`), the systemd units, pruning for real (the removal is only ever run
against a mock), `on_done: shutdown`, and a run longer than 20 minutes.

## Evaluating a checkpoint

`evaluate.py` plays the policy in a checkpoint against fixed opponents on a Showdown server of
its own, and writes one line per opponent to `RUN\eval\results.jsonl` and a report to
`RUN\eval\REPORT.md`. The policy plays alone, without PSBot, and takes its best legal action.

One evaluation, for example of the newest checkpoint of a run (it starts a server on the port
if none is listening there, and stops it afterwards):

```powershell
.venv\Scripts\python.exe evaluate.py --run-dir runs\laptop_test --checkpoint newest --opponents random,maxpower,simple,psbot --games 200 --port 8100 --server-dir C:\Users\you\showdown-bot\server
```

For a whole run, beside the trainer, never stopping:

```powershell
.venv\Scripts\python.exe evaluate.py --run-dir runs\laptop_test --watch --config cloud\run_laptop_test.json
```

It reads the `eval` section of the config: `every_hours`, `games`, `opponents`, `port` (9000
unless set: never a training port), `foulplay`. It evaluates the newest checkpoint when the last
evaluation is `every_hours` old, once more when the phase in `state.json` changes, and a last
time when the run is done, after which it writes `eval\final.done` (on the box the supervisor
waits for that file before it shuts the machine down). A checkpoint is told apart by its name
and its time, so a name that a later phase writes again is evaluated again. `status.py` shows
the evaluator's heartbeat and the last six results per opponent, and adds `EVAL_STALE` to its
first line when the evaluator has stopped refreshing `eval\watch.json` without saying it stopped.

| Opponent | What it is | Reference |
|---|---|---|
| `random` | poke-env RandomPlayer | The floor. A working policy beats it. |
| `maxpower` | poke-env MaxBasePowerPlayer | |
| `simple` | poke-env SimpleHeuristicsPlayer with the stat fix | PSBot beats it in about 91% of games, so that is what imitation can reach. |
| `psbot` | The imitation teacher | 50% means level with the teacher. |
| `foulplay` | pmariglia's search bot (`bench/FOULPLAY.md`) | Skipped with a note where it is not installed (it is not on the laptop). |

200 games give a win rate to about plus or minus 7 points. The report says "better" or "worse"
only when the intervals do not overlap or a two-proportion test gives p < 0.05. A line with no
finished game, or marked `"valid": false`, is in the file and the report but not in `status.py`'s
table.

What one full evaluation costs on the laptop while a training run goes beside it. Measured once,
2026-10-05: the barely trained test checkpoint (imitation, update 2, 32,845 decisions), 200 games
against each of the four opponents, 32 battles at a time, the policy on the GPU, beside a PPO run
on 2 servers with 32 battles.

| Opponent | Games a minute | 200 games took | Win rate of that checkpoint |
|---|---:|---:|---|
| `random` | 89 | 134 s | 69.0% (62.3 to 75.0) |
| `maxpower` | 126 | 96 s | 26.5% (20.9 to 33.0) |
| `simple` | 134 | 90 s | 2.0% (0.8 to 5.0) |
| `psbot` | 33 | 366 s | 0.5% (0.1 to 2.8) |

| | Alone | With the evaluation beside it |
|---|---|---|
| The evaluation, start to exit | | 14 minutes: 11.4 of games, 2.6 of starting its server and loading the checkpoint once per opponent |
| GPU memory in use | 2.6 to 2.7 GB (the training run) | 2.8 to 3.2 GB. The evaluation's own share is 0.2 to 0.5 GB, of which torch held at most 302 MB. |
| Free RAM | 3.3 to 4.3 GB | 0.2 to 2.0 GB for most of it |
| The training run | 640 decisions a second (its first minutes, so inside the CPU's boost) | 167 decisions a second over the 13.6 minutes |

So on this laptop it is memory, not the GPU, that the evaluation costs: with 2 training servers
the machine was out of RAM and swapping for the whole evaluation, and training ran at about a
quarter of its speed. During the last two and a half minutes another job (not part of this
measurement) filled the GPU to 7.9 GB, so the `psbot` figure is a little low: it was 35 a
minute before that. Under that load `taskkill` failed, and the evaluation server and both training
servers were left running after their owners had exited. That is fixed (see "Fixed along the
way") and was the reason for it. On the box, with 124 GB of RAM and a 46 GB GPU, neither limit
applies. What will matter there is CPU: 32 evaluation battles, the PSBot engine and four Foul
Play processes beside the training servers on 16 vCPUs. The evaluator's unit runs at `Nice=5`
for that reason. It has not been measured there.

If the GPU cannot be used the evaluation falls back to the CPU and says so in its lines. That
works and is very slow: one forward pass for one battle took 2.1 s on this laptop's CPU, against
about 30 ms on the GPU.

For a quick look at one pairing without any files, `play.py` still works:
`.venv\Scripts\python.exe play.py --agent policy --checkpoint checkpoints\learner_update_000123.pt --opponent simple --games 400 --port 8100`.
Give the exact file name: for the `policy` agent a missing file is an error.

## What was measured on the laptop

| Run | Scale | Result |
|---|---|---|
| Imitation, SimpleHeuristics as teacher | 2 servers, 32 battles | 443 decisions a second, 30 updates in 5 minutes, policy loss 1.81 to 1.10 |
| Imitation, PSBot as teacher | 2 servers, 32 battles | 110 decisions a second |
| Imitation, PSBot as teacher, the commands above as written | 4 servers, 128 battles | 175 decisions a second, about 1.6 hours per million |
| PPO | 2 servers, 32 battles | 222 decisions a second, 6.4 GB of the 8 GB GPU |
| PPO | 6 servers, 192 battles | 425 decisions a second over 6 minutes, 6.6 GB GPU, free RAM down to 1 GB with a browser open |
| PPO, the commands above as written, 12 minutes | 4 servers, 128 battles | 390 decisions a second, flat from minute 5 on; free RAM down to 1.1 GB with a browser open |
| Warmup, and PPO with `--grad-accum 3` | 2 servers | ran clean |
| Stop, restart, change of phase | | each restart resumed from the newest checkpoint |
| Under the supervisor: trainer start to first telemetry line, to first decisions | 2 servers | 18 to 33 s, 23 to 38 s (once 90 s on a busy machine) |
| Under the supervisor: clean stop of the trainer through the stop file | 2 servers | 5 to 8.5 s, the save itself under 1 s |

The first three minutes of any run are 10 to 60% faster than what follows (the CPU's short
boost), so judge speed after minute five. At the sustained 390 a second, 250 million decisions
take about 7.4 days of the laptop running flat out. Six servers were not faster in any way that
lasts: the one GPU shared by learner and inference is the limit, not the simulation.

## Limits to know before a long run

- **Memory.** Six servers and six workers left 1 GB free with Chrome and other apps open. Close
  them, or stay at four servers.
- **GPU.** The default minibatch (768 by 256) does not fit 8 GB. 256 with `--grad-accum 3` is
  the same effective batch. The JEPA modes are estimated above 9 GB and were not tried.
- **Disk.** A checkpoint is 219 MB and one is written every 25 updates, roughly 10 GB a day.
  A run by hand never removes one. Under the supervisor see "Checkpoints and disk".
- **The upstream number.** "250M states in two days on a 3090" was written for an earlier, much
  lighter model. It says little about this one.
- **Sleep and restarts.** Closing the lid on AC power sleeps this machine, and a Windows update
  restarts it. Either ends the run. By hand nothing restarts it, and on the laptop nothing
  starts `pipeline.py` again either: start it yourself and it resumes. On the box systemd does.
- **Long runs.** Until a run of several hours has been watched, cap each trainer: by hand with
  `--minutes 240` and a restart of the servers, under the supervisor with
  `watchdog.recycle_hours` (6 in `run_vm.json`), which stops and restarts servers and trainer
  cleanly and is not counted as a failure.
- **The Python environment.** PyTorch and NumPy are not in `.venv`. They load from the user-wide
  Python 3.14 packages, which other projects share. A `pip install` for another project during
  a run can change them under it.
- **Editing the code during a run.** Each trainer start imports the source files as they are
  on disk at that moment. A file saved half-way through a change while a trainer starts can
  leave it running with rollout loops that have died (seen once, see above): the supervisor
  catches it only after `startup_grace_minutes`.
- **Silent failures.** A failed update is logged and skipped. By hand, watch that `update` in
  the telemetry line keeps counting. The supervisor restarts after `update_stall_minutes`.
  A rollout worker whose loop has died keeps answering heartbeats, so `train.py` does not
  notice: with every worker dead the supervisor's stall rule fires, with only some dead the
  run just gets slower.
- **Learning rate.** `config.py` holds the rate for 500,000 optimizer steps and then, because
  `lr_total_steps` is smaller than that (it reads `1_00_000`), drops it by 27 times in one step.
  That point is tens of millions of decisions in, so it does not touch a pilot. Decide what the
  schedule should be before a full run.
- **Showdown's own log.** Under the supervisor each server's output is kept in
  `RUN\logs\servers\`. Lines that say CRASH there at the moment the servers are stopped are the
  servers being stopped. On Windows, servers started in the same second from one folder can
  also log `EPERM ... rename ... the-studio.json`: they write the same settings file. They
  keep running.

## The cloud option

Prices fetched 2026-10-05, speeds extrapolated from the laptop measurements, so treat the times
as rough.

| Instance | GPU | Est. decisions a second | 250M decisions | Spot cost | On-demand cost |
|---|---|---|---|---|---|
| g5.4xlarge | A10G 24 GB, 16 vCPU | 600 to 950 | 3 to 5 days | $50 to $106 | $119 to $188 |
| g6e.4xlarge | L40S 48 GB, 16 vCPU | 600 to 1,100 | 2.6 to 4.8 days | $53 to $97 (us-east-2) | $190 to $348 |
| g6e.8xlarge | L40S 48 GB, 32 vCPU | 870 to 1,800 | 1.6 to 3.3 days | $49 to $101 (us-east-2) | $175 to $361 |

New AWS accounts have a GPU instance quota of zero, separately for spot and on-demand, so the
quota request comes first. On Linux none of the Windows workarounds are needed: PyTorch ships
Triton there, and `start_servers.py` works unchanged.

## Fixed along the way

Found by running it, all in this repository now:

- `train.py` took no arguments, so changing the phase or the scale meant editing `config.py`.
- After a change of phase the update counter restarts, and resume picked the highest-numbered
  checkpoint, which was the old phase's. Every restart of the new phase threw its progress away.
  Resume now takes the newest file.
- A stopped run lost everything since its last periodic checkpoint. It now saves on the way out.
- `grad_accum_steps` in `config.py` was never passed to the update.
- Every finished battle left a task waiting forever in the worker, and every new battle woke all
  of them. A finished battle's late messages are now dropped. The same change stops the PSBot
  teacher's engine from rebuilding battles that are already over.
- `start_servers.sh` is bash with the upstream author's `D:` path. `start_servers.py` replaces it
  and starts every instance with `--skip-build`, since each start otherwise rebuilds the server
  under the instances already running from the same folder.
- Each Showdown server's output went to the Showdown folder, where nobody looked and every run
  added to the same files. `start_servers.py --log-dir` puts it under the run.
- `status.py` said "last update 0.2 min ago" for a trainer that had not made one yet, and showed
  an evaluation in which no game finished as a win rate of 0%.
- The evaluator refused to start for up to ten minutes after a reboot when its old process
  number had been given to something else.
- On Windows, stopping "a process and everything under it" went through `taskkill /T`, which has
  nothing to go on once the process itself has gone, and which failed outright when the laptop
  was out of memory: one evaluation server and both training servers stayed up after
  `pipeline.py` and `evaluate.py` had exited, and the supervisor still called the run "stopped".
  Each child now runs in a Windows job object, which holds everything it starts and is ended
  with one call (on Linux the process group already did that). What a stop could not stop is
  recorded, shown by `status.py`, and makes `pipeline.py --stop` exit with code 1.
- A group without a name made `cloud/install_services.sh` render broken unit files.
