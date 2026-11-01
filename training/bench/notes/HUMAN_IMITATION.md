# The top players' imitation term in the live learner (branch v2-humanaux)

`human_imitation.py` adds the top players' own decisions to the PPO loss as a weighted cross-entropy:
in every k-th minibatch, coef x CE(pi, the player's action) on one packed sequence of 256 decisions
from human/rebuild.py's shards. It is ported from branch stage3 (bench/STAGE2.md section 2 there) to
this code line, behind flags that are off by default. Off, every update is the commit before it, bit
for bit (tested). It is learner-side only: the workers, the inference actors and the observation do
not change, and it adds no parameter.

```
python train.py --mode ppo ... --human-imitation-coef C --human-imitation-data ~/dev/human_shards
```

**Short answer on the coefficient.** At the live run's learning rate (3e-4, held until optimizer step
801,000) the term's first effect is to make the policy less sure of itself, as stage3 found: the
network is over-confident on top players' decisions (CE 2.1 nats at 55% agreement), and the
cross-entropy flattens it. In the 20-update smoke runs from the live checkpoint, coef 0.02 (stage3's)
and 0.01 raised the self-play entropy from 0.44 to 0.60-0.65, still climbing, while agreement with
held-out players moved by half a point. Recommended for the live run: **0.012 on every 4th minibatch** (`--human-imitation-coef 0.012 --human-imitation-every 4`, the pull of 0.003 on every one at a quarter of the cost): in the smoke it kept KL, the value loss and the heads where the run is, with entropy 0.47 to 0.50 (off: 0.44-0.45), and costs the live learner about 0.4 s an update instead of 1.6 s. It is sized for safety, not for a proven gain: stage3 found no strength from the top players' labels (section 6). Switch-on and way back: section 7.

## 1. Flags (train.py; all off by default)

| flag | default | what |
|---|---|---|
| `--human-imitation-coef` | None (off) | weight of the top players' CE in the PPO loss; absent or 0: nothing is imported |
| `--human-imitation-data` | None | shard directory (needed with the coef; refused at parse without it) |
| `--human-imitation-splits` | `~/dev/human-data/splits` | the collector's split files, read only for shards without `ep_split` (stage3's layout) |
| `--human-imitation-salt` | `~/dev/human-rebuild/salt` | the rebuild's salt, with the split files |
| `--human-imitation-seqs` | 1 | human sequences (256 decisions) per PPO minibatch |
| `--human-imitation-every` | 1 | add the term to every k-th minibatch |
| `--human-imitation-switch-weight` | 1.0 | the players' switches (actions 4-9) count W times |
| `--human-imitation-double-switch-weight` | 1.0 | switched by choice on a turn the opponent also switched: W times more |
| `--human-imitation-cap` | 0.03 | largest share of the train decisions one player may hold |
| `--human-imitation-shuffled` | off | the control arm: labels permuted within a legal mask |
| `--human-imitation-pool` | 120000 | decisions held in the learner (21 kB each with this observation: 2.5 GB); an eighth swapped per update |
| `--human-imitation-holdout` | 8192 | held-out players' labelled decisions in the fixed agreement set, at most `cap` of it per player (245 at the defaults) |
| `--human-imitation-holdout-every` | 50 | updates between held-out measurements (and at the first update) |
| `--human-imitation-set-coef` | 0.0 | new: the SetHead also learns the shards' exact set labels; needs `--set-head-coef` |

Values that cannot be (a negative coef, a weight of 0, a cap above 1, a coef, weight, cap or
SetHead coef that is not a finite number, the SetHead part without `--set-head-coef`, a coef without
data) are refused by train.py's parser (and so by check_restart), like this line's other flags. Data that cannot be had (no directory, no shard of this run's observation, no train side)
never stop a run: the learner prints `[learner] human imitation OFF: <reason>` and every update
line ends `hum=off`. `cloud/check_restart.py` refuses such a start beforehand.

The update line, with the term configured, ends with `hum_n hum_ce hum_agree hum_sw hum_dsw hum_ho
hum_ho_pp hum_ho_ce hum_ho_upd hum_total hum_pool hum_wait hum_err hum_nonfinite hum_drop`
(`hum_set_loss hum_set_n` with the SetHead part): the train decisions of the update's first epoch,
their CE and top-1 agreement (the train accuracy), the held-out players' top-1 agreement (pooled over
the capped set, and per player) and CE, the data in memory, minibatches without a batch ready,
minibatches whose term raised (`hum_err`: left out, harmless), minibatches whose term was not finite
(`hum_nonfinite`: see section 2, it means the weights are gone), and the episodes dropped at load for
a value that is not finite (`hum_drop`, each counted once). Every figure is a finite number or `na`,
so the supervisor's NaN guard never counts it.

## 2. What changed from stage3

- **The observation check is this line's.** A shard loads only when it records the run's
  `own_tera_type`, `obs_matchup` and `obs_movesets`, no `obs_abilities`, and (when it says) the same
  `obs_dim` as the network's input (10,295 for the live run); each loaded shard's rows are checked
  against the width again. Stage3's shards (no matchup block) and the abilities builds are refused.
- **The split comes from the shard.** Branch v2-human's rebuild writes `ep_split` per side: `train`
  sides train, the held-out players' own sides (`test_player`) measure, `loose` (the other side of a
  held-out player's game) and `test_game` are never used. Shards without it read the split files as
  stage3 did.
- **The learner waits for the GPU less.** The term's figures are summed on the GPU and read once per
  update; the weight sum is taken on the CPU.
- **Values that are not finite never reach the term** (review fix, see section 9). When a shard
  loads (on the CPU, in the loader thread, about 0.18 s a shard), any episode with an inf or NaN in
  its observation or mask is dropped whole and counted in `hum_drop`. This matters because the
  stage3 way of guarding (replace a non-finite term by zero with `torch.where`) does not stop the
  gradient: log_softmax's backward computes `grad - exp(out) * sum(grad)`, and with NaN logits that
  is `0 x NaN = NaN`, so one bad row, even a weightless history row, made every weight's gradient
  NaN and one optimizer step made every weight NaN, while the logged term read 0.0. Settings that
  are not finite are refused for the same reason (`--human-imitation-coef inf`: `inf x 0 = NaN`).
  Past those two checks only a network that is already NaN can give a non-finite term; it is
  counted in `hum_nonfinite` and left in the loss on purpose, so the update's `loss=` is NaN at the
  update it happened and the supervisor's NaN guard quarantines from that update (a zeroed term
  would have shown a finite loss there, and a checkpoint written at that update could have escaped
  the quarantine).
- **The held-out set caps each player** (review fix, section 9): no player above `cap` x
  `--human-imitation-holdout` (245 of 8,192 at the defaults; a player's first episode is always
  taken). Stage3's set took whole shards until 8,192: on the test rebuild one account was 80% of it,
  and on the full 40,000 games (modelled from box d's index) one account would be about half, so
  `hum_ho` mostly measured agreement with one prolific, probably grinding or bot, account.
- **The SetHead part** (`--human-imitation-set-coef`, off by default): the shards carry
  `set_labels`, the opponent's real team from the re-simulation in `obs_matchup.build_set_labels`'
  layout (6 x 8 with the moveset block). With the flag, the human batch is run with `return_z` and
  `ppo_core.compute_set_head_loss` is added on those labels (padding unlabelled). It needs
  `--set-head-coef`, so the head already exists and the optimizer's layout cannot change. **Not
  recommended now:** the live SetHead already reads the human games as well as its own (human/agree.py
  on the test rebuild, checkpoint 5550: ability 0.78-0.79, item 0.80, against the live run's own
  0.78-0.81 and 0.78-0.82), and the labels come from the same set generator as self-play's.
- **The loader starts with the learner** (beside the resume), at nice +10 on Linux.
- Checkpoints written with the term carry `human_imitation` (its settings); nothing reads it to
  resume, and an older release loads such a checkpoint as any other.

## 3. Tests (box c, ~/dev/humanaux, GPU 0, nice 15)

`tests/test_human_imitation.py`: 31 passed (27 on the CPU, 4 on the GPU with
`PSPPO_HEAD_DIR` = a git archive of train-v2 dcc5116 and `PSPPO_HUMAN_SHARDS` = the test rebuild);
26 at 75e55f2, 5 more with the review fixes (section 9).

| test | what it holds |
|---|---|
| off is nothing | no flag (or the live run's flags, or coef 0): no setting, the config unchanged, `LearnerActor(human_imitation=None)`, `ppo_update(human=None)`, main() passes the learner nothing |
| values refused, data not read at parse | the thirteen impossible values exit at parse (six of them not finite: coef inf and nan, switch and double-switch weight inf, cap nan, SetHead coef inf); `check()` refuses inf for each; a missing directory is accepted at parse |
| double switch, shuffle | stage3's, ported |
| roles from `ep_split`, cap, weights | 12 train sides, 3 held-out, 4 unused (loose and test_game); player cap; switch and double-switch weights; nothing of the held-out players or their opponents in the pool |
| stage3 layout | shards without `ep_split` read the split files; without readable files: off, with the reason |
| built otherwise (6 cases) | no matchup, matchup only, no own Tera, abilities, another `obs_dim`, no fields: never loaded, off with "rebuild" |
| width without `obs_dim` | rows checked against the network's width |
| batches, the term, SetHead part | whole sequences, weightless padding, padding unlabelled; coef x weighted CE; gradients reach the policy, never the value head; held-out agreement and CE; the line parses with no non-finite value; the SetHead part adds its loss on the gathered labels and is off without a head |
| values that are not finite (3 cases: inf, -inf, NaN) | a weightless history row of one train side and of one held-out side holds the value: both episodes are dropped at load (`hum_drop=2`), the other 7 train sides and the other held-out side load, every batch the term draws is finite and so is every gradient after backward |
| a term that is not finite | (a network already NaN) counted as `hum_nonfinite=1`, `hum_err=0`, returned as it is, so the update line's `loss` is the non-finite field the supervisor reads |
| why the rows are dropped | the reviewers' mechanism held as a test: a zeroed `torch.where` term over a weightless inf row reads 0.0 and still leaves every weight's gradient NaN |
| the held-out cap | one account with 30 episodes first in the shard and 20 others: at cap 0.2 of 200 the set is 200 decisions of 9 players, the account 40 (20%), none above; at cap 0.01 each player's first episode is still taken (20 players) |
| `ppo_update` | `human=None` and a term still loading are bit for bit the update without it; with it the update changes; a failing term is counted and left out; `every` |
| resume (2 cases, with and without the opponent-action head) | a checkpoint written without the term loads with it: "Optimizer state loaded successfully", weights, fp32 masters, Adam moments and steps, every group's rate, the scheduler's step, update, decisions and episodes all equal; and back |
| check_restart | the live run's arguments plus the term: OK and "human data: 2 of 2 shards fit"; off; other observation, other width, missing directory, coef without data: refused |
| GPU: flags off vs dcc5116 | a whole `_perform_update`, every feature off and the live run's features on (opponent-action head, boosted rows x2, read bonus with its column): every weight, master, Adam state, rate and step bit-identical |
| GPU: real shards | the test rebuild on the network the live run builds (matchup, movesets, SetHead): index 1,563 train sides (48,762 labelled decisions of 252 players), 1,277 held-out sides; masks are the observation's own; term and backward reach the policy and the SetHead; no held-out player's side in the pool. **0.028 s per minibatch** for the term (0.031 s with the SetHead part), gather 0.3 ms |
| GPU: a whole learner update on real shards | `_perform_update` with the term and the live heads: the line parses, hum_err 0, hum_nonfinite 0, hum_drop 0, hum_wait 0, hum_ho measured (capped set: 1,392 decisions of 30 players at 2,048, none above 61) |

Other suites on this tree (box c): check_restart, train_v2, lr_restart, stage_lr, infer_actors,
opponent_seats with test_human_imitation, CPU: 192 passed, 14 skipped, 1 failed
(`test_infer_actors.py::test_infer_batch_with_fast_sample_returns_the_same_values_and_histories`,
which fails the same way on dcc5116 when run after the other modules and passes alone in both
trees). GPU (`PSPPO_REF_DIR` = b5680bd's files): speed_update, train_v2, speed_ref, widen: 51 passed,
17 skipped (fixtures not on the box: old checkpoints, git history, widening checkpoints). After the
review fixes (section 9), the same suites again: CPU 197 passed, 14 skipped, the same 1 failed; GPU 51
passed, 17 skipped; check_restart on the live run.json copy with `--human-imitation-coef 0.012
--human-imitation-every 4` against checkpoint 007800: "OK to start", rate 3.000e-04 at the resume.
`cloud/check_restart.py` from this tree on a copy of the live run.json (repo_dir pointed at the
tree), against a copy of the live checkpoint 007800, with and without
`--human-imitation-coef 0.01 --human-imitation-data <test shards>`: "OK to start" both times, rate
3.000e-04 at the resume as written, "human data: 12 of 12 shards fit this run's observation; 1,563
train sides, 48,762 labelled train decisions; 1,277 held-out players' sides, 41,478 decisions".

## 4. The smoke: 20 updates from the live checkpoint

Box c (g6e.4xlarge, one L40S shared by the learner and one inference actor, 16 vCPU), code 75e55f2,
`~/dev/humanaux/smoke/run_arm.sh`: each arm resumed a copy of the live run's checkpoint 007800
(288.6M decisions, optimizer step 390,379, rate 3.00e-04, taken from the S3 backup) and ran 20
updates with the live run's learner settings: 36,864 steps an update, minibatch 768, sequences of 256,
2 epochs with target_kl, the SetHead 0.1, the opponent-action head 0.1, the read bonus (0 at these
decisions), boosted rows x2, setupnet seats 0.15, the rate schedule (hold 800,000, total 1,600,000).
6 Showdown servers (11300-11305) with 64 battles each, as the live run's 36 have, nice 15. Human data:
the test rebuild (12 shards, 1,563 train sides, 48,762 labelled train decisions of 252 players; the
held-out set 8,746 decisions of 18 players), `--human-imitation-holdout-every 5`. Every arm cleaned up
after itself (servers, trainer, Ray: nothing left on the box, port and GPU free).

Means over updates 2-20 (the first holds the start's queue and the compile); entropy and the human CE
as "first 5 -> last 5"; hum_ho at updates 1 and 20 (held-out players, pooled top-1; its CE in brackets).

| arm | kl | clip | early stops | entropy | value loss | hum_ce (train) | hum_agree | hum_ho (CE) | s / minibatch | s / update at 50 minibatches | decisions/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 0.0204 | 0.244 | 13 of 19 | 0.444 -> 0.452 | 2.127 -> 2.128 | | | | 0.089 | 5.14 | 1,829 |
| 0.001 | 0.0211 | 0.249 | 13 | 0.453 -> 0.470 | 2.131 -> 2.117 | 2.10 -> 2.06 | 0.541 -> 0.542 | 0.555 (1.90) -> 0.559 (1.85) | 0.119 | 6.77 | 1,885 |
| 0.003 | 0.0198 | 0.251 | 12 | 0.456 -> 0.487 | 2.126 -> 2.133 | 2.07 -> 1.95 | 0.545 -> 0.548 | 0.557 (1.89) -> 0.562 (1.77) | 0.120 | 6.72 | 1,840 |
| 0.012, every 4 | 0.0204 | 0.256 | 9 | 0.465 -> 0.495 | 2.134 -> 2.146 | 2.02 -> 1.96 | 0.548 -> 0.541 | 0.561 (1.89) -> 0.562 (1.75) | 0.097 | **5.56** | 1,872 |
| 0.01 | 0.0205 | 0.276 | 6 | 0.506 -> **0.592** | 2.117 -> 2.145 | 1.93 -> 1.65 | 0.544 -> 0.551 | 0.552 (1.85) -> 0.552 (1.55) | 0.122 | 6.83 | 1,762 |
| 0.02 (stage3's) | 0.0188 | 0.278 | 5 | 0.518 -> **0.643** | 2.141 -> 2.124 | 1.84 -> 1.47 | 0.552 -> 0.566 | 0.557 (1.76) -> 0.564 (1.41) | (GPU shared with the tests) | | 1,711 |
| 0.012, every 4, after the review fixes | 0.0195 | 0.250 | 13 of 20 | 0.449 -> 0.503 | 2.135 -> 2.133 | 2.05 -> 1.95 | 0.548 -> 0.541 | capped set: 0.513 (2.31) -> 0.516 (2.08) | 0.096 | 5.53 | 1,859 |

The last row is the same arm rerun on the fixed code (section 9), 2026-10-17 08:23-08:31 UTC, with
the held-out set capped per player: 4,053 decisions of 30 players, none above 245 (6.0% of the set;
the test rebuild has too few held-out players to fill 8,192 at the cap). Training figures are those
of the first run within noise (entropy 0.428 0.450 0.446 0.444 0.454 0.450 0.454 0.467 0.478 0.474
0.483 0.486 0.472 0.489 0.494 0.511 0.500 0.510 0.511 0.496 0.500, under 0.52 throughout), hum_err,
hum_nonfinite, hum_drop and hum_wait 0 in every update, the same cost a minibatch. The held-out
figure is not: on the capped set agreement starts at 0.513 (per player 0.520), not 0.561. The
uncapped set's 4.8 points came from the one account that was 80% of it. Compare the live run's
`hum_ho` only with itself on the capped set.

Entropy, update by update (off holds 0.43-0.48):

```
off    0.439 0.436 0.456 0.455 0.447 0.426 0.433 0.454 0.455 0.451 0.454 0.457 0.475 0.457 0.475 0.459 0.468 0.441 0.441 0.453
0.001  0.436 0.438 0.451 0.456 0.474 0.447 0.454 0.456 0.470 0.464 0.457 0.450 0.448 0.461 0.458 0.456 0.472 0.476 0.468 0.478
0.003  0.441 0.441 0.457 0.462 0.456 0.466 0.462 0.468 0.459 0.500 0.478 0.480 0.472 0.477 0.483 0.489 0.483 0.492 0.478 0.491
0.012e4 0.438 0.453 0.451 0.465 0.474 0.483 0.490 0.466 0.486 0.495 0.486 0.499 0.488 0.488 0.509 0.492 0.499 0.500 0.489 0.493
0.01   0.437 0.482 0.496 0.519 0.499 0.535 0.523 0.536 0.536 0.551 0.562 0.564 0.561 0.584 0.588 0.597 0.598 0.583 0.583 0.600
0.02   0.458 0.494 0.506 0.507 0.522 0.560 0.553 0.568 0.580 0.608 0.602 0.607 0.603 0.619 0.623 0.654 0.633 0.628 0.646 0.652
```

- **KL, clip, the value loss, the heads: sane in every arm.** KL 0.019-0.021 and the value loss
  2.12-2.15 as off; the opponent-action and SetHead accuracies as off (0.877-0.891, 0.809-0.818);
  hum_err 0 and hum_wait 0 in every update of every arm; no NaN. Fewer early stops with the larger
  coefficients (the flatter policy moves less per step on the rollout rows).
- **The term's first effect is calibration, and it flattens the policy.** The live network is
  over-confident on the top players' decisions: CE 2.1 nats at 55% agreement (held out 1.85-1.90).
  The cross-entropy brings that CE down fast (0.02: 2.10 to 1.47 in 20 updates) and the self-play
  entropy goes up with it: 0.44 to 0.60 at 0.01 and 0.65 at 0.02, still rising at update 20, past
  stage3's turn-off line (0.55) within 10 updates, which is about a minute of the live run. At 0.003
  (and 0.012 every 4th minibatch) it rises by 0.03 to about 0.49; at 0.001 by 0.02, near off's noise.
- **Agreement with the held-out players barely moves in 20 updates**: within 0.5 points at every
  coefficient but 0.02 (+0.7). Their CE falls with the coefficient (1.85-1.90 to 1.41-1.85): that is
  the calibration, not new picks. This is stage3's finding again (HUMAN.md 9.2: "NLL is mostly
  calibration"; the shuffled control's NLL equals the real one's).
- **The cost is per human minibatch, not per coefficient**: +0.031 to 0.033 s each (0.119-0.122 s
  against 0.089 s), so +1.6 s on an update of 50 minibatches with the term on every minibatch, +0.4 s
  with every 4th. The decisions a second on box c do not show it (its workers, not its learner, set
  the pace: 1,762-1,885 in every arm).

## 5. What it costs the live run

The live learner (its own L40S) takes 5.54 s per update at 49.9 minibatches on average, and the
updates come 6.09 s apart: 6,208 decisions/s (the last 300 updates of metrics.jsonl, 2026-10-17 06:20
UTC). It is busy 91% of the time. The term on every minibatch adds about 1.6 s (7.1 s an update): the
learner becomes the limit, about 5,200 decisions/s (-16%). On every 4th minibatch it adds about 0.4 s
(5.9 s): still inside the present 6.09 s between updates, so little or nothing. The loader thread runs at nice +10 and holds about
2.5 GB in the learner (box a has 200 GB free); the shards take about 1.3 GB on disk for 40,000 games
(the test rebuild: 67 MB for 2,106).

## 6. The coefficient: 0.012 on every 4th minibatch (0.003 a minibatch)

```
--human-imitation-coef 0.012 --human-imitation-every 4 --human-imitation-data /home/ubuntu/dev/human_shards
```

Switch weight 1, double-switch weight 1, cap 0.03, pool 120,000, held out 8,192 every 50 updates, the
SetHead part off, real labels. Why:

1. **Stage3's evidence says the term is not a proven gain, so it is sized for safety.** Fine-tuning
   the old network on 1.1M top-player decisions raised held-out agreement 54.8% to 59.1% and did not
   win more (47.0% of 2,100 games against its start; a shuffled-label control 48.3%). Inside PPO at
   0.02 (with switch x2 and double switch x5), at 1.1e-5 to 6e-5: 43% and 46% against the control,
   real labels against shuffled 49-51, one arm down to 90% against SimpleHeuristics, held-out
   agreement up the same with real and shuffled labels, entropy 0.45-0.59 to 0.60-0.78 in 20 updates.
   Its advice: not in the live run as it stood; if tried, beside a shuffled arm, judged by games.
2. **The live rate makes the same coefficient stronger.** Here the rate is 3e-4, 5 to 27 times
   stage3's. At 0.02 and 0.01 the entropy left the band within 10 updates and kept climbing (section
   4). Stage3's coefficient is not usable on this run.
3. **0.003 a minibatch is the largest tested that keeps the run in its band**: entropy +0.03 over 20
   updates and flattening (0.49), KL, clip, value loss and the heads as off. 0.001 does almost nothing
   to the CE (2.10 to 2.06). 0.012 on every 4th minibatch behaves as 0.003 on every one (entropy 0.465
   to 0.495, CE 2.02 to 1.96) and costs a quarter (0.4 s an update, not 1.6 s), on a learner that is
   the run's limit.
4. **Weights 1**: stage3's switch x2 and double switch x5 came with the arm that lost against
   SimpleHeuristics; there is no evidence for them here.
5. **No SetHead part**: section 2.

What 20 updates cannot say: where the entropy settles after thousands of updates, and whether the
picks (not the calibration) move toward the top players' and make the bot stronger. The rules below
decide that on the run.

## 7. Switching it on in the live run (box a), and back

Prerequisites: the full rebuild of the 40,000 games is synced to `s3://<your-bucket>/human/shards/`
(human/rebuild_all.sh on box d prints "shards synced"); the test rebuild alone (48,762 decisions, all of
them in the pool at once) would be repeated every few minutes. Look at the rebuild's own check too
(`obs_nonfinite` should be 0): rebuild_all.sh syncs whatever its exit code, and although the learner
drops such episodes at load (`hum_drop`), a count above a handful means the rebuild has a bug worth
knowing before training on it.

```
A=$(bash C:/Users/you/ps-ppo-v2/cloud/v2/boxssh.sh a)      # the ssh command
# 1. the release: v2-speed's files (heuristic cache included) with this branch over them; nothing running is touched
$A 'test ! -e ~/rel/v2-humanaux && cp -a ~/rel/v2-speed ~/rel/v2-humanaux'
git -C C:/Users/you/ps-ppo-v2-humanaux archive --format=tar v2-humanaux | $A 'tar -x -C ~/rel/v2-humanaux && echo "v2-humanaux $(date -u +%FT%TZ)" > ~/rel/v2-humanaux/DEPLOYED'
# 2. the data
$A 'mkdir -p ~/dev/human_shards && nice -n 15 aws s3 sync --only-show-errors s3://<your-bucket>/human/shards/ ~/dev/human_shards/ --exclude "*" --include "shard-*.npz" --include "manifest.json"'
# 3. run.json: repo_dir and three arguments of the PPO phase, nothing else; the old file kept
$A 'cd ~/runs/v2/scratch && cp -p run.json run.json.pre-humanaux && ~/venv/bin/python -c "
import json; p=\"run.json\"; r=json.load(open(p))
r[\"repo_dir\"]=\"/home/ubuntu/rel/v2-humanaux\"
r[\"phases\"][\"ppo\"][\"extra_train_args\"]+=[\"--human-imitation-coef\",\"0.012\",\"--human-imitation-every\",\"4\",\"--human-imitation-data\",\"/home/ubuntu/dev/human_shards\"]
json.dump(r,open(p,\"w\"),indent=1)"'
# 4. the check: must end "OK to start", with "human data: N of N shards fit this run's observation; ..."
$A '~/venv/bin/python ~/rel/v2-humanaux/cloud/check_restart.py --config ~/runs/v2/scratch/run.json --run-dir ~/runs/v2/scratch'
# 5. apply (the supervisor reads run.json at its start): stop, keep the start point, start
$A 'sudo systemctl stop psppo-v2-scratch-train && mkdir -p ~/runs/v2/scratch/pre_humanaux && cp -p ~/runs/v2/scratch/checkpoints/$(ls -t ~/runs/v2/scratch/checkpoints | head -1) ~/runs/v2/scratch/pre_humanaux/ && sudo systemctl start psppo-v2-scratch-train'
```

The resume changes no rate and no counter (section 3; check_restart says so for the real file): the
term adds no parameter, so the optimizer state loads as it is.

**The first hour.** The trainer log says `human imitation: coef 0.012, 1 sequence(s) every 4
minibatch(es) ...` and `[learner] human imitation ON: <shards> shards with train sides (0 of <shards>
built otherwise ...), <N> labelled train decisions ..., held out <M> decisions ...`; every update line
ends with `hum_n=... hum_ce=... hum_agree=... hum_ho=...`, `hum_err=0 hum_nonfinite=0` and, after the
first update, `hum_wait=0`. The ON line also gives the held-out set (decisions, players, "at most 245
each, the largest x%") and the episodes dropped for a value that is not finite. Expect `hum_ce` to fall from about 2.0, entropy to rise from about 0.45 toward 0.5, the
value loss and KL as before, `sec=` up by about 0.4 s.

**Turn it off when**: `hum_nonfinite` above 0 even once (that update's loss is NaN too: the weights
are gone and the supervisor is already quarantining; go all the way back, below, and find the cause
before switching on again); entropy above 0.55 for 30 minutes (stage3's line, crossed within 10
updates at 0.01); KL above 0.05 or the value loss up by more than 0.05 for an hour; `hum_err` above 0
again and again (each one left a minibatch without the term, nothing worse; the log says why);
`hum_drop` climbing past a handful (a bad rebuild); decisions/s more than 10% under the hour before
the switch; an evaluation clearly under the run's band. **Keep it only if it earns its place**: after
6 hours `hum_ho` (held-out top-1 on the capped set: no player above 3% of it) at least 2 points above
its first value with entropy back under 0.5, and the run's evaluator (every 4 hours) level with or
better than its last two evaluations before the switch. Read `hum_ho_pp` (the per-player mean) beside
it: if the two disagree in sign, the move is a few players', not the held-out players' in general. A head to head of the newest
checkpoint against `pre_humanaux/` would see a few points (stage3 used 400 games); this line has no
checkpoint-against-checkpoint script yet (stage3's bench/stage2_proof/h2h.py builds stage3's
observation). Agreement without wins is what stage3 found.

**Back** (the term off, training on from where it is: the release before, the newest checkpoint):

```
$A 'cp -p ~/runs/v2/scratch/run.json.pre-humanaux ~/runs/v2/scratch/run.json && sudo systemctl restart psppo-v2-scratch-train'
```

The checkpoints the term wrote load in v2-speed as any other (the extra `human_imitation` key is not
read). **All the way back** to the weights before the switch: stop the unit, move (never delete) every
checkpoint newer than the one in `pre_humanaux/` into `~/runs/v2/scratch/humanaux_aside/`, copy
`run.json.pre-humanaux` over `run.json`, start the unit: the learner resumes the newest checkpoint left,
the one the switch started from.

## 8. Not measured here

- Anything past 20 updates: where the entropy settles, and whether the held-out agreement moves past
  the calibration.
- Strength: no games were played between arms. Stage3's games say the term did not help the old network.
- A shuffled-label arm at the recommended coefficient (stage3's control): it needs a second run.
- The full 40,000-game data (the collector stood at 27,315 of 40,000 at 06:18 UTC, then the rebuild);
  the smoke used the test rebuild of 2,106 games.
- The term only weights decisions by player and by switching; the set-up plays the owner asked about
  (a sun setter and Solar Beam, Tera Water Belly Drum then Aqua Jet) are a small part of the data and
  get no weight of their own. A set-up weight (setup_moves.py's moves, as the switch weight does for
  switches) would be the next flag if the term earns its place.

## 9. Review fixes (after 75e55f2 and 34e11af)

Two confirmed defects, both fixed in `human_imitation.py`, with tests (section 3), the suites and the
smoke rerun on box c (section 4).

1. **The non-finite guard did not stop the gradient** (major; the code, the docstring, section 2 and
   the old test all claimed it did). `torch.where(isfinite(term), term, 0)` zeroes only the forward
   value; the backward of log_softmax and of the layers multiplies that zero by the NaN, so one bad
   row (even a weightless history row) gave NaN gradients to every weight it reached, and the next
   clip and AdamW step (no finite check there) wrote NaN into the weights, the fp32 masters and the
   Adam moments, while the update line showed a finite loss and `hum_err=1`. The loader is seeded,
   so after the supervisor's quarantine and restart the same row would come back and the second NaN
   within six hours marks the live run failed (`on_failed: idle`). Reproduced on box c by both
   reviewers (448 of 448 and 112 of 112 weight gradients NaN). Fixed by keeping such values out
   instead of guarding after them: episodes with an inf or NaN in the observation or mask are dropped
   when a shard loads (`Shard`, CPU, counted once each in `hum_drop`); coef, both switch weights, cap
   and the SetHead coef must be finite (`HumanImitationSettings.check`, so train.py and check_restart
   refuse `inf`); the zeroing is gone, so a term that is still not finite (a network already NaN) is
   counted in `hum_nonfinite` and shows as `loss=nan` on the update it happened, where the
   supervisor's quarantine starts. `hum_err` now counts only terms that raised (left out, harmless).
   The test rebuild has no such row (0 of 131,191); the full 40,000-game rebuild has not been
   checked yet, which is why the load check matters.
2. **The held-out set had no per-player cap** (minor). It took whole shards until 8,192 decisions;
   on the test rebuild one held-out account was 80.3% of it, and modelled on the full data one
   account would be about 49% (effective players about 4). `hum_ho`, on which the keep rule rests,
   measured that account. Now no player passes `cap` of the set (245 of 8,192 at the default 3%,
   the train side's cap); a player's first episode is always taken, and the ON line prints the
   largest share. On the test rebuild the set is 4,053 decisions of 30 players. Filling 8,192 at
   245 each takes at least 34 players; the full data's held-out side has about 205 (the review's
   read of box d's index).

What did not change: the term itself on finite data (the same batches, the same loss; the training
figures of the rerun match the first run), the flags' defaults, the optimizer, the checkpoints, and
the switch-on steps in section 7 (only the turn-off and keep rules read the new fields).
