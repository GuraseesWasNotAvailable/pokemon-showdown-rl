# The league run: M24381 widened, a league of opponents, fresher data, a learning-rate restart

The league run trains the strongest network so far, main update 024381 (M24381: +72 Elo over Foul Play alone,
level with every network once the app's search is beside it), further by PPO on this code line, with the v2
inputs added by widening (it plays exactly as M24381 on day one), against a league of opponents instead of
itself alone, on fresher data (worker queues a fifth as long, so when the learner is the slower side its data
is about 1 update old from the queues instead of 4 to 6), and with a learning-rate restart that spends most of
its 300M decisions coming down. Branch `league` (from `topstart`), worktree
`C:\Users\you\ps-ppo-v2-league`. Nothing has been launched on box a; the owner's session launches it
(section 8).

| | |
|---|---|
| Base | M24381 (sha256 1cd07632...aa1c3b) widened by `tools/widen_checkpoint.py --main-line` to `M24381w.pt` (sha256 7254bb90...0265af, `s3://<your-bucket>/league/start/`), on the main line's own observation (`--obs-main-line`) |
| New inputs | matchup block (1,068), moveset block (660), SetHead (fresh, with move and Tera layers), opponent-action head (fresh, attached in PPO), all by zero columns or new heads |
| League | self-play 60% of pairs, setupnet 15%, frozen networks 25% (M24381 35%, SFIN 25%, 4 past saves of the run 40% of the frozen battles) |
| Fresher data | `--learn-queue-episodes 32` (each worker's queue to the learner: 32 episodes, was 160), measured by `--lag-stats` |
| Learning rate | 6e-5 (2 x M24381's 3e-5), ramp 1,000 steps, held to 100M decisions, down to 1/27 at 300M (in decisions) |
| Length | 300M PPO decisions after a warmup of 0.5M to 1.5M |
| Bars | 100M, 200M, 300M: head to head against frozen M24381, 400 games; 55% or more calls it better (section 7) |

## 1. What the owner decides

| # | Decision | Recommended | Why |
|---|---|---|---|
| D1 | When to launch | after the B7 test on box a ends (ladder-pick cells on ports 11500-11599 use GPU 0 and about 13 cores; its C cells are timed searches) | the league run takes all 4 GPUs and the CPUs; both at once would bias the B7 test's search cells |
| D2 | Foul Play seats | none in this run | about 23 helper cores at 5% of pairs (the seats' own sizing rule: 7 pairs x 16 rooms = 112 Foul Play sides x 0.455 s a decision at 88k iterations / 2.2 s; box a has 48 vCPUs, nearly all in use) and no measured gain anywhere (r3: -7.5% throughput, no benchmark change; the fp10 wave arm stopped at 7M). Can be added later with `foulplay:0.03 --seat-foulplay-iterations 88000` |
| D3 | The bars' harness | `bench/league_h2h.py` (this line's players, ladder-pick's B-cell counts) | ladder-pick's players (branch decide) refuse league checkpoints (they carry `obs_settings` with `main_line`, `matchup`, `movesets`, which decide does not build). Porting `obs_main.py` to decide/ladder-pick is the follow-up for the app and for C cells with search (section 9) |
| D4 | Length | 300M | the restart's evidence is that gains come while the rate comes down (section 5); 300M at the expected 4,200 to 5,000 learned decisions a second (section 3.1) is 17 to 20 h |
| D5 | The rate | 6e-5 | section 5: KL 0.004 at 6e-5, 0.010 at 1.2e-4; 6e-5 is inside what the main line tested on its own networks |

## 2. The base: M24381 on its own observation, widened

### 2.1 Why porting the observation, and how much it was

M24381 was trained on the MAIN line's observation (release r5/r6): obs_tera.py's own Tera type (the request's
type, a disguised Zoroark's corrected, no STAB restored after a Tera), obs_abil.py's six fixes (abil_sets,
abil_announced, abil_history, item_unknown, spe_conditions, regenerator) and switch_names.py in the action mask,
8,567 numbers. This line (search 73f524b and the v2 work) has none of those and its own `own_tera_type` writes
poke-env's type and restores STAB. Same size, other bytes: played on this line's observation, M24381 differs
on every one of 12,048 recorded decisions (224 columns) and its greedy move changes on 1,118 (9.3%; the
negative control in 2.2).

The port took about an hour and is exact:
- `obs_tera.py`, `obs_abil.py`, `switch_names.py`, `randbats_abilities_gen9.json` are the app's files byte for
  byte (git blob sha equal to branch local-app's, held by tests/test_obs_main.py);
- `obs_main.py` puts their classes on `MainLineBase`, this line's `ObservationAssembler` with the blocks as
  asked, this line's own Tera write and STAB restore made no-ops (obs_tera sets the attribute
  `own_tera_type` to True after the base's `__init__`, which the base reads), and the app's
  `create_action_mask` (switch_names) line for line. MRO: AbilityObservationAssembler, OwnTeraObservationAssembler,
  MainLineBase, ObservationAssembler. The overlays only touch the first part; the blocks follow;
- `ObsConfig.main_line` (`train.py --obs-main-line`, which sets `own_tera_type` False: the main line's own
  Tera type is part of it), `RunConfig.make_assembler`, the training line's `hybrid.PolicyNet` (evaluate.py,
  gauntlet.py), check_restart's observation settings, the learner (refuses a checkpoint whose `main_line`
  differs from the run's; writes the `obs_settings` entry the main line's learner wrote, with `main_line`,
  `matchup`, `movesets` named, so the window app refuses the network instead of playing it on 8,567 numbers);
- `tools/widen_checkpoint.py --main-line`: the matchup block (with a fresh SetHead) then the moveset block, both
  by zero columns in model, target_model, the fp32 masters and both Adam moments; `run_cfg.model.ff_expansion`
  set to the weights' 2.0 (the main line's config said 4.0, its layers were built at 2.0, and this line builds
  them from the config); `run_cfg.obs` main_line True, own_tera_type False; a `widened_main_line` record.

### 2.2 The proof: the training worker against the window app on 300 recorded battles

`bench/league_exactness.py` replays box c's 300 recorded battles (`~/dev/streams/streams.jsonl`, 12,048
decisions, the app support's test set) through this line's training worker (`worker.SyncBridgePlayer` with
worker.py's poke-env patches and the assembler `RunConfig.make_assembler` builds from the checkpoint) and this
line's network as the learner builds it (aux heads attached, every tensor loaded strictly), one decision at a
time with its history; against the window app's own player (branch local-app's `bench/v2_exactness.py app`)
on the original M24381. Box c, 2026-10-20 13:16 to 13:45 UTC, `~/dev/league/exact/`:

| a | b | compared | observation | logits | value logits | expected value |
|---|---|---|---|---|---|---|
| league worker, M24381w (fp32) | app, M24381 | 12,048 | first 8,567 numbers identical on 12,048 | bit identical 12,048 | 12,048 | 12,048 |
| league worker, M24381 on main_line without blocks (no widening) | app, M24381 | 12,048 | all 8,567 identical | 12,048 | 12,048 | 12,048 |
| league worker, M24381w | the same, not widened | 12,048 | prefix identical | 12,048 | 12,048 | 12,048 |
| inference actor's bf16 recurrent path, M24381w | the same path, not widened | 12,048 | prefix identical | 12,048 | 12,048 | 12,048 |
| this line's players (`players.batched_policy_player`, hybrid.PolicyNet: what evaluate.py and gauntlet.py play), M24381w | app, M24381 | 12,048 | prefix identical | 12,048 | 12,048 | 12,048 |
| NEGATIVE CONTROL: league worker, M24381w on this line's own observation (own_tera_type, no main_line) | app, M24381 | 12,048 | 0 identical (224 columns differ) | 0 identical; greedy move equal on 10,930 (90.7%) | 0 | 0 |

So the widened network plays exactly M24381 on day one, on the training path, the inference path and the
evaluation path, and the comparison sees a wrong observation at once.

### 2.3 The start file

`league_setup.py` keeps `M24381w.pt` as it is for the frozen seat and writes the run's first checkpoint
`learner_update_000000.pt` as a LEAGUE START: the same model and target_model tensors (checked equal), counters
at 0, no optimizer, `run_cfg.learner.mode` "imitation". Found by check_restart in the rehearsal's setup: left at
M24381's mode "ppo", a PPO start from it would be a resume at 900,057,460 decisions with the rate at the floor
(`DO NOT START ... factor of 0.20`, exit 3). With the league start, warmup and PPO both see a phase change
("the checkpoint is of phase 'imitation' and the start is 'warmup' / 'ppo'", exit 0), as run T's start did.

## 3. The league

### 3.1 What plays in the other seat

| Pairs | Share | Opponent | Trained sides per battle | Why (evidence) |
|---|---|---|---|---|
| self-play | 60% | the network itself | 2 | the run's main signal (S and T: 85%) |
| setupnet | 15% | the network, a set-up rule's move with p 0.5 (max stage 4) | 1 | as S and T, with `--boosted-row-weight 2`: 57% of the bot's ladder losses are one boosted foe sweeping (REPORT_LADDER_AND_BASELINE_RUN.md); the gate is 55 to 80% (OPPONENT_SEATS.md: 59.5%) |
| frozen | 25% | M24381 35%, SFIN 25%, 4 past saves 40% (10% each) | 1 | below |

- **Frozen M24381 (8.75% of pairs).** The anchor: the run's own start, which it must keep beating, and the bar's
  opponent, so a regression shows in training before any evaluation (the `[Seats]` line counts it per net).
- **SFIN (6.25%).** The strongest network of another line (the from-scratch v2 run's 600M final, +47 Elo
  against Foul Play, 48.8% head to head against M24381 in ladder-pick): an opponent at parity with another
  style. It plays on ITS observation (this line's own Tera type and both blocks, no main line; the seat builds
  it with an assembler of its own).
- **Four past saves (10%).** The run's own checkpoints of the PPO phase (not the start or warmup's saves, which
  play as M24381), re-picked every 30 minutes uniformly from those at least 10 minutes old
  (`--frozen-past 4:0.40 --frozen-past-pick uniform --frozen-refresh-min 30`; until the first PPO save, the
  newest file there is): fictitious
  self-play against its own history, so a strategy that only beats the current self cannot cycle. A battle
  keeps the net it started with (its history is that net's); a refresh changes what new battles draw; a net no
  battle holds is dropped.
- **Not in it: PSBot** (M24381 beats it about 83%: too weak to teach) and **Foul Play** (D2: about 23 helper
  cores at 5% of pairs at 88k iterations, no gain measured anywhere).
- **25% frozen** is OpenAI Five's 80/20 of self-play and past selves, with the setupnet seats as a further
  15%. More would cut the data further: a frozen or setupnet pair trains one side, not two, so with 40% of the
  pairs taken the trained decisions per battle are 0.6 x 2 + 0.4 x 1 = 1.6 of 2 (80%). On box a the learner
  takes 5,500 to 6,200 decisions a second of about 6,150 the games supplied (run T, SPEED.md). Against run T
  (setupnet 15%, so 1.85 trained sides a battle) the league trains 1.6 / 1.85 = 86% as many sides per battle.
- **The main line's observation costs CPU too.** Replaying the 300 recorded battles through the worker's two
  players (12,048 decisions, the same harness and box): 17.7 s with the main line and both blocks, 13.8 s with
  this line's own observation and both blocks, 10.8 s for the main line without blocks: obs_abil's fixes cost
  about 0.32 ms a decision (most of it the 15 stat bounds of every Pokemon recomputed), the blocks 0.58 ms. Obs
  building is 51 to 61% of a rollout worker's CPU (the speed profile of 2026-10-17), so the games' supply falls
  by roughly 10 to 15% more.
- **The frozen pool's own cost** (section 7.4, box c): 3% of the games' decisions a second.
- **Expected: the run is rollout-bound at about 4,200 to 5,000 learned decisions a second** (T: 5,500 to 6,200;
  0.865 x 0.97 x 0.85 to 0.9 of T's supply), so 300M takes 17 to 20 h. Measure it in the first hour
  (section 9); if it is under 4,000 the levers are the frozen share (25%: each point is about 0.3% of the speed
  and a bit of the league), the number of past slots (fewer nets a call) and the moveset block's cost.

### 3.2 How a frozen net plays (frozen_pool.py, opponent_seats.FrozenSeatPlayer)

- The nets live in frozen actors (`FrozenPoolActor`, made only with a frozen share): one per inference actor,
  on that actor's GPU with half of its share (box a: 3 inference actors at 0.5 and 3 frozen actors at 0.5 beside
  the learner's 1.0; box c: 0.175 and 0.175 beside 0.65), asked by that inference actor's workers. Each net is its
  own module built by the run's `make_model`, loaded once from its file, every parameter `requires_grad` False,
  every call `no_grad`. The frozen actors never talk to the weight store, the inference actors are plain
  InferenceActors as always, and the learner never sees the modules. (First built inside the inference actors,
  the pool took 57% of the actor's time on box c: section 7.4. In a process of its own it waits for nothing.)
  A checksum of every net's parameters is taken at load and checked on every `[Frozen]` line ("frozen weights
  unchanged", or `FROZEN NETS CHANGED: [...]`).
- The seat draws its net at its first decision of a battle (the slot weights, a random stream of its own seeded
  by the run's seed, the port and the pair, so a restarted worker draws the same), builds the observation of
  that net's own run_cfg, and asks the worker's `FrozenClient`: one thread that batches the frozen seats'
  questions (2 ms, and whatever waits while a call is out), asks the worker's frozen actor once
  (`infer_frozen`), and answers each future on the players' loop. The rollout loop and its training-inference
  calls are not touched. The actor samples at temperature 1 under the observation's own mask, as self-play does;
  only the action comes back. A seat keeps its draw for 16 battle starts, so a pair's rooms mostly play one net
  at a time (a call then needs about 1.5 forwards, not one per net); over the run every net keeps its share.
- Nothing of a frozen decision reaches the learner: no STEP, DONE or IMITATION is put on the worker's queue;
  no episode slot is taken; the tags are the seat's own (username and battle), in the pool's own history
  table. When the pool does not answer (an error or a 30 s timeout) the network plays the rest of that battle
  as for Foul Play's hand-over: STEP events and one CLEANUP at the end, so those steps are dropped, and the
  game is a mixed one (not counted for the net).
- Lines every 30 s: `[Seats] ... frozen: pairs N, games G, network won X% ..., by net: m24381 ..., sfin ...,
  past0 ...` and `[Frozen] D decisions in C calls (F forwards, B s of the actors' time, M ms a call); battle
  sides started by net: ...; K nets held in A actors, R picks of the past saves; past slots now [...]; frozen
  weights unchanged`.

## 4. Fresher data

**Where the lag comes from.** A worker hands an episode to the learner when the battle ends, into a queue of
`learn_max_pending_batches x learn_max_episodes` = 5 x 32 = 160 episodes that drops its OLDEST when full; a
thread takes up to 32 at a time from the front (the oldest) and waits on the learner (`submit_packed_batch`,
whose own queue holds 5 batches). While the learner is the slower side every queue is full, so the learner
trains on episodes about a whole queue old: 36 workers x 160 episodes x about 40 decisions is about 230,000
decisions (6.2 updates of data), which the games refill in 230,000 / their rate seconds; in updates that is
6.2 x (the learner's rate / the games' rate): about 4 updates on box a before v2-fast (4,000 learned of 6,150),
about 6 when the two are nearly level. On top of that, and whatever the queues, a decision waits for the end of
its battle: about 15 to 30 s, 2 to 5 updates on box a. When the games are the slower side (the league run is
expected to be: section 3.1) the queues stay near empty and only the battle's length counts.

**The flag.** `--learn-queue-episodes 32` makes each worker's queue 32 episodes (the in-flight batch and the
learner's queue are as before), so when the learner is the slower side its data is a fifth as old from the
queues (about 1 update instead of 4 to 6), and more of the old is dropped instead. When the games are the
slower side the queues stay near empty either way; then the flag drops a few episodes while an update runs
(box c: 2 to 5% at 32, a worker making about 6 episodes a second through a 7 s update).

**The measurement.** `--lag-stats` adds to every update line: `kl0` (the approx KL of the update's first
minibatch before any optimizer step: the behaviour policy's log-probabilities against the policy the update
starts from, so it holds the lag and the inference/learner mismatch of history and bf16), `age_s`, `age_p50`,
`age_p90` (seconds from each decision to the update) and `lag_mean`, `lag_p50`, `lag_p90` (how many weight
pushes of the learner came after the decision: a lower bound by at most one, since a push is stamped when the
learner hands it over and the inference actors load it up to 1 s later); and every 30 s a `[Lag]` line with the workers' queues (episodes
handed in, dropped as the oldest, held now). A single minibatch's kl0 is noisy (+-0.002): read it as a mean over
tens of updates.

The A/B: section 7.5.

## 5. The learning rate and the length

**Schedule:** `--lr 6e-05 --lr-hold-decisions 100000000 --lr-total-decisions 300000000` (the rate counted in
decisions, topstart-lr-decisions): a ramp over the first 1,000 optimizer steps (about 10 updates), 6e-5 to
100M decisions, then 6e-5 x 1/(8p+1)^1.5 with p = (D - 100M) / 200M, 2.2e-6 (1/27) at 300M. The zone
multipliers stay (PPO: backbone 0.5, policy 1.0, value 2.0). **Length: 300M PPO decisions.**

The evidence, all of it from this project's runs:

| Run | Rate | What happened |
|---|---|---|
| baseline run (B22640 line) | at its 1.11e-5 floor, +164M | flat: Foul Play 58.5% against 60.0% at its start, KL about 0.002 ("the LR on its floor so the run could not move") |
| r5 (the main line, M24381's own) | 3e-5 held to 774M | "no collapse and no gain" (59.0%) |
| STAGE2 (main line, new inputs switched on) | 3e-5 and 6e-5 | both recovered the new-input dip alike (3e-5 was chosen); at the floor it "did not recover at all in 20 updates" |
| r2b | a glide down | "the only clean training gain": Foul Play 48.8 to 56.8 (+8.0, z 2.9) |
| S (from scratch) | 3e-4 to 357M, then down to 1/27 at 600M | the anneal paid: Foul Play 43.7% (281M) to 52.3% (366M); rating S330 to SFIN +113 Elo of it during the decline |
| scratchx | the floor 1.1e-5, 600M to 800M | no gain (S800 -26 Elo against SFIN) |
| T (top players first) | as S | its last three Foul Play points against the three before them +7.8 (p 0.007), during the anneal |
| the waves (from B22640) | 3e-5 held | no arm clearly ahead of its control at about 39M |

What follows: holding a rate adds nothing measurable in either direction; the gains came while the rate came
down; at the floor nothing moves. So the restart holds only long enough for the new inputs (zero columns, a
fresh SetHead and opponent-action head) and the league to take effect, then spends two thirds of the run
coming down. **6e-5** is twice the rate M24381 was trained at and inside what STAGE2 tested on the main line's
own networks (3e-5 and 6e-5 recovered alike). 3e-4 (S's) is for random weights: T's start lost agreement
quickly at it. **300M** gives 200M of decline (S's decline was 243M) and fits one long day of box a.

**The probe** (box c, from the league start, the league seats, 2026-10-20): KL grows with the rate and the
entropy does not move at either rate; no update stopped early on KL.

| Rate | Updates | KL | clip | entropy | kl0 |
|---|---|---|---|---|---|
| 6e-5 (the rehearsal, updates 11 to 20) | 10 | 0.0038 (0.0026 to 0.0045) | 0.095 | 0.428 | 0.0013 |
| 1.2e-4 (`probe-lrhi`, updates 11 to 13; 9.2e-5 to 1.15e-4 at updates 8 to 10: 0.0065 to 0.0076) | 3 | 0.0097 (0.0090 to 0.0105) | 0.159 | 0.440 | 0.0021 over all 13 |

Both are well under the 0.02 target (the stop is at 1.5 x 0.02 per epoch). 1.2e-4 moves the policy about 2.5
times as far an update and is untested beyond a few updates on this network; 6e-5 is inside what the main line
itself tested. **D5 (the owner):** 6e-5 (recommended), or 1.2e-4 for a faster but riskier restart (the bars'
stop rule at 100M is the guard either way).

## 6. Off by default

Every change is behind a flag that is off by default: `--obs-main-line`, `--opponent-seats frozen:...` with
`--frozen-*`, `--learn-queue-episodes`, `--lag-stats`, `widen_checkpoint.py --main-line`. Off:

- **The probe** (`bench/league_offcheck.py`, run from branch topstart's code 3dba0dd and from league's, box c,
  `~/dev/league/off/`): for run T's warmup and PPO arguments, run S's, the waves' seat arguments (setupnet,
  psbot + foulplay, fp10), a learning-rate restart and no arguments: the RunConfig as a checkpoint saves it, the
  lines train.py logs (features, ppo settings), the opponent seats' plans for 36 x 4, 6 x 4 and 3 workers and
  their log lines, the speed options, the learner's keywords, the seats' line, check_restart's observation
  settings, a learner checkpoint's keys and run_cfg, the worker queue's size (160), the observations the
  training worker builds on 60 recorded battles (2,400 decisions, sha256 of every byte) for four observations
  (own Tera type, + matchup, + movesets, without own Tera type), and two PPO updates on those observations on
  the GPU (fast_update off and on) with their figures and the weights after them (sha256). **Equal, but for one
  string**: the seats config's repr gains `frozen=None` (a new dataclass field; nothing logs it, its plan and
  every line from it are equal). The same code twice gives equal files (the GPU update is deterministic here),
  so the comparison can see a difference.
- **The test suites** (box c, `~/dev/league/suite/`): the whole suite of branch topstart (3dba0dd) and of
  league (29d7149, then the review's fixes), `pytest tests --continue-on-collection-errors`. topstart: 1,041
  passed, 118 skipped, 3 failed, 1 error; league: every one of those the same, plus its 36 new tests passed. The
  4 that fail in both are the known ones (test_tera's inductor C++ compile error on the CPU, test_ppo_health's
  collection). The run also caught one real regression, fixed: two tests of test_extras build a
  `SyncLearnerClient` without its `__init__`, and the new episode counters raised on them (now class defaults).
  **The final commit (c9dd9be, after the review's fixes and the frozen actors), the whole suite again**
  (2026-10-20 15:00 to 15:18): 1,089 passed, 118 skipped, 3 failed, 1 error: every one of topstart's 1,163
  tests with topstart's outcome, and the 48 new ones passed (`~/dev/league/suite/final.xml`).
- `RunConfig.as_dict` leaves `main_line` out when it is off, so every run's checkpoints carry the run_cfg they
  did; `OpponentSeatsConfig` gets a `frozen` field (None); `SyncLearnerClient` counts handed-in and dropped
  episodes (two integers, reported only with `--lag-stats`); the learner's `_lag_info`, `obs_settings` entry
  and main-line refusal run only with their flags (a run without main_line loading a checkpoint without it
  never reaches the refusal); `ppo_update(record_first_kl=True)` is passed only with `--lag-stats` (popped from
  its keywords, a tensor kept from the first minibatch, read once after the loop).

## 7. The bars (fixed before any training)

**The measure.** Head to head, networks alone, against frozen M24381 (`M24381w.pt`, the run's own start, which
plays exactly as M24381: section 2.2), with `bench/league_h2h.py`: ladder-pick's B cells (300 games with the
league network through ours, the app's player, against M24381 through the batched player, then 100 with the
two swapped; each cell's sides balanced), scored as the league network's wins out of 400 with a 95% Wilson
interval (about +-4.9 points at 50%). On **box c** from the S3 copies of the run's checkpoints (the backup
copies `~/runs/v2/league` every 10 minutes), so box a is never touched for them. Beside each B: Foul Play
alone at 88,000 iterations, 400 games (`cloud/v2/topstart_eval.sh` with the league release), for the record.

| Bar | Checkpoint | Better | Go on | Stop |
|---|---|---|---|---|
| L100 | the first at or after 100M PPO decisions (the end of the hold) | >= 55% | over 45% and under 55% | <= 45%: the restart hurts. Stop (section 10) and report |
| L200 | the first at or after 200M (half way down) | >= 55% | over 45% | <= 45%: stop and report |
| L300 | the final (300M) | >= 55%: BETTER. Then the app port (section 9.3), the C cells with the app's search and the ladder | 50% to 55%: level, not better; keep M24381 on the ladder | under 50%: worse; keep M24381 |

Expected from the evidence: L100 about 50% (a hold at a modest rate moves little: r5, the waves), L200 and L300
50 to 56% (the decline is where the earlier runs gained). For information, at L300 also B against SFIN (400).
The `[Seats]` line gives a free running read of the same thing during training (the trained network against
frozen M24381 at temperature 1, both sides sampling): it should drift above 50% if the run improves; it is
not a bar.

Commands (box c; `C=$(bash C:/Users/you/ps-ppo-v2/cloud/v2/boxssh.sh c)`; NNNN the checkpoint's update):

```
$C 'mkdir -p ~/dev/league/bars && aws s3 cp s3://<your-bucket>/main-g6e12/runs/v2/league/checkpoints/learner_update_00NNNN.pt ~/dev/league/bars/L100.pt --only-show-errors'
$C 'cd ~/dev/league/rel-final && setsid nohup ~/venv/bin/python bench/league_h2h.py run --new L100=$HOME/dev/league/bars/L100.pt --ref M24381=$HOME/dev/league/ckpt/M24381w.pt --root $HOME/dev/league/bars/L100 --port 11450 > $HOME/dev/league/bars/L100.log 2>&1 < /dev/null &'
$C 'cd ~ && REL=~/dev/league/rel-final ALLOW_PORT=1 setsid nohup bash ~/dev/league/rel-final/cloud/v2/topstart_eval.sh L100 ~/dev/league/bars/L100.pt 11460 0 > /dev/null 2>&1 < /dev/null &'
$C 'cd ~/dev/league/rel-final && ~/venv/bin/python bench/league_h2h.py report --root ~/dev/league/bars/L100; cat ~/dev/topstart/eval-L100/runs.log'
```

The checkpoint's decisions: its line in `s3://.../main-g6e12/runs/v2/league/metrics.jsonl` (`total_steps` of the
PPO phase). A checkpoint is saved every 200 updates (about 7.4M decisions).

The harness ran once on box c (2026-10-20 14:01 to 14:04, `~/dev/league/h2htest`): the rehearsal's update 20
against M24381w, 40 + 20 games: 33 of 60 (55.0%, 95% 42.5 to 66.9; noise at that size), both cells complete,
no invalid order, exception or forced random move. A game of networks alone took 3 to 4 s: 400 games take about
20 minutes at 6 in parallel.

## 7.2 The rehearsal (box c, 2026-10-20 13:40 to 14:00 UTC)

`~/dev/league/rehearsal`, built by `league_setup.py --rehearsal --base builtin` from release a850b2a: the real
run's arguments at box c's size (6 servers on 11400-11405, 384 battles, one L40S: learner 0.65 and one
inference actor 0.35, `--infer-actors 1`), warmup 100k to 300k, PPO with the league seats, queue 32, lag stats,
lr 6e-5 (hold 1.5M of a 3M schedule), the frozen pool re-picked every 5 minutes from saves 2 minutes old. The
whole test suite of both branches ran beside it, so its speed is box c's under load, not box a's. This was the
code before the review (the frozen pool then inside the inference actor, past saves of every phase, draws per
battle); the arms of sections 7.4 and 7.5 ran the final design (release 0d01fce, the same Python as c9dd9be).

| | |
|---|---|
| Setup | the checks of 8.4 all exit 0 (check_restart warmup and ppo: "phase 'imitation' and the start is ..."; `--check-run`: 6 nets, frozen seats OK; can-start 0) |
| Warmup | `PHASE CHANGE DETECTED (imitation -> warmup)`; vloss 2.09 at update 1, 2.06 at update 7 (M24381's value head fits self-play at once); 8 updates, 302k decisions |
| PPO start | `PHASE CHANGE DETECTED (warmup -> ppo)`; seats: setupnet 4 of 24 pairs, frozen 6 of 24, the network p1 in half of each; frozen nets m24381 35%, sfin 25% on its own observation (own_tera_type True, no main line), 4 past saves 40% |
| Learning rate | 5.82e-06 at update 1, +5.8e-06 an update (the 1,000-step ramp), 6.00e-05 from update 11 |
| PPO at 6e-5 (updates 11 to 20) | KL 0.0038 (0.0026 to 0.0045), clip 0.095, entropy 0.428 (0.44 at update 1), n_mb 96 every update (no KL stop), vloss 2.08, set_loss 0.75 (1.05 at update 1: the fresh SetHead learning), opp_acc 0.848 (0.64 at update 1: the fresh opponent-action head) |
| Lag (queue 32, the games the slower side) | lag_mean 0.40 updates, lag_p90 1; age_s 31 s (p90 50 s: the battle's length); kl0 0.0013 mean (-0.0008 to 0.0042 per update); `[Lag]` 0 to 4.6% of episodes dropped (4.6% in the first minute, while the queues met the first updates) |
| League | 55,894 frozen-seat decisions, every one answered by the pool (the network's 0, mixed games 0); the network against frozen M24381 51.2% of 568 (itself: 50% as it should), against SFIN 62.4% of 410 (both sampling at temperature 1), against the past saves 45 to 60%; against setupnet 57.1% of 1,010 (the gate: 55 to 80%); `frozen weights unchanged` on every line |
| Past saves | picked at the start (only the start was 2 minutes old: 4 x learner_update_000000), again 5 minutes later (000000, 000005, 000008) and after the restart (000010, 000008, 000005, 000011); battles in progress kept their net (up to 10 nets held, the old ones dropped as their battles ended) |
| Stop and resume | at update 11 by `pipeline.stop`: the trainer saved learner_update_000011.pt in 6.5 s, the servers stopped, no port left; check_restart on the stopped run exit 0 ("policy rate at the resume: 6.000e-05 ... OK to start"); the restart printed "policy rate at the resume: 6.000e-05 at optimizer step 1056 and 441,674 decisions", resumed from 000011, the pool re-picked, the seats played on |
| Checkpoints | learner_update_000020.pt: run_cfg.obs main_line, matchup, movesets True and own_tera_type False; `obs_settings` with the main line's seven and main_line, matchup, movesets; 8 set_head and 4 opp_action_head tensors; M24381w.pt's sha256 unchanged (the frozen copy is only read) |
| Faults | no NaN, no Traceback, no ERROR in the trainer logs; the warnings are the base code's ("blocking ray.get inside async actor", 5 in the warmup without seats too; a NumPy array not writable, as in the learner) |
| Speed (box c, under the test suites) | about 1,000 learned decisions a second, rollout-bound (the learner's `sec` 7.1 s of a 37 to 40 s update cycle). Not box a's number: measure it there (section 9) |

## 7.3 The independent review and what was fixed

A hostile review of the whole diff (a separate agent, reading only, 2026-10-20 13:47 to 14:08) found no
blocker. It confirmed: off is off (as_dict, obs_settings, the refusal branch, the ppo_update keyword, the
workers' keywords, LAG_STATS, the seat plans and lines, InferenceActor and the spawn path, check_restart);
the lag's direction, kl0 before any optimizer step (also with fast_update and grad_accum), the slot weights,
plan_seats' placements of the other kinds, Wilson, the eviction counts; the widening (ff_expansion repair,
target_model, SetHead); the league start's phase change in pipeline, check_restart and evaluate; that no
frozen net can be written and no frozen decision reaches the learner; SFIN's observation; the concurrency of
the client and the refresh; the run.json. Fixed after it:

| | Finding | Fix |
|---|---|---|
| S1 | the past slots drew from every save in the directory: the start and warmup's saves (which play as M24381) diluted "the run's own history" for hours | past saves are only this phase's (`run_cfg.learner.mode` read with an mmap load, 0.01 to 0.03 s a file, cached); until there is one, the newest file there is |
| S2 | the frozen calls' cost on the single-threaded inference actors was not measured; one forward per distinct net per call | measured (section 7.4): 57% of the inference actor's time. The pool moved into actors of its own (FrozenPoolActor, one per inference actor, half its GPU share); a seat keeps its draw for 16 battle starts (1.5 forwards a call instead of up to 6); the checksum is one fused read per net; the `[Frozen]` line reports forwards and the actors' busy seconds |
| M1 | the lag is a lower bound, not an upper one (a push is stamped before the actors load it) | the docstring and section 4 say so |
| M2 | a raw main-line checkpoint into a run without main_line loads silently (as it always did) | kept (off is as before), the comment made exact |
| M3 | a bad frozen net was only caught when the trainer started | `resolve` compares every tensor's name and shape with the run's network; `league_setup.py --check-run` runs it before the start (section 8.4) |
| M4 | the head to head counted a tie against the new network in one cell and for it in the other | a tie is half a game in both |
| nits | a slot of weight 0 drawn on rounding; `kl0=nan` in a mode without the PPO loss; a half-done setup left files that refuse a re-run | the last slot with a weight; kl0 printed only when there is one; every destination checked before anything is copied |

## 7.4 What the frozen pool costs (box c)

Three short arms from the league start, PPO only, 440k decisions each, the rehearsal's size (6 servers, 384
battles, one GPU), one after another on an otherwise quiet box c (`~/dev/league/probe-r2on`, `-r2off`, `-r4on`;
`rate.py` over the second half of each arm's telemetry):

| Arm | Frozen pool | Training inference, decisions/s | Frozen, decisions/s | All decisions/s | Learned/s | Frozen calls |
|---|---|---|---|---|---|---|
| r2on | inside the inference actor (the first design) | 1,367 | 204 | 1,571 | 1,249 | 5.6 decisions and 1.58 forwards a call, 15.8 ms a call: **58% of the actor's time** |
| r2off | no frozen seats (setupnet 15% only) | 1,810 | 0 | 1,810 | 1,524 | |
| r4on | its own actor (FrozenPoolActor, half of the GPU share) | 1,542 | 215 | 1,757 | 1,297 | 5.9 decisions and 1.62 forwards a call, 19.3 ms a call: the frozen actor 70% busy |

- The first design cost the games 13% of their speed (1,571 against 1,810 decisions a second); the frozen actor
  of its own costs 3% (1,757): the training inference gets its time back (1,542 against 1,367), and what is
  left is CPU (box c is CPU-bound: the frozen actor's process, the frozen seats' own observations).
- Learned decisions: 1,297 against 1,524 without frozen seats (-15%): 13.5% of it by construction (a frozen pair
  trains one side: 0.8 trained sides per decision against 0.925 with setupnet alone) and the 3% above.
- The league itself in r4on: the start network won 48.6% of 712 against frozen M24381 (itself), 54.2% of 683
  against SFIN, 43 to 55% against the past slots (the start again, early in the arm); every one of 77,406
  frozen-seat decisions answered by the pool.
- A call takes 16 to 19 ms for about 1.5 forwards of 6 decisions: a forward of this network costs about 10 ms
  whatever its batch (launch-bound, as the training inference's 10.8 ms a call). The 16-battle draws keep the
  forwards near one a call; on box a each frozen actor serves 12 workers and will run near full, batching more
  decisions a call as it fills (each client sends what waited while its call was out).
- The frozen actor is up 6 s after it is made (the 6 nets loaded, a forward of each at two batch sizes: 1.5 s);
  train.py waits for it before the workers start, so the first questions do not wait for a compile.
- The direct check of FrozenPoolActor on box c's GPU (`fpa_smoke.py`: M24381w, SFIN and a past save, 400 recorded
  observations in calls of 8 over 3 slots): 400 of 400 answers legal under the observation's mask, the
  histories dropped on forget, the weights unchanged.

## 7.5 The worker queue, A/B

Box c is the games' side's box (the rehearsal is rollout-bound: lag 0.4 updates at queue 32), so for the A/B
the learner was made the slower side: minibatch 16 and seq 16 (2,304 minibatches an epoch, about 92 s an
update, 380 learned decisions a second against about 1,000 the games make), PPO from the league start with the
league seats, `--lag-stats`. The queue sizes are box a's in data: box a's 160 episodes x 36 workers x about 40
decisions = 6.2 updates of data, so 1,240 episodes for box c's 6 workers of about 31; box a's 32 is 248 here.
(kl0 is one minibatch's KL, 16 rows here: too noisy to read in this emulation; it is 768 rows in the run.)

| Arm (updates 4 to 14) | Queue (box c / box a equivalent) | Dropped as the oldest | Data age, mean / p90 (s) | lag_mean (updates) | lag_p90 | KL of the update | Learned/s |
|---|---|---|---|---|---|---|---|
| q1240 | 1,240 / 160 | 41.8% | 200 / 258 | 2.67 (2.49 to 2.76) | 3 | 0.052 | 375 |
| q248 | 248 / 32 | 64.8% | 44 / 94 | 1.12 (1.09 to 1.16) | 1.6 | 0.040 | 377 |

- The small queue trains on data 4.5 times fresher (44 s against 200 s) at the same speed (the learner sets
  it); it drops more of the old data instead (65% against 42% here, where the learner is far slower than the
  games). The update's own KL falls with it (0.040 against 0.052): the data is nearer the policy being trained.
- In updates the lag here is the queue's time over the update's (92 s): on box a, with 6.5 s updates and the
  learner nearly as fast as the games, 160 episodes are about 5 to 6 updates and 32 about 1, plus the battle's
  own length. If the league run is rollout-bound, as expected, the queues stay near empty and the flag costs a
  few dropped episodes a minute (box c's rehearsal: 0 to 4.6%); `[Lag]` and `lag_mean` on box a will say which.

## 8. The launch on box a (the owner's session)

`A=$(bash C:/Users/you/ps-ppo-v2/cloud/v2/boxssh.sh a)`. A `~` inside single quotes is the box's home. Every
command below must exit 0 (or print what it says) before the next.

| | |
|---|---|
| Release | `~/rel/league`: a copy of `~/rel/topstart` (its `heuristic/cache`) with branch `league`'s `git archive` over it |
| Run | `~/runs/v2/league` (`league_setup.py`): run.json from run T's live one, the league start, `league/M24381w.pt`, `league/SFIN.pt` |
| Units | `psppo-v2-league-train`, `psppo-v2-league-eval` (written, not enabled, by `--units`) |
| Ports | servers 10900-10935, evaluator 10990 (free on 2026-10-20 13:42; the B7 test uses 11500-11599) |
| GPUs | trainer env CUDA 0,1,2,3, PSPPO_GPU_SPLIT=1: learner a GPU, 3 inference actors a GPU each, the frozen nets inside them |
| Disk | 300M at save_every 200: about 41 PPO checkpoints of 229 MB (9.4 GB) plus warmup's; 66 GB free at 13:42. Do not start under 30 GB |
| Time | warmup about 10 minutes; PPO 300M at the rate measured in the first hour (4,200 to 5,000 learned decisions a second expected: 17 to 20 h) |

### 8.1 Before anything (read only)

```
$A 'pgrep -af "[l]adderpick|[g]auntlet.py" | head -3; ss -ltn | grep -cE ":(109[0-9]{2}|115[0-9]{2})\b"; df -h / | tail -1; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; cat ~/runs/v2/guard.env; ls ~/dev/KEEP_BOX ~/runs/v2/topstart/eval/final.done; uptime'
```

The B7 test done (no ladderpick or gauntlet process), 0 listeners on 10900-10999 and 11500-11599, 30 GB free or
more, the GPUs idle, `KEEP_BOX` there, run T's evaluator done (`final.done`).

### 8.2 The release

```
$A 'test ! -e ~/rel/league && cp -a ~/rel/topstart ~/rel/league && echo copied'
cd C:/Users/you/ps-ppo-v2-league && S=$(git rev-parse --short league) && git -c core.autocrlf=false archive --format=tar league | $A "tar -x -C ~/rel/league && echo 'league $S $(date -u +%FT%TZ) (git archive, LF, over a copy of ~/rel/topstart)' > ~/rel/league/RELEASE && cat ~/rel/league/RELEASE"
# the check: every tracked file's sha256 in box a's release against box c's replica of the same commit
# (~/dev/league/rel-final, the git archive of the branch head, its code that of c9dd9be, checked file for file on 2026-10-20). The diff must be empty.
C=$(bash C:/Users/you/ps-ppo-v2/cloud/v2/boxssh.sh c)
T=$(mktemp -d) && git ls-tree -r --name-only league > $T/files && $A 'cd ~/rel/league && tr "\n" "\0" | xargs -0 sha256sum' < $T/files > $T/a.sha && $C 'cd ~/dev/league/rel-final && tr "\n" "\0" | xargs -0 sha256sum' < $T/files > $T/c.sha && diff $T/a.sha $T/c.sha && echo "release == league, $(wc -l < $T/files) files"
```

### 8.3 The start files

```
$A 'mkdir -p ~/dev/league/start && cd ~/dev/league/start && test ! -e M24381w.pt && aws s3 cp s3://<your-bucket>/league/start/ . --recursive --only-show-errors && sha256sum -c M24381w.pt.sha256 && sha256sum ~/runs/v2/scratch/checkpoints/learner_update_016041.pt'
```

`M24381w.pt: OK` (7254bb90...0265af, the file the exactness test ran on) and SFIN `a1ad3087...a59a3` (the copy box
c's tests used, equal to box a's since 2026-10-18).

### 8.4 The run directory, the units, the checks

```
$A 'cd ~/rel/league && ~/venv/bin/python cloud/v2/league_setup.py --rel ~/rel/league --start ~/dev/league/start/M24381w.pt --sfin ~/runs/v2/scratch/checkpoints/learner_update_016041.pt --write --units'
$A 'R=~/runs/v2/league; P=~/venv/bin/python; cd ~/rel/league && $P cloud/check_restart.py --config $R/run.json --run-dir $R --phase warmup; echo "exit $?"'
$A 'R=~/runs/v2/league; P=~/venv/bin/python; cd ~/rel/league && $P cloud/check_restart.py --config $R/run.json --run-dir $R --phase ppo --checkpoint $R/checkpoints/learner_update_000000.pt; echo "exit $?"'
$A 'R=~/runs/v2/league; cd ~/rel/league && ~/venv/bin/python cloud/v2/league_setup.py --check-run $R; echo "exit $?"'
$A 'R=~/runs/v2/league; cd ~/rel/league && ~/venv/bin/python evaluate.py --run-dir $R --check-config $R/run.json; echo "exit $?"'
$A 'R=~/runs/v2/league; P=~/venv/bin/python; cd ~/rel/league && $P pipeline.py --config $R/run.json --run-dir $R --dry-run | head -20; $P pipeline.py --config $R/run.json --run-dir $R --can-start; echo "can-start $?"'
```

What they say: the setup prints run.json and "league start ... model and target_model equal to M24381w's,
counters 0, no optimizer, mode 'imitation' (was 'ppo' at 900,057,460 decisions)"; both check_restart runs end
"the checkpoint is of phase 'imitation' and the start is 'warmup' / 'ppo': the optimizer and the schedule
start afresh" (the PPO one prints "full rate up to 100,000,000 decisions, then down to 1/27 at 300,000,000");
`--check-run` prints the frozen nets (m24381 35%, sfin 25% on its own observation, 4 past saves 40%), "pool on
the CPU: 6 nets", "frozen seats OK". The setup refuses (exit 2) a base that is not run T's run.json, a start
that is not a widened main-line network with a SetHead, an SFIN that is not a v2 network with both blocks, a
release without the league's files, an existing state.json, checkpoints already there.

The run.json the setup writes (built from box a's live run T run.json, which equals the setup's builtin copy
of it but for `_notes`; checked 2026-10-20 13:58), `_notes` left out here:

```
{
 "repo_dir":"/home/ubuntu/rel/league",
 "python":"/home/ubuntu/venv/bin/python",
 "showdown_dir":"/home/ubuntu/pokemon-showdown",
 "servers":{"count":36,"base_port":10900,"heap_mb":2048,"ready_timeout_seconds":420,"command":["nice","-n","5","{python}","start_servers.py","--dir","{showdown_dir}","--ports","{ports}","--heap-mb","{heap_mb}","--log-dir","{server_log_dir}"]},
 "trainer":{"stop_timeout_seconds":420,"failure_stop_timeout_seconds":120,"env":{"CUDA_VISIBLE_DEVICES":"0,1,2,3"}},
 "train_defaults":{"battles":2304,"rooms_per_pair":16,"minibatch":768,"seq_len":256,"grad_accum":1,"steps_per_update":36864,"teacher":"psbot","save_every":200,"extra_train_args":["--obs-matchup","--obs-movesets","--obs-main-line","--infer-actors","3","--infer-sync-s","1","--rollout-nice","5"]},
 "phases":{
   "imitation":{"skip":true,"min_decisions":0,"max_decisions":1},
   "warmup":{"skip":false,"min_decisions":500000,"max_decisions":1500000,"plateau":{"metric":"vloss","window_decisions":300000,"min_improvement_pct":2.0,"smooth_updates":4},"train":{"save_every":5},"extra_train_args":["--set-head-coef","0.1"]},
   "ppo":{"skip":false,"target_decisions":300000000,"extra_train_args":["--set-head-coef","0.1","--opp-action-aux-coef","0.1","--boosted-row-weight","2","--opponent-seats","setupnet:0.15,frozen:0.25","--frozen-nets","m24381=/home/ubuntu/runs/v2/league/league/M24381w.pt:0.35,sfin=/home/ubuntu/runs/v2/league/league/SFIN.pt:0.25","--frozen-past","4:0.40","--frozen-refresh-min","30","--frozen-past-min-age-min","10","--learn-queue-episodes","32","--lag-stats","--lr","6e-05","--lr-hold-decisions","100000000","--lr-total-decisions","300000000","--fast-update","--fast-ingest"]}
 },
 "watchdog":{"stall_minutes":10,"startup_grace_minutes":20,"update_stall_minutes":30,"max_restarts_per_hour":6,"backoff_seconds":[30,120,300,600,900],"max_fruitless_starts":5,"nan_updates_limit":3,"telemetry_error_minutes":5,"recycle_hours":6},
 "disk":{"min_free_gb":10},
 "retention":{"enabled":false,"keep_newest":4,"milestone_every":500,"keep_hours":0,"check_minutes":10},
 "on_done":"idle",
 "on_failed":"idle",
 "eval":{"every_hours":4,"games":200,"opponents":["simple","psbot"],"port":10990,"foulplay":{"games":200,"every_hours":4,"parallel":2,"iterations":88000}}
}
```

### 8.5 Start it

```
$A 'test -e ~/runs/v2/guard.env.pre-league || cp -p ~/runs/v2/guard.env ~/runs/v2/guard.env.pre-league; sed -i "s/^ARMS=.*/ARMS=\"league\"/" ~/runs/v2/guard.env; cat ~/runs/v2/guard.env'
$A 'sudo systemctl disable --now psppo-v2-topstart-eval psppo-v2-topstart-train; systemctl is-active psppo-v2-topstart-eval psppo-v2-topstart-train'
$A 'sudo systemctl enable --now psppo-v2-league-train && systemctl is-active psppo-v2-league-train'
```

guard.env becomes `ARMS="league"` (the guard re-reads it every 5 minutes and would restart the evaluator of an
arm in ARMS, so it goes first); the topstart units only idle (T is done: its supervisor holds nothing) and go
off so a reboot does not bring them back; nothing of T's files changes. The guard starts
`psppo-v2-league-eval` 3.5 h after the mtime of the league's first trainer log (the warmup's, frozen after
about 10 minutes). `KEEP_BOX` stays. The backup unit already copies `~/runs/v2/league` to S3.

## 9. What to watch, the follow-ups

### 9.1 Commands

```
$A 'tail -3 ~/runs/v2/league/logs/pipeline.log'
$A 'L=$(ls ~/runs/v2/league/logs/trainer_*.log | tail -1); grep -E "PHASE CHANGE|schedule in decisions|policy rate at the resume|fast_ingest on|rollout nice|CPU priorities|frozen nets:|Opponent seats:|league:" $L | cut -c1-260; grep -E "\[Seats\]|\[Frozen\]|\[Lag\]" $L | tail -3 | cut -c1-700; grep -ciE "nan|traceback|REFUSED|CHANGED" $L'
$A 'python3 -c "import json; [print(r[\"phase\"], r[\"update\"], r[\"total_steps\"], r[\"vloss\"], r[\"kl\"], r[\"ent\"], r[\"clip\"], r[\"n_mb\"], r[\"sec\"], r[\"lr\"]) for r in map(json.loads, list(open(\"/home/ubuntu/runs/v2/league/metrics.jsonl\"))[-5:])]"'
$A 'L=$(ls ~/runs/v2/league/logs/trainer_*.log | tail -1); grep "\[learner\] upd" $L | tail -20 | grep -oE "kl0=[^ ]+|lag_mean=[^ ]+|age_s=[^ ]+" | paste - - - | tail -5'
$A 'df -h / | tail -1; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; uptime'
```

### 9.2 What to expect, and when to act

| When | Expect (the rehearsal and the arms on box c, sections 7.2 to 7.5) | Act if |
|---|---|---|
| Start | `PHASE CHANGE DETECTED (imitation -> warmup)`; `rollout nice 5: 36 of 36 workers`; servers up | REFUSED, a restart loop |
| Warmup | vloss about 2.1 from update 1 (M24381's value head fits at once; T's start began at 13.6); only the value head trains; ends on the plateau or at 1.5M | vloss above 4, NaN |
| PPO start | `PHASE CHANGE DETECTED (warmup -> ppo)`; `league: learn_queue_episodes=32 lag_stats=True`; `frozen nets: m24381 35.0% ... sfin 25.0% (... own_tera_type: True ...) ... 4 past saves 40.0%`; `Opponent seats: setupnet 22 of 144 pairs ...; frozen 36 of 144 pairs ...`; `learning-rate schedule in decisions ... until 100,000,000 ... 1/27 at 300,000,000`; `frozen seats: each inference actor's GPU share split in two: inference [0.5, 0.5, 0.5], frozen actors [0.5, 0.5, 0.5]`; `frozen actors ready: [{'warmup_s': ..., 'nets': 6}, x3]` | a seat kind "NOT used", `PAIRS NOT STARTED`, the frozen actors not ready after 15 minutes |
| First hour | lr ramps to 6.00e-05 by update 11; KL 0.002 to 0.005 at the full rate, clip about 0.10, n_mb 96 (no early stop), entropy about 0.43 and steady; set_loss falling (1.05 to 0.75 in 15 updates); opp_acc rising; `[Seats]` network about 50% against the frozen nets (SFIN about 52 to 56%), about 60% against setupnet; `[Frozen] ... frozen weights unchanged`, no "did not answer" | entropy under 0.30 or above 0.60; KL above 0.02 for several updates; `FROZEN NETS CHANGED`; the pool not answering (`battles the pool did not answer`) |
| Speed | 4,200 to 5,000 learned decisions/s (section 3.1): measure over the first hour from pipeline.log; `[Lag]` dropped share; `[Frozen]` ms a call and the frozen actors' busy seconds (near full is expected) | under 4,000/s: check nice levels (SPEED.md), the frozen calls (`[Frozen]` calls a minute), load |
| Every 30 min | `[Frozen] ... R picks of the past saves; past slots now [...]` (R grows by 1) | a refresh FAILED |
| Lag | `lag_mean` a few updates (the episode's length), `age_s` 20 to 40 s; dropped share a few %; kl0 small (mean over 20 updates) | lag_mean above 5 (the queues full and old) |
| 6-hourly restart | `policy rate at the resume` equal to check_restart's; the pool re-picks; the seats' counts start again | a rate that moved |
| 100M, 200M, 300M | the bars (section 7), on box c | the stop rules |

### 9.3 Follow-ups (not in this branch)

- **The app and ladder-pick for league networks.** Port `obs_main.py` (and its four files, which decide already
  has) into decide/local-app's `hybrid.PolicyNet`: a checkpoint whose run_cfg says `main_line` with the blocks
  gets `obs_main.make_assembler(matchup, movesets)` and `make_model(matchup, movesets)`, and `unknown_obs_settings`
  accepts `main_line`/`matchup`/`movesets` for it; `export_v2.py`'s naming (`millions()` treats only arm
  "scratch" as from scratch); a loadcheck rule; then the same exactness check (app on the exported league
  network against this branch's worker on the 300 streams). About 2 hours. Needed for L300's C cells with
  search and for the ladder.
- **Foul Play seats** if the owner wants them (D2): `foulplay:0.03 --seat-foulplay-iterations 88000` at a
  restart, after measuring the speed it costs.

## 10. The undo

Nothing here deletes anything.

- **Before the start:** nothing to undo. `~/rel/league`, `~/dev/league/start`, `~/runs/v2/league` and the units
  (written, not enabled) can stay.
- **Stop the run** (a stop rule, a bar, the owner's call), guard first: `$A 'sed -i "s/^ARMS=.*/ARMS=\"\"/"
  ~/runs/v2/guard.env'` (or `cp -p ~/runs/v2/guard.env.pre-league ~/runs/v2/guard.env`), then
  `$A 'sudo systemctl disable --now psppo-v2-league-train psppo-v2-league-eval'`. The supervisor stops the
  trainer through its stop file, which saves a checkpoint first (6.5 s in the rehearsal). To resume later:
  `league` in ARMS again and enable the train unit; it continues from the newest checkpoint.
- **Drop the league or the queue flag only** (a seat misbehaves, the speed is too low): stop the train unit,
  keep the old file as `run.json.pre-<what>`, take the flags out of `phases.ppo.extra_train_args`
  (`--opponent-seats` back to `setupnet:0.15` and the `--frozen-*` flags; or `--learn-queue-episodes 32`; or
  `--lag-stats`), run check_restart without `--phase` (exit 0), start the unit. Checkpoints are the same with
  or without these flags; the frozen nets are only read.
- **Never** drop `--obs-main-line` from a running league run: the learner refuses the checkpoint (it would play
  on another observation); check_restart says DO NOT START.
- **A wrong start file** (before PPO): stop as above, move `~/runs/v2/league` aside
  (`mv ~/runs/v2/league ~/runs/v2/league.aside-$(date -u +%m%d%H%M)`), and run 8.4 again. The setup refuses a
  directory with a state.json.
- **The learning-rate schedule:** never edit `--lr`, `--lr-hold-decisions`, `--lr-total-decisions` in a running
  run without check_restart (exit 3 when the rate or the schedule would move; `--allow-rate-change` only on
  purpose).
- **Back to run T's units:** `cp -p ~/runs/v2/guard.env.pre-league ~/runs/v2/guard.env`; T is finished, so there
  is nothing to restart.
