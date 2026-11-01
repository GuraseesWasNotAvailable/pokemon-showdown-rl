# Run T's start T0 and gate G1: passed, launch T from T0

**T0 passes gate G1.** Against Foul Play at a fixed 88,000 iterations it wins 38.3% (153 of 399 games), and P0
wins 15.1% (60 of 398). Both ran on box c with the same Foul Play build. T0's capped held-out top-1 is 0.5865
against the bar of 0.50. By the fixed rule (TOPSTART.md section 2), T launches from T0 on box a once scratchx is
done (TOPSTART.md section 3; disk decision D1 is still the owner's).

On bar B1, T0 passes the agreement checks: 0.5865 against the bars 0.55, 0.5076 (P0 + 5 points) and 0.5351
(SFIN + 2). Its test_game agreement is within 0.13 points of test_player. It **fails B1's overfit check**: T0
itself, measured on a fixed sample of its own train sides, agrees 0.6483, which is **6.2 points** above held-out
(the bar is 3). The training log's window reading at the chosen step (108,000) was 4.1. B1 does not hold up the
launch, but section 5 explains it. The control C0 is about as good on agreement (0.5869) and does not overfit
(+0.9). Against Foul Play it wins 33.2%, 5.1 points under T0 (95% interval of the gap -1.5 to +11.7). That gap is a
reading, not proof.

This is step D2 of TOPSTART.md (commit d52d02a). It ran entirely on box c on 2026-10-18, 08:36 to 12:28 UTC.
Nothing ran on box a or box d, apart from reading box a's Foul Play commit. Section 7 is an independent check of
every number here, from the raw files (12:31 to 12:55 UTC); it corrected the overfit figure, T0's one Foul Play
error and three rounded slice values, and found one step of the plan (3.2) that needs a fix before it runs. The
version before the check is `TOPSTART_G1.eb56a9b.md`.

## 1. What ran

| | |
|---|---|
| Box | c (1 L40S, 16 vCPU, 124 GB), GPU 0; idle before (load 0.0, GPU 0 MiB, no job) and after (no process, port or GPU memory left) |
| Release | `~/dev/topstart/rel`, `RELEASE`: topstart d52d02a (git archive, LF, plus the heuristic cache equal to box a's) |
| Data | `~/dev/topstart/full/ALL`: 781 shards; 169,241 train sides, 5,024,032 train decisions, 4,917 train players; held-out digest ho866789933298928784 (32,785 decisions, 226 players); 0 episodes dropped |
| Foul Play | 8f541f80 on box c, the same commit as box a and box d (checked 08:37); every search with more than one option at the root ran exactly 88,000 iterations a world (min = max = 88,000 in all three runs; a root with one option stops after one batch of 1,000) |
| Ports | 10791 (P0), 10792 (T0), 10793 (C0): the range D2 was given. Not the plan's 11691 to 11693. No other effect |
| T0 | `topstart.py train` with every default of the plan (warmup 1,000, hold, plateau anneal from 60k, at most 150k steps, abort 40k:0.30). 08:38 to 11:45 (11,207 s) |
| C0 | `--init ckpt/P0_ms.pt --warmup 100 --min-hold 0 --max-passes 3 --abort-at off --best-after 0`, 09:16 to 10:06 (2,986 s). It ran beside T0 on the same GPU. RAM was at most 59 of 124 GB used at every check, and GPU memory at most 7.8 GB |
| Measures | `topstart.py measure` on T0, C0, P0 (`ckpt/P0_ms.pt`) and SFIN (`ckpt/S/ppo_learner_update_016041.pt`). P0 was measured again under d52d02a and matched the earlier `P0_ms.json` exactly in every number |
| Gate games | `cloud/v2/topstart_eval.sh`: Foul Play 400 games (4 at a time), then simple and psbot 200 each. P0 used S's own `imitation_learner_update_000324.pt`, not the widened copy. Times: P0 08:38 to 09:15, C0 10:06 to 10:47, T0 11:45 to 12:26 |

## 2. T0's training curve

The full curve is in `T0/info.json` and `T0/history.json` (S3, section 6). The held-out set is the capped one
(32,785 decisions of 226 players). "Train" is the train top-1 of the 2,000 steps before each measurement (the
log's window: the model as it trained, on the batches it trained on). It is the only train-side reading the tool
records; section 3 adds a direct one.

| Step | Capped top-1 | Per player | CE | Value loss | SetHead loss | Train (window) | Train minus held-out | lr share |
|---|---|---|---|---|---|---|---|---|
| 2,000 | 0.2381 | 0.2351 | 1.7320 | 2.5089 | 0.7145 | 0.2331 | -0.5 | 1 |
| 4,000 | 0.2927 | 0.2879 | 1.7035 | 1.6635 | 0.7025 | 0.2405 | -5.2 | 1 |
| 6,000 | 0.3881 | 0.3794 | 1.5797 | 1.5253 | 0.6870 | 0.3545 | -3.4 | 1 |
| 8,000 | 0.4252 | 0.4253 | 1.4968 | 1.4636 | 0.6521 | 0.4042 | -2.1 | 1 |
| 10,000 | 0.4458 | 0.4450 | 1.4650 | 1.4751 | 0.6379 | 0.4296 | -1.6 | 1 |
| 20,000 | 0.4955 | 0.4947 | 1.3517 | 1.3235 | 0.5967 | 0.4974 | +0.2 | 1 |
| 30,000 | 0.5231 | 0.5241 | 1.2839 | 1.2375 | 0.5988 | 0.5214 | -0.2 | 1 |
| 40,000 (abort check) | 0.5490 | 0.5487 | 1.2309 | 1.2102 | 0.5625 | 0.5479 | -0.1 | 1 |
| 50,000 | 0.5632 | 0.5670 | 1.1928 | 1.2185 | 0.5535 | 0.5668 | +0.4 | 1 |
| 60,000 | 0.5738 | 0.5751 | 1.1645 | 1.2065 | 0.5479 | 0.5756 | +0.2 | 1 |
| 70,000 | 0.5752 | 0.5777 | 1.1581 | 1.2482 | 0.5462 | 0.5931 | +1.8 | 1 |
| 80,000 | 0.5826 | 0.5858 | 1.1424 | 1.1924 | 0.5390 | 0.5971 | +1.5 | 1 |
| 90,000 | 0.5841 | 0.5845 | 1.1383 | 1.2086 | 0.5398 | 0.6136 | +3.0 | 1 |
| 92,000 | 0.5882 | 0.5928 | 1.1340 | 1.2163 | 0.5395 | 0.6094 | +2.1 | 1 |
| 98,000 (plateau) | 0.5837 | 0.5798 | 1.1359 | 1.1975 | 0.5357 | 0.6169 | +3.3 | 1 |
| **108,000 (the start)** | **0.5865** | 0.5841 | **1.1299** | 1.1862 | 0.5256 | 0.6273 | **+4.1** | 0.655 |
| 116,000 | 0.5841 | 0.5828 | 1.1516 | 1.2023 | 0.5207 | 0.6639 | +8.0 | 0.181 |
| 123,000 (end) | 0.5828 | 0.5819 | 1.1584 | 1.1983 | 0.5177 | 0.6926 | +11.0 | 0 |

- **Abort rule:** 0.5490 at step 40,000 against the bar of 0.30. Not fired.
- **Plateau and anneal:** the best gained under 0.5 points over 20,000 steps at step 98,000, so the anneal ran from
  98,000 to 123,000. Held-out agreement was flat at 0.581 to 0.588 from step 78,000 to the end.
- **The start:** by the tool's rule (top-1 decides; within 0.2 points the lower CE does), step 108,000 at 0.5865 and
  CE 1.1299 beat step 92,000 at 0.5882 and CE 1.1340. T0.pt is that model, with total_steps 82,510,003 (rows
  seen up to it).
- **Size:** 123,000 steps, which is 18.2 passes over the train shards (6,773 steps a pass) and 94.0M rows seen. No
  step was non-finite. The log has no Traceback or error line. The data wait was 10.4 s on the first group and
  0.0 after.
- **Speed:** 14.1 to 14.2 steps a second alone, and 5.3 to 7.4 while C0 shared the GPU (09:16 to 10:06). The
  batches are a function of (seed, pass, group), so sharing changed the time, not the result.
- **Against the rehearsal:** the rehearsal measured 0.235 at 2k and 0.445 at 8k, after annealing from 6k. T0
  measured 0.238 at 2k and 0.425 at 8k at the full rate. These are the same early curve.

**C0** (arm C's recipe from P0, cosine from step 0): 0.5467 at 2k, 0.5693 at 8k, 0.5798 at 12k, 0.5849 at 16k and
0.5869 at the end (20,319 steps, 3 passes). Its train window at the end is 0.5914, 0.45 points above held-out.
Its SetHead starts fresh, because P0 has none.

## 3. Bar B1 (`measure`, counts only)

| Measure | P0 | SFIN | C0 | T0 |
|---|---|---|---|---|
| **capped held-out top-1** | 0.4576 | 0.5151 | 0.5869 | **0.5865** |
| capped, per-player mean | 0.4548 | 0.5082 | 0.5817 | 0.5841 |
| capped CE | 1.6358 | 2.7648 | 1.1388 | 1.1299 |
| capped value loss (Monte Carlo targets) | 8.92 | 10.01 | 1.16 | 1.19 |
| test_player, all 784,253 decisions | 0.4543 | 0.5226 | 0.5814 | 0.5819 |
| test_player, per-player mean (541 players) | 0.4537 | 0.5029 | 0.5780 | 0.5827 |
| test_game, 265,905 decisions | 0.4540 | 0.5136 | 0.5800 | 0.5832 |
| test_game minus test_player (points) | -0.03 | -0.89 | -0.14 | **+0.13** |
| capped by set: 40k / bulk | 0.4599 / 0.4564 | 0.5102 / 0.5177 | 0.5843 / 0.5884 | 0.5852 / 0.5873 |
| capped by period: 2025 / 2026-01..06 / 2026-07+ | 0.468 / 0.455 / 0.460 | 0.494 / 0.521 / 0.511 | 0.583 / 0.588 / 0.586 | 0.582 / 0.587 / 0.587 |
| test_player by set: 40k / bulk | 0.4619 / 0.4484 | 0.5384 / 0.5102 | 0.5817 / 0.5811 | 0.5796 / 0.5837 |
| test_player by period: 2025 / 2026-01..06 / 2026-07+ | 0.443 / 0.449 / 0.461 | 0.507 / 0.510 / 0.537 | 0.580 / 0.581 / 0.582 | 0.583 / 0.583 / 0.580 |
| test_game by set: 40k / bulk | 0.4550 / 0.4536 | 0.5112 / 0.5146 | 0.5782 / 0.5807 | 0.5792 / 0.5848 |
| SetHead on test_player: ability / item | no head | 0.804 / 0.796 | 0.748 / 0.738 | 0.791 / 0.768 |
| SetHead on test_player: moves / Tera | no head | 0.840 / 0.512 | 0.771 / 0.482 | 0.833 / 0.512 |
| SetHead loss, test_player | no head | 0.508 | 0.628 | 0.526 |
| train window minus held-out at the saved step (points) | | | +0.45 | +4.1 |
| **train top-1, the saved network on a fixed train sample** (30 train shards, 177,283 decisions) | 0.4552 | 0.5149 | 0.5957 | **0.6483** |
| **that minus capped held-out (points)** | -0.24 | -0.02 | +0.87 | **+6.17** |

The window is what `topstart.py train` logs: the train top-1 of the 2,000 steps before a measurement, each batch
read before its own update, by a network that changes over the window. The sample row is the direct reading:
every 26th of the 780 train shards (all their train sides), each saved network in eval mode, top-1 on the
labelled rows (section 7). P0 and SFIN, which never saw these sides, sit within 0.25 points of their held-out
figure, so the train and held-out populations agree and the gap of T0 is memorisation.

| B1 check | Bar | T0 | |
|---|---|---|---|
| capped top-1 | >= 0.55 | 0.5865 | pass |
| over P0 | >= P0 + 5 = 0.5076 | +12.9 points | pass |
| over SFIN | >= SFIN + 2 = 0.5351 | +7.1 points | pass |
| test_game against test_player | within 1.5 points | +0.13 | pass |
| train against held-out | at most 3 points above | **+6.2** (the saved network on train sides; the log's window said +4.1) | **fail** |
| abort | not below 0.30 at 40k | 0.5490 | not fired |

**B1: not passed in full.** Five checks pass and the overfit check fails, by about twice the bar. In the log's
window the gap was within 3 points up to step 92,000 (+2.1), reached +3.3 by 96,000 to 98,000 and +4.1 at the
saved step, then widened to +11 during the anneal. The cause is 18 passes over 5.0M decisions: the network
memorises the train sides, while held-out players stay at 0.58 to 0.59. The overfit costs nothing in agreement on
unseen players: every slice (set, period, test_player, test_game) is within 0.6 points of 0.585, the same flatness
as C0, which does not overfit. On SetHead, T0 is close to SFIN (moves 0.833 against 0.840, Tera equal at 0.512)
and well above C0.

## 4. Gate G1 (Foul Play at 88,000 iterations, all on box c)

| | Foul Play (400) | simple (200) | psbot (200) |
|---|---|---|---|
| P0 (S's imitation_learner_update_000324.pt) | 60-338 of 398 (2 errors): **15.1%**, 95% 11.9 to 18.9 | 159-41: 79.5%, 73.4 to 84.5 | 61-139: 30.5%, 24.5 to 37.2 |
| **T0** | 153-246 of 399 (1 error): **38.3%**, 95% 33.7 to 43.2 | 192-8: 96.0%, 92.3 to 98.0 | 130-70: 65.0%, 58.2 to 71.3 |
| C0 | 133-267 of 400: **33.2%**, 95% 28.8 to 38.0 | 176-24: 88.0%, 82.8 to 91.8 | 114-86: 57.0%, 50.1 to 63.7 |

| Difference (points, 95% interval, two-proportion p) | Foul Play | simple | psbot |
|---|---|---|---|
| T0 minus P0 | **+23.3** (+17.3 to +29.2), p 1e-13 | +16.5 (+10.3 to +22.7), p 5e-7 | +34.5 (+25.3 to +43.7), p 5e-12 |
| C0 minus P0 | +18.2 (+12.4 to +24.0), p 2e-9 | +8.5 (+1.3 to +15.7), p 0.02 | +26.5 (+17.1 to +35.9), p 9e-8 |
| T0 minus C0 | +5.1 (-1.5 to +11.7), p 0.13 | +8.0 (+2.7 to +13.3), p 0.003 | +8.0 (-1.5 to +17.5), p 0.10 |

Errors are games the Foul Play side forfeited after its own process failed. They are left out of the rates, as
evaluate.py always does. P0 had 2: one `ZeroDivisionError` in Foul Play's `select_move_from_mcts_results`, and one
`AttributeError: 'NoneType' object has no attribute 'max_hp'` in Foul Play's state update. T0 had 1, a panic in
poke-engine (`pyo3_runtime.PanicException: Invalid PokemonMoveIndex: 4`). C0 had none. Counting every error game
against the gate (T0's as a loss, P0's as wins) gives T0 38.2% and P0 15.5%: the verdict does not move. On the network side, every run had 0 fallbacks, 0 invalid orders and 0 exceptions.

**Verdict by the fixed rule:**

- FP(T0) >= FP(P0): 38.3% against 15.1%. Yes.
- T0's capped held-out top-1 >= 0.50: 0.5865. Yes.

**G1 passes, so launch T from T0.**

Against section 7's expectations: P0 was expected at 20 to 25% and measured 15.1%, below the range, which only
widens the margin. T0 was expected at 30 to 42% and measured 38.3%, inside the range. Arm C's 41.1% is an earlier
measurement, not part of this set. C0's 33.2% is the same recipe measured here, on the same box and build as P0
and T0.

## 5. What it means for launching T

- **The start is ready.** `s3://<your-bucket>/topstart/start/T0.pt` (34,091,275 bytes, sha256
  `48e4e7317467b3d5eb0e22009dee9d11880b4f31df06fd23b4584f5a7ca27915`) has `T0.pt.sha256` and `T0.pt.json` beside
  it. Step 3.4 fetches exactly these and checks them with `sha256sum -c`. The S3 copy was read back and its sha256
  matches.
- **The setup accepts it.** On box c, `topstart_setup.start_problem` returns none for T0 and for C0: both have
  set_head.*, all three observation settings, mode imitation, and no optimizer or scheduler. A dry run of
  `topstart_setup.py --rel ~/dev/topstart/rel --start ~/dev/topstart/start/T0.pt --fast` against S's run.json copy
  (`~/dev/topstart-integ/chk/S_run.json`, `--run` to a new path) exited 0 and wrote nothing.
- **Steps 3.1 to 4 run as written once scratchx is done, with one fix to 3.2.** D1 (disk) is still the owner's
  decision before 3.3, and without it T stops itself between 150M and 200M decisions. Step 3.2's check compares
  every tracked file of branch topstart on box a with box c's replica `~/dev/topstart/rel`, which is still
  d52d02a's archive. The branch now also has this note and `TOPSTART_G1.eb56a9b.md`, which the replica lacks, so
  `sha256sum` fails on box c and the check never prints its line. Before 3.2, put the branch's notes into the
  replica (only `bench/notes/` files; no code changes between d52d02a and the branch):

  ```
  cd C:/Users/you/ps-ppo-v2-topstart && git diff --quiet d52d02a topstart -- . ':!bench/notes' && git -c core.autocrlf=false archive --format=tar topstart bench/notes | $C 'tar -x -C ~/dev/topstart/rel && ls ~/dev/topstart/rel/bench/notes | grep TOPSTART'
  ```

  It stops before writing anything if a file outside `bench/notes/` differs. Then 3.2 runs as written.
- **T0 over C0 is a reading, not proof.** Foul Play is +5.1 (interval -1.5 to +11.7) and psbot is +8.0 (-1.5 to
  +17.5). Only simple (+8.0, p 0.003) clears the test. Agreement is equal (0.5865 against 0.5869). T0 is the
  start because the rule says so and because it is at least as strong. C0 is not needed.
- **What B1's failed overfit check means for PPO.** T0 agrees with its own train sides 6.2 points more than with
  unseen players (C0: 0.9). It is sharper on positions like its training sides than a 3-pass start would be. Two
  things to watch in the first PPO hour (section 5 of the plan), both readings and not rules:
  - The rehearsal's start (8,000 steps, no overfit) had entropy 1.43 at PPO update 1. T0 may start lower.
  - A run of KL early stops in the first updates would be the sign of an over-confident start.

  If B4 at 100M shows no lead, a shorter start for any rerun is worth trying: about 80,000 steps, where held-out
  is already 0.583 and the log's window gap is +1.5, at two-thirds of the time.
- **Time.** T0 took 3.1 h (the plan said 1.6 to 2.9 h): it ran to 123,000 steps, and it shared the GPU with C0
  for 50 minutes. Each 400-game evaluation took 37 to 41 minutes on box c (the plan said about 1 h).

## 6. Files

Box c (left in place, nothing deleted): `~/dev/topstart/start/{T0,C0}.pt` (with `.sha256`, `.json`),
`~/dev/topstart/T0/` (`resume_020212` to `resume_121791`, `best_*.pt`, `info.json`, `history.json`,
`log.jsonl`; 1.8 GB), `~/dev/topstart/C0/` (545 MB), `~/dev/topstart/measure/{P0,P0_ms,SFIN,C0,T0}.json`,
`~/dev/topstart/eval-{P0,T0,C0}/`, `~/dev/topstart/{T0,C0}.log`, `~/dev/topstart/G1_SHA256SUMS`.

S3 (account <aws-account-id>). Every object was read back and its sha256 matches `G1_SHA256SUMS` (31 of 31):

| Path | What |
|---|---|
| `s3://<your-bucket>/topstart/start/T0.pt` | the start, sha256 `48e4e7317467b3d5eb0e22009dee9d11880b4f31df06fd23b4584f5a7ca27915` |
| `s3://<your-bucket>/topstart/start/C0.pt` | the control, sha256 `6d897d0ac960f332f2207c3d6a13a821376998d1b4818c63c28c3890710b27fa` (34,074,571 bytes) |
| `s3://<your-bucket>/topstart/start/{T0,C0}.pt.{sha256,json}` | the sha256 files and the training info (counts only) |
| `s3://<your-bucket>/topstart/work-boxc/` | `measure/*.json` and `.log`, `{T0,C0}/{info,history}.json` and `log.jsonl`, `{T0,C0}.log`, `eval-{P0,T0,C0}/runs.log`, `eval/results.jsonl`, `eval/REPORT.md`, and `G1_SHA256SUMS` |

The uploads hold counts only. The only long hex strings in them are the Foul Play commit, the code's sha256 and
the checkpoints' sha256. No player hash, salt or name is in them. The shards, `data/` and the resume and best
files were not uploaded.

## 7. Independent check (2026-10-18, 12:31 to 12:55 UTC)

Every number above was recomputed on box c from the raw files with separate scripts (`~/dev/topstart-verify/`,
also in `s3://<your-bucket>/topstart/verify-g1/` with `SHA256SUMS`, read back equal; counts only, scanned
for every player and game hash of the shards: none). The scripts read the shards and the Foul Play logs
directly, not through `TopData`, `HumanImitation` or evaluate.py's tally. Box a was only read (its Foul Play
commit); box d was not touched; nothing was deleted; nothing of the check is left running.

| What | How | Result |
|---|---|---|
| G1 wins and losses | every `Winner:` line of the 15 Foul Play logs against each process's own login name | P0 60-338, 2 games never ended; T0 153-246, 1; C0 133-267, 0. Equal to results.jsonl |
| G1 intervals and differences | Wilson 95% and the normal difference interval, recomputed | equal to the tables of section 4, to the digit |
| 88,000 iterations | every `Iterations` line (162,696), every process's `Fixed iterations: 88000` line (15 of 15) | 160,172 at 88,000, 2,524 at 1,000, none other. Every 1,000 search is a root with one option (each world's choice visited 100%), except the one search cut short by P0's ZeroDivisionError crash, a game left out as an error |
| Same box and build | results.jsonl, `git status`, file times | all three on box c, Foul Play 8f541f80 (clean tree; box a's checkout has the same commit), poke-engine 0.0.48 built 2026-10-16 and unchanged since; no release or Foul Play file changed after 08:34; the policy played its best legal action in all three |
| Each network on its own observation | each checkpoint loaded by evaluate.py's player (`hybrid.PolicyNet`) | P0: own Tera type and matchup on, movesets off, input 9,635; T0 and C0: all three on, input 10,295. 98 of 98 tensors equal to the file for each |
| T0 is the chosen model | T0.pt against `T0/best_108000.pt`; the best and plateau rules on `history.json` | 106 of 106 tensors equal; the rule picks 108,000 (0.5865, CE 1.1299 over 92,000's 0.5882, CE 1.1340); the plateau rule fires first at 98,000; 0.5490 at 40,000 |
| No held-out row trained on | the split rebuilt from every shard's `ep_split` | 169,241 train sides in 780 shards, 5,024,032 labelled decisions, 4,917 players, no duplicate side: T0's counts exactly. 0 players both train and test_player; 0 games with a train side and a test_player, test_game or loose side (in any copy of the game). The trainer reads only the index's `train` episodes, and T0's rows seen (93,968,710 in 18.16 passes, about 5.17M a pass) fit the 5,176,205 train rows, not train plus held-out (6.3M or more) |
| The capped held-out set | rebuilt with its own code (the shuffled order, the 3% cap) | 1,107 episodes in 50 shards, 32,785 decisions, 226 players, none a train player, digest ho866789933298928784: the plan's |
| B1 agreement | each network run one episode per sequence (no packing), top-1 on the labelled rows | T0 0.5865, C0 0.58698 (written 0.5869 above), P0 (P0_ms) 0.4576, SFIN 0.5151, per player and per set and period equal to section 3 (P0_ms is P0 with zero columns added: 97 tensors equal, 1 widened) |
| B1 overfit | the saved networks on every 26th train shard (177,283 decisions) | T0 0.6483 (+6.2 over held-out), C0 +0.9, P0 -0.2, SFIN 0.0 |
| T0 in S3 | `aws s3 cp ... - \| sha256sum` on box c, account <aws-account-id> | `48e4e731...7a27915`, equal; `T0.pt.sha256` names `T0.pt`, so step 3.4's `sha256sum -c` works; all 31 objects equal to `G1_SHA256SUMS`, none holds a hash |
| The setup takes T0 | `topstart_setup.start_problem` | none: set_head.*, all three settings, mode imitation, no optimizer or scheduler |

**Corrected here:** the overfit figure (+6.2 measured, not +4.1; the verdict was already fail), T0's Foul Play
error (a poke-engine panic, not the `max_hp` error), and three rounded values (SFIN's test_game minus test_player
-0.89, T0's capped 2025 slice 0.582, C0's test_player 2026-07+ slice 0.582). **Found:** step 3.2's check needs the
notes in box c's replica first (section 5). **Not refuted:** the gate. T0 beats P0 by 23.3 points (interval +17.3
to +29.2), still 22.8 points with every error game counted against T0, and its capped top-1 is 0.5865 against
0.50. P0's 15.1% sits under the plan's guess of 20 to 25%, which no measurement backs (S's evaluator never played
its imitation checkpoint; its update 50 won 23.1% of 199 on box a); with P0 at 25% the gate would still pass.
