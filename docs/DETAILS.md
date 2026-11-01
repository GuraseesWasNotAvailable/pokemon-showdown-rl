# The full story: training a Pokemon Showdown agent from 1,500 to 2,466

This is the long version of the [README](../README.md). It covers what we built, every training run
in order with its numbers, how we measured, what the app does, how the bot did on the ladder, and
what did not work. The environment itself has its own page: [ENVIRONMENT.md](ENVIRONMENT.md).

All dates are 2026 and times are UTC. "FP" is [Foul Play](https://github.com/pmariglia/foul-play), the
open-source search bot we used as a yardstick. "FP@88k" is Foul Play searching exactly 88,000
iterations per sampled world. "Decisions" are trained decisions. Win rates against an opponent
come with 95% Wilson intervals: about +-7 points at 200 games and +-4.9 at 400.

Contents:
1. [In short](#1-in-short)
2. [Where it started](#2-where-it-started)
3. [The model](#3-the-model)
4. [The trainer](#4-the-trainer)
5. [Every training run, in order](#5-every-training-run-in-order)
6. [How we measured](#6-how-we-measured)
7. [The playing side: app, search and PSBot](#7-the-playing-side)
8. [The ladder](#8-the-ladder)
9. [What did not work](#9-what-did-not-work)
10. [What we would try next](#10-what-we-would-try-next)
11. [Infrastructure and cost](#11-infrastructure-and-cost)
12. [The weights](#12-the-weights)
13. [Showdown's bot rule and responsible use](#13-showdowns-bot-rule-and-responsible-use)

---

## 1. In short

- **What it is.** A 14.5M-parameter transformer that sees the whole battle state and its last 64
  turns. It first copies a hand-written bot, then trains by PPO self-play on local Showdown servers,
  2,304 battles at a time, 4,300 to 6,200 learned decisions a second on one cloud machine.
- **How good.** On the public ladder it played 1,407 rated games, reached a peak of **2,466**, and
  played at about 2,330 to 2,400. Alone, the best network (main run, update 24381) beats Foul Play
  at a fixed budget 60.2% of the time. With Foul Play's search beside it as a veto, it scores 59 to 66%.
- **What we learned.** Five separate training lines all ended within about 30 Elo of each other:
  - the main run;
  - a continuation run;
  - a run from scratch with every new technique;
  - a run that started from 140,000 games of strong human players;
  - a league run against frozen past networks.

  The one lever that reliably moved strength was letting the learning rate come down. Better
  starting points, extra training signals and more decisions at a low learning rate did not move the
  plateau.

## 2. Where it started

**The base project.** [Nebraskinator/ps-ppo](https://github.com/Nebraskinator/ps-ppo) (MIT) had:
- the transformer (PokeTransformer);
- Ray-distributed PPO self-play against local Showdown servers;
- a chat plugin that keeps battles running;
- the observation layout;
- imitation of poke-env's SimpleHeuristicsPlayer.

Its README reported over 85% against SimpleHeuristics and above 1,900 ladder Elo as a pure network.
It shipped no trained weights.

**PSBot.** Before any network, we wrote PSBot, a JavaScript bot that plays inside the Showdown web
page. It scores every legal move and switch with a full damage calculator and looks one turn ahead
(section 7.4). It played 426 ladder games, peaked at 1,929 and played at about 1,500 to 1,700.

**The goal.** Merge the two: let the network learn from PSBot first, then improve by self-play and
beat both. The work below lives on this repository's two code trees: `training/` (the newest
training line) and `app/` (the playing side).

**Bugs fixed in the inherited trainer.** These mattered as much as anything new:

| Bug | Effect | Fix |
|---|---|---|
| `lr_total_steps` written `1_00_000` (100k), smaller than the 500k hold | the learning rate dropped 27x in one optimizer step at step 501,001 (about 192M decisions) | a smooth decline. It gave the main run's only clean gain: FP 48.8% to 56.8% (+8.0 points, z 2.9) |
| value support built at +-1.6 while targets used +-1.5 | every reported value 6.7% too large | support matches the targets |
| `ff_expansion` 4.0 in the config never passed | every layer built at 2.0 | kept at 2.0 for compatibility, now explicit |
| `grad_accum_steps` never passed | always 1 | passed |
| KL early stop never fired in the default mode | no brake on large updates | fixed (UberMayinch) |
| recurrent history wiped on every weight sync | the model lost its memory of the game every second | fixed (UberMayinch) |
| poke-env never reads the own Tera type | the network chose to Terastallize without knowing into what | read from the request, see [ENVIRONMENT.md 4.3](ENVIRONMENT.md#43-the-tera-type-nobody-could-see) |
| a refused choice answered twice | refusal floods, workers swelling to 117.5 GB | one answer per request, see [ENVIRONMENT.md 4.1](ENVIRONMENT.md#41-refused-choices-a-feedback-loop) |
| resume picked the highest-numbered checkpoint, even one from an older phase | runs resumed from the wrong place | newest checkpoint of the right phase |

## 3. The model

| | |
|---|---|
| Input | 64 tokens a decision: field, 12 Pokemon (6 ours, 6 theirs), 48 moves, a register, actor, critic |
| Tokenizers | small MLPs per token type, with embeddings for species (64), items (32), abilities (64), moves (64), and shared number banks (0 to 100, stats up to 800, power up to 251) |
| Encoder | 3 pre-LayerNorm layers, d_model 512, 8 heads, feed-forward 2x, attention logits soft-capped at 50 |
| Memory | each layer attends to the field token of the last 64 decisions of the same battle; training packs 256 consecutive decisions with episode-aware block masks |
| Readout | actor and critic tokens cross-attend over the turn's tokens |
| Policy | 14 logits (4 moves, 6 switches, 4 Tera moves), masked to the legal ones |
| Value | 51 bins, two-hot, on [-1.5, 1.5] |
| Size | 14,490,708 parameters; a network-only file is about 32 MB (mostly bf16) |
| Precision | bf16 compute with fp32 master weights in the optimizer |

Optional heads, used only during training and dropped from the exported files:
- **SetHead** predicts each opposing Pokemon's ability, item, moves and Tera type from its token.
- **Opponent-action head** predicts whether the opponent stays in or switches, and to which slot.

## 4. The trainer

**Layout on Ray.**
- One learner on its own GPU.
- Up to three inference actors on the other GPUs.
- One rollout worker per Showdown server.
- In the league run, frozen-network actors that serve opponent networks.

**PPO settings.**

| Setting | Value |
|---|---|
| Decisions per update | 36,864 |
| Minibatch | 768 (3 packed sequences of 256) |
| Epochs | 2 |
| Clip | 0.1 |
| Entropy coefficient | 0.02 |
| Value coefficient | 0.5 |
| Grad-norm clip | 0.5 |
| Optimizer | AdamW |
| GAE | gamma 0.999, per-state lambda in [0.55, 0.95] |
| Learning-rate multipliers during PPO | backbone 0.5, policy 1.0, value 2.0 |
| KL early stop | from the second epoch, an update stops when its mean approx-KL passes 1.5 x 0.02 |

**Phases** (`pipeline.py` runs them in order and supervises them):

| Phase | What trains | Typical length |
|---|---|---|
| Imitation | the whole network copies PSBot playing both seats | 12M to 27.5M decisions; PSBot is learned to 60%+ agreement within about 7M |
| Warmup | only the value head, on the policy's own games | 0.5M to 2.2M |
| PPO | everything, self-play or league | 300M to 900M |

The supervisor:
- restarts servers and the trainer on stalls, dead servers, NaN losses or unreadable checkpoints;
- recycles the trainer every 6 hours;
- runs an evaluator on the newest checkpoint every 3 hours.

**The learning-rate bug that hid for a week ("steps vs decisions").** The schedule counted optimizer
steps and assumed 96 per update. With KL early stops firing, an update averaged about 49.6 steps. In
a 600M run, the hold would have ended at about 593M decisions and the decline at 1.19 billion, so the
learning rate would never have come down. We found it on 17 October and fixed it twice:
- **Restart.** One restart with recomputed step counts.
- **Schedule in decisions.** A schedule counted in decisions (`--lr-hold-decisions`,
  `--lr-total-decisions`), which neither KL stops nor restarts can move.

The from-scratch run went from 43.7% to 52.3% against Foul Play right after its decline began.

**Speed work.** All of these give bit-identical updates:

| Flag | Gain |
|---|---|
| Three GPU inference actors | about 5,070 to 6,205 decisions/s |
| `--fast-update` (minibatch loop with no GPU syncs) | +21 to 28% |
| `--fast-ingest` (observations to the GPU as they arrive, advantages at once, weight push on its own thread) | +8 to 12% |
| `--rollout-nice 5` | Ray had started every actor at nice 15 while 108 Showdown processes ran at nice 0, starving the single-threaded learner |
| `--learn-queue-episodes 32` | data reaches the learner about 1 update old instead of 4 to 6 |

## 5. Every training run, in order

| # | Run | Dates | Decisions | Best result |
|---|---|---|---|---|
| 1 | Main run | Oct 5 to 8 | 900M | **M24381**: 60.2% vs FP@88k alone; ladder peak 2,466 (at 437M) |
| 2 | Baseline (continuation) | Oct 13 to 14 | +164M from 674M | flat: FP 58.5% (start 60.0%) |
| 3 | Imitation test (arms A/B/C) | Oct 14 | 25M each | PSBot imitation then top-player fine-tune: 41.1% vs FP |
| 4 | Full run from arm C | Oct 14 to 15 | 600M | head start gone by 60M; 54.8% vs FP@88k |
| 5 | Eight technique arms ("waves") | Oct 15 to 16 | 5M to 60M each | no arm clearly ahead of its control |
| 6 | From scratch, every technique (S) | Oct 16 to 18 | 600M | level with M24381 (48.8% head to head) |
| 7 | Extension of S | Oct 18 | +200M | slightly worse (41.0% vs M24381) |
| 8 | Top-player start (T) | Oct 18 to 20 | 600M | level (49.2% vs M24381), second-best rating |
| 9 | League from M24381 | Oct 20 to 21 | 300M | level (52.5% vs M24381), 57.5% vs S's final |

### 5.1 Main run, October 5 to 8

- **Setup.**
  - Random weights, PSBot imitation for 27.5M decisions, warmup for 2.2M, then PPO.
  - One L40S GPU (g6e.16xlarge), 44 Showdown servers, 2,816 battles at once.
  - Target 600M, raised to 900M on October 7.
- **Releases during the run:**

| Release | From | Change | Against FP in its window |
|---|---|---|---|
| r1 | 2.6M | speedups (learner 1.5x, inference 1.6 to 2.4x, observation 3.1x), supervisor, evaluator | 40.5% to 54.0% by 158M |
| r2b | 348M | refusal-flood fix, smooth learning-rate decline | 56.8% of 400 (+8.0 points) |
| r3 | 425M | Foul Play as the opponent in 10% of battles | ran 46 minutes |
| r4b | 435M | + search teacher: Foul Play's veto as a KL target on about 1% of decisions | 60.7% pooled over 1,197; peak 65.5% at 649M |
| r5 | 674M | own Tera type and the ability fixes in the observation | 59.0% pooled over 1,599 |

- **Result.** It ended at 900,020,375 decisions (`learner_update_024381.pt`, "M24381"). The final
  evaluation was FP 60.5%, PSBot 81.2% and SimpleHeuristics 97.8%. From 400M to 900M it was flat:
  59.5% +-1.8 over 2,996 games against Foul Play.
- **On the ladder.**
  - Update 011850 (437M) went 136 to 84 over 220 games, from 1,969 to 2,358, and set the
    account's peak of **2,466** on October 8.
  - M24381 went 55 to 45 and peaked at 2,464.
- **Verdict.** PPO did nearly all its work early. The last 500M decisions (about 36 machine-hours)
  bought nothing measurable. M24381 became the reference for every later test.

### 5.2 Baseline run, October 13 to 14

- **Setup.** A continuation of the main run from 674M with fixes, as a 12-hour run: +164M decisions
  on an 8-GPU g7.48xlarge.
- **Result.** Flat. Against Foul Play it went from 60.0% to 58.5%, and PSBot from 81.8% to 80.2%.
- **Why.** The learning rate was on its floor (KL about 0.002 per update).
- **On the ladder.** Its final peaked at 2,449.

### 5.3 Imitation test, October 14

**Question.** Is starting from strong human players better than starting from PSBot?

**Data.** 35,919 public replays of players rated 2,100+ (2.3M decisions). Each player's view was
rebuilt exactly from the replay's input log. Usernames never leave memory; the shards carry salted
hashes only.

**Rule fixed before any game.** 800 games per arm against Foul Play.

| Arm | Recipe | vs FP | vs SimpleHeuristics |
|---|---|---|---|
| A | PSBot imitation from random weights | 22.1% | 86.5% |
| B | copy top players from random weights | 7.0% | 39.0% |
| C | A, then fine-tune on the same human games | **41.1%** | 94.5% |

C beat A by 18.9 points (p 4e-16). B reached only 24% agreement with held-out humans. It
memorised, it did not generalise, and its value targets also had a bug.

### 5.4 Full run from arm C, October 14 to 15

Arm C's network went through warmup and 600M decisions of PPO. Its head start was gone within about
60M decisions. At 451M it was at 53.0% against Foul Play, against the main run's 62.5% at a similar
point. The final was 54.8% against FP@88k, and it peaked at 2,462 on the ladder.

**Verdict.** The starting point matters less than the PPO budget.

### 5.5 The waves, October 15 to 16

Eight arms ran from one start, each testing one idea against a control:

| Arm | Idea |
|---|---|
| a0 | control: learning-rate restart only |
| read | a read bonus: +0.30 for hitting a switch-in super effectively when the attack would not have been on the Pokemon that left. Zero-sum, added to the advantage only, faded to 0 between 20M and 45M |
| head | the opponent-action head |
| readhead | read + head |
| setup15 | "setupnet" opponents in 15% of battles: the network with a set-up rule substituted half the time, never trained on |
| setupw2 | setup15 + double policy-loss weight where the foe has a stat boost |
| musethead | the matchup block (by widening) + SetHead |
| fp10 | Foul Play opponents in 10% of battles |

At about 26M decisions (FP@88k, 200 games):

| Arm | vs FP |
|---|---|
| a0 | 50.0% |
| readhead | 61.0% |
| setupw2 | 63.8% |
| musethead | 55.0% |

By 39M they had converged: a0 57.5%, readhead 52.3%, setupw2 54.5% and musethead 52.5%. No arm was
clearly ahead, so every technique went into one from-scratch run instead.

The waves were interrupted once when their cloud machine was deleted with everything on its disk.
From then on, every run was backed up to S3 every 10 minutes.

### 5.6 From scratch with every technique (S), October 16 to 18

- **Setup.**
  - Random weights, PSBot imitation for 12M with the matchup block, warmup, then 600M of PPO.
  - SetHead, the opponent-action head, the read bonus, set-up seats with a 2x weight on boosted
    positions.
  - The moveset block was added by widening at 61M.
- **Two mid-run events.**
  - A PPO-time top-player imitation term flattened the policy (entropy 0.47 to 0.59 in an hour) and
    was switched off by its pre-set rule.
  - The learning-rate restart at 357M finally let the rate come down.
- **Against Foul Play** (FP@88k, 200 games):

| Decisions | vs FP |
|---|---|
| 70M | 30.2% |
| 138M | 35.7% |
| 206M | 42.7% |
| 281M | 43.7% |
| **366M, after the decline began** | **52.3%** |
| 425M | 48.7% |
| 540M | 52.5% |
| 600M | 51.0% |

- **Final (SFIN).**
  - Head to head vs M24381: 48.8% of 400.
  - Alone vs FP@88k: 56.0% (M24381: 60.2%).
  - With the app's search: level with M24381.
  - Its rating gap to M24381 shrank from 113 Elo to 67 to 19 during the learning-rate decline.

### 5.7 Extension of S to 800M, October 18

- **Setup.** 200M more decisions at the floor learning rate.
- **Result.** Head to head vs M24381 41.0% (p 0.0004), and vs its own 600M final 48.0%.
- **Verdict.** More decisions at a low rate did not help.

### 5.8 Top-player start (T), October 18 to 20

- **Data.** About 140,000 public games of players rated 2,100+: 5.0M training decisions from 4,917
  players, and a capped held-out set of 32,785 decisions from 226 others.
- **Offline start T0.** Imitation from random weights, with a value target on the shaped return and
  SetHead.
  - Held-out top-1 agreement: 58.65%. PSBot's imitation start reached 45.8%, S's final 51.5%.
  - Its gate, set before training: beat the PSBot start against Foul Play and reach 50% agreement.
    It passed: 38.3% against Foul Play versus 15.1%.
- **PPO.** 600M decisions with every technique, the learning rate held to 357M then brought down,
  in decisions.
- **Course.** T led S by 10 to 15 points early. S caught up during its decline, the same pattern as
  arm C.
- **Final test (2,000 games).**
  - T600 alone vs FP@88k: 52.2%.
  - Head to head vs M24381: 49.2%.
  - Vs SFIN: 53.5%.
  - With the search: level.

### 5.9 League run from M24381, October 20 to 21

- **Goal.** Keep training the strongest network, but against something other than itself.
- **The start.** M24381 was trained on an older version of the observation. It was ported exactly
  onto the new code line, then widened with the matchup and moveset blocks. It played bit for bit as
  M24381 on 12,048 recorded decisions. A negative control on the wrong observation matched 0 of
  12,048, and its greedy move changed on 9.3% of them.
- **Opponents.**

| Opponent | Share of battles |
|---|---|
| Self-play | 60% |
| Setupnet | 15% |
| Frozen networks | 25%: M24381 35%, SFIN 25%, four past saves of the run 40%, re-picked every 30 minutes |

  Frozen networks are served by their own actors, at a 3% speed cost.
- **Learning rate.** 6e-5 held to 100M, then down to 1/27 at 300M. Fresher data, with the learner
  queue at 32 episodes.
- **Bars set before training.** 400 games head to head against frozen M24381 at 100M, 200M and
  300M. 55% or more means better; 45% or less at 100M or 200M means stop.

| Checkpoint | vs M24381 | vs FP@88k | Other |
|---|---|---|---|
| 103M | 50.2% | 57.5% | |
| 207M | 54.8% (49.9 to 59.6) | **63.2%** | |
| 300M | **52.5% (47.6 to 57.3)** | 57.4% | **vs SFIN 57.5% (52.6 to 62.3)** |

- **During training.** Its win share against the frozen networks climbed from 53.4% to 55.4% over
  hundreds of thousands of games.
- **Verdict.** Level with M24381 by the pre-set bar; clearly better than the from-scratch line. We
  ported both league checkpoints into the app anyway, so the ladder can judge them.

## 6. How we measured

- **Foul Play at a fixed budget.** Time-based Foul Play drifted with machine load (about 23% fewer
  iterations once other jobs shared the cores). From October 15 every sampled world is searched
  exactly 88,000 iterations: the median at 200 ms on 122,766 worlds. Strength follows iterations:
  Foul Play at 22,867 iterations won only 30% against itself at 118,591.
- **The Foul Play bridge.** It asks Foul Play's own code for its move in our battles. It showed 0
  differences over 881 decisions and won 52.4% of 254 games against the real Foul Play process.
- **Pre-registered bars.** Every test's pass and stop rule was written in a plan file before the
  first game or training step. After each report, an independent recount from the raw per-game files
  reproduced every number.
- **Head to head.** Every game is a challenge on a local server, with sides balanced. Our side is
  the exact player the app runs.
  - One caveat: the side played through the batched player scored higher in every test. Later fits
    include a side term.
- **Bradley-Terry ratings.** Maximum likelihood on the Elo scale over all games, with FP@88k
  pinned at 0:

| Fit | Elo vs FP@88k |
|---|---|
| 4,000 games, Oct 17 | M24381 +75, B22640 (baseline final) +44, S at 293M -34, S at 330M -41 |
| 8,000 games, Oct 20 | **M24381 +67**, T600 +46, B22640 +40, SFIN +38, S800 +16 |

## 7. The playing side

### 7.1 The window app

"PSBot Lookahead" is one C# file (`app/app/PSBotLookahead.cs`), built with the compiler that ships
with Windows. It runs `ladder.py` as a child process and shows what it does.

| Feature | Detail |
|---|---|
| Who plays | network only; network + search; best network + guarded search; language models alone (Claude, GPT) |
| Thinking time | slider from 0.5 to 7 s a move. Deeper searches each guessed opponent team longer; Wider guesses more opponent teams (`--test-think` prints the whole table) |
| Network list | every run family, each played on the observation it was trained on: main line, baseline, train-v2 runs, league. `LATEST` and `BEST ON LADDER` marks. Incomplete or still-writing files are refused |
| Live views | the Elo of the run, game by game; a "neurons firing" view of the network's feed-forward layers and attention for the move it just chose, in 2D or 3D |
| Safety | the password is typed, sent once over stdin and never stored or logged; one bot per account; an 8 GB free-memory guard; "stop after these games" and "stop now"; every child process dies with the window |
| Test mode | `--test-server PORT --test-user NAME` plays on a local server with no password |

### 7.2 ladder.py

The runner the app launches.
- It plays the network's best legal action, or the search's choice.
- It sends `/timer on` every game.
- It reconnects within Showdown's 60-second forfeit window, rejoins games in progress and answers
  the pending request without deciding twice.
- It spaces orders 300 ms apart and resends any dropped by the typing limit.
- It records every battle and checks the search's health.
- `--arm-settings` gives each game its own setting, so several settings can be compared at once.

### 7.3 The search

- **The engine.** Foul Play (pinned, installed separately, run inside WSL on Windows) is used as an
  adviser through a bridge.
- **Sampling.** For each decision it samples full opponent teams consistent with what has been
  revealed, about 2.2 per decision, and runs its tree search on each.
- **Veto.** An action is sound if its search share is at least 0.25 of the best share. Among sound
  actions, the network's favourite is played.
- **Guarded mode** (`--trust 0.03`). The search may only choose among moves the network gives at
  least 3%. A study of 518 ladder losses found that 45% of overrules picked a move the network gave
  under 2%. Against Foul Play the guard tied (60.0% vs 60.0%, 600 games).
- **What the search adds.** +23.6 points against Foul Play over the network alone (68.2% vs 44.6%).
  Head to head against the network alone it adds nothing (50.0% of 306).
- **More time does not help.** 5 s a move tied 1 s, both on a box (67.0% vs 67.0%) and on the
  ladder (60.0% vs 59.9%, 474 games). The number of guessed teams stays at 2.2 at any budget. That
  motivated the "Wider" style.
- **Settings test** (1,200 games, guarded, vs FP@88k):

| Setting | vs FP@88k |
|---|---|
| 2 s Wider | 64.0% |
| 4 s Deeper | 64.0% |
| 6 s Wider | 58.5% |
| 4 s Wider | 57.5% |
| 6 s Deeper | 57.0% |
| 2 s Deeper | 54.0% |

  All six are compatible with being equal (p 0.24). The app defaults to 2 s Wider, guarded.
- **Opponent prediction** is shown but not acted on. Acting on it lost 20.5 points.

### 7.4 PSBot

A JavaScript userscript that plays inside the Showdown web client.
- **How it scores.** Every option is valued in units of expected HP lost by each side, plus a KO
  bonus.
- **What it models:**
  - a full damage calculator (stats, boosts, Tera, weather, terrain, screens, items, about 80
    abilities);
  - opposing sets taken from the random-battle data;
  - turn order;
  - the one-on-one played forward turn by turn;
  - switches scored after the incoming hit.
- **Accuracy.** Predicted damage is within 1% of actual on average over 2,300 hits.
- **Strength.** It won 90.6% of 3,000 games against SimpleHeuristics, and runs headless at about
  170 decisions a second.
- **How it was used:**
  - as the imitation teacher;
  - as an evaluation opponent (networks beat it 72% to 86%);
  - briefly, as a prior.

## 8. The ladder

**Record.** 1,407 rated games by the network bot (October 5 to 20): 797 wins, 607 losses and 3
with no result. The rating went from 1,510 to a peak of **2,466** on October 8, with network 011850
and search.

| Mode | Wins and losses | Win % |
|---|---|---|
| Network + search (veto) | 330-214 | 60.7 |
| Wider search | 165-136 | 54.8 |
| Network alone | 93-66 | 58.5 |
| Read-ahead lines | 73-44 | 62.4 |
| Guarded search | 45-33 | 57.7 |

Win rates are not comparable across rows: they come from different networks, dates and rating bands.

**Level, not peak.** Above 2,350 the bot fell 11.7 wins short of expectation over 251 games
(performance about 2,354). Its true level is about 2,330 to 2,400. Swings of 100 points are normal
for a constant-strength player over 42-game sessions.

**How it loses** (518 games studied):
- In 68% of losses, one opposing Pokemon knocked out three or more of ours. In 34.5% of losses
  that Pokemon had used a set-up move.
- Losses turn late, at a median of turn 18 of 27.

**Language models** played too. None beat the plain search:
- GPT-6 Luna deciding over the search: 27-23.
- GPT-6 Sol alone: 16-37.
- Claude Fable 5.1 alone: 4-12.

## 9. What did not work

| Idea | Result |
|---|---|
| More PPO at the floor learning rate | main run 400M to 900M flat; baseline +164M flat; S extension +200M slightly worse |
| A held, not declining, learning rate | no collapse and no gain |
| A better start (PSBot + humans, or top players) | big early lead, gone by 60M to 600M decisions of PPO |
| Copying top players from random weights | 7% against Foul Play |
| Fine-tuning a strong network on top players | agreement up from 54.8% to 59.1%, strength +0.0 over 2,100 games |
| PPO-time top-player imitation term | flattened the policy; switched off |
| Foul Play as a training opponent | no benchmark change, -7.5% throughput |
| Search teacher on about 1% of decisions | no benchmark change |
| Richer observation (own Tera, ability fixes) | no gain live, though it fixed real errors |
| Read bonus, opponent-action head, set-up seats, matchup + SetHead (waves) | no arm clearly ahead at 39M |
| A separate win head or value net | equal to the existing value head |
| SetHead set guesses inside the search | the chance of the opponent's true set rose from 45.7% to 46.6% (bar +4.6): stopped |
| More search time, language-model deciders, set-up counter rules | none beat the plain search |

## 10. What we would try next

1. **A bigger network.** The size is the one thing no run changed. Widening (zero columns) lets it
   start exactly as M24381.
2. **Distil the search into the network** on most decisions, not 1%.
3. **Defend against set-up sweeps**, which decide most ladder losses. Train from those positions.
4. **Calibrate the value head.** S lost 9 of 18 ladder games after its own estimate passed 70%.
5. **Train against stronger and different opponents**, not only copies of its own family.

## 11. Infrastructure and cost

| Machine | GPUs | Price | Used for |
|---|---|---|---|
| EC2 g6e.16xlarge, on demand | 1 x L40S | $7.58/h | main run |
| EC2 g7.48xlarge, on demand | 8 x RTX PRO 4500 | $28.51/h | baseline, imitation test, arm C run |
| EC2 g6e.12xlarge, spot | 4 x L40S | about $9/h | waves, S, extension, T, league |
| EC2 g6e.4xlarge, spot | 1 x L40S | about $2.3/h | imitation starts, rehearsals, tests |
| EC2 g6.4xlarge, on demand | 1 x L4 | about $0.9/h | tests, replay collection |
| Laptop, RTX 4060 | 1 | 0 | early tests (about 390 decisions/s) |

**How the boxes ran:**
- Spot boxes stop on interruption and resume from the newest checkpoint.
- Every run directory is synced to a versioned S3 bucket every 10 minutes, never with deletes.
- A guard stops a box after 120 idle minutes.
- Releases are deployed as `git archive` copies checked file by file.

**Cost.**
- Recorded or computed from logged hours: the main run about $540, the baseline about $322, and
  the imitation test plus arm C's run about $1,040.
- The later runs on the spot box were roughly $80 to $350 each.

## 12. The weights

All in [release v1.0](https://github.com/GuraseesWasNotAvailable/pokemon-showdown-rl/releases/tag/v1.0). Each file is the network only: no optimizer, no
target network, no training-only heads.

| File | Run and decisions | Observation | How good |
|---|---|---|---|
| `learner_update_024381.pt` | main run, 900M | main line (own Tera, ability fixes) | best alone: +67 to +75 Elo over FP@88k |
| `learner_update_011850.pt` | main run, 437M | base layout | played the 2,466 peak |
| `v2-league_update_008124_300M.pt` | league, 300M from 24381 | main line + matchup + movesets | level with 24381; 57.5% vs SFIN |
| `v2-league_update_005600_207M.pt` | league, 207M | same | 54.8% vs 24381, 63.2% vs FP@88k |
| `v2-topstart_update_016127_595M.pt` | top-player run, 600M | own Tera + matchup + movesets | +46 Elo over FP@88k |
| `v2-scratch_update_016041_591M.pt` | from scratch, 600M | own Tera + matchup + movesets | +38 to +47 Elo over FP@88k |

The app (`app/`) plays all six, building each one's observation from the settings stored in the
file. Put them in `app/weights/`. The training code (`training/`) can continue the league and v2
files.

## 13. Showdown's bot rule and responsible use

On 20 October 2026 Pokemon Showdown changed its matchmaking. An account registered as a bot is now
paired only with players who opted in with `/allowbots`, and everyone else is opted out by default
(`server/ladders.ts`, commits `de1cf598c7` and `7777a79491`). Our last ladder game was on October 20,
before the change. Every game in section 8 was played before this rule existed.

If you use this code:
- train and test on local servers (`--test-server` in the app, `--server local:PORT` in
  `ladder.py`);
- on the public ladder, use an account registered as a bot, and do not use an unregistered account
  to get around the opt-in;
- keep one bot per account, the battle timer on, and no chat; the app enforces all three;
- stop with "stop after these games" rather than abandoning games.

---

Built by Gurasees Singh Dhanoa and UberMayinch, equal contributors, on top of Nebraskinator/ps-ppo.
See the [README](../README.md) for credits and licences.
