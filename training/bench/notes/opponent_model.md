# Predicting the opponent, and playing on it: what is there, what was measured, what is not done

2026-10-06, branch `search`. New: `opponent_model.py`, `opponent_model.json` (the fitted numbers),
`tests/test_opponent_model.py`, `bench/opp_model_replay.py`, `bench/predict_run_report.py`.
Changed: `search_player.py` (mode `predict`), `ladder.py` (`--player predict`, four flags, the
outcome notes), one line of `tests/test_search_player.py` (the list of modes).
Port 8660 only (a local server). No public server, no account. Nothing in `ps-ppo` or
`ps-ppo-play` was touched.

## The result in five lines

1. **The opponent's next action is predicted right 47% of the time on people** (the likeliest of
   about 10 options; test split, 60 games, 1,320 turns, search at 500 ms = 229,000 iterations per
   decision on average, median 123,000). The search alone: 45%. The best baseline: 32%. Top-3: 81%.
2. **The gain over the search alone is in the probabilities, not in the top pick**: log loss 1.48
   against 1.81 for the search's own shares and 1.87 for the best baseline.
3. **It is roughly calibrated up to 60% and overconfident above**: where it says 60 to 70% it is
   right 58%, where it says 70 to 80% it is right 62%, where it says 80 to 90% it is right 83%.
4. **"It will switch" is the weak read.** When the model says so with 60% or more, the opponent
   switched on 57% of 65 test turns, and the Pokemon named came in on 40%. Above 70% it was right
   on 15 of 29. A double switch is therefore a bet at a little better than even, and the code
   treats it so.
5. **The `predict` mode runs clean and changes little**: 132 games against the network alone, all
   finished, no invalid order, no exception. It left the mix choice on 89 of 3,955 predictions
   (2.3%), with the read right on 50 of the 82 that could be scored (61%). Three double
   switches in 132 games. **Nothing shows that it wins more than `mix`**, and the games in which
   it changed a move were won less often than the others (20 of 51 against 59 of 81): see
   "Real games".

## 1. The model (`opponent_model.py`)

`OpponentModel.load().predict(ctx, advice)` gives `Prediction.dist`: a probability for every
`move <id>`, `switch <species id>` and `switch ?` (a Pokemon not shown yet), plus `p_switch`,
`p_tera` and `tera[move]`. `ctx` is the situation as `bench/opp_dataset.py` writes it. At play
time `OpponentTracker` builds it from the battle's lines WITH THE SAME CODE that made the
training rows (`GameRows`), so the model sees at play time what it was fitted on.

Two sources: the search's shares for the opponent's options (`advice.opponent`), and what
people did in the recorded games. Four small parts, all fitted on the train split only:

| Part | Form | Fitted (500 ms fit, the one shipped) |
|---|---|---|
| P(switch) | logistic regression on the logit of the search's switch share and the situation | search logit 0.50, type disadvantage +0.67, advantage -0.45, just in after a faint -1.94, just in otherwise -0.42, on the field one turn +0.21, we switched last turn +0.59, a stat drop +1.08, a boost -0.91, statused +0.24, constant -1.10 |
| P(target given a switch) | the search's target shares to a power, mixed with an even spread over shown Pokemon and "?" | weight of the search 0.7, power 0.5, "?" counts as 1 Pokemon |
| P(move given it stays) | mixture, weights by EM, temperature by a grid | search 0.69 (T = 1.0), its last move 0.06, its strongest shown move 0.09, even over shown moves 0.03, even over every move it may have 0.14 |
| P(Tera given a move) | a table by the search's Tera share of its move shares, six bands drawn towards the overall 4.6% | 1.3%, 0.9%, 3.2%, 5.2%, 10.2%, 30.0% (bands under 0.05, 0.10, 0.20, 0.35, 0.50, above) |

Without a search answer the same parts run on the situation alone (the "prior" numbers in the
file, fitted on all 298 train games).

Read the P(switch) numbers as corrections of the search: people switch more than the search
expects after a stat drop, at a type disadvantage and after we switched, and much less right
after a replacement came in.

### How it was fitted

`bench/opp_model_replay.py` replays the recorded games through the helper (inside WSL, the
scout's stream builder and helper class unchanged) and keeps the whole opponent side of every
answer. Runs (all 6 helpers, nothing else of mine running):

| Replay | Games | Decisions answered | Iterations per decision: mean / median / lowest tenth | Asked to answered |
|---|---|---|---|---|
| all aligned games, 150 ms | 368 (259 train, 109 test) | 10,538 of 10,564 | 64,467 / 39,000 / 15,000 | 386 ms mean |
| train, 500 ms | 90 | 2,699 of 2,699 | 367,192 / 214,000 / 122,000 | 1,120 ms mean |
| test, 500 ms (two runs joined) | 60 | 1,718 of 1,718 | 228,597 / 123,000 on the scored turns | 1,166 to 1,458 ms |

(The 26 unanswered decisions of the first run are one game the helper could not start.)

**The search time matters for calibration, so the model is fitted at the time it is played at.**
The first fit used the 150 ms replay (258 train games). Scored with 500 ms searches it was
overconfident: where it said 70 to 80% it was right 59% (196 turns), and its fitted temperature
of 0.7 sharpens shares that a longer search has already sharpened. The fit that is shipped
(`opponent_model.json`) uses the 90 train games replayed at 500 ms: temperature 1.0, mean gap
between said and happened 0.069 instead of 0.089. Both sets of numbers are kept:
`runs/opp_pred/model/params_fit_all_ms150.json` and `params_fit_train90_ms500.json` (git-ignored).

### Test split, search at 500 ms (60 games, 1,320 scored turns, 178 switches)

`python opponent_model.py eval --replay runs/opp_pred/model/replay_train90_ms500.jsonl --eval-replay runs/opp_pred/model/replay_test60_ms500.jsonl`
(the scorer is `bench/opp_predict_eval.score`, the baselines its own, fitted on train).

| Predictor | top-1 | top-3 | log loss | switch or stay | switch AUC | switch was top-1 when it switched | right target given a switch (already shown) | top-1 on move turns (move already shown) |
|---|---|---|---|---|---|---|---|---|
| **model, fitted at 500 ms (shipped)** | **47.0%** | 80.7% | **1.481** | **87.4%** | 0.864 | 37.1% | 73.6% (44.4%) | **48.5%** (57.7%) |
| model, fitted at 150 ms | 46.8% | 81.3% | 1.492 | 87.1% | 0.870 | 44.4% | 73.6% (44.4%) | 47.2% (54.6%) |
| the search's own shares | 44.6% | 79.9% | 1.809 | 86.2% | 0.840 | 49.4% | 73.0% (44.4%) | 43.8% (47.9%) |
| model without the search | 33.1% | 62.8% | 1.820 | 86.7% | 0.792 | 20.5% | 62.8% (43.6%) | 35.0% (59.2%) |
| best baseline, column by column | 31.7% | 60.5% | 1.874 | 86.6% | 0.701 | 36.9% | 62.8% (43.6%) | 36.2% (60.0%) |

What it is NOT better at: naming the Pokemon that comes in when that Pokemon has been shown
(44.4%, the even guess gets 43.6%), and the turns where the opponent repeats a shown move
(57.7% against 60.0% for "it repeats its last move"). "Switch or stay" is 87.4% against 86.6%
for "it never switches": one point.

### The same at 150 ms (109 test games, 2,401 scored turns, 62,984 iterations per decision, median 38,000)

| Predictor | top-1 | top-3 | log loss | switch or stay | switch AUC |
|---|---|---|---|---|---|
| model fitted at 150 ms | 48.6% | 82.6% | 1.435 | 87.3% | 0.868 |
| the search's own shares | 45.3% | 82.5% | 1.680 | 86.7% | 0.832 |
| model without the search | 34.3% | 64.1% | 1.796 | 86.2% | 0.806 |
| best baseline, column by column | 33.0% | 62.0% | 1.840 | 86.0% | 0.716 |

So a search with a quarter of the iterations predicts people no worse (45.3% at 150 ms, 44.6% at
500 ms, different games in part): more search time does not buy a better read of people.

### When it says 80%, is it right 80% of the time?

Test split, 500 ms, the shipped fit. "Said" is the probability of its likeliest action.

| Said between | Turns | Said on average | Was right |
|---|---|---|---|
| under 20% | 15 | 19.0% | 13.3% |
| 20 and 30% | 192 | 25.9% | 32.3% |
| 30 and 40% | 320 | 35.0% | 44.7% |
| 40 and 50% | 288 | 44.8% | 37.9% |
| 50 and 60% | 201 | 54.6% | 56.2% |
| 60 and 70% | 165 | 64.8% | 57.6% |
| 70 and 80% | 93 | 74.4% | 62.4% |
| 80 and 90% | 46 | 84.2% | 82.6% |

Mean gap 6.9 points. It never said 90% or more. P(switch) by itself:

| Said between | Turns | Said | It switched |
|---|---|---|---|
| under 5% | 566 | 1.7% | 0.9% |
| 5 and 10% | 207 | 7.2% | 4.3% |
| 10 and 20% | 200 | 14.0% | 17.0% |
| 20 and 30% | 101 | 24.7% | 22.8% |
| 30 and 40% | 92 | 34.9% | 33.7% |
| 40 and 50% | 56 | 44.9% | 37.5% |
| 50 and 60% | 33 | 54.2% | 54.5% |
| 60 and 70% | 36 | 64.6% | 61.1% |
| 70 and 80% | 23 | 74.6% | 52.2% |
| 80% and more | 6 | 83.9% | 50.0% |

Log loss of P(switch) 0.289 against 0.396 for the base rate. Good up to 70%, and above that the
29 turns say 52%: the model's highest switch probabilities are not to be believed at their face.

The reads the `predict` mode acts on (a read is "it switches" when P(switch) is over the bar,
else its single likeliest action when that is over the bar):

| Bar | Reads (of 1,320 turns) | Right | "Its move is X" | "It switches" | and the Pokemon named came in |
|---|---|---|---|---|---|
| 50% | 571 | 59.7% | 60.5% of 473 | 56.1% of 98 | 40.8% |
| 60% (the default) | 347 | 62.8% | 64.2% of 282 | 56.9% of 65 | 40.0% |
| 70% | 159 | 67.3% | 70.8% of 130 | 51.7% of 29 | 34.5% |
| 80% | 49 | 79.6% | 83.7% of 43 | 3 of 6 | 2 of 6 |

Tera (723 move turns on which it still could, 33 with a Tera): log loss 0.155 against 0.186 for
the base rate, and the bands hold (said 30.0%, happened 28.3% on 46 turns; said 1.1%, happened
1.2% on 330).

## 2. Acting on it: mode `predict`

`opponent_model.choose()` is a pure function. The mix choice (network and search, rule veto unless
told otherwise) stands unless ALL of these hold. Every number is a setting
(`PredictSettings`, and `--predict-confidence`, `--predict-gain`, `--predict-risk`, `--payoff-ms`):

| Test | Default | Kept reason when it fails |
|---|---|---|
| a read: P(switch), or else its likeliest action, at least | 0.60 | `unsure` |
| the payoff table has something for the mix choice | | `no table` |
| another action has cells behind at least this share of the predicted probability, for it and for the mix choice | 0.70 | `coverage` |
| that action is worth more against the WHOLE predicted distribution by at least (engine scale, `sigmoid(0.0125 * difference) - 0.5`) | 0.08, about 26 points of evaluation | `no gain` |
| the cell against the opponent's second most likely action exists | | `blind` |
| against that second action, and every other one it takes with 15% or more, the new action is worse than the mix choice by at most | 0.08 | `risk` |
| the same read has not already been wrong against this Pokemon in this battle (the player's memory, not `choose()`'s; added after the main run) | on | `bitten` |

A candidate must be legal and have at least 1% of the search's visits. Terastallization is not
this rule's business: a move is compared in the form the mix rule would give it.

A double switch is the case "read = it switches, the best action against the Pokemon coming in
is a switch of ours". Nothing is coded for it apart: it falls out of the table.

### The payoff table: what is used, exactly

The bridge's `payoff="full"`, `payoff_ms=400`: for every sampled world, pairs (our option, the
opponent's option) in the order of the search's visits (ours times theirs), each pair's outcomes
from `generate_instructions`, each outcome's value from a 1 ply search of the position after it
(2 plies in all), until the budget is spent. The mean over the worlds that have the cell is used.
The unit is the engine's evaluation.

The engine's own matrix (`payoff="engine"`) is not used, and this is measured, not assumed: its
pruning stops a row of ours at the first cell that shows the row is not the best, and the
opponent's switches are the last columns. 24 test games, 494 tables each, 500 ms search, 400 ms
table, 6 helpers:

| | full (used) | engine (pruned) |
|---|---|---|
| time for the table: mean / p95 / max | 378 / 419 / 451 ms | 403 / 411 / 442 ms |
| request answered after: mean / p95 / max | 1,553 / 1,690 / 1,886 ms | 1,599 / 1,716 / 2,004 ms |
| cells known | 42.5% | 35.1% |
| cells known in the opponent's switch columns | 35.6% | **20.1%** |
| cells known in its move columns | 45.7% | 44.1% |
| share of (our share x its share) on known cells: mean / median / lowest tenth | 85.7% / 92.7% / 58.0% | 72.7% / 75.5% / 46.2% |
| rows of ours complete | 24.4% | 18.8% |

So the table is thin: on average two or three of our actions have a full row, and every action
has the cell against the opponent's likeliest action. An answer to a predicted switch that the
search itself hardly visited has no row, and is then not a candidate (`coverage`). 60 of 1,128
worlds came back with a negative depth, 1,058 with 2 and 10 with 3. Probably the depth counter
running over on positions where the battle has ended (not checked). Their values were used as
they came.

### What is counted

Per decision, in the record (`battles/<tag>.jsonl`, and `decisions_of(tag)`): `predict` =
`top` (three likeliest with probabilities), `p_switch`, `p_tera`, `read`, `confidence`, `second`,
`candidate`, `gain`, `risk`, `coverage`, `why`, `changed`, `mix` (the action the mix rule chose),
`payoff_ms`, `payoff_known`, and when the move was changed `position` (both Pokemon with HP,
its shown moves and bench, both actions in words). After the turn the player reads what the
opponent did from the battle's own lines and adds `outcome` = `did`, `name`, `top1_right`,
`p_true`, `read_right` (a second note line, `{"outcome": ...}`).

Per game, in `results.jsonl` (arm `predict` only): `pred_made`, `pred_scored` (the log says what
it did: not when it fainted first or could not move), `pred_top1_right`, `pred_p_true` (a sum),
`pred_reads` (confident ones), `pred_reads_scored`, `pred_reads_right`, `pred_changed`,
`pred_changed_scored`, `pred_changed_right`, `pred_kept` (why the mix choice was kept, by reason).

`python bench/predict_run_report.py DIR/results.jsonl [--json f] [--examples N]` adds a run up.

## 3. Real games

Local server on port 8660, `gen9randombattle`, both sides `learner_update_001475.pt`. The opponent
is `ladder.py --player net` under another account. Ours: `--player predict --parallel 6 --helpers 6
--search-ms 500 --deadline 2.5 --think-ms 0`, default thresholds, the model fitted at 500 ms.
Another agent's server (port 8806) and whatever it played were on the laptop at the same time. Files (git-ignored): `runs/predict/main/` and
`runs/predict/run2/`, each with `results.jsonl`, `battles/`, `out.txt`; `predict_report.json` is
`bench/predict_run_report.py` of the run.

### Main run: 84 games (10:25 to 10:39), the rule WITHOUT "once bitten"

| | |
|---|---|
| Games started / finished | 84 / 84 |
| Lost for inactivity, forfeited, invalid orders, decisions that raised, random orders forced, helper exits | 0 of each |
| Predictions that raised, battles the tracker could not follow | 0, 0 |
| Record against the network alone | 51-33 (as p1 11-6, as p2 40-27). Not a strength measurement |
| Decisions | 2,997: 54 with one legal action, 1 Revival Blessing, 2,942 put to the search |
| Search answered | 2,941 of 2,942 (1 `narrow`) |
| Iterations per decision: mean / median / lowest tenth | 157,757 / 91,000 / 55,000 |
| Request to order: median / p95 / p99 / max | 1,607 / 1,808 / 2,019 / 2,681 ms |
| Over the 2.5 s deadline | 2 of 2,997 (2,547 ms on the first decision of the run, 2,681 ms once) |
| Payoff table: time mean / max; cells known | 382 ms / 1,230 ms (28 of 2,623 over 500 ms); 46.8% |
| **Predictions made** | 2,623 (every turn the search answered and the opponent had a choice to make) |
| Scored (the log says what it did) | 2,349. The other 274: it fainted first, could not move, or the game ended |
| **Its likeliest action was what it did** | **1,178 of 2,349 (50.1%)**. Mean probability given to what it did: 0.336 |
| Confident reads (60% or more) | 601, scored 563, **right 406 (72.1%)** |
| of them "it switches" | 115 scored, right 79 (68.7%). It switched on 336 of the 2,349 scored turns (14.3%) |
| **The mix choice was left** | **68 decisions (2.6% of the predictions) in 33 of the 84 games** |
| on those, the read was right | 40 of 62 scored (64.5%) |
| what was played instead | 12 times a switch of ours, 56 times another move. 15 of the 68 on an "it switches" read, 53 on "its move is X" |
| **Double switches (read "it switches", and we switch)** | **0** |
| The mix choice was kept because | `unsure` 2,022, `no gain` 456, `coverage` 39, `blind` 19, `risk` 19 |

Calibration against this opponent (the network, not people), the model's likeliest action:

| Said between | Turns | Said | Was right |
|---|---|---|---|
| under 30% | 394 | 25.7% | 27.2% |
| 30 and 40% | 620 | 34.9% | 40.8% |
| 40 and 50% | 480 | 44.7% | 53.3% |
| 50 and 60% | 373 | 54.8% | 57.6% |
| 60 and 70% | 253 | 64.7% | 65.2% |
| 70 and 80% | 159 | 74.6% | 77.4% |
| 80 and 90% | 69 | 83.5% | 85.5% |

The model is fitted on people and is better calibrated on the network than on them (62% where
it said 70 to 80% on the recorded people, 77% here): the network plays closer to what the
search expects than people do.

What the run says about the rule:

- **No double switch happened.** On the 120 confident "it switches" reads the best other action
  was a switch of ours 23 times and none passed (`no gain` 19, `risk` 2, `blind` 2). What passed on a switch
  read were 15 moves: an attack that is better against the Pokemon coming in, or a set-up move.
  The table is the limit: a switch of ours that the search hardly visited has no row.
- **The games in which the move was changed were won less often: 12 of 33, against 39 of 51
  without a change.** They are also the longer games (38.5 turns on average against 26.8). Among
  the 45 games of 28 turns or more: 8 of 27 with a change, 12 of 18 without. This is an
  association, and a losing position invites changes (at low HP every action of ours is worth
  the same against the opponent's attack, so the risk test passes easily). It does not show
  that the changes lose games. It does not show that they win any. A `mix` arm beside a
  `predict` arm, over some hundreds of games, is what would tell.
- **The same wrong read was acted on seven times in eight turns** (third example). After the run the rule
  got "once bitten": a read that was wrong against a Pokemon is not acted on again against it in
  that battle (`PredictSettings.once_bitten`, on by default, `--no-predict-once-bitten`). In this
  run it would have kept the mix choice on 7 of the 68.

### Three turns on which it changed the move (main run)

1. **Right read, and a switch of ours.** `battle-gen9randombattle-149887`, turn 8. Our Haxorus
   (45%) faces Scizor (33%, no move shown yet). The mix choice was Scale Shot
   with Terastallization. Expected: Bullet Punch 60%, Close Combat 15%. Played: switch to
   Moltres (gain 0.24, risk 0). What happened: Scizor used Bullet Punch, Moltres resisted it
   (278 to 230 HP) and Scizor fell to 23% from its Life Orb. The game was lost all the same.
2. **Right "it switches" read, the Pokemon named came in.** `battle-gen9randombattle-149901`,
   turn 8. Our Cinderace (88%) faces Serperior (57%). The mix choice was High Jump Kick.
   Expected: a switch 83% (to Glimmora 57%, to a Pokemon not shown 26%). Played: U-turn (gain
   0.13, risk 0). What happened: Serperior went out, Glimmora came in, U-turn hit it (55% to
   33%) and brought in our Gliscor against it. The same read and the same answer were right
   again on turn 12. The game was won.
3. **Wrong read, seven times in eight turns.** `battle-gen9randombattle-149935`, turns 8 to 15. Our
   Sandaconda (16% HP) faces Latias (36%). The mix choice was Stone Edge every turn. Expected: a
   switch, 61 to 83% each turn (a Pokemon not shown 28 to 51%). Played: Earthquake (turn 8, into
   Latias's Levitate: no effect), Coil, Glare, Glare, Coil, Glare, Earthquake. What happened:
   Latias stayed every time and used Draco Meteor four times and Psyshock twice (once it
   was paralysed). The risk test passed each time with 0: at 16% HP against a faster Latias
   every action of ours was worth the same if it attacked. The game was won. The read is the
   model's weak one, and nothing remembered that it had just failed: hence "once bitten".

### Run 2: 48 games with "once bitten" (10:43 to 10:50)

| | |
|---|---|
| Games started / finished | 48 / 48 |
| Lost for inactivity, forfeited, invalid orders, decisions that raised, random orders forced, helper exits | 0 of each |
| Record against the network alone | 28-20 (as p1 4-5, as p2 24-15). Not a strength measurement |
| Decisions | 1,552: 39 with one legal action, 1,513 put to the search, 1,513 answered |
| Iterations per decision: mean / median / lowest tenth | 154,777 / 79,000 / 46,000 |
| Request to order: median / p95 / p99 / max | 1,603 / 1,818 / 2,216 / 3,135 ms |
| Over the 2.5 s deadline | **10 of 1,552**. The event loop ran more than 0.5 s late 8 times (2.1 s at the worst): I ran my analysis scripts beside this run, which the search player's notes warn against |
| Payoff table: time mean / max; cells known | 392 ms / 1,129 ms; 39.6% |
| Predictions made / scored | 1,332 / 1,187 |
| Its likeliest action was what it did | 626 of 1,187 (52.7%) |
| Confident reads | 318, scored 287, right 205 (71.4%). "It switches": right 50 of 65 (76.9%) |
| The mix choice was left | 21 decisions (1.6% of the predictions) in 18 games; read right on 10 of 20 scored |
| what was played instead | 7 times a switch of ours, 14 times another move |
| **Double switches** | **3** (read "it switches" and we switched): one right, one wrong, one where the opponent could not move |
| The mix choice was kept because | `unsure` 1,014, `no gain` 235, `coverage` 27, `blind` 14, `risk` 13, **`bitten` 8** |
| Games with a change / without | won 8 of 18 / 20 of 30 |

4. **A double switch** (run 2). `battle-gen9randombattle-149973`, turn 11. Our Tentacruel (100%)
   faces Magearna (100%, shown: Thunder Wave; its shown bench: Cyclizar, Urshifu-Rapid-Strike).
   The mix choice was Surf. Expected: a switch 73% (a Pokemon not shown 30%, Urshifu 23%,
   Cyclizar 20%); second most likely Volt Switch 12%. Played: switch to Grimmsnarl (gain 0.12,
   risk 0, 85% of the predicted probability behind known cells). What happened: Magearna went
   out and Cyclizar came in, in the same turn as our Grimmsnarl. The read "it switches" was
   right, the Pokemon named most likely was not the one. The game was lost.
   The two others: turn 2 of `149986`, our Spiritomb (50%) against Gallade, expected a switch
   63%, played switch to Brambleghast, Gallade stayed and used Leaf Blade, which Brambleghast
   resisted (240 to 142 HP): wrong read, bounded cost, game won. Turn 17 of `149964`, our
   Oranguru (10%) against Slaking, expected a switch to Passimian 66%, played switch to
   Golduck, Slaking was loafing (Truant): nothing to score.

## Tests

`python -m pytest tests/test_opponent_model.py -q -p no:cacheprovider`: 25 tests, no Foul Play, no
server, no checkpoint. The model on hand-made situations (a distribution over what it can do,
the directions of the switch probability, a fit on made-up rows that recovers their rates);
`choose()` on hand-made tables (every kept reason, the double switch, a wrong read that would
cost too much, Tera forms, 3,000 random positions never give an illegal action and never leave
the mix choice without a confident read, a gain and a bounded risk); the tracker on the seven
recorded battles (a context at every turn, every turn closed once and in order); the player in
`predict` mode on the recorded battles with a fake bridge: without a table it plays move for
move what `mix` plays and its counts add up, with a made-up table it leaves the mix choice only
on a confident read, a read that failed against a Pokemon is not acted on again against it, a
model that raises costs nothing, a deadline too short for search plus table is refused.
After the last change: `tests/test_opponent_model.py`, `tests/test_search_player.py` and `tests/test_tera.py`
together, 97 passed in 49 s (25 + 56 + 16). The run before it, made while my notes script and
another agent's games were using the machine, had 1 failure in 97: the search player's timing
test `test_a_search_that_never_answers_costs_the_deadline_and_no_more` counted 2 late decisions
where it allows 1. Alone and in the rerun it passes.

## Commands

```
# inside WSL, from /mnt/c/Users/you/ps-ppo-search (6 helpers at most):
~/psppo-bench/foul-play/.venv/bin/python bench/opp_model_replay.py --split train --allow-v1 --games 90 --ms 500 --helpers 6 --tag train90_ms500
        [--split test|all] [--skip N] [--payoff-ms 400 --payoff-mode full|engine] [--iterations N]
# Windows (python = C:\Users\you\ps-ppo\.venv\Scripts\python.exe), from the worktree:
python opponent_model.py fit  --replay runs/opp_pred/model/replay_train90_ms500.jsonl          # writes opponent_model.json
python opponent_model.py eval --replay runs/opp_pred/model/replay_train90_ms500.jsonl --eval-replay runs/opp_pred/model/replay_test60_ms500.jsonl --json out.json
python ladder.py --server local:PORT --user NAME --player predict --games N --parallel 6 --helpers 6 --search-ms 500 --deadline 2.5
        [--predict-confidence 0.60 --predict-gain 0.08 --predict-risk 0.08 --payoff-ms 400 --no-predict-once-bitten] --out DIR/results.jsonl --stop-file DIR/x.stop
python ladder.py ... --arms mix,predict            # the comparison that was not run
python bench/predict_run_report.py DIR/results.jsonl --examples 5
```

`runs/opp_pred/model/` (git-ignored): `replay_all_ms150.jsonl`, `replay_train90_ms500.jsonl`,
`replay_test60_ms500.jsonl` (= `replay_test24_ms500_full400.jsonl` + `replay_test36_ms500.jsonl`),
`replay_test24_ms500_engine400.jsonl`, the `eval_*.json` behind every table here.

## API for the next agent

- `from opponent_model import OpponentModel, OpponentTracker, PredictSettings, Prediction, choose, search_view, fold_search, column_probabilities, fit, evaluate, reliability, truth_name`
- `OpponentModel.load(path=None).predict(ctx, advice_or_view_or_None) -> Prediction(dist, p_switch, p_tera, tera, with_search)`; `.ranked()`, `.top(n)`.
- `OpponentTracker(tag).feed(line, role)` for every line of the room, `.context(turn)` (None when not there), `.closed()` -> `[(turn, label, name or None)]`, `.broken`.
- `choose(prediction, advice.payoff_table, advice.payoff_theirs, search_view(advice), mix_action, mask, shares14, PredictSettings()) -> (action, info)`.
- Player: `search_player()(mode="predict", predict=PredictSettings(...), opponent_model=..., on_outcome=fn, ...)`. Also an arm: `--arms mix,predict`.
- `search_player.MODES` now ends with `"predict"`; `ladder.PLAYER_MODES` the same.

## Not done, not measured

- **Whether `predict` wins more than `mix`.** Not measured: no mix arm was played beside it, and
  with 2 changed decisions in 100 it would take a great many games to see. The records above
  are against the network alone and say nothing about it. The one thing they do say points the
  wrong way (games with a change won 20 of 51, without 59 of 81), and may be the position's
  doing and not the change's. Until a `mix,predict` run says otherwise, `mix` is the mode to
  put on the ladder.
- **The thresholds are first guesses.** 0.60, 0.08, 0.08, 0.70, 0.15 were set from the train
  tables and from what the engine's scale means, before the games, and not changed after. No
  value was tried against another. "Once bitten" was added after the main run because of what
  that run showed, and run 2 (48 games) is the only play it has had.
- **The risk test is easy to pass from a lost position.** At low HP against a faster Pokemon
  every action of ours is worth the same if it attacks, so the risk is 0 and any gain against
  the read is taken. Many of the changes are of this kind (the Sandaconda example). A floor
  on our own HP, or a risk measured over more than 2 plies, was not tried.
- **Calibration above 60% is off by 7 to 12 points on the test split** and the model was not
  corrected for it (that would have been tuning on the test split). A fit on more than 90 games
  at 500 ms is the first thing to try: the 150 ms replay of all games took 12.5 minutes, the 259
  train games at 500 ms would take about 25.
- **The switch read is weak** (57% right at a bar of 60%, the incoming Pokemon named right 40%
  of the time, 44% when it had been shown: the even guess). Whom people bring in is not
  predicted better than by the search, which is not better than an even guess over the shown
  bench. The recorded games hold no feature for it here (no "who resists what we just used").
- **The model was fitted on people at 1100 to 1500 playing an older bot.** In the real games here
  the opponent is the network, which it predicts better than people (see below). Whether people
  at 1900 are more like the network or like the recorded people is not known.
- **The old bot double switched in 4.4% of the opponent's switches**, so nothing here says how
  people react to an opponent that does it often.
- **The table is 2 plies and incomplete.** An answer the search hardly looked at has no row.
  Asking the helper for chosen cells (our candidates x the predicted actions) needs a new
  request in `bench/fp_bridge_server.py` (after the search, with the worlds kept per battle):
  not done, those files are not mine. It is the change that would let a real double switch be
  found more often.
- **The table can overrun its budget**: one cell is never cut short. 28 of 2,623 tables in the
  main run took over 500 ms, the slowest 1,230 ms, and 12 of 4,549 decisions over both runs
  ended after the 2.5 s deadline (the slowest at 3.1 s). `--deadline 2.3` or `--payoff-ms 300`
  buys the margin; neither was run.
- Choice-item locks, forced switches and pivot targets are not predicted (no prediction is made
  at a forced switch request). The Tera part is scored but `choose()` only uses it to split a
  move's probability between its two columns.
- `opponent_model.py` imports `bench.opp_dataset` and `bench.opp_predict_eval` (the tracker and
  the options are the scout's code, on purpose). The move pools come from
  `heuristic/cache/gen9randombattle.json`, read from `C:\Users\you\ps-ppo\heuristic\cache` when the
  worktree has none; without the file the model still runs, on the moves the search sampled.
- `bench/search_run_report.py` does not know the `pred_*` fields; `bench/predict_run_report.py` does.
- Not run on Linux. Not run on the public ladder. `ladder_graph.py` shows a `predict` arm like
  any other arm, without the prediction counts.
