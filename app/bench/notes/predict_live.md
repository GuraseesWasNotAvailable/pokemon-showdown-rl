# Live predictions, the guard, and the set-up test opponent

2026-10-07, branch `predict-live` (from `search` at e65a7e7). Files: `search_player.py`, `ladder.py`,
`setup_moves.py` (new), `bench/setup_player.py` (new), `bench/guard_replay.py` (new),
`bench/ladder_opp.py`, `bench/gauntlet.py`, `tests/test_predict_live.py` (new),
`tests/test_search_player.py` (the list of modes).

Three things, asked for after 9 games on the ladder ("we are getting beat by random teams which
just quiver dance once and sleep all my pokemon", "also im not seeing predictions, I want that"):

1. **Live predictions** (on by default with a search): for every decision the opponent's three
   likeliest actions, what we play, and after the turn whether the read was right. No move changes.
2. **The guard** (`--player mix-guard`, or `--guard`; OFF by default): a small change of the veto
   rule against set-up moves and sleep, taken from the look at the 7 losses.
3. **A set-up test opponent** (`bench/setup_player.py`): SimpleHeuristics that sets up, puts to
   sleep and heals, for measuring the guard on a local server.

## 1. Live predictions

What prints, one line per decision and one per turn, `[1408]` being the game (last digits of its
number), so two games at once stay readable:

```
[4405] T5 Cyclizar: predict Rapid Spin 38%, Draco Meteor 22%, Knock Off 22% | we play Hyper Voice
[4405] T5 read: right (Rapid Spin)
[4405] T6 Cyclizar: predict Knock Off 54%, Rapid Spin 28%, Draco Meteor 8% | we play switch Scovillain
[4405] T6 read: wrong, it played Rapid Spin (our 2nd guess)
[4402] T17 Articuno-Galar: predict Hurricane 72%, Psyshock 16%, Calm Mind (set-up) 6% | we play U-turn + Tera [guard: boosted +4]
[4402] game   4: WIN  in 19 turns vs opp2140140 (1000) ... [mix, search 23/24, likeliest right 6/18, top 3 16/18]
```

Set-up and sleep moves are marked `(set-up)` / `(sleep)`. A turn the log cannot score says so
(`read: not scored (the game ended in this turn)`, `read: not scored, it was locked into Outrage`).

Where the numbers come from: every answer of the bridge already carries the opponent's side of
the search (its options and their share over the sampled worlds). `opponent_model.OpponentModel`
turns that, plus base rates from 426 recorded human games, into a probability for each action
(likeliest right 47% on held-out human games, top 3 81%; calibrated up to about 60%). The same
tracker as the "predict" mode follows each battle (`OpponentTracker`, fed every line). No
payoff table is asked for and the search time is unchanged: the prediction is a few dictionary
operations on the answer already in hand. A forced switch gets no prediction (the opponent
chooses nothing then), nor does a decision the search did not answer.

What is kept:

| Where | What |
|---|---|
| console | the two lines above; the game line with `likeliest right k/n, top 3 m/n`; at the end "predictions of the opponent's action: likeliest right ..., in the top 3 ..." and the search summary's "opponent predicted on N turns, scored M: its likeliest action was right on K (x%), what it did was in the top 3 on L (y%)" |
| `battles/<tag>.jsonl` | a note `{"prediction": {turn, line, top, p_switch, played, mode, guard}}` per decision, `{"outcome": {battle, turn, mode, predict}}` per turn (`predict.outcome` has `text`, `name`, `top1_right`, `top3_right`, `p_true`); the decision note's record has `predict` too |
| `results.jsonl` | per game `pred_made`, `pred_scored`, `pred_top1_right`, `pred_top3_right`, `pred_p_true` (sum of the probability given to what happened) |

`--no-predictions` turns all of it off (no tracker, nothing printed). The player option is
`live_predictions=True` (search_player); `bench/gauntlet.py` turns it on without printing.

**Printed by a thread of their own** (`ladder.ConsoleLines`, after the review of 2026-10-07):
the prediction lines are made on the event loop before the order is sent. In the old Windows
console (conhost with QuickEdit) a click or a selection in the window holds every write until
Esc, and a `print` there held the move and the websocket's keep-alive pings with it. The
prediction, read and game lines now go into a queue that one thread prints, in order; a held
window holds only that thread (at most 5000 lines wait, more are dropped and counted). The game
line starts with the game's `[tag]` like its prediction lines.

**Illusion**: the tracker (`bench/opp_dataset.GameRows`) follows `|replace|`. The revealed
Pokemon becomes the active one, with the disguise's HP and status and the moves used since it
came in; the disguise gets back its own state, or is forgotten when it was never seen itself.
Before, the lines kept the disguise's name after the reveal ("T25 Zapdos-Galar: predict Hyper
Voice 67%") and its faint was put on the disguise.

## 2. The guard (`MixSettings.guard`, mode `mix-guard`)

OFF unless asked for, so a measurement can compare `--arms mix,mix-guard`. Veto rule only.

**When it fires** (`guard_reason`, `Threat`):
- the opponent's active Pokemon is at +2 or more in attack, defence, special attack, special
  defence or speed (`guard_boost`), or at +1 with a set-up move as its last action;
- or the read puts 40% or more (`guard_read`) on a set-up move, or on a sleep move while ours can
  still fall asleep (the prediction above; the search's own shares when there is none);
- or the search's favourite is one of our passive moves while the network gives 70% or more
  (`guard_sure`) to an action that is not ("network sure").

**What it does** (`search_player._guarded_choice`):
- our passive moves are left out (`setup_moves.is_passive`: our own set-up, Roost / Recover /
  Rest, Protect, Substitute, hazards, screens, Defog), except when it fired on a sleep read alone
  (Substitute is an answer to sleep). Haze, phazing, Encore, Taunt, burns and paralysis stay in;
- Sucker Punch and Thunderclap are left out when the read puts 40% or more on a status move;
- the veto is measured against the search's largest share among what is left, so a set-up move
  the search favours does not veto the attack the network wants;
- the network's likeliest action left is played when the veto keeps it, else the search's
  largest share among what is left: never an action that neither of the two picked.

If everything is passive, nothing is left out. The record of a decision it fired on has `guard`:
why, what it left out, `net_pick`, `search_pick`, `by`, `veto` (what the plain veto would have
played), `changed`, and the threat's numbers. Each game counts `guard_fired` and `guard_changed`.

**Why these and not more.** From the look at the 7 losses (5 had a sweeper we did not stop):
the veto rule played Milotic on a forced switch that neither side wanted (game 1 T3), our own
Swords Dance vetoed the network's Shadow Sneak on the decisive turn (game 4 T42), Forretress
spent its Tera on Toxic Spikes against a +2 Dondozo (game 4 T7), Oricorio looped Roost, and
Sucker Punch went into Agility (game 6 T17). Larger hand-written answers (pick the phazer, hit
on the side it did not boost) were left out on purpose: they would be a policy of their own,
measured on nothing.

**Replayed on the owner's ladder games** (`bench/guard_replay.py`, the 34 battle files of
2026-10-07 on the box, 935 searched decisions). It recomputes the veto from each decision note
(the replayed veto equals what was played on every mix decision) and the guard from the same
numbers; the search is not asked again.

| | decisions | guard fired | move changed |
|---|---|---|---|
| all games | 935 | 236 | 28 |
| games won | 438 | 91 | 8 |
| games lost | 448 | 142 | 20 |

The positions the look named:

| Game, turn | Played | Guard plays | |
|---|---|---|---|
| 2693998716 T3 (forced, +2 Fezandipiti) | Milotic (neither's pick) | Spiritomb (the search's) | as the look asked |
| 2694007454 T7 (+2 Dondozo) | Tera Toxic Spikes | Volt Switch (the search's) | as the look asked |
| 2694007454 T42 | Swords Dance (search veto) | Shadow Sneak (network sure 79%) | as the look asked |
| 2694015275 T17 (Agility read) | Sucker Punch | switch Lumineon | Sucker Punch left out; whether the switch is better is not known |
| 2693998921 T22-T23 (Chimecho) | Knock Off, Exeggutor | the same | not fixed: both sides agreed; needs a longer search (look, fix 3) |
| 2694007454 T13 / T23 forced | Dusknoir, Honchkrow | the same | not fixed: the network's pick survives the veto |

Some changes in won games look doubtful (our own Bulk Up or Curse + Recover left out against a
Pokemon that is boosting too). That is what the measurement is for: the replay says where the
rule acts, not whether it wins.

## 3. The set-up test opponent (`bench/setup_player.py`)

SimpleHeuristicsPlayer with, on a move request in singles, in this order: switch when
SimpleHeuristics would; Sleep Talk when asleep; a recovery move when boosted and below half HP;
a sleep move when the target can fall asleep (no status, Sleep Clause, Substitute, Yawn,
Safeguard, terrains, Insomnia-like abilities, powder against Grass / Overcoat / Safety Goggles);
a set-up move when not about to be knocked out (`in_danger`: half HP or less, hit 4x by its types,
2x below 85%, or the opponent at +2 in an attacking stat below 85%) and below +4 (`max_stage`) in
what it raises; otherwise SimpleHeuristics. Every move it picks is one of
`battle.available_moves`. It counts what it did (`setup_counts`).

```
python bench/ladder_opp.py --kind setup --port P --games 30 --parallel 2        local ladder
python bench/gauntlet.py opp --kind setup --port P --pairs ACC:CHA:OURS ...     head to head (also --kind heuristic)
```

## Tests and runs (box, 2026-10-07)

`tests/test_predict_live.py`: the move tables, the words, the guard on the numbers of the lost
games and on 3,000 random positions (never illegal, never a third action), mix with predictions
on the 7 recorded battles (the same moves as without, no payoff asked, every scored turn counted),
mix-guard on the same battles, the ladder's counts, the set-up opponent's rules. The existing
`tests/test_search_player.py` and `tests/test_opponent_model.py` pass unchanged (but for the list
of modes).

Two runs of `ladder.py` on local servers (ports 9601, 9602), the owner's network
(`learner_update_011850.pt`, the same file), search 500 ms, deadline 2.5 s, network on the CPU
with 2 threads, `bench/ladder_opp.py` laddering against it. Box under the training run's load
(load average 56 to 77 on 64 cores). Not measurements: 20 and 10 games against weak bots, to see
the console and the counts.

| | mix with predictions vs SimpleHeuristics | mix-guard with predictions vs the set-up opponent |
|---|---|---|
| games | 20 (2 at a time, 3 helpers) | 10 (1 at a time, 2 helpers) |
| record | 18-2 | 10-0 |
| decisions, answered by the search | 454, 448 (0 fell back) | 253, 252 (0 fell back) |
| predictions made | 398 | 225 |
| scored (the log says what it chose) | 326 | 192 |
| likeliest right | 139 (43%) | 76 (40%) |
| what it did in the top 3 | 288 (88%) | 165 (86%) |
| not scored | 72: fainted before it moved 65, asleep 3, locked into Outrage 2, no action in the log 2 | 33 |
| request to order, mean / slowest | 1,178 / 1,295 ms | 1,196 / 1,294 ms |
| search time per decision | 1,191 ms | 1,199 ms |
| event loop late by > 0.5 s | 0 of 1,076 checks | 0 of 1,211 |
| orders refused, raised, forced random | 0, 0, 0 | 0, 0, 0 |
| guard fired / changed the move | - | 85 / 12 (boosted 53, set-up read 11, set up last turn 9, network sure 4, sleep read 2 of the printed lines) |

The request-to-order time is the search's time plus 2 to 4 ms: the prediction costs nothing a
player could see. Every battle file holds one `prediction` note per decision and one `outcome`
note per turn; every game line of `results.jsonl` has the `pred_*` counts. The opponents did set
up and put to sleep (Shift Gear, Calm Mind, Dragon Dance, Bulk Up, Spore in the printed reads).
Examples of the guard changing a move, as printed:

```
[4404] T7 Suicune: predict Substitute 31%, Calm Mind (set-up) 28%, Scald 17% | we play Secret Sword [guard: boosted +2, instead of Calm Mind]
[4406] T3 Stantler: predict switch to one not seen yet 42%, Body Slam 26%, Thunder Wave 19% | we play Leaf Blade [guard: network sure 75%, instead of Swords Dance]
[4408] T17 Moltres-Galar: predict Fiery Wrath 33%, Nasty Plot (set-up) 31%, Hurricane 16% | we play Iron Head [guard: boosted +3, instead of Sucker Punch]
```

Likeliest right 40 to 43% against SimpleHeuristics is below the 47% on held-out human games: the
model was fitted on people, and these bots switch and pick moves by their own fixed rules.

## Measured (box, 2026-10-07, learner_update_011850.pt, 500 ms, Foul Play at 450 ms = 0.90 of our iterations)

Wins, Wilson 95% interval in brackets; first and second player split evenly in every arm.

| Opponent | mix (control) | mix-guard | predict |
|---|---|---|---|
| Foul Play, fair compute | 96-61, 61.1% (53.3 to 68.4) | 88-68, 56.4% (48.6 to 63.9) | 37-54, 40.7% (31.1 to 50.9) |
| Set-up test bot | 175-8, 95.6% (91.6 to 97.8) | 171-12, 93.4% (88.9 to 96.2) | not run |
| Network alone | 78-80, 49.4% (41.7 to 57.1) | 83-74, 52.9% (45.1 to 60.5) | not run |

- Guard minus mix, average of the three opponents: -1.1 points (-6.5 to +4.3); on the games where
  the guard changed a move, -1.5 (-8.7 to +5.8). No gain. It was measured only against bots.
- Predict minus mix against Foul Play: -20.5 points (-32.5 to -7.6), z -3.2; -24.2 within the two
  sessions that ran both arms (59/91 against 37/91). Predict stays off.
- The set-up test bot is too easy (mix 95.6%) to measure an anti-set-up rule.
- With this checkpoint mix only ties the network alone (49.4%); with u2575 it won 61%. Whether
  the search still pays for its 1.2 s a move against Foul Play is not measured (`--arms net,mix`).

How well the prediction read the opponent:

| Opponent | Turns scored | Likeliest right | Top 3 | Shown % against what happened |
|---|---|---|---|---|
| Foul Play | 10,754 | 61.5% | 86.8% | too low: shown 35-45% was right 48-62%; ECE 0.14 |
| Network alone | 8,543 | 51.9% | 84.7% | right, ECE 0.04 |
| Set-up bot | 5,666 | 42.5% | 84.7% | too high above 50%: shown 85% was right 66% |
| All | 24,963 | 53.9% | 85.6% | ECE 0.065 |

Reports: `bench/measure_report.py` and `bench/guard_affected.py` on the box folders
`~/dev/predict-measure-data/{fp,fp3,fp4,su,su3,su4,nt,nt3,nt4}`.

## Not done

- **The guard showed no gain against bots** (above). Its one fair test left is human set-up
  sweepers, which needs a harder set-up opponent (the network plus forced set-up) or ladder games.
- The look's other fixes are not built: "never a third action" outside the guard, Tera only when
  one side picked the Tera form, a longer search on the turns the guard fires.
- The guard reads boosts from poke-env's view of the battle; Haze / Clear Smog resets are seen
  as poke-env sees them.
