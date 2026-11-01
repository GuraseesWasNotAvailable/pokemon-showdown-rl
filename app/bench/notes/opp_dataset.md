# What real opponents do: dataset, scorer, base rates, replay through the search

Built 2026-10-06 on branch `search`. Everything below was run; the numbers are from those runs.
Source: 426 rated gen9randombattle games an older bot of the owner played against people
(`C:\Users\you\showdown-bot\battles\<date>\battle-*.json`, read only). No server was started and
no port of 8630-8639 was used: none of this needs one.

| File | What it is |
|---|---|
| `bench/opp_dataset.py` | log parser, one row per opponent decision, the split. Writes `runs/opp_pred/dataset.jsonl` (34 MB), `games.jsonl`, `dataset_summary.json` |
| `bench/opp_predict_eval.py` | the scorer, the baselines, the base rates. Writes `runs/opp_pred/eval_baselines.json` |
| `bench/opp_replay.py` | a recording as a websocket message stream (`messages_up_to`, `events_up_to`, `build_stream`) and its replay through `bench/fp_bridge_server.py` |

Names: a game is `g` + 12 hex of a salted hash, an opponent `opp` + 10 hex, our own name is
`replayme` in the streams. No real name is written by any of the three scripts, in `runs/` either.

## 1. The dataset

`python bench/opp_dataset.py` (Windows venv python; 20 s). Row kinds: `turn` (what the opponent
chose at the start of a turn), `forced` (whom it sent in after a faint), `pivot` (whom it sent in
after its own U-turn and the like). `label.known` is false when the log cannot say what it chose,
`label.locked` when it had no choice. `ctx` is what was known before the decision, `our_action`
what we did in the same turn (not context). The format is in the file's docstring.

Split by game, seed 20261006, 70/30:

| | train | test |
|---|---|---|
| games | 298 | 128 |
| rows of all kinds | 8,281 | 3,479 |
| turn rows | 7,045 | 2,970 |
| turn rows with a known, free choice (the scored ones) | 6,215 | 2,648 |
| of those: moves / switches | 5,220 / 995 | 2,262 / 386 |
| turn rows unknown | 807 | 319 |
| turn rows locked into a move | 24 | 3 |
| forced rows (after a faint) | 1,086 | 451 |
| pivot rows | 150 | 58 |
| turns with a Tera | 186 | 76 |

Turn rows are 10,015, which is the sum of the recorded game lengths. Unknown is 11.2% of turns
(1,126): fainted before it moved 877, asleep 87, paralysed 42, flinched 38, frozen 27, the game
ended in that turn 43, nothing in the log 7, Taunt and Heal Block 4, recharge 1.

Limits of the labels:
- The scored rows leave out the turns where the opponent was knocked out before it moved. Those
  are not a random 9% of turns: they are turns in which it was slower and in range.
- `locked` is only what the log marks (`[from]lockedmove`: Outrage, the second turn of a charging
  move, recharge). A Choice item lock is not marked: such an opponent can only repeat or switch.
- A Tera is a flag on the label, not part of the action name.
- Zoroark's Illusion is recorded as the Pokemon that was shown. Abilities copied by Trace show as
  the copied ability.
- "Type disadvantage" is a rule on types only: our best STAB type against its types is more
  effective than its best STAB type against ours (Tera types counted once shown).

## 2. The scorer and the baselines

`python bench/opp_predict_eval.py`. A predictor is `ctx -> {action name: probability}` with names
`move <move id>`, `switch <species id>` and `switch ?` for a switch to a Pokemon not shown yet
(the truth is `switch ?` when the Pokemon that came in had not been shown). Ties are scored by
their expected value. Log loss is `-ln(max(p, 1e-4))`. Use it from code:

```python
from bench.opp_predict_eval import score, Fitted
from bench.opp_dataset import load_rows
rows = list(load_rows()); test = [r for r in rows if r["split"] == "test"]
print(score(my_predictor, test))          # my_predictor(ctx) -> {"move earthquake": 0.6, "switch ?": 0.1, ...}
```

Test split, 128 games, 2,648 scored turns (386 switches, 148 of them to a Pokemon already shown;
2,262 moves, 1,019 of them a move already shown). Mixing weights and switch rates were fitted on
the train split only (they are printed and saved with the results).

| predictor | top-1 | top-3 | log loss | switch or stay | switch AUC | switch was our top-1 | right target given a switch | same, target already shown | top-1 on move turns | same, move already shown |
|---|---|---|---|---|---|---|---|---|---|---|
| uniform over its known options (revealed moves, revealed bench) | 15.5% | 40.5% | 5.03 | 49.9% | 0.524 | 37.4% | 62.3% | 43.9% | 11.8% | 26.2% |
| uniform over revealed moves + its species' move pool + switches | 13.3% | 39.8% | 2.07 | 85.5% | 0.521 | 12.8% | 62.3% | 43.9% | 13.4% | 14.3% |
| repeats its last move | 27.4% | 49.5% | 1.98 | 85.4% | 0.499 | 1.9% | 62.3% | 43.9% | 31.8% | 63.1% |
| its most used move so far | 26.5% | 48.9% | 1.99 | 85.4% | 0.499 | 1.9% | 62.3% | 43.9% | 30.7% | 60.7% |
| most damage to our active, among revealed moves (pool when none) | **32.6%** | 53.3% | 1.92 | 85.4% | 0.533 | 2.5% | 62.3% | 43.9% | 37.7% | 62.7% |
| most damage, among revealed moves and pool | 29.8% | 51.2% | 1.97 | 85.4% | 0.522 | 0.2% | 62.3% | 43.9% | 34.9% | 38.5% |
| base rate: 16.0% switch, the rest uniform | 16.1% | 46.0% | 2.02 | 85.4% | 0.548 | 20.6% | 62.3% | 43.9% | 15.3% | 17.1% |
| base rate by bucket (type disadvantage, HP under 1/3, just came in) | 16.9% | 47.4% | 2.00 | 85.4% | **0.707** | 32.0% | 62.3% | 43.9% | 14.3% | 14.9% |
| prior_combo: bucket switch rate, then 0.2 last move, 0.2 strongest move, rest uniform | 31.5% | **61.6%** | **1.86** | 85.4% | 0.699 | 15.9% | 62.3% | 43.9% | 34.2% | 48.7% |

What a later claim has to beat: **top-1 32.6%, top-3 61.6%, log loss 1.86, switch-or-stay 85.4%
(that is "it never switches": no baseline beats it), switch AUC 0.707, target given a switch
62.3% (43.9% when the target had been shown)**. Every baseline spreads its switch mass evenly,
which is why the target columns are the same in every row: that column is the uniform guess.

"Damage" is base power x STAB x type effectiveness against our active Pokemon, nothing else.
The move the opponent used is in its species' random battle pool in 99.7% of move turns
(7,458 of 7,482), so the pool (`ctx.opp.active.pool`, from `heuristic/cache/gen9randombattle.json`)
covers what it can click.

After a faint (451 test rows; not in the table, computed once by hand with the same rule): the
uniform guess over revealed bench plus `switch ?` names the replacement 58.4% of the time, and
69.2% of the 179 times it had been shown.

## 3. Base rates

All 426 games (train and test separately are in `runs/opp_pred/eval_baselines.json`). Intervals
are 95% Wilson intervals.

**Switching** (scored turns):

| Situation | Switch rate |
|---|---|
| all scored turns | 15.6% (14.8 to 16.4), 1,381 of 8,863 |
| it has somebody to switch to | 16.6% (15.8 to 17.4), 1,381 of 8,313 |
| type disadvantage | **30.5%** (28.7 to 32.4), 724 of 2,372 |
| strong disadvantage: we hit it super effectively, it does not hit us so | **35.6%** (33.4 to 37.9), 603 of 1,694 |
| even types | 13.3% (12.2 to 14.4), 476 of 3,585 |
| type advantage | 7.7% (6.7 to 8.8), 181 of 2,356 |
| HP under 1/3 | 17.8% (15.3 to 20.5), 150 of 844 |
| HP 1/3 to 2/3 | 14.9% (13.4 to 16.5), 292 of 1,963 |
| HP over 2/3 | 17.1% (16.1 to 18.1), 939 of 5,506 |
| just came in, any way | **7.4%** (6.5 to 8.4), 227 of 3,070 |
| came in by its own switch last turn | 6.2% (5.0 to 7.8), 72 of 1,158 |
| came in after a faint | **1.4%** (0.9 to 2.2), 18 of 1,302 |
| came in by a pivot move | 7.4% (4.5 to 12.0), 14 of 189 |
| on the field for 1 turn | **30.6%** (28.7 to 32.5), 679 of 2,221 |
| on the field for 2 turns or more | 15.7% (14.5 to 17.1), 475 of 3,022 |
| it is statused | 20.4% (17.0 to 24.4), 91 of 445 |
| it has a boost | 7.5% (6.4 to 8.8), 134 of 1,782 |
| it has a stat drop | 28.1% (24.6 to 31.9), 167 of 594 |
| we switched last turn | **36.6%** (34.2 to 39.1), 535 of 1,461 |

So: low HP by itself says nothing about switching. The type matchup, a stat drop, and above all
"we just brought something in" do. The Pokemon it switches to had already been shown in 40.3% of
switches (556 of 1,381). We switched in the same turn in 4.4% of its switches (61 of 1,381): the
old bot almost never double switched, so these games say little about how people answer one.
The type-disadvantage rate is 32.4% in train and 26.1% in test: read the table to about 5 points.

**The same move twice:**

| | |
|---|---|
| it used a move, and had used one before with this Pokemon still on the field: the same move | 40.2% (38.8 to 41.6), 1,805 of 4,491 |
| the same, counting a switch as not repeating | 32.0% (30.8 to 33.2), 1,805 of 5,640 |
| with 1 / 2 / 3 / 4 moves revealed | 37.5% / 45.9% / 38.5% / 37.2% |
| the move it used had been revealed before | 44.3% (43.1 to 45.4), 3,311 of 7,482 |

More than half of all move turns show a move we had not seen yet. That is the ceiling of every
predictor that only knows revealed moves, and the reason the move pool is in the rows.

**Terastallization:**

| | |
|---|---|
| games in which the opponent Terastallized | 61.5% (56.8 to 66.0), 262 of 426 |
| per turn in which it could and used a move | 4.5% (4.0 to 5.1), 243 of 5,393 |
| turn of the Tera: mean / median / quartiles | 15.2 / 14 / 8 to 21 (games last 23.5 turns on average, median 23) |
| by turn 0-2 / 3-5 / 6-8 / 9-11 / 12-14 / 15-17 / 18-20 / 21+ | 3.4% / 11.1% / 11.8% / 14.1% / 10.7% / 13.0% / 9.5% / 26.3% |
| in the first / second / last third of the game | 20.2% / 31.7% / 48.1% |
| its Pokemon left when it did (6, 5, 4, 3, 2, 1) | 35, 50, 44, 50, 36, 28 |
| its Pokemon had just come in | 41.2% (35.1 to 47.4), 100 of 243 |
| it was at a type disadvantage | 53.5% (47.2 to 59.7), 130 of 243 |
| its HP was over 2/3 | 76.5% (70.8 to 81.4), 186 of 243 |
| it had a boost | 31.7% (26.2 to 37.8), 77 of 243 |
| the Tera move was a damaging move / was Tera Blast | 69.5% / 4.5% |
| the Tera type was one of its own types | 21.4% (16.7 to 27.0), 52 of 243 |

(243 of the 262 Tera turns have a known move; the others were turns where it then could not move.)
People Tera late, at high HP, and half the time out of a bad type matchup: mostly a defensive
type change, 4 times in 5 to a type the Pokemon does not have.

## 4. Replay through the bridge helper

**The recordings do not hold the websocket messages and do not hold the request JSON.** `log` is a
flat list of battle lines with no `|request|` line in any of the 426 games. What they do hold is
enough to rebuild both:

| Needed | Where it comes from |
|---|---|
| message boundaries | the server writes a `\|t:\|` line each time it resolves choices; the opening is cut the way the server sends it (checked against the real streams in `tests/fixtures/foulplay_bridge_replay.jsonl.gz`) |
| which batch a decision belongs to | request ids count up by two per batch: batch k is rqid 2k+2 for p1 and 2k+3 for p2. In 368 of 426 games every recorded decision sits on a batch of the right kind (new turn or forced switch) and every new turn has a decision |
| `side.pokemon` | `myTeam` (ident, details, stats, moves, ability, item, Tera type as the first request listed them) + the decision's `mine` (order, condition, active) + our item losses and Tera read from the log |
| the active part | `decisions[].req` (move ids, pp, maxpp, disabled, trapped, maybeTrapped, canTera): only in the 288 games of recording version 2. In the 138 version-1 games it is rebuilt: every move offered, pp 16, Tera offered until used |
| requests without a decision | `{"wait": true}` (the opponent is choosing a replacement) |
| what we sent | `decisions[].choice`, fed as `chose` |

Streams built: **368 of 426 games** (8,949 move requests, 1,615 forced switch requests, 1,510
wait requests), 255 of them with the recorded active part. The other 58 are refused with the
reason (a request decided twice, a turn with no recorded decision, the bot joined midway).

Replayed through the helper as it stood in the worktree at the time (protocol version 2, the
other agent's edit in progress, sha1 `543ad176b9ff`; Foul Play `8f541f80`, poke-engine 0.0.48),
in WSL, 3 helpers, `decide` asked at every move and forced switch request, `--ms 50`:

| Run | Games | Decisions asked | Answered | Errors, failed battles | Iterations per decision, mean (median) | Helper time per decision | Wall time |
|---|---|---|---|---|---|---|---|
| test split, version 2 only | 24 | 633 (527 moves, 106 forced) | **633** | 0 | 47,051 (29,000) | 123 ms | 31 s |
| train split, version 1 and 2 mixed | 60 (44 v2, 16 v1) | 1,823 (1,545 moves, 278 forced) | **1,823** | 0 | 53,526 (34,000) | 122 ms | 83 s |

84 games, 2,456 decisions, every one answered, no battle Foul Play's code could not follow. The
16 version-1 games (463 decisions) went through as well. No helper was left running. The search's
largest-share move is the old bot's recorded move in 48.8% and 48.9% of decisions.

The helper's reply already carries the opponent's side of the search (`opponent.options`, added
by the other agent). `opp_replay.py` stores it per turn as `opp_pred`, Tera folded into the move
and switches to unseen Pokemon as `switch ?`, so the scorer reads it as it is:

`python bench/opp_predict_eval.py --pred runs/opp_pred/replay/decisions_<tag>.jsonl`

| The search's own prediction of the opponent, 50 ms | rows | top-1 | top-3 | log loss | switch or stay | switch AUC | switch was top-1 | target given a switch (already shown) | top-1 on move turns (move already shown) |
|---|---|---|---|---|---|---|---|---|---|
| 24 test games, 47,051 iterations per decision | 483 | **52.8%** | 81.0% | 1.78 | 88.0% | 0.869 | 47.4% | 76.3% (40.7%) | 53.8% (64.8%) |
| best baseline on the same rows, column by column | 483 | 32.2% | 60.3% | 1.91 | 84.3% | 0.681 | 35.2% | 58.2% (41.4%) | 37.9% (69.0%) |
| 60 train games (16 of them version 1), 53,526 iterations | 1,387 | **46.1%** | 79.7% | 1.75 | 83.8% | 0.827 | 49.1% | 72.3% (41.3%) | 45.6% (52.9%) |
| best baseline on the same rows (weights fitted on these games) | 1,387 | 31.7% | 62.1% | 1.87 | 83.9% | 0.718 | 37.8% | 62.5% (48.2%) | 36.1% (62.3%) |

Both runs together: top-1 47.9% on 1,870 turns against about 32% for the best baseline. The
search is well ahead on top-1, top-3 and on ranking who switches (AUC 0.83 to 0.87 against 0.68
to 0.72). It is NOT ahead on naming the Pokemon that comes in when that Pokemon has been shown
(41% against 41% to 48%: the uniform guess), nor on switch-or-stay accuracy in the larger run
(83.8% against 83.9% for "never switches"), nor on the turns where the opponent uses a move it
has already shown (64.8% and 52.9% against 69.0% and 62.3% for "its strongest" or "its most used
revealed move": the search's lead is all on moves not yet revealed), and its log loss gains are small because it puts
no mass at all on some actions that were then taken (25 of the test rows, 24 of them moves that
were an option in none of its sampled worlds). The 24-game number is 7 points above the 60-game one: quote the
pooled 48%, not 53%.

The function for the next agent:

```python
from bench.opp_replay import load, messages_up_to, events_up_to, build_stream, games_index
game = load("g...")                 # a hash from runs/opp_pred/games.jsonl, or a path to a battle-*.json
msgs = messages_up_to(game, 7)      # list[str]: every message up to and including the request for turn 7
evts = events_up_to(game, 7)        # the same, with {"type": "chose", "choice": ...} in their places
stream = build_stream(game)         # {"game", "room", "user": "replayme", "side", "exact_active", "events"}
# feed: {"op":"feed","key":"replayme|<room>","user":"replayme","room":<room>,"data":msg}
# and {"op":"chose","key":...,"choice":...} for every chose event, in order, then {"op":"decide",...,"rqid":evts[-1]["rqid"]}
```

`replay_game(helper, game, ms, iterations)` in the same file does exactly that for a whole game,
and `python bench/opp_replay.py replay --split test --games 24 --ms 50 --helpers 3 [--iterations N]`
runs it from WSL with Foul Play's python (`--allow-v1` adds the version-1 games; `check` builds
every stream without a helper). Games that can be replayed: `aligned` in `runs/opp_pred/games.jsonl`.

## Not done

- **Not checked for exactness.** The rebuilt requests were never compared with real ones (there
  are none for these games). The proof is that Foul Play's own code followed 84 games to the end
  and answered every request. A wrong item or a missing disabled flag would not raise.
- Version-1 requests are approximations (see the table). Use `has_req` in `games.jsonl` to stay
  on the 255 exact-as-recorded games.
- The search names the opponent's active Pokemon as our rows do in 520 of 527 test requests. The
  7 others are forme names; a switch target with a forme name may be scored as a miss.
- 58 games have no stream. Their rows are in the dataset all the same.
- No predictor for Tera, for forced switches or for pivot targets is scored. The rows are there
  (`label.tera`, kinds `forced` and `pivot`).
- Nothing was tuned on the test split. The baselines' weights come from train. The search has no
  fitted part, so its train and test numbers are both out of sample, but the baselines beside it
  in the 60-game row are in sample.
- Only 50 ms was run (47,000 to 54,000 iterations per decision on this shared laptop). How the
  opponent prediction changes with iterations was not measured.
- These opponents are the ones an older, weaker bot met (the rating of each is in the row as
  `opp_rating`). Whether people at 1900 switch and Tera at the same rates is not known from this.
