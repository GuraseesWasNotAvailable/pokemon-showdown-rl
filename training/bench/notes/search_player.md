# The search player: what is there, what was measured, what is not done

2026-10-06, branch `search`. Files: `search_player.py` (new), `ladder.py`, `ladder_graph.py`,
`tests/test_search_player.py` (new), `bench/search_run_report.py` (new). An earlier attempt at this
part was cut off by an outage and left `search_player.py` and the `ladder.py` changes uncommitted,
with no tests, no notes and no graph. Its design was kept (it is sound); what review and real
games changed is listed under "Changed after review".

Everything below ran on the laptop that morning with other agents' jobs on it (4 Foul Play
helpers of another worktree, test runs, my own test commands). Ports used: 8640 (a server the
earlier attempt had left running, gone by 09:07) and 8641 (mine). No public server, no account.

## What plays

```
python ladder.py                                   the network alone, as before (nothing is started)
python ladder.py --player mix                      the network and the search combined (rule: --mix-rule, default veto)
python ladder.py --player search                   the search's largest share
python ladder.py --arms net,mix --games 60         every new game gets one of the listed modes, balanced
        --search-ms 500 --helpers 4 --deadline 2.5 [--mix-rule veto|pool|shortlist] [--out FILE] [--no-save-battles]
python bench/search_run_report.py RESULTS.jsonl    per arm: record, fallbacks, who was overruled, Tera, iterations, time
```

Modes: `net`, `search`, `mix`, and `mix-veto`, `mix-pool`, `mix-shortlist` (the rule in the name).

`search_player.search_player()` is `players.batched_policy_player` with `foulplay_bridge.FoulPlayMixin`
in front. Every decision the network answers exactly as today (same observation, same batched
forward pass, same history). For a battle whose mode is not `net`, the same request goes to the
search first (it thinks while the network answers), and the answer is waited for until
`deadline - 0.05 s` after the request.

### The combination (`search_player.mix`, pure, settings in `MixSettings`)

| Rule | What it does | Why |
|---|---|---|
| `veto` (default) | Every legal action with at least `keep` (0.25) times the largest share of the search is "sound". The network's likeliest sound action is played | The search sees tactics a few turns deep but scores with a material count, so it only removes what it found to lose something. The network's long view picks among the rest |
| `pool` | Largest `net_weight * log max(p, 1e-3) + search_weight * log max(share, 1e-2)` | The plain weighted sum of log probabilities, with a floor under each |
| `shortlist` | Of the network's likeliest moves (at least 0.10 of its top probability, 3 at most) the one with the largest share of the search | The reverse of veto: the network proposes, the search disposes |

Terastallization in all three: the network's probability of a move is that of its plain and Tera
forms together (it cannot see its Tera type, bench/notes/tera.md), and when both forms are among
the candidates the search decides: with Tera when the Tera form has the larger share and at least
`tera_share` (0.20). In `search` mode Tera actions are actions like any other.
Every number in `MixSettings` is a first guess. None was tuned.

An action the search has no visit on (it believes we are trapped, say) is not vetoed: no opinion.

### It never costs a game

Every decision of a searched battle ends in one of these, and each is counted per game and per run:

| Reason | When | Played |
|---|---|---|
| (answered) | the search answered in time with at least two of our legal actions | the mode's choice |
| `single` | one legal action (a locked move, a trapped Pokemon with one move, one fainted Pokemon to revive) | it. The search is not asked |
| `revive` | the request asks whom Revival Blessing revives | the network's choice. Not asked |
| `timeout` | no answer by the deadline (an answer that is there when the timer fires is still used) | the network's move |
| `expired` | the helper did not start a search that could not have ended by the deadline (it was queued behind another) | the network's move, at once |
| `error` | bridge closed or down, helper failed, Foul Play could not follow the battle, anything unexpected | the network's move |
| `illegal` | nothing the search looked at is a legal action of ours, or its favourite is not one | the network's move |
| `narrow` | the search saw ONE action where the request offers several | the network's move |
| `weak` | fewer than `min_iterations` (10,000) iterations | the network's move |
| `off` | the player was given no bridge | the network's move |

`single` and `revive` are not failures and are not in a game's `fallbacks` count.
`foulplay_note()` is called with every order: by `choose_move` for its own, and at the client's
`send_message` for the ones that do not come from it (the random order `players.ChoiceAuditMixin`
forces after three rejections, poke-env's own `/choose default`). A request that arrives after the
battle has ended sends nothing. A decision that raises plays a random legal order, counted.

### What is written

- `results.jsonl`, per game with an arm: `arm`, `decisions`, `asked`, `searched`, `fallbacks`,
  `fallback_reasons`, `disagreed`, `overruled` (the network's move was not played),
  `search_overruled`, `tera_added`, `tera_dropped`, `iterations_per_decision`, `search_ms_mean`,
  `decision_ms_mean`, `decision_ms_max`, `late` (decisions over the deadline), `tera`, `tera_turn`,
  `tera_type`, `tera_by` (net / search / mix), `tera_net`, `tera_search` (did each side want Tera on
  that decision), `role`, `search_ms`, `mix_rule`. A game without an arm has none of these.
- `battles/<battle tag>.jsonl` next to the results file (default `runs/ladder/battles/`), one JSON
  object per line: `{"t", "dir": "in"|"out"|"note", "data"}`. Battle rooms only, so nothing of the
  login. `note` lines carry the arm, every decision's record and the result line. Written by a thread
  of its own, not by the event loop. ON by default for every player, the plain network included.
- A decision's record: `battle`, `turn`, `rqid`, `mode`, `legal`, `mask`, `forced`, `net`, `search`,
  `played`, `source`, `asked`, `reason`, `ms`, `net_ms`, `wait_over_ms`, `p_net`, `p_played`,
  `probs[14]`, and when the search answered `iterations`, `worlds`, `search_ms`, `shares[14]`,
  `share_net`, `share_search`, `share_played`, `score_*`, `disagree`, `net_overruled`,
  `search_overruled`, `tera_added`, `tera_dropped`, `opponent_active`, `opponent` (its six
  likeliest replies as `[name, share, seen]`).
- The graph (`ladder_graph.py`): with arms, a table "By arm" (games, record, the level
  `performance()` gives for the arm's own games, decisions answered by the search, iterations,
  fallbacks, network overruled, Tera games and median turn), the dots in the arm's colour (filled
  for a win, hollow for a loss) and a legend. A page whose games have no arm is byte for byte what
  it was (tested against the committed version).

## Real games

Local server, `gen9randombattle`, both sides `learner_update_001475.pt`. The opponent is
`ladder.py --player net` (the network alone) under another account; the two find each other on
the local ladder. Search 500 ms, deadline 2.5 s, rule veto, no pause before moves. Nothing was
tuned on these games and nothing was changed in the rule or its settings because of them.

### Main run: 124 games, 6 at a time on 6 helpers (08:26 to 08:48, 21 minutes)

`runs/search_player/main/` (git-ignored): `ours/results.jsonl`, `ours/battles/` (124 files), `report.json`.
The code was the one of 08:26: before the battle files moved to their own thread and before
"an answer that is there when the timer fires is used" (see below).

| | mix (veto) | search |
|---|---|---|
| Games | 62 | 62 |
| Record against the network alone | 43-19 (69%) | 36-26 (58%) |
| as p1 / as p2 | 22-4 / 21-15 | 17-13 / 19-13 |
| Games not finished, lost for inactivity, forfeited | 0 | 0 |
| Orders the server refused as invalid; decisions that raised; random orders forced | 0; 0; 0 | 0; 0; 0 |
| Decisions | 2,041 | 2,232 |
| Not asked: one legal action | 51 | 40 |
| Asked | 1,990 | 2,192 |
| Answered and used | 1,822 (91.6% of asked) | 2,018 (92.1%) |
| Fallbacks to the network's move | 168: timeout 150, expired 12, narrow 6 | 174: timeout 155, expired 19 |
| Games without a single fallback | 30 | 30 |
| The two wanted different moves | 828 (45% of answered) | 979 (49%) |
| Network overruled | 302 (16.6%) | 979 (48.5%) |
| Search overruled | 577 (31.7%) | 0 |
| Tera added to the network's move / taken off it (decisions) | 30 / 32 | 37 / 29 |
| Games with Tera | 55 (89%) | 58 (94%) |
| Median Tera turn | 13 | 17 |
| The Tera order came from | the mix 49, a fallback decision 6 | the search 52, a fallback decision 6 |
| The network wanted Tera on that decision too | 25 of 55 | 21 of 58 |
| Won, of the games with Tera | 41 of 55 | 32 of 58 |
| Iterations per answered decision: mean / median / lowest tenth | 214,411 / 100,000 / 48,000 | 220,951 / 103,000 / 48,000 |
| Answered decisions under 60,000 iterations | 355 (19%) | 385 (19%) |
| Search, asked to answered: mean / p95 / max | 1,229 / 1,571 / 2,405 ms | 1,235 / 1,617 / 2,378 ms |
| Request to order: median / mean / p95 / p99 / max | 1,217 / 1,413 / 2,469 / 4,739 / 16,718 ms | 1,219 / 1,433 / 2,461 / 4,960 / 17,063 ms |
| The network's part of it: median / p99 / max | 106 / 929 / 4,375 ms | 104 / 1,047 / 4,396 ms |
| Decisions over the 2.5 s deadline | 85 (4.2%) | 95 (4.3%) |

Helpers over the run: 3,924 decisions answered, 94 not started for lack of time, **6 helper exits**
(each replaced and its battles rebuilt by the bridge; why they left was not looked into).
Server errors seen: 3 times `[Unavailable choice] Can't switch: the active Pokemon is trapped`
(normal: a trapping ability is only learnt by trying), nothing else.

The records are 62 games each, on a machine where 8% of the asked decisions fell back, against
the same network on a local ladder. 95% intervals: mix 57% to 79%, search 46% to 70%. They show
that it runs, and they are not the measurement.

### Run 2: 30 games in the ladder bot's shape, 3 at a time on 4 helpers (09:09 to 09:16)

`runs/search_player/run2/`. The code with everything but the last fix (the answer that is there
when the timer fires). Nothing heavy was run next to it.

| | mix (veto) | search |
|---|---|---|
| Games, record | 15, 10-5 | 15, 6-9 |
| as p1 / as p2 | 7-2 / 3-3 | 5-2 / 1-7 |
| Decisions, asked, answered and used | 493, 482, 482 | 555, 538, 536 |
| Fallbacks | 0 | 2 (narrow) |
| The two wanted different moves | 216 (45%) | 265 (49%) |
| Network overruled / search overruled | 84 (17.4%) / 141 (29.3%) | 265 / 0 |
| Games with Tera, median turn | 14, turn 10 | 14, turn 19 |
| Iterations per answered decision: mean / median / lowest tenth | 451,323 / 184,000 / 100,000 | 256,822 / 144,000 / 67,000 |
| Answered decisions under 60,000 iterations | 13 (2.7%) | 25 (4.7%) |
| Search, asked to answered: mean / p95 / max | 1,164 / 1,262 / 1,447 ms | 1,192 / 1,319 / 1,556 ms |
| Request to order: median / p95 / p99 / max | 1,167 / 1,282 / 1,363 / 1,527 ms | 1,187 / 1,325 / 1,402 / 1,581 ms |
| The network's part: median / p99 / max | 65 / 176 / 251 ms | 66 / 173 / 394 ms |
| Decisions over the 2.5 s deadline | 0 | 0 |

0 timeouts, 0 expired, 0 helper exits, 0 invalid orders, 0 exceptions. The event loop was at most
245 ms late in 1,631 checks. So with the machine left alone the deadline holds with a second to
spare, and the lateness of the main run is the machine's, not the decision's.

### Run 3: queueing on purpose, 18 games, 6 at a time on 2 helpers, arms net, mix, search (09:18 to 09:22)

`runs/search_player/run3/`. The final code. Up to four searched battles share two helpers, so a
decision waits behind another battle's search.

| | |
|---|---|
| Games | 18 (6 per arm), all finished. net 5-1, mix 5-1, search 2-4 |
| Asked decisions | 407 |
| Answered and used | 264 (65%): request to order 2,236 ms median, 2,621 ms at the slowest |
| Not started by the helper (`expired`) | 141: the network's move after 2,095 ms median, 2,442 ms at the slowest |
| `timeout` | 2 (2,450 and 2,452 ms) |
| Over the 2.5 s deadline | 2 of 612 decisions, by 115 and 121 ms. Both are answers that were in hand when a timer fired 165 and 171 ms late |
| `net` arm decisions | 197, 84 ms median, 423 ms at the slowest |
| Event loop | at most 273 ms late in 673 checks |
| Invalid orders, exceptions, forced random orders, inactivity losses, forfeits, helper exits | 0 of each |

So under queueing nobody waits past the deadline for a search, a third of the decisions are the
network's, and a decision can end later than the deadline by as much as the event loop is late
(about a quarter of a second with six games in one process). The 0.05 s kept back (`RESERVE_S`)
does not cover that. It is not raised, because at 500 ms a second-in-line search ends 2.3 s after
the request and 0.1 s less would turn all of those into `expired`. Whoever needs "never over
2.5 s" sets `--deadline 2.3`. One helper per game at a time (run 2) has no such decisions at all.
The page with three arms: `runs/search_player/run3/ours/elo.html`.

All runs together (mix 83 games, search 83, net arm 6): 172 games started, 172 finished, no
game lost for inactivity, no forfeit, no invalid order, no exception.
mix 58-25, search 44-39 against the network alone, in three runs under three loads: not a measurement.

### What the main run showed that was wrong, and what was done

1. **The first decision took 4.4 s** (the network's first forward pass of the process). Fixed: one
   forward pass at construction, at one row and at the number of games played at once. First
   decision after the fix: 1.17 s. It still took 4.4 s once, when six games started together
   with the opponent process starting on the same GPU: that is why the warm-up now also runs a
   batch of the playing size, and why `net_ms` is in every record.
2. **The search "chose" Solar Beam four turns running with four moves on offer**: Foul Play's state
   had us locked into the move (sun, no charge turn), the engine searched one option, 1,000
   iterations per world in 12 ms. Now a fallback (`narrow`). 6 such decisions in the main run.
3. **Decisions over the deadline come in bursts, and the cause is outside the decision code.**
   180 of 4,273 decisions (4.2%) took longer than 2.5 s, up to 17.1 s. 173 of them are `timeout`
   fallbacks whose timer fired late: four to six battles end their wait in the same instant, 3 to
   17 s after the request, so the event loop itself was not run. 13 had the network's own
   forward pass take over a second. All 180 fall in 14 bursts. The three largest later ones
   coincide with heavy commands I ran next to the games (a torch import and a CPU forward pass,
   08:31:30 to 08:33:50; a system query, 08:36 to 08:38; the first pytest run, 08:43:50 to
   08:48:30). The bursts of the first four minutes (37 decisions) coincide with nothing of mine;
   another worktree started 4 Foul Play helpers about then. Not established: why the loop is
   held for as long as 17 s instead of merely slowed. No timeout inside the process can fire
   while its loop is not run, so this cannot be fixed in the decision code. What was done:
   `wait_over_ms` in every record and an event-loop watch in the run's summary say how late the
   loop was, and see 4.
   **78 answers were thrown away.** The helpers answered 3,924 decisions and the player used
   3,846 (3,840 and the 6 narrow ones). A loop that is held up finds the timer and the answer
   due together, and the timer wins, so 78 of the 305 "timeouts" had an answer that had arrived
   in time. Fixed after this run: an answer that is there when the timer fires is used (tested by
   holding the loop up on purpose).
4. **The battle files were written by the event loop**, one flush per message. Moved to a thread.
   Whether that was part of the freezes is not known; it is the only disk write per message.
5. **Without `foulplay_note` the helper cannot follow the battle at all**: replaying a saved battle
   without notes stopped Foul Play's loop on message 20 (`KeyError: ''`). So the note is not only
   about what the search infers; hence the note at `send_message` for orders `choose_move` did not make.

## Changed after review (against the earlier attempt's uncommitted files)

- `read_advice`: the search's favourite must be a legal action of ours (`illegal` otherwise, where
  before the best legal one was taken), and `narrow` (above).
- A request after the end of a battle returns an order with an empty message. Before, it
  returned `/choose default`, which the server answers with another error.
- Orders not made by `choose_move` are noted when they are sent.
- An answer of the search that is already there when the wait's timer fires is used (the event
  loop was late, not the search). Before, it was thrown away and counted as a timeout.
- Warm-up, `net_ms`, `wait_over_ms`, `mask`, the event-loop watch (`loop_lag`, in the run's summary).
- `tera_added`, `tera_dropped`, `tera_net`, `tera_search`.
- `BattleRecorder` writes on its own thread. `arm_fields()` is a function (tested).
- The end of a run prints games ended for inactivity or by a forfeit.
- New: all of `ladder_graph.py`'s arms, `tests/test_search_player.py`, `bench/search_run_report.py`.

## Tests

`python -m pytest tests/test_search_player.py -q -p no:cacheprovider`: 56 tests, 56 passed in 40 s (Windows, the laptop still busy). No Foul Play, no
server, no checkpoint needed.
- The combination: each rule on hand-made numbers, floors, Tera in each rule, 3,000 random
  positions per rule never give an illegal action.
- `read_advice`: every reason. `ArmPicker`: balance, the same battle keeps its arm, a restart
  continues from the games already recorded.
- The player on the seven recorded battles (real poke-env objects, the real helper's recorded
  answers through a fake bridge, a stub network): `net` plays move for move what
  `batched_policy_player` plays and tells the search nothing; `search` plays the largest share;
  the four mix modes play legal actions and count who was overruled; a search that never answers
  costs the deadline and no more; nine kinds of failure each play the network's move and are
  counted; three battles on one helper keep the deadline; arms per battle; Tera turn in the
  summary; a decision that raises; an order forced past `choose_move` is noted; nothing is sent
  after the end; warm-up leaves no trace.
- `ladder.py`: arguments, arm counts, the result line's arm fields, the recorder (battle rooms
  only, never raises, does not overwrite). `ladder_graph.py`: no arms = the old page exactly,
  arms table, dot colours, mixed pages, more arms than colours.
- Not covered by a test: `ladder.py`'s `drive()` loop itself (it is what the real games ran).

## For whoever measures next

- Let each arm meet the same opponent: `--arms net,mix` (or `net,search,mix`) in ONE process. The
  `net` arm is the search player in `net` mode, tested equal to the plain player.
- The local ladder only pairs two accounts at once when their local ratings are within 50
  points; otherwise it waits for the server's matcher, which runs once a minute. Two accounts
  alone therefore start about one game a minute once their ratings drift apart. The main run
  needed 21 minutes for 124 games at 6 at a time.
- Who is p1 is the server's choice, and it was not even: mix played 26 games as p1 and 36 as p2
  (22-4 and 21-15). Check the side before reading a record.
- Watch `iterations_per_decision` (per game in the results) and the median, not the mean: endgames
  with few options run to millions of iterations and pull the mean up (mean 214,000, median
  100,000 in the main run). A fifth of the searched decisions had under 60,000.
- WSL: twice a cold start failed (`Wsl/Service/CreateInstance/CreateVm/0x800705b4`, a timeout)
  right after the previous helpers had left. `ladder.py` then stops before logging in ("the
  search cannot be used"). Keep something running inside WSL between runs.
- Do not run anything heavy next to a measurement. A torch import and one CPU forward pass in
  another process were enough to freeze the ladder process's event loop for seconds at a time.
- `python bench/search_run_report.py RESULTS.jsonl --json out.json` gives every number in the
  tables here. The per-decision records are in `battles/`.

## Not done, not measured

- **Strength.** Nothing here measures it. The records above are from runs made to show that the
  player runs; the main one on an overloaded machine, with 8% of the asked decisions played by
  the network. No comparison with a `net` arm of real size (6 games only). Not against Foul Play,
  SimpleHeuristics or people.
- **The rule and its numbers are untried guesses**: `keep` 0.25, `tera_share` 0.20, the floors of
  `pool`, `top` 3 and `ratio` 0.10 of `shortlist`, `min_iterations` 10,000. Only `veto` was played
  in real games. `pool` and `shortlist` have unit tests and replayed battles, no real game.
- **p1 and p2 are not balanced** by the ladder and differ a lot in these games (mix 29-6 as p1,
  24-18 as p2 over the main run and run 2; search 22-15 and 20-20). The Tera scout saw the same
  network win 55.3% as p1 against itself. Not followed up.
- **The 17 s freezes are explained only by coincidence in time** with heavy processes on the same
  machine. Why the loop is held instead of slowed was not found (candidates not tested: the disk
  writes of the old recorder, the bridge's lock while a helper is replaced, paging). Run 2 and
  run 3 had none, but they were also the runs with nothing heavy beside them.
- **6 helper exits in the main run**: counted by the bridge, cause not looked into. Each costs its
  battles some timeouts while the helper is replaced.
- **Tera quality.** Only how often, when, and who decided. The scout's checks ("worse defence
  and no boost", Tera when weak) were not run on the search player; the per-decision records and
  battle files to do it with are there.
- **Opponent prediction is recorded, not used.** Each record holds the search's six likeliest
  opponent replies. The combination uses only our side's shares. No explicit double-switch logic.
  The payoff table (`payoff_ms`) is not asked for.
- **A switch the search never looked at is not vetoed.** If Foul Play believes we are trapped and
  we are not, the network may switch without the search's view. Not counted separately.
- `net_overruled` counts the move, not the Tera form. With the merged prior a move whose plain
  and Tera forms together outweigh the network's single likeliest action counts as "the
  network's move", so a few "overruled" are the merge and not the search. Not counted apart.
- `mix` plays Tera on the Tera form's share alone when the plain form was vetoed, also below
  `tera_share`. Intended (the search turned away from the plain move), not measured.
- **`--save-battles` is on by default**, also for the plain network: a file per battle in
  `runs/ladder/battles/`, about 150 to 300 kB each. Nothing prunes them.
- The warm-up is in the search player only. The plain player (`--player net` without arms) is
  untouched and still takes about 4 s for its first decision.
- `drive()` in `ladder.py` has no test of its own. Not run on Linux. Not run on the public ladder.
- The local server on port 8640 that the earlier attempt had started was gone at 09:07 (I did not
  stop it). I started one on 8641 and stopped it by its PID at the end.
- About 100 minutes for this second attempt, against the 90 asked.
