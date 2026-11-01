# The measurement: net, mix, predict

2026-10-06, branch `search`. New: `bench/gauntlet.py` (plays and reports), `bench/gauntlet_tera.py`
(what each arm did with Tera, from the saved battles). No player code was changed.
Port 8680 only (one local server). No public server, no account. Nothing in `ps-ppo` or
`ps-ppo-play` was touched.

**Stalls and errors first.** No player stalled or failed: 972 reported games, 33,701 decisions,
0 invalid orders, 0 exceptions, 0 forced random orders, 0 games lost for inactivity, 0 forfeits,
3 fallbacks to the network's move in 23,026 searched decisions. One interruption that was not a
player's: **the laptop ran out of battery and hibernated at 12:25:41** (System log, Kernel-Power 42,
"Sleep Reason: Battery"; back at 14:40:00) in the middle of the final run against the network.
The six games in flight (5 predict, 1 mix) were abandoned before our side recorded them (they are
in the battle files and in the opponent's opp.jsonl after all: mix won its 1, predict won 2 and lost
3; counted, mix is 102/166 = 61.4% and predict 93/167 = 55.7%, see the review below), and that run
ended at 165 + 162 games where 200 + 200 were planned. One decision was over the 2.5 s deadline,
by 16 ms, while I ran a script of my own beside the games.

Network: `learner_update_002575.pt` on both sides (ours and the "net" opponent). The checkpoint
that is on the ladder (u1475) was not measured here.

## The result

| Opponent | Arm | Games | Record | Won (95% Wilson) | as p1 | as p2 |
|---|---|---|---|---|---|---|
| The network alone | net (control) | 145 | 79-66 | 54.5% (46.4 to 62.4) | 38/73 | 41/72 |
| The network alone | **mix** (veto) | 165 | 101-64 | **61.2% (53.6 to 68.3)** | 46/83 | 55/82 |
| The network alone | predict | 162 | 91-71 | 56.2% (48.5 to 63.6) | 42/81 | 49/81 |
| Foul Play | net | 130 | 58-72 | 44.6% (36.3 to 53.2) | 28/65 | 30/65 |
| Foul Play | **mix** (veto) | 110 | 75-35 | **68.2% (59.0 to 76.1)** | 41/55 | 34/55 |
| Foul Play | predict | 110 | 54-56 | 49.1% (39.9 to 58.3) | 28/55 | 26/55 |

| Difference | Points | 95% interval | z |
|---|---|---|---|
| mix minus net, against Foul Play | +23.6 | +11.0 to +35.0 | 3.8 |
| mix against the network alone, minus an even 50% | +11.2 | +3.6 to +18.3 | 2.9 |
| mix minus predict, against the network alone | +5.0 | -5.6 to +15.5 | 0.9 |
| mix minus predict, against Foul Play | +19.1 | +6.1 to +31.2 | 2.9 |
| mix minus predict, both opponents (weighted by games) | +10.7 | +2.4 to +18.9 | 2.5 |
| predict minus net, against Foul Play | +4.5 | -8.1 to +16.9 | 0.7 |

1. **Look-ahead helps.** `mix` won 61.2% against the same network playing alone and 68.2%
   against Foul Play held to 150 ms (about a third of our search's iterations: see the review below;
   with a later run pooled, about 66%),
   where the network alone won 44.6%: 23.6 points more (pooled, +20, +10 to +30), and both are
   outside the noise (z = 2.9 and 3.8). In Elo that is about +79 against the network alone
   (95% +25 to +133), against an even 50%; over the same session's own net-vs-net control (54.5%)
   the +6.7 points are inside the noise (-4.2 to +17.5, p 0.25).
2. **Prediction on top of it is not shown to help, and may cost a few points.** `predict` was below `mix`
   against both opponents (56.2% against 61.2%, and 49.1% against 68.2%). Against Foul Play it
   was no better than the network alone. A later run did not repeat this (predict 69.2%, mix 61.5%
   against Foul Play): pooled, mix minus predict is +9.1 points (-1.1 to +19.1, p 0.10).
3. **Inside or outside the noise.** `mix` over `net` is outside it. `predict` under `mix` is
   outside it against Foul Play (z = 2.9) and with both opponents together (10.7 points, z = 2.5)
   in this run, but not once the later run is pooled in (+9.1, p 0.10; all three: +7.1, p 0.06),
   and inside it against the network alone (z = 0.9): to settle that one alone takes about
   1,500 games per arm if the gap is the 5 points seen, about 380 per arm if it is the 10.7
   points of both opponents together.

`mix` (rule veto) is the mode for the ladder. `predict` as built should not go on it.

### Read with the review of the numbers (2026-10-06, recomputed from the raw records)

Every count and interval above was reproduced from the server's own messages. What changes is
how far they reach:

- **Foul Play here is not an equal-compute opponent.** It ran at 150 ms on 2 helpers: median
  85-86k iterations per decision in the mix and predict games (862 of 7,576 decisions under its
  own 60k target), where our mix search had a median of 253k, about 3.0 times as many per game.
  Quote these results as "against Foul Play at 150 ms". The live evaluator's Foul Play (200 ms,
  8 at a time) is a different benchmark, and the 44.6% for the network alone here is not the
  live 50-56%.
- **mix against Foul Play: about 66%, not 68%.** A later run with the same checkpoint and settings
  (cut short at 65 games per arm) had mix at 61.5%. Both pooled: 115/175 = 65.7% (58.4 to 72.3).
  The 68.2% leans on one 11-minute stretch (09:47-09:58 UTC, 25 of 27 won; post hoc, chi-square 11.7
  on 3 df); without it 50/83 = 60.2%. mix minus net, pooled: +19.9 points (+9.6 to +29.6, Fisher
  p 0.0002). That gain is the solid part.
- **predict: not shown to help, may cost a few points.** The later run had predict at 45/65 =
  69.2% against mix at 61.5%, so the two runs disagree (z = 2.54). Pooled: mix minus predict
  +9.1 points (-1.1 to +19.1, p 0.10); over all three opponents' runs +7.1 (-0.2 to +14.4, p 0.06).
  About 380 games per arm would settle a 7-10 point gap.
- **mix against the network: 61% (54 to 68) head to head**, against an even 50% (p 0.005). Over
  the session's own net-vs-net control (54.5%) it is +6.7 (-4.2 to +17.5, p 0.25): not separable.
- The opponent network's checkpoint is not written in its own run files (only in the launch
  commands); gauntlet.py opp should record the path and its sha256.

## What was played

Local server, `gen9randombattle`, challenges (not the ladder): see "How" below. Search 500 ms,
1 thread, deadline 2.5 s, bridge timeout 4 s, rule veto, `predict` with its defaults
(confidence 0.60, gain 0.08, risk 0.08, payoff table 400 ms "full", once bitten on), no pause
before moves, battle timer on, as `ladder.py` sets them.

| Phase | When (local) | Ours | Opponent | Games at once / helpers | Games |
|---|---|---|---|---|---|
| Selection | 11:26 to 11:44 (17.2 min) | mix-veto, mix-pool, mix-shortlist | network alone | 6 / 6 | 150 |
| Final | 11:45 to 12:25 (41 min, cut by the hibernation) | mix, predict | network alone | 6 / 6 | 327 (+6 abandoned) |
| Control | 11:25 to 12:25, and 14:40 to 14:47 | net | network alone | 1 / none | 145 (129 + 16) |
| Foul Play | 14:57 to 15:41 (44 min) | mix, predict | Foul Play, 150 ms | 4 / 4 ours + 2 Foul Play's | 220 |
| Foul Play, net | 15:02 to 15:44 (42 min) | net | Foul Play, 150 ms (the same 2 helpers) | 1 / none | 130 |

102 minutes of games. Fewer games than asked for in every part (50 per rule where 100 were asked,
165 and 162 where 300 were asked, 110 to 130 where 200 were asked): the six helpers allow about
8 searched games a minute, and the full list is about 220 minutes of them.

The laptop was on battery during the selection and the final run (which is why it went down)
and on the mains during the Foul Play runs. Other agents' jobs were on it throughout (a soak
test, pytest and mutation runs; 62% processor load before I started anything).

### Selection (not the result)

| Rule | Games | Record | Won (95%) | as p1 | as p2 | Network overruled | Search overruled |
|---|---|---|---|---|---|---|---|
| veto | 50 | 31-19 | 62.0% (48.2 to 74.1) | 15/25 | 16/25 | 19.7% | 34.0% |
| pool | 50 | 30-20 | 60.0% (46.2 to 72.4) | 15/25 | 15/25 | 12.8% | 37.7% |
| shortlist | 50 | 23-27 | 46.0% (33.0 to 59.6) | 7/25 | 16/25 | 20.6% | 33.7% |

Veto was kept: most wins, and the default. **Fifty games each do not separate the rules**: veto
and pool are one win apart, and veto minus shortlist is 16 points with an interval of -3.4 to
+33.8 (z = 1.6). All three ran clean (0 fallbacks, 0 late decisions in 5,075 searched decisions).
`predict` in the final run is built on the kept rule. The selection games are not in the result.

## Every arm

"Network overruled" is the search changing the network's move; "search overruled" is the
network keeping a move that was not the search's favourite. Both as a share of the decisions
the search answered. Time is from the request to the order, over all decisions of the arm.

| | net vs net | mix vs net | predict vs net | net vs FP | mix vs FP | predict vs FP |
|---|---|---|---|---|---|---|
| Games | 145 | 165 | 162 | 130 | 110 | 110 |
| Games with Tera | 105 (72.4%) | 143 (86.7%) | 134 (82.7%) | 95 (73.1%) | 93 (84.5%) | 95 (86.4%) |
| Tera turn: median (mean; quartiles) | 18 (19.5; 10, 25) | 13 (14.8; 7, 20) | 13 (14.9; 7, 20) | 18 (19.7; 11, 25) | 13 (16.2; 7, 21) | 12 (14.1; 6, 19) |
| Decisions | 5,300 | 5,452 | 5,348 | 4,843 | 3,748 | 3,810 |
| Put to the search / answered | none | 5,332 / 5,332 | 5,233 / 5,231 | none | 3,673 / 3,672 | 3,713 / 3,713 |
| Fallbacks to the network's move | | 0 | 2 (timeout 1, narrow 1) | | 1 (narrow) | 0 |
| The two wanted different moves | | 48.9% | 46.6% | | 46.4% | 45.9% |
| Network overruled | | 1,135 (21.3%) | 1,029 (19.7%) | | 819 (22.3%) | 821 (22.1%) |
| Search overruled | | 1,671 (31.3%) | 1,684 (32.2%) | | 1,056 (28.8%) | 1,102 (29.7%) |
| Time per decision: mean / 95th percentile / slowest | 64 / 101 / 2,019 ms | 1,142 / 1,262 / 1,872 ms | 1,462 / 1,662 / 2,516 ms | 64 / 99 / 360 ms | 1,136 / 1,229 / 1,444 ms | 1,424 / 1,623 / 1,874 ms |
| Over the 2.5 s deadline | 0 | 0 | 1 | 0 | 0 | 0 |
| Iterations per decision: mean / median / lowest tenth | | 304,623 / 168,000 / 108,000 | 316,479 / 168,000 / 106,000 | | 455,552 / 253,000 / 186,000 | 469,124 / 258,000 / 190,000 |
| Decisions under 60,000 iterations | | 61 (1.1%) | 52 (1.0%) | | 0 | 0 |
| Game length, turns / seconds | 31.7 / 3.8 | 28.9 / 38.5 | 28.7 / 49.0 | 32.4 / 19.1 | 29.7 / 41.1 | 30.0 / 51.5 |
| Invalid orders, exceptions, forced random orders, inactivity, forfeits | 0 | 0 | 0 | 0 | 0 | 0 |

- The control's 2,019 ms decision is the first one after the laptop came back from hibernation.
- The one late decision (2,516 ms, a `timeout` whose timer fired 66 ms late; game 155196, turn 2,
  11:46) coincides with my trying `bench/gauntlet_tera.py` on the trial battles. The search
  player's notes say not to run anything beside a measurement. I did not do it again.
- The event loop was more than 0.5 s late twice in about 23,500 checks (629 ms at the worst).
- Helpers: 0 expired questions, 0 timeouts in the bridge, 0 exits while games were played. The
  bridge of the final run counts 1 hang and 1 exit: both across the hibernation (the status
  line of 12:25:41 has none).
- `[Unavailable choice]` (a trapped Pokemon learnt by trying): 14 on our side in all runs, normal.

### Foul Play as the opponent

`foulplay_player.FoulPlayBridgePlayer`, `pick="choice"` (the drawn move, as it plays on the
ladder), 150 ms, 2 helpers of its own, timeout 60 s. 12,346 decisions asked, 12,346 answered,
0 fallbacks, 0 errors, 0 expired, 0 helper exits.

| Its games against | Decisions | Iterations per decision: mean / median / lowest tenth | Under 60,000 | Asked to answered: mean / 95th / slowest |
|---|---|---|---|---|
| mix and predict | 7,576 | 141,000 / 85,000 to 86,000 / 57,000 to 58,000 | 862 (11.4%) | 501 to 512 / 879 to 891 / 1,223 ms |
| net | 4,770 | 134,000 / 80,000 / 51,000 to 52,000 | 777 (16.3%) | 507 to 510 / 783 to 801 / 1,242 ms |

So Foul Play searched 2.3 times the 60,000 iterations `bench/FOULPLAY_BRIDGE.md` asks for on
average and 1.3 to 1.4 times at the median, and about one decision in eight was under it. Five
battles shared its two helpers, so its answers waited in line (501 ms on average where the search
itself takes about 350): that delays the game and does not shorten the search. Our own helpers
were not starved by it: 456,000 to 469,000 iterations per decision, none under 60,000.

## Prediction

| | against the network | against Foul Play |
|---|---|---|
| Predictions made / scored | 4,625 / 4,008 | 3,277 / 2,932 |
| The likeliest action was what it did | 2,128 (53.1%) | 1,802 (61.5%) |
| Confident reads (60% or more), scored, right | 1,334, 1,208, 826 (68.4%) | 950, 887, 736 (83.0%) |
| The mix choice was left | 91 decisions (2.0%) in 63 of 162 games | 87 (2.7%) in 52 of 110 games |
| The read was right on those | 57 of 86 scored (66%) | 71 of 84 scored (85%) |
| What was played instead of what | another move for a move 50, a switch for a move 19, a move for a switch 10, Tera taken off 10, other 2 | another move for a move 46, a switch for a move 15, a move for a switch 14, Tera taken off 6, another switch for a switch 5, other 1 |
| Double switches (read "it switches" and we switch) | 2 | 1 |
| Risk test: risk exactly 0 | 77 of 91 | 71 of 87 |
| Our active Pokemon's HP at the change | mean 66%, under 25% on 8 | mean 62%, under 25% on 7 |
| Games without a change / with one / with two or more: won | 72 of 99 / 16 of 43 / 3 of 20 | 37 of 58 / 13 of 31 / 4 of 21 |
| Kept because | unsure 3,291, no gain 1,049, forced 606, coverage 63, risk 54, bitten 48, blind 26, no table 3 | unsure 2,328, no gain 753, forced 436, coverage 43, risk 40, blind 19, bitten 7 |

- The read is good and it does not pay. Against Foul Play 85% of the reads acted on were right,
  and that is the opponent `predict` lost the most against.
- "Games with a change are lost more often" is again what the opponent-model notes saw, and by
  itself it proves nothing (a long or losing game has more changes). What is new here is the
  comparison of the arms, which is random: the games were dealt between `mix` and `predict` by
  the draw, the two differ in these changes and in nothing else I could find (the same search
  and iterations, the same mix choice on every other decision; `predict` also asks the helper
  for the table, after the search), and `predict` came out 10.7 points lower (95% 2.4 to 18.9).
- A likely reason, not tested: the change replaces a choice backed by 300,000 to 470,000
  iterations and the network by the best reply in a 2 ply table whose values are a material
  count. The risk test almost never binds (risk 0 on 148 of 178 changes).
- Three double switches in 272 games. The play the owner asked for is not happening.

## Tera

`python bench/gauntlet_tera.py <run folders>` reads the saved battles. Defence is the scout's
stand-in (bench/notes/tera.md): the worst multiplier of the opposing active Pokemon's own types
against ours, before and after. "Boosted" is the scout's rule: the move played with the Tera is
an attack of the Tera type (Tera Blast is not counted as boosted, as there).

| | net vs net | mix vs net | predict vs net | net vs FP | mix vs FP | predict vs FP |
|---|---|---|---|---|---|---|
| Games with Tera | 72.4% (64.6 to 79.0) | 86.7% (80.6 to 91.0) | 82.7% (76.2 to 87.8) | 73.1% (64.9 to 80.0) | 84.5% (76.6 to 90.1) | 86.4% (78.7 to 91.6) |
| Tera turn, median | 18 | 13 | 13 | 18 | 13 | 12 |
| Tera per turn on which it was offered | 3.2% | 5.2% | 4.8% | 3.1% | 4.6% | 5.4% |
| The network's own best action was Tera on that decision | all | 53 of 143 | 52 of 134 | all | 34 of 93 | 26 of 95 |
| The search's favourite was a Tera action | | 131 of 143 | 123 of 134 | | 82 of 93 | 89 of 95 |
| Decisions where Tera was added to / taken off the network's move | | 90 / 52 | 82 / 71 | | 59 / 31 | 69 / 39 |
| The Tera move was an attack of the Tera type | 28 (27%) | 58 (41%) | 56 (42%) | 26 (27%) | 40 (43%) | 34 (36%) |
| Defence better / same / worse | 57 / 35 / 13 | 76 / 45 / 22 | 76 / 41 / 17 | 44 / 41 / 10 | 50 / 26 / 17 | 53 / 26 / 16 |
| Boost or better defence | 67 (64%) | 110 (77%) | 105 (78%) | 63 (66%) | 71 (76%) | 71 (75%) |
| Worse defence and no boost | 13 (12.4%) | 13 (9.1%) | 8 (6.0%) | 9 (9.5%) | 8 (8.6%) | 12 (12.6%) |
| Weak to its types, Tera removes the weakness: Tera rate | 11.2% (53 of 474) | 16.2% (60 of 370) | 14.2% (62 of 436) | 8.9% (34 of 382) | 15.3% (38 of 249) | 14.5% (41 of 282) |
| Weak to its types, Tera leaves it: Tera rate | 3.6% (11 of 307) | 1.3% (2 of 155) | 3.3% (7 of 211) | 8.5% (12 of 142) | 6.5% (8 of 124) | 4.3% (5 of 117) |
| HP of the Pokemon when it Terastallized, mean | 85% | 76% | 80% | 91% | 78% | 78% |
| Won with Tera / without (not causal) | 48 of 105 / 31 of 40 | 92 of 143 / 9 of 22 | 80 of 134 / 11 of 28 | 39 of 95 / 19 of 35 | 64 of 93 / 11 of 17 | 47 of 95 / 7 of 15 |

What the search did to Tera, net against mix, both opponents together (275 games each):

- **More often**: 85.8% of games (236) against 72.7% (200). z = 3.8.
- **Earlier**: median turn 13 against 18.
- **It is the search's Tera now**: the network's own best action was Tera on 87 of the 236
  decisions (37%); on the other 149 the search added it. The search's favourite was a Tera
  action on 213 (90%).
- **More often for the boost**: the Tera move is an attack of the Tera type on 41.5% (98 of 236)
  against 27.0% (54 of 200). z = 3.2.
- **Better aimed when weak**: Tera on 15.8% of the turns where it removes a weakness (98 of 619)
  and on 3.6% where it leaves it (10 of 279). The network alone: 10.2% (87 of 856) and 5.1%
  (23 of 449). The network alone against Foul Play does not tell the two apart at all (8.9% and 8.5%).
- **Still some Teras that worsen the defence and boost nothing**: 8.9% (21 of 236) against 11.0%
  (22 of 200). Not a difference these counts can show.
- With the network alone a game with Tera is won less often than one without (87 of 200
  against 50 of 75); with `mix` it is the other way round against the network (92 of 143
  against 9 of 22) and level against Foul Play. An association, as in the scout's notes.

Whether a Tera was right is judged by the type chart only, as there. No damage calculation.

## How (bench/gauntlet.py)

`ours` builds the player with `ladder.make_player`, so it is the ladder's player, and replaces
only the way games are found: on a local server the ladder pairs two accounts at once only while
their ratings are close and lets the server choose the side, which came out uneven in the earlier
runs. Here every game is a challenge:

- The opponent process holds two accounts per player of ours. One accepts every challenge (we
  challenge it: we are p1). The other keeps one challenge to us standing (we accept: we are p2).
- Our process starts a game only when it has a free seat and takes the side it has played less.
  The side planned was the side played in all 972 games.
- The arm is drawn when the battle's first message arrives, balanced within the side
  (`RolePicker`): every arm has the same number of games as p1 and as p2, to within one.
- The searched arms and the `net` arm are separate processes of ours against separate accounts
  of the same opponent process, so a `net` game never takes a helper or a seat from a searched one.
- A run stopped by its time limit finishes the games in progress: nothing is cut by length.

The network alone as the opponent is `players.batched_policy_player` (audited), greedy, as on
the ladder. The control is the search player in `net` mode against it.

With the sides balanced the network against itself (u2575) shows no advantage for p1: 38 of 73
as p1, 41 of 72 as p2.

## Found on the way

- **A name that is part of the other name switches the search off for that game.** My first
  trial used `smS` against `smSa`: every decision of ours as p1 fell back with `error` ("Foul Play
  took p1 for both sides: the player's name 'smS' must not be part of the opponent's name or the
  other way round", as FOULPLAY_BRIDGE.md says). The player keeps playing the network's move and
  counts it, so nothing breaks, but on the public ladder a game against an account whose name
  contains the bot's name (or the reverse) is played without the search. Not fixed (player code).
- poke-env's own `send_challenges` with `max_concurrent_battles` lets one more battle start on the
  server than there are seats and holds it; the search player gives such a battle its arm and its
  helper while it waits. `gauntlet.py` therefore starts games itself, one at a time.

## Not done, and what to hold against the numbers

- **Fewer games than asked** (see "What was played"). The intervals above are what they buy.
- **Six games in flight at the hibernation are missing** (5 predict, 1 mix). A game in flight at
  a given moment is more often a long one, so this is not a random loss. At most about 1.5
  points on the predict arm against the network.
- **The control is 54.5%, not 50%.** 79 of 145 is inside the noise of an even game (z = 1.1), and
  the two sides are the same network with the sides balanced, so I read it as luck and compare
  `mix` with 50%. Read against the control instead, `mix` is +6.7 points (95% -4.2 to +17.5),
  which is not outside the noise. The comparison against Foul Play does not depend on this:
  there the three arms are all ours and `mix` is 23.6 points over `net`.
- **The selection decided nothing** between veto and pool. Pool was not measured further.
- **One checkpoint.** u2575 on both sides. Nothing here says what the search adds to u1475, the
  one on the ladder (the earlier runs with it were not measurements).
- **Bots only.** The network and Foul Play, on a local server. Not people, not the ladder.
- **Six games on six helpers**, not the three on four recommended for the ladder: the ladder
  shape gets more iterations per decision than the final run did (median 168,000 here).
- **Foul Play through the bridge**, not the real process, with about one decision in eight under
  60,000 iterations.
- **`predict` was measured as built**, with untuned thresholds. This says the mode loses to
  `mix`; it does not say a prediction cannot help. Why it loses was not established.
- The Tera table is the type chart, not a damage calculation, and our own types are the
  species' types.
- `bench/gauntlet.py` and `bench/gauntlet_tera.py` have no unit tests; they were exercised by
  the runs. The trial run (18 games, with the name problem) is in `runs/gauntlet/smoke` and in
  no number here.
- 102 minutes of games; about three hours of work without the 2 h 14 min the laptop was down.

## Commands

```
python = C:\Users\you\ps-ppo\.venv\Scripts\python.exe, from the worktree; start a server first:
python start_servers.py --dir C:\Users\you\showdown-bot\server --ports 8680

# the opponent: two accounts (ACCEPTS:CHALLENGES:OURS) per player of ours
python bench/gauntlet.py opp --port 8680 --kind net --checkpoint CKPT --pairs oppacca:oppchaa:ggsearch,oppaccb:oppchab:ggplain --out DIR/opp.jsonl --stop-file DIR/opp.stop
python bench/gauntlet.py opp --port 8680 --kind foulplay --fp-ms 150 --fp-helpers 2 --pairs fpacca:fpchaa:hhsearch,fpaccb:fpchab:hhplain --out DIR/opp.jsonl --stop-file DIR/opp.stop

# ours
python bench/gauntlet.py ours --port 8680 --user ggsearch --arms mix,predict --mix-rule veto --per-arm 200 --parallel 6 --helpers 6 --checkpoint CKPT --opp-accept oppacca --opp-challenge oppchaa --out DIR/final/results.jsonl --stop-file DIR/final/stop --minutes 52
python bench/gauntlet.py ours --port 8680 --user ggplain --arms net --per-arm 160 --parallel 1 --gap-s 28 --checkpoint CKPT --opp-accept oppaccb --opp-challenge oppchab --out DIR/control/results.jsonl --stop-file DIR/control/stop

python bench/gauntlet.py report DIR [DIR2] --json out.json     # every arm, the differences between arms, the opponent
python bench/gauntlet_tera.py DIR/final DIR/control --json out.json
```

- Names: no name may be part of another one that it plays against.
- Stop ours with its stop file (it finishes the games in progress), then the opponent with its own.
- Keep something running inside WSL while helpers are needed (`wsl.exe -e sleep 21600`).
- Keep the laptop on the mains.
- Per run folder: `results.jsonl` (ladder.py's line per game plus `phase`, `planned_role`, `ending`),
  `decisions.jsonl` (one line per decision: arm, ms, iterations, net, search, played, probs,
  shares, mask, the predict record), `status.jsonl` (every 30 s), `summary.json`, `battles/`.
  Opponent: `opp.jsonl` (per game, with the iterations and time of every Foul Play decision),
  `opp_status.jsonl`, `opp_summary.json`.
- My runs (git-ignored): `runs/gauntlet/vsnet/{select,final,control}`, `runs/gauntlet/vsfp/{search,plain}`,
  `runs/gauntlet/vsnet/report.json`, `runs/gauntlet/vsfp/report.json`, `runs/gauntlet/tera_report.json`.
