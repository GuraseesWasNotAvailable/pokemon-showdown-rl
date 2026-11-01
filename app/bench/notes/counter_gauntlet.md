# Round 3's counter rules against bots: today's mix against mix-counter

**Answer first: no cost measured, and no gain either.** Over 793 games against three bots, mix-counter
(both round 3 rules: "use the counter" and "don't hit the boosted defence") is level with today's mix:
pooled **-1.1 points (-5.0, +2.8)**. The rules never slowed a decision (mean difference within 4 ms,
0 late decisions, the slowest decision 1,571 ms against mix's 1,587 ms). They raised no errors, and
they caused no fallbacks or invalid orders. So the gauntlet rules out a pooled loss of more than about
5 points, and the ladder A/B (`launchers/laptop/Look Play Counter AB.cmd`) can go ahead. Two flaws
turned up in lost games. Both are rare, and neither is fixed here (see section 4).

Box, 2026-10-08, 02:55:57 to 04:32:42 UTC. Owner's network `learner_update_011850.pt` (sha256
edd11dd6...) in both arms. Code: branch counterfix (d573f7e) plus this commit's `bench/gauntlet.py`
changes (section 5).

## 1. Wins per opponent

Search 500 ms, deadline 2.5 s, rule veto, the same player and helpers for both arms, arms balanced
within each side. Wins with Wilson 95% intervals; the difference with a 95% Newcombe interval.

| opponent | games | mix (today) | mix-counter | mix-counter minus mix |
|---|---|---|---|---|
| set-up test bot | 300 | 147-3, 98.0% (94.3-99.3); p1 74/75, p2 73/75 | 143-7, 95.3% (90.7-97.7); p1 69/75, p2 74/75 | -2.7 (-7.5, +1.7) |
| Foul Play, fair compute | 198 | 60-39, 60.6% (50.8-69.7); p1 30/50, p2 30/49 | 68-31, 68.7% (59.0-77.0); p1 30/49, p2 38/50 | +8.1 (-5.2, +21.0) |
| network alone (011850) | 295 | 75-73, 50.7% (42.7-58.6); p1 42/74, p2 33/74 | 79-68, 53.7% (45.7-61.6); p1 43/74, p2 36/73 | +3.1 (-8.2, +14.3) |
| **pooled** (inverse variance) | 793 | | | **-1.1 (-5.0, +2.8)** |
| pooled, opponents weighted equally | | | | +2.8 (-3.1, +8.8) |

- **Inverse-variance pooling leans on the set-up bot** (79% of the weight): near 100% wins its
  difference has the smallest error. The pooling uses Agresti-Caffo standard errors (one win and one
  loss added per arm) so that an opponent both arms always beat cannot take all the weight. With
  plain normal errors, as in round 1, it is -1.3 (-4.9, +2.4).
- **Fair compute held:** Foul Play's iterations were 0.91 of ours at the median and 0.87 at the mean
  (round 1: 0.92 and 0.89). 6,969 Foul Play decisions answered, none without an order, 186 under 60,000
  iterations (mostly late-game positions with few options).
- **Today's mix matches round 1** (95.5%, 59.4% and 45.7% then), so the conditions are comparable.
- **Fewer games than asked for Foul Play and the network.** The aim was 150 per arm per opponent.
  The set-up bot got its 150; Foul Play got 99 per arm in its 44 minutes. The network got 148 and 147,
  because the 95-minute budget ended. Against one opponent, these runs only show differences of 10 to 20 points.

**Set-up sweep losses** (`bench/setup_ladder_scan.py --by-arm`: games lost to an opponent's Pokemon that
set up and then made 2 or more KOs):

| opponent | mix | mix-counter | difference, points of games |
|---|---|---|---|
| set-up test bot | 3 (2%) | 5 (3%) | +1.3 (-2.8, +5.8) |
| Foul Play | 10 (10%) | 13 (13%) | +3.0 (-6.2, +12.3) |
| network alone | 35 (24%) | 24 (16%) | -7.3 (-16.4, +1.8) |
| pooled (inverse variance) | | | +0.4 (-3.0, +3.8); weighted equally -1.0 (-5.5, +3.5) |

These bots do not answer the ladder's question. The rules were built for the sweeps people run on
the ladder, and only the ladder A/B can show whether they cut those.

## 2. How often the rules fired (mix-counter games)

Per game, with the totals in brackets. "Looked" = a set-up position (opponent at +2, +1 right after
a set-up move, a set-up read of 50% or more, or +2 in any stat).

| per game | set-up bot (150) | Foul Play (99) | network alone (147) |
|---|---|---|---|
| set-up positions looked at | 6.80 (1,020) | 5.26 (521) | 5.96 (876) |
| played a counter | 0.43 (65) | 0.14 (14) | 0.16 (23) |
| switched to a counter | 0.31 (46) | 0.13 (13) | 0.11 (16) |
| kept mix's pick: an attack in KO range | 1.63 (245) | 1.31 (130) | 1.23 (181) |
| kept: mix already played or switched to a counter | 0.38 (57) | 0.20 (20) | 0.23 (34) |
| kept: no counter can work | 3.92 (588) | 3.38 (335) | 4.19 (616) |
| attacks into a boosted defence | 1.07 (161) | 0.84 (83) | 0.98 (144) |
| swapped to the other defence | 0.06 (9) | 0.09 (9) | 0.07 (11) |
| games with a changed move | 35 of 150 | 20 of 99 | 33 of 147 |
| decisions changed | 120 (3.3%) | 36 (1.0%) | 50 (1.0%) |
| counters played | Haze 36, phazing 16, Encore 9, sleep 4 | phazing 7, Encore 4, Haze 2, sleep 1 | Encore 8, Haze 7, sleep 6, phazing 2 |

- In all: 206 changed decisions in 88 of 396 games: 102 counters played, 75 switches to a counter
  holder, 29 swaps to the other defence. "Use the counter" does nearly all the work. "Don't hit the
  boosted defence" swapped 29 of 388 attacks into a boosted defence (7%), as the replay of this
  morning's games predicted: there is rarely an attack of the other kind on the board.
- Games with a change were won 31 of 35, 15 of 20 and 15 of 33; games without one were won 112 of
  115, 53 of 79 and 64 of 114. This is a count, not a comparison: a game with a set-up position is a
  harder game.

## 3. Clean: errors, fallbacks, late decisions, decision time

| | set-up bot: mix / mix-counter | Foul Play: mix / mix-counter | network: mix / mix-counter |
|---|---|---|---|
| decisions (asked the search) | 3,290 (3,268) / 3,613 (3,595) | 3,294 (3,226) / 3,548 (3,491) | 5,081 (4,961) / 5,026 (4,916) |
| fallbacks to the network (failures) | 0 / 1 ("narrow": the search saw one action) | 0 / 0 | 0 / 0 |
| late (over the 2.5 s deadline) | 0 / 0 | 0 / 0 | 0 / 0 |
| **decision_ms_max** | **1,587 / 1,566** | **1,531 / 1,519** | **1,573 / 1,571** |
| ms per searched decision: mean | 1,201 / 1,201 | 1,198 / 1,199 | 1,204 / 1,203 |
| p95 / p99 | 1,258, 1,290 / 1,253, 1,286 | 1,255, 1,285 / 1,255, 1,281 | 1,259, 1,289 / 1,260, 1,292 |
| mix-counter minus mix, mean ms | -0.8 (-3.3, +1.7) | +1.3 (-1.3, +3.9) | -1.2 (-3.3, +0.9) |

- The mix-counter player builds its board on every searched decision, so a cost would show in every
  row; none does. Decisions where the rule looked at a set-up position took 1,202, 1,199 and 1,196 ms
  on average, the same as the rest.
- Errors: counter rules raised 0 times; 0 exceptions, 0 invalid orders, 0 random orders, 0 stalled
  games, 0 inactivity, 0 forfeits; helpers: 0 decision errors, timeouts, expiries, exits or hangs; the
  event loop was never 0.5 s late (worst 416 ms).
- 9 "unavailable choice" answers from the server (a switch while trapped by an unseen ability): 6 in
  mix-counter games, 3 in mix games, **none on a turn the rule switched**. They are mix's own picks
  (in both su games the rule changed nothing).

## 4. Two flaws seen in the lost games (not fixed: this run measures d573f7e as built)

`report.txt` lists, for every lost mix-counter game, each move the rules changed. Two things stand out:

1. **"Don't hit the boosted defence" can undo "use the counter".** In Foul Play game 278391,
   rule 1 kept mix's Dragon Tail (a phazing counter) against a Pokemon at +1 to +2 Defence. Then
   rule 2 swapped it for Shadow Ball on six turns in a row, because Dragon Tail is a physical attack
   into the boosted Defence. Then the game was lost. It happened in only this game (6 of the 29 swaps).
   The fix is one condition in `search_player._board_rules`: rule 2 should not swap away a
   move that rule 1 counts as a working counter ("kept: mix plays a counter").
2. **Encore locks in whatever the opponent did on that turn.** If the opponent moves first,
   that is this turn's move, not the set-up move that triggered the rule. Of the 21 Encores the rule
   chose, 17 landed: 12 locked a set-up move, 5 locked an attack (Bug Buzz, Moonblast, Knock Off,
   Dragon Claw, Dragon Ascent). 3 were chosen while our Pokemon was asleep, and 1 missed
   (`encore.txt`). In set-up bot game 278288, Vivillon (+1 Speed and faster) put Dewgong to sleep. When
   Encore finally landed on turn 19, it locked Vivillon into Bug Buzz, which knocked Dewgong out.
   A speed check (Encore only when we move first, or when the opponent's likeliest move is the set-up
   move again) would remove most of these.

Neither is frequent enough to block the ladder A/B: together they touched 6 of the 396 mix-counter games
(278391, and the five Encores that locked an attack).

## 5. How it was run

- **Order and budget**, as asked: the set-up test bot, then Foul Play at fair compute, then the
  network alone. Each phase aimed at 150 games per arm, and no game started more than 95 minutes after
  the start. The set-up bot was capped at 26 minutes and took 22. Foul Play got the time left minus 30
  minutes (43). The network got the rest (29). `bench/counter_gauntlet/master.sh`, `phase.sh`,
  `server.sh` (copies of the box's `~/dev/counter-gauntlet`).
- **Settings** (round 1's method, `bench/notes/setup_prove.md` section 2): local server on port
  10500 (`start_servers.py --no-security`), `bench/gauntlet.py ours --arms mix,mix-counter --per-arm 150
  --search-ms 500 --deadline 2.5 --mix-rule veto`, predictions on as on the ladder, seeds 41, 42 and 43.
  Set-up bot and network: 7 games at a time on 8 helpers (11 processes with the server and the
  opponent). Foul Play: 5 games at a time on 6 helpers, and Foul Play 450 ms on 3 helpers of its own
  (12 processes). nice 15, `OMP_NUM_THREADS=3`, no GPU. Round 1 used 5 games on 6 helpers (3 against
  Foul Play): more games at a time changes the speed, not the conditions, which are the same for both arms.
- **Changes to `bench/gauntlet.py`:** each decision line in `decisions.jsonl` now keeps the rules'
  `counter` record, and our side may have up to 12 helpers (it was capped at 6, a laptop-era limit).
  Nothing in the players changed.
- **A false start, not counted:** the first start (02:49) asked for 8 helpers and the old cap of 6
  stopped the set-up phase at once. The master then went on to Foul Play, which was stopped after 11
  games (4-7) so the order could be kept. Those games are set aside in `~/dev/counter-gauntlet/aborted`.
- **The live run:** 3,394 decisions a second before the start. During the run, 3,088 to 3,280 (10-minute windows).
  The one exception was a dip to 1,703 at 03:41, from the trainer's own planned restart (start 14 at
  03:35). Another job (`~/dev/counter-replay`) also ran Foul Play helpers on the box at times.
- **Reproduce the tables:** `python bench/counter_gauntlet_report.py ~/dev/counter-gauntlet/g/su
  ~/dev/counter-gauntlet/g/fp ~/dev/counter-gauntlet/g/nt --json report.json`;
  `python bench/setup_ladder_scan.py --dir ~/dev/counter-gauntlet/g/<phase> --quiet --by-arm --arms
  mix,mix-counter`; `python bench/counter_gauntlet/encore_check.py ~/dev/counter-gauntlet/g/{su,fp,nt}`.
  Committed: `bench/counter_gauntlet/report.txt`, `report.json`, `sweeps.txt`, `encore.txt`, and
  `games.jsonl` (one line per game: arm, side, result, turns, fallbacks, late, decision_ms_max, and the
  rule counts). The decision and battle files stay on the box in
  `~/dev/counter-gauntlet/g/{su,fp,nt}` (184 MB).
