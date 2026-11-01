# Search plumbing (bridge): what is there, what was measured, what is not done

2026-10-06, branch `search`. Files: `foulplay_bridge.py`, `bench/fp_bridge_server.py`,
`tests/test_foulplay_bridge.py`, new `bench/fp_bridge_measure.py`, and the section "Search at
play time" in `bench/FOULPLAY_BRIDGE.md`, which has the full tables. `foulplay_player.py` is
unchanged. Everything here ran on the laptop that evening with the live ladder bot and other
test runs on it. Ports 8620-8629 were not used: no Showdown server was needed for this part.

## What a Windows player calls

```python
import foulplay_bridge as fb
ok, why = fb.available()            # (True, "foul-play 8f541f80... at /home/you/psppo-bench/foul-play in WSL")
bridge = fb.FoulPlayBridge(helpers=4, search_ms=500, timeout=4.0)
bridge.start()                      # BEFORE queueing: 0.4 to 11 s, and up to 44 s when WSL itself has to start

class P(fb.FoulPlayMixin, Player):
    foulplay_enabled = True         # set self.foulplay = bridge before Player.__init__
    async def choose_move(self, battle):
        advice = await self.foulplay_advice(battle)                      # search_ms of the bridge
        advice = await self.foulplay_advice(battle, payoff_ms=300)       # + the engine's payoff matrix
        advice = await self.foulplay_advice(battle, iterations=20000)    # exact iterations per world
        ...
        self.foulplay_note(battle, order)    # REQUIRED when the order sent is not Foul Play's own
```

Our side, as before: `advice.policy[14]`, `advice.visits[14]`, `advice.score[14]`, `advice.mask[14]`,
`advice.best_action`, `advice.action`, `advice.iterations`, `advice.worlds`, `advice.ms`,
`advice.wall_ms`. Actions 10-13 are the Tera moves: their share is the search's opinion on
Terastallizing now.

New, the opponent (every decision, no extra cost):

| | |
|---|---|
| `advice.opponent` | list of `OpponentOption(name, fp, share, visits, score, worlds, seen)`, largest share first. `name` is `move earthquake`, `move earthquake terastallize`, `switch garchomp`, or `none` (our forced switch). `share` sums to 1 |
| `score` | from the OPPONENT's side, 0 to 1. Checked: our total score + its total score = the iterations |
| `seen` | switches only. False = the sampling invented that Pokemon to fill the team. Do not read such a share as a prediction of that species |
| `advice.opponent_active`, `advice.opponent_revealed` | Foul Play's names (lower case, no spaces: `ogerponwellspring`) |
| `advice.opponent_worlds[i]` | `chance`, `iterations`, `options` in that world, `reserve`, and `active`: the set drawn for the active Pokemon (`species`, `moves`, `item`, `ability`, `tera_type`, `hp`, `max_hp`, `level`, `status`, `terastallized`) |

New, the payoff table (only with `payoff_ms`), after `fill_actions` (the mixin calls it):

| | |
|---|---|
| `advice.payoff_table[a][c]` | our action `a` (0-13) against `advice.payoff_theirs[c]`. Mean over the worlds that have the cell. None = no world has it |
| `advice.payoff_known[a][c]` | share of the worlds behind the cell |
| `advice.payoff_depth`, `advice.payoff_worlds` | plies per world, and the table per world |
| unit | the engine's evaluation, positive good for us. NOT the 0 to 1 of the Monte Carlo scores. The engine's MCTS turns it into a score with `sigmoid(0.0125 * (value - value of the root))` |

## Findings

1. **poke-engine 0.0.48 gives a pair value only for the pairs its pruning needed.**
   `iterative_deepening_expectiminimax(state, ms)` returns `matrix` (ours by theirs) and
   `depth_searched`, but it prunes: about half the cells are NaN (46 to 49% known over 60
   decisions, 68 of 174 known on 4 full states). Depth: 2 in 100 ms, 2 in 300 ms, 3 in 1,000 ms
   on a 12 x 7 position; 3, 4, 4 on a forced switch. Through the bridge with the budget shared
   among the worlds: mean 2.43 at 100 ms, 2.66 at 300 ms, 3.11 at 1,000 ms.
2. **A complete table can be built** from `generate_instructions` (every outcome of a pair with
   its chance) + `State.apply_instructions` + a 1 ply search of each child: `payoff="full"`.
   2 plies, no pruned cell, 4 to 7 ms per outcome: 643 ms mean with a 1,000 ms budget (80% of the
   cells), 957 ms mean and up to 3 s with a 3,000 ms budget (100%). Does not fit a 2 s answer.
3. The module does not offer the evaluation of a state by itself, nor a search without pruning.
   `IterativeDeepeningResult.get_safest_move()` is broken upstream (its index never advances).
4. `monte_carlo_tree_search(state, ms, iterations, threads)`: `iterations > 0` is exact
   (`[20000, 20000]` checked). `threads=2`: about 1.5 times the iterations for about 10% more
   time; `threads=4`: 2.3 to 2.6 times for 21 to 25% more. Tree quality per iteration with
   threads: not measured. Both are request fields now.
5. **The engine holds the interpreter lock** in both searches (a Python thread ticked once in
   500 ms). One helper is one decision at a time.
6. **Killing wsl.exe does not kill the Linux process.** Measured: helper still there 7 s after
   taskkill of its wsl.exe, its stdin still open. Hence the tag on the command line (stopped by
   `pkill` inside WSL) and the line every 2 s without which an idle helper leaves after 20 s.
7. **Every wsl.exe start costs 2 to 9 s on the loaded laptop**, and WSL shuts down when nothing
   runs in it (`available()` took 17 and 20 s when it had to start, 6 s otherwise).
8. The pipe through wsl.exe costs 1 to 10 ms per decision, the same as inside WSL.

## Measured (iterations per decision, all worlds; time from asked to answered)

Windows to WSL against inside WSL, as many decisions at once as helpers, 24/48/72 decisions per row:

| Helpers | ms | Win to WSL iterations mean | asked to answered ms mean (p95) | Inside WSL iterations mean | ms mean (p95) |
|---|---|---|---|---|---|
| 2 | 100 | 67,208 | 245 (262) | 65,958 | 244 (260) |
| 2 | 250 | 155,208 | 576 (603) | 158,416 | 576 (599) |
| 2 | 500 | 316,791 | 1,145 (1,181) | 312,916 | 1,141 (1,177) |
| 2 | 1000 | 636,625 | 2,287 (2,360) | 551,208 | 2,288 (2,348) |
| 4 | 100 | 65,937 | 243 (261) | 55,770 | 249 (281) |
| 4 | 250 | 156,562 | 577 (601) | 128,312 | 583 (608) |
| 4 | 500 | 311,062 | 1,159 (1,232) | 252,812 | 1,154 (1,195) |
| 4 | 1000 | 554,895 | 2,293 (2,389) | 509,083 | 2,316 (2,460) |
| 6 | 100 | 50,194 | 253 (285) | 48,333 | 255 (279) |
| 6 | 250 | 121,402 | 590 (617) | 114,972 | 589 (609) |
| 6 | 500 | 236,847 | 1,153 (1,195) | 217,902 | 1,162 (1,198) |
| 6 | 1000 | 419,486 | 2,341 (2,494) | 399,444 | 2,317 (2,439) |

3 decisions at a time (the ladder bot's shape), from Windows, 36 decisions per cell:

| Helpers | 500 ms | 750 ms | 1000 ms |
|---|---|---|---|
| 3 | 247,416 it, 1,167 ms (max 1,245) | 358,416 it, 1,735 ms (max 1,922) | 474,361 it, 2,295 ms (max 2,444) |
| 4 | 280,305 it, 1,160 ms (max 1,240) | 401,472 it, 1,715 ms (max 1,780) | 526,750 it, 2,292 ms (max 2,353) |
| 6 | 298,944 it, 1,156 ms (max 1,214) | 434,305 it, 1,725 ms (max 1,857) | 570,722 it, 2,292 ms (max 2,365) |

A decision takes 2.3 times the search time, always. The iterations are what moves with the load:
22 to 34% fewer with 6 searches at once than with 2.

## Recommended for the ladder bot (3 games at once, answer in about 2 s)

`search_ms=500`, `helpers=4`, `timeout=4.0`, `bridge.start()` before queueing, 1 thread.
1.16 s per decision on average, 1.32 s at the slowest of 252 measured, 237,000 to 317,000
iterations per decision. That leaves 0.7 s for the network and, if wanted, `payoff_ms=300`.
750 ms fits (1.73 s, slowest 1.92 s) and leaves nothing. 1,000 ms does not fit (2.3 s).
Watch `bridge.stats()["recent_iterations_per_decision"]`: the time stays, the iterations go.

## Tests

`python -m pytest tests/test_foulplay_bridge.py -q`: 95 tests, 20 of them new.
Windows (helpers inside WSL): 88 passed, 7 skipped, 312 s. Inside WSL: 90 passed, 5 skipped,
187 s. The 75 old ones: all pass or are skipped for the reason they were skipped before (the 2
that wait for `worker.py`; the 3 of the exact check on Windows). One old assertion was changed:
the helper's protocol version is 2. No helper was left inside WSL after either run (`pgrep`).

## Not done, not measured

- **Whether any of it makes the bot stronger.** No game was played with the search in this part.
- **How well `advice.opponent` predicts real opponents.** It is the search's opinion against
  guessed sets. Nobody compared it with recorded games here.
- The exact check (`bench/fp_bridge_replay.py`) and `bench/foulplay.py` do not start through WSL
  from Windows. Inside WSL the exact check passes with the changed helper (3 tests).
- `calibrate()` works from Windows but its target (60,000 iterations) is the teacher's, not a
  ladder bot's. Use `bench/fp_bridge_measure.py`.
- Threads: only iterations and time, on 36 decisions per setting. No strength comparison.
- The complete payoff table searches each child 1 ply only. Deeper needs more than a 2 s answer.
- A helper orphaned by a killed wsl.exe stays for up to 20 s (idle, about 55 MB).
- `--pool fork` through WSL was not run.
- Revival Blessing, a rejoined room and Foul Play's own failures are as before: no answer.
