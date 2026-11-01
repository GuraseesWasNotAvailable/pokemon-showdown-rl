# Set-up fixes: what was built, what the puzzles say, what to play

Built 2026-10-07 on branch `setupfix-play` (from predict-live-merged d7ace06). Everything is **off
by default**: without the new flags the ladder bot plays exactly as before.

**Superseded (2026-10-08).** The proof and its review (`bench/notes/setup_prove.md`) found that no
play-time setting measured better: `--veto-fix` is level with today's mix against top players, and
`--setup-defer` gains only where the top player and the engine agree, exactly as the 500 ms search
alone does, and loses 5.4 points where they disagree. Look Play stays as it is. The table below is the
build job's reading, kept as it was.

## What to play next (recommendation, superseded: see setup_prove.md)

| flag | what it does | set-up puzzles, against the plain veto | play it? |
|---|---|---|---|
| `--veto-fix` | never a move neither side picked; our set-up move favoured by the search does not veto an attack the network gives 0.7+ that is estimated to KO; Tera only when one side picked the Tera form | +8 sound (15 fixed, 7 broken), +21 on the oracle's best, -5 on the top player (z -0.8); the "neither side picked" moves fall from 108 to 32 | **yes**: small, safe, removes the third actions |
| `--setup-defer` | in a set-up position, the search's best | +64 sound (z 5.5), +2.1 points on the agreed set, but -27 on the top player (z -1.7) | only as a measured experiment |
| `--setup-switch-sure 0.8` | with the above, the network's sure switch stands | -14 sound, **-72 on the top player (z -4.3)** | **no** |
| `--guard` / `mix-guard` | the earlier guard | **-76 sound (z -6.4), -58 on the top player (z -4.5)** | **no** |
| `--think-longer 2000` | a 2,000 ms search on set-up turns | not measurable on the puzzles (see below) | optional; expected effect small |

So the two earlier recommendations "play mix-guard" and "never veto a network switch it is 80%
sure of" are both **refuted** by the 1,889 set-up puzzles. The guard drops our passive moves in
set-up positions, and top players use them (14% of their choices, 69% sound). The network's sure
switches are the network's main weakness (its switches are 48% sound), so protecting them hurts.

Ladder command for the next session (Look Play.cmd with one flag added):

    ladder.py --games 20 --player mix --predictions --parallel 2 --helpers 3 --search-ms 500 --deadline 2.5 --veto-fix --checkpoint ...

## The measurement

`bench/setup_puzzles/score.py --variants` on branch setupfix scores each fix on the 1,889 set-up
puzzles (held-out 2100+ players, oracle = Foul Play at 5,000 ms; see that README), with the
ladder's network 011850 and the 500 ms search answers stored with the puzzles. A puzzle's threat
is its opponent's boosts, whether it just set up, the 500 ms search's read of it, and our passive
actions; the KO estimate cannot be made (no stats), so `sure_attack` is not in these numbers.

Network 011850 (017900 reads the same within a point):

| player | sound, all 1,889 | = oracle best | = top player | agreed set (982) |
|---|---|---|---|---|
| plain veto (today) | 86.7 | 65.3 | **56.2** | 81.4 |
| `--veto-fix` (no_third, tera_own) | 87.1 | 66.4 | 56.0 | 81.7 |
| `--setup-defer` | 90.0 | 71.4 | 54.8 | 83.5 |
| `--setup-defer --veto-fix` | 90.1 | 71.6 | 54.8 | 83.7 |
| `--setup-defer --setup-switch-sure 0.8` | 85.9 | 67.7 | 52.4 | 79.3 |
| `mix-guard` | 82.6 | 61.5 | 53.1 | 75.1 |
| search alone, 500 ms | 91.1 | 72.9 | 53.6 | 83.7 |

1,477 of the 1,889 puzzles are set-up positions by the live rule (`setup_reason`); the rest are a
+1 boost with no set-up seen. 95% intervals are about +-2 points per column; the paired counts
above are the sharper test.

How to read `--setup-defer`: it moves the mix toward the search, which is the same engine as the
oracle, so its gain on "sound" and "oracle best" is partly shared judgement. Against the top
players it loses 1.4 points. Whether the engine or the top player is right in those positions is
exactly what the puzzles cannot settle, so it should be tried on the ladder only as a measured
arm, not switched on.

## The longer search (`--think-longer MS`)

- Asked at once (one search, of MS) when the opponent's active Pokemon is at +1 or more in attack,
  defences or speed, or used a set-up move last turn. Asked again after the ordinary 500 ms answer
  when that answer's read of the opponent puts 0.4 or more on a set-up move; the long answer
  replaces the ordinary one when it comes in time, else the ordinary one plays.
- Waited for up to `minimum_deadline(MS) + 1.0` s (6.4 s at 2,000 ms). The ladder's timer was
  150 s every turn in all 100 recorded games, so time is not a constraint; it is still checked
  (`|inactive|Time left:` lines, 30 s margin).
- A second engine thread when a helper is free (3 helpers, 2 games: one is).
- On a real helper on the box (`bench/think_longer_check.py`, 40 recorded decisions, quiet helper):

  | search | wall time, median (max) | iterations, median | best differs from 500 ms |
  |---|---|---|---|
  | 500 ms (the ladder's) | 1.20 s (1.25) | 223,000 | |
  | 2,000 ms, 1 thread | 4.70 s (4.85) | 854,500 | 11 of 40 |
  | 2,000 ms, 2 threads | 4.94 s (5.25) | 1,355,500 | 12 of 40 |

  None came later than the wait. The laptop's helpers run in WSL beside the games, so expect
  them a little slower; the 1 s margin is for that.
- Expected effect is small: the ladder reruns found a longer search changes the outcome of at most
  2 of the 14 sweep losses, and on the puzzles the 500 ms search is within a point of the oracle's
  own repeat rate. `bench/think_longer_check.py` measures its time and iterations on a real helper.

## Files

- `search_player.py`: `MixSettings.no_third`, `sure_attack`, `tera_own`, `setup_defer`,
  `setup_switch_sure`; `setup_reason`, `ko_estimate`, `ThinkLonger`, `parse_time_left`; the
  player records `fix` (what the plain veto would have played) and `long` per decision, and counts
  them per game and in `search_summary()`.
- `ladder.py`: `--veto-fix`, `--sure-attack`, `--setup-defer`, `--setup-switch-sure`,
  `--think-longer`, `--think-longer-read`, `--think-longer-boost`, `--think-longer-threads`.
- `setup_moves.py`: `largest_boost`, `setup_share` (shared with the search teacher on setupfix).
- `tests/test_setup_fix.py`; `bench/think_longer_check.py`.
