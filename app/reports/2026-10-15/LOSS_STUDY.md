# Why the bot loses, and what we are doing about it

Two patterns lose most of our ladder games: one opposing Pokemon sweeping the team (68% of losses),
and the search overruling the network with a move the network all but ruled out. The second is
cheap to fix: a guarded search (`mix-trust`) is built, tested and on its way through a 600-game
test. The sweeps need training work (set-up defence, better set prediction), planned below.

Written 2026-10-15 for the owner and the collaborator. Script: `reports/2026-10-15/loss_study.py`.

## The data

518 public ladder games of the network with the search (mix), 318 won and 200 lost, from the
battle records of the owner's laptop (`psbot-local/runs/ladder/battles`, `ps-ppo-search/runs/ladder/battles`),
2026-10-07 to 15, several networks. Every decision record has the network's choice and probability,
the search's choice, the move played and the network's own win estimate.

## What the losses look like

| Pattern | In losses | In wins |
|---|---|---|
| One opposing Pokemon knocked out 3 or more of ours | **68.0%** | 5.0% |
| ... and it had used a set-up move (Swords Dance, Calm Mind, Iron Defense ...) | 34.5% | |
| An opposing Pokemon reached +4 boosts at some point | 43.5% | 33.0% |
| An opposing Pokemon reached +2 at some point | 76.5% | 73.9% |
| Our Pokemon statused, per game | 1.35 | 1.09 |
| Hazards on our side (kinds) / cleared, per game | 0.54 / 0.28 | 0.46 / 0.18 |
| Turns | 31.2 | 24.5 |
| The network's win estimate once reached 60% (a "throw") | 31.5% | (29.2% of wins once dipped under 40%: mostly noise) |
| Tera used / median turn | 89% / 12 | 79% / 12 |

- The losses turn late: the first turn the win estimate stays under 30% is turn 18 (median) of 27,
  and a set-up move had been used before it in 64% of those losses.
- The sweepers are many different species, at most 2 losses each: set-up walls (Carbink,
  Registeel, Probopass with Iron Defense; Espathra, Florges, Phione, Hatterene with Calm Mind or
  Take Heart) and Swords Dance users (Kingambit, Bisharp). The bot lets them set up instead of
  statusing, phazing or hitting them hard early.

## The search overruling the network

The search overrules the network on about 7 to 9 moves a game. After an overrule, the network's own
win estimate at the next decision falls; after the network's own move it rises. The same holds in
every band of win estimate, so it is not just that overrules happen in bad positions:

| Network's win estimate | After its own move | After an overrule | After an overrule to a move it gave under 2% |
|---|---|---|---|
| 0 to 10% | +0.35 pts | -0.07 | -1.08 |
| 20 to 30% | +1.06 | -0.30 | -1.18 |
| 40 to 50% | +0.36 | -0.53 | -2.16 |
| 50 to 60% | +0.90 | -1.70 | -2.82 |
| 70 to 80% | +0.53 | -2.49 | -5.15 |
| all, weighted as overrules happen | **+0.70** | **-0.90** | **-2.12** |

1,778 of the 3,911 overrules (45%) chose a move the network gave under 2%. Caveat: the judge here is
the network's own value head, which may be biased against moves it would not make. Only a game test
can settle it; that is what the next section runs.

## Fix 1 (built): the guarded search, `mix-trust`

- **What it does.** The veto rule as before (the search marks as unsound every move it gave less
  than a quarter of its favourite's share, and the network plays its likeliest sound move), but only
  among the moves the network gives 3% or more (and always its likeliest). The search can still turn
  the network from its first choice, never to a move the network all but rules out.
- **Code.** `search_player.py`: `MixSettings.trust` (0 by default, so every other mode is unchanged)
  and the arm `mix-trust` (0.03); `ladder.py --player mix-trust`. Tests in
  `tests/test_search_player.py`, including 2,000 random positions where it never plays a move under
  3% other than the network's likeliest. 207 tests pass.
- **In the window app.** A new "Who plays" choice, "Bot: best network + guarded search (new)", which
  selects the best ladder network automatically: the main run's update 11850 ("BEST ON LADDER, Main
  run, saved 7 Oct 00:31", the account's peak of 2,466).
- **The test (running on the box, `~/dev/trust`).** Network 011850, mix against mix-trust, 300 games
  each, sides balanced, 500 ms of search (the app's 1 s setting), against Foul Play at 150 ms on a
  private server. Pass: mix-trust wins more than mix with a 95% interval that clears zero, then a
  ladder confirmation.

### Result (2026-10-15, 08:55 to 11:35 UTC, `~/dev/trust`)

| Arm (network 011850, 500 ms search, against Foul Play 150 ms) | Record | Win rate (95% Wilson) | Network overruled |
|---|---|---|---|
| mix (plain veto) | 180-120 | 60.0% (54.4 to 65.4) | 21.7% of decisions |
| mix-trust (guarded) | 180-120 | 60.0% (54.4 to 65.4) | 13.7% of decisions |

- Difference mix-trust minus mix: +0.0 points (95% interval -7.8 to +7.8), z 0.00. 600 games,
  every one finished normally, 0 search errors, 0 timeouts.
- **It does not pass.** The guard does what it was built to do (overrules fall by about a third),
  but against Foul Play that changes nothing: the far-fetched overrules it removes were not
  deciding games against this opponent, or their effect is under what 300 games per arm can see
  (about 4 points).
- What is still open: the loss study's signal came from the ladder against people, where the
  search's guesses about the opponent are worse. The guarded mode stays in the window as an option
  ("Bot: best network + guarded search (new)") at no measured cost; a ladder A/B or a head-to-head
  against mix would settle whether it helps there.

## Thinking time: 5 s a move does not beat 1 s

The owner asked whether letting the search think longer helps. Test on the box (`~/dev/think`,
2026-10-15 08:28 to 12:21 UTC): the full run's arm C network at 454M decisions, arm mix, against
Foul Play at 150 ms on a private server, 300 games per setting, sides balanced.

| Setting (window app) | Search budget | Record | Win rate (95% Wilson) | Iterations per decision (mean / median) | Time per decision (median) |
|---|---|---|---|---|---|
| 1 s | 500 ms | 201-99 | 67.0% (61.5 to 72.1) | 790k / 417k | 1.18 s |
| 5 s | 2,200 ms | 201-99 | 67.0% (61.5 to 72.1) | 2.85M / 1.54M | 5.31 s |

- Difference 5 s minus 1 s: +0.0 points (95% interval -7.5 to +7.5), z 0.00.
- The longer setting really did search about 3.7 times as much per move, so the budget reaches the
  search. It just does not change the result: the network decides most of the strength, and the
  search's extra iterations go into worlds sampled from the same set guesses.
- 0 invalid orders and 0 exceptions in either run; at 5 s, 76 of 10,392 answers fell back to the
  network because the search ran out of time.
- Use the 1 s setting: same strength, games four times faster. More search time is not where the
  next gain is; better set prediction for the search (Fix 3) and training are.

## Fix 2 (planned): set-up defence

- Train against opponents that set up: the box's gauntlet already has a set-up opponent
  (`bench/gauntlet.py --kind setup`, SimpleHeuristics that sets up and puts to sleep), and PPO could
  start a share of its battles from set-up positions.
- Run the counter rules of branch `counterfix` (use the counter on the team, do not hit the boosted
  defence; `mix-counter`) in a 400-game test: so far they were only tested on puzzles.

## Fix 3 (planned): predict the opponent's sets

Sample the search's worlds from a set predictor trained on the 35,919 top-player games instead of
Foul Play's usage statistics, so the search stops being surprised by set-up sets. See
`reports/2026-10-15/OWN_SEARCH_IDEAS.md`, idea 2.
