# The counter rules after round 3's review (counterfix bbf9a75)

**Result: the fixed mix-counter is level with mix on the puzzles (+0.09 points, 95% -0.54 to +0.75,
pre-registered reading: the A/B goes ahead), about even by the search on the ladder replays, and live
on the box it ran with no error, no late decision and no invalid order. It now changes very few
decisions: 18 in 149 ladder games, against about 67 before.** So the ladder A/B is a check that it does no
harm, not a way to show a gain (see "How to read the A/B").

## What changed (review finding -> fix)

| finding (review of d573f7e) | fix |
|---|---|
| Puzzles: switching to a benched counter cost about 80% of the -3.01 | `counter_switch` (new, off): no switch to a benched holder. Only the active's own counter is played. |
| The KO exception never fired at full HP (32 of 69 overridden attacks were into a full-HP threat) | `Threat.counter_ko`: a KO at full HP counts at twice its HP, or when Sturdy, Focus Sash, Multiscale, Shadow Shield, Disguise and Ice Face are ruled out (`setup_counter.full_hp_guard`: known ability, else the search's sampled sets, else every possible ability; known item, else two or more sampled sets without a Sash). Other doubts (Assault Vest, Eviolite, Levitate...) still block it. `Threat.ko` (sure_attack's) is unchanged. |
| The other-defence swap undid a counter mix chose (Dragon Tail -> Shadow Ball six turns running) | No swap when mix's move is a working counter, whether or not the counter rule fired. |
| Encore locked in attacks; picked while our Pokemon slept | Encore counts only when we move first (speed with stages, Scarf, paralysis, Quick Feet, weather abilities, Trick Room) or the read puts 50% or more on a set-up move again. No counter while our active is asleep or frozen. |
| Switch loops (game 278209) and giving up our own boosts | Gone with the switches. |
| A counter played in place had no survival check | Priority -6 counters (Roar, Whirlwind, Dragon Tail, Circle Throw): not when the threat's best hit (two_hit_ko's, top roll) is at least our HP. Priority 0 counters: not when the threat moves first and its best hit is at least our HP. Prankster counts +1. |
| The 0.8 ratio swapped to weaker or less accurate moves, KO moves and priority could be dropped | The other kind must do at least as much damage times accuracy (`other_side_ratio` 1.00). Never off a priority move, never off an attack in `counter_ko` for one outside it. Shell Side Arm, Photon Geyser, Light That Burns the Sky, Tera Blast, Tera Starstorm are neither judged nor swapped in. |
| Smaller gaps | Wind Rider stops Whirlwind; Prankster status moves fail on Dark types; Haze not when it clears more of our positive stages than of theirs; sleep moves need 75% accuracy (no Hypnosis, Sing, Grass Whistle, Dark Void); Safeguard, Leaf Guard in the sun, Flower Veil. |
| Not from the review, found now | `ladder.py --arms` balanced each new game against ALL earlier games of the checkpoint with that arm name. ps-ppo-predict has 220 earlier "mix" games for 011850, so mix-counter would have got every game for the next 220. Each A/B game now records `"experiment": "mix,mix-counter"`, the balance counts only those, and `setup_ladder_scan.py --experiment` reads only those. |

Not changed: Sleep Clause still counts any sleeping opponent (a Rest included); hazards on entry are not
in the holders' two-hit check (no switches now); the reconnect fix (review: correct) is as built.

## Checks

**Tests on the box** (~/dev/counter-fix, nice 15, CPU only, one process): 343 passed, 6 skipped (the 6
known skips) across test_counter, test_foulplay, test_foulplay_bridge, test_opponent_model,
test_predict_live, test_search_player, test_setup_fix, test_setup_round2, test_tera. Both fuzz tests
ran against a3b1cec and 478d82a: with the flags off mix() is unchanged on 4,500 random positions with 11
settings each. New tests cover every fix above and the A/B balance.

**Ladder replay** (bench/counter_replay.py through the real player, both sets, 149 games; the replayed mix
matched live play on 1,473 of 1,473 and 3,042 of 3,043 decisions; 0 rule errors):

| kind of game | games | changed decisions per 100 (95%) | games with a change | before the fixes |
|---|---|---|---|---|
| set-up sweep losses | 25 | 0.8 (0.4 to 1.7) | 4 (16%, 6 to 35) | 3.1 per 100, 8 games |
| other losses with a set-up | 25 | 0.2 (0.1 to 0.8) | 2 (8%, 2 to 25) | 0.6, 5 games |
| losses without a set-up | 10 | 0.0 (0 to 1.2) | 0 | 0.0 |
| wins | 89 | 0.4 (0.2 to 0.7) | 7 (8%, 4 to 15) | 1.4, 22 games |

The search asked again on every change (rule's action minus mix's, points of win chance, 95% over games):

| | sweep losses (7 changes) | wins (9 changes) |
|---|---|---|
| live effort | +0.7 (-0.8, +4.0) | -4.4 (-6.2, -3.1) |
| 500 ms | +0.7 (-0.5, +3.7) | -5.9 (-8.9, -2.9) |
| before the fixes, live effort | -4.3 (27 changes) | -8.0 (34 changes) |

Counter moves in sweep losses scored +4.0 (2 changes: Whirlwind on +1/+1 Arceus in 8092, Roar on +2
Pecharunt in 4869). Rough arithmetic: about -0.4 games in 89 wins, +0.05 in 25 sweep losses. Net near zero.
Game 1202 (Encore on +6 Scrafty) no longer changes: our Jumpluff was paralysed and slower, and the read
was under 50%, so Encore could have locked in an attack.

**Puzzles** (setupfix ROUND3B.md, pre-registered in ROUND3B_PLAN.md, CHOOSE half, never scored by a
counter rule):

| set | puzzles | difference, by game (95%) | changed | fixed / broken |
|---|---|---|---|---|
| where the rule acts (main) | 2,264 | **+0.09 (-0.54, +0.75)** | 73 | 29 / 27 |
| top player and judge disagreed | 1,079 | +1.58 (+0.65, +2.52) | 42 | 22 / 5 |
| agreed (500 ms judge) | 1,185 | -1.27 (-2.18, -0.42) | 31 | 7 / 22 |
| agreed, 5,000 ms judge only | 340 | -2.06 (-3.86, -0.31) | 9 | 1 / 8 |
| before the fixes (CONFIRM, main) | 2,426 | -3.01 (-4.03, -1.97) | 226 | 42 / 115 |

**Live smoke on the box** (port 10510, mix against mix-counter, the set-up test bot then Foul Play at
fair compute, about 12 minutes each): see the section at the end.

## How to read the A/B

- Each game is mix or mix-counter, balanced from this A/B's own games. The end of each run prints the
  per-arm record and the set-up sweep losses per arm (`setup_ladder_scan.py --by-arm --experiment`).
- The rule changes about 1 decision in 125 in sweep losses and 1 in 250 in wins. Its likely effect on
  the win rate is under 1 point either way. No practical number of ladder games can show that: a 100
  games-per-arm A/B has a 95% interval of about +-13.6 points on win rate and +-10 on the sweep-loss
  share; 380 per arm gets the sweep-loss share to about +-5.
- **Reading, fixed now:** at 100 games per arm, go back to plain mix if the win-rate difference's upper
  95% end is below 0 (a clear loss). Otherwise keep alternating. The sweep-loss counts mean little before
  about 380 games per arm, and even then only a large change would show.
- What the replay says about sweep losses: in 17 of 25 there was no counter the rule could use, and with
  the fixes it acts in 4. Set-up sweeps need more than this rule; the counter is rarely the answer.

## Live smoke on the box (partial)

Port 10510, mix against mix-counter against the set-up test bot, parallel 4 on 5 helpers, from 05:34 UTC.
At 46 games (23 per arm, the last reading taken): mix 23-1, mix-counter 23-0. In mix-counter games the
rules looked at 113 set-up positions and changed no decision (no counter could work 64, an attack in KO
range 30, a counter on the bench only 10, our Unaware 7, mix played a counter 2; the boosted defence hit 8
times, no swap). Late decisions 0, fallbacks only the 15 single-action ones, decision time +7.4 ms (0.5
to 14.3) of about 1,200 ms, slowest 1,461 ms against a 2,500 ms deadline. The Foul Play phase was not
run: the phase was stopped early and its final report was not read. Far too few games for any win-rate
claim; it shows only that the fixed rules run live without trouble and act rarely.
