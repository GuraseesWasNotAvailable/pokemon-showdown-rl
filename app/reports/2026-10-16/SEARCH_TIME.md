# Does thinking longer make the ladder bot worse?

No, not measurably. On the owner's ladder games the 5 s setting (2,200 ms of search) won 60.0% of
150 games and the 1 s setting (500 ms) 59.9% of 324: a difference of +0.1 points (95% interval
-9.4 to +9.4). It is not better either, and it does change how the bot plays. With the same network,
the longer search overrules the network 4 points more often. The extra time makes the search more
confident inside the same 2 or so guessed opponent teams, and it reads the opponent no better.

Written 2026-10-16 for the owner and the collaborator. Script: `reports/2026-10-16/search_time.py`
(read-only, one process, about 5 s; it reuses `loss_study.py`'s parsing of the battle records).

## The data

- **474 public ladder games of the network with the search under the veto rule** (`mix` and
  `mix-trust`), 2026-10-06 to 16, 15,216 decisions (14,660 answered by the search). Sources: the
  battle records and `results.jsonl` in `psbot-local/runs/ladder` and `ps-ppo-search/runs/ladder`.
- **How the setting is recorded.** Each game's `search_ms` in `results.jsonl`, and each decision's
  `asked_ms` in the newer records. They agree in every game. The app maps 1 s to 500 ms (deadline
  2.5 s), 2 s to 900 ms (2.7 s) and 5 s to 2,200 ms (6.0 s) (`PSBotLookahead.cs`, `ThinkSearchMs` and
  `ThinkDeadline`). `search_ms` is Foul Play's time per sampled opponent team ("world"). A decision
  takes about 2.3 times it: 1.17 s and 4.98 s (medians measured).
- Only 1 s and 5 s were played on the ladder. There are no 2 s games.

| Setting | Mode | Networks (games) | When (UTC) |
|---|---|---|---|
| 1 s | mix | main 011850 (100), main 024381 (90), baseline 021725 (64), baseline 022640 (50), arm C 426M (20) | Oct 6 to 15 |
| 5 s | mix | arm C 426M (29), arm C 454M (34), main 011850 (27) | Oct 15 |
| 5 s | mix-trust (guarded) | main 011850 (45), v2 setupw2 56M (6), v2 setupw2 60M (9) | Oct 15 to 16 |

Left out: `mix-lines` (117) and `sol-lines` (86), both at 1 s, which add look-ahead lines or a model
on top of the search, and the model-only games (no search). They are shown once below for context.

**The setting was never randomised.** All 5 s games come from Oct 15 to 16, partly with other
networks. Two networks played at both settings (main 011850 and arm C 426M, 220 games). The 1 s games
of main 011850 are the account's climb from 1,969 on Oct 6, played against weaker opponents. Three
controls deal with this:
- a performance rating: the Elo strength that explains the score from the opponents' ratings alone;
- a comparison within one rating band;
- an Elo model that holds each network's own strength fixed.

## 1. Win rate and rating per setting

| Setting (veto rule) | Games | W-L | Win rate (95% Wilson) | Opponents (mean) | Performance rating (95%) | Rating change a game (95%) | Total |
|---|---|---|---|---|---|---|---|
| 1 s (500 ms) | 324 | 194-130 | 59.9% (54.5 to 65.1) | 2,242 | 2,322 (2,281 to 2,363) | +1.8 (-0.3 to +3.9) | +584 |
| 5 s (2,200 ms) | 150 | 90-60 | 60.0% (52.0 to 67.5) | 2,261 | 2,342 (2,283 to 2,401) | -0.3 (-3.4 to +2.9) | -40 |

- **Difference, 5 s minus 1 s:** +0.1 points of win rate (-9.4 to +9.4).
- **Same rating band (games started at 2,250 to 2,400):** 62.3% of 138 at 5 s against 60.8% of
  158 at 1 s, +1.6 (-9.5 to +12.5). Rating change a game: +1.0 (-2.3 to +4.2) against +1.9 (-1.1 to
  +4.9). The 1 s total of +584 is mostly the Oct 6 climb. In the same band, both settings gain about
  1 to 2 points a game.
- **Elo model, each network's strength held fixed:** 5 s is +31 Elo (-95 to +158), or +4.5 points
  of win rate at an even match (-13.3 to +21.3). Using only the two networks played at both settings
  gives the same +31 (-95 to +158).
  - Model checks: with a free slope on the opponent's rating, +17 (-108 to +141); with the opponent's
    rating minus our own, -44 (-170 to +82). Every version includes zero.
- **Without the 3 games where the search played under half the moves** (an internet outage, see
  section 2): 59.9% of 147, unchanged.

By mode and network (veto rule unless named):

| Group | Games | W-L | Win rate (95%) | Opponents | Performance rating (95%) | Total rating change |
|---|---|---|---|---|---|---|
| 1 s, mix | 324 | 194-130 | 59.9% (54.5 to 65.1) | 2,242 | 2,322 +/- 41 | +584 |
| 5 s, mix | 90 | 56-34 | 62.2% (51.9 to 71.5) | 2,248 | 2,348 +/- 78 | -79 |
| 5 s, mix-trust | 60 | 34-26 | 56.7% (44.1 to 68.4) | 2,282 | 2,332 +/- 92 | +39 |
| main 011850, 1 s, mix | 100 | 60-40 | 60.0% (50.2 to 69.1) | 2,145 | 2,223 +/- 73 | +377 |
| main 011850, 5 s, mix | 27 | 15-12 | 55.6% (37.3 to 72.4) | 2,253 | 2,297 +/- 138 | -89 |
| main 011850, 5 s, mix-trust | 45 | 26-19 | 57.8% (43.3 to 71.0) | 2,285 | 2,344 +/- 107 | +45 |
| arm C 426M, 1 s, mix | 20 | 13-7 | 65.0% (43.3 to 81.9) | 2,278 | 2,390 +/- 162 | -26 |
| arm C 426M, 5 s, mix | 29 | 19-10 | 65.5% (47.3 to 80.1) | 2,203 | 2,341 +/- 142 | -34 |
| arm C 454M, 5 s, mix | 34 | 22-12 | 64.7% (47.9 to 78.5) | 2,280 | 2,394 +/- 126 | +44 |
| main 024381, 1 s, mix | 90 | 51-39 | 56.7% (46.4 to 66.4) | 2,282 | 2,333 +/- 75 | -50 |
| baseline 021725, 1 s, mix | 64 | 41-23 | 64.1% (51.8 to 74.7) | 2,247 | 2,361 +/- 94 | +218 |
| baseline 022640, 1 s, mix | 50 | 29-21 | 58.0% (44.2 to 70.6) | 2,343 | 2,403 +/- 100 | +65 |
| v2 setupw2 56M + 60M, 5 s, mix-trust | 15 | 8-7 | 53.3% | 2,274 | | -6 |
| (context) 1 s, mix-lines | 117 | 73-44 | 62.4% (53.4 to 70.6) | 2,040 | 2,147 +/- 72 | +537 |
| (context) 1 s, sol-lines | 86 | 44-42 | 51.2% (40.8 to 61.4) | 2,239 | 2,248 +/- 76 | -85 |

- **Same network at both settings.**
  - Main 011850: 41-31 (56.9%) at 5 s against 60-40 at 1 s, -3.1 points (-17.7 to +11.5). But its
    5 s opponents were 128 points stronger (2,273 against 2,145). Its performance rating is 2,327
    against 2,223, +103 (-8 to +215).
  - Arm C 426M: +0.5 points (-24.3 to +26.7), with a performance rating -49 (-265 to +167).
- **The stretch that looks bad** is main 011850 with mix at 5 s on Oct 15, 08:15 to 09:15 UTC: 15-12
  and -89 rating. The next 45 games of the same network at 5 s (guarded) went 26-19, +45. A 27-game
  run has a 95% interval of 37% to 72%.
- **What 150 games can say.** They rule out a 5 s penalty larger than about 9 points of win rate
  (about 95 Elo). The mechanism in section 2 predicts at most a point or two, which would need more
  than 10,000 games per setting to see on the ladder. The box test of the same question gave the same
  answer: arm C 454M against Foul Play, 300 games each, 67.0% against 67.0%, difference -7.5 to +7.5
  (`LOSS_STUDY.md`).

How the games ended:

| Setting | Played out | Opponent forfeited | Opponent timed out | We timed out | We forfeited |
|---|---|---|---|---|---|
| 1 s | 247 | 68 | 8 | 1 | 0 |
| 5 s | 105 | 44 | 1 | 0 | 0 |

- **Opponents gave up more often at 5 s:** 30.0% against 23.5%, +6.5 (-1.8 to +15.4). They did so
  in positions the network already rated about 92% won (median), so this changes little.
- **No game was lost on the timer at 5 s.** The 4 games ever lost on time (1 mix and 3 mix-lines,
  all at 1 s) all came after an internet drop.

## 2. Mechanism checks per setting

| Setting, mode | Searched decisions | Network overruled (95%) | Overrules to a move the network gave < 2% (95%) | Iterations a decision (median) | Iterations a world (median) | Worlds a decision | Full-width depth (turns) |
|---|---|---|---|---|---|---|---|
| 1 s, mix | 10,142 | 21.6% (20.5 to 22.8) | 40.7% (38.0 to 43.4) | 328k | 170k | 2.21 | 3.58 |
| 5 s, mix | 2,701 | 25.8% (23.7 to 28.0) | 45.1% (41.0 to 49.1) | 889k | 430k | 2.21 | 3.77 |
| 5 s, mix-trust | 1,817 | 13.4% (11.5 to 15.2) | 2.9% (1.1 to 5.1) | 1,350k | 650k | 2.23 | 4.01 |

- **More search, more overrules, with the same network** (mix at both settings; 95% intervals by
  resampling games):
  - Main 011850: +3.1 points of decisions (-0.6 to +6.5).
  - Arm C 426M: +6.9 (+0.0 to +13.2).
  - The two together: +4.3 (+1.2 to +7.4). That is about 1.3 more overrules a game, at about 30
    searched decisions a game.
  - Overrules to a move the network gave under 2%: +3.1 points of decisions (+1.0 to +5.2).
  - Every mix game, all networks: +4.1 (+1.7 to +6.6).
- **The search's answer gets sharper.** The search's favourite gets 0.563 of the visits at 1 s and
  0.582 at 5 s. The veto keeps 2.41 moves on average at 1 s and 2.34 at 5 s, and the entropy of the
  shares falls from 1.23 to 1.15. The veto drops any move with under a quarter of the favourite's
  share, so a more peaked answer drops more of the network's choices.
- **The number of worlds does not change.** It is 2.21 against 2.22 sampled opponent teams a
  decision (Foul Play samples 4 early and 2 later at any budget). The extra time goes into more
  iterations inside each world: 2.5 times as many for 4.4 times the time.
- **The guarded mode** cuts overrules to 13.4%, and only 2.9% of those go to a move the network
  gave under 2%.

Fallbacks, timing and the battle timer:

| Setting | Decisions | Search fell back (error) | Played by the network after a reconnect | Decision time median / 95th pct / max | Deadline | Within 0.5 s of it / past it | Lowest timer bank: median game / worst game | Games whose bank fell under 120 s / 60 s |
|---|---|---|---|---|---|---|---|---|
| 1 s | 10,382 | 1 (0.01%) | 38 (0.4%), 2 games | 1.17 / 1.21 / 1.56 s | 2.5 s | 0 / 0 | 150 / 80 s | 2 / 0 |
| 5 s | 4,834 | 49 (1.0%), 2 games | 159 (3.3%), 4 games | 4.98 / 5.29 / 6.74 s | 6.0 s | 9 / 1 | 150 / 30 s | 7 / 2 |

- **The 5 s fallbacks are an internet outage, not the setting.** Every 5 s error and reconnect is in
  5 games of the first 5 s session (arm C 426M, Oct 15 05:17 to 06:25 UTC). The console log shows
  the laptop's connection failing: "keepalive ping timeout", then "getaddrinfo failed", at 11:04,
  11:07 and 11:45 local time. The same DNS failures appear on Oct 13 and 14, at 1 s. The 2 games
  whose timer bank fell under 60 s are 2 of these outage games.
- **Late moves.** One decision of 4,518 went past the 6.0 s deadline: 6.74 s, after waiting 0.79 s
  for a free helper. It was still played, and it is the 1 game with `late` in `results.jsonl`.
  - The margin is thinner at 5 s: the 95th percentile is 0.7 s under the deadline, against 1.3 s at
    1 s. On a busier laptop, the 5 s setting is the one that would start falling back to the
    network's move.
- **The ladder timer is no pressure.** Showdown sends "Time left: N sec this turn | M sec total" with
  every request. In the median game the bank never leaves its 150 s cap at either setting: a 5 s
  decision is paid back by the next turn.
- **Game length.** 152 s against 230 s a game (median), 6.1 against 9.0 s a turn, for the same
  number of turns (27.7 and 28.0).

## 3. The network's own win estimate after overrules

This uses `loss_study.py`'s method: the network's win estimate at the next decision minus the one at
this decision, positions matched in 10 bands of estimate and weighted as the overrules fall. Searched
decisions only. Units are points of win probability, with 95% intervals by resampling games.

| Setting, mode | Pairs (own / overrule / under 2%) | After its own move | After an overrule | After an overrule to a move it gave < 2% | Overrule minus own |
|---|---|---|---|---|---|
| 1 s, mix | 2,723 / 760 / 393 | +0.49 (+0.19 to +0.80) | -0.96 (-1.54 to -0.33) | -2.90 (-3.72 to -2.08) | -1.44 (-2.18 to -0.69) |
| 5 s, mix | 1,922 / 691 / 313 | +0.61 (+0.26 to +0.97) | -0.72 (-1.34 to -0.04) | -1.76 (-2.83 to -0.61) | -1.33 (-2.07 to -0.53) |
| 5 s, mix-trust | 1,516 / 241 / 7 | -0.02 (-0.47 to +0.44) | +0.23 (-0.66 to +1.27) | (7 pairs) | +0.25 (-0.76 to +1.40) |

- **Overrules are no better at 5 s.** The gap for 5 s mix minus 1 s mix is +0.10 points (-0.94 to
  +1.15): the value head judges a 5 s overrule as costly as a 1 s one. So the 5 s setting adds about
  1.3 overrules a game of the same quality.
- **The two settings are judged on different networks.** The 1 s pairs come only from baselines
  021725 and 022640, because the older records (main 011850 and arm C 426M at 1 s, main 024381) have
  no win estimate. The gaps by network are:
  - baselines: -0.96 and -2.15;
  - arm C 426M: -1.68;
  - arm C 454M: -1.38;
  - main 011850 with mix: -1.19.
- **The judge is the network's own value head**, which may be biased against moves it would not
  make. At the game level, overrules do not separate wins from losses:
  - The overrule rate is 22.2% in wins against 20.9% in losses at 1 s, and 21.2% against 20.4% at
    5 s.
  - A logistic model with the network and the opponent's rating held fixed gives -6 Elo per 10
    points of overrules to moves under 2% (-57 to +46).
  - The box A/B of mix against mix-trust was 60.0% against 60.0%.
  - So each overrule probably costs less than the value head says. At 1.3 extra overrules a game,
    the cost of 5 s is at most a point or two of win rate.

## 4. Why more search does not help here

1. **The extra time buys depth, not a better picture of the opponent.** The search still guesses
   about 2.2 opponent teams per decision, drawn from usage statistics, and spends 2.5 times the
   iterations inside them. Its own read of the opponent's next action (`opponent_model.py` over the
   search's view, scored after every turn) does not improve:
   - top-1 right: 48.8% of 2,800 at 1 s against 49.0% of 2,133 at 5 s, +0.2 (-3.7 to +3.9);
   - probability on what the opponent did: 33.9% against 34.2%, +0.3 (-1.8 to +2.5).
2. **A sharper answer from the same few guesses means more vetoes.** The veto's bar is relative (a
   quarter of the favourite's visits) and was a first guess at 500 ms. A longer search concentrates
   its visits, so with the same network it overrules 4.3 points more decisions. Most of the extra
   (3.1 of the 4.3 points) go to moves the network gave under 2%.
3. **The overrules it adds are no better:** the value-head gap is the same (-1.33 against -1.44).
4. **It is not the timer and not the fallbacks.** The bank stays at 150 s, one decision of 4,518
   went past the deadline, and the fallbacks were an internet outage.

**Net.** At 5 s the bot is the same bot with about 1.3 more overrules a game, each of the same
doubtful quality. Any cost is a point or two at most, far inside the +/-9 points that 150 games can
see. The measured changes point the wrong way (more overrules, no better read of the opponent), so
there is no reason to expect a longer search, spent as it is now, to help. **Keep the app at 1 s:**
it is just as strong, games are 34% shorter, and the deadline margin is twice as large.

## Fixes, ranked

1. **Spend the time on more worlds, not deeper ones.**
   - **Why.** It goes straight at the measured cause. The number of worlds is the thing the budget
     never changes. Averaging shares over 8 to 16 sampled teams instead of 2 to 4 should flatten the
     vetoes that come from one world's artefact. It should also catch the rare sets and items that 2
     guesses miss (Weakness Policy, Sitrus Berry with Cheek Pouch).
   - **No new code.** The deep arm already has the knobs (`SearchBudget.worlds_x`, `--deep-worlds`).
     A 5 s budget as 500 ms per world on 4x the worlds, one world after another in the battle's
     helper, needs 5.37 s by `minimum_deadline`. That fits the 6.0 s deadline and today's helper
     count: `ladder.py --arms mix,mix-deep --deep-search-ms 500 --deep-worlds 4 --deep-deadline 6.0`.
   - **Test.** On the box against Foul Play: 1 s against this setting, 300 games each. Then look at
     this report's read accuracy and overrule rate as secondary checks before any ladder games.
2. **Better worlds.**
   - **What.** Sample the opponent's sets from a predictor trained on the 35,919 top-player games
     instead of usage statistics, and drop the sets that the opponent's revealed moves and items rule
     out (`LOSS_STUDY.md` Fix 3, `OWN_SEARCH_IDEAS.md` idea 2).
   - **Why.** The search's read of the human is stuck at 49% top-1 and 34% on what they did at any
     budget. That ceiling is the opponent model, not search depth. This is the biggest upside and the
     biggest job.
3. **Make the veto aware of the budget.**
   - **Why.** At 5 s the same network is overruled 4 points more often.
   - **Options.** Play the guarded mode (`mix-trust`) whenever the budget is over 500 ms: at 5 s it
     overrules 13.4%, 2.9% of those to moves under 2%, and its value-head gap is +0.25 (-0.76 to
     +1.40). Its ladder effect is +80 Elo (-79 to +239), unproven. Or lower `keep` as iterations
     grow, or veto only what every sampled world rejects.
   - **Limit.** This only stops longer search costing a point or two. It does not make longer search
     help, so it ranks last.

## Reproduce

    python reports/2026-10-16/search_time.py

It reads `C:\Users\you\psbot-local\runs\ladder` and `C:\Users\you\ps-ppo-search\runs\ladder`
(`results.jsonl` and `battles\*.jsonl`) and prints every table above. It is read-only and starts
nothing.
