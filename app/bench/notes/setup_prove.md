# Set-up fixes, proved on the box: puzzles, gauntlets, a short training run

**Round 2 (2026-10-07, 19:30 to 20:10 UTC): no play rule beats today's mix in set-up positions.** 16
pre-registered rules (mix-guard, the guard with Sucker Punch at 25%, never vetoing a sure network switch,
playing the network's sure action, blends of the network and the search, a looser veto) and their
combination were scored on 6,389 set-up puzzles (round 1's 1,889 and 4,500 new ones from every 2100+ game
collected), split by game. On the choose half only `--setup-blend 8` kept the agreed set, at +0.2 points
(-0.8, +1.2) over mix, short of the +1 the plan asked for; so the winner is "none", the confirm half was not
scored, and Look Play stays as it is. Details: `bench/setup_puzzles/ROUND2.md` on branch setupfix. The
flags (`--setup-switch-trust`, `--setup-net-trust`, `--setup-blend`, `--setup-keep`, `--guard-sucker`) are
off by default; `tests/test_setup_round2.py` checks that off is today's play.

**After the review (2026-10-07, ~20:30 to 21:10 UTC).** With the 5,000 ms oracle as the only judge of the
agreed set (round 1's puzzles of the choose half, 474), `--setup-blend 8` still holds it, -0.2 (-2.1, +1.7),
and `--setup-blend 4` does not, -3.6 (-6.0, -1.2). By player instead of by game the intervals barely move
(blend 8: +0.22, (-0.81, +1.24)). So blend 8 is the one candidate worth a ladder look, and nothing ships. To
keep measuring it in real games: `launchers/laptop/Look Play Blend AB.cmd` runs this worktree with
`--arms mix,mix-blend8` (the arms `mix-blend4` and `mix-blend8` are `search_player.ARM_FIXES` entries:
`--setup-blend` on that arm only) and ends with `bench/setup_ladder_scan.py --by-arm`, each arm's record
and set-up sweep losses with 95% Wilson intervals and the difference with a 95% Newcombe interval. On
round 1's 100 ladder games split into two fake arms at random, the arms already differ by +16 points of
wins (-3, +34): read the A/B only through its intervals.

2026-10-07, box, ~16:46 to ~18:40 UTC. Owner's network `learner_update_011850.pt` (sha256 edd11dd6...)
for every puzzle and gauntlet number; the training arms start from a copy of the live newest
`learner_update_018725.pt` (release r5). Settings named as in `bench/notes/setup_fix.md`:

- **a** = `--veto-fix` (no third action, Tera only when one side picked it, a sure attack in KO range is not vetoed for our own set-up move)
- **b** = `--setup-defer` (in a set-up position the search's best)
- **c** = `--think-longer 2000` (a 2,000 ms search on set-up turns)

**Answer first (revised after the review, 2026-10-08): no play-time setting measured better, so
Look Play stays as it is.**

- **a** (`--veto-fix`) is level with today's mix on the one column the engine does not judge: top
  player -0.2 (-0.9 to +0.4). Its +1.1 on the oracle's best is the engine agreeing with itself.
- **b** (`--setup-defer`, alone or with a) plays the 500 ms search's own pick in set-up positions
  (401 of the 405 decisions a+b changes). Its one fair-looking gain, +2.3 on the agreed set, is
  exactly what the 500 ms search alone scores there (+2.3, -0.1 to +4.8): choosing the puzzles where
  the top player and the engine agree rewards copying the engine. On the other 907 puzzles a+b is
  -5.4 (-8.0 to -2.8) against top players; over all puzzles -1.4 (-3.1 to +0.3); in 242 bot games
  -1.2 (-9.4 to +7.1). Six settings times four columns were looked at on the same puzzles, and the
  one positive interval starts at +0.1. So b is **an unproven bet, not a measured gain**.
- **The only test left for b is people:** `ladder.py --arms mix,mix-fixdefer` (launcher
  `launchers/laptop/Look Play Setup AB.cmd`, which now alternates today's mix and a+b game by game;
  before the review it alternated a and a+b and had no control). 50 games per arm show only a
  difference of about 25 points.
- **Training:** the proposed `--search-teacher-setup-ms 1000 --search-teacher-setup-weight 3`
  starves the teacher (labels -94%). `--search-teacher-setup-weight 3` alone gives set-up turns 38%
  more labels at no cost in speed or latency; since the review its weight is normalised, so it moves
  the teacher's emphasis to set-up turns without making the whole term stronger (section 3). Its
  effect on play is not shown.

The sections below are the proof as it was run; where the review changed a reading, it says so.

## 1. Puzzles: today's mix against each setting (1,889 scored set-up puzzles, network 011850)

Judge: Foul Play at 5,000 ms (the oracle). "Sound" = in its sound set, "best" = its best, "top
player" = what the 2100+ player did. The first three columns are Wilson 95% intervals; the paired
columns are the difference against mix on the same puzzles, in points, with a 95% interval and the
counts (decisions it fixed / broke).

| setting | sound | = oracle best | = top player | sound vs mix | best vs mix | top player vs mix |
|---|---|---|---|---|---|---|
| mix (today) | 86.7 (85.1-88.1) | 65.3 (63.2-67.4) | 56.2 (54.0-58.4) | | | |
| a | 87.1 (85.5-88.5) | 66.4 (64.3-68.5) | 56.0 (53.8-58.2) | +0.4 (-0.1, +0.9) 15/7 | **+1.1 (+0.4, +1.9)** 37/16 | -0.2 (-0.9, +0.4) 18/22 |
| b | 90.0 (88.6-91.3) | 71.4 (69.3-73.4) | 54.8 (52.5-57.0) | **+3.4 (+2.2, +4.6)** 101/37 | **+6.0 (+4.3, +7.8)** 204/90 | -1.4 (-3.1, +0.2) 115/142 |
| a+b | 90.1 (88.7-91.4) | 71.6 (69.5-73.6) | 54.8 (52.6-57.1) | **+3.4 (+2.2, +4.7)** 105/40 | **+6.2 (+4.5, +8.0)** 212/94 | -1.4 (-3.1, +0.3) 120/146 |

The agreed set (982 puzzles where the top player played the oracle's best; one label, both judges):

| setting | right | vs mix |
|---|---|---|
| mix | 81.4 (78.8-83.7) | |
| a | 81.7 (79.1-84.0) | +0.3 (-0.5, +1.1) |
| b | 83.5 (81.1-85.7) | +2.1 (-0.1, +4.3) |
| a+b | 83.7 (81.3-85.9) | **+2.3 (+0.1, +4.5)** |

**The other half (added after the review; `prove.py` prints it next to the agreed set now).** Top-player
agreement, paired against mix: on the agreed set, on the 907 puzzles where the top player and the
oracle disagree, and on the 526 where the top player is outside the oracle's sound set.

| setting | agreed (982) | disagreed (907) | top player outside the sound set (526) |
|---|---|---|---|
| mix (today), level | 81.4 | 29.0 | 18.1 |
| a | +0.3 (-0.5, +1.1) | -0.8 (-1.9, +0.3) | -0.4 (-1.4, +0.7) |
| b | +2.1 (-0.1, +4.3) | **-5.3 (-7.8, -2.8)** | **-6.5 (-9.0, -3.9)** |
| a+b | +2.3 (+0.1, +4.5) | **-5.4 (-8.0, -2.8)** | **-6.5 (-9.1, -3.8)** |
| search 500 ms alone | +2.3 (-0.1, +4.8) | -7.9 (-10.8, -5.1) | -8.8 (-11.7, -5.8) |
| network alone | -13.7 (-16.2, -11.1) | **+6.8 (+4.1, +9.6)** | **+14.4 (+10.8, +18.1)** |

Every setting that leans on the search gains on the agreed set and loses the same way where the top
player and the engine part, and a+b sits between today's mix and the search alone on both. Where top
players leave the engine, the network alone is with them far more often than the search. Which side
is right in those positions the puzzles cannot say: only games against people can. Box:
`~/dev/setup-final-out/prove.json` and `prove.txt` (every published number reproduced).

**c needs a 2,000 ms answer per puzzle.** Six helpers asked it for 10 minutes
(`oracle.py --prefix long --oracle-ms 2000 --search-ms 0`, one engine thread): 758 puzzles, a random
part of the set (episodes in the file's random order). It fires on 753 of them (all set-up
positions), and its best differs from the 500 ms best on 160 (21%). On those 753:

| setting | sound | = oracle best | = top player | agreed set (396) | vs mix, agreed | vs a+b, agreed |
|---|---|---|---|---|---|---|
| mix | 86.3 | 64.7 | 58.7 | 83.1 | | |
| a+b | 89.6 | 72.4 | 57.6 | 88.1 | +5.0 (+1.6, +8.5) | |
| c alone | 89.6 | 67.5 | **54.8** (vs mix -3.9, -6.2 to -1.5) | 82.1 | -1.0 (-3.9, +1.9) | |
| a+b+c | 92.6 | 74.6 | 56.8 | 88.9 | +5.8 (+1.8, +9.8) | +0.8 (-2.7, +4.2) |

- **a** is safe and small: one point closer to the oracle's best, no loss against the top player.
- **b** carries the effect. It moves the mix toward the engine (+6 on the oracle's best) and is 1.4
  points further from the top player (interval -3.1 to +0.2). Where both judges agree it gains 2.3.
  What it changes (405 decisions, a+b vs mix): the top player's own action goes from 146 matched to
  120. The losses are mostly an attack the top player made replaced by a switch (35, against 12 the
  other way) or by a passive move (17 against 7); the gains are mostly a switch replaced by the attack
  the top player made (23 against 11). So in set-up positions the search leaves more often than top
  players do. *Review:* the agreed-set gain is the search's own (the 500 ms search alone gains the same
  2.3), and on the disagreed set b loses 5.3; see the table above. It is not a measured gain.
- **c** on its own moves the plain veto toward the engine and away from the top players (-3.9 points,
  significant). On top of a+b it adds +2.9 sound (shared judge) and +0.8 on the agreed set (noise).
  The 2,000 ms search is closer to the 5,000 ms oracle partly because it is the same engine searched
  longer; the agreed set is the fair test, and there c adds nothing measurable.
- Not scored: the sure-attack part of a (a puzzle holds no stats).

Code: `bench/setup_puzzles/prove.py` (branch setupfix). Box: `~/dev/setup-prove/prove.json`,
`~/dev/setup-prove/puz-long/long-*.jsonl`.

## 2. Gauntlets: mix against a+b (`--arms mix,mix-fixdefer`), owner's network 011850

The best setting on the fair puzzle comparison was taken to be a+b (the review showed that comparison leans on the engine; see the answer at the top). c was left out of the games: it adds nothing
measurable on the agreed set and costs about 5 s on set-up turns, which would have cut the number of
games. Local server (port 10300), `bench/gauntlet.py`, search 500 ms, deadline 2.5 s, rule veto,
first and second player balanced by the picker within each arm. Wins with Wilson 95% intervals.

| opponent | mix (today) | a+b | a+b minus mix | games a+b changed / their wins |
|---|---|---|---|---|
| Foul Play, fair compute (450 ms; its iterations 0.92 of ours at the median, 0.89 at the mean) | 19-13, 59.4% (42.3-74.5); p1 9/16, p2 10/16 | 19-12, 61.3% (43.8-76.3); p1 10/16, p2 9/15 | +1.9 (-22.2, +26.1) | 25 of 31, won 15 (60.0%) |
| set-up test bot | 42-2, 95.5% (84.9-98.7); p1 21/22, p2 21/22 | 40-3, 93.0% (81.4-97.6); p1 21/22, p2 19/21 | -2.4 (-12.2, +7.4) | 33 of 43, won 30 (90.9%) |
| network alone (011850) | 21-25, 45.7% (32.2-59.8); p1 9/23, p2 12/23 | 22-24, 47.8% (34.1-61.9); p1 12/23, p2 10/23 | +2.2 (-18.2, +22.6) | 42 of 46, won 19 (45.2%) |
| **pooled** (inverse variance) | | | **-1.2 (-9.4, +7.1)** | 100 of 120 |

- **Scaled down.** 63, 87 and 92 games per opponent (31 to 46 per arm), not 200 per arm: the six
  helpers allow about 3 games a minute against Foul Play and 7 to 10 against the bots, and the whole
  job had about 120 minutes. 200 per arm on all three is about 4 hours of games. So the gauntlet only
  rules out a change of more than about 10 points pooled, and of more than 20 against one opponent.
- **The fixes act in almost every game:** a+b played another action than the plain veto in 100 of
  its 120 games (326 decisions: the search's best in a set-up position 204, no third action 114, a
  sure attack in KO range 7, Tera 1). Games without a change are short, easy games, so "won with a
  change" against "won without" is not a comparison.
- Clean: 0 fallbacks other than single-option turns, 0 late decisions, 0 helper expiries or exits,
  0 invalid orders. Foul Play: 2,063 decisions answered, 53 under 60,000 iterations.
- The first Foul Play start stopped at once (`--helpers 3 --parallel 3`: our side needs one helper
  more than games); it was run again as `fp2` (4 ours + 2 Foul Play's, 3 games at a time).

Code: `search_player.ARM_FIXES` (arms `mix-fix`, `mix-defer`, `mix-fixdefer`, `mix-fixdefer-long`,
`mix-fix-long`), `bench/gauntlet.py --think-longer`, `bench/setup_gauntlet_report.py`,
`bench/setup_prove/*.sh`. Box: `~/dev/setup-prove/g/{fp2,nt,su}`, `~/dev/setup-prove/gauntlet.json`.

## 3. Training: the set-up teacher focus on and off (branch setupfix)

Three arms, each its own Ray cluster (`RAY_ADDRESS=local`, `RAY_TMPDIR=/tmp/r_setupprove`; its own
GCS server and session, the live run's cluster untouched), 2 Showdown servers (ports 10310-10311),
128 battles, the live run's steps per update (36,864), minibatch 768, resumed from a copy of the live
newest `learner_update_018725.pt`, r5's flags (`--keep-history --foulplay-opponent-share 0.10
--foulplay-search-ms 60 --foulplay-helpers 1 --search-teacher-helpers 2 --own-tera-type
--obs-abilities all --search-teacher-switch-weight 3 --lr 3e-05 --lr-hold-steps 1544335
--lr-total-steps 4340147`), code setupfix (b2fd4e3; r5 is cd72859, see section 5). GPU through `~/dev/gpu.lock`, one arm at a time, each under 17 minutes.

- **off**: the flags above, with `--search-teacher-setup-weight 1.0001` only so that set-up turns are
  counted (a 0.01% weight: nothing measurable changes).
- **on**: `--search-teacher-setup-ms 1000 --search-teacher-setup-weight 3`, the start the build job
  proposed. Stopped at 15 updates: its result was already clear.
- **weight only**: `--search-teacher-setup-weight 3` (set-up turns searched as long as the others,
  asked first and counted 3 times). Added when "on" turned out to starve the teacher.

| per update (mean) | off (20 updates) | on (15) | weight only (15) |
|---|---|---|---|
| teacher labels | 324.6 | **20.6 (-94%)** | 316.1 (-3%) |
| set-up turns labelled | 80.5 (1,610; 25% of labels) | **18.2** (273; 88% of labels) | **110.8** (1,662; 35% of labels) |
| set-up label mass (labels x weight) | 80.5 | 54.6 | **332** |
| teacher answer latency, mean | 2.55 s | **6.37 s** (79% over 5 s) | 2.59 s |
| questions dropped busy / late / expired (whole arm) | 3,326 / 0 / 0 | 16,464 / 294 / 28 | 2,104 / 25 / 0 |
| kl (whole policy) | 0.00181 | 0.00181 | 0.00185 |
| kl on labelled steps (kl_lab) | 0.0026 | 0.0196 | 0.0025 |
| agree (network's likeliest in the search's set, labelled steps) | 0.827 | 0.746 | 0.849 |
| steps per second / seconds per update | 774 / 47.6 | 774 / 47.5 | 770 / 47.8 |
| learner seconds per update | 13.3 | 13.3 | 13.5 |
| network vs Foul Play pairs (rollouts) | 55.0% of 171 | 55.0% of 120 | 53.6% of 125 |

**The proposed "on" setting starves the teacher.** A 1,000 ms question holds a helper ten times as long
as an ordinary 100 ms one (the mean answer's iterations rose from 93,403 to 400,882), and weight 3 lets set-up
questions wait three times longer, so the two helpers queue: answers come in after 6.4 s on average,
the busy drops go up five-fold, and the labels fall from 325 to 21 per update. Even the set-up turns
get fewer labels (18 a update against 81 in off), so the weight of 3 does not make up for it. The
trainer itself is not slowed (774 steps a second in every arm): the whole cost lands on the helpers.
Set-up turns are common: a quarter of the labelled decisions in off (the opponent at +1 or more).

**Weight only does what was wanted, at no cost.** Set-up turns get 38% more labels (111 against 81
an update), each counted 3 times, so their share of the teacher's term goes from a quarter to about
60%, while all labels fall only 3% and the answer latency, drops, speed and KL stay
where off has them.

**Agreement on the set-up puzzles, the networks alone, before and after** (`bench/setup_puzzles/netcmp.py`,
stage2 build of the same 1,901 puzzles, `--own-tera-type on --obs-abilities all`; 1,889 scored;
paired against the start):

| network | sound | = oracle best | = top player | agreed set (982) | moved: KL from start / picks changed |
|---|---|---|---|---|---|
| start (live 018725) | 70.2 (68.1-72.2) | 50.0 (47.7-52.2) | 52.5 (50.2-54.7) | 67.9 (64.9-70.8) | |
| off, 15 updates | 70.2 (+0.0; -0.6, +0.6) | 50.2 (+0.3) | 52.7 (+0.2) | 68.4 (+0.5; -0.5, +1.5) | 0.0085 / 62 |
| off, 20 updates | 69.8 (-0.4; -1.1, +0.2) | 49.9 (-0.1) | 52.2 (-0.3) | 67.6 (-0.3) | 0.0113 / 77 |
| on, 15 updates | 69.7 (-0.5; -1.2, +0.2) | 49.3 (-0.7; -1.4, +0.1) | 52.4 (-0.1) | 67.3 (-0.6; -1.6, +0.3) | 0.0111 / 84 |
| weight only, 15 updates | 70.4 (+0.2; -0.4, +0.7) | 50.2 (+0.2) | 52.7 (+0.2) | 68.1 (+0.2; -0.8, +1.2) | 0.0095 / 64 |

Fifteen to twenty updates move the network very little on these positions (60 to 85 of 1,889 picks
change) and no arm is outside the noise. "on" leans the wrong way on every column, which fits a
term fed by few, late labels; weight only and off are level. The puzzles cannot say yet whether the
set-up focus teaches the network anything: that needs hundreds of updates, and the network-alone
score on the puzzles is the cheap check along the way (`netprobs.py` + `netcmp.py`, about a minute
per checkpoint).

**Changed after the review: the set-up weight is normalised.** The arms above ran with set-up rows
simply counted 3 times. A set-up turn is any opponent boost of +1 or more, so 34% of the labels in
"weight only" were marked, and the whole teacher term was 0.658 + 3 x 0.342 = 1.68 times as strong:
weight 3 also raised the teacher's coefficient by about 70%. `ppo_core.search_teacher_term` now
divides the set-up factors by their mean over each minibatch's labelled rows: at 34% marked, a
set-up row weighs 1.79 and any other row 0.60, and the term's size is what it was. Weight 1 is the
term of before, bit for bit (the code path is unchanged), and the switch weight is not normalised,
as before. The queue part of the flag (set-up turns asked first) is unchanged, so the label counts,
latency and drops of "weight only" hold; its KL and puzzle rows were measured unnormalised. Tests:
`tests/test_setup_teacher.py` checks the normalised term by hand (43 passed, 2 skipped with
`test_search_teacher.py`, on the box).

Two more things the review found: on stage2 and r5 only boosts mark a set-up turn ("by the read" was
0 in every arm, because stage2's bridge returns no read of the opponent), and the mark (+1 or more,
whatever caused it) is wider than the play side's set-up position (+2, or +1 right after a set-up
move). `--search-teacher-setup-boost 2` narrows the mark to +2; it was not measured.

Box: `~/dev/setup-prove/arms/{off,on,wonly}` (logs, checkpoints), `~/dev/setup-prove/arms.json`,
`~/dev/setup-prove/netcmp.json`, puzzles `~/dev/setup-prove/puz-s2` (stage2 build).
Code: `bench/setup_prove/run_arm.sh`, `arms.sh`, `arms2.sh`, `arm_summary.py`, `bench/setup_puzzles/netcmp.py`
(branch setupfix).

## 4. What goes where (revised after the review)

| change | where | verdict | why |
|---|---|---|---|
| a `--veto-fix` | Look Play | **no change** | not measured better: top player -0.2 (-0.9, +0.4), agreed set +0.3 (-0.5, +1.1); its +1.1 on the oracle's best is the engine's own judgement; never played alone in games |
| b `--setup-defer`, with a | Look Play | **no change; an unproven bet** | in set-up positions it is the 500 ms search (401 of 405 changes); agreed set +2.3 = the search alone's +2.3; disagreed set -5.4 (-8.0, -2.8); top player overall -1.4 (-3.1, +0.3); games -1.2 (-9.4, +7.1). Measure it against people with `--arms mix,mix-fixdefer` if wanted |
| c `--think-longer 2000` | not now | no | adds nothing on the agreed set (+0.8 over a+b, noise), alone it moves away from top players (-3.9, significant), and costs about 5 s on set-up turns |
| mix-guard, the network's sure switch kept | never | no | refuted earlier on the puzzles (z -4.5 and -4.3 against top players) |
| `--search-teacher-setup-ms 1000 --search-teacher-setup-weight 3` | next release | **no** | starves the teacher: labels -94%, set-up labels -77%, answers 6.4 s late |
| `--search-teacher-setup-weight 3` (normalised, no longer search) | next release after r5 | **can go in; effect not shown** | set-up labels +38% (each weighed 1.79, the others 0.60, the term's size kept); all labels -3%; no change in speed or latency; its effect on play is not shown (15 updates, unnormalised) |

**Look Play now: no change.** The Desktop launcher keeps `--player mix` without the fix flags. If you
want the ladder to measure b against people, run `launchers/laptop/Look Play Setup AB.cmd` (after
`git -C C:\Users\you\ps-ppo-predict merge --ff-only setupfix-play`, when no ladder game is running):
`--arms mix,mix-fixdefer`, today's mix against a+b game by game; the result line of each game names
its arm and the run ends with a "by arm" table. With every flag off the merged code plays exactly what
d7ace06 played (fuzzed on 3.84M cases by the review). 50 games per arm show only a large difference.
`Look Play Setup.cmd` (a+b only) stays on the branch for after such a test; it is not a recommendation.

**Next training release after r5:** `--search-teacher-setup-weight 3` and nothing else of the set-up
teacher. Watch the `[SearchTeacher]` line's "set-up turns ... labelled" and the network-alone puzzle
score per checkpoint against the r5 line (`netprobs.py` + `netcmp.py`).

## 5. Problems and what was scaled

- **Review fixes (2026-10-08):** the verdict above; `prove.py` prints the disagreed set; the A/B
  launcher has today's mix as its control; the set-up weight is normalised; with `--think-longer`, a
  second search thread is given back when the network's inference raises; the sure-attack KO
  estimate counts no KO at full HP, through a known Assault Vest or Eviolite (or an unseen item on a
  Pokemon that still evolves), or into an ability that could stop it (`search_player.ko_doubt`); the
  `--veto-fix` help says that the Tera form is played when the search vetoed only the plain form.
  Not fixed (c is not recommended): a late long search asked before the move still leaves that turn
  to the network alone, not to a 500 ms answer.

- **Gauntlets scaled to 31-46 games per arm and opponent** (242 games), not 200: about 4 hours of
  games at six helpers did not fit. They rule out large effects only.
- **c was not played in games**, only scored on 758 puzzles (a random 40% of the set) with one
  engine thread.
- **"on" stopped at 15 updates**, "weight only" was run for 15; off for 20 (scored at 15 and 20).
- **The judge of the puzzles is Foul Play**, the same engine as our search; any setting that plays
  the search more scores better on "sound" and "oracle best" partly by shared judgement. The agreed
  set and "= top player" are the fair columns.
- **Training ran setupfix (b2fd4e3)**, not r5's cd72859. cd72859 adds review fixes (`obs_abil.py`:
  absorb heals; `hybrid.py`; restart checks); the trainer, learner, PPO and teacher files are the same
  at both commits apart from setupfix's teacher change, and all three arms ran the same code.
- The first Foul Play phase stopped at once on a helper-count check (no games) and was run again.
- Nothing of mine is left running on the box (checked after every arm and at the end); the live run,
  its services, ports and `~/runs/main` were only read (a checkpoint copied). Nothing was deleted.
