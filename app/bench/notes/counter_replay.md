# Round 3 counter rules: the ladder replay

Run 2026-10-08 on the box, branch `counterfix` (the rules of d573f7e, arm `mix-counter`: "use the
counter" and "don't hit the boosted defence"). Two sets of the owner's own ladder games, all played
by mix (network 011850 + 500 ms search):

- **this morning**: 49 games, `~/dev/ladder-1008/data` (29-20, 11 set-up sweep losses)
- **round 1**: 100 games, `~/dev/setup-ladder/data` (60-40, 14 set-up sweep losses)

Each battle is fed through the real search player with the decision notes' own network and search
numbers (`bench/counter_replay.py`), so mix decides exactly as it did live, and the rules see the
board poke-env built. Then every decision a rule changed is asked again of the search, 2 repeats,
once at 500 ms (round 1's way) and once at the live search's own effort (`bench/counter_rerun.py`).

## Results first

- **The rules change few decisions in the sweep losses, and as many games among the wins.** They change
  a decision in 8 of the 25 sweep losses (32%, 95% interval 17 to 52%) and in 22 of the 89 wins (25%,
  17 to 35%). Per decision: 3.1 per 100 in sweep losses (2.2 to 4.5), 1.4 per 100 in wins (1.0 to 2.0),
  none in the 10 losses without a set-up.
- **They answer the actual sweeper in few games.** Of the 8 sweep losses with a change, the first change
  is aimed at the Pokemon that swept in 4, and answers it directly (removes its boosts or locks it) in
  2: game 8092 (Whirlwind on +1/+1 Arceus) and game 1202 (Encore on Scrafty, from turn 29). In the other
  4 games the first change is aimed at a different Pokemon that set up earlier. Example: game 7639
  switches to Swampert for Yawn against Minior, but Florges swept.
- **The cap is the team, not the rule.** In 17 of the 25 sweep losses nothing changes. In 10 of them the
  team had no counter at all: no phazing move, Haze, Encore, Taunt, sleep move or Unaware. In the other 7
  the counter could not work: it would be knocked out in two hits, Sleep Clause applied, the boost had
  already landed (Taunt), or the last move was not a set-up move (Encore).
- **The search scores the rules' moves below mix's, and much further below in wins.** At the live
  search's effort, the rule's action scores 4.3 points of win chance below mix's in sweep losses (95%
  interval over games 1.3 to 8.0 below) and 8.0 points below in wins (5.4 to 10.9 below). None of the
  27 changes in sweep losses is better at that effort. Round 1's 500 ms rerun gives the same picture
  (sweep losses 4.0 below, wins 9.2 below).
- **The switches are the costly part.** The switch to a benched Pokemon that holds a counter is 23 of
  the 34 changes in wins. The search scores it 9.3 points below mix (5.5 to 13.2) in wins and 7.5 below
  in sweep losses. Played counter moves score 2.9 below in sweep losses and 5.1 below in wins. "Don't hit
  the boosted defence" swaps are about even (0.6 below, 5 changes in 2 sweep losses).
- **With the switches off, the rule is more targeted but the upside is still small.** (This variant is
  a replay only, `counter_hp` 1.01, no arm built.) It changes 5 of 25 sweep losses and 9 of 89 wins (10%,
  5 to 18%). In sweep losses the search scores it 2.2 below mix (0.7 above to 3.4 below); in wins 7.4 below.
- **Checks:** the replayed mix played what was played live on 1,473 of 1,473 searched decisions (this
  morning) and 3,042 of 3,043 (round 1). The reruns gave no errors, and every changed action was among
  the search's options. The live-effort pass reproduced the live iterations (median 342k and 316k) and
  the same number of sampled worlds on all 134 answers.

## What it means for the ladder A/B

Rough arithmetic, not a measurement. Suppose each of the 2 to 4 sweep losses where a rule answers
the sweeper turned into a win. That is 2 to 4 games in 149, or +1.3 to +2.7 points of win rate. Now
suppose the search is right about the 22 wins the rules touch, at 7.6 points each for the first
change. That costs about 1.7 games in 149, or about -1.1 points. The net is near zero either way.

A ladder A/B cannot resolve an effect of that size soon. With 100 games per arm, the 95% interval on
the win-rate difference is about +-13.6 points, and on the sweep-loss rate about +-9.7 points.
Bringing the sweep-loss interval to +-5 points takes about 380 games per arm. Bringing the win-rate
interval to +-5 points takes about 740 per arm.

My reading: as built, `mix-counter` is more likely to cost a little than to gain. If the A/B runs, an
arm with the switches off is the better candidate (it needs a new arm, see below). The larger lever
on set-up sweeps is not in play rules: 17 of 25 sweep losses had no counter the rule could use.

One caution on that reading. The search is the thing the rule overrides, so "the search scores it
lower" is not proof the move is worse. Game 1202 is the clearest case. At turns 29 to 37 the network
put 0.91 to 1.00 on Encore against a Scrafty at +2 to +6 from Bulk Up, the search preferred Strength Sap,
mix followed the search, and Scrafty swept 4. The search still scores Encore 3 to 6 points lower in
every rerun. Only games can settle who is right there.

## Every set-up sweep loss with a change (8 of 25)

The first change is the only one played in the game's own position. Any later change follows a history
the rule would already have altered. "Gap" is the search's score of the rule's action minus mix's, in
points of win chance, for the two repeats. Verdicts were set before the runs: "clearly worse" means 5
or more points below in both repeats; "worse" means 2 or more below in both; "better" means 2 or more
above in both; anything else is "about even".

| set | game | swept by (set-up turn, KOs) | changes (of set-up positions looked at) | first change: rule instead of mix | aimed at | answers the sweep? | gap 500 ms | gap at live effort | later changes (gap 500 ms) |
|---|---|---|---|---|---|---|---|---|---|
| morning | 1202 | Scrafty (T27, 4) | 11 of 18 | T5 switch Jumpluff (Encore) instead of switch Sneasler | Scrafty, the sweeper, before it set up (read 59%) | partly: brings in an Encore holder | +1/+0 even | +1/+1 even | T27 switch Jumpluff -2/-2; T29 to T37 Encore 9 times, -3 to -6 each (Strength Sap kept by mix) |
| morning | 8092 | Arceus (T7, 2) | 1 of 3 | T8 Whirlwind instead of Sticky Web | Arceus-Fairy at SpA +1 SpD +1, the sweeper | yes: drags it out, boosts gone | +11/+5 better | +1/+3 even | - |
| morning | 1226 | Regidrago (T35, 4) | 1 of 3 | T3 switch Milotic (Dragon Tail) instead of U-turn | Raikou at SpA +1 SpD +1, not the sweeper | no: another set-up Pokemon | -3/-3 worse | -6/-7 clearly worse | - |
| morning | 4869 | Dragonite (T31, 2) | 1 of 5 | T13 Roar instead of Earthquake | Pecharunt at SpA +2, not the sweeper | no: another set-up Pokemon (it took 1 KO) | +1/+2 even | -0/+1 even | - |
| morning | 9987 | Necrozma (T19, 4) | 3 of 13 | T12 switch Archaludon (Dragon Tail) instead of Weather Ball | Necrozma, the sweeper, before it set up (read 65%) | partly: brings in a phazer (it later went Fairy: Dragon Tail failed) | +0/-4 even | -1/+1 even | T28, T29 Body Press instead of Draco Meteor (0% estimated): +0/+0 |
| round 1 | 7454 | Dondozo (T3, 3) | 3 of 22 | T12 Hydro Pump instead of Flip Turn | Dondozo at Atk +4 Def +4, the sweeper | partly: hits the unboosted side (22% vs 3% estimated) | +1/+1 even | -0/+0 even | T17 Hydro Pump -3/-3, T22 Hydro Pump -1/-1 |
| round 1 | 5357 | Kingambit (T28, 3) | 6 of 10 | T7 switch Clodsire (Unaware) instead of Rapid Spin | Altaria at Atk +1 Spe +1, not the sweeper | no: another set-up Pokemon | -10/-9 clearly worse | -10/-8 clearly worse | T8 and T9 switch Clodsire -10/-15 and -17/-18; then against Kingambit: T10 switch Toedscruel -4/-4, T11 and T12 switch Clodsire -4/-4 and -7/-9 |
| round 1 | 7639 | Florges (T16, 4) | 1 of 17 | T5 switch Swampert (Yawn) instead of Thunderbolt | Minior before Shell Smash (read 62%), not the sweeper | no: another set-up Pokemon | -15/-10 clearly worse | -12/-18 clearly worse | - |

Game 8092 is a close call that the search cannot settle. Four reruns of that one decision gave gaps of
+11/+5 and -6/-2 at 500 ms, and +1/+3 and +4/-0 at the live effort.

## The 17 sweep losses without a change

| set | game | swept by | why nothing changed |
|---|---|---|---|
| morning | 4910 | Chimecho (6 KOs) | no counter on the team (17 set-up positions) |
| morning | 2447 | Ceruledge (4) | no counter on the team |
| morning | 3496 | Scrafty (2), Carbink (2) | no counter on the team (29 positions) |
| morning | 4136 | Toxtricity (2) | no counter on the team; twice an attack was estimated to KO |
| morning | 0365 | Revavroom (2) | Vivillon's Sleep Powder: Sleep Clause (the clause check counts any sleeping foe, a Rest too) |
| morning | 4151 | Frosmoth (6) | Reshiram's Dragon Tail: Reshiram would be knocked out in two hits |
| round 1 | 8716 | Fezandipiti (2) | Milotic's Haze: two-hit KO; mix played a counter twice itself |
| round 1 | 8921 | Chimecho (4) | Encore: its last move was not a set-up move; 4 times an attack was estimated to KO |
| round 1 | 3541 | Lapras (3) | no counter on the team |
| round 1 | 5275 | Calyrex (4) | Lumineon's Encore: two-hit KO, or the last move was not a set-up move |
| round 1 | 5539 | Cinccino (2) | no counter on the team |
| round 1 | 3018 | Annihilape (2) | no counter on the team |
| round 1 | 2867 | Florges (3) | no counter on the team |
| round 1 | 0184 | Calyrex (2) | no counter on the team |
| round 1 | 3715 | Skarmory (4) | Mightyena's Taunt: the boost had landed (52 positions) |
| round 1 | 0196 | Revavroom (3) | Pyroar's Taunt: the boost had landed; 7 times an attack was estimated to KO |
| round 1 | 8359 | Porygon-Z (4) | no counter on the team |

## Changes per kind of game (95% Wilson intervals)

| set | kind of game | games | decisions | changed | per 100 decisions | games with a change |
|---|---|---|---|---|---|---|
| both | set-up sweep losses | 25 | 863 | 27 | 3.1 (2.2 to 4.5) | 8 (32%, 17 to 52) |
| both | other losses with a set-up | 25 | 947 | 6 | 0.6 (0.3 to 1.4) | 5 (20%, 9 to 39) |
| both | losses without a set-up | 10 | 307 | 0 | 0.0 (0.0 to 1.2) | 0 (0%, 0 to 28) |
| both | wins | 89 | 2,399 | 34 | 1.4 (1.0 to 2.0) | 22 (25%, 17 to 35) |
| morning | set-up sweep losses | 11 | 373 | 17 | 4.6 (2.9 to 7.2) | 5 (45%, 21 to 72) |
| morning | wins | 29 | 794 | 5 | 0.6 (0.3 to 1.5) | 3 (10%, 4 to 26) |
| round 1 | set-up sweep losses | 14 | 490 | 10 | 2.0 (1.1 to 3.7) | 3 (21%, 8 to 48) |
| round 1 | wins | 60 | 1,605 | 29 | 1.8 (1.3 to 2.6) | 19 (32%, 21 to 44) |

This morning's set looks well targeted (4.6 vs 0.6 per 100) because of one game: 1202 alone holds 11
of its 17 sweep-loss changes. Round 1, twice the size, has the rules acting as often in wins as in
sweep losses.

## The search asked again: the rule's action against mix's, both sets

Mean gap in points of win chance. The 95% interval comes from a bootstrap over games, because the
changes within one game are not independent. Small cells (1 to 3 games) have intervals that mean
little.

| asked at | kind of game | kind of change | changes | clearly worse | worse | about even | better | mean gap (95% interval) |
|---|---|---|---|---|---|---|---|---|
| live effort | sweep losses | counter move played | 11 (3 games) | 0 | 9 | 2 | 0 | -2.9 (-3.8 to +1.6) |
| live effort | sweep losses | switch to a holder | 11 (5 games) | 7 | 1 | 3 | 0 | -7.5 (-10.6 to -0.9) |
| live effort | sweep losses | other defence | 5 (2 games) | 0 | 1 | 4 | 0 | -0.6 (-1.0 to +0.0) |
| live effort | sweep losses | all | 27 | 7 | 11 | 9 | 0 | **-4.3 (-8.0 to -1.3)** |
| live effort | wins | counter move played | 8 | 0 | 6 | 2 | 0 | -5.1 (-7.5 to -2.9) |
| live effort | wins | switch to a holder | 23 | 15 | 5 | 2 | 1 | -9.3 (-13.2 to -5.5) |
| live effort | wins | other defence | 3 (1 game) | 1 | 0 | 1 | 1 | -5.5 |
| live effort | wins | all | 34 | 16 | 11 | 5 | 2 | **-8.0 (-10.9 to -5.4)** |
| live effort | other losses with a set-up | all | 6 | 3 | 1 | 2 | 0 | -8.1 (-14.4 to -1.3) |
| 500 ms | sweep losses | all | 27 | 8 | 10 | 8 | 1 | -4.0 (-6.9 to -0.3) |
| 500 ms | wins | all | 34 | 19 | 9 | 6 | 0 | -9.2 (-12.2 to -6.6) |
| 500 ms | other losses with a set-up | all | 6 | 3 | 2 | 1 | 0 | -9.1 (-15.8 to -1.2) |

First changes only (one per game, the clean counterfactual), live effort: sweep losses -3.4 over 8
games (-7.7 to +0.2), wins -7.6 over 22 games (-11.2 to -4.5).

**By trigger** (mean gap at live effort). The switches fired on a set-up read alone, with nothing
boosted yet: 5 in sweep losses (-4.5), 12 in wins (-6.7). The switches fired on a boost: 6 in sweep
losses (-10.0), 11 in wins (-12.2). Counter moves are all boost-triggered except one Taunt in a win.

**Changes outside the sweep losses (gap 500 ms, then live effort):**

| set | game | result | turn | mix | rule | against | gap 500 ms | gap live |
|---|---|---|---|---|---|---|---|---|
| morning | 0842 | won | T21 | Triple Axel | Encore | Minior Atk/SpA/Spe +2 | -18/-22 | -5/-19 |
| morning | 6759 | lost (other) | T33 | Sludge Bomb | Haze | Sylveon SpA/SpD +1 | -2/-2 | -1/-1 |
| morning | 6872 | won | T16 | Shell Smash | Spore | Stantler SpA/SpD +1 | -4/+6 | +0/+1 |
| morning | 6614 | won | T26, T27, T30 | switch Salamence, Outrage, switch Regidrago | switch Lumineon (Encore) | Altaria after Dragon Dance | -4/-11, -8/-12, -7/-6 | -11/-10, -10/-9, -4/-4 |
| round 1 | 1031 | won | T4, T8 | Earthquake, switch Alcremie | Roar, switch Vivillon | Feraligatr +1/+1, Koraidon (read) | -7/-4, -5/-5 | -9/-1, -5/-5 |
| round 1 | 2984 | won | T18 | switch Avalugg | Encore | Kyurem-Black +1/+1 | -3/-2 | -3/-2 |
| round 1 | 8140 | lost (other) | T5, T6 | switch Veluza, Aqua Jet | switch Muk (Haze) | Excadrill Atk +2 | -15/-10, -14/-12 | -7/-14, -13/-11 |
| round 1 | 0328 | won | T11, T18 | switch Braviary, switch Entei | switch Appletun (Dragon Tail) | Darkrai, Porygon-Z (reads) | -3/-3, -7/-8 | -4/-3, -2/+3 |
| round 1 | 1408 | won | T13 | Strength Sap | switch Quagsire (Unaware) | Calyrex-Ice Spe +2 | -6/-6 | -6/-6 |
| round 1 | 9928 | won | T8, T10 | Hurricane, Hurricane | switch Skeledirge (Unaware) | Gogoat +1/+1, Gouging Fire (read) | -13/-9, -10/-10 | -9/-11, -12/-13 |
| round 1 | 1004 | won | T13 to T15 | Body Press | Flash Cannon | Ditto Def +2 | -6/+2, +6/-25, -2/-15 | +10/+5, +4/-6, -14/-32 |
| round 1 | 2613 | won | T16 | switch Meganium | Haze | Sandaconda after Coil | -6/-4 | -6/-4 |
| round 1 | 4842 | won | T9, T12 | Heavy Slam, Knock Off | switch Volcanion (Haze) | Wigglytuff SpA +2, Dunsparce (read) | -7/-7, -3/-3 | -4/-7, -4/-3 |
| round 1 | 6874 | won | T15 | Heavy Slam | switch Plusle (Encore) | Carbink Def +2 | -33/-34 | -34/-34 |
| round 1 | 0754 | won | T14, T15 | Ice Beam, switch Iron Leaves | switch Duraludon (Dragon Tail) | Drednaw (read, then +2) | +5/-4, -1/-1 | +4/+4, -1/-6 |
| round 1 | 6767 | won | T2 | Sticky Web | switch Brute Bonnet (Spore) | Jolteon (read) | -19/-12 | -11/-6 |
| round 1 | 7755 | won | T2 | Thunderbolt | switch Uxie (Encore) | Deoxys-Speed (read, never set up) | -10/-8 | -10/-6 |
| round 1 | 8407 | won | T3, T16 | Close Combat, Sleep Talk | switch Swampert (Yawn) | Kleavor Atk +2, Gouging Fire (read) | -20/-20, -1/-5 | -17/-15, -7/-6 |
| round 1 | 9138 | won | T8 | Flip Turn | Roar | Charizard +1/+1 | -8/-4 | -8/-3 |
| round 1 | 9600 | lost (other) | T10 | Leaf Storm | switch Torkoal (Yawn) | Furret (read) | -24/-22 | -20/-22 |
| round 1 | 1180 | won | T8 | Earthquake | Taunt | Minun (read) | -15/-15 | -12/-2 |
| round 1 | 5290 | won | T18 | Fire Blast | switch Meganium (Encore) | Torterra (read, never set up) | -7/-6 | -6/-6 |
| round 1 | 1148 | lost (other) | T30 | Flip Turn | Surf | Diancie Def +2 | +1/+2 | +0/+0 |
| round 1 | 1284 | won | T7 | Brave Bird | switch Volcanion (Haze) | Girafarig (read) | -12/-14 | -15/-15 |
| round 1 | 5868 | won | T17, T18, T22 | Calm Mind, Calm Mind, Focus Blast | switch Brute Bonnet (Spore) | Latias (read, then +1, +4) | -20/-18, -22/-22, -16/-17 | -16/-16, -21/-21, -15/-15 |
| round 1 | 6492 | won | T16 | Fire Blast | Yawn | Cobalion SpA/SpD +1 | -6/-5 | -4/-4 |
| round 1 | 6915 | lost (other) | T2 | switch Bruxish | switch Volcanion (Haze) | Cobalion Atk +1 | -5/-4 | -5/-5 |

## A variant without the switches (replay only)

`--mix '{"counter_hp": 1.01}'`: no benched Pokemon qualifies, so "use the counter" only plays our active's
own counter. "Don't hit the boosted defence" is unchanged. No arm for it exists yet.

| kind of game | games with a change | changed decisions (per 100) | gap at live effort (95% interval) |
|---|---|---|---|
| set-up sweep losses | 5 of 25 (20%, 9 to 39) | 16 (1.9, 1.1 to 3.0) | -2.2 (-3.4 to +0.7); 0 clearly worse |
| wins | 9 of 89 (10%, 5 to 18) | 11 (0.5, 0.3 to 0.8) | -7.4 (-9.8 to -4.2); 5 of 11 clearly worse |
| other losses with a set-up | 2 of 25 | 2 | -0.3 |
| losses without a set-up | 0 of 10 | 0 | - |

Its first changes in sweep losses: 1202 Encore at T29 (Scrafty +2/+2), 8092 Whirlwind, 4869 Roar on
Pecharunt, 9987 Body Press, and 7454 Hydro Pump. Two of these answer the sweeper directly.

## How it was run

On the box, in `~/dev/counter-replay` (a copy of this worktree), nice 15, CPU only, at most 7 processes
(one replay or rerun driver plus 6 Foul Play helpers, one core each). Each pass took under 3 minutes.

    python bench/counter_replay.py --scan ~/dev/ladder-1008/scan.json --json out/replay_1008.json ~/dev/ladder-1008/data/battles/*.jsonl
    python bench/counter_replay.py --scan ~/dev/setup-ladder/scan.json --json out/replay_r1.json <the 100 scanned battles>
    python bench/counter_rerun.py --replay out/replay_1008.json --ms 500 --repeat 2 --parallel 6 --json out/rerun_1008.json
    python bench/counter_rerun.py --replay out/replay_1008.json --live-effort --repeat 2 --parallel 6 --json out/rerun_live_1008.json
    (the same for r1; the variant adds --mix '{"counter_hp": 1.01}' to the replay, files out/nosw_*)
    python bench/counter_replay_report.py --set NAME REPLAY RERUN RERUN_LIVE [--set ...]   (the tables above)

Run scripts: `run_replay.sh`, `run_rerun.sh`, `run_rerun_live.sh`, `run_variant.sh`. Outputs are in
`~/dev/counter-replay/out` (`report.md`, `nosw_report.md`, the replay and rerun `.txt` and `.json` files).

- **Tools.** `counter_replay.py` now keeps, per changed decision: the rqid, both actions as poke-env orders
  them, the board (both active Pokemon, the opponent's boosts, the trigger), and a judgement against
  scan.json's set-up events. It prints the changes per kind of game with Wilson intervals and a table of
  the sweep losses, and takes `--mix` for variants. `counter_rerun.py` is round 1's `ladder_rerun.py`
  (branch setupfix), asked by rqid, keeping every option's share and score; `--live-effort` sets each
  world's iterations to the live search's at that decision. The helper is `bench/fp_bridge_server.py`
  (the same file, by md5, as the ladder's d7ace06 and round 1's code).
- **Why not `setup_round2_ladder.py`.** The counter rules read a board (our moves and bench, the
  opponent's types, abilities and HP) that only a rebuilt poke-env battle gives. setup_round2_ladder.py
  calls mix() on the notes' numbers alone, so it cannot run them.
- **Score.** The search's per-option score is the mean playout result from our side (0 to 1). Repeats
  sample other opponent sets, so the score level moves between repeats; the gap within a repeat is
  steadier (median difference between the two repeats' gaps: 0.2 points this morning, 1.1 points in
  round 1).
- **500 ms on the box is weaker than live.** It made 206k and 219k iterations (medians) where the live
  search made 342k and 316k on these decisions. That is why the live-effort pass exists. The two passes
  agree on everything above except single close calls such as 8092.
- **Caveats.** Only the first change of a game is played in the game's real position. The "answers the
  sweep?" column is a judgement from the scan's events (who set up, when, and who took the KOs) plus what
  the move does. Nothing here says what would have happened after the change. Sleep Clause is checked
  cautiously: any sleeping foe counts, even one that used Rest.
