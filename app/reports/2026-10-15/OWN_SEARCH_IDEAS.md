# Our own search: three ideas to beat Foul Play

Three ways to build a search of our own that plays better than Foul Play's: let our network steer
it, predict the opponent's hidden sets from data, and solve each turn as the simultaneous game it
is. Two of the three are already half built on branch `oursearch` (v1, frozen at 444759a, never
measured). This page says what each idea is, why it should help, what exists, and the first step.

Written 2026-10-15 for the owner and the collaborator. Numbers come from the files cited.

## Where the search stands today

- **Foul Play helps against bots, not against our own network.** With Foul Play's search beside it
  (mix, the veto rule) the network wins 68.2% against Foul Play, +23.6 points over the network
  alone (44.6%). Against the same network playing alone mix wins only 50.0% over 306 games
  (`pick:bench/notes/gauntlet.md`; `oursearch:bench/OURSEARCH.md` section 1).
- **Foul Play's own score is a weak judge.** Before turn 20 it predicts who wins no better than a
  plain HP count (AUC 0.58 against 0.64 at turns 6 to 10, 220 ladder games), yet it overrules the
  network on about 20% of ladder moves (`bench/OURSEARCH.md` section 1).
- **More thinking time has not shown a gain** on the ladder: 1 s and 5 s a move both won exactly as
  many games as the ratings predicted (65% of 20, 70% of 10, p 0.78). A 300 + 300 game test against
  Foul Play is running on the box today (`~/dev/think`, 500 ms against 2200 ms).
- **The biggest single weakness is set-up sweeps:** 57% of losses are one boosted foe sweeping
  (`REPORT_LADDER_AND_BASELINE_RUN.md` on `baseline-run`).
- **Outside evidence that a network-driven search pays:** Jaxcalibur (peak 2,557, the only gen9
  random battles bot clearly above us) gets +100 to +150 Elo from its search over its network alone:
  the policy picks candidates, the value head scores leaves, 32 worlds from the network's own set
  predictions, about 2 turns deep, mixing between near-equal moves. mikumiku37 (VGC) took a bot
  from 1,747 to #1 at 1,857 with the one-turn version (`bench/OURSEARCH.md` section 1).

## Idea 1: a search steered by our network

**What.** Keep Foul Play's battle simulator (poke-engine) for the rules, but replace its guesswork:
our policy chooses which of our moves are worth exploring, and our value head scores the positions
the search reaches, instead of Foul Play's hand-written evaluation. AlphaZero, Jaxcalibur and
mikumiku37 work this way.

**What exists.** v1 on `oursearch` does it one turn deep: our top candidates against their options,
one turn simulated per pair in sampled worlds, each resulting position converted into our
network's observation and valued (`bench/OURSEARCH.md` sections 2 to 6, converter in
`bench/OURSEARCH_CONVERTER.md`). The adversarial review found it correct (leaf values equal to the
real next decision's to 1e-6, 0 invalid orders in 7,526 requests) but expects it to come out level
with mix, because the value head is not yet better than a coin's edge as a judge: the value gate
failed narrowly (`bench/OURSEARCH_REVIEW.md`; `bench/OURSEARCH_GATE.md`).

**First steps.**
1. Run the pre-registered MEASURE on v1 as it is (section 13 of `OURSEARCH.md`): about 800 box
   games, 2 to 4 box-hours. It is the baseline every later version is judged against.
2. Make the judge better before searching deeper: train the value head on search-labelled
   outcomes, or give it more capacity, until it beats Foul Play's score and the HP count in the
   gate.
3. v2: two turns deep and 32 worlds, the Jaxcalibur shape.

## Idea 2: predict the opponent's hidden sets from data

**What.** In random battles every Pokemon gets one of a few fixed sets. Foul Play samples the
opponent's unseen moves, items and abilities from usage statistics. Instead, learn which set it is
from what has been revealed, using the 35,919 top-player games already on the box
(`~/dev/armb-human/data`), and sample the search's worlds from that model.

**Why it should help.** Every search decision is only as good as its worlds. A search that thinks a
Pokemon has Choice Band when it has Swords Dance + Leftovers plays the wrong move with full
confidence, and set-up sweepers are exactly what beats us (57% of losses).

**What exists.** An opponent model predicts their next action (47.0% top-1, 80.7% top-3 on 1,320
held-out human turns, used by v1's solver), but nothing predicts their set. Jaxcalibur samples
worlds from its own network's set predictions.

**First step.** On held-out human games, score how often Foul Play's sampled set matches the set
the game later reveals, against a small model trained on the same data. If the model is clearly
more accurate, plug it into the bridge's world sampling (Foul Play side) or v1's (ours) and
measure.

## Idea 3: solve each turn as the simultaneous game it is

**What.** In Pokemon both players choose at the same moment. Foul Play's search (decoupled UCT)
behaves as if the opponent sees our move first, and always plays its single best move. Treat each
turn as a matrix game instead: our options against theirs, solve it with regret matching or CFR,
and play the mixed answer when moves are close, so the bot cannot be read and exploited.

**Why it should help.** PokaiTrainer measured it: a best-response exploiter beat decoupled UCT
83.6% of the time but a CFR player only 49.6% (`bench/OURSEARCH.md` section 7).

**What exists.** v1 already solves the one-turn matrix game (Bayesian maximin blended with the
opponent model, regret matching+, logged exploitability gap) and draws its move from the mixed
solution with a trust region around the network (`bench/OURSEARCH.md` sections 7 and 8).

**First step.** Comes with idea 1's MEASURE. Beyond one turn: simultaneous-move MCTS (regret
matching at every node) with our value head cutting the tree at depth 2.

## Order and how to judge

1. MEASURE v1 now (cheap, 2 to 4 box-hours): the baseline.
2. Idea 2's accuracy test (a day of work, CPU only): it can improve Foul Play and our search alike.
3. Improve the value head until it passes the gate, then v2 (ideas 1 and 3 at depth 2).

Judge every version the same way: 400 or more games on the box against Foul Play at a fixed
iteration count (not a fixed time, which drifts with load), plus head to head against today's mix.
Win 55% or more against both, then confirm on the ladder.
