# Tera diagnosis: the network alone

What was asked: the owner says the bot does not Terastallize enough and does not use it properly.
Find out what is true, with real games on a local server.

## Verdict

1. **The Tera path has no bug.** Mask, choice, order and wire were followed in 3,600 games. All
   3,446 Tera orders the player sent were answered by the server's `-terastallize` line, none was
   rejected, and the mask offered Tera on exactly the decisions on which the request did
   (87,319 of 87,319).
2. **"Not enough" is not true of the checkpoint on the ladder.** `learner_update_001475` Terastallizes
   in 91.5% of its games against itself (732 of 800) and 75.7% against SimpleHeuristics (454 of
   600). Foul Play: 88.6%. The newest checkpoint, `learner_update_002575`, does it less: 76.3% against
   itself (610 of 800), 44.8% against SimpleHeuristics (269 of 600).
3. **"Not properly" has a real cause: the network cannot see its own Tera type.** poke-env 0.15
   never reads `teraType` from the request, so the observation's Tera type block of the player's
   own Pokemon is empty until the Pokemon has Terastallized: 0 of 490,108 (decision, own Pokemon)
   pairs showed it. The network decides whether to Terastallize without knowing into what. It has
   been trained that way from the first step, so this is a missing input, not a wire that can be
   reconnected: showing the type to the trained networks changes nothing (75.7% to 76.5%, 44.8% to
   45.5%, 91.5% to 89.8%, 76.3% to 72.5%, all inside the noise; win rates the same).
4. So for these checkpoints it is **a learned habit under a blindfold**. The search player should
   decide Tera (poke-engine knows the Tera type), and training should later get the input
   (`ObservationAssembler(own_tera_type=True)`, added here; since wired into training as
   `ObsConfig.own_tera_type`, on by default, `--no-own-tera-type` for the old observation).
   No "Tera more" rule was added to the network player.

## The path, as read and as measured

| Step | Code | What it does | Checked how |
|---|---|---|---|
| Is Tera offered | `obs_assembler.create_action_mask` | action `10+k` is legal when move `k` is usable and `battle.can_tera` (poke-env sets it from the request's `canTerastallize`) | mask against the raw request at every decision: 0 disagreements in 143,632 decisions, 87,319 of them with Tera offered; `tests/test_tera.py` |
| Which action | `hybrid.combine` (beta 0, no veto, greedy: ladder.py's settings) | best legal log-probability | `tests/test_tera.py`; the probe recomputes the action from the logits it taps |
| Action to order | `obs_assembler.map_index_to_order`, `Player.create_order` | `(move, {"terastallize": True})`, sent as `/choose move <id> terastallize` | the sent text, tapped at `ps_client.send_message` |
| The server's answer | | `\|-terastallize\|p1a: <name>\|<Type>`, and the next request has no `canTerastallize` and `"terastallized":"<Type>"` | tapped at `ps_client._handle_message` |
| "We used Tera" in the observation | `obs_global` (`battle.used_tera`), `obs_pokemon` flag | set after the `-terastallize` line | wrong on 0 of 143,632 decisions |

One exchange from a wire log (`--wire`, the 8-game trial before run 1), unedited apart from
shortening the two requests:

```
request rqid 18  active[0].canTerastallize = 'Dragon' | Dialga, L73  teraType = 'Dragon'  terastallized = ''
>>> battle-gen9randombattle-147155|/choose move thunderwave terastallize
<<< |-terastallize|p1a: Dialga|Dragon
<<< |move|p2a: Arceus|Earthquake|p1a: Dialga
<<< |move|p1a: Dialga|Thunder Wave|p2a: Arceus
request rqid 20  canTerastallize = None | Dialga, L73  terastallized = 'Dragon'
```

Orders sent with `terastallize` against `-terastallize` lines for our side, per match: 217/217,
132/132, 368/368, 299/299 (run 1), 237/237, 137/137, 364/364, 311/311 (run 2): 2,065 of 2,065, and
1,381 of 1,381 in the probe matches below. Orders rejected as invalid: 0. Decisions that raised: 0.
(44 orders were answered "[Unavailable choice]": the server revealing a trap, which is normal play.)

## How often, and when (measured)

`bench/tera_usage.py`, the ladder player unchanged (`players.batched_policy_player`, greedy), 8
battles at a time on the GPU, local servers on ports 8610 and 8611. Every match was played twice
(run 1, run 2), nothing was tuned between them. Self-play counts both sides, so 200 games are 400
player-games. No game ended on a forfeit or the timer.

| Match | Tera in (run 1, run 2) | Pooled, 95% interval | Turn of the Tera, median | Won |
|---|---|---|---|---|
| u1475 against itself | 368/400, 364/400 | **91.5%** (732/800), 89.4 to 93.2 | 14, 14 | 50% |
| u1475 against SimpleHeuristics | 217/300, 237/300 | **75.7%** (454/600), 72.1 to 78.9 | 14, 13 | 93.8% (563/600) |
| u2575 against itself | 299/400, 311/400 | **76.3%** (610/800), 73.2 to 79.1 | 17, 16 | 50% |
| u2575 against SimpleHeuristics | 132/300, 137/300 | **44.8%** (269/600), 40.9 to 48.8 | 12, 12 | 95.7% (574/600) |
| Foul Play (bench/FOULPLAY_BRIDGE.md) | | 88.6% | | |

The policy's probability on the four Tera actions together, on turns where Tera is offered (move
requests before Tera is used; 15 to 22 such turns a game):

| Match (run 2) | Offered turns | Mean | Median | Above 0.1 | Above 0.5 | A Tera action is the best action |
|---|---|---|---|---|---|---|
| u1475 against itself | 6,805 | 0.0585 | 0.00014 | 9.1% | 5.1% | 5.35% of turns |
| u1475 against SimpleHeuristics | 4,591 | 0.0575 | 0.00014 | 9.9% | 4.8% | 5.16% |
| u2575 against itself | 8,418 | 0.0408 | 0.00009 | 6.6% | 3.5% | 3.69% |
| u2575 against SimpleHeuristics | 5,564 | 0.0320 | 0.00009 | 5.9% | 2.2% | 2.46% |

Run 1 gave the same within a few tenths of a point (mean 0.0581, 0.0547, 0.0392, 0.0309). The
policy is decided: under 0.01 on 81 to 86% of the offered turns, and Tera is used at most once a
game, so a low figure per turn is what any player has. The share of games is the figure to
compare, and there the ladder checkpoint is level with Foul Play. In the games without Tera the
highest probability it ever gave Tera had a median of 0.09 to 0.27: it considered it and did not.

The Tera comes on average 56 to 67% of the way through the game, with 3.6 to 3.9 own and 4.0 to
4.2 opposing Pokemon left in self-play, own HP 82 to 84%. Turns 1 to 3: 6 to 13% of the Teras.

## Are the games with Tera won more often? No, less. Not causal.

| Match (both runs) | Won with Tera | Won without |
|---|---|---|
| u1475 against itself | 47.8% (350/732) | 73.5% (50/68) |
| u2575 against itself | 43.9% (268/610) | 69.5% (132/190) |
| u1475 against SimpleHeuristics | 92.3% (419/454) | 98.6% (144/146) |
| u2575 against SimpleHeuristics | 91.8% (247/269) | 98.8% (327/331) |

Self-play, same network on both sides: in the games where only one side Terastallized, that side
won 14 of 60 (23%, u1475) and 29 of 132 (22%, u2575). Where both did, the first to do it won 144
of 305 (47%) and 114 of 228 (50%).

This does not say Tera loses games. The side that is ahead often wins before it needs Tera, and
the side under pressure uses it: the same selection would give these numbers if every Tera were
correct. What it does say is how the policy uses Tera: as a reaction when it is in trouble (see
the next section), and not as the way to win a game it is level in.

## What the Tera is used for, and what the blindfold costs

For every offered turn the probe wrote down the Tera type (from the request), the active
Pokemon's types, its moves, and the opposing active Pokemon's types. "Defence" below is the worst
multiplier of the opposing active Pokemon's own types against ours, before and after the Tera. It
is a stand-in for the threat (the real moves are mostly unknown), so read it as a tendency.

The Tera decisions themselves (run 2):

| Match | Tera decisions | Defence better | Defence worse | The move played gets the Tera boost | Boost or better defence | Neither | Worse defence and no boost |
|---|---|---|---|---|---|---|---|
| u1475 against itself | 364 | 185 (51%) | 41 (11%) | 98 (27%) | 233 (64%) | 99 (27%) | 32 (9%) |
| u1475 against SimpleHeuristics | 237 | 111 (47%) | 33 (14%) | 70 (30%) | 144 (61%) | 66 (28%) | 27 (11%) |
| u2575 against itself | 311 | 161 (52%) | 41 (13%) | 101 (32%) | 216 (69%) | 65 (21%) | 30 (10%) |
| u2575 against SimpleHeuristics | 137 | 90 (66%) | 7 (5%) | 51 (37%) | 110 (80%) | 21 (15%) | 6 (4%) |

So it is mostly a defensive Tera: half of them improve the matchup against what is in front, under
a third boost the move played, 12 to 27% are played with a status move. About one in ten makes the
defence worse and boosts nothing, and with the ladder checkpoint more than a quarter do neither
good thing by this measure.

The trigger is "I am weak to what is in front of me", whatever the Tera type turns out to be.
Offered turns on which the active Pokemon takes 2x or 4x from the opposing Pokemon's types,
split by what its hidden Tera type would do (run 2, self-play, both sides):

| Checkpoint | Tera would remove the weakness | Tera would leave it as it is |
|---|---|---|
| u1475 | Tera on 16.5% of 1,040 turns (95% 14.4 to 18.9) | on 11.6% of 406 turns (8.8 to 15.1) |
| u2575 | on 11.8% of 1,277 turns (10.1 to 13.6) | on 6.2% of 529 turns (4.5 to 8.6) |

On neutral turns it is 1.5 to 3% whichever way the type would go. The network does tell the two
columns apart a little, which it can only do by guessing the Tera type from the species and the
moves. That guess is limited: in the server's own set list (`data/random-battles/gen9/sets.json`,
879 sets of 509 species) 39% of the sets have exactly one Tera type, the rest have two to eight,
and the chance of naming the type when the exact set is known is 64.6% on average (not weighted
by how often a set is generated). So it Terastallizes into a type that does not help at 70% (u1475)
and 53% (u2575) of the rate at which it Terastallizes into one that does.

**The probe.** `--own-tera-type` writes the request's Tera type into the block (the observation
then showed it in 328,310 of 328,310 pairs, against 0 of 490,108 without). On the trained networks:

| Match | Tera in, type hidden | Tera in, type shown | Won, hidden | Won, shown |
|---|---|---|---|---|
| u1475 against SimpleHeuristics | 75.7% (454/600) | 76.5% (459/600) | 93.8% | 94.2% (565/600) |
| u2575 against SimpleHeuristics | 44.8% (269/600) | 45.5% (273/600) | 95.7% | 93.5% (561/600) |
| u1475 against itself | 91.5% (732/800) | 89.8% (359/400) | | |
| u2575 against itself | 76.3% (610/800) | 72.5% (290/400) | | |

No difference outside the noise (the largest, 95.7% against 93.5% won, is z = 1.66), and the split
above does not get sharper with the type shown (u1475: 14.9% against 13.6%; u2575: 9.6% against
9.1%). A lit Tera block on a Pokemon that has not Terastallized never occurred in training, and
the networks do not read it. The input only helps a network trained with it.

## A second, smaller defect in the observation (found, not changed)

`obs_moves.py` builds the STAB flag from `mon.type_1`, `mon.type_2` and, "if terastallized", the
Tera type. It tests `getattr(mon, 'terastallized', False)`, an attribute poke-env 0.15 does not
have (it is `is_terastallized`), and after a Tera poke-env's `type_1` is the Tera type and `type_2`
is None. So after Terastallizing, the flag says a move of the ORIGINAL type has lost its STAB,
which the game does not do. Seen on the test fixture: Espeon (Psychic, Tera Fairy), STAB flags of
Morning Sun / Alluring Voice / Psychic / Shadow Ball are `0 0 1 0` before the Tera and `0 1 0 0`
after it; the game still gives Psychic its 1.5x. Left as it is for the same reason as above: the
checkpoints were trained on it, and it belongs with the retraining.

## What was changed

| File | What |
|---|---|
| `bench/tera_usage.py` | the measurement: plays the ladder player on a local server, taps logits, orders and server lines, prints and stores the numbers above. `--opponent simple\|self\|...`, `--own-tera-type`, `--wire N`, `--summarize files` |
| `obs_assembler.py` | `own_tera_type_name(mon)`: the Tera type of an own Pokemon from the request (`mon._last_request["teraType"]`), lower case, None when unknown. `ObservationAssembler(own_tera_type=False)`: when True, writes that type into the own Pokemon's Tera type block. **Default False: the observation of every existing checkpoint is byte for byte what it was** (`tests/test_speed_obs.py` still passes, 6 passed 1 skipped). |
| `tests/test_tera.py`, `tests/fixtures/tera_request.json` | 16 tests on a real captured request and a real poke-env Battle: mask, order text, combine, the empty block by default, the flag changing exactly the six own blocks and nothing else, the state after `-terastallize`. All pass. |

Results and wire logs: `runs/tera/run1`, `runs/tera/run2` (git-ignored). 3,600 games in all: 1,600
in run 1, 2,000 in run 2 (run 2 added the probe in self-play and the columns on what the Tera did).

## For the agents after this one

- **Tera should be decided by the search, not by the network's Tera logits.** The network's
  probability on actions 10 to 13 is formed without the Tera type. When mixing the network into
  the root, treat its split between a move and the same move with Tera (`p[k]` against
  `p[10+k]`) as a weak prior at best: one option is to add the two and let the search split
  them. The engine has the type (the bridge passes the request).
- `obs_assembler.own_tera_type_name(battle.active_pokemon)` gives the own Tera type; the
  request's `active[0].canTerastallize` holds the same string for the active Pokemon.
  `battle.can_tera` is a bool in poke-env 0.15. `Pokemon.tera_type` is None for the own team
  until the Tera: do not rely on it. (poke-env's own SimpleHeuristicsPlayer does, and so never
  Terastallizes in random battles: 0 of 2,400 games here. It is a Tera-free opponent.)
- Reference rates for a search player: Foul Play 88.6% of games; the ladder network 91.5% in
  self-play. A much lower rate from a new player is a sign that something is off; a similar rate
  says nothing about quality. Better checks of quality from `tera_usage.py`: "worse defence and
  no boost" (now 9 to 11% of Teras) and the split "would remove the weakness / would leave it"
  (now 16.5% against 11.6%). The probe class works on any player built on
  `batched_policy_player`; a search player needs its own tap for the chosen action.
- For training later: construct the assembler with `own_tera_type=True` in the worker and in the
  players together (the layout and sizes do not change, so a checkpoint can be trained further),
  and fix the STAB flag in the same step. Neither is wired in. Until a network has been trained
  with it, leave the flag off: on the present checkpoints it buys nothing.

## Seen on the way, not explained: the side that challenges

In self-play the same network sits on both sides, and the player that sends the challenge is
always p1 (checked: every Tera line of the first player is `p1a`, of the second `p2a`). As p1,
u1475 won 112/200, 115/200 and 105/200 against itself: 332 of 600, 55.3% (95% 51.3 to 59.3,
z = 2.6). u2575 won 101, 95 and 103 of 200: 299 of 600, 49.8%. This may be chance (two checkpoints
were looked at), or the ladder checkpoint plays better as p1 than as p2. It was not followed up.
**Anyone measuring one player against another head to head should let each side challenge in
half of the games**, or five points of this can end up in the result.

## Not done, not known

- The Tera rate in the bot's real ladder games was not measured (the live bot and its folder were
  not touched). Self-play is the nearest thing here to an opponent of its own strength.
- Whether a Tera was right was not judged against a search or a damage calculation, only against
  the type chart and the opposing Pokemon's own types. The 9 to 11% "worse and no boost" can
  contain correct plays (a Tera against a coverage move, a Tera for a later turn).
- "Won less often with Tera" is an association. Nothing here measures what Tera is worth.
- The probe ran on networks that never saw the input. How much a network trained with its own
  Tera type gains is not known.
- Foul Play's Tera was not analysed the same way (what its Teras do to the matchup): only its
  88.6% is known.

## Reproduce

```
python start_servers.py --dir C:\Users\you\showdown-bot\server --ports 8610-8611 --log-dir runs/tera/serverlogs
set C=C:\Users\you\ps-ppo\runs\vm\checkpoints\learner_update_001475.pt
python bench/tera_usage.py --checkpoint %C% --opponent simple --games 300 --concurrency 8 --port 8610 --wire 4
python bench/tera_usage.py --checkpoint %C% --opponent self   --games 200 --concurrency 8 --port 8610
python bench/tera_usage.py --checkpoint %C% --opponent simple --games 300 --concurrency 8 --port 8610 --own-tera-type
python bench/tera_usage.py --summarize runs/tera/run2/*.json
python -m pytest tests/test_tera.py -q -p no:cacheprovider
```

300 games against SimpleHeuristics take 80 to 100 seconds, 200 games of self-play 200 to 260.
