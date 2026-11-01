# How Pokemon Showdown became a reinforcement-learning environment

Pokemon Showdown is a battle simulator played in the browser. It was never meant to be a training
environment: there is no reset function, no observation tensor and no reward. This page explains
how the code turns it into one that runs about 2,300 battles at once on one machine. It covers what
we had to work out about the protocol along the way, and how each piece was proven correct.

Code references are relative to [`training/`](../training/) unless they start with `app/`. The
starting point was [Nebraskinator/ps-ppo](https://github.com/Nebraskinator/ps-ppo) (MIT). Section 10
says exactly which parts came from there and which were added here.

Contents:
1. [The shape of the environment](#1-the-shape-of-the-environment)
2. [The server: battles without a ladder](#2-the-server-battles-without-a-ladder)
3. [The client: poke-env, one thread per worker, one GPU call per batch](#3-the-client)
4. [Reverse engineering: what the protocol does and what poke-env gets wrong](#4-reverse-engineering-the-protocol)
5. [The observation](#5-the-observation)
6. [The action space](#6-the-action-space)
7. [The reward](#7-the-reward)
8. [Episodes and trajectories](#8-episodes-and-trajectories)
9. [Throughput](#9-throughput)
10. [How we know it is right](#10-how-we-know-it-is-right)
11. [What came from upstream and what was added](#11-what-came-from-upstream-and-what-was-added)

---

## 1. The shape of the environment

```
 Showdown server (Node)              rollout worker (Python, one per server)            GPU
 ┌───────────────────────┐  websocket ┌───────────────────────────────────────┐  Ray  ┌──────────────┐
 │ rlspawn plugin keeps  │◄──────────►│ asyncio thread: poke-env players      │──────►│ inference    │
 │ N battles per pair    │  protocol  │   parse lines, build observation,     │ batch │ actor(s)     │
 │ simulator process     │   lines    │   park a future per decision          │◄──────│ transformer  │
 └───────────────────────┘            │ RL thread: drain decisions, one call, │actions└──────────────┘
                                      │   finish episodes, ship to learner    │──────► learner (PPO)
                                      └───────────────────────────────────────┘ episodes
```

- One **episode** is one seat of one battle. In self-play both seats are our players, so every
  battle gives two episodes.
- One **step** is one decision: Showdown sends a `|request|`, our player answers with a `/choose`.
- The **format** is `gen9randombattle`. The server builds both teams at random, so there is no team
  building to learn.

## 2. The server: battles without a ladder

**A stock server in a training mode.** Each port runs
`node pokemon-showdown start --no-security --skip-build --port P` (`start_servers.py`).
`--no-security` turns off login checks and throttles, so players can be unregistered guests and
nothing rate-limits them. The battle timer stays off unless a format forces it. `--skip-build`
matters when several servers share one checkout: without it, every new server rebuilds `dist/`
underneath the ones already running. `start_servers.py` restarts any server that exits and prints a
line the supervisor (`pipeline.py`) watches for.

**Battles are made by a chat plugin, not by challenges or the ladder.**
`pokemon-showdown/server/chat-plugins/rlspawn.ts` (from upstream, unchanged) adds four commands:

| Command | What it does |
|---|---|
| `/rlautospawn A, B, gen9randombattle, N` | Keeps N live battles between players A and B. A 50 ms timer counts the pair's live rooms and creates the missing ones with `Rooms.createBattle` (at most 50 per tick). An empty team means "random battle team" |
| `/rlactive` | Returns the caller's live battle rooms as one CSV frame |
| `/rlrescue <room>` | Expires a room the caller plays in |
| `/rlautooff` | Stops a pair's spawner |

A reaper expires any plugin-made battle older than 5 minutes. Because the spawner keeps refilling
battles for a pair after its trainer has gone, the servers are restarted before every run.

**How many at once.** Upstream's default was 800 battles. The laptop recipe uses 4 servers and 128
battles. The cloud runs used 44 servers and 2,816 battles, and later 36 servers and 2,304 battles
(64 per server, 16 rooms per pair of players). The cloud box runs the servers and rollout workers at
nice 5, so the single-threaded learner is never starved of CPU.

## 3. The client

**poke-env 0.15** (`requirements.txt` pins `>=0.15,<0.16`) parses the protocol into a `Battle`
object. Each player connects to `ws://127.0.0.1:P/showdown/websocket` as a guest with a random name.
At start-up one side of each pair sends `/rlautospawn` in the lobby.

**Two threads per worker** (upstream's design, kept; `worker.py`):
- An **asyncio thread** runs every poke-env player of that worker. When Showdown asks for a move,
  `choose_move` builds the observation, computes the reward potential, parks a future and puts
  `("STEP", tag, obs, phi)` on a queue.
- A **synchronous RL thread** waits for the first event, sleeps 3 ms, drains the rest and makes one
  blocking Ray call, `inference_actor.infer_batch(tags, obs_batch)`. The actions go back through
  `loop.call_soon_threadsafe(future.set_result, action)`.

**Inference.** The inference actor holds each battle's recurrent history: the field token's input
to each layer for the last 64 decisions, keyed by player and battle. It is cleared when the battle
ends. This project moved that history into one pooled GPU tensor and added `--infer-actors N`, so
three GPUs can serve decisions while a fourth trains. Observations travel as float16 and are
widened on the GPU.

**Keeping client and server in sync** (upstream, kept). Every 30 seconds each player asks
`/rlactive`, then:
- joins battles the server has and it does not;
- cleans up battles with no action for 60 seconds, then rescues them;
- drops battles gone from the server.

Every 3 minutes it re-sends `/rlautospawn`.

## 4. Reverse engineering the protocol

Most of the work in this section was finding places where the training client and the real game
disagree. The method was the same each time:
1. Record the raw websocket lines of many battles.
2. Compare what Showdown actually sent (the `|request|` JSON, the protocol lines, the simulator's
   own source) with what poke-env's `Battle` object and our observation said.
3. Count every mismatch over hundreds of thousands of decisions.
4. Fix it, and pin the fix with a test built from a real position.

### 4.1 Refused choices: a feedback loop

When Showdown refuses a choice it sends `|error|[Invalid choice] ...`. Upstream answered every
refusal with `/choose default`, and poke-env also asked the policy again. That made two answers per
refusal and left the client one decision behind the server. It started a flood:
- 43,122 refusals in 16 seconds with scripted players (4,478 a second at the peak);
- 40,304 handler tasks waiting on one room's lock;
- in a training run, an episode of 66 steps for 22 turns.

On the cloud box the rollout workers once swelled to 117.5 GB and the trainer was killed at 472 of
497 GB of RAM. This flood is the suspected cause; that was never proven.

The fix (`worker.py`):
- Answer at most **one** `/choose default` per request, and never for the four messages that mean
  "you cannot choose now": `There's nothing to choose`, `Sorry, too late`, `Can't do anything`,
  `Can't undo`.
- Strip the `[Invalid choice]` line before poke-env sees it, so the policy is not asked twice and
  the episode gets no phantom step.

`tests/test_worker_choice_storm.py` replays the flood and checks the episode equals the same battle
without the error.

### 4.2 Rooms that close without a result

The plugin's reaper ends old rooms with `|expire|` then `|deinit`, with no `|win|` or `|tie|`.
poke-env never released the room's seat (`max_concurrent_battles`), so the episode slot, the
trajectory and the inference history stayed taken. Once every seat was full, the next battle's first
message waited forever while holding its room's lock. Now a `|deinit` with no result releases
everything, and the battle's steps are dropped.

Late lines for rooms that already finished (such as a `|deinit` after the result) used to make
poke-env park a coroutine forever. Each new battle woke all of them, so the cost grew with every
game played. Those lines are now dropped, with a bounded memory of finished rooms.

### 4.3 The Tera type nobody could see

poke-env 0.15 (and 0.16.1) never reads `teraType` from the request JSON. Your own Pokemon's Tera
type stayed empty until the `-terastallize` line, which is after the decision it should inform.
Over 490,108 (decision, Pokemon) pairs the network never once saw its own Tera type.

poke-env does keep each Pokemon's raw request entry in `Pokemon._last_request`, filed by its
`ident`. The fix reads `teraType` and `terastallized` from there (`obs_tera.py`, `obs_assembler.py`).
The action mask was checked against the raw request on 143,632 decisions, 87,319 of them with Tera
offered, with zero disagreements. Every Tera order was confirmed on the wire as
`/choose move <id> terastallize` (2,065 of 2,065 and 1,381 of 1,381 answered by `-terastallize`).

Two related details:
- **STAB after Tera.** After a Pokemon Terastallizes, poke-env reports only the Tera type, so the
  encoders dropped STAB on its original types. Gen 9 keeps that STAB. The own-Tera option restores
  it.
- **Zoroark.** A disguised Zoroark that Terastallizes is reported as the Pokemon it is posing as, so
  poke-env files the Tera type on the wrong Pokemon. The own request is trusted instead, and a
  mismatching Tera block is overwritten. This was 230 of 759,624 own blocks, all on teams with a
  Zoroark.

### 4.4 The Transform switch-name clash

poke-env sends a switch as `/choose switch <name>`. Showdown resolves that name in `Side.chooseSwitch`,
checking the active Pokemon first, by name or species. Suppose our active Ditto Transformed into the
opponent's Garchomp, and we also have a Garchomp on the bench. Then `/choose switch Garchomp`
resolves to the active Ditto and is refused: `Can't switch: You can't switch to an active Pokemon`.

`switch_names.switch_name_taken(active, mon)` spots this case. It checks that the active Pokemon is
`transformed` and that its temporary base stats, height and weight match the Pokedex entry of the
benched Pokemon's name. Types are not compared, because a Transformed Pokemon that Terastallizes
loses its temporary types. The action mask then drops that one switch. It happened about once in
100,000 decisions, always with Ditto active. The fixture `app/tests/fixtures/ditto_transform_switch.json`
is a real position from a training server.

### 4.5 Abilities and items poke-env misses

An audit of 42,058 decisions found six holes. Each fix writes only into slots the observation
already had (`obs_abil.py`).

| Fix | What was wrong | Evidence |
|---|---|---|
| `abil_sets` | The "possible abilities" slots came from the Pokedex, which names an ability the species cannot have in random battles 71% of the time | 441 of 509 species have exactly one random-battle ability, so it is known for certain |
| `abil_announced` | poke-env ignores `[from] ability: X` on lines like `-weather`, `-heal`, `-damage`, and `[from] item:` on `-status` (Toxic Orb) | 9,893 and 2,677 (decision, Pokemon) pairs showed "unknown" after the game had announced it |
| `abil_history` | Bare `-ability\|POKEMON\|X` lines did not reach the last-turn block | 3,352 of 3,368 such lines left it empty |
| `item_unknown` | The flag meant "holds nothing" where poke-env meant "not revealed yet" | wrong on all 101,324 unknown items |
| `spe_conditions` | Swift Swim, Chlorophyll, Sand Rush, Slush Rush and Surge Surfer double Speed only in their weather or terrain; Unburden only after the item is gone | |
| `regenerator` | A benched opponent certain to have Regenerator showed the HP from before it healed a third | 1,726 pairs too low |

Also handled:
- **Hidden formes.** Greninja-Bond switches in as "Greninja". Its set abilities are merged into the
  shown species, after 48 decisions showed Protean wrongly.
- **Illusion moves.** Moves a Zoroark used in disguise stay on the Pokemon it posed as, even after
  `|replace|`. Once an Illusion user is revealed, impossible moves are moved back.
- **Known gap.** Broken-Illusion HP is not repaired: 98 wrong HP pairs out of 175,818.

### 4.6 Smaller quirks

| Quirk | Handling |
|---|---|
| Struggle | The request offers `struggle`, which is not one of the 4 known moves. The mask sets action 0 and maps it to Struggle |
| Recharge, or any move outside the known 4 | No mask bit matches, so a safety fallback forces action 0, which falls through to the first available move |
| Forced switches, trapping, Revival Blessing | poke-env's `available_switches` handles them; switches map to the 6 fixed team slots. Foul Play cannot answer a Revival Blessing request (1 battle in 43,485 live decisions), so the network answers it |
| `[Unavailable choice]` | The server reveals a trap or a disabled move the player could not know. Counted as normal play, not a fault |
| Ties | `\|tie` scores 0. Upstream scored anything but a win as a loss |
| Typing limit (public server only) | Showdown queues messages closer than 600 ms and drops one that finds the queue full. `app/ladder.py` spaces orders 300 ms apart and resends a dropped order with its request id |

## 5. The observation

The observation is a flat float16 vector that the model cuts back into tokens.

**Base layout, 8,567 numbers** (`obs_assembler.py`; upstream's layout, with this project's fixes
written into it):

| Block | Size | Contents |
|---|---|---|
| Pokemon bodies | 12 x 351 | HP % and absolute HP; 5 stats as (min, estimate, max) from the evidence so far; 7 boosts as one-hots; level, weight and height bins; active, fainted, Terastallized flags; status counter; fog-of-war numbers (item unknown, ability unknown, unrevealed moves); types and Tera type; 156 volatile effects; status; gender; slot identity |
| Pokemon ids | 12 x 2 | species and item ids |
| Ability ids | 12 x 4 | the confirmed ability, then up to 3 possible ones |
| Move ids | 12 x 4 | 48 move ids, "unknown" for unrevealed slots |
| Move scalars | 48 x 85 | accuracy, power, PP, priority, type, category, target, STAB, status inflicted and its chance, owner, slot, usable now, expected hits, self-boosts |
| Global | 41 | turn; who has used Tera; weather and its duration; terrains and Trick Room; hazards and screens for both sides |
| Last-turn transitions | 10 + 2 x 45 | per side: the move used, actor, target, ability and item revealed; 38 event flags (moved first, crit, super effective, fainted, switched, pivoted, hazards set) and boost changes |
| Action mask | 14 | the legal actions |

The transition block holds every protocol line since this player's last decision, minus `t:`,
`request` and `turn` lines.

**Two blocks added by this project**, both from a pinned copy of the random-battle set data for
509 species:
- **Matchup block, 1,068 numbers** (`obs_matchup.py`):
  - for each of the 48 moves, the log2 damage multiplier against each of the 6 opposing Pokemon;
  - the chance that an unrevealed ability or item makes the target immune;
  - for each Pokemon, its top ability and item candidates with probabilities, and whether a Choice
    item is possible or the item is gone.
  
  The type rules agree with Showdown's own engine on 20,000 random cases (`tests/test_matchup.py`).
- **Moveset block, 660 numbers** (`obs_movesets.py`):
  - for each Pokemon, its top 8 unrevealed move candidates with probabilities;
  - a Tera type distribution;
  - its threat to each enemy slot;
  - role flags (set-up, priority, recovery, pivot, status, hazards, hazard removal, phazing).
  
  Draw probabilities are exact over the set roles still consistent with what was shown.

Total with both blocks: **10,295 numbers**. A network trained without the blocks gains them by
**widening**: zero-initialised input columns, so it plays bit for bit the same on day one
(`tools/widen_checkpoint.py`).

**Tokens.** The model (`ppo_core.py`, upstream's PokeTransformer) cuts each decision into 64
tokens:

| Token | What |
|---|---|
| 0 | field (global block, transitions, action mask) |
| 1 to 12 | Pokemon, 6 ours then 6 theirs |
| 13 to 60 | the 48 moves |
| 61 | a register token |
| 62, 63 | actor and critic |

Each layer also attends to the field token of the last 64 decisions of the same battle. That is its
memory of the game.

## 6. The action space

14 actions (upstream's design, `obs_assembler.py`):

| Index | Action | Sent as |
|---|---|---|
| 0 to 3 | use move slot k | `/choose move <id>` |
| 4 to 9 | switch to team slot k (fixed team order) | `/choose switch <name>` |
| 10 to 13 | Terastallize and use move k | `/choose move <id> terastallize` |

Mask rules:
- a move bit is set for each known move the request lists as available, and its Tera bit when
  Tera is allowed;
- Struggle sets bit 0;
- switch bits come from the available switches;
- if nothing is legal, bit 0 is forced, which becomes `/choose default`;
- the main-line observation also drops the Transform name-clash switch (section 4.4).

## 7. The reward

`r_t = 0.5 x (0.999 x Phi(s_t+1) - Phi(s_t))` on every step, and the result (+1 win, -1 loss, 0
tie) on the last (`rl_reward/`, [`training/RL_REWARD.md`](../training/RL_REWARD.md)).

- **Phi** is `0.98 x tanh(w . f(s))` over 19 "mine minus theirs" features:
  - Pokemon left and HP;
  - status counts;
  - boosts;
  - hazards and screens;
  - an unused Tera;
  - unrevealed Pokemon.
- **Fit.** Logistic regression on 2,987 public replays of players rated 2,351 to 2,547 (91,293 turn
  states). Out of fold, AUC 0.718, calibrated within a few points at every level.
- **It cannot be gamed.** The shaping is potential-based: with gamma equal to the learner's 0.999,
  the shaping terms cancel exactly over a game. It cannot change which policy wins most, and it pays
  nothing for winning by more. It is also an odd function with no intercept, so the two seats'
  potentials are zero-sum.
- **What it replaced.** Upstream paid +-0.1 per faint, which pays for margin and raises the critic's
  variance.
- **Rejected.** Action bonuses (super-effective hits, setting hazards) were rejected because "it pays
  for doing a thing, not for a position, so it can always be farmed".
- **Live check.** Over 30 games and 1,334 decisions (`check_reward.py`), the cancellation held to
  5e-8, and no return left the value head's [-1.5, 1.5] range.

## 8. Episodes and trajectories

- **Per step.** The RL thread stores the observation, the action, its log-probability, the expected
  value and its variance (from the 51-bin value head), and Phi.
- **When the game ends.** On `|win|` or `|tie|` the episode is closed and queued for the learner.
  Stalled or closed rooms are dropped.
- **The worker's queue.** It holds 160 episodes and drops the oldest when full. The league run
  shrank it to 32, so data reaches the learner about one update old instead of four to six.
- **The learner.** It computes GAE with gamma 0.999 and a per-step lambda of
  `delta^2 / (delta^2 + Var)`, clamped to [0.55, 0.95], where Var is the critic's own variance
  (upstream).
- **Extra per-step columns** feed optional heads: read labels, the opponent's next action, set
  labels, boosted rows. With none of them on, an update is bit-identical to the code without them.

## 9. Throughput

| Setup | Learned decisions a second |
|---|---|
| Laptop, RTX 4060, 4 servers, 128 battles | about 390 |
| Cloud, one L40S, 44 servers, 2,816 battles | 4,700 to 4,800 |
| Cloud g6e.12xlarge (4 L40S), 36 servers, 2,304 battles, 3 inference GPUs, speed flags | 5,500 to 6,200 |
| Same box, league run (frozen-network opponents) | 4,300 to 4,600 |

On the laptop, building the observation is 84% of a worker's CPU time per decision (1.30 of 1.56
ms). Table-driven encoders that cache what does not change between decisions write the same bytes
as the original functions (`tests/test_speed_obs.py`). The speed flags `--fast-update` and
`--fast-ingest` each give bit-identical updates (+21 to 28% and +8 to 12%).

## 10. How we know it is right

- **Tests.** 41 test files with about 940 tests, plus property tests for the reward. The ones that
  matter most for the environment:
  - `test_worker_choice_storm.py`, `test_worker_memory.py`, `test_worker_handover.py`: the
    refusal handling, what a worker keeps per battle, and what the learner receives at every ending;
  - `test_tera.py`: a real request through a real poke-env `Battle`, down to the order text on the
    wire;
  - `test_speed_obs.py`: the fast encoders against the original ones on at least 10,000 replayed
    decisions;
  - `test_matchup.py`: the type rules against Showdown's own engine, 20,000 cases;
  - `test_widen.py`: a widened network plays exactly like the original.
- **Exactness tool.** `bench/league_exactness.py` (training) and `app/bench/v2_exactness.py` (app)
  replay 300 recorded battles, 12,048 decisions, through the training worker and through the window
  app's player. They compare every decision's observation byte for byte, and the logits and values
  bit for bit. Results:
  - the league network, the 207M and 300M checkpoints and 24381: identical on all 12,048;
  - a negative control on the wrong observation: identical on 0 of 12,048, so the check really does
    catch a wrong observation.

## 11. What came from upstream and what was added

| Piece | Upstream (Nebraskinator/ps-ppo) | Added here |
|---|---|---|
| Server | `rlspawn.ts` battle spawner, a bash launcher | `start_servers.py` (any OS, `--skip-build`, restarts the supervisor can see), supervisor, nice levels |
| Client | poke-env thread bridge, one Ray call per batch, `/rlactive` reconciliation | refusal flood fix, closed-room release, late-line drop, pooled inference history, several inference GPUs, opponent seats, fresher learner queue |
| Observation | the 8,567-number layout, 64-token model input | own Tera type from the request, STAB after Tera, Zoroark fixes, six ability and item fixes, the Transform switch fix, matchup and moveset blocks, widening, fast encoders |
| Actions | 14 actions, mask, order mapping | the switch-name clash filter, Tera path audits |
| Reward | +-1 plus +-0.1 per faint | potential-based shaping fitted on top players' games, ties scored 0 |
| Teacher | poke-env's SimpleHeuristicsPlayer | PSBot, our own heuristic bot, run headless under Node |
| Verification | none in the repository | about 940 tests, recorded-battle replays, exactness tooling |

Some notes and docstrings inside the code trees are older than the code around them. For example,
upstream's README describes an earlier, smaller observation, and `config.py` calls the Phi cap
"about 0.85" where the fitted weights say 0.98. The code and the numbers above are the current
ones.
