# Opponent seats (Stage 3: set-up)

`opponent_seats.py` puts an opponent of a fixed kind in one seat of a share of the PPO pairs.
The other seat of such a pair is `worker.SyncBridgePlayer`, unchanged, and is trained on like any
self-play seat. The opponent's seat is never trained on.

```
python train.py --mode ppo ... --opponent-seats setupnet:0.15,psbot:0.05,foulplay:0.05 \
    --setupnet-p 0.5 --setupnet-max-stage 4 --seat-foulplay-iterations N
```

(The flags come from `opponent_seats.add_arguments`, which train.py calls. See "Wiring".)

## The kinds

| Kind | Who moves | When it falls back |
|---|---|---|
| `setupnet` | The network, asked through the ordinary STEP event. Where a set-up rule applies, the rule's move replaces the network's with probability **p**. | Never needed: every decision is the network's first. |
| `psbot` | PSBot (`psbot_bridge.PSBotMixin`, `heuristic/engine.js`). | No legal answer from the engine: the network moves for that decision. |
| `foulplay` | Foul Play at a **fixed number of iterations per sampled world** (the bridge's `iterations` option), so its strength does not depend on how loaded the box is. | No answer in time, a helper down or hung, a battle older than 150 s: the network plays the rest of that battle. |

### setupnet's rules

`setup_rules.py` holds rules 2 to 5 of decide's `bench/setup_player.py`. They are checked in this
order on a move request:

1. **Asleep with Sleep Talk:** Sleep Talk.
2. **Boosted and below half HP:** a recovery move.
3. **A sleep move** when the target can fall asleep. It cannot when it has a status, Sleep Clause
   applies, or there is a Substitute, Safeguard, Electric or Misty Terrain, or Insomnia and the
   like. Powder moves also fail on a Grass type, Overcoat or Safety Goggles.
4. **A set-up move** when not in danger, while some stat it raises is below `max_stage`. Of
   several, the one that raises most.

Rule 1 (switch out on a bad matchup) and rule 6 (SimpleHeuristics' own move) are the network's
own choice here.
- **A switch the network chose is kept** (`SetupNetSettings.keep_switches`).
- **When the network picked the rule's move itself**, its own order stands, Tera included.

### The two knobs

- **`--setupnet-p`:** per decision on which a rule applies, the chance that the rule's move is
  played. A rule applies on about 17% of the seat's decisions, so the rule moves are about
  0.17 x p of its decisions.
- **`--setupnet-max-stage`** (set-up strength, 1 to 6, default 4): how far rule 4 keeps
  boosting a stat before the seat attacks.

## What reaches the learner: nothing of an opponent seat

- **An opponent seat never puts DONE or IMITATION on the worker's queue.**
- **Where the network moves for it** (always for setupnet, as the fallback for the others), it
  puts the ordinary STEP. At the end of the battle it puts one CLEANUP. The worker then drops
  those steps, the side's history in the inference actor, and its episode slot.
- **The base class's DONE** is swallowed for the length of one call (`OpponentSeat._quietly`), as
  stage3's `fp_opponent.py` did.
- **A room the server closes** goes through `SyncBridgePlayer._room_closed`, which only ever
  puts a CLEANUP.

## The plan

`plan_seats(pairs_per_worker, rooms_per_pair, settings)` decides which pairs get which kind.
- **Shares are of all the run's pairs:** each kind gets round(share x pairs), at least 1.
- **Dealing:** the pairs are dealt one at a time round the workers, each kind going on from
  where the previous one stopped.
- **Placement:** a worker's opponent pairs are its last pairs.
- **Seats:** the opponent's seat alternates, so the network is p1 in half of a kind's pairs.
- **Foul Play:**
  - its pairs go to at most `max_helpers` workers;
  - helpers are sized as in stage3: a decision every 2.2 s per side, at the search time
    `--seat-foulplay-ms` that the iterations stand for.
- **Nothing in it is random.** A worker that Ray starts again gets the same seats. setupnet's
  draws are seeded by `(--opponent-seats-seed, port, pair)`.

**A kind that cannot be used in a worker** is played as ordinary self-play pairs, and the
heartbeat says so. That covers PSBot's engine or Foul Play not starting, and a teacher mode.

## The gate: the network against setupnet

`bench/seats_smoke.py --seats setupnet:1` runs the training's own rollout code on a local server.
- **The worker:** `OpponentSeatsRolloutWorker`, using `RolloutWorker`'s own rollout loop.
- **The inference actor:** `InferenceActor` itself, on the GPU, sampling at temperature 1 as in
  training.
- **The network:** `ckpt.pt` from `~/dev/think`.
- **The setup:** every pair a setupnet pair, 4 pairs x 16 battles at once, about 1,000 games per
  row, seed 1.

The plan's gate: the network must win **55 to 80%** against the seat.

| p | max stage | games | network won | rule moves / seat decisions | set-up moves played | the network's own set-up choices |
|---|---|---|---|---|---|---|
| 0 | 4 | 1019 | 49.1% ±3.1 | 0 | 6.7% | 6.7% |
| 0.25 | 4 | 1003 | 55.9% ±3.1 | 2.9% | 8.9% | 6.3% |
| **0.5** | **4** | 1015 | **59.5% ±3.0** | 5.6% | 11.3% | 6.3% |
| 0.75 | 4 | 1029 | 62.6% ±3.0 | 7.7% | 12.5% | 5.6% |
| 1.0 | 4 | 1019 | 65.5% ±2.9 | 10.2% | 14.4% | 5.8% |
| 1.0 | 2 | 1002 | 59.6% ±3.0 | 7.3% | 11.7% | 5.7% |
| 1.0 | 6 | 1024 | 69.7% ±2.8 | 11.6% | 16.1% | 5.9% |
| 0.5 | 6 | 1011 | 60.1% ±3.0 | 6.1% | 11.7% | 6.4% |

**What the table shows:**
- **p = 0 is a mirror match** (49%), as it should be.
- **Every p from 0.25 up passes the gate.**
- **The seat loses more the more it sets up:** to this network, a rule-driven set-up is on
  average a worse move than its own.
- **The default is p = 0.5, max stage 4** (59.5%). It sets up nearly twice as often as the
  network does on its own.
- **For more set-up exposure,** p = 1.0 at max stage 4 (65.5%) still passes.

**Throughput:** a 1,000-game row took about 75 s in one process (64 battles at once).
- No crashes, and no opponent-seat episode reached the learner in any run.
- The seat's CLEANUPs matched its counted games (one still in flight when a run stopped).
- 1 refused choice in about 8,000 games.

**The other kinds, live** (2 pairs x 8 battles, one of them the opponent pair, 40 games each):
- **psbot:** PSBot played 1,355 of the seat's 1,357 decisions (2 were in flight at the stop);
  the network won 80%.
- **foulplay at 3,000 iterations per world:** a small N, only to check the plumbing. Foul Play
  played 1,348 of 1,356 decisions, every answer at the fixed N, with 2 helpers; the network won
  80%.
- In both, nothing of the seat reached the learner, there were no crashes, and the self-play
  pair's and the trained seat's episodes all did.

## Wiring into train.py (done)

train.py now takes the flags (`opponent_seats.add_arguments`), refuses a bad value before Ray starts
(`from_args`), logs `describe_plan` at the start, builds each worker from `worker_class(entry)` /
`worker_kwargs(entry)` when there is a plan (without one the workers are built exactly as before), and
logs `report_line` from the workers' heartbeats every 30 s. The snippet it follows:

```python
import opponent_seats
opponent_seats.add_arguments(ap)                                   # in parse_args
seats = opponent_seats.from_args(args)                             # None without --opponent-seats; ValueError on a bad value
plan = opponent_seats.plan_seats(pairs_per_worker, rooms_per_pair, seats) if seats else None
if plan: logger.info(opponent_seats.describe_plan(plan))
for i, port in enumerate(server_ports):
    entry = plan["workers"][i] if plan else None
    Remote = ray.remote(num_cpus=1)(opponent_seats.worker_class(entry)).options(max_restarts=-1, max_task_retries=0)
    Remote.remote(cfg=cfg_ref, inference_actor=infer, learner_actor=learner, pairs=count, server_port=port,
                  **opponent_seats.worker_kwargs(entry))
# telemetry: logger.info(opponent_seats.report_line([b.get("opponent_seats") for b in worker_beats]))
```

**Without `--opponent-seats`,** `worker_class` is `worker.RolloutWorker` and `worker_kwargs` is
empty, so the run is today's.

**Foul Play's N:** `--seat-foulplay-iterations` is required with a foulplay share; Stage 0 picks
it. It is iterations per sampled world. `SeatStats.extra["iterations_not_fixed"]` counts answers
whose helper did not search exactly N.

## Tests

`tests/test_opponent_seats.py` (on the box: `python -m pytest tests/test_opponent_seats.py -q`) covers:
- **the plan:** shares, rounding, alternation, determinism, the Foul Play cap, the flags;
- **the rules:** on poke-env Battles built from a real request;
- **the override:** p, switches, agreement, the seed;
- **the seats on the seven recorded battles:** setupnet puts STEP per decision + one CLEANUP;
  PSBot (a stand-in engine) and Foul Play (the fake helper, fixed iterations) put nothing while
  they answer, and STEP + CLEANUP when the network moves; never a DONE;
- **the worker's own rollout loop:** the trained seat's battle reaches the learner, the setupnet
  seat's does not;
- **the fallbacks** to self-play, and the log line.
