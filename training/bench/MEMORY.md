# Memory of a training run

Written 2026-10-06, after the cloud run (44 servers x 64 battles, release r1) lost two rollout
workers to Ray's memory monitor at 472 of 497 GB, 5.5 hours after its trainer started. Three
sittings; the last one (from 03:40 UTC) made no training run on the laptop (the owner forbade
them at 03:30 UTC) and says so wherever a figure comes from an earlier one.

## Answer

- **Nothing is kept per finished battle, and the servers do not grow.** r1 already drops a
  battle that ends with a result. Most of the box's 44 workers stay at 0.45 to 0.55 GB for hours;
  a server is 1.7 GB (all its processes) from ten minutes after its start, flat.
- **What fills the box is a few workers that swell**, one after the other: 0.5 GB to tens of GB
  at 2 to 4 GB a minute (the largest seen: 100.2 GB), back down, 3 to 15 GB kept. One Python
  thread of such a worker runs at 100 % and its server's `sockets.js` carries several times the
  traffic of the others, with no more work in the simulator.
- **The cause found: one choice the server refuses ends in a flood of refusals** between a
  worker's player and its server, thousands a second, every one of them a handler task waiting
  on the room's lock (about 3 KB each). Reproduced on the laptop twice with scripted players
  and r1's player code on a real server (43,122 refusals in 16 s, 34,684 in 8 s) and once inside
  a training run (28,280 in 18 s, before the rule). With the changes below the same refused
  choice gets one answer and nothing follows.
- **Not proven: that the box's swollen workers are this flood.** Nothing inside the box's
  processes could be looked at. What speaks for it is the signature above, which a flood has and
  nothing else found here has. What does not fit yet: on the laptop a flood lasted 8 to 18
  seconds and cost about 0.1 GB; on the box a worker's thread has been seen at 100 % for 16
  minutes and the worker at 39 GB. Why it lasts that long there is a guess (below), not a
  measurement. **The first hour after the restart with the new code settles it** (what to look
  at is at the end).
- **A second defect, fixed and tested:** a room the server closes after five minutes was never
  given back (battle, seat, episode slot, steps, inference history: for the life of the worker).

## The box, read only

`RssAnon` of every `ray::RolloutWorker`, RSS of the node processes by role, `ps`/`top` CPU time.

Second trainer (started 2026-10-05 21:27:18 UTC, r1):

| UTC | after start | node used | workers over 1 GB | the largest (GB) |
|---|---|---|---|---|
| 23:31:43 | 2 h 04 | 221 GB | 8 | 54.9, 35.3, 18.0, 7.7, 7.0, 6.6, 4.5, 1.2 |
| 23:36 to 23:38 | 2 h 10 | 223 -> 236 GB | | one at 67.3 -> 70.9, one at 31.8 -> 36.5 (1 GB every 15 s), the 35.3 one back at 3.8 |
| 23:53:04 | 2 h 26 | 334 GB | 11 | 85.9, 81.9, 24.7, 14.1, 8.6 |
| 00:02:22 | 2 h 35 | | | the 85.9 and the 81.9 both at 13.9 |
| 01:29:21 | 4 h 02 | 300 GB | 18 | 100.2 (a worker that was at 0.45 at 23:53), 14.8, 13.7, 10.8, 9.9 |
| 01:52:46 | 4 h 25 | 306 GB | 22 | 40.8, 29.2, 29.2, 22.2, 13.5 (all workers together 225 GB) |

It reached its planned restart at 6 hours (below) without a kill. Third trainer (started
03:29:37 UTC, r1 again, the same thing from the start):

| UTC | after start | node used | workers over 1 GB | the largest (GB) | episodes, decisions |
|---|---|---|---|---|---|
| 03:44:39 | 15 min | 86.9 GB | 0 | 0.51 (all 44 between 0.44 and 0.51, 20.2 GB together) | 5,934,813 and 201,062,325 |
| 03:56:25 | 27 min | 88.0 GB | 1 | 1.42, 0.55 | 6,031,169 and 204,458,374 |
| 04:08:13 | 38 min | 89.4 GB | 1 | 2.02, 0.56 | 6,128,692 and 207,853,749 |
| 04:21:04 | 51 min | 123.6 GB | 3 | 30.25, 1.96, 1.25 | 6,228,349 and 211,323,393 |
| 04:23:05 and :35 | 53 min | | | the same worker at 38.5, then 39.4 GB: 1.8 GB a minute | |
| 04:32:19 | 63 min | 113.5 GB | 2 | 22.36 (the same worker, on its way down), 1.95 | 6,321,516 and 214,608,504 |
| 04:36:40 | 67 min | 98.3 GB | 3 | 6.19, 2.54, 1.95 | 6,356,977 and 215,863,671 |

That worker at 04:23: 58 threads, one at 99.9 % with 16 min 46 s of CPU time (the whole process
18 min 05 s in 53 minutes), the others idle. The busiest `sockets.js` had used 8 min 15 s of
CPU, the next two 4 min 17 s and 1 min 52 s, the mean of the 44 about a minute (which one is
that worker's server was not read). Steps per episode stayed 34.8 to 35.2 over these
samples and the run made 4,700 to 4,800 decisions a second.

The other workers of the second trainer: 0.46 to 0.54 GB each at every sample, CPU time 32 to
33 minutes in 4 hours. The swollen ones: 45 to 90 minutes of CPU, all of it in one thread.

Servers, 44 of each, sum of RSS:

| role | 23:31 | 23:53 | 01:29 | per server per hour | 15 min after the 03:29 restart |
|---|---|---|---|---|---|
| `pokemon-showdown start` (the server) | 22.54 GB | 22.54 | 22.50 | 0 | 20.38 |
| `room-battle.js` (the simulator) | 24.81 | 24.89 | 24.99 | 2 MB | 24.29 |
| `sockets.js` | 5.78 | 6.44 | 8.08 | 27 MB on average, all of it on some servers (largest 248 MB) | 4.02 |
| the six others | 22.95 | 22.95 | 23.03 | 0 | |

That is 1.7 GB a server, flat. It is Showdown's own doing and it has a ceiling: a battle room
lives 10 minutes after its last line once it is empty (`TIMEOUT_EMPTY_DEALLOCATE` in
`server/rooms.ts`), so each server holds its last ten minutes of battles, in the server process
(the rooms) and in the simulator (the battles). The figures in the task ("the node processes
went from 69.6 to 70.4 GB in 127 s") were taken 11 minutes after a restart, while that was
still filling. Nothing in `rlspawn.ts` keeps anything: its only state is one timer per pair.
Nothing to fix on the server side, and nothing was changed there: no rebuild.

CPU time of the three processes of each server at 01:37 (4 h 10 after start):

| | `sockets.js` | the server | the simulator |
|---|---|---|---|
| 26 servers | 260 to 289 s | 668 to 722 s | 1,726 to 1,803 s |
| 18 servers | 310 to 2,031 s | 730 to 1,630 s | 1,636 to 1,763 s |

18 servers have carried up to seven times the message traffic of the others, and their
simulators have done no more work than the others (a little less). 18 is also the number of
workers over 1 GB eight minutes earlier. (Which server belongs to which worker was not read.)

## What was found in worker.py

### 1. A refused choice ends in a flood of refusals (fixed)

worker.py patches poke-env's message handler: every `|error|[Invalid choice]` is answered with
`/choose default`, at once, before the handler takes the room's lock. poke-env's own answer
comes after it, under the lock: it asks the player again (`choose_move`, a step for the worker)
and sends that choice. So r1 answers every refusal twice.

What follows, seen second by second with scripted players on a real local server
(`bench/choice_flood.py`: r1's `SyncBridgePlayer`, its patch and its message handling; a move is
"wait, then a random legal move", a second for side a, 0.05 s for side b; one choice of side a
is made invalid; no learner, no inference actor, no Ray):

1. The server takes the default. The second choice, a decision later, lands on the NEXT turn
   (the other side had already moved): the client is now one decision behind the server.
2. Being behind, some of its choices are refused. Each refusal adds a default and one more
   asking that waits on the room's lock: 2 handlers waiting after the refused choice, 29 half a
   minute later, 14 refusals in all.
3. The game ends on the server (the other side's player has its `|win|`). This side's `|win|`
   waits in the lock's queue behind those askings, one decision each. Everything this side
   still sends is refused ("too late", "nothing to choose"), the patch answers each refusal
   with another default, and that goes round: 71 refusals in the first second, then 1,105,
   2,209, 3,090, up to 4,478 a second, every one a handler task waiting on the lock.
4. 15.7 seconds later the `|win|` comes through, the player leaves the room, and it is over:
   43,122 refusals, 40,304 handler tasks at the peak, +133 MB (about 3 KB a task), +47 MB kept.

| scripted players, one refused choice | r1 | r1, second run | this tree |
|---|---|---|---|
| refusals, and defaults sent | 43,122 | 34,684 | 1 |
| most in one second | 4,478 | 6,788 | 1 |
| handler tasks alive at once | 40,304 | 34,668 | 1 |
| private memory before, peak, after (MB) | 1,727, 1,860, 1,774 | 1,727, 1,834, 1,752 | 1,727, 1,728, 1,728 |

It does not always come to that: with both sides at 0.3 s r1 bounced 10 times and stopped, and
with side b at 3 s the second choice simply replaced the default (1 refusal). The flood needs a
client that has fallen behind and a game that ends meanwhile.

In a training run (the second sitting, before the rule: 2 servers, 128 battles, r1's worker.py,
one choice made invalid in one battle at 110 s): defaults sent 9, 65, 72, 88, 1,227, 4,582,
8,531, 15,578, 23,857, 28,280 at two-second steps; 23,861 handler tasks alive at the peak; the
battle went to the learner 18.7 s after the refused choice **with 66 steps for 22 turns**, a
loss. Every asking-again is recorded as a step, though its choice is not what was played. The
first version of the fix (no default for an error that says nothing can be chosen, poke-env
still asking again) stopped the flood but not that: 106 steps for 26 turns. With the fix as it
is now, three choices made invalid in three battles: 3 defaults, episodes of 32, 29 and 33
steps for 28, 24 and 28 turns.

With bare websocket clients (the first sitting): one stray `/choose default` from a side with a
wait request while the other side holds its forced switch for 3 s is bounced 7,574 times, 2,525
a second. The server answers these from `room-battle.ts` in the server process: the simulator
never hears of them.

Why a flood would last 16 minutes on the box and not 16 seconds: not measured. It lasts for as
long as the askings ahead of the `|win|` take, one decision each, and a decision is slow in a
worker whose one event loop is busy with thousands of refusals a second (it also serves the
worker's other 63 battles). On the laptop 30 askings took 16 s. Neither the number of askings
nor the time of one could be read on the box.

**The fix.** A refused choice gets exactly one answer, `/choose default`, and only when a
default can be taken:

- not for the four errors that say this side can choose nothing at this moment
  (`worker.NOTHING_TO_CHOOSE`): a default is one more choice and gets the same error back;
- not twice for one request (`default_is_due`), so no error text, known or not, can go round;
- and the training player does not ask the policy again (`SyncBridgePlayer.
  _handle_battle_message` keeps these errors from poke-env): the server has the default.

`evaluate.py`, `ladder.py` and `play.py` do not import worker.py: nothing changes for them, nor
in what they read from `player.battles`.

How often a first refused choice happens in training is not known. The two 18-minute laptop
runs of the first sitting saw none in 256,000 decisions each. The box swells its first worker
within half an hour (16.7 million decisions an hour).

### 2. A room the server closes is never given back (fixed)

`rlspawn.ts` closes every battle room that is five minutes old and not finished: `|expire|`,
`|deinit`, no `|win|`. poke-env reports a finished battle only on `|win|` or `|tie`, and the
repair loop that was meant to find such battles is in `_handle_query`, which poke-env 0.15 never
calls (the `/rlactive` answers are logged as unhandled and dropped). So r1 keeps, for the life
of the worker: the Battle, its seat in poke-env's count of running battles (`rooms_per_pair`
seats a player; when all are taken the next battle's `|init|` waits for a seat forever and holds
its room's lock, and everything the server sends that room queues behind it), its episode slot
(`learn_max_pending_episodes`), its steps in `_traj` and its history in the inference actor.

Seen on the laptop (first sitting) in the worker that ran under tracemalloc (slow, so its
battles reached five minutes): after 17 minutes 12 rooms stood behind a held lock, 74 rooms had
lines nobody read, 347 handler tasks were alive (122 in the other worker), 365 of them parked.
With the fix, same run: 20 battles given back, 0 parked, 128 locks, 12 rooms with unread lines.

Now `|deinit` for a battle without a result gives everything back through the worker's own
CLEANUP event. The learner gets nothing for such a battle, as in r1 (r1's steps for it sat in
`_traj` and never left).

## What the learner is handed

Checked against release r1's worker.py (from git, commit 8227671) on recorded battles of a real
server, with the real players, room locks and rollout loop, Ray replaced by a function of the
observation (`tests/test_worker_memory.py`, `tests/test_worker_choice_storm.py`, 42 tests):

| A battle ends with | r1 | now |
|---|---|---|
| a win, a tie, a forfeit (ppo, warmup, imitation) | one episode | the same episode, array for array; the same messages to the server; nothing kept, as in r1 |
| the room closed by the server | no episode; battle, seat, slot and steps kept | no episode; nothing kept |
| the worker stopping | no episode | no episode; the steps so far are the same |
| closed before its first decision | | no slot given back that was never taken |
| **a choice the server refused during it** | the battle's episode plus one step for every asking-again (1 in the tests; 66 steps for 22 turns and 106 for 26 in the training runs above, where a battle without a refusal has little more than a step a turn) | **the episode of the same battle without the refusal, array for array** |

The last row is the one place where the learner is handed something else than r1 hands it, and
it is on purpose: r1's extra steps are observations of a state with actions that were not
played, as many of them as the flood's timing happened to make. There is no way to stop the
flood and keep them. If r1's steps must stay as they are for such battles, this change is not
the one to deploy; nothing else in it depends on that row.

`choose_move`, `_battle_finished_callback`, the rollout loop, `_finalize_trajectory`,
`_cleanup_trajectory` and the learner client are r1's code to the letter
(`tests/test_ppo_health.py::test_worker_is_r1_but_for_run` lists what differs).
The Foul Play opponent share is not in this tree's worker.py, nor in r1's.

Two differences in what is played, not in what is recorded. r1's second choice after a refusal
replaced the default or was taken unseen as the next turn's move; now the default stands. And
a side that r1 left a decision behind the server played every later turn on the observation of
the turn before; that no longer happens.

## Before and after on the laptop

Per worker per 1,000 episodes, from the first sitting (before the rule; 2 servers, 128 battles,
4 pairs a worker as on the box, resumed from `learner_update_002575.pt`, 18 minutes each, the
first 3 left out). "Private" is the commit charge (Windows trims the working set, so RSS is not
comparable). Worker 8800 ran under tracemalloc in both runs and made a tenth of the episodes.
The second column has the room fix only; the choice fix came later and was not run for 15
minutes (its 16-minute run with three refused choices: workers +19 and +7 MB over 13 minutes,
8.4 and 3.3 MB per 1,000 episodes, no flood).

| | r1 worker.py | with the room fix |
|---|---|---|
| run | 7,839 episodes, 289 decisions a second | 7,366 episodes, 276 a second |
| worker 8801 | 1,173 -> 1,198 MB, 76 MB an hour, 2.75 MB per 1,000 of its episodes | 1,193 -> 1,208 MB, 10 MB an hour, 0.39 MB per 1,000 |
| worker 8800 (slow) | 1,256 -> 1,293 MB, 123 MB an hour | 1,274 -> 1,300 MB, 79 MB an hour |
| the 2 servers | 775 -> 1,020 MB | 977 -> 928 MB |
| the 2 simulators | 735 -> 1,280 MB | 908 -> 1,131 MB |

Neither run met a refused choice, so this table shows that nothing grows per episode, with or
without the changes; it does not show the box's failure, nor that it is gone. A few MB per
1,000 episodes in runs this short is the noise of a 1.2 GB process (the workers of the second
sitting's five- to seven-minute runs read between -15.7 and +38.5). The server and simulator rows
are the ten-minute filling described above, not a leak. The before and after of the failure
itself is the scripted table further up: 43,122 refusals and +133 MB in 16 seconds, against 1.

## The planned restart

It has now run for real on the box, 6 hours after the second trainer's start (`pipeline.log`):

| | the box, 2026-10-06 03:27:17 UTC |
|---|---|
| reason | `planned restart after 6 hours` |
| trainer stopped through its stop file | exit code 0, `graceful: true`, `killed: false`, 92.7 s |
| final checkpoint confirmed | `learner_update_005335.pt` (196,483,375 decisions) |
| servers | stopped at 03:28:49, started at 03:28:52, all 44 answering at 03:29:35 |
| next segment | trainer started 03:29:37, `[learner] resumed from .../learner_update_005335.pt`, "Training started" 03:32:40 |
| time without training | 5 min 23 s |

No failure counted. The timer counts from each trainer start (`_watch` in pipeline.py), and
the config is read once, when the supervisor starts: a changed `watchdog.recycle_hours` takes
effect with `sudo systemctl restart psppo-train`, not before.

On the laptop: `tests/test_pipeline.py` (pipeline.py as a real process against the stand-in
trainer and servers), 120 passed and 1 skipped in 17 minutes, among them
`test_e2e_planned_restarts_are_not_failures` (every planned restart saves on its way out, no
update lost or repeated, none counted as a failure). And once with the real trainer, in the
first sitting, before the rule: 2 servers, 32 battles, `recycle_hours` 0.05: planned restart
after 3 min 01 s twice, exit code 0 in 6.5 s and 9.3 s, `learner_update_002586.pt` and
`...002592.pt` confirmed and resumed from, servers answering 19 s and 30 s later.

## recycle_hours

- Servers: nothing to restart them for. 2 MB an hour each.
- With the changes deployed and the first hour clean (below): the workers are flat too, and
  the box sits near 90 GB of 497. Keep `watchdog.recycle_hours` at 6 for the first day as a
  net (5.4 minutes in 6 hours, 1.5 %), then 24.
- As long as r1 runs: 6 is too long. The first trainer was killed at 5.5 hours, the second was
  at 300 GB after 4 hours with single workers at 100 GB, the third had a worker at 39 GB after
  53 minutes. 3 hours (3 % of the time): the most seen in a first three hours was 334 GB.
  A kill is not a disaster either: the first one cost 76 seconds and no update, the trainer
  saved a checkpoint on its way out.

## To be checked on the box at the restart

None of this could be shown on the laptop without a training run.

1. **One hour in, every rollout worker under 0.6 GB** (`RssAnon`). With r1 the third trainer had
   its first worker over 1 GB after 27 minutes and one at 30 GB after 51; the second had 8 over
   1 GB after 2 hours. Four hours in with none over 1 GB: fixed. One worker in the GBs and
   growing: not fixed, whatever else is true.
   `for p in $(pgrep -f "^ray::RolloutWorker"); do awk -v p=$p '/^RssAnon/{print $2/1048576, p}' /proc/$p/status; done | sort -n | tail -5`
2. **No worker thread at 100 %** (`top -H -b -n 1 -p <pid>`), and the CPU time of the 44
   `sockets.js` processes alike (r1: 18 of them up to seven times the others).
3. **Node "used" flat near 90 GB** from 15 minutes after the start (r1: 124 GB after 51 minutes,
   221 GB after 2 hours).
4. With `"trainer": {"env": {"PSPPO_MEMWATCH": "/home/ubuntu/runs/main/logs/memwatch",
   "PSPPO_MEMWATCH_SECONDS": "300"}}` in run.json every worker writes its counters there:
   `client_tasks` stays in the hundreds (a flood: tens of thousands); `defaults_sent` and
   `choices_refused` say how often a choice is refused at all (expected: a handful an hour;
   thousands would mean the policy's legal-move mask is wrong, another matter);
   `defaults_withheld` counts refusals that r1 would have bounced; `rooms_closed` counts
   battles the server closed at five minutes (unknown on the box; a large number would mean
   battles that stall, worth knowing).
5. **Steps per episode and decisions a second as before** (34.8 to 35.2 and 4,700 to 4,800 in
   the trainer's Telemetry lines under r1), and the update line's kl and clip in their usual
   range. A drop in steps per episode of a few tenths would be r1's extra steps going away.

## Tests

- `tests/test_worker_memory.py` (17) and `tests/test_worker_choice_storm.py` (25): 42 passed.
  Until the last sitting a run of the first file never left: its r1-out-of-seats test parked a
  coroutine on poke-env's loop for good, and poke-env's exit hook waits, spinning, for every
  task of that loop (two such runs sat at a full core for half an hour here before they were
  noticed and stopped). `Harness.feed` now takes a stream that cannot go on off the loop, and
  the run leaves two seconds after its last test.
- `tests/test_ppo_health.py`: 31 passed. `test_under_ray_a_dead_loop_makes_train_py_exit_with_1`
  was left out in the last sitting because it starts Ray; it passed in the second sitting, and
  what it tests (`RolloutWorker.run`, `_guarded_rl_loop`) has not changed since.
- `tests/test_speed_*.py`: 85 passed, 6 skipped. `tests/test_pipeline.py`: 120 passed, 1 skipped.

## Tools

- `bench/choice_flood.py`: the scripted players above (`--worker r1` or `new`, `--report`).
- `memwatch.py`: `PSPPO_MEMWATCH=<dir>` makes every rollout worker write its memory, the size of
  every per-battle container, live objects by type and the counters above every 30 s
  (`PSPPO_MEMWATCH_SECONDS`). `PSPPO_MEMWATCH_TRACE=<port>` adds tracemalloc for one worker (it
  slows that worker tenfold).
- `bench/memsample.py --roots <pids> --out f.jsonl`, then `--report f.jsonl --memwatch <dir>
  --log <trainer log>`: every process under the trainer and the server launcher, by role.
