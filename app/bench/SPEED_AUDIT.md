# Speed audit: where the time goes in training, and what would make it faster

Written 2026-10-05 from the code and from small measurements on the Windows laptop (RTX 4060
Laptop 8 GB, Core Ultra 7 155H, three other jobs running beside it). Nothing here was measured
on the cloud box, and no training run was started for it. The one end-to-end number, 388
decisions a second, comes from the saved output of the 12-minute PPO run of the same day.

The tools are next to this file: `throughput.py` (the measurement to run on the box),
`micro_showdown_sim.py`, `micro_obs.py`, `micro_psbot.py`, `micro_serialize.py`, `micro_gpu.py`
and `micro_lr_schedule.py`. No existing file was changed. The two exact GPU changes and the
per-turn attention exist as working, checked reference code inside `micro_gpu.py`.

## The short version

| | Finding | Evidence |
|---|---|---|
| 1 | The learner spends 38% of every minibatch building an attention mask it could build for almost nothing. `create_block_mask` materialises a 16,384 by 16,384 boolean tensor per packed sequence, every minibatch. Building the same `BlockMask` from the episode ids at block level gives bit-identical outputs and a minibatch 1.4 to 1.5 times faster, with a third of the memory. | 181 ms to 119 ms, and 198 to 137, back to back in one process. Outputs compared with `torch.equal`. |
| 2 | One inference call costs 23 to 25 ms at the batch the laptop ran (about 30), and a call for a single decision already costs 15: most of a call is fixed overhead on one Python thread. The same mask builder (6 to 7 ms), 96 tiny tensor clones to store the per-battle cache (4.4 ms), the encoder's many small operations. The cheap mask and a pooled cache take the call to 10 to 13 ms. The overhead does not shrink on a faster GPU. | Breakdown with the GPU synchronised around each part. |
| 3 | A rollout worker is one Python thread, and 84% of what it does per decision is `ObservationAssembler.assemble` (1.30 of 1.56 ms). 620 property lookups into poke-env's move table per decision, for values that never change. | Replay of 10,280 recorded decisions through the real worker code. |
| 4 | In the imitation phase the teacher is the limit: one PSBot engine process per worker, 8.3 ms per decision, 121 decisions a second at best. | Replay of 3,480 requests into one engine. |
| 5 | Mixed precision is already in place, in the stronger form: the weights themselves are bfloat16. Standard autocast on float32 weights would be 19% slower. There is nothing to gain there. | Same minibatch three ways. |
| 6 | The learning rate falls 27 times in one optimizer step at step 501,001, which is decision 194 million under the upstream defaults: 78% of the way through a 250-million-decision run. | The scheduler built by the learner's own code. |

What to do on the cloud box, in order: measure the baseline at 16 ports with `bench/throughput.py`,
make the two exact GPU changes (rows 1 and 2), give the imitation phase more engines, then look
at the CPU table `throughput.py` prints to decide whether the workers or the GPU is the next limit.

## 1. The loop, with references

Line numbers are as of this audit (train.py had uncommitted edits by another task at the time).

### One decision

| Step | Where | What happens |
|---|---|---|
| Simulator | `server/room-battle.ts:1324` `RoomBattleStream._write`, `sim/battle-stream.ts:49`, `sim/battle.ts:3267` `sendUpdates` | One child process per server runs every battle of that server. A choice is applied, and when both sides have chosen the turn runs. Output is one `update` chunk, then one `request` per side. With `debugsimprocesses` on (it is, in config-example.js), every choice first sends the battle's whole input log to the parent (`room-battle.ts:1326`). |
| Server main | `room-battle.ts:731` `listen`, `:762` `receive`, `rooms.ts:1972` | Adds each line to the room log, broadcasts, parses and re-serialises each request to stamp its id. No throttle: `--no-security` sets `nothrottle` (`config-loader.ts:35`). |
| Sockets process | `server/sockets.ts:249` | Splits the update per player and writes websocket frames. Nagle is off (sockjs sets no-delay). |
| Websocket to poke-env | `poke_env/ps_client/ps_client.py:246` `listen`, `:156` `_handle_message`, patched by `worker.py:55` | One asyncio task per message, split on newlines and bars, one lock per battle. All players of a worker run on poke-env's single loop thread. |
| Parsing | `worker.py:202` `_handle_battle_message`, `poke_env/player/player.py:258` | The worker keeps the turn's protocol lines for the observation (`worker.py:224`), then poke-env updates its `Battle` line by line, and for a request parses the JSON and calls `_handle_battle_request`. |
| Observation | `worker.py:237` `choose_move`, `obs_assembler.py:90` `assemble`, `rl_reward/potential.py:49` | 8,567 numbers as float16 (17 kB): 12 Pokemon bodies, 48 moves, global, last-turn events, the 14-action mask. Then the reward potential. Then a future is parked and a `STEP` event goes on a queue (`worker.py:275`). |
| Worker's second thread | `worker.py:494` `_run_rl_loop`, `:457` `step` | Blocks for the first event, sleeps 3 ms, drains the rest, stacks them, casts to float32 (`:525`), and makes one blocking Ray call for the lot (`:531`). |
| Inference actor | `inference.py:184` `infer_batch` | A plain synchronous Ray actor: one call at a time, one call per worker, no batching across workers. `RolloutConfig.infer_min_batch`, `infer_max_batch`, `infer_wait_ms`, `infer_max_pending` and `InferenceConfig.server_min_batch`, `server_max_wait` are read by nothing. Per call: gather each battle's cache from per-tag lists and concatenate (`:142`), forward, value variance, masked sample, store the new cache with three clones per battle (`:168`), four copies to the CPU. |
| Model, inference path | `ppo_core.py:664` `forward`, `:511` `encode_features`, `:594` to `:629`, `:631` `readout_policy` | 64 tokens for the turn (field, 12 Pokemon, 48 moves, register, actor, critic) plus 64 history tokens: the field token's input to each layer for the last 64 turns of that battle. That history is what the code calls the key-value cache. A block mask is built for every call (`:601`). All 128 tokens go through every layer, although the 64 history rows only serve as keys and values and their outputs are thrown away (`:624`). |
| Back | `worker.py:458`, `player.py:354`, `room-battle.ts:617` `choose` | The future is set from the other thread, the order is mapped and sent, the server hands it to the simulator. |

End of a battle: `worker.py:560` `_finalize_trajectory` stacks the episode and computes the shaped
reward, `SyncLearnerClient` (`worker.py:121`) sends up to 32 episodes per call. Its queue holds 160
episodes and silently drops the oldest when full (`worker.py:131`).

### One learner update

| Step | Where | What happens |
|---|---|---|
| Intake | `learner.py:401`, `:202` `_loop`, `ppo_core.py:808` `gae_from_episode` | Advantages per episode in a Python loop over steps, then the steps are appended to a CPU buffer. |
| Trigger | `learner.py:267` | An update starts when the buffer holds `steps_per_update` steps, and it uses everything in the buffer. |
| The update blocks the actor | `learner.py:327` `_perform_update` | It is `async def` but calls `ppo_update` synchronously, so the actor answers nothing while it runs: no intake, no `get_stats`. The trainer's telemetry line therefore only appears between updates, and the workers' episodes arrive in a burst afterwards. |
| Batching | `ppo_core.py:783` `iter_minibatches` | Whole episodes in shuffled order, concatenated, cut every `minibatch_size` steps. What does not fill a minibatch is dropped (`:799`). Each minibatch is viewed as `minibatch_size / batch_seq_len` sequences of `batch_seq_len` turns (`:579`), so an episode can be cut by a sequence boundary and loses its earlier history there. |
| Size | | Upstream: 36,864 steps, minibatch 768 = 3 sequences of 256 turns = 3 x 16,384 tokens, 48 minibatches an epoch, 2 epochs, 96 optimizer steps. Laptop: 16,384 steps, minibatch 256 = 1 sequence, 64 minibatches an epoch, gradient accumulation 3, 44 optimizer steps. |
| Attention in training | `ppo_core.py:584`, `:187` `pokemon_batched_episodic_mask` | A token attends to the 64 tokens of its own turn and to the field tokens of up to 64 earlier turns of the same episode. The mask is built by evaluating that rule on every token pair. |
| Loss and step | `ppo_core.py:1077` to `:1102` | Clipped PPO loss, two-hot value loss, entropy. Gradients go into float32 master weights (`:925`), are clipped, stepped with AdamW, and copied back into the bfloat16 weights (`:937`). In epoch 2 the update stops early when the mean KL passes 1.5 x `target_kl` (`:1110`). |
| Publishing | `learner.py:365`, `inference.py:234` | Weights go to a small store actor. The inference actor looks at most every 5 s, and when it loads new weights it clears every battle's history (`inference.py:253`). |

### Precision

`PokeTransformer.enable_bf16_recurrent_path` (`ppo_core.py:482`, the conversion at `:501` to `:506`)
converts the parameters themselves to bfloat16: all seven embedding tables, the four subnets,
the three transformer layers, the readout, both heads, and the three learned tokens. It is called
once when each actor builds its network (`inference.py:121`, `learner.py:443` and `:446`), before
any forward. Left in float32: `action_emb` and `jepa_predictor` (JEPA modes only) and the value
support buffer. Logits are cast to float32 before the loss (`ppo_core.py:640`, `:1078`). So both
the forward and the backward run in bfloat16 with no autocast involved, and the optimizer keeps
float32 masters so small updates are not rounded away.

### What is compiled

Only `flex_attention`: both actors replace `ppo_core.flex_attention` with `torch.compile` of it
(`inference.py:119`, `learner.py:198`). The encoder, the feed-forward blocks and the readout run
eagerly. Two things to know. Past torch's recompile limit of 8, `torch.compile` gives up on the
function and runs it eagerly from then on, silently, and eager flex attention materialises every
token pair: the learner path then needs tens of gigabytes. This happened inside the benchmark
(many dtypes and masks in one process) and is why it raises the limit. And flex attention has no
backward on a CPU at all.

### How the GPU is shared

`train.py:149` gives the inference actor `num_gpus=0.35` and the learner 0.65. That is Ray's
bookkeeping only. They are two processes with two CUDA contexts on one device, time-sliced by
the driver, with nothing giving inference priority.

On the laptop, from the saved 12-minute PPO run (`bench/throughput.py --parse-log`): 388
decisions a second after minute five, one update every 42.3 s, the learner deaf for at least 32 s
of each (76%). The same 128 minibatches take 23.3 s with the GPU to itself (128 x 182 ms), so
sharing costs the learner about 1.5 times.

What an update does to inference there: little. Allowing for when each counter in a telemetry
line was sampled, inference requests ran at 324 to 455 a second (mean 389, 17 intervals) in the
intervals that fall mostly inside an update, and at 413 in the one interval after minute five
with no update running. So on the laptop the update is not what holds inference back. What
does is the shape of the loop: every worker waits for one inference call per cycle, the calls
of all workers are served one after another, and in between the worker's one thread has to
parse and assemble everything that came back. The saved run made 13.3 inference calls a second
over four workers, so one worker went round every 300 ms carrying 29 decisions. That is why the
cost of a call, and not only the GPU's speed, sets the pace. The 662 a second seen in the first
25 seconds is the CPU's short boost, as TRAINING.md says.

Measured directly with two processes on the 4060 (`micro_gpu.py --sections share`, one process
looping learner minibatches, the other timing inference calls of 32):

| | Alone | While the other runs |
|---|---|---|
| Inference call, median (90th percentile, worst) | 22.0 ms (28.1, 32.5) | 31.7 ms (35.6, 41.0) paced at 13 calls a second, 32.0 ms back to back |
| Learner minibatch | 182 ms | 231 ms |

So a running update makes each inference call about 1.4 times slower and never stalls it for
long (worst call 41 ms), and inference makes the learner about 1.3 times slower. That is
Windows time-slicing on this card. Linux, the L40S and MPS may behave differently.

## 2. Measurements

All on the laptop. Single-thread figures move by a factor of up to 1.7 with what else the
machine is doing (the observation benchmark gave 1.56 ms per decision on a quiet run and 2.60 ms
with other jobs busy). Ratios taken inside one process are the reliable part.

### CPU per decision, by part

| Part | ms per decision | How measured |
|---|---|---|
| Rollout worker, message to order, PPO mode | 1.56 (2.60 on a busy machine) | 120 recorded battles, 10,280 decisions, through `PSClient._handle_message` and `SyncBridgePlayer` |
| of which `ObservationAssembler.assemble` | 1.30 | |
| of which poke-env `parse_request` | 0.060 | 1.11 requests per decision (one in ten is a wait) |
| of which reward potential | 0.047 | |
| of which poke-env `parse_message` | 0.044 | 9.6 lines per decision at 4.6 microseconds |
| of which request JSON (orjson) | 0.018 | 2.9 kB per request |
| of which order mapping and creation | 0.005 | |
| of which everything else (split, asyncio, locks, queue) | 0.07 | |
| The same in imitation with SimpleHeuristics | 1.66 | the teacher is 0.085 |
| PSBot engine, per decision, with its feeds | 8.28 | one node process, 3,480 decisions; median round trip 6.1 ms, 90th percentile 17.1 |
| Worker's second thread: stack and cast a batch of 29 to float32 | 0.74 per call, 0.026 per decision | stacking alone, kept as float16, is 0.02 per call |
| Ray round trip of one inference call, idle echo actor, batch 29 (994 kB) | 1.7 per call, 0.06 per decision | 1.5 with float16; 2.5 at batch 128 |
| Ray round trip of one learner submission (32 episodes, 21 MB) | 4.0 per call | plus 5.9 to concatenate |
| Simulator | 0.44 | 300 battles in one node process: 36.0 ms a battle, 81.3 choices a battle |
| Showdown main and sockets processes | not measured | needs a running server; `throughput.py` reports them |

Inside `assemble`, by profile: `encode_moves_inplace` 57%, `encode_pokemon_body_inplace` 35%.
poke-env's `Move.entry` property is called 620 times per decision, `get_combat_bounds` 51 times,
`np.clip` on a scalar 31 times at 3.1 microseconds each, `get_move_metadata` 24 times.

### GPU, RTX 4060, nothing else on it

Inference, `InferenceActor.infer_batch` exactly as coded:

| Batch | ms per call | ms per decision | The same with the cheap mask | and the pooled cache |
|---|---|---|---|---|
| 1 | 14.9 | 14.9 | | |
| 8 | 20 to 21 | 2.6 | 12 to 18 | 10 to 17 |
| 32 | 23 to 25 | 0.75 | 16 to 18 | 10 to 13 |
| 64 | 27 to 28 | 0.43 | 19 to 25 | 14 to 20 |
| 128 | 38 to 40 | 0.30 | 32 to 49 | 22 to 24 |
| 256 | 60 to 68 | 0.25 | 61 to 68 | 39 to 41 |

Where a call of 32 goes (26.9 ms with the GPU synchronised around each part): block mask 6.9,
encoder 4.1, three layers 6.2, readout 1.7, cache store 4.4, cache gather 0.5, transfer and
sampling 3.1. Storing the cache costs 0.14 ms per battle (4.5 ms at 32, 10 to 13 at 128, 27 at
256). One indexed tensor does the gather and the store together in 0.3 ms at 32 and 1.2 ms at 256.

Learner, one minibatch of 256 steps (one sequence of 16,384 tokens), as `ppo_update` runs it:

| Variant | ms per minibatch | Peak memory | Same maths |
|---|---|---|---|
| As coded | 181 to 198 (mask 69, rest of forward 27, backward 74 to 90, optimizer 14) | 3.5 GB | |
| Block mask built from episode ids | 119 to 137 | 1.1 GB | yes, bit for bit |
| Attention per turn with plain matmuls, float32 scores | 149 to 156 | 2.1 GB | yes to 5e-7 in float32 |
| The same, bfloat16 scores | 105 to 107 | 1.3 GB | rounding-level differences |
| Real `ppo_update` on the same data | 184 to 200 | | cross-check of the timing harness |

The per-turn prototype at the upstream minibatch of 768 takes 434 ms and 5.5 GB, so it fits the
laptop where the coded path does not.

Precision, same minibatch at reduced size (128 steps): bfloat16 weights as coded 87.8 ms,
float32 weights 372.8 ms, float32 weights under bfloat16 autocast 104.9 ms. Inference forward at
batch 64: 16.6, 38.7 and 19.9 ms.

### What this says about the GPU budget

Per decision on the 4060, each alone: the learner 2 epochs x 0.71 ms = 1.42 ms, inference 0.86 ms
at the batch of 29 the run had. Together 2.28 ms, which is 439 decisions a second if the GPU
were shared without loss, against 388 measured. With the cheap mask and the pooled cache it is
0.93 to 1.07 plus 0.31 to 0.41, so 1.5 to 1.8 times more on the GPU side.

### Cores per 1,000 decisions a second

PPO, from the parts above on a quiet laptop performance core: worker 1.7, simulator 0.44,
inference actor up to 0.8 (it is one thread and mostly overhead), Showdown main and sockets an
unmeasured few tenths. About 3.3 core-milliseconds per decision, so 3.3 cores per 1,000.

Do not plan with that number. On the loaded laptop a worker delivered 100 to 240 decisions a
second, not the 640 the bare path suggests. A server and its worker form one mostly serial
pipeline: simulator, main, sockets, the worker's parsing thread, its second thread, the
inference call, and back. The saved PPO run moved 97 decisions a second through each of its
four pipelines, about 10 ms of pipeline time per decision, with no single stage full: the
inference actor, at 13.3 calls of about 32 ms, was busy under half the time. The threads inside
a worker share one interpreter lock, the socket and process hops are not in the benchmark, and
the laptop's cores slow down under sustained load. So the number of server and worker pairs is
the first throughput knob, until the inference actor or the GPU fills. Plan the cloud box at
150 to 250 decisions a second per pair until `throughput.py` says otherwise: a worker whose
"busiest" figure is near 1.0 is pinned.

Imitation with PSBot: 8.3 for the engine plus about 2.2 for the worker plus the simulator: about
11 cores per 1,000 on a quiet core, and one engine per worker caps each worker at 121 a second.

### What to expect on the cloud box (estimates, not measurements)

| Limit | As coded | After candidates 2 and 3 |
|---|---|---|
| Inference actor, one thread. Its fixed cost per call is CPU work and will not shrink with the GPU. | About 40 calls a second. At 30 decisions a call that is 1,200 a second, at 64 a call 2,300. | 75 to 100 calls a second at 32 a call (2,500 to 3,200), 50 to 70 at 64 a call (3,200 to 4,600). |
| Learner, if the L40S is 4 to 5 times the 4060 | 0.3 ms of GPU per decision, so 3,000 a second with the GPU to itself | 0.2 ms, 4,500 |
| Workers, at 150 to 250 decisions a second each | 16 ports: 2,400 to 4,000. 24 ports: 3,600 to 6,000. | the same until candidate 5 |
| Imitation with PSBot, one engine per worker, engines slower than on the laptop's fast cores | 16 ports: about 1,200. 24 ports: about 1,800. | candidate 4: several times that, until the 64 vCPUs are full |

So the order in which the box will run out is most likely: the inference actor's thread, then
the learner's share of the GPU, then the workers. That order is a prediction. The first
`throughput.py` run replaces it with a measurement: GPU utilisation, the inference actor's CPU
(1.0 means its thread is full), and the busiest rollout worker.

## 3. The Showdown side

Each server is ten node processes (seen on the running servers, read-only): main, the simulator
(`room-battle.js`), sockets, and seven that do nothing in this workload: team validator,
verifier, friends, two artemis, battlesearch, datasearch. As they stood, not in use at the
time, they held 226 MB per server together (main 87, simulator 84, sockets 7, the seven
others 48).

| Question | Answer |
|---|---|
| Which processes work per battle | The simulator (all battle logic and team generation), main (rooms, routing, request ids), sockets (frames). |
| What sets the simulator count | `exports.subprocesses.simulator` in config/config.js, 1. `network` is the sockets count, 1. |
| More servers or more simulators per server | More servers, because a server comes with a worker and the worker is the tighter limit. One simulator process does about 2,250 choices a second on a laptop core, a worker at best 640 decisions. A second simulator per server would sit idle. |
| A battle against team generation | 36.0 ms of simulator CPU per battle (81 choices at 0.40 ms). Two teams cost 1.56 ms, 4.3%. Setting a battle up, teams and first turn included, is 3.3 ms, 9%. |
| Logging and replays | Off already: `logchat` and `logchallenges` are false, unrated battles are not logged, no replay is saved unless `autosavereplays` is set. Battle rooms never write their log to disk (`roomlogs.ts:130`). |
| Timers | The battle timer only starts with `forcetimer` or a blitz format. Off. |
| Throttles | Chat throttle, connection and battle counters are all off under `--no-security` (`users.ts:1431`, `monitor.ts:188` on). |
| Debug cost | `debugsimprocesses = true` sends about 1 kB of input log from the simulator to main on every choice: 5.5 microseconds on the simulator side (1.4% of a choice) and 9.7 on main. Set it to false. |
| Announcements | `reportbattles` posts every new battle to the lobby, where every worker player sits (`worker.py:114`), and `reportbattlejoins` posts joins into each battle room. A few short frames per battle. Set both to false. |
| The plugin, per battle | `rlspawn.ts:181`: every 50 ms per pair it counts that pair's live battles by walking every room on the server (`:27`), and creates what is missing, at most 50 at a time. A finished battle is replaced within 50 ms. Ended rooms stay in the room table for 10 minutes after both players leave (`rooms.ts:20`), so the walk grows with the finish rate, but it skips ended rooms at once. |
| The plugin's reaper | Any battle older than 5 minutes is destroyed (`rlspawn.ts:229`). The worker also abandons a battle whose side has not acted for 60 s (`worker.py:358`). Both throw away the episode. This is what limits battles per worker: a battle's wall time is sides per worker x decisions per side / worker rate. |

One decision in every 34 has a single legal action: 2.9% (1.1% moves, 1.9% forced switches), the
same from the simulator's requests and from the mask the worker builds.

## 4. Candidates, ranked

"Same maths" means the learner sees the same numbers and takes the same steps. Estimates for the
cloud box are estimates.

| # | What | Where | Gain | Same maths | Effort |
|---|---|---|---|---|---|
| 1 | Baseline and scale sweep on the box | commands only | sets the starting point, likely the largest single step | yes | small |
| 2 | Build the block mask from episode ids | `ppo_core.py:584` and `:601` | learner minibatch 1.4 to 1.5 times faster, one third of the memory, 6 ms off every inference call | yes, bit for bit | small |
| 3 | Keep the per-battle cache in one tensor | `inference.py:142` to `:181` | 4.5 ms off a call of 32, 13 to 27 ms off a call of 128 to 256 | yes | small |
| 4 | More PSBot engines per worker | `psbot_bridge.py:208` `shared_engine`, `:297` | imitation phase 2 to 4 times on the box | yes | small |
| 5 | Static tables in the observation assembler | `obs_moves.py:84`, `obs_pokemon.py:208`, `:97` | worker thread about 2.4 times (1.56 to about 0.65 ms) | yes, checked byte for byte | medium |
| 6 | Batch inference across workers | `inference.py:184`, `worker.py:531` | fewer, larger calls: up to 2 times on inference at 16 workers | same policy, different random draws | medium |
| 7 | CUDA MPS on Linux | launch script | unknown, 0 to 30% | yes | small |
| 8 | Send float16 to inference | `worker.py:525` | 0.7 ms of the worker's interpreter lock per call, half the bytes | yes | small |
| 9 | Hold the update's observations on the GPU | `ppo_core.py:749`, `learner.py:346` | estimated 5% of an update on the L40S | yes | small |
| 10 | Showdown config | config/config.js | 1 to 2% of server CPU | yes | small |
| 11 | Attention per turn without flex attention | `ppo_core.py:226`, `:562` | learner a further 1.1 to 1.25 times, no compile step, less memory | float32: yes. bfloat16 scores: rounding level | large |
| 12 | Advantages without torch scalars | `ppo_core.py:844` | 32.7 to 1.2 microseconds of learner CPU per step | yes, bitwise on the test | small |

Not worth doing: any change of precision (already bfloat16, autocast is slower); skipping
inference for single-action decisions (2.9% of calls, and it changes what the model sees); a
second simulator per server; trimming the idle Showdown processes (48 MB a server).

### 1. Baseline and scale sweep

TRAINING.md's commands are sized for the laptop: 4 ports, 128 battles, minibatch 256 with
accumulation 3. On the box start from the upstream update shape and more workers:

```
python bench/throughput.py --servers start --ps-dir /home/ubuntu/pokemon-showdown --ports 8000-8015 \
    --warmup-minutes 5 --minutes 10 --out /home/ubuntu/runs/bench/ppo_16x32.json -- \
    --mode ppo --battles 512 --rooms-per-pair 16 --steps-per-update 36864 --minibatch 768 \
    --grad-accum 1 --ckpt-dir /home/ubuntu/runs/bench/ckpt
```

Sweep ports 8, 16, 24 and battles per port 32, 48, 64. Read three things from the result: GPU
utilisation, the busiest rollout worker, and the inference actor's CPU. More battles per worker
is the cheap way to larger inference batches, since each worker's call carries whatever sides
are waiting. Rooms per pair only sets how many battles share one pair of websocket connections:
16 and 32 behave alike for speed (it would matter with candidate 4 if engines went per player).
Limits to respect:
at most 110 battles per worker (`learn_max_pending_episodes = 220` sides, `worker.py:390`, and
that semaphore is awaited on one event loop and released from another, so do not let it fill),
and battles short enough for the 5-minute reaper. Ports 9000 and up are for evaluation, and an
evaluation running beside the trainer takes GPU and CPU from it, so measure with it in the state
the long run will have. Risk: none to learning, the update shape is the upstream default.

### 2. Block mask from episode ids

Change: in `condition_tokens`, replace both `create_block_mask` calls. Training: compare episode
ids per turn (256 by 256 per sequence), require key turn at or before query turn and within
`kv_cache_len`, pool to blocks of two turns, and call `BlockMask.from_kv_blocks` with the same
`mask_mod`. Inference: the sequence is exactly one block, always partial, so the block tensors
are constants. `fast_create_block_mask` in `bench/micro_gpu.py` is a working reference (it reads
the ids from the closure only so the benchmark could swap it in untouched code; pass them in
directly). Evidence: every `BlockMask` field equal to torch's, logits and value logits
`torch.equal`, for four training batches (256, 64, 31 and 30 turns per sequence, 2 to 6
episodes, the 31 not filling whole blocks) and four inference turns. Learner minibatch 181 to 119 ms and 198 to 137 ms in the same
process, peak memory 3.5 to 1.1 GB. Risk: none to learning, old checkpoints load (no parameter
changes). It depends on torch's default block size of 128 being a multiple of the 64 tokens per
turn: assert it. Measure: `python bench/micro_gpu.py --sections learn,fastmask`, then
`throughput.py` before and after.

### 3. Pooled cache in the inference actor

Change: one tensor `[slots, layers, 64, 512]` and one length vector on the GPU, a dict from tag
to slot with a free list, gather with one index, store with one indexed assignment, free the
slot in `clear_cache`. `make_pooled_actor` in `bench/micro_gpu.py` is a working subclass.
256 slots are 50 MB. Evidence: 5.0 ms to 0.3 ms for gather and store at 32, and the call as a
whole 16 to 18 ms to 10 to 13 ms on top of change 2. Risk: none to learning. Size the pool for
twice the concurrent battles plus slack, and decide what happens when it is full (the reference
raises). A side effect to keep: new weights clear every history today (`inference.py:253`).

### 4. More PSBot engines per worker

Change: `shared_engine()` returns one engine per process. Make it a pool of N and pick by the
key's hash, or give each player its own engine (a worker has two players per pair). Every
message of a battle side carries the same key, so one side always lands on the same engine and
the engine's per-battle state stays whole. Default N = 1 keeps today's behaviour. Evidence: one
engine answers 121 decisions a second flat out, the laptop runs gave 45 to 55 per worker, and
the worker's own thread can do several times that. Risk: none to learning (same teacher, same
answers). Each engine is a node process of about one core when busy, so on the box N x workers
must fit the 64 vCPUs. Measure: `throughput.py` with `--mode imitation --teacher psbot`, the
role `psbot_engine` shows their CPU.

### 5. Static tables in the assembler

Change: everything `encode_moves_inplace` writes for a move depends on the move id alone, except
PP, the STAB flag, the availability flag and the owner and slot one-hots. (poke-env also keeps a
base-power override on the move object, set from the raw id of Hidden Power variants: key the
table on what `Move.base_power` returns, not on the id alone.) Precompute one 85-number row per
move id and write it with one slice assignment, then the few dynamic fields. Compute
`get_move_metadata` once, not 24 times per decision. In the body encoder cache the three stat
bounds per (species, level, item, ability, revealed moves) and replace `np.clip` on scalars with
`min` and `max`. Evidence: the profile above and the primitive costs (writing one 85-number row
from a table takes 0.5 microseconds, against 130 ns per scalar write and 0.9 to 4 microseconds
per helper call today).
Expected `assemble` 1.30 to about 0.4 ms. Risk: the observation format must not move by a bit.
The check exists: `python bench/micro_obs.py --streams ... --out <dir>` saves 6,000 real
observations, and after the change `--no-save --compare <dir>/obs_samples.npz` must report
`identical: true` (it does today, run twice). Record the streams with
`bench/micro_showdown_sim.py --record 150`.

### 6. Inference batching across workers

Change: make `infer_batch` put the request on a queue and return a future, and let one loop
drain the queue into a single forward (a tag is never in two requests at once, each side has one
pending decision). The dead `infer_wait_ms` and `infer_max_batch` are the natural knobs. Gain:
the fixed cost of a call is paid once per forward and not once per worker. On the 4060 after
changes 2 and 3, four calls of 32 cost 40 to 52 ms and one call of 128 costs 22 to 24. On the
box the fixed part is a larger share. The same effect is partly available for free by running
more battles per worker. Risk: the policy is the same but actions are sampled in a different
order, so runs are not reproducible against old ones, and one slow forward now delays every
worker. Do it only if the baseline shows the inference actor near one full core with the GPU
not full. An idea in the same direction that was not tried: compiling the encoder and the
readout too (they run eagerly, about 6 ms of small operations per call).

### 7. CUDA MPS

Change: start `nvidia-cuda-mps-control -d` before the trainer on the box, so the two actors'
kernels can overlap instead of taking turns. No code. Linux only, so not measured here.
Measure with `throughput.py`, same arguments, with and without.

On the other ways to share the GPU. Pausing inference while an update runs would only lose
decisions: on the laptop an update barely slows inference as it is. There is no priority to
set between two processes without MPS. Update size (`steps_per_update`) does not change the GPU
work per decision, which is two epochs over every step either way. It changes how long the
learner is deaf at a stretch and how stale the acting policy is, so leave it at the upstream
36,864 unless the learning itself is being studied.

### 8 to 10, 12. Small exact ones

Float16 to inference: `worker.py:525` casts the stacked batch to float32 (0.74 ms at 29, on the
interpreter lock the parsing thread needs) and the model then casts it to integers and bfloat16
anyway. The learner already feeds float16. Observations on the GPU: `AsyncEpisodeDataset.add_steps`
forces observations to the CPU (`ppo_core.py:749`) and `_perform_update` builds a second dataset,
which copies the 632 MB buffer again and then ships 13 MB per minibatch through pageable memory.
One transfer per update and indexing on the GPU is the same data in the same order. Showdown
config: `debugsimprocesses`, `reportbattles`, `reportbattlejoins`, `reportjoins` to false in the
box's config/config.js. Advantages: the loop at `ppo_core.py:844` does five torch scalar
operations per step and costs 32.7 microseconds per step (measured, 40-step episode), which at
3,000 decisions a second is a tenth of the learner's thread. The same recurrence on float32
NumPy scalars costs 1.2 and gave bitwise equal advantages and returns on the test episode.

### 11. Attention per turn

Change: compute each layer per turn: queries are the turn's 64 tokens, keys and values are those
64 plus the field tokens of the earlier turns, gathered with a sliding window in training and
from the cache at inference. Plain matmul, tanh cap, softmax. `dense_train_forward` and
`dense_infer_forward` in `bench/micro_gpu.py` are working references using the model's own
weights. Evidence: against the compiled path in float32, logits differ by 5e-7 and gradients of
all 92 trained tensors by 4e-8, with a 64-turn window that binds, a shortened window, and a
cache that rolls over. A negative control without history differs by 0.18. Speed: 105 ms per
minibatch with bfloat16 scores, 149 to 156 with float32 scores, against 119 to 137 for change 2
alone. Risk: with bfloat16 scores the outputs move at rounding level (largest logit difference
0.005 on an untrained network), so it must be optional and off by default. Its real value is
dropping torch.compile, Triton and the silent eager fallback, and the 5.5 GB minibatch of 768.

### Settings: TRAINING.md's commands against config.py

| Setting | config.py | Laptop command | What the difference does |
|---|---|---|---|
| `mode` | `ppo_with_jepa` | `imitation`, `warmup`, `ppo` | The JEPA modes add a second network's forward and a predictor to every minibatch. Never run here. |
| Ports (workers) | 10 | 4 | One worker and, in imitation, one teacher engine per port. |
| `target_concurrent_battles` | 800 (80 a worker) | 128 (32 a worker) | Sides waiting per inference call, and wall time per battle. |
| `rooms_per_pair` | 32 | 16 | Connections only. |
| `steps_per_update` | 36,864 | 16,384 | Length of one update, not the work per decision. |
| `minibatch_size` x `grad_accum_steps` | 768 x 1 | 256 x 3 | Same effective batch. With 64 minibatches an epoch, 3 leaves one over, so each epoch ends with one extra optimizer step on a third of a batch: 44 steps an update where 768 x 1 would give 42. |
| `batch_seq_len`, `update_epochs` | 256, 2 | same | |
| `save_every_updates` | 25 | 25 | 219 MB each, never pruned. |

Settings in config.py that look like knobs and are read by nothing: `infer_min_batch`,
`infer_max_batch`, `infer_wait_ms`, `infer_max_pending`, `infer_timeout_s`, `open_timeout`,
`learn_min_episodes`, `learn_wait_ms`, `server_min_batch`, `server_max_wait`, `keep_last`,
`ff_expansion`, `use_twohot_value`, `use_hp_aux`, and the three `lr_*_mult` fields.

## 5. The learning-rate schedule

From `learner.py:169` `_init_scheduler`, run through `bench/micro_lr_schedule.py`. With
`lr_warmup_steps = 1,000`, `lr_hold_steps = 500,000`, `lr_total_steps = 100,000`:

| Optimizer step | Multiplier of each group's rate |
|---|---|
| s below 1,000 | (s + 1) / 1,000 |
| 1,000 to 501,000 | 1 |
| 501,001 and every step after | 1 / 27 = 0.037 |

The anneal is `1 / (8p + 1)^1.5` with `p = (s - 501,000) / max(1, lr_total_steps - 501,000)`.
The span is negative, so it becomes 1 and the whole anneal happens in one step.

The scheduler steps once per optimizer step (`ppo_core.py:1102`). Under the upstream defaults
an update is 48 minibatches x 2 epochs = 96 optimizer steps for 36,864 decisions: 384 decisions
per step, 388 counting the few hundred an update runs over. So the fall lands at decision
194 million (192.4 million at exactly 384), 78% of the way through 250 million, and the last
56 million decisions run at 1/27 of the rate: 1.1e-5 on the policy head, 2.2e-5 on the value head, 1.1e-6 on the backbone.
With the laptop's command (256 x 3) it is 381 decisions per step and 191 million. PPO updates
that stop early on KL take fewer steps, which pushes it later.

Three more facts. The count restarts at zero in each phase (`learner.py:536` skips the scheduler
on a change of mode) and is restored on a same-phase resume. On a same-phase resume
`learner.py:571` writes the full rate into every group and the scheduler only corrects it at its
next step, so the first optimizer step after each restart runs at the full rate. And the
multipliers in `LearnerConfig.__post_init__` (backbone 0.5 in PPO) are not what runs: the table
in `learner.py:152` is, and it says 0.1.

The smallest correct fix is one digit: `lr_total_steps: int = 1_000_000` in `config.py:162`,
plus a guard in `__post_init__` that refuses `lr_total_steps <= lr_warmup_steps + lr_hold_steps`.
Then the largest single-step change is 0.002%, the rate glides from 1 at step 501,000 to 1/27 at
step 1,000,000, and a 250-million-decision run ends at 0.167 of the rate. That keeps the shape
the formula was written for. Whether the run should end at 0.167 or lower is the owner's call:
to finish the anneal exactly at 250 million with the upstream update shape, set
`lr_total_steps = 644,000` and shorten the hold to match.

## 6. bench/throughput.py

```
python bench/throughput.py --servers start --ps-dir <showdown> --ports 8000-8015 \
    --warmup-minutes 5 --minutes 10 --out <file>.json -- <train.py arguments>
```

It starts the servers through `start_servers.py` (or measures running ones with `--servers use`,
which it never signals), runs `train.py` with `--minutes` set to warmup + window + margin, reads
the telemetry lines as they arrive, samples every process under what it started every 5 s, and
writes one JSON. Decisions a second are taken between lines that follow a finished update,
because `total_steps` only moves when the learner is free and jumps by a whole backlog after
each update. Roles come from command lines, and for Ray actors on Windows, where a process cannot
retitle itself, from the `:actor_name:` line Ray writes at the top of each worker log. It stops
only what it started: every process is remembered by pid and start time together with the
command it was found under, so leftovers are found even after their parent has gone.

Exit code 0 is a complete window. 3: the trainer stopped before the window ended (the result
still holds what was seen, and the last lines of the trainer's output). 4: no telemetry within
`--startup-seconds`. 5: the trainer did not exit by itself. With `--train-stop-file` it hands
`train.py` a `--stop-file` and creates that file when it has to stop the trainer early, which is
the only polite way on Windows. The file gets a new name every run and is never removed.

In the result, `cpu.by_role.<role>.busiest_single_process` is the number to look at first for
`rollout_worker` and `inference_actor`: near 1.0 means one Python thread is full.
`cpu_ms_per_decision` and `cores_per_1000_decisions_per_second` are the measured versions of
the estimates in section 2.

| Tested | How |
|---|---|
| Parsers | `--self-test`: real telemetry and learner lines from the laptop run, with Ray's prefix and colour codes |
| Rate, cycle and busy-time maths | `--self-test`: a synthetic run with a known rate and a learner that goes deaf during updates |
| The 388 decisions a second | `--parse-log` on the saved 12-minute run: 388.1, cycle 42.3 s, batch 29.2 |
| Role classifier | `--self-test`: 35 command lines, Windows and Linux forms. And `--probe-servers` on the running servers: 4 main, 4 simulator, 4 sockets, 28 other |
| Actor names from Ray logs | `--self-test --ray-logs <session>/logs` on a real session: 1 learner, 1 inference actor, 5 rollout workers. And `bench/micro_serialize.py --with-ray` on a live actor |
| CPU accounting | `--self-test` on synthetic samples. End to end on a real child process using half a core: 0.486 reported |
| The whole run loop | a stand-in trainer built into the script (`--train-script bench/throughput.py -- --_fake-train ...`): 400.02 reported for a true 400. With train.py's 5 s telemetry period and back-to-back 20 s updates: 400.0, a 25.0 s cycle and 20.0 s busy per update, all as set |
| Stopping by pid | stand-in servers with children and a trainer that leaves an orphan: 2 and 6 processes killed by pid, none left, nothing else touched |
| Stop file and give-up path | stand-in trainer that prints nothing: gives up, creates the stop file, the trainer exits by itself |

Not tested, and why: a real `train.py` run from start to finish (not allowed here: no ports, and
the GPU and memory were in use); `--servers start` against the real `start_servers.py` and real
ports, including the wait for the ports; anything on Linux, in particular the `ray::` process
titles, SIGINT to the trainer and SIGTERM to the servers, and its own handling of SIGTERM and
SIGHUP; per-role GPU memory (nvidia-smi gives no per-process memory on Windows); more than one
GPU; Ctrl+C; `--train-stop-file` against the real `train.py --stop-file` (only against the
stand-in). It reads `[Telemetry]` lines in the format of `train.py:250`: if that line changes,
the parser must follow. It does not read `metrics.jsonl`. A window has to span a few update
cycles: with fewer than two telemetry lines in it there is no rate, and the result says so.

The learner's busy time per update is an estimate from gaps in the telemetry, a lower bound.
An exact figure needs the learner to print how long each update took.

## 7. Noticed on the way, not speed

- New weights clear every battle's history (`inference.py:253`). With an update every 20 to 40 s
  and battles of 25 s or more, a large share of decisions are made by a policy that has just lost
  its memory of the game, while the learner replays them with the memory present. Not a bias (the
  stored log-probability is the true one), but it widens the gap PPO has to clip.
- A rollout worker died with a Windows heap corruption (0xc0000374, then an access violation
  inside poke-env's message handler) in the Ray session of 10:48. `train.py` exits when that
  happens. Whether Linux shows it is unknown.
- `keep_last = 500` is read by nothing: checkpoints are never pruned.
- `ModelConfig.ff_expansion = 4.0` is read by nothing: the layers are built with 2.0
  (`ppo_core.py:358`). Do not "fix" this, it would change the architecture.
- The learner allocates and checkpoints a second copy of the network (`target_net`) in modes that
  never use it.
- The worker's queue to the learner drops the oldest episodes without a word once 160 are
  waiting. On the laptop a worker queued about 95 during an update, by arithmetic (2.55 episodes
  a second for 37 s). The margin shrinks with longer updates or busier workers.
- The learner drops what does not fill the last minibatch of each update, about 1% of the data.

## 8. Reproducing the numbers

```
python bench/micro_showdown_sim.py --ps-dir <showdown> --battles 300 --record 150 --out <dir>/sim
python bench/micro_obs.py --streams <dir>/sim/streams.jsonl --battles 120 --out <dir>/obs --profile
python bench/micro_obs.py --streams <dir>/sim/streams.jsonl --mode imitation --keep-obs 0
python bench/micro_psbot.py --streams <dir>/sim/streams.jsonl --battles 40
python bench/micro_serialize.py --obs <dir>/obs/obs_samples.npz --with-ray
python bench/micro_gpu.py --obs <dir>/obs/obs_samples.npz --out <dir>/gpu
python bench/micro_gpu.py --obs <dir>/obs/obs_samples.npz --sections share
python bench/micro_lr_schedule.py
python bench/throughput.py --self-test
```

`micro_gpu.py` uses the GPU in bursts of a few calls each, about 90 seconds in all, and caps its
own memory at 85% of the card so a mistake fails instead of spilling into system memory. Run it
with nothing else on the GPU: one of the runs here overlapped another job and every figure in it
was 1.2 to 1.4 times higher.

## 9. Results

Added later on 2026-10-05, after the changes were made. Sections 1 to 8 are as the audit wrote
them: their line numbers, and every figure called "as coded", describe the code before these
changes (commit b5680bd, which is what the cloud box ran its imitation phase with).

### What was changed

| # | Candidate | Done | Where it is now | Held to the old code by |
|---|---|---|---|---|
| 2 | Block mask from episode ids | yes | `ppo_core.py`: `episodic_block_mask`, `inference_block_mask` | `tests/test_speed_mask.py` |
| 3 | History in one pooled tensor | yes | `inference.py`: `_kv_pool`, `_build_batched_kv_cache`, `_store_new_kv_cache` | `tests/test_speed_kvpool.py` |
| 5 | Tables in the observation assembler | yes | `obs_moves.MoveEncoder`, `obs_pokemon.PokemonBodyEncoder`, `obs_abilities.AbilityEncoder` | `tests/test_speed_obs.py` |
| 8 | float16 to inference | yes | `worker.py` sends what the assembler made, `inference.py` widens it on the GPU | `tests/test_speed_kvpool.py`, `tests/test_speed_update.py` |
| 9 | The update's data on the GPU | yes, when it is under a tenth of the memory free there, and cut on the CPU as before when the move still runs out of memory | `ppo_core.py`: `update_data_fits`, `AsyncEpisodeDataset.on_device`, `iter_minibatches` | `tests/test_speed_update.py` |
| 12 | Advantages without torch scalars | yes | `ppo_core.gae_from_episode` | `tests/test_speed_update.py` |
| 4 | More PSBot engines per worker | no | | `psbot_bridge.py` was outside this change, and on the box the imitation phase already uses 12.7 of 16 hardware threads |
| 6, 7, 10, 11 | Batching across workers, MPS, Showdown config, per-turn attention | no | | 6 changes the random draws, 7 and 10 are settings on the box, 11 moves the outputs at rounding level |

"Old code" in every comparison is commit b5680bd, imported next to the working tree by
`bench/micro_ref.py`, so both run on the same inputs in one process. What each test holds:

| Test file | What must be equal, bit for bit | On what |
|---|---|---|
| `test_speed_mask.py` (40 tests) | Every field of the `BlockMask` against torch's `create_block_mask`. Policy logits, value logits and values of the old and the new model. Gradients of every tensor the loss reaches. | 19 sequence shapes (1 to 256 turns, 31 and 255 not filling whole blocks, windows of 0, 1, 8, 64, 100 and 300 turns) times 6 episode layouts. Real minibatches cut the way the learner cuts them: an episode cut by a sequence boundary, several episodes in one sequence, episodes of up to 142 turns. Inference over 80 consecutive turns, so the history fills and rolls over, with one battle restarting. |
| `test_speed_kvpool.py` (6 tests) | The tensor the model is handed, logits, value logits, sampled actions, log-probabilities, value variance, and the stored histories, of the old and the new `InferenceActor` under the same seed. And that a call which fails while new sides are being given rows stores nothing (added after the review, see below). | One random run of 1,327 calls and 16,220 rows: 1,796 battles started, 2,194 ended, 527 came back under the same tag, up to 160 alive in a pool that starts with 8 rows (it grew 5 times), 413 repeated tags inside one call, new weights and a new temperature 4 times. The old actor is fed float32, the new one float16. The failed call: a pool of two rows, the failure made up once while the pool grows and once while a row is wiped, against a twin actor that never had the failed call. |
| `test_speed_obs.py` (7 tests) | The observation's bytes, the old assembler and the new one called on the same battle at every decision, and the count of names missing from the vocabulary. | 63,537 decisions of 750 recorded battles through the real worker path: a Terastallized Pokemon in 53,739 of them, a transformed one in 259, Zoroark in 1,069 (an Illusion ending in 42), 7,293 forced switches, Recharge offered 14 times and Struggle 70, more than four known moves 32 times. gen9randombattle has no team preview. Then 12,000 made-up Pokemon over every item, ability and stat case the bounds look at. |
| `test_speed_update.py` (33 tests) | After two whole updates in each of imitation, warmup and PPO: every weight, every fp32 master, every Adam moment and step count, the rates, the scheduler, and every loss figure. Advantages and returns of 8,000 episode cases. The rows of every minibatch cut on the GPU, and cut on the CPU when the GPU refuses the data. The learning-rate multiplier of every parameter group at every optimizer step probed, old learner against new, in the three phases. One test holds nothing to the old code: a checkpoint saved in the hold and resumed with `lr_total_steps = 1,000,000` has, from there on, the rates of a run that had that length from its first step. | The new update run three ways (data on the GPU, on the CPU, choosing by itself), against the old update, which is first shown to repeat itself exactly. The refused move: made up, and for real with this process capped at the GPU memory it already holds. The schedule: steps 0 to 1,200, around 501,000 and 1,000,000, and every 997th step to 1,100,000. |
| `test_speed_ref.py` (5 tests) | Nothing about the model: that `bench/micro_ref.py` hands out the old code on a machine without git history only from a directory holding the commit's files to the byte. | git made unavailable two ways, against a complete directory, an empty one, one with a file missing, one with today's `ppo_core.py` under the old name, and one with converted line endings. |

The audit's own check also still holds: `bench/micro_obs.py --no-save --compare` against the
6,000 observations the old code saved reports `identical: true`.

### Old against new, one process, taking turns

RTX 4060, `bench/micro_before_after.py`. Other jobs were running on the machine, so read the
ratios.

| What | Old | New | |
|---|---|---|---|
| Learner, whole `ppo_update`, ms per minibatch of 256 | 265 | 161 | 1.65 times |
| its peak GPU memory | 3.8 GB | 1.4 GB | |
| Inference call, batch 1 | 17.5 ms | 11.3 ms | 1.56 times |
| batch 8 | 21.0 ms | 12.4 ms | 1.69 times |
| batch 32 | 24.2 ms | 13.1 ms | 1.86 times |
| batch 64 | 31.4 ms | 16.3 ms | 1.93 times |
| batch 128 | 48.9 ms | 27.8 ms | 1.76 times |
| `ObservationAssembler.assemble`, per decision (5,220 decisions) | 1.42 ms | 0.34 ms | 4.2 times |
| Rollout worker, message to order, per decision (`micro_obs.py`, 10,217 decisions of 120 battles, old and new taking turns three times in one session) | 1.75 to 1.83 ms | 0.57 to 0.60 ms | 3.1 times |
| Advantages, per step, episode of 40, dynamic lambda | 22.6 us | 2.7 us | 8 times |
| Worker's batch of 29 before sending | 0.46 ms, 994 kB | 0.03 ms, 497 kB | |

The worker row first read "2.53 ms, 0.52 ms, 4.8 times". Those were two separate `micro_obs.py`
runs 18 minutes apart, on 60 and on 120 battles, and in the earlier one the code that did not
change was 1.5 to 1.85 times slower as well (poke-env's request parsing 0.095 against 0.060 ms
per decision, the reward potential 0.077 against 0.047), so the machine was busier and the ratio
overstated the change by about half. The row above is a re-measurement after the review: a clean
checkout of the old code and the working tree, the same 120 recorded battles, in the order old,
new, old, new, old, new. (Section 2 has 1.56 ms for the old code on a quiet machine.)

The reviewers ran `micro_before_after.py` again, on a GPU they report as quiet (four rounds,
real observations): learner 181 to 187 ms against 119 to 121 ms per minibatch (1.54 times), peak
memory 3,755 against 1,414 MiB; an inference call 27.8 against 16.9 ms at batch 1, 37.4 against
18.9 at batch 32 and 64.6 against 27.2 at batch 128 (1.6 to 2.4 times). The ratios hold. The
absolute figures differ between the two sessions in both directions (their learner was faster,
their inference calls slower), which is one more reason to rely on the ratios only.

### End to end on the laptop

`bench/throughput.py` against the real `train.py`: 4 servers, 128 battles, PPO, minibatch 256
with accumulation 3, 16,384 steps an update, each run started from a copy of the same imitation
checkpoint written by the old code, 4 minutes discarded and 5 measured. Old is a clean checkout
of the commit before the changes. In the order they were run: old, new, old, old, new.

**The old code's figures for minutes 4 to 9 are not a baseline, and the 2.5 to 4 times between
them and the new code's is not a speedup.** A 4-server run takes this laptop's free memory from
12 GB to about 1 GB within its first minute, whichever code it is, and a second task was using
the machine. By minute 4 the old runs had 0.1 to 0.5 GB of RAM left and had collapsed: 3 updates
in the 5 measured minutes, one every 87 to 116 s, where their own first four minutes had one
every 37 to 43 s, and where the same code on the same laptop held 390 decisions a second that
morning (TRAINING.md). The new code lost a tenth in one run and nearly a third in the other. Every
figure in the table below that is taken over minutes 4 to 9, which is every row but the second
and the last, carries that collapse.

What the runs do support:

| Comparison | Old | New | Ratio | What it is worth |
|---|---|---|---|---|
| PPO end to end, minutes 0 to 4 of these runs, decisions a second | 418, 427, 449 | 650, 660 | 1.5 | Three runs against two, before the memory ran out, but inside the start-up minutes that TRAINING.md says to discard (the CPU's short boost, in both). This laptop only. |
| PPO end to end, the new runs against the morning's sustained figure | 390 | 468, 582 | 1.2 to 1.5 | Different sessions, and the morning's machine was quieter. |
| Imitation with PSBot end to end, 2 servers | 110 (TRAINING.md) | 121 | 1.1 | Different sessions. In this phase the teacher's engine is the limit (candidate 4, not done), so these changes buy next to nothing there. |
| The parts, one process, taking turns (table above) | | | learner 1.5 to 1.65, inference call 1.6 to 2.4, worker 3.1 | Solid, but they are parts. None of them is the end-to-end figure. |

So: about 1.5 times end to end in PPO on this laptop, and roughly nothing in the imitation
phase. For the cloud box there is no measurement at all. Expect at most about that 1.5 in PPO,
and possibly less: on its 16 hardware threads PPO is likely to be bound by the CPU, and of the
9.3 ms of CPU a decision cost in the new runs here, 3.5 ms are the Showdown processes (simulator,
main, sockets), which these changes do not touch. The imitation phase the box is running is
bound by the PSBot engines, so restarting it on the new code buys little until PPO starts.
Before a gain is quoted for the box, measure it there with `bench/throughput.py` at the box's
own shape (minibatch 768, accumulation 1, 36,864 steps an update, its number of ports), in PPO,
the old code and the new, at least two runs of each.

The five runs in full:

| | Old 1 | New 1 | Old 2 | Old 3 | New 2 |
|---|---|---|---|---|---|
| Decisions a second, minutes 4 to 9 (the old runs had collapsed: not a baseline) | 141 | 582 | 191 | 176 | 468 |
| Decisions a second, minutes 0 to 4 | 427 | 650 | 418 | 449 | 660 |
| Seconds per update | 31 (from gaps) | 24.5 (timed) | 31 (from gaps) | 33 (from gaps) | 27.4 (timed) |
| Seconds per update cycle | 116 | 28 | 87 | 94 | 35 |
| Inference calls a second, mean batch | 4.2, 28.9 | 18.6, 30.5 | 7.2, 27.8 | 6.3, 28.7 | 15.6, 30.9 |
| GPU utilisation, mean | 31% | 60% | 42% | 42% | 54% |
| GPU memory, most | 6.7 GB | 3.9 GB | 6.7 GB | 6.3 GB | 3.0 GB |
| CPU ms per decision: rollout workers | 5.2 | 2.9 | 6.6 | 6.2 | 2.8 |
| inference actor | 2.40 | 1.17 | 2.50 | 2.31 | 1.15 |
| learner | 1.61 | 1.25 | 2.08 | 1.84 | 1.17 |
| simulators | 2.41 | 2.10 | 2.59 | 2.47 | 2.10 |
| Showdown main, sockets | 0.76, 0.51 | 0.78, 0.62 | 0.88, 0.63 | 0.83, 0.58 | 0.77, 0.59 |
| Ray's own processes | 1.56 | 0.52 | 1.35 | 1.42 | 0.61 |
| Cores per 1,000 decisions a second | 14.5 | 9.3 | 16.7 | 15.7 | 9.2 |
| Busiest single process: worker, inference, learner | 0.19, 0.34, 0.23 | 0.44, 0.68, 0.73 | 0.32, 0.48, 0.40 | 0.28, 0.41, 0.32 | 0.36, 0.54, 0.55 |
| RAM left on the machine, least | 0.5 GB | 0.3 GB | 0.2 GB | 0.1 GB | 0.1 GB |

What limits the laptop now is the learner: it was busy 87% and 79% of the time in the two new
runs (24.5 s of a 28 s cycle), against a quarter to a third before, and an update of 130
minibatches takes 0.19 to 0.21 s per minibatch while inference shares the GPU.

A smaller run with the new code, 2 servers and 32 battles with 4,096 steps an update, made 542
decisions a second in minutes 1 to 4, wrote a checkpoint every second update and printed nothing
the old code does not print, apart from the new key on the update line. TRAINING.md has 222 for
the old code at that scale, from another session and over a window it does not record, while
542 is from a run's first minutes (the CPU's short boost). The two are not a ratio to quote.

An imitation run with the new code, resumed in its own phase from the old code's checkpoint (2
servers, 32 battles, PSBot as teacher): the optimizer and the schedule came back, updates 3 to
15 printed `sec=2.0 acc=0.28` at the end of their lines, and it made 121 decisions a second with
each of the two PSBot engines at 1.0 core, which is candidate 4's point.

`bench/throughput.py` had never met the real `train.py` when section 6 was written. It has now,
eight times on Windows (five runs at 4 servers, three at 2): it started and stopped its own
servers by PID, passed the stop file through, and exited 0 each time with a complete window. In
one of the eight it reported the actors under Ray's own processes: it had taken their names
from the logs of a Ray session of that morning, whose driver once had a pid this run happened
to reuse. It now takes the session from a Ray process's own command line, and before one is
seen only a session made since the run began. The run after that fix named every actor. It
now also reports the learner's busy share from `sec=`. It is still untested on Linux.
`bench/micro_gpu.py` keeps its
`fastmask` and `kvpool` sections as a second opinion that needs no git history: they now time
the code as it is against the old ways (torch's dense mask by switching
`ppo_core._FAST_BLOCK_MASKS` off, the per-battle lists as `ListCacheActor` in that file). Its
last run: learner minibatch 203 ms and 3.5 GB with the dense mask, 147 ms and 1.1 GB without;
storing the histories of a call of 32 took 3.85 ms as lists and 0.13 ms in the pool, of a call
of 128 14.7 ms and 0.53 ms.

### Changed with these, not speed

| What | Before | Now |
|---|---|---|
| `lr_total_steps` | 100,000, less than the hold, so the rate falls 27 times in one optimizer step at step 501,001 | **Not changed.** These changes first set it to 1,000,000 and made `LearnerConfig` refuse the old value. The review took both out again. The two schedules are the same number through step 501,000 (192 million decisions into PPO at 96 optimizer steps an update), and from step 501,001 the longer one runs at up to 27 times the old rate (0.24 of the full rate against 0.037 at step 600,000, 0.16 where a 250-million-decision run ends). That is a decision about what the model learns in the last 23% of the run, not a speed change, and it is the owner's. `config.py` holds the committed value again, with the reason at the line, and `tests/test_speed_update.py` holds the schedule to the old code's at every step. Section 5 has the arithmetic. The decision can be made at any trainer restart before step 501,000 of the PPO phase: the schedule is rebuilt from `config.py` at every start and only the step count comes from the checkpoint. |
| First optimizer step after a restart in the same phase | at the full rate, whatever the schedule said: 11 times too high for the old code's own checkpoint (89 steps into the warmup), 27 times too high after the hold, and right in the hold, where the full rate is the scheduled one | at the scheduled rate. A run interrupted and resumed at step 88, 250,000 or 750,000 has exactly the rates of one that was not. A change of phase was never affected: it starts a fresh schedule. |
| Checkpoint save | written to a temporary file and renamed | the temporary file is synced to disk before the rename, and on Linux the directory after it, so a box that loses power cannot come back with a torn newest checkpoint |
| The `[learner] upd=` line | seven keys, ending in `n_mb` | ` sec=` (how long the update took, exactly) and, in imitation modes, ` acc=` (the share of steps where the policy's best legal action is the teacher's) appended at the end. `pipeline.py` reads the longer line unchanged, and `bench/throughput.py` now reports the learner's busy share from `sec`. |

A checkpoint written by the old code loads with nothing missing or unexpected, gives the old
code's outputs bit for bit on the learner's path and over 70 turns of the inference path, and
resumes into imitation (optimizer, masters and schedule restored, at the rate it was saved
with), warmup and PPO (the weights, a fresh schedule), each followed by an update. Every laptop run
above started PPO from such a checkpoint.

### Changed after the review

The changes above were reviewed the same day. What the review found and what was done:

| Found | Before | Now | Held by |
|---|---|---|---|
| The schedule's length was moved inside the speed work | `lr_total_steps` 1,000,000 and a guard, which changes training from optimizer step 501,001 on | Back at the committed 100,000 with no guard (the first row of the table above). The owner decides. | `test_schedule_is_the_one_the_old_code_builds`, `test_schedule_constants_are_the_ones_the_run_was_planned_on` |
| A failed inference call could leave a side with another battle's history | When a call failed while new battle sides were being given rows (the pool has to grow and the GPU is out of memory), a side that had already been handed a freed row stayed registered on it, unwiped. At its next decision the network was given the dead battle's history and length, until that battle ended or new weights arrived. The old code stored nothing on a failed call. | The rows handed out in a failed call go back and those sides are forgotten, so the call stores nothing. A call that does not fail runs the same statements as before. | `test_a_failed_call_leaves_no_side_holding_another_sides_history` |
| An update was dropped when its data did not fit after all | `update_data_fits` looks at the card's free memory at one instant. If the move then ran out of memory, the error left `ppo_update`, the learner logged "PPO Update failed" and that update's decisions were gone (at least 36,864 on the box), at every update for as long as it lasted. | That pass is cut on the CPU, as every pass was before these changes: the same rows in the same order, so the same update. The next pass asks the GPU again. | `test_minibatches_are_cut_on_the_cpu_when_the_gpu_refuses_the_data`, `test_a_real_out_of_memory_on_the_move_is_survived`, `test_an_update_goes_through_when_the_gpu_refuses_its_data` |
| The comparisons with the old code could not run on the box | `bench/micro_ref.py` needed `git show`, and the box has no history | A directory holding the commit's files to the byte is accepted ("Running the checks again" below) | `tests/test_speed_ref.py` |
| Two figures in this section said more than was measured | The worker row (4.8 times) and the reading of the end-to-end table (2.5 to 4 times) | Corrected in place above, the worker row from a new measurement | |

### Running the checks again

```
python bench/micro_ref.py                                  # writes the old files into the system temp folder
python bench/micro_showdown_sim.py --ps-dir <showdown> --battles 600 --record 600 --ipc-messages 0 --out <dir>/sim
python bench/micro_obs.py --streams <dir>/sim/streams.jsonl --battles 120 --out <dir>/obs

PSPPO_OBS_SAMPLES=<dir>/obs/obs_samples.npz  PSPPO_STREAMS=<dir>/sim/streams.jsonl  PSPPO_OLD_CKPT=<a checkpoint the old code wrote> \
    python -m pytest tests/test_speed_mask.py tests/test_speed_kvpool.py tests/test_speed_obs.py tests/test_speed_update.py tests/test_speed_ref.py -q -p no:cacheprovider -s -rs
python bench/micro_before_after.py --obs <dir>/obs/obs_samples.npz --streams <dir>/sim/streams.jsonl --out <file>.json
```

Without the three variables the tests still run: on made-up observations that stay inside every
embedding table, without the replay of recorded battles, and without the checkpoint tests. They
use the GPU for about five minutes in all, 3.8 GB of it for moments (that is the old mask), so
on the box they take a little from a run in progress. Without a GPU they run the short
sequences only and skip whole updates, which need the attention's backward pass.

Read the last line of the run, not only its colour. With a GPU, the three variables and the old
code at hand it is `91 passed` and nothing skipped (86 for the first four files). Every skipped
test is a comparison that was not made, and `-rs` prints why. The case to watch is a machine
without the repository's history, which is what the cloud box is (`cloud/vm.sh deploy` sends a
tar file, no `.git`): `git show` cannot produce the old files there, and until the review that
meant every comparison with the old code was skipped and the run still ended green. Tried on the
laptop in a copy of the tree made the way the deploy makes it: the first four files ended
`22 passed, 63 skipped` with exit code 0 without the old files, and with them every test ran
and passed. The old files are made where the history is and sent along:

```
PSPPO_REF_DIR=<dir> python bench/micro_ref.py              # here: writes the eleven old files into <dir>
PSPPO_REF_DIR=<that directory on the other machine> python -m pytest tests/test_speed_mask.py ...   # there
```

`bench/micro_ref.py` takes such a directory only when git cannot be asked and every file in it
has exactly the commit's bytes (git's own blob ids, kept in `REF_BLOBS`), so today's files or
files with converted line endings cannot pass for the old code. On such a machine two of the
five tests in `test_speed_ref.py` skip, because they check against git itself.

Not tested, and why: anything on Linux or on the L40S (no access to the box); a run longer than
ten minutes; a minibatch of 768 (the box's shape) end to end, since it never fitted the laptop
with the old code to compare against; the pool growing in a real run (it starts with room for
every battle side, and grew only in the test that starts it at 8 rows); a real power loss for
the checkpoint sync (the order of sync and rename is tested, the disk is not); a real
out-of-memory while the history pool grows (the failure is made up in the test, whereas the one
on the update's data is torch's own).
