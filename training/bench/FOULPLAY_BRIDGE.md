# Foul Play as an adviser: the bridge

`foulplay_bridge.py` asks Foul Play what it would play in a battle that one of OUR poke-env players
is playing, and returns the move together with what its search saw for every legal action.
`foulplay_player.py` is a poke-env player that plays that move. `bench/fp_bridge_server.py` is the
helper process behind both. Everything below was run for real in WSL on the laptop on 2026-10-05.
None of it has run on the VM yet.

A review of the first version found five things. What was done about each is in
["After the review"](#after-the-review); the rest of this document is as it stands after them.

`bench/foulplay.py` and this are different things:

| | `bench/foulplay.py` | the bridge |
|---|---|---|
| What runs | Real Foul Play processes that log in under their own names and play their own battles | Our process, our websocket, our battle. Only the thinking is Foul Play's |
| What you get | A win rate against the real thing | Foul Play's move for our position, the visits and scores of every root action, and a player class |
| Good for | The evaluation opponent, unchanged from upstream | An imitation teacher, a training opponent, look-ahead at play time |

## Files

| File | Runs under | What it is |
|---|---|---|
| `foulplay_bridge.py` | our Python (3.12 on Linux, 3.14 on Windows in the tests), standard library only | `FoulPlayBridge` (the pool of helpers), `FoulPlayMixin` (gives a poke-env player the advice), `available()`, the mapping onto our 14 actions, `teacher_problems()` (when a run should stop) and `calibrate()` (how hard Foul Play searches on this machine) |
| `foulplay_player.py` | our Python, needs poke-env | `FoulPlayBridgePlayer` |
| `bench/fp_bridge_server.py` | FOUL PLAY's virtualenv python | the helper: JSON lines on stdin and stdout |
| `bench/fp_bridge_replay.py` | any Python; it starts Foul Play's | the exact check: the helper against Foul Play's own unmodified battle loop, state for state |
| `tests/test_foulplay_bridge.py` | pytest | 95 tests, 81 of them need neither Foul Play nor a server |
| `bench/fp_bridge_measure.py` | our Python | iterations per decision and the time from asked to answered, at several search times and helper counts, on this machine as loaded as it is |
| `tests/fixtures/foulplay_bridge_trace.jsonl.gz` | | seven recorded battles with the real helper's answers (60 KB) |
| `tests/fixtures/foulplay_bridge_replay.jsonl.gz` | | nine recorded battles for the exact check: three of a real Foul Play process, five of random play, one of SimpleHeuristics (48 KB) |

Foul Play is found the way `bench/foulplay.py` finds it: `$FOULPLAY_DIR`, default `~/foul-play`,
its python at `<dir>/.venv/bin/python` or `$FOULPLAY_PYTHON`. `bench/install_foulplay.sh` installs
it. On the VM that is `/home/ubuntu/foul-play`, so nothing has to be set. On Windows it is looked
for inside WSL and the helpers run there: see ["Search at play time"](#search-at-play-time-windows-the-opponents-side-a-payoff-table).

## Use

```python
from foulplay_player import FoulPlayBridgePlayer, available

ok, why = available()                      # (True, "foul-play <commit> at <dir>") or (False, reason)
opponent = FoulPlayBridgePlayer(search_ms=100, helpers=4, battle_format="gen9randombattle",
                                account_configuration=..., server_configuration=..., max_concurrent_battles=16)
await policy_player.battle_against(opponent, n_battles=100)
print(opponent.foulplay_summary())         # decisions, fallbacks and why
```

Any player of your own, for the advice without playing it:

```python
from foulplay_bridge import FoulPlayMixin

class MyPlayer(FoulPlayMixin, Player):
    foulplay_enabled = True                # the mixin is off until this is set
    foulplay_search_ms = 100

    async def choose_move(self, battle):
        advice = await self.foulplay_advice(battle)
        # advice.choice        "move 2 terastallize": the move Foul Play drew
        # advice.action        its index among our 14 actions, None if it is not one of them
        # advice.best_action   the action with the largest share of the search
        # advice.policy[14]    the share of every action (sums to 1), advice.visits[14], advice.score[14]
        # advice.mask[14]      the legal mask, as ObservationAssembler.create_action_mask makes it
        order = ...                        # whatever this player decides to send
        self.foulplay_note(battle, order)  # REQUIRED when the order is not the one Foul Play was asked to pick
        return order
```

`order, advice = await self.foulplay_decision(battle)` does the common case in one call: Foul
Play's move as a poke-env order, already noted. `order` is `None` when the bridge failed, when the
move is not legal for poke-env, or when the request asks whom Revival Blessing revives (Foul Play
has no answer to that one); the caller falls back and notes what it played. Each case has its own
count in `player.foulplay_counts`.

The note is not optional. Foul Play reasons about the opponent's speed, Choice Scarf and set from
the move it believes it selected (`battle.user.last_selected_move`). In the first trial run, with
the note missing, its own code stopped on the next turn with `KeyError: 'switchvikavolt'`.

A player that LEARNS from Foul Play asks a third way:

```python
order, action, advice = await self.foulplay_teacher(battle)
if action is not None:
    record(action)          # Foul Play's label: this index plays exactly Foul Play's move
    play(order)
else:
    order = something_else(battle)     # no answer in time, a helper down, a battle Foul Play's
    self.foulplay_note(battle, order)  # code could not follow, Revival Blessing, a move the 14
    play(order)                        # actions cannot say: played, counted, and NOT recorded
```

A move made because Foul Play had no answer is another bot's move. Recorded as the teacher's
step it teaches the student that bot, and nothing in a run shows it. `teacher_problems(asked,
unanswered, ...)` says when there are too many of them for the games still to be Foul Play's.

Settings without touching code: `FP_BRIDGE_SEARCH_MS` (100) and `FP_BRIDGE_HELPERS` (1) are read
when the process-wide bridge is first made.

Two commands for a machine the bridge has not run on:

```bash
python bench/fp_bridge_replay.py check           # does the helper track a battle exactly as this Foul Play does? (20 s)
python foulplay_bridge.py calibrate --helpers 4  # iterations per decision at 20 to 100 ms, here, now (2 minutes)
```

## What Foul Play does inside (commit 8f541f80, poke-engine 0.0.48)

| Question | Answer | Where |
|---|---|---|
| How do messages reach the battle state? | `fp.run_battle.pokemon_battle` reads whole websocket messages. `start_battle` takes the room title to find the opponent's name (the title minus its own name), waits for the opponent's `\|player\|` line to learn which side it is, keeps the lines after `\|start`, and builds its side from the first `\|request\|` JSON (`Battler.initialize_first_turn_user_from_json`). After that every message goes to `update_battle`. | `fp/run_battle.py`, `fp/modes/random_battle.py`, `fp/modes/base.py` |
| When is the state updated? | Lines are only collected (`battle.msg_list`) until a `\|request\|` arrives. Then `request()` stores the JSON, `process_battle_updates` runs every collected line through its handler (`switch`, `move`, `-damage`, 48 kinds) with the inference around them (speed ranges, Choice Scarf, damage rolls against the possible sets, Heavy-Duty Boots), and a move is picked unless the request says `wait`. This matches the server, which sends a turn's log first and the request after it. | `fp/battle/protocol.py` |
| Team preview? | Not in random battles. | |
| State per battle | One `Battle`: two `Battler`s (active, reserve, side conditions, `last_used_move`, `last_selected_move`), weather, field, Trick Room, the last request JSON, `rqid`, `wait`, `force_switch`, `time_remaining`. | `fp/battle/state.py` |
| How is a decision made? | `async_pick_move(battle)`: copy the battle, bring the copy up to date from the request JSON (`update_from_request_json`), `find_best_move(copy)` in a thread, record `last_selected_move`, `format_decision`. | `fp/modes/base.py` |
| The search | `find_best_move` asks the mode for `(worlds, time per world)`, samples that many complete opponent teams (`prepare_random_battles`), turns each into a poke-engine state string and runs `monte_carlo_tree_search(state, ms)` on each in a `ProcessPoolExecutor` it creates for that one decision. | `fp/search/main.py` |
| How many worlds, how long | `--search-parallelism` (P) times 4 worlds at half the search time while it is early (at most 3 opposing Pokemon seen, the active one alive and with no move revealed), otherwise P times 2 worlds at the full time. Half as many worlds when 60 s or less are left on its turn clock. With P = 1 a decision is 4 x ms/2 or 2 x ms of search either way. Measured: 4 worlds in 11.6% of 6,658 decisions. | `RandomBattleMode.search_params` |
| How the hidden sets are sampled | For every opposing Pokemon seen so far, one full set (moves, item, ability, Tera type) is drawn from the pkmn/randbats sets still consistent with what was revealed, weighted by how often the set is generated. The unseen slots are filled with random species and sets, redrawn up to ten times when the team breaks the generator's rules (species clause, types, weaknesses). Every world counts 1/worlds. | `fp/search/random_battles.py` |
| What the search returns | Per world an `MctsResult`: `side_one` and `side_two`, each a list of root options with `move_choice`, `visits`, `total_score`, and `total_visits` (the iterations). | poke-engine |
| What the score is | Not a win probability. A leaf is scored `sigmoid(0.0125 * (evaluation - evaluation of the root position))`, 1 or 0 when the battle is over. So 0.5 means "the engine's material count is where it is now", and scores of different positions are not on one scale. | `poke_engine/src/mcts.rs` |
| How the move is chosen | Per option the share is the mean over worlds of visits / iterations. Options with at least 75% of the largest share are kept, and ONE IS DRAWN at random in proportion to its share. So Foul Play is not deterministic even for identical searches. Measured: the drawn move is the largest share in 87.9% of decisions. | `select_move_from_mcts_results` |
| The search budget | Time, not iterations. The engine runs batches of 1,000 iterations and looks at the clock between them. One option only: 1,000 iterations and out. | `mcts.rs` |
| Move names | The engine names root options `earthquake`, `earthquake-tera`, `switch garchomp`. `format_decision` turns them into `/choose move earthquake terastallize` (by move id, not slot) and `/switch 3` (the Pokemon's place in the request, 1-based). | `fp/modes/base.py` |
| Forced switches, trapping | `forceSwitch` in the request becomes `force_switch` for the engine, which then offers switches only. `trapped` and `maybeTrapped` both become `force_trapped`: no switch is searched. | `fp/search/poke_engine_helpers.py` |
| Global state | `FoulPlayConfig` (one module-level object: user name, format, search time, parallelism), the pokedex and move tables (loaded once, modified per format by `apply_mods`), and one `RandomBattleMode` with the set data, shared by every battle and reloaded from disk at every battle start. Nothing in them belongs to one battle except the user name, which is read exactly once, when a battle's title is parsed. | `fp/config.py`, `fp/data`, `fp/modes` |
| So: many battles in one process? | Yes, once the user name is set before a battle's lines are processed and the search settings before a decision. Proven here by 16 battles at a time on 4 helpers, and by 240 battles held open in one helper. What one process cannot do is search two decisions at once: the engine call holds the interpreter lock. One process, one core. | |

## How the helper uses Foul Play's code

The checkout on disk is never edited. In memory:

| Foul Play's own | What the helper does with it |
|---|---|
| `fp.run_battle.pokemon_battle` | Runs unchanged, once per battle. Its "websocket" is an object whose `receive_message()` waits for the next fed message, so the coroutine is advanced by hand with `send()`, one message at a time. No event loop, no thread, and the helper always knows whether a battle is waiting for input, finished (the coroutine returned the winner) or broken (it raised). |
| `async_pick_move`, as called inside that loop | Replaced by a function that notes "Foul Play would move now" and the battle object. The search then runs when the player asks. |
| `async_pick_move`, the original | Is what runs when the player asks: the same copy, update, `find_best_move`, `last_selected_move`, `format_decision`. |
| `ProcessPoolExecutor` in `fp.search.main` | Replaced by an executor that runs the search in the helper itself (`--pool inline`, the default). Foul Play forks a new worker for every decision, which calls the same function with the same arguments. `--pool fork` keeps Foul Play's pool. |
| `select_move_from_mcts_results`, `find_best_move` | Wrapped to keep what they were given (every world's results, the prepared battle). They return what the originals return. |
| `FoulPlayConfig.username` | Set to the battle's player before every step. |
| `FoulPlayConfig.search_time_ms`, `.parallelism` | Set before every decision. |
| the pokedex and move tables | Checked after every finished battle, as Foul Play does. If they changed, the helper exits and the bridge starts a clean one. |

One thing is added on top of Foul Play's own behaviour: `chose`. Foul Play records the move it
picked as the move it made. A player that sends something else says so, and the helper sets
`last_selected_move` exactly as `async_pick_move` would have.

## The protocol

JSON lines. `key` is `<user name>|<battle tag>`: both sides of a self-play game can live in one process.

| To the helper | Reply | Meaning |
|---|---|---|
| `{"op":"ping","id":1}` | version, pid, commit, engine, `clock` | `clock` is the helper's `time.monotonic()`: the client sends deadlines only to a helper that reads the clock it reads itself |
| `{"op":"feed","key":K,"user":U,"room":R,"data":D}` | none | one websocket message of the room, the lines after the `>room` header |
| `{"op":"decide","id":2,"key":K,"ms":100,"rqid":7,"deadline":81234.5}` | the decision, or `ok: false` with the reason | Foul Play's move for the request it was fed last. With `rqid`, a question about another request than the one Foul Play is at is refused instead of answered. With `deadline` (a time on that clock), a decision that could not be made before it (about 2.5 x `ms`) is not searched: `{"ok":false,"expired":true}` at once |
| `{"op":"chose","key":K,"choice":"move 2 terastallize"}` | none | what the player really sent |
| `{"op":"end","key":K}` | none | forget the battle |
| `{"op":"sync","id":3}`, `{"op":"stats","id":4}`, `{"op":"exit"}` | | |
| `{"op":"beat"}` | none | "the bridge is still there", sent every 2 s to a helper inside WSL (see "Search at play time") |
| `decide` with `"iterations"`, `"threads"`, `"payoff_ms"`, `"payoff"` | the decision, with `"payoff"` when asked | optional, protocol version 2: see "Search at play time". Every decision of version 2 also carries `"opponent"` and `"clock"` |

A decision:

```json
{"id":2,"ok":true,"key":"p1a|battle-gen9randombattle-7",
 "choice":"move 2 terastallize", "best":"move 2 terastallize",
 "fp_choice":"earthquake-tera", "fp_message":"/choose move earthquake terastallize",
 "options":[{"choice":"move 1","fp":"stealthrock","visits":412,"total_score":177.6,"score":0.431,
             "policy":0.012,"considered":false}, ...],
 "worlds":4, "iterations":57000, "iterations_per_world":[15000,14000,13000,15000], "search_ms":50,
 "ms":241.7, "turn":5, "rqid":7, "force_switch":false, "tera_dropped":false}
```

`visits` and `total_score` are summed over the sampled worlds, `score` is their quotient, `policy`
is the share Foul Play itself computes and draws from, `considered` marks the options it draws
among. `choice` is what Foul Play drew, `best` the largest share.

Answers without a move are `{"ok":false,"error":"..."}`: a wait request, a battle that is over or
unknown, a question about another request than the current one, `"revive":true` for the
request that asks whom Revival Blessing revives (see "Robustness"), and `"expired":true` for a
question nobody waits for any more (see "After the review", F3).

A battle whose lines Foul Play's loop could not follow (its code raised) is marked failed: every
later decision for it answers with the exception's text, and the helper goes on with its other
battles. The real Foul Play process would have crashed at that point. It happened in one live
battle and in 2 of 12 recorded ones, all three after Revival Blessing, and in none since the
helper handles that move as described below.

## The client

| Requirement | How |
|---|---|
| A pool of N helpers, one core each | `FoulPlayBridge(helpers=N)`. Helpers start when a battle first needs them. |
| Battles pinned to a helper | At its first line a battle goes to the helper with the fewest battles and stays there. |
| Never block the event loop | `feed()` and `note()` put a line on that helper's queue and return. One thread per helper writes the queue, one reads the answers. A decision is a `concurrent.futures.Future`; `advice()` awaits it from any loop. A question is put on the queue the moment `advice()` is called, behind every line fed before it and ahead of every line fed after. |
| Timeouts | `advice(timeout=...)`, default 60 s. The caller gets `FoulPlayBridgeError`; the helper is left alone and the next question is answered. The timeout travels with the question as its deadline: when the question's turn comes and a search could no longer end in time, the helper does not search it and says `expired` at once. A helper that is asked more than it can answer therefore keeps answering as many as it can, in time, instead of answering all of them late. |
| A helper that dies | Its reader thread sees the end of its output. The bridge starts another, feeds it every stored line and noted choice of the battles it held, and asks again what was waiting (once: a question that kills two helpers fails by itself). |
| A helper that hangs | A helper that has work and answers nothing for `hang_timeout` (60 s) is stopped by its process group, and the same recovery follows. A long queue is not a hang. |
| A helper that cannot start | After `max_restarts` (5) failures within a minute the slot is left down for 30 s. Meanwhile lines are still stored and decisions fail with the helper's own reason (its last log line, or its fatal message). |
| Finished battles are forgotten | The mixin calls `forget()` from `_battle_finished_callback`. The stored lines go, the helper drops its state, and late lines for the room are ignored (the last 20,000 ended battles are remembered for that). |
| So are battles whose room is gone | A battle does not always end with `\|win\|`. The training server (`rlspawn.ts`) closes every battle room older than five minutes, and the player then gets `\|expire\|` and `\|deinit` and never a result, so poke-env never reports the battle as finished. `\|deinit` (the server sends it to a player it takes out of a room, for whatever reason) therefore forgets the battle too. |
| Memory does not grow with battles played | Stored bytes follow the battles in progress: 0 after every run below. A cap of 4,096 stored battles evicts the one unused longest if a player never reports its battles as finished. |
| Everything stopped at exit | Every helper is the leader of its own process group and is stopped by that group's id in `close()` and at interpreter exit. A helper also exits when its stdin closes, and checks once a second that its parent is still its parent. |
| A room that is joined again | Seen on the live server: a player that sends `/join` for a room it is IN gets nothing at all (so `worker.py`'s `/join` for a battle it has no record of changes nothing here). Only a player that was out of the room (`/leave`, answered with `\|deinit`) is sent the whole battle again, from `\|init\|`, in one message, followed by the current request. Whatever happened while it was out it has missed. Foul Play cannot pick a battle up midway, and an old state must not be fed the same lines twice. So an `\|init\|` for a battle that was followed before starts it over; the new one never reaches a first request, and its decisions are errors (counted fallbacks), not guesses from a state that missed lines. |

## Our 14 actions

`obs_assembler.py` numbers the actions 0-3 moves, 4-9 switches, 10-13 the same moves with
Terastallization (the comment in `config.py` lists them in another order). A move's index is its
place among `battle.active_pokemon.moves`, a switch's index is the Pokemon's place in
`battle.team`. Neither is the slot number of the request, which is what a Showdown choice string
uses, and the request lists the active Pokemon first.

`worker.py` turns the teacher's order into an index with `ObservationAssembler.map_order_to_index`
and plays `map_index_to_order` of it. `foulplay_bridge.py` goes the same way: choice string to
poke-env order (`psbot_bridge.choice_to_order`), order to index. It holds copies of the assembler's
three functions without numpy (`legal_mask`, `order_to_index`, `index_to_target`), and uses the
assembler's own when the player has one, as `worker.SyncBridgePlayer` does. A choice only gets an
index when playing that index plays that very choice. Anything else is reported in
`advice.unmapped` and counted, not forced onto action 0.

What the comparison of Foul Play's root options with the request found:

| Case | What Foul Play does | What the bridge does | Seen |
|---|---|---|---|
| A Pokemon locked into a move (Outrage, a charging turn). The request has one move and no `canTerastallize`. | Its engine still offers `outrage-tera`, and it may send `/choose move outrage terastallize`. The server plays that as plain Outrage: `Side.chooseMove` returns from its locked-move branch before it looks at the modifier. | `choice` is the plain move, `tera_dropped` is true, and the visits of both options are added on the plain move's index. `fp_message` stays literal. | 6 of 43,485 live decisions (Foul Play rarely locks itself), and 1 of 259 when replaying SimpleHeuristics games. |
| `maybeTrapped` in the request | Treated as trapped: no switch is searched. | poke-env still lists the switches, so they are legal actions with 0 visits. | 33 of 6,658 decisions (0.5%) had a legal action with no visit at all. |
| A move the request offers that poke-env does not list among the active Pokemon's moves, next to moves it does | | No index: `advice.action` is `None` and `foulplay_decision` still returns the order, which is legal. The assembler would call it action 0 and action 0 would play another move. | 0 of 43,485. Covered by a unit test. |
| Struggle, Recharge | One move in the request under its own id. | Action 0, which plays it. | covered by unit tests |
| The request after Revival Blessing (whom to revive) | Takes it for a forced switch and offers the living Pokemon, or fails. | No answer (`revive`). See "Robustness". | 1 battle in 43,485 live decisions, 12 recorded for it |

## Measured

WSL2 on the laptop (Core Ultra 7 155H, shared with other jobs), everything pinned to 4 cores with
`taskset`, Showdown `3661ce4` on Node 24.16.0, Foul Play `8f541f80` with poke-engine 0.0.48 in
Python 3.12.15, poke-env 0.15.0. Intervals are 95% Wilson intervals. "Simple" is poke-env's
`SimpleHeuristicsPlayer` as shipped.

### a. Does it play like Foul Play?

| Check | Result |
|---|---|
| **The exact check** (`bench/fp_bridge_replay.py check`, added after the review). Recorded battles are replayed through Foul Play's own unmodified battle loop and through the helper, without any search, and compared at every decision: the engine state string of every sampled world, the search time per world, everything in Foul Play's `Battle` object, the message Foul Play would send, and `last_selected_move` after the helper's `chose` | **0 differences** in 359 decisions and 776 sampled worlds of 9 battles (the fixture in the repository: 3 games of a real Foul Play process, 5 of random play with Transform, Illusion, locked moves, recharge, Struggle, forced switches and called moves, 1 of SimpleHeuristics). The same 359 decisions with all 9 battles interleaved in one helper, and 76 of them again in a battle rebuilt from its stored lines and choices: identical. On 22 battles (all 16 of the real process, 881 decisions, 1,898 worlds): 0 differences. With one line not fed: 70 differences in the first two battles. With one choice noted wrongly: 4 |
| Bridge at 100 ms against Simple, 120 games | 111 won, **92.5% (86.4 to 96.0)**. The real process at 200 ms: 131 of 144, 91.0% (`bench/FOULPLAY.md`). After the review's changes: 113 of 120, 94.2% (88.4 to 97.1) |
| Bridge at 100 ms against the REAL process at 100 ms, through `bench.foulplay.run_matches`, 110 games | Bridge won 60, lost 50: **54.5% (45.2 to 63.5)**. 0 errors, Foul Play's own log agrees with the count. |
| Every other game played against the real process: a 4 game trial before (0 won), 16 games with the real process logging at DEBUG (10 won), and 24, 24 and 16 games after later changes to the bridge (11, 13 and 8 won) | With the 110 above: **102 of 194, 52.6% (45.6 to 59.5)**. 0 errors and 0 fallbacks in all of them. |
| The same after the review's changes: 60 games against the real process at 100 ms | 31 won, 51.7% (39.3 to 63.8), 1,978 decisions, 0 errors, 0 fallbacks. All 254 games together: **133 of 254, 52.4% (46.2 to 58.4)** |
| Search effort in those games, iterations per sampled world | real process mean 48,555 (median 29,000), bridge mean 50,187 (median 30,000) |

The sharper check. The real process, at DEBUG, logs every message it receives and every move it
sends. Those received messages were fed to the bridge under the process's own name, and at each of
the 582 points in 16 games where the real process sent a move, the bridge was asked twice. The
bridge was then told the real move, so both followed the same game to its end.

| On the same 582 positions | Same move sent | Same largest-share option |
|---|---|---|
| Real process and bridge | 74.9% (71.2 to 78.3) | 81.6% (78.3 to 84.6) |
| Real process and the bridge's second decision | 73.4% (69.6 to 76.8) | |
| Bridge and bridge | 74.7% (71.1 to 78.1) | 82.6% (79.4 to 85.5) |

| Also on those positions | Real | Bridge |
|---|---|---|
| Worlds sampled: the same number | 582 of 582 | |
| Largest share, mean | 0.485 | 0.474 (mean difference to the real one 0.065, between two bridge decisions 0.062) |
| Options within 75% of the largest share, mean | 1.48 | 1.48 |
| Iterations per world, mean (median) | 52,391 (37,000) | 46,781 (31,000) |
| Bridge errors | | 0 |

The bridge agrees with the real process exactly as often as it agrees with itself. The number of
worlds depends on the tracked state (Pokemon seen, moves revealed), and it never differed.

**How often do two runs of the same search agree?** On 2,715 live positions (bridge against
Simple, every decision asked twice at 100 ms):

| | All positions | With more than one legal action (2,677) |
|---|---|---|
| Same drawn move | **74.1% (72.4 to 75.7)** | 73.7% |
| Same largest-share move | **77.9% (76.3 to 79.4)** | 77.5% |
| Total variation distance between the two share vectors | | mean 0.154, median 0.107 |
| Share one search gives to the other search's best move | | mean 0.466 |

That is the ceiling for imitation at 100 ms: a student that reproduced Foul Play's search perfectly
would match the teacher's recorded move in about 74% of positions, and its largest-share move in
about 78% (82% on the positions of Foul Play against itself, above). Three things make two
searches differ: the two or four opponent teams that are sampled, the search itself, and the final
draw among the options within 75% of the best.

### b. The labels

6,658 decisions in 246 games (120 against Simple, 126 against the real process), search time 100 ms.

| | |
|---|---|
| Answered by Foul Play | 6,658 of 6,658. No fallback, no bridge error, no battle Foul Play's code could not follow |
| The choice maps to a legal action index of ours | **6,658 of 6,658 (100%)**. Exceptions: none. Every one also passed the checks against `ObservationAssembler`: same mask, same index as `map_order_to_index`, and `map_index_to_order` of it plays the same order |
| Options of the search with no index of ours | 0. The shares on our actions sum to 1 in every decision |
| Terastallization dropped on a locked move | 0 here (see the table above for the one case found elsewhere) |
| Moves / switches / Tera moves | 72.9% / 23.9% / 3.3% |
| By action 0 to 13 | 17.5, 17.9, 18.1, 19.4 \| 2.4, 4.6, 4.1, 4.2, 4.1, 4.5 \| 0.9, 0.8, 0.7, 0.9 (%) |
| Games in which it Terastallized | 88.6% |
| Forced switches | 12.5% of decisions |
| Decisions per game | 27.1 (22.6 against Simple, 30.7 against the real process) |

How peaked the search is, on the 6,521 decisions with more than one legal action:

| | |
|---|---|
| Entropy of the visit shares | mean 1.39 nats, median 1.48, 10th to 90th percentile 0.49 to 2.10. Uniform over the legal actions would be 2.09 |
| Entropy relative to uniform | mean 0.66 |
| Largest share | mean 0.51, median 0.47. Above 0.9 in 9.0% of decisions, above 0.5 in 44.2%, below 0.3 in 17.9% |
| Effective number of actions, exp(entropy) | mean 4.7 of 8.8 legal |
| A second option within 75% of the best | 24.2% of decisions |

So the search is far from one-hot: the single recorded move throws away most of what it knows,
and it is a noisy label besides (74% repeatable). The share vector is the better target.

Over all the live games behind this document (the few short trial runs before them are not
counted), at 20, 50, 100 and 500 ms and against every opponent: 43,485
decisions of `FoulPlayBridgePlayer`, each checked against the assembler in the same way. 43,479
were answered and every one of those is a legal action index of ours. The 6 without an answer are
one battle with Revival Blessing (below). In 6 decisions Terastallization was asked for on a
locked move and played as the plain move.

### c. Cost

The iteration counts in this section are what this laptop gave when the first version was
measured. A few hours later the same cores gave about half as many per millisecond ("After the
review", F2). The times per decision and the decisions per second are limited by the clock and
did not change; the iterations, which are what makes Foul Play strong, did.

One helper, nothing else running, 259 decisions of recorded games replayed back to back:

| `search_ms` | Decisions a second on one core | Time in the helper, mean (95th) | Asked to answered | Overhead beyond the search | Iterations per world, mean (median) | Iterations per decision |
|---|---|---|---|---|---|---|
| 2 (the floor) | 30.5 | 24.6 ms (42.6) | 32.6 ms | 20.8 ms | 1,226 (1,000) | 2,794 |
| 20 | **16.8** | 54.1 ms (62.6) | 59.4 ms | 15.2 ms | 11,953 (7,000) | 27,226 |
| 50 | **8.1** | 118.3 ms (129.2) | 123.5 ms | 21.2 ms | 30,425 (18,000) | 69,302 |
| 100 | **4.3** | 225.6 ms (241.3) | 230.6 ms | 31.6 ms | 63,889 (35,000) | 145,524 |
| 200 | **2.2** | 446.3 ms (478.6) | 452.2 ms | 58.3 ms | 118,200 (62,000) | 269,234 |

The pipe and the client add about 5 ms to a decision. With Foul Play's own pool (`--pool fork`, a
fork per decision): 12.8 decisions a second at 20 ms and 3.8 at 100 ms, 70.5 and 253.8 ms each, and
10,040 and 50,214 iterations per world, so the in-process search is the cheaper way to the same
function. Iteration counts moved by about 20% between runs on this shared laptop in both
directions (see the two comparisons with the real process above), so read them as "the same
effort", not to the percent.

Memory:

| | |
|---|---|
| A helper, just started | 42 MB |
| After its first battle (set data loaded) | 54 MB |
| Per battle held open | 0.17 MB in the helper (240 battles: 95.7 MB), 73 KB of stored lines in the client |
| During live runs | 55 to 70 MB per helper at 100 ms |
| Feeding | 3.1 ms per message in the helper, Foul Play's inference included. Rebuilding a whole battle after a helper died: 173 ms on average |

Strength at shorter search times. Bridge against bridge, both sides through the same helpers:

| Side A | Side B | Games | A won | A's share |
|---|---|---|---|---|
| 20 ms | 100 ms | 80 | 24 | **30.0% (21.1 to 40.8)** |
| 50 ms | 100 ms | 160 (two runs of 80: 45 and 41 won) | 86 | **53.8% (46.0 to 61.3)** |
| 100 ms, always the largest share (`pick="best"`) | 100 ms, the drawn move | 80 | 49 | 61.3% (50.3 to 71.2) |

50 ms loses nothing against 100 ms that 160 games can show and costs half. 20 ms is clearly
weaker against an equal: its search is visibly flatter (largest share 0.40 on average, against
0.51 at 100 ms) and its drawn move is the largest share in 80.5% of decisions, against 87%.
`bench/FOULPLAY.md` found the same plateau from the other side: 200 ms against 50 ms and 1000 ms
against 200 ms made no difference.

Against Simple, over the runs of this document: 20 ms won 243 of 270 (90.0%, 85.8 to 93.0), 50 ms
148 of 170 (87.1%, 81.2 to 91.3), 100 ms 244 of 264 (92.4%, 88.6 to 95.0). The three do not come out in the
order of their search time, so this opponent does not tell them apart: it loses to all of them,
and what is left is the luck of the teams. Only the head to head games above say something about
search time.

How repeatable a label is at each setting (every decision asked twice, against Simple):

| `search_ms` | Positions | Same drawn move | Same largest-share move | Distance between the two share vectors, mean |
|---|---|---|---|---|
| 20 | 1,576 | 68.5% (66.2 to 70.8) | 81.7% (79.7 to 83.5) | 0.093 |
| 50 | 1,558 | 72.6% (70.3 to 74.8) | 78.7% (76.6 to 80.7) | 0.122 |
| 100 | 2,715 | 74.1% (72.4 to 75.7) | 77.9% (76.3 to 79.4) | 0.154 |

A longer search does not make the label more repeatable. The shares of a short search are flatter
and closer to each other from run to run; a long search commits more, and to different things
depending on the two or four teams it happened to sample. The largest-share move stays near 80%
at every setting, the drawn move is the noisier of the two everywhere.

### d. Robustness

| What was done | Result |
|---|---|
| 16 battles at once on 4 helpers, 48 games against Simple at 100 ms | 45 won. 1,098 decisions, 1,092 answered. The other 6 were one battle in which Foul Play used Revival Blessing: see below. No bridge error, no helper restart |
| One of 4 helpers killed with SIGKILL while it held 5 of 16 running battles and 4 decisions were waiting in it. Done three times at 100 ms (32 games each) and once at 50 ms (40 games) | Every time all games finished and every decision was answered (771, 675, 714 and 848 of them), 0 fallbacks. The 5 battles were rebuilt from their stored lines (201 lines in the first run, 139 in the third) and the 4 waiting decisions asked again |
| The opponent forfeits while our decision is being searched (500 ms setting, about 1.1 s a decision), done three times (12, 12 and 6 games) | In all 30 games the end arrived during the search. The 30 decisions were answered anyway, then poke-env processed the end (it handles one room's messages one after the other). The server refuses the late order, poke-env asks again, and the finished battle is answered with a default order without asking Foul Play (30 times). No error, nothing left over |
| The same at the bridge itself: a battle forgotten while its decision waits | Unit test: the waiting decision is answered, a second question is an error at once, the other battle is unaffected |
| A 300 turn game: both sides only ever switch, both ask Foul Play every turn (20 ms), one of the 2 helpers killed after 12 s. Done twice | 300 turns reached, 599 decisions asked and 599 answered, both times. The killed helper's battle (about turn 160, 489 and 476 stored lines) was rebuilt and its waiting decision asked again: 0.63 s and 0.49 s for that one turn. Time in the helper per decision 54.9 ms in the first 50 turns, 58.2 ms in the last 50 |
| 200 games at 20 ms, finished battles dropped from the players as `worker.py` does, memory sampled every 20 games | Client process 91.8 MB after 20 games, 91.9 MB after 200. Helpers 52 to 57 MB throughout. Stored lines 0 bytes and 0 battles at every sample. 4,600 decisions, 0 fallbacks |
| The client process killed with SIGKILL while it held 3 real helpers, one of them inside a search of 4 worlds at 2.5 s each | The two idle helpers were gone within 0.5 s (their stdin closed). The searching one was gone 2 s after the kill: a helper cannot look up while the engine runs, so it notices at the end of the world it is searching. At 100 ms that is at once |
| A battle line that arrives while the interpreter is shutting down | Found in the last runs: a player on the process-wide bridge got its room's `deinit` after the exit hook had closed the bridge, made a new bridge and started a helper in the middle of the shutdown. The helper was stopped, but nothing should start there: once the exit hook has run no bridge is made and no helper started (unit test, and 0 such lines in the two runs after the change) |
| Processes left behind | None after any run. Every script ends with a `pgrep` for helpers, Foul Play processes and servers |

poke-env itself keeps every battle object of a player for good, which is what grows in a plain
`battle_against` loop (50 to 111 MB over 120 games in the first run). `worker.py` drops finished
battles, and so did the 200 game run.

**Revival Blessing.** Foul Play follows the move in the battle log, but it cannot answer the
request that follows it, the one that asks whom to revive. Its engine takes that request for a
forced switch and offers the LIVING Pokemon, which the server refuses (12 of 12 recorded cases).
When none lives, its own code divides by zero (the battle in the first row above). And a revive
recorded as "the move I selected was a switch" stops its speed inference a turn later with a
`KeyError`, because no switch line follows (2 of those 12 battles). A real Foul Play process
does not get past this request. The helper therefore answers it with "no answer" (`revive` in the
error), keeps `last_selected_move` at Revival Blessing, and the battle goes on: in 12 battles
recorded for this (560 random games, the move forced whenever possible) all 488 decisions after
the revive were answered, against 442 before the fix. `FoulPlayBridgePlayer` revives a random
fainted Pokemon and counts it (`foulplay_counts["revive"]`); `worker.py` with the requested
change lets the old teacher choose and counts `teacher_fallbacks`. It came up once in the 43,485
live decisions.

## After the review

A second pair of eyes went through the first version looking for ways its answers could differ
from Foul Play's, or it could fail under load. Five findings. Each was first reproduced, here, on
the same four cores, then fixed, then run again. One was not what it said.

### F1. A move Foul Play did not decide was recorded as Foul Play's label (critical: real)

The change first asked of `worker.py` let SimpleHeuristics move whenever Foul Play had no answer,
and `choose_move` recorded that move as the teacher's step like any other. Nothing stopped the
run. Reasons for "no answer": a helper down or behind, a decision past its timeout, a battle
Foul Play's own code could not follow (the review saw one among 300 recorded sides of random
play: its engine panics on an opposing Pokemon with more than four known moves), Revival
Blessing.

| Run | First request | Amended request |
|---|---|---|
| Unit test (`test_the_worker_never_records_another_bots_move_as_foul_plays_label`): the real `worker.SyncBridgePlayer` from a patched copy, a recorded battle, a helper that answers every decision with an error | 15 of 15 SimpleHeuristics moves recorded as teacher steps. The test fails | 0 recorded, 15 counted as fallbacks, all 15 noted to Foul Play. The test passes |
| Live, 12 games, two `SyncBridgePlayer`s in imitation mode, players that wait 0.02 s for a search of 100 ms (so Foul Play answers nothing) | asked 514, answered 0, **514 teacher steps recorded** | asked 532, answered 0, **0 teacher steps recorded**, and `teacher_problems()` says stop |
| Live, 24 games, healthy, 50 ms, largest share | | 1,669 teacher steps, 1,669 answered, 0 outside the action mask, 0 fallbacks |

What changed:

- `FoulPlayMixin.foulplay_teacher(battle)` returns `(order, action, advice)`. `action` is the
  index among our 14 actions that plays exactly Foul Play's move, or `None` when there is no label
  on this request, whatever the reason. The rule lives in one place and is tested there.
- `worker.py` (requested): `_teacher_order` returns `(order, labelled)`; an unlabelled step is
  played, told to Foul Play, counted in `teacher_fallbacks`, and not put on the event queue. In
  `jepa_pretraining` it is still recorded: that mode trains no policy, the action is only what
  the world model is conditioned on, and the move really played is the right one for that.
- `teacher_problems(asked, unanswered, ...)` and `train.py` (requested): the run stops, the way
  it stops for a crashed worker (the final checkpoint is saved, exit code 1), when more than
  max(50, 2% of the decisions asked) had no answer. The same count covers
  `FoulPlayOpponentPlayer`: a policy that is said to train against Foul Play does not quietly
  train against SimpleHeuristics.

Not done: a skipped step leaves a gap in that battle's recorded sequence (the next observation
follows two turns after the one before it). In `imitation_with_jepa` that is one wrong pair for
the world model's loss per fallback. At the rates of a healthy run (0 in 5,697 teacher steps
before the review, 0 in 17,392 after) it is nothing, and the stop rule bounds it at 2%.

### F2. The setting is a time, the strength is iterations (high: real, and seen on this laptop)

The first version recommended 50 ms from games played on this laptop. Foul Play's search stops by
the clock, so the same 50 ms is a weaker player on slower or shared cores, with no error anywhere.
The review called this plausible. It happened here between the first version's runs and these,
on the same four cores, a few hours apart (the laptop is shared with training jobs):

| Iterations per decision, recorded battles replayed | 20 ms | 35 ms | 50 ms | 75 ms | 100 ms |
|---|---|---|---|---|---|
| First version, 1 helper, a free core | 27,226 | | 69,302 | | 145,524 |
| Now, 4 helpers on 4 cores (`python foulplay_bridge.py calibrate --helpers 4`) | 14,890 | 26,507 | 38,568 | 60,153 | 80,278 |
| Now, 8 helpers on the same 4 cores: starved on purpose | 7,174 | 11,894 | 16,617 | 24,503 | 32,664 |
| Time per decision in the starved run, against the row above it | 104 ms vs 63 | 134 vs 94 | 166 vs 126 | 221 vs 180 | 276 vs 233 |

Now 50 ms is 38,568 iterations, not 69,302, and 60,000 iterations take 75 ms. Helpers that
share their cores do 2.5 times less in the same milliseconds, and the only other sign is a
decision that takes 40 ms longer. In the live runs after the review, 100 ms gave 67,000 to 70,000
iterations per decision where the first version's runs had 119,000 to 146,000; in the
`PokeEnvVectorWrapper` run, where four pairs of players and the server shared the four cores with
three helpers, it gave 35,916.

What is known about iterations and strength, all head to head, bridge against bridge:

| Iterations per decision | Against | Games | Won |
|---|---|---|---|
| 22,867 (20 ms then) | 118,591 (100 ms then) | 80 | 24, **30.0%** (21.1 to 40.8) |
| 58,818 to 67,664 (50 ms then) | 119,414 to 130,840 (100 ms then) | 160 | 86, **53.8%** (46.0 to 61.3) |
| 40,099 and 54,348 (66 ms now, two runs of 80 ten minutes apart) | 114,093 and 141,151 (180 ms now) | 160 | 73, **45.6%** (38.1 to 53.4): 36 and 37 |

So about 60,000 iterations per decision is known to hold level with twice as many. 40,000 to
54,000 came out 4 points under level against 2.7 times as many, which 160 games can neither
confirm nor exclude. 23,000 is clearly weaker. That is where the two numbers in
`foulplay_bridge.py` come from: `TARGET_ITERATIONS = 60,000` to choose the search time by, and
`MIN_ITERATIONS = 45,000`, under which a run stops. (The same 66 ms were 54,348 iterations in one
of those two runs and 40,099 in the other.)

What changed:

- `python foulplay_bridge.py calibrate --helpers N` (and `calibrate()`): asks every decision
  point of the recorded battles at each search time, with N helpers working at once, and prints
  the table above and the shortest search time that reaches 60,000 iterations per decision. Run
  it on the machine the run will be on, with as many helpers as the run will have, while
  whatever else the run needs is running. It took 2 minutes here.
- `bridge.stats()` has `recent_iterations_per_decision` (over the last 5,000 decisions answered)
  next to the mean over the whole run, which hides a helper that lost its core an hour in.
- `teacher_problems(...)` and `train.py` (requested): the run stops when Foul Play searched
  fewer than `foulplay_min_iterations` (45,000) per decision over its last decisions, judged
  from 2,000 decisions on. 0 switches the check off.
- `train.py` (requested): a rollout worker that uses Foul Play is declared to Ray as
  `1 + foulplay_helpers` CPUs, not 1, and a run that needs more CPUs than Ray has is refused
  with a sentence saying so (10 workers with 4 helpers each are 50; on this laptop, 22, that is
  refused; a worker asking for CPUs Ray does not have would otherwise wait for ever without a
  word). Ray does not pin anything: this is a count, the stop rule above is the guard.

Still unmeasured: the VM. One command there settles it.

### F3. A decision nobody waits for was still searched (high: real)

A caller that gave up told nobody, so its question was searched in full when its turn came. Once
a helper was more than a timeout behind, every answer came too late, each still cost a search,
and questions piled up without end. The review's run, repeated here before any change, and again
after (12 recorded battles on one helper at 200 ms, each player waits 1 s, moves on and asks
again, 30 s; the helper can answer 2.2 decisions a second):

| | Before | After |
|---|---|---|
| A caller that waits 50 ms for a 200 ms search | searched anyway: 459 ms for nobody | not searched, told at once |
| Asked in 30 s | 349 | 359 to 600 |
| Answered while the caller waited | **2** | **29 to 60** (1 to 2 a second) |
| Questions waiting in the helper, at 5 s steps | 51, 101, 150, 199, 248 | 12, 10, 12, 1, 1 |
| Lines queued for the helper | 159 growing to 1,067 | 0 to 18 |
| A stats question to the helper | no answer in 60 s | 0.05 to 0.44 s |

That run squeezes time: a caller waits 1 s for a decision that takes half a second. In the
proportions of a real run (24 battles on one helper at 100 ms, each player waits 4 s, which is
16 decisions; 40 s; the helper can answer 4.3 a second):

| | Without deadlines | With |
|---|---|---|
| Answered while the caller waited | 15 of 231 (0.4 a second) | **164 of 246 (4.1 a second)**: all the helper can do |
| Not answered | 216, each after the full 4 s | 82: 76 told by the helper when their turn came, 6 after the full 4 s |
| Questions waiting in the helper, at 5 s steps | 32, 40, 47, 54, 69, 76, 83 | 24, 24, 24, 24, 24, 24, 24 |
| Lines queued for the helper | 67 growing to 319 | 30 to 45 |

What changed: every question carries the time until which its answer is of use (`advice()` and
`advice_sync()` pass their timeout). When its turn comes and a decision (about 2.5 times the
search time) could no longer be made before then, the helper answers `expired` at once and goes
on to the next. The time is on `time.monotonic()`, which every process of a machine reads alike
and which never jumps; a helper shows its reading when it starts and is sent deadlines only if it
reads the same (unit test with a helper whose clock is off).

Tried and taken out again: refusing a question in the client, before it is queued, when its
helper is too far behind. Players that are told at once play their fallback move at once, their
battles run ahead, and the lines of those battles alone then keep the helper busy: 2 answers in
30 s again, with 17,208 refusals and 38,310 lines queued. A caller that waits for its turn in the
helper is what keeps the two in step.

A helper that is asked more than it can answer still means fallbacks, each after a wait. F1's
rule then stops the run. The requested `worker.py` waits 20 s (`foulplay_timeout_s`), not the
bridge's default of 60, which is also the time after which `worker.py` means to take a battle
for stalled.

### F4. A battle that is joined again loses Foul Play (medium: not as described; something else was)

The finding: `worker.py`'s repair of a stalled battle sends `/rlrescue` and `/join`, the server
replays the whole battle from `|init|`, the bridge starts the battle over, and Foul Play is lost
for the rest of it. Tried on the live server (the build with `rlspawn.ts`), a player holding its
move while the commands are sent:

| Sent by a player in a running battle | What the server sent back |
|---|---|
| `/join <room>` (the room it is in) | nothing |
| `/rlrescue <room>`, then `/join <room>` (what `worker.py` sends) | nothing, to either player. The battle stayed in the server's `/rlactive` list and was played to its end afterwards |
| `/leave`, then `/join <room>` | `\|deinit`, then the whole battle from `\|init\|` in one message, then the current `\|request\|` |

So that repair does not make the server replay anything. (`/rlrescue` does nothing at all: the
plugin looks the room up under its name without the hyphens and never finds it. That is a bug in
`rlspawn.ts` and `worker.py`, outside these files, and without it the room would be closed, not
replayed. And in a run the two commands are not even sent: the loop that sends them is never
called under poke-env 0.15, see "Sizing a training run".) A replay only follows a `/leave`,
which nothing in this repository sends during a battle. And there starting over is the right
thing: the player was out of the room and missed whatever happened. Left as it was, with a test
that says why.

What the live look did find: a battle whose room goes away is never reported as finished, and
the bridge kept it for good. `rlspawn.ts` closes every battle room older than five minutes
(`|expire|`, then `|deinit`, no `|win|`), and with a search bot as the teacher a battle can take
that long (see "Sizing a training run"). Each such battle left its lines in the client (73 KB)
and a session in a helper (0.17 MB), and went on counting as a battle of that helper, so new
battles were pinned around it, until 4,096 battles were stored and the oldest was dropped. Now
`|deinit` forgets the battle. Live, a `FoulPlayBridgePlayer` whose opponent holds its move:

| | What the bridge held of the battle |
|---|---|
| Before `/leave` | 1 battle, 8,914 bytes of lines, 1 session in the helper |
| 1 s after `/leave` (`\|deinit`) | nothing: 0 battles, 0 bytes, 0 sessions. (The bridge as it went into the review, same run: still 1 battle, 8,862 bytes, 1 session) |
| After `/join` (`\|init\|`, the whole battle again, the request) | the battle from its start, which has no first request: all 70 later decisions of that game were errors and counted fallbacks, none an answer from a state that had missed lines. The game was played to its end |
| A battle made by `/rlautospawn`, standing still | the server closed the room 339 s after it was asked for: `\|expire\|`, `\|deinit`. poke-env reported no finished battle. The bridge had forgotten it a second later |

The requests to `worker.py` do the same for the worker's own books: see "Sizing a training run".

The second half of the finding stands: a battle Foul Play's own code cannot follow has no Foul
Play until it ends. Those are Foul Play's failures (the real process stops there). With F1 they
cost labels, not wrong labels.

### F5. No test would catch a small error of state (medium: real)

Head to head, 102 of 194 only excluded a bridge much weaker than Foul Play. Agreement on the
same position is 75% between two runs of the same code. Neither sees a helper that is wrong in
one position out of fifty. `bench/fp_bridge_replay.py` does (first row of "Does it play like Foul
Play?"), and three tests run it where Foul Play is installed: the check itself, and the check on
input broken on purpose, which must fail.

| Run of `bench/fp_bridge_replay.py check` | Decisions compared | Result |
|---|---|---|
| The fixture in the repository, 9 battles | 359, with 776 sampled worlds | 0 differences. 31 s |
| The same with one line (a move of the opponent) not fed, first 2 battles | 46 | **70 differences**, the first at the next decision: state strings and `Battle` contents |
| The same with one `chose` naming another move than the one sent | 46 | **4 differences**: `last_selected_move`. The engine states did not show it, which is why it is compared by itself |
| All 16 games of the real process and 6 others, 22 battles | 881, with 1,898 worlds | 0 differences. 70 s |

Both replays run with the same `PYTHONHASHSEED`: Foul Play's own tracking depends on the order of
a set in one rare place, so two Foul Play processes can differ from each other by themselves.

### The earlier runs again, after the changes

Same four cores, same scripts as in "Measured". Every decision was again checked against
`ObservationAssembler` (mask, index, round trip).

| Run | Result |
|---|---|
| Bridge at 100 ms against Simple, 120 games | 113 won, 94.2% (88.4 to 97.1). 2,783 decisions, all answered, all a legal action index of ours |
| Bridge at 100 ms against the REAL process at 100 ms, 60 games through `run_matches` | 31 won, 51.7% (39.3 to 63.8). 1,978 decisions, 0 errors |
| 16 battles on 4 helpers, one helper killed after 12 s while it held 5 battles and 4 decisions waited | 32 of 32 games finished, 789 of 789 decisions answered. 5 battles rebuilt from 102 stored lines, the 4 decisions asked again |
| The opponent forfeits while our decision is being searched (500 ms) | 12 of 12 times the end arrived during the search, 12 answered, 12 finished battles then answered with a default order |
| 300 turns of switches at 20 ms, a helper killed after 12 s | 599 of 599 decisions answered, the slowest in 0.51 s (the rebuilt battle: 483 lines) |
| 100 games at 20 ms, finished battles dropped as `worker.py` drops them | Client 92.5 MB after 20 games, 92.9 MB after 100. Helpers 55 to 59 MB. 0 bytes stored at every sample |
| Search of 66 ms against 180 ms, 160 games (the table under F2) | 10,529 decisions on both sides together, all answered, 0 with a problem |
| The last run, with the files as they are handed over: 24 games against Simple, 16 at a time on 4 helpers, a helper killed after 8 s while it held 5 battles and 4 decisions waited | 22 won. 615 of 615 decisions answered, 0 fallbacks, nothing left running |

After the changes, in all: 19,028 live decisions of `FoulPlayBridgePlayer`, every one answered
and every one a legal action index of ours by the assembler's own check, and 18,548 more
through the patched copies of `worker.py` ("Wiring it in"), with no fallback, no error, no
expired question and no battle Foul Play's code could not follow in any of them. (The two runs
that were made to fail on purpose are not in these numbers.)

## Choosing the settings

| Setting | Recommended | Why |
|---|---|---|
| `search_ms` | whatever gives **60,000 iterations per decision** on the machine of the run, under the load of the run: `python foulplay_bridge.py calibrate --helpers <all of them>`. That was 50 ms on this laptop in the first version's runs and 75 ms a few hours later | A search of 59,000 to 68,000 iterations was level with one of twice that (86 of 160). One of 23,000 lost 56 of 80. The setting is a time and means nothing by itself |
| `foulplay_min_iterations` | 45,000 | The run stops below it. Searches of 40,000 to 54,000 iterations won 73 of 160 against searches 2.7 times as long (45.6%, 38.1 to 53.4): at the edge of what was measured as good enough, and well above the 23,000 that lost 56 of 80 |
| `search_ms` as an opponent | twice the teacher's | A training or evaluation opponent is asked far less often than a teacher. The head to head with the real process was played at about 67,000 to 130,000 iterations per decision |
| `pick` | `"best"` for a teacher | The largest share is the more repeatable label (see the table above) and it did not play worse than the drawn move: 49 of 80 against it. `"choice"` is Foul Play as it plays on the ladder, and the default. |
| `helpers` | at most one per core that is free for it | A helper is one core while it searches and 55 to 70 MB. Helpers that share cores, with each other or with anything else, search less in the same time (the table under F2) |
| `foulplay_timeout_s` | 20 | How long a player waits for Foul Play. Under `worker.py`'s 60 s stall limit, and long enough that only a helper that is really behind runs into it |
| What to learn from | `advice.policy`, not only `advice.action` | The share vector has 4.7 effective actions on average. One recorded move is right about 74% to 78% of the time by the teacher's own standard. |

## Wiring it in

Nothing in the existing files was edited. The changes they need are listed in the report of this
work (config.py, worker.py, train.py, play.py, evaluate.py) as exact text replacements, 24 of
them: the first version's 17 with the review worked in. Foul Play as a third imitation teacher
with its search time, helper count, pick, waiting time and iteration floor as settings; a teacher
step that is recorded only when Foul Play decided it; a player class for the second seat of a
training pair and a share of the pairs that get it; the heartbeat fields `train.py` stops a run
on; the helpers declared to Ray; and `fpbridge` as an agent in `play.py` and an opponent in
`evaluate.py`. They were applied to COPIES of the files as they were late on 2026-10-05 (other
work had changed `worker.py`, `train.py` and `config.py` since the first version) and run, in
WSL unless it says otherwise, with the bridge as it is handed over:

| Check | Result |
|---|---|
| `worker.SyncBridgePlayer` in imitation mode with `imitation_teacher = "foulplay"`, two of them playing each other as `check_teacher.py` does. 3 helpers, 50 ms, largest share. (Before the review, four runs: 5,697 teacher steps, 0 fallbacks) | 24 games in 118 s. 1,669 teacher steps recorded (1,312 moves, 316 switches, 41 Tera moves), 1,669 answered by Foul Play, 0 outside the action mask, 0 observations of the wrong size, another bot had to move 0 times |
| The same with Foul Play made to answer nothing (F1) | 12 games: asked 532, answered 0, **0 teacher steps recorded**, 532 fallbacks counted and noted, and `teacher_problems()` says stop. With the first version's request: 514 recorded |
| `worker.PokeEnvVectorWrapper` in imitation mode, driven as `RolloutWorker` drives it: one pair, 16 battles at a time on the server's `/rlautospawn`, both sides taught, 4 helpers at 50 ms, 4 minutes. A teacher run in small | 101 games finished, none given up, no room closed by the server. 7,026 teacher steps, **29.3 a second on 4 cores**. 31.7 steps and 33 s a game (the longest 77 s). Asked 7,045, answered 7,026 (the rest were in flight at the end), 0 fallbacks, 0 expired, 0 timeouts. Heartbeat: 54,234 iterations per decision over the last 5,000; `teacher_problems()`: go on |
| `worker.SyncBridgePlayer` in ppo mode against the new `worker.FoulPlayOpponentPlayer`, the inference actor stood in for by a thread that answers every STEP with a random legal action. (Before the review, four runs: Foul Play won all 84) | 24 games, Foul Play won all 24. No DONE event from the Foul Play seat, episode slots 6 before and 6 after, no battle left in either player, 0 fallbacks |
| `worker.PokeEnvVectorWrapper` in ppo mode: 4 pairs with 2 rooms each, `foulplay_opponent_share = 0.5`, 60 s | 2 of the 4 pairs got Foul Play. 22 games finished against it (it won all 22) next to 43 self-play games in the other two pairs. No DONE event from a Foul Play seat. Heartbeat: asked 445, 0 fallbacks, 0 errors, 0 helper restarts, and **35,916 iterations per decision at 100 ms**: three helpers shared the four cores with eight players and the server. That is under the floor, and `train.py` would stop such a run once 2,000 decisions are in. With `FOULPLAY_DIR` pointing nowhere the wrapper refused to start and said why |
| `play.py --agent fpbridge --opponent simple` | 12 of 12, 268 decisions, 0 fallbacks |
| The requested `train.py`, on Windows with Ray stood in for by a stub that starts nothing, 14 cases | Goes on: a healthy run (10 workers, 43,490 asked, 10 without an answer, 62,000 iterations), 1.5% without an answer, a starved run with the floor at 0, a starved run before 2,000 decisions, one starved worker among ten (58,000 on average), a run without Foul Play. `SystemExit(1)` with the reason in the log: 347 of 349 without an answer, 3% of 100,000 without an answer, 22,867 iterations per decision. Refused with a sentence: 10 workers with 4 helpers each (50 CPUs) on 22 CPUs. Declared to Ray: 5 CPUs per worker with Foul Play as the teacher, 3 with a share of the pairs against it and 2 helpers, 1 with PSBot or with no pair against it |
| The two unit tests on the real `worker.SyncBridgePlayer`, on Windows with ray and torch installed | Both pass on the patched copy. The first fails on a copy with the first version's request (15 of 15 fallback moves recorded), the second on a copy that gives the episode slot back and not poke-env's seat |

torch and ray are not installed in the WSL virtualenv. `config.py` only asks torch whether CUDA is
there and `worker.py` only calls ray inside `RolloutWorker`, so both were empty stand-ins for the
import there. `RolloutWorker` as a Ray actor and `evaluate.py` were not run with the changes:
their copies were only compiled.

### Sizing a training run

Decisions are the bottleneck, not battles. A helper makes one decision at a time, in about 2.5
times the search time (0.12 s at 50 ms, 0.23 s at 100 ms), and every battle in progress has a
decision waiting for each side that Foul Play plays. With S such sides on one helper each
decision waits about S times that, and a battle of n decisions lasts n times as long.

Three clocks limit the wait:

| Clock | Where | What happens |
|---|---|---|
| 5 minutes from the start of a battle | `rlspawn.ts` on the server (`RL_MAX_LIFESPAN_MS`, checked once a minute) | the room is closed, finished or not, and the battle's steps are lost |
| 20 s for one decision | `foulplay_timeout_s` | another bot moves: not recorded as a label, counted as a fallback |
| 60 s without a move | `worker.py`'s repair loop | meant to drop the battle from the worker. It does not run under poke-env 0.15 (below) |

The first one decides. A game of two Foul Plays has about 31 decisions a side, and a long one 60
and more, so a decision has to come within about 4 s for the long games to end in time:

| Search time | One decision takes | Sides per helper, at most |
|---|---|---|
| 50 ms | 0.12 s | about 32 |
| 100 ms | 0.23 s | about 16 |

Measured with `worker.PokeEnvVectorWrapper` in imitation mode, both sides taught:

| Sizing | What happened |
|---|---|
| 8 sides per helper at 50 ms (16 battles, 4 helpers), 4 minutes | 1.1 s from one decision of a side to its next, 33 s a game, the longest 77 s. 101 of 101 games finished |
| 40 sides per helper at 100 ms (20 battles, 1 helper), 9 minutes: what the first version of these notes recommended | 9.4 s from one decision of a side to its next. No game finished in the first 3 minutes. The 27 games that finished took 236 s on average, the longest 319 s, and the server closed 4 of the first 20 battles under their players (8 sides). No fallback, no timeout: nothing else was wrong |

What a closed room costs is more than its steps. `worker.py` takes an episode slot for every side
of a battle (`learn_max_pending_episodes`, 220 per worker) and gives it back when the battle
ends or when its repair loop cleans the battle up. That repair loop is in `_handle_query`, and
poke-env 0.15.0, the version pinned here and on the VM, never calls `_handle_query`: it logs
the server's answer to `/rlactive` as an unhandled message. So the slot of a battle whose room
the server closes is never given back. Nor is the battle's seat in poke-env's own count
(`max_concurrent_battles`): poke-env gives a seat back on `|win|` or `|tie` only. The room the
server opens in the closed one's place is then never played, because the player has no seat for
it. A pair loses a battle for good with every room that is closed, without an error.

Measured with the wrapper run above at 40 sides per helper and 100 ms, 9 to 10 minutes each (the
server closed 3 to 5 of the pair's 20 battles in each run, the first of them between minutes 6
and 7):

| At the end of the run | Requested change without the last request to `worker.py` | With it |
|---|---|---|
| Sides of closed rooms that gave their episode slot back | 0 of 10 | 10 of 10 |
| Battles the pair was still playing, of the 20 rooms the server keeps open for it | 15. The other 5 rooms had been opened and were waiting for players that had no seat for them | 20 |
| Battles each player still held | 20, 5 of them gone from the server | 20, all in progress (and 20 of 20 seats taken in poke-env's count) |

That last request makes the player give up a battle when the server takes it out of the room
(`|deinit` without a result): the episode slot and the trajectory go back through the worker's
own CLEANUP event, the seat goes back to poke-env, and Foul Play's copy is dropped. It is older
than the bridge and not one of the review's findings. It matters here because a teacher that
searches is what makes battles long enough to be closed.

`RolloutConfig.target_concurrent_battles` is 800 today: over 10 workers that is 80 battles and
160 taught sides per worker, 40 per helper with 4 helpers. That is at the edge for a search of
50 ms (4.8 s a decision, 2.5 minutes for an average game, the five minutes for a long one) and
too many for 100 ms. More battles than the helpers can serve add nothing: the helpers are the
limit, and `train.py` stops the run when more than 2% of the decisions go unanswered.

## Search at play time: Windows, the opponent's side, a payoff table

Added on 2026-10-06 for a ladder bot that runs on Windows with the GPU and wants look-ahead next
to its network. Everything below was run on the laptop as it was that evening, with a live
ladder bot and other test runs on it. Notes with the raw numbers: `bench/notes/bridge.md`.
`bench/fp_bridge_measure.py` repeats the measurements.

### Windows: the helpers run inside WSL

Foul Play does not build on Windows. A Windows process now gets the same helpers, started inside
WSL:

```
wsl.exe [-d <distro>] --exec /usr/bin/env PYTHONUNBUFFERED=1 ... <foul-play>/.venv/bin/python -u
        /mnt/c/.../bench/fp_bridge_server.py --foulplay-dir <foul-play> --format gen9randombattle
        --pool inline --parent-watch beat --own-group --tag fpb.<pid>.<random>.<bridge>.<slot>.<generation>.
```

The same JSON lines travel over wsl.exe's stdin and stdout. Nothing else changes for the caller:
`available()`, `FoulPlayBridge`, the mixin and `FoulPlayBridgePlayer` are used as on Linux.

| Question | Answer |
|---|---|
| When is WSL used? | Only on Windows, and only when the installation asked for is inside WSL: `FOULPLAY_DIR` starts with `/`, or is a `~/...` that is no folder on the Windows side, or is not set and there is no native `~/foul-play`. A Windows path keeps the old behaviour. `FOULPLAY_WSL=0` never, `FOULPLAY_WSL=1` always |
| Where is Foul Play looked for? | `FOULPLAY_DIR` (a path inside WSL), else `~/psppo-bench/foul-play`, else `~/foul-play`. Its python: `FOULPLAY_PYTHON` (a path inside WSL) or `<dir>/.venv/bin/python`. `FOULPLAY_WSL_DISTRO` names a distribution other than the default, `FOULPLAY_WSL_EXE` another wsl.exe |
| How is it checked? | One sh script, sent on wsl.exe's stdin, makes the checks `locate()` makes on Linux (checkout, virtualenv, set data, commit, the imports) and answers in key=value lines. `available()` says `foul-play <commit> at <dir> in WSL` |
| Paths | `bench/fp_bridge_server.py` is given as `/mnt/<drive>/...` (`wsl_path()`; `\\wsl.localhost\<distro>\...` becomes the path inside). The helper reads that one file through the Windows mount |
| Settings | The Windows environment does not reach a process inside WSL. The helper's settings are on its command line, the bridge's `env=` goes through `env(1)` |
| Deadlines | A helper inside WSL reads another kernel's clock (the two differed by 529,001 s here). The bridge keeps the difference, taken from every answer, and gives a deadline on the helper's clock. Short by the time an answer travels (milliseconds), so a deadline is at most that much early |
| The cost of the pipe | Asked to answered minus the time in the helper: 1 to 10 ms, the same as inside WSL (table below) |
| Starting | 0.4 to 11 s for the helpers of a bridge on the loaded laptop in 13 runs (each is a wsl.exe start plus loading Foul Play), 44 s once while WSL itself was starting, 0.3 s inside WSL. When the WSL virtual machine is not running, the first call starts it: `available()` took 17 and 20 s then, 6 s otherwise. WSL shuts down when nothing runs in it, so call `bridge.start()` before a session, not at the first decision. The helpers keep it up while a bridge is open |
| A helper that dies | As on Linux: wsl.exe ends with the helper's exit code, the bridge starts another and rebuilds its battles |

Stopping, which is where WSL differs. Measured with the PIDs inside WSL (`wsl_leftovers()` lists
them by the tag on their command lines):

| What happens | What was seen | What the bridge does |
|---|---|---|
| `close()` | A helper whose stdin closes leaves and its wsl.exe ends: 0.1 to 0.6 s in the runs here, 3 s once | All helpers are asked at once and waited for together (8 s at most). What is still there is stopped inside WSL with one `pkill` by tag. 0 left after every run |
| A wsl.exe is killed (taskkill of its tree) | The helper inside WSL SURVIVES: its stdin stays open and its parent, the WSL relay, stays. Still there after 7 s | Whenever the bridge itself kills a wsl.exe (a hang, a helper still searching at `close()`), it stops the helper inside WSL by its tag as well |
| The Windows process is killed outright | Its wsl.exe processes live on, see their stdin close, and the helpers leave | Nothing needed |
| The Windows process is killed with its tree | No stdin closes (row 2) and nobody is left to stop the helper | The bridge sends a helper inside WSL a line every 2 s (`{"op":"beat"}`). A helper that is not working and has heard nothing for 20 s leaves by itself (`--parent-watch beat`). A search that takes longer than that is not silence: the helper is busy |
| One more wsl.exe | Costs 2 to 9 s on the loaded laptop, whatever it runs | So the bridge starts none that it can avoid: none in a clean `close()`, one for all helpers otherwise |

Not done: `bench/fp_bridge_replay.py` (the exact check) and `bench/foulplay.py` start Foul Play's
python themselves and were not taught WSL. Run them inside WSL. The three tests of the exact
check are skipped on Windows for that reason.

### The opponent's side of the search

poke-engine's `MctsResult` has `side_two`: the opponent's root options with visits and scores, in
every sampled world. Every decision now returns it (protocol version 2, at no extra search):

```python
advice = await player.foulplay_advice(battle)
advice.opponent            # [OpponentOption(name="move earthquake", share=0.41, visits=..., score=..., worlds=2, seen=None), ...]
advice.opponent_active     # "garchomp"
advice.opponent_revealed   # its Pokemon shown so far, the active one excluded
advice.opponent_worlds     # per sampled world: {"chance", "iterations", "active": {"species", "moves", "item", "ability",
                           #   "tera_type", "hp", "max_hp", "level", "status", "terastallized"}, "reserve": [...], "options": [...]}
```

| Field | Meaning |
|---|---|
| `name` | `move earthquake`, `move earthquake terastallize`, `switch garchomp`, or `none` when the opponent has no move to make (our forced switch) |
| `share` | The mean over the worlds of visits / iterations, weighted as Foul Play weights the worlds. The shares sum to 1. This is the search's own prediction of the opponent |
| `score` | Mean playout result from the OPPONENT's side. Checked on real results: our total and its total add up to the iterations (16,868.6 + 12,131.4 = 29,000) |
| `worlds` | In how many of the sampled worlds it was an option at all. A move of a set that was drawn in one world only has `worlds` 1 |
| `seen` | Switches only. False when the Pokemon has not been shown: the sampling invented it to fill the team, so that share is "it switches to something we have not seen", not a prediction of that species |
| `opponent_worlds[i]["active"]` | The complete set that was drawn for the opponent's active Pokemon in world i: the four moves, item, ability and Tera type the search assumed |

What it is and is not. It is the opponent's best play as the search sees it, against sets that
are guesses, searched 2 or 4 worlds at a time. A Pokemon locked by a Choice item has one move in
every world (seen in the first recorded battle: Wugtrio with Choice Band, `move stompingtantrum`
and switches only). How well it predicts real opponents was not measured here: that is the job
of whoever compares it with recorded games.

### A payoff table

Can poke-engine 0.0.48's Python module give the value of each pair (our option, the opponent's
option) at the root? The module offers:

| Call | What it gives |
|---|---|
| `monte_carlo_tree_search(state, duration_ms, iterations, threads)` | Per side, per root option: visits and total score. No value per pair. `iterations > 0` runs exactly that many and ignores the time; `threads` searches one tree with several threads |
| `iterative_deepening_expectiminimax(state, duration_ms)` | `side_one`, `side_two`, `matrix` (row by row, ours by theirs), `depth_searched`. The values are the engine's evaluation (positive is good for us), not the 0 to 1 of the Monte Carlo search. **The search prunes (alpha-beta): a row is dropped as soon as it is known to be worse than the best row, and its remaining cells are NaN.** In a position with 12 x 7 options 60% of the cells were NaN (106 of 174 over 4 states) |
| `generate_instructions(state, ours, theirs)` | Every outcome of one pair with its chance (a hit, a crit, a miss, a secondary effect). Switches are named by the Pokemon alone, no move is `none` |
| `State.apply_instructions(outcome)` | The state after that outcome, as a new `State` |
| `calculate_damage` | Damage rolls |
| not offered | The evaluation of a state by itself, and a search without pruning. (`IterativeDeepeningResult.get_safest_move()` in the module never advances its index into the matrix, so it does not do what it says) |

So: the engine's own matrix gives the value of SOME pairs, the ones its pruning needed. A value
for EVERY pair can be built from the other calls. Both are an optional part of a decision:

```python
advice = await player.foulplay_advice(battle, payoff_ms=300)                  # the engine's matrix
advice = await player.foulplay_advice(battle, payoff_ms=1500, payoff="full")  # every cell
advice.payoff_theirs    # the columns: the opponent's options by name
advice.payoff_table     # [14][columns]: our 14 actions by the opponent's options, the mean over the worlds
                        # that have the cell, weighted by the world's chance. None: no world has it
advice.payoff_known     # [14][columns]: the share of the worlds behind each cell
advice.payoff_depth     # plies searched, per world
advice.payoff_worlds    # the same table per world
advice.payoff           # as the helper sent it: per world "choices", "theirs", "matrix", "depth", "cells", "known"
```

| | `payoff="engine"` (default) | `payoff="full"` |
|---|---|---|
| How | `iterative_deepening_expectiminimax` on each world's state, the budget shared among the worlds | Every pair is played out with `generate_instructions`, every outcome weighted by its chance, and each resulting state is searched 1 ply |
| Depth | Whatever the time allows | 2 plies, always |
| Missing cells | Pruned ones, about half | None, unless the budget runs out. Pairs are visited in the order of their Monte Carlo visits, so the cells that are there are the ones that matter |
| Cost | The budget, exactly | About 4 to 7 ms per outcome of a pair: 555 ms for one world of 84 pairs and 152 outcomes |

The budget is added to the decision's time. A caller's deadline accounts for it.

How deep the engine's matrix gets, measured through Windows and WSL on 60 decisions of recorded
battles, 3 at a time (the budget is for the whole decision, 2.5 worlds on average):

| `payoff_ms` | Per world | Depth, mean (lowest to highest) | Cells with a value | Asked to answered at 250 ms search |
|---|---|---|---|---|
| 100 | about 40 ms | 2.43 (2 to 5) | 49% | 666 ms |
| 300 | about 120 ms | 2.66 (2 to 5) | 47% | 867 ms |
| 1000 | about 400 ms | 3.11 (2 to 6) | 46% | 1,571 ms |

On a single full position (12 of our options by 7 of the opponent's) directly: depth 2 in 100 ms,
2 in 300 ms, 3 in 1,000 ms. A forced switch (3 options against none): 3, 4 and 4. Ten times the
time buys one ply.

The complete table, 36 decisions:

| `payoff_ms` | Time used, mean | Cells with a value | Asked to answered at 250 ms search, mean (95th, slowest) |
|---|---|---|---|
| 1000 | 643 ms | 80% | 1,222 ms (1,611, 1,637) |
| 3000 | 957 ms | 100% | 1,541 ms (2,794, 3,603) |

So a complete 2-ply table costs about a second per decision on average and up to three, which
does not fit a 2 second answer next to a search. The engine's matrix fits, and half of it is
missing by construction. Whether either improves play was not measured here.

### Exact iterations, and threads

Two more optional fields of a decision, both poke-engine's own arguments:

| Request | What it does | Measured |
|---|---|---|
| `iterations=20000` | Every world is searched for exactly that many iterations, whatever the clock says | `iterations_per_world == [20000, 20000]`, 120 ms in the helper. The same strength on a loaded machine, at the price of a time that is no longer bounded |
| `threads=2` | One world's tree is searched by 2 threads | 3 decisions at a time, 250 ms: 220,583 iterations per decision in 646 ms, against about 153,000 in 575 ms with one thread (the search part of the payoff runs above). 500 ms: 400,722 in 1,261 ms against 247,416 in 1,167 ms |
| `threads=4` | | 250 ms: 354,722 in 717 ms. 500 ms: 648,361 in 1,407 ms. The time limit is overrun by 20 to 25% |

Threads buy iterations with cores, not with time: 1.4 to 1.6 times the iterations with 2 threads
and 2.3 to 2.6 times with 4, for 8 to 12% and 21 to 25% more time. Whether a tree searched by several threads is as
good per iteration as one searched by one was not measured.

### Iterations and time on the laptop as it is

`python bench/fp_bridge_measure.py --helpers 2,4,6 --ms 100,250,500,1000 --points 12`, once from
Windows (helpers inside WSL through wsl.exe) and once inside WSL (helpers as direct children),
one run after the other. As many decisions at a time as helpers, each of the first 12 decision
points of a recorded battle asked once per search time. Iterations are per decision, all worlds
together (2.5 worlds on average: 4 at half the time early in a game, 2 at the full time later).

| Helpers at once | Search time | Windows to WSL: iterations, mean (median, lowest) | asked to answered ms, mean (95th, slowest) | Inside WSL: iterations, mean (median, lowest) | asked to answered ms, mean (95th, slowest) |
|---|---|---|---|---|---|
| 2 | 100 | 67,208 (64,500, 45,000) | 245 (262, 262) | 65,958 (64,000, 34,000) | 244 (260, 276) |
| 2 | 250 | 155,208 (149,500, 79,000) | 576 (603, 604) | 158,416 (151,500, 98,000) | 576 (599, 611) |
| 2 | 500 | 316,791 (274,500, 188,000) | 1,145 (1,181, 1,217) | 312,916 (287,000, 199,000) | 1,141 (1,177, 1,185) |
| 2 | 1000 | 636,625 (551,000, 380,000) | 2,287 (2,360, 2,418) | 551,208 (529,500, 426,000) | 2,288 (2,348, 2,352) |
| 4 | 100 | 65,937 (64,000, 42,000) | 243 (261, 271) | 55,770 (56,000, 31,000) | 249 (281, 288) |
| 4 | 250 | 156,562 (154,500, 98,000) | 577 (601, 603) | 128,312 (122,500, 77,000) | 583 (608, 614) |
| 4 | 500 | 311,062 (297,000, 203,000) | 1,159 (1,232, 1,316) | 252,812 (251,000, 156,000) | 1,154 (1,195, 1,210) |
| 4 | 1000 | 554,895 (535,500, 284,000) | 2,293 (2,389, 2,463) | 509,083 (506,500, 350,000) | 2,316 (2,460, 2,477) |
| 6 | 100 | 50,194 (51,500, 16,000) | 253 (285, 319) | 48,333 (50,000, 17,000) | 255 (279, 326) |
| 6 | 250 | 121,402 (125,000, 35,000) | 590 (617, 638) | 114,972 (120,000, 33,000) | 589 (609, 699) |
| 6 | 500 | 236,847 (238,500, 69,000) | 1,153 (1,195, 1,253) | 217,902 (218,000, 83,000) | 1,162 (1,198, 1,234) |
| 6 | 1000 | 419,486 (400,000, 273,000) | 2,341 (2,494, 2,726) | 399,444 (391,500, 221,000) | 2,317 (2,439, 2,672) |

24, 48 and 72 decisions per row. What it says:

- **The path through wsl.exe costs nothing that can be seen.** Time in the helper against asked
  to answered: 1 to 10 ms on both paths. The iterations of the two paths differ by what two runs
  a minute apart differ by on this laptop (the Windows run came out higher in most rows, which
  cannot be the path).
- **The time is the setting, the iterations are what the laptop gives.** A decision takes 2.3
  times the search time whatever else runs: 0.24, 0.58, 1.15 and 2.3 s. With 6 searches at once
  the same time bought 22 to 34% fewer iterations than with 2 (50,194 against 67,208 at 100 ms,
  419,486 against 636,625 at 1,000 ms), and the weakest single decision at 100 to 500 ms fell to
  about a third (16,000 against 45,000 at 100 ms).
- **Iterations per millisecond of search time: 620 to 670 with 2 at once, 550 to 660 with 4, 420
  to 500 with 6.**

A ladder bot's shape, 3 decisions at a time, from Windows:

| Helpers | 500 ms: iterations, mean (lowest) | asked to answered ms, mean (95th, slowest) | 750 ms | | 1000 ms | |
|---|---|---|---|---|---|---|
| 3 | 247,416 (145,000) | 1,167 (1,228, 1,245) | 358,416 (216,000) | 1,735 (1,839, 1,922) | 474,361 (308,000) | 2,295 (2,367, 2,444) |
| 4 | 280,305 (201,000) | 1,160 (1,201, 1,240) | 401,472 (310,000) | 1,715 (1,757, 1,780) | 526,750 (401,000) | 2,292 (2,333, 2,353) |
| 6 | 298,944 (157,000) | 1,156 (1,196, 1,214) | 434,305 (228,000) | 1,725 (1,839, 1,857) | 570,722 (325,000) | 2,292 (2,348, 2,365) |

36 decisions per cell. The three rows ask the same thing (3 battles are on 3 helpers in each: a
battle is pinned to one helper), so their differences are the laptop's load from one minute to
the next, not the helper count.

### Recommended for a ladder bot that plays 3 games at once and answers within about 2 seconds

| Setting | Value | Why |
|---|---|---|
| Search time | **500 ms** | A decision is answered in 1.16 s on average and 1.32 s at the slowest of 252 measured, which leaves 0.7 s of the 2 s for the network, poke-env and a payoff table. It bought 237,000 to 317,000 iterations per decision here, four to five times the 60,000 that was level with twice as many ("After the review", F2). 750 ms fits too (1.73 s, slowest 1.92 s) and leaves nothing. 1,000 ms does not fit: 2.3 s |
| Helpers | **4** | One per game and one over. Each of the 3 battles then has a helper to itself also while one game ends and the next begins, and no search waits behind another. More than that buys nothing: a battle is pinned to one helper |
| Payoff table | `payoff_ms=300` (the engine's matrix) if it is used at all | Depth 2 to 3, half the cells, 0.3 s. With 500 ms of search that is 1.46 s per decision. The complete table does not fit |
| Threads | 1 | 2 threads would give about 1.5 times the iterations for about 10% more time, and the strength of that was not measured |
| `timeout` | 4 s for the ladder | The helper is told, and does not start a search it cannot finish in time |
| Before queueing | `bridge.start()` | Starting the helpers takes up to 11 s, and up to 44 s when WSL itself has to start |
| Watch | `bridge.stats()["recent_iterations_per_decision"]` | The time stays, the iterations go when the laptop is busier. Under about 60,000 the search is no longer what was measured as strong |

None of this says that the search makes the network play better. It says what the search costs.

### When the helpers go away in a game (WSL restarting): `fail_when_down`

Seen on 2026-10-07 at 02:54 IST on the laptop: the WSL virtual machine was torn down and made
again (Hyper-V's events: its network adapter deleted, a new one 3 s later), which takes every
helper with it at once. ladder.py now builds its bridge with `fail_when_down=True`: a question
whose helper is not up fails at once with `.down` (search_player counts it as the fallback
"down", a failure for ladder.py's health rule) instead of waiting in the queue of a helper that
cannot answer it before the deadline. Not up means: no process, not yet answered its first
question (Foul Play still loading, through wsl.exe that can be a minute), or nothing answered for
`timeout` while it had work (frozen; the watchdog replaces it after `hang_timeout`). With
`background_spawn` the watchdog also starts a slot that was left down after repeated failures as
soon as its cooldown is over, without waiting for a battle to ask for it. `helpers_up()` says how
many can answer now; ladder.py prints a line when it reaches 0 and when it is back.

Tested on the box (Linux helpers, so "WSL gone" is every helper killed at once and "WSL hung" is
every helper stopped with SIGSTOP; `bench/ladder_faults.py`): killed, the network's moves were
played at once until the new helpers had loaded, under a second there; frozen, at most the first
4 s of decisions waited for the deadline, then the network's moves at once until the watchdog had
replaced them (8 s). The games went on in both cases. Two tests in `tests/test_foulplay_bridge.py`.

A battle whose room was joined again after a lost connection is not followed by the search any
more at all (search_player `battle_rejoined`, the fallback "rejoined"): see F4 above for why the
helper cannot pick it up midway.

## Limits

- **Not run on the VM.** Everything was measured in WSL (Ubuntu 26.04, a standalone Python
  3.12.15, 4 pinned cores of a shared laptop whose speed changed by a factor of two within the
  day). Two commands settle the VM before a run: `python bench/fp_bridge_replay.py check` (does
  the helper track a battle as the Foul Play installed there does) and `python foulplay_bridge.py
  calibrate --helpers <all of them>` (which search time gives 60,000 iterations there).
- **Not run inside Ray.** `worker.SyncBridgePlayer`, `FoulPlayOpponentPlayer` and
  `PokeEnvVectorWrapper` with the requested changes were run by themselves, the wrapper driven as
  `RolloutWorker` drives it. The requested `train.py` was run with Ray stood in for by a stub.
  `RolloutWorker` as a Ray actor, the learner and `evaluate.py` were not run with the changes. No
  policy network was involved anywhere: the "policy" in the training-opponent check answered
  with random legal actions.
- **The share vector is not consumed yet.** `worker.py` records one action index per step, and
  the requested change keeps it that way. Soft targets need a change to the trajectory format and
  the imitation loss, which is not part of this work. The same change could carry a mark for
  "played, not a label", which would close the gap a skipped step leaves (F1).
- **Revival Blessing** (above): one request per use has no Foul Play answer. That step is played
  by another bot and not recorded.
- **A player that is taken out of a room and joins it again** has no Foul Play for the rest of
  that battle (F4). Nothing in this repository leaves a room during a battle.
- **Foul Play's own failures are reproduced, not repaired.** A battle whose lines its code cannot
  follow gets errors from then on, and the player falls back. Apart from Revival Blessing none
  occurred in the live decisions (43,485 before the review, 38,175 after); the review found one
  in 300 recorded sides of random play.
- **The training server closes battle rooms after five minutes**, finished or not (`rlspawn.ts`).
  With a teacher that searches, a battle lasts as long as its decisions wait: see "Sizing a
  training run".
- **`worker.py`'s repair loop does not run.** It lives in `_handle_query`, which poke-env 0.15.0
  never calls. If it did run, its `/rlrescue` would not find the room and its `/join` would be
  ignored by the server (F4). The one case it was needed for here, a room the server closes, is
  covered by the last request to `worker.py`. A battle that stalls for another reason is still
  nobody's: it stays until the server's five minutes are over. This is outside the bridge's files.
- **PSBot as the teacher has the fallback that F1 removed for Foul Play**: when its engine has no
  answer, SimpleHeuristics moves and the step is recorded as PSBot's. The requested change
  leaves that as it was.
- **User names.** Foul Play finds the opponent's name by removing its own from the room title.
  When the opponent's name is contained in the player's name it follows the battle from the wrong
  side. The helper notices (both sides end up with the same id) and fails that battle with a
  message that says so; it does not advise. `worker.py`'s names (`p<run><nnn>a` and `...b`) are fine.
- **The search is limited by time.** On a machine with no free core every helper searches less in
  the same milliseconds: half as much on this laptop a few hours after the first measurements,
  and 2.5 times less with two helpers per core (F2). The search time is therefore chosen by
  iterations and the run is stopped when they fall under the floor.
- **A stopped run is started again by the supervisor.** `train.py` leaves with exit code 1, as
  it does for a crashed worker, and `pipeline.py` then restarts it from the newest checkpoint,
  up to `max_restarts_per_hour` times before it marks the run failed. A cause that does not go
  away (too few cores for the helpers) therefore costs that many starts.
- **The first line of the first battle starts the helper**, which takes about 0.3 s on that one
  call (a check that Foul Play can be imported, once per process, and the process start). Call
  `bridge.start()` beforehand where even that matters.
- **Scores are not win probabilities** and not comparable between positions (see the table on
  Foul Play's inside). They rank the options of one decision.
- **Only random battle formats.** The helper refuses anything else. gen9randombattle is the one
  that was run.
- **Windows.** Foul Play does not build there. Since 2026-10-06 a Windows process starts the
  helpers inside WSL ("Search at play time"), and `available()` is True when Foul Play is
  installed there. The exact check and `bench/foulplay.py` still have to be run inside WSL.
- **The opponent's side and the payoff table are returned, not judged.** How well the search's
  shares predict real opponents, and whether a payoff table improves play, was not measured.

## Tests

```bash
python -m pytest tests/test_foulplay_bridge.py -q
```

95 tests (75 before 2026-10-06). 81 need neither Foul Play nor a server: `available()` in every failure case; the
protocol, the pinning and the ordering against a fake helper started through the real code path;
a helper killed, a helper killed while a decision waits, a decision that kills every helper, a
helper that cannot start, a helper that comes back, a hang, a long queue that is not a hang,
timeouts, the event loop staying alive, a battle that ends while its decision waits, a room
joined again, `close()`, the interpreter exiting, and a client killed outright; the 14 actions on
made-up battles, held against `ObservationAssembler`; the mixin on a stand-in player; the
hand-driven coroutine the helper rests on; and the seven recorded battles replayed through
poke-env's own message handler, where every one of the real helper's 160 recorded answers has to
map onto a legal action and survive the round trip through the assembler (among them a Pokemon
locked into Outrage, a Ditto that transforms, and Revival Blessing).

Added after the review, each failing without its fix: a decision nobody waits for is not searched
(sixteen callers on a slow helper); a helper asked more than it can answer keeps answering in
time (six players for three seconds: 3 answers without deadlines, about 15 with); a caller that
cannot wait for a search is told at once; a helper with another clock gets no deadlines; a
decision asked again after a helper died keeps its deadline; a battle whose room is gone is
forgotten, here and in the helper, and the helper it was on is free again; a room that is left
and joined again answers errors, not a stale state; `foulplay_teacher()` gives a label only for
Foul Play's own move (six ways to have none); the search effort in `stats()`, over the run and
over the last decisions; `teacher_problems()` at its thresholds; `calibrate()` choosing the search
time by iterations.

2 run the real `worker.SyncBridgePlayer` and are skipped until `worker.py` has the requested
changes (or `FOULPLAY_TEST_WORKER` names a copy that has them) and where ray and torch are
missing. One: Foul Play's own moves are recorded, another bot's moves are not, except in
`jepa_pretraining`; with the first version's request applied it fails with 15 of 15 such moves
recorded. The other: a battle whose room the server closes gives back its episode slot, its
trajectory, poke-env's seat for it and Foul Play's copy of it.

8 run the real helper and are skipped where Foul Play is not installed: a recorded battle with
the helper killed halfway, a battle with Revival Blessing, a pair of user names that makes Foul
Play mistake the sides, a caller that gave up (no search may run for it), `calibrate()` on the
real search, and the exact check three times: as it is, with one line not fed, and with one
choice noted wrongly. The last two must fail, and do.

Added on 2026-10-06 (20 tests). Without Foul Play or WSL: the opponent's side as it arrives in
an `Advice`, and an old helper that does not send it; a payoff table asked for, with the request
fields it travels in, and put on our 14 actions (the mean over the worlds, a pruned cell, a
column one world lacks, an action that is not legal); the payoff budget counted into a deadline;
the helper's own code for the opponent's names and shares, for the engine's matrix with its
pruned cells, for the complete table (outcomes weighted by their chance, a panic in the engine,
a budget that runs out) and for the value of a searched state, each against stand-ins for
poke-engine; a helper that hears nothing from its bridge leaves, one that hears it or is busy
stays; Windows paths as WSL sees them; when WSL is used and when not; what the probe script's
answers become (every failure `locate()` reports natively, wsl.exe missing, wsl.exe complaining
in UTF-16); the probe script itself on a made-up checkout (where there is an sh); the command
line a helper inside WSL is started with; deadlines on a clock 100,000 s away and the line that
says the bridge is there; an ordinary helper gets neither a tag nor that line. With Foul Play:
the real helper's opponent side, payoff tables of both kinds and exact iterations on a recorded
battle. On Windows with Foul Play inside WSL: real advice through wsl.exe with the helpers'
PIDs inside WSL checked before and after `close()`, and the Windows process killed, once alone
and once with its wsl.exe processes, after which no helper may be left inside WSL (30 s each).

On 2026-10-06, on the busy laptop: on Windows (Python 3.14, the helpers inside WSL) **88 passed
and 7 were skipped** in 312 s: the 2 that wait for `worker.py`, the 3 of the exact check (it
starts Foul Play's python itself: run it inside WSL) and 2 that need an sh or a Linux process.
Inside WSL (Python 3.12.15, Foul Play installed, no ray) **90 passed and 5 were skipped** in
187 s: the 2 that wait for `worker.py` and the 3 that are for Windows. Before that day: 73
passed and 2 skipped on Linux in 77 s, 65 passed and 10 skipped on Windows.
