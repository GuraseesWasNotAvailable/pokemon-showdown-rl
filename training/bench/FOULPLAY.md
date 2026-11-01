# Foul Play as a benchmark opponent

`bench/foulplay.py` plays gen9randombattle games between one poke-env player and Foul Play on a
local Showdown server and returns the result counted from the player's side.
`bench/install_foulplay.sh` installs Foul Play at a pinned commit. Both were run for real in WSL
on the laptop on 2026-10-05 (numbers below). Neither has run on the VM yet.

## What Foul Play is

[Foul Play](https://github.com/pmariglia/foul-play) is pmariglia's Showdown bot. For every
decision it samples several complete teams the opponent could have, from the published random
battle sets and from what the battle has revealed, and runs a Monte Carlo tree search on each
with its own Rust engine, [poke-engine](https://github.com/pmariglia/poke-engine). It learns
nothing and needs no weights, so it is the same opponent on every evaluation.

How strong it is supposed to be, from the author's own write-up
([pmariglia.github.io/posts/foul-play](https://pmariglia.github.io/posts/foul-play/), 2025-05-11):

| Format | GXE | Peak Elo | Peak ladder rank |
|---|---|---|---|
| gen9randombattle | 88% | 2341 | Top 50 |
| gen9randombattleblitz | 84% | 2018 | 1 |

Those numbers were measured on the public ladder with far more search than this benchmark gives
it. In the [Smogon thread](https://www.smogon.com/forums/threads/re-introducing-foul-play-a-competitive-pokemon-battle-bot.3767378/)
the author says he usually sets "about 7 seconds per decision", that "16GB of RAM and 4 cores"
are enough for those results, and that he suspects "diminishing returns to the number of battles
searched after maybe 8". The benchmark default is 0.2 s of search on one core. Treat the ladder
numbers as what the bot can reach, not as a rating measured for the opponent used here. What was
measured here is below: against itself, 1 s of search did not beat 0.2 s.

## What is pinned

| Thing | Pin | Why |
|---|---|---|
| Foul Play | commit `8f541f80f91f960603673796559f796b10ccf0fb` (main, 2026-10-03) | Its command line has changed between versions. |
| poke-engine | 0.0.48, built with `--features poke-engine/terastallization --no-default-features` | Foul Play's own requirement. gen9 needs Terastallization, and the default build is gen4. |
| Rust | 1.99.0, installed by rustup only when `cargo` is missing | The version the build was tested with. |
| Python | 3.11 or newer for Foul Play's virtualenv. Tested with 3.12.15 | Ubuntu 24.04 ships 3.12. |
| Random battle sets | [pkmn/randbats](https://github.com/pkmn/randbats) commit `85c017b6`, sha256 `6a4b6fc8...793c`, 509 species | Foul Play's knowledge of every possible set. Generated from Showdown commit `1ce9b34f`; nothing under `data/random-battles/gen9` changed between that and the server commit this project runs (`3661ce4`), so the sets match the server. |

If the Showdown server is ever moved to a newer commit, move `SETS_COMMIT` and `SETS_SHA256` in
the install script with it, or Foul Play will reason about sets the server no longer deals.

## Install

```bash
bash bench/install_foulplay.sh            # into ~/foul-play
bash bench/install_foulplay.sh /some/dir  # then: export FOULPLAY_DIR=/some/dir
```

What it does, each step skipped when it is already done:

1. Checks for git, curl, a C linker and a Python with `venv`. On the VM it apt-installs what is
   missing (`sudo -n`, so it never waits for a password). `FOULPLAY_NO_APT=1` forbids that.
2. Fetches the pinned commit into the directory.
3. Installs Rust with rustup if there is no `cargo`. PATH and dotfiles are not touched.
4. Creates `<dir>/.venv` and installs Foul Play's requirements there. Nothing goes into the
   training venv. The engine is built from source with `--no-cache-dir`: pip caches the wheel it
   builds and cannot tell which features a cached wheel has, so a cache hit from a build for
   another generation would silently give an engine without Tera.
5. Downloads the set data, checks its checksum, and puts it where Foul Play looks. Left to
   itself Foul Play downloads it in its first battle, and when that download fails it caches an
   empty list and plays on without knowing any set.
6. Checks the result: imports, a 100 ms search that must offer a Tera move, `run.py --help`.

It removes nothing. Measured in WSL: 57 s the first time (Rust download and engine build on 8
cores), 20 s into a second directory with Rust already there, 1 s when run again on either.

## Use

```python
from bench import foulplay

ok, why = foulplay.available()     # (True, "foul-play <commit> at <dir>") or (False, reason)
result = await foulplay.run_matches(player, 100, "127.0.0.1", 9000,
                                    parallel=8, search_time_ms=200, log_dir="/home/ubuntu/runs/main/eval/logs")
```

`player` is any poke-env 0.15 `Player`, already connected to that server, built with
`max_concurrent_battles` of at least `parallel`. The result:

| Key | Meaning |
|---|---|
| `games`, `wins`, `losses`, `ties` | Games that ran to their end, from the player's side. `games` is their sum. |
| `errors` | Games that did not. `games + errors` is always the number asked for. |
| `seconds`, `foulplay_commit`, `search_time_ms` | As named. |
| `parallel` | The value used. Lowered to the player's `max_concurrent_battles` when that is smaller. |
| `log_agrees`, `foulplay_log` | Foul Play prints the winner of every game. `foulplay_log` is that count, `log_agrees` says it equals the count above. |
| `foulplay_mean_iterations` | Search iterations per sampled team. The health check of the opponent: see "Settings". |
| `games_per_minute`, `game_seconds_mean`, `foulplay_decisions`, `foulplay_cores_per_process`, `foulplay_rss_mb_max`, `restarts`, `log_dir` | Cost and bookkeeping. |
| `error`, `failures`, `notes`, `timed_out` | Only when something went wrong: why. |

`run_matches` does not raise when Foul Play or the server fails. Those games are `errors`.
Optional arguments beyond the three above: `search_parallelism` (1), `game_timeout_s` (240 s, more
for long searches), `overall_timeout_s`, `start_timeout_s` (60), `max_restarts` (2).

From a shell, with the training venv:

```bash
python bench/foulplay.py                          # installed and usable? exit code 0 or 1
python bench/foulplay.py --play 8 --port 9000     # 8 real games, poke-env's RandomPlayer against it
```

## How a run works

| Question | Answer | Why |
|---|---|---|
| How does Foul Play log in? | Each process is started as `python bench/foulplay.py --child <dir> <Foul Play's arguments>` with Foul Play's own Python. That replaces one method, the login, in memory and then runs Foul Play as it is. The checkout is not edited. | Foul Play's guest login posts to play.pokemonshowdown.com for a signed assertion, even for a local server, and then sleeps 3 s. A `--no-security` server accepts an empty assertion, which is what poke-env sends. Without the replacement every benchmark would depend on the internet and on the name not being registered by somebody. No password argument is needed or passed. |
| How is it told the server? | `--websocket-uri ws://host:port/showdown/websocket` | |
| Who challenges? | Foul Play (`--bot-mode challenge_user`). The player accepts. | Showdown allows a user one outgoing challenge at a time, so one player cannot challenge several processes at once. And Foul Play, by its code, drops a challenge that arrives while it is still leaving its previous battle and then waits forever. When Foul Play challenges, it does so exactly when it is ready. |
| How do several games run at once on one player? | `parallel` processes, each with its own name (`FP<run>s<slot>r<attempt>`), each playing its share of the games one after the other with `--run-count`. One coroutine on the player's own event loop accepts their challenges. | It follows poke-env's `Player._accept_challenges` step for step, and adds a time limit per step. poke-env's own `accept_challenges` would wait forever if one process died. |
| How does a process end? | It exits by itself after `--run-count` games. | |
| Why are the processes started 0.4 s apart? | So that their first challenges do not arrive together. | Showdown refuses a challenge to a user who already has three waiting, and Foul Play does not ask twice. The player accepts within milliseconds, so three never wait once the run is going. |
| How is a game's result read? | From the player's battle object (`won`), for battles that ended by themselves while their process was being watched. | Counting `player.n_won_battles` would count a forfeited game. |
| What is an error? | A process that exits early, never gets a battle within `start_timeout_s`, or has a game running past `game_timeout_s`. The process is stopped, its open battle is forfeited so the player is free again, and the slot starts a new process for its remaining games (at most `max_restarts` times). Each failure costs one game. | The forfeit shows as a loss on the player object. It is not in the returned counts. |
| What is left running? | Nothing. Every process is started in its own process group and stopped by that group's id, on return, on timeout and on cancellation. If the benchmark process itself is killed, each Foul Play process notices within a second and stops itself and its search workers. | |

Foul Play turns the battle timer on in every game. The player has Showdown's normal clock to
answer, which is far more than a policy needs, and loses on time if it stops answering.

Do not use the player for anything else while `run_matches` is running.

## Settings

| Setting | Default | What it does |
|---|---|---|
| `search_time_ms` | 200 | Foul Play's `--search-time-ms`. Early in a game it searches 4 sampled teams at half this, later 2 at the full time, so a decision takes a little over twice this (0.47 s at 200 ms). |
| `search_parallelism` | 1 | Foul Play's `--search-parallelism`: that many teams searched at the same time, and proportionally more teams. It is also the number of CPU cores one process keeps busy. |
| `parallel` | 4 | Processes, which is games at once, which is cores at `search_parallelism` 1. |

The search is limited by time, not by iterations. On a machine with no free core each process
gets less CPU, finishes fewer iterations in the same 200 ms, and plays a weaker game without
anything else looking different. `foulplay_mean_iterations` in every result shows it. Measured
below: a fourfold cut in search time made no measurable difference to its results, so ordinary
load is harmless, but keep the number in `results.jsonl` and distrust an evaluation where it is
far below the others.

## Measured

WSL2 on the laptop (Core Ultra 7 155H, 22 logical cores, three other jobs running), Showdown
`3661ce4` on Node 24.16.0, Foul Play in Python 3.12.15, poke-env 0.15.0. Win rates are the
player's, with 95% Wilson intervals. "Simple" is poke-env's `SimpleHeuristicsPlayer` as shipped,
without this repository's stat fix.

Against the benchmark's opponent, Foul Play at `search_time_ms` 200 and `search_parallelism` 1:

| Player | Games | Player won | Win rate |
|---|---|---|---|
| poke-env `RandomPlayer` | 103 | 0 | 0.0% (0.0 to 3.6) |
| poke-env `SimpleHeuristicsPlayer` | 144 | 13 | 9.0% (5.4 to 14.8) |
| `SimpleHeuristicsPlayer` with the repository's stat fix (`--opponent simple` in `play.py`) | 40 | 3 | 7.5% (2.6 to 19.9) |
| **PSBot, the imitation teacher** (`players.PSBotPlayer`, 3,175 decisions, none fell back to random) | 100 | 18 | **18.0% (11.7 to 26.7)** |

Every one of those runs finished with 0 errors and with Foul Play's own log agreeing with the
count. Foul Play used Tera in these games and loaded the pinned set file.

What the search settings cost, with Simple as the player:

| `search_time_ms` | `search_parallelism` | `parallel` | Games | Simple won | Games a minute | Seconds a game | Seconds a decision | Iterations per sampled team | Cores per process | Peak MB per process |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 1 | 4 | 40 | 3 | 37.1 | 5.7 | 0.26 | 42,000 | 0.99 | 144 |
| 200 | 1 | 4 | 40 | 6 | 19.2 | 11.1 | 0.47 | 99,000 | 0.99 | 156 |
| 200 | 1 | 8 | 48 | 4 | 37.2 | 11.0 | 0.48 | 72,000 | 0.99 | 152 |
| 200 | 2 | 4 | 40 | 1 | 17.8 | 11.4 | 0.51 | 63,000 | 1.86 | 229 |
| 500 | 1 | 4 | 40 | 4 | 7.9 | 26.2 | 1.15 | 250,000 | 1.00 | 198 |
| 1000 | 1 | 4 | 24 | 1 | 4.1 | 48.9 | 2.27 | 474,000 | 1.00 | 265 |

A game against Simple is about 22 Foul Play decisions. Against PSBot the games are longer
(30 decisions, 14.3 s a game, 15 games a minute at `parallel` 4). One process uses one core at
`search_parallelism` 1, whatever the search time. At `parallel` 8 the iterations per team fell
from about 100,000 to 72,000, most likely because only six of the laptop's cores are fast ones,
and the results did not move.

Does more search make it stronger? Foul Play against itself, one core each:

| Side A | Side B | Games | A won | A's share |
|---|---|---|---|---|
| 200 ms | 50 ms | 60 | 29 | 48.3% (36.2 to 60.7) |
| 1000 ms | 200 ms | 40 | 17 | 42.5% (28.5 to 57.8) |

No difference that 40 to 60 games can show, across a twentyfold range of search time. So the
opponent at 200 ms is not a weakened Foul Play in any way these games can detect, and it is not
sensitive to the machine being busy.

Failure handling, against the real thing, with the code as it is now:

| What was done | Result |
|---|---|
| One Foul Play process killed with SIGKILL in the middle of a game (12 games asked, 3 processes) | 11 games, 1 error, a replacement process played the rest, 49 s, nothing left running |
| One process frozen with SIGSTOP, `game_timeout_s` 40 (9 games, 3 processes) | 8 games, 1 error, 56 s, nothing left running |
| `overall_timeout_s` 20 on a 12 game run | Returned after 22 s with 3 games, 9 errors and `timed_out`, the 3 open battles forfeited, nothing left running |
| The benchmark process itself killed with SIGKILL during a run | All 3 Foul Play processes gone within 1.0 s |
| Player built with `max_concurrent_battles=1`, `parallel=3` | Ran with `parallel` 1 and said so |
| Two `run_matches` calls on one player | Each counted only its own games |

The win rates and the settings table come from runs made while the bookkeeping was still being
fixed (see "What the real runs caught"). The fixes do not touch how either side plays.

## What the real runs caught

Three faults that the first version had, each found by running it and fixed:

| Fault | Seen as | Fix |
|---|---|---|
| Foul Play challenges again within milliseconds of a game ending. The books were only brought up to date five times a second, so that challenge looked like one sent during a game and was dropped. | First 20 game run: 12 games, 8 errors, every process stuck after its first game. | The books are brought up to date at the moment a challenge arrives. |
| A process that exits right after its last game could be judged before the player had processed the game's final message. | Not seen in a real run. Found by reasoning from a unit test failure that turned out to have another cause. | After a process exits, the player gets up to 3 s to see the game end, 15 s when Foul Play's log says it ended. |
| A challenge was marked as accepted only after the accept had been sent, which leaves a window in which the battle is already there. | Not seen. Found reading the code. | Marked before the accept is sent. |

## Recommended for the VM

100 games, `parallel=8`, `search_time_ms=200`, `search_parallelism=1`, and the same settings for
the whole training run so the curve is comparable.

| Why | |
|---|---|
| Cost | 8 of the 64 vCPUs for 3 to 4 minutes, and about 0.17 GB of memory per process. On the laptop `parallel` 8 gave 37 games a minute against Simple; games against a PSBot-level player are a third longer. Every few hours that is nothing next to training. |
| Level | PSBot, the imitation teacher, wins 18% against it. A policy that only copies its teacher should start near there, so the number can move both ways. Random (0%) and Simple (9%) mark the bottom. |
| Resolution | 100 games put a 95% interval of about plus or minus 10 points around 50%, and plus or minus 6 around 10%. Enough to see a trend across checkpoints, not to compare two neighbouring ones. |
| Stability | Foul Play at 50 ms and at 1000 ms did no better or worse against Foul Play at 200 ms, so load on the box has a wide margin before it changes the opponent. |

`search_parallelism` 2 doubles the CPU for no measured gain, and 500 or 1000 ms multiply the time
by 2.3 and 4.3 for no measured gain against the opponents tried here. If the policy ever wins
most of its games at 200 ms, add a second opponent entry at a longer search time instead of
changing this one.

## Limits

- **Not run on the VM.** Everything was measured in WSL on Ubuntu 26.04 with a standalone
  Python 3.12.15. The apt step of the install script has not been executed anywhere (WSL was
  off limits for apt), and Ubuntu 24.04's own Python 3.12.3 has not been tried.
- **Not run against the policy.** The players tested are poke-env's `RandomPlayer` and
  `SimpleHeuristicsPlayer`, and the repository's `PSBotPlayer`. `HybridPlayer` with a
  checkpoint has not played it.
- **Game speed on the VM is an estimate.** The VM's cores are slower per thread than the
  laptop's fast ones. Read `foulplay_mean_iterations` from the first evaluation there.
- **Logs grow.** Each game adds 60 to 75 KB to the process logs in `log_dir`, so 6 to 7.5 MB
  per 100 games. Nothing removes them.
- **One change to Foul Play's behaviour:** the login, as described above. Nothing else.
- **Private attributes of poke-env.** The accept loop uses `Player._challenge_queue` and
  `Player._battle_semaphore`, which poke-env 0.15 has. `requirements.txt` pins `poke-env<0.16`.
  With a version that lacks them `run_matches` returns all games as errors and says why.
- **Windows.** Foul Play needs Rust to build, and the laptop has none, so `available()` is False
  there. The unit tests do not need it.

## Tests

```bash
python -m pytest tests/test_foulplay.py -q
```

33 tests, no Foul Play and no server needed: `available()` in every failure case, the counting,
and `run_matches` end to end against a fake Foul Play started through the real `--child` path,
including a process that never challenges, one that is never accepted, one that crashes at
start, one that crashes in a game, a game that never ends, the overall timeout, cancellation,
and a check that stopping a process takes its own child processes with it. They pass on Windows
(Python 3.14) and on Linux (Python 3.12).
