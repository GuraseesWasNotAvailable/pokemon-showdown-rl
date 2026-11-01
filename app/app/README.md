# PSBot Lookahead.exe

A window that plays rated gen9randombattle games on play.pokemonshowdown.com with a chosen network
(the newest by default) and the look-ahead search ("mix", the veto rule), on this laptop: the
network on the GPU, Foul Play's search helpers in WSL. Double-click `PSBot Lookahead.exe` in the
folder above this one, or the "PSBot Lookahead" shortcut on the Desktop.

Build it with `build-app.cmd` (the C# compiler of .NET Framework 4 that ships with Windows). The
exe runs `ladder.py` from its own folder with the Python of a `.venv` in that folder or the one
above it (else the first `python.exe` on PATH), and lists the networks in `weights\` next to the
exe. Each can be changed with `python=`, `checkpoints=`, `armc=` and `v2=` lines in
`PSBot Lookahead.settings`. The trained networks are in the repository's v1.0 release.

## The window

- Username (remembered), password (masked, never stored, cleared when a run ends), games
  (default 20), games at once (1 to 3, default 2), who plays, thinking (a slider and a search
  style with the search, the Claude list without), the API key, network.
- Who plays: one list, the bot or one model alone.
  - Bot: network + search (Lookahead): today's bot ("mix").
  - Bot: network only: the network's best move, no search.
  - Claude Fable 5.1 / Opus 5.5 / Sonnet 5 / Haiku 4.5 alone, GPT-6 Sol alone, GPT-6 Luna alone:
    the model picks every move from a text of the position and the legal moves. No network and
    no search are behind it, and no network file is used. Roughly 5 cents a move for Fable 5.1,
    2 for Opus 5.5, 1 for Sonnet 5, and under half a cent for Haiku 4.5, Sol and Luna. The run
    stops asking the model at $20 of API use (ladder.py --max-model-usd).
  When a model gives no usable answer (a timeout, an API error, a refusal, a move that is not
  legal) a simple rule plays that move: the strongest attack, or the healthiest switch-in.
  The modes where a model had the last word over the bot (Luna decides, Sol decides) and reading
  ahead left the window on 2026-10-14: on the ladder neither played better than the bot itself
  (bench/notes/lines.md). They are still arms of ladder.py.
- Thinking (the Claude models but Haiku): low, medium or high (ladder.py --claude-effort). Low is
  the default and keeps a move at about 10 s or less. There is no cut-off at 10 s: an answer is
  waited for up to 25 s (--claude-timeout), then the rule plays.
- API key: Anthropic's for the Claude models, OpenAI's for Sol and Luna; the label says which.
  Masked, never stored, cleared when a run ends. Left empty, the bot uses the variable
  ANTHROPIC_API_KEY or OPENAI_API_KEY when this computer has it. Before the first game the bot
  asks the model one tiny question and two of a game's size; a key or a model that does not work
  stops the run with the API's own message and nothing is played.
- Network (the bot's modes only): the list holds the main line's three newest files and the baseline
  run's; tick "all" for every file. The newest whole file is selected. A file that does not end like a whole zip archive
  (an interrupted copy) is marked "incomplete file: not usable" and Start refuses it; one written
  in the last two minutes is marked and Start asks first. The list is read again when it is
  opened and when Start is pressed, so a network copied while the window is open shows up; Start
  takes the newest whole one unless one was picked by hand.
  Files named `baseline_update_N.pt` in the same folder are networks of the search branch's baseline
  run. They are listed after the main line as "baseline run, update N" and are never the default
  pick. They carry their own numbering (it overlaps the main line's), and the bot builds the
  observation that branch trained them on (`obs_search.py`: own Tera types, STAB kept after a Tera).
  Files named `armC-full_update_N_xM.pt` in `weights\armc\` (or an
  `armc=` line in the settings) are the full run from arm C (see that folder's README.md), listed
  last as "arm C full run, update N (xM)", never the default pick. They are search-branch networks
  too, so the bot builds the same observation for them.
  Files named `v2-<arm>_update_N_<M>M.pt` in `weights\` are networks of branch
  train-v2, exported on the training box by `tools/export_v2.py` (network only, run_cfg kept).
  They are listed as "NEW TRAINING: <what the arm
  trained with>, M new moves"; the from-scratch run is "scratch", "from scratch, every technique"
  (M = updates x 36,864). A network trained with train-v2's matchup block (the from-scratch run,
  musethead) is played on exactly that branch's observation and network (`obs_v2.py`: the matchup
  and moveset blocks, its action mask; the training-only heads are left out at load), checked
  against train-v2's own worker and network on recorded battles by `bench/v2_exactness.py`. One
  whose run built the network otherwise (attention heads, layers, history length, feed-forward
  width in its run_cfg) is refused at load instead of played as another network.
- Lookahead: search 500 ms per decision on games-at-once + 1 helpers (at most 4); the move is played
  within 2.5 s, the network's when the search has not answered by then.
  Thinking (the search modes only, in the place of the Claude list and of the API key's row): a
  slider from 0.5 to 7 s a move in half seconds (default 1 s) and a search style, both remembered
  (think= seconds, searchstyle=). Deeper, the default, is the search as before: Foul Play guesses
  2 opponent teams a move (4 at half the time early in a game) and searches each longer, the time
  over 2.26 to 100 ms (1 s: 500 ms, cap 2.5 s; 2 s: 900 ms, 2.7 s; 5 s: 2200 ms, 6.0 s, as the old
  list had them). Wider keeps 500 ms a team and guesses that many times more teams, one after
  another in the battle's helper (ladder.py --player mix-deep --deep-worlds W: W 2 at 2 s up to 6 at
  6.5 s; below 2 s it is Deeper; each W more, 2 more teams, is about 1.1 s, so some neighbouring stops play the
  same); in the guarded mode it adds --trust 0.03, the guard of mix-trust.
  Wider is new and not yet tried on the ladder (reports/2026-10-16/SEARCH_TIME.md: 5 s searched
  deeper won no more than 1 s). Every cap is search_player.minimum_deadline + 50 ms to the next
  tenth, never under 2.5 s, and at most 8.2 s: 1.8 s under the battle timer's 10 s for the order
  spacing (ladder.py --order-max-wait-s 1.5) and the trip to the server. The label by the slider
  gives the time the plan expects. `--test-think` prints the whole table. A decision
  takes about 2.26 times the search time (10 ladder games, 2026-10-15). The search has a time budget,
  not a number of turns ahead. After each game the log and the search line say how long moves took
  and how many search iterations a decision had (ladder.py's game_end: decision_ms, iterations). Foul Play in WSL is checked
  on opening when Lookahead is selected and at least 8 GB of memory is free; "Check Foul Play"
  checks it at any time. When the check fails the label says why in red, Lookahead stays
  selected, and Start asks whether to play the network alone this once (the saved choice stays
  Lookahead).
- Start warns when less than 8 GB of memory is free (the default answer is No).
- Stop after these games: no new games, the ones in progress are finished.
  Stop now: asks first; the games in progress are lost on time.
  Closing the window during a run asks: finish the games in progress (the window waits and then
  closes), stop now, or cancel.
- Status: running or not, games played and the record, rating at the start and now, the games in
  progress with their turn. The log shows every order sent (and what the search did for it), every
  result and rating change. Elo graph writes `runs/ladder/elo.html` and opens it.
- The column on the right, drawn from the run's own "@app" lines:
  - Elo, this run: the rating before the first game, then after each game (green dot a win, red a
    loss), the dashed line where it started.
  - What it looks at: where the network's decision token puts its attention for the move it just
    made, in each of its 3 layers (L1 to L3, 8 heads averaged) and in the readout that picks the
    move (Out). Columns: the field, your 6 Pokemon and their 6 (each with its 4 moves), and the
    memory of earlier turns. Each row is shaded against its own largest share; a yellow mark is the
    Pokemon in battle; hovering a cell gives the percentage.
  - Neurons firing: the same token's 1024 feed-forward neurons per layer, one dot each, grey when
    silent and red to white the harder it fires, with the share that fires.
  - The "3D" pill on that panel (remembered, on by default) draws the network as 3D blocks
    instead: the input encoder, the 3 layers, the readout and the move head, each block's face as
    big as its parameters (square root: 4.5M, 2.1M each, 7k), every neuron a cell on it that
    flashes as it fires; the encoder's 14 cells are the readout's attention, the head's the move
    probabilities. Click the pill for the flat diagram of the strongest neurons and their weights.
  One game is followed at a time (the first that moves, until it ends). Both network panels come
  from `ladder.py --app-net-view` (net_viz.py), which reads the forward pass that picks the move:
  no second pass, the same moves (tests/test_net_viz.py). A model alone has no network to show.

## One bot per account

Two clients logged in as one account both get every battle and both send orders. So:

- Only one window opens; a second double-click says the window is already open.
- Start looks for another `ladder.py` playing on the same server on this computer (the Desktop's
  Look Play.cmd, a command line). One that plays as the same account, or whose account is not on
  its command line (Look Play.cmd asks for it), stops the start; for the second kind Start asks,
  default No, in case it plays a different account.
- `ladder.py` takes a lock per server and account (a named mutex) and exits at once when another
  run holds it. Look Play.cmd runs the `search` branch's ladder.py, which does not take it, which
  is why the window also looks at the running processes, at Start and every 15 s during a run
  (a bot that appears is reported in red).

## Where things go

- `runs/ladder/results.jsonl` and `runs/ladder/battles/`: ladder.py's results and battle logs.
- `runs/app/run-<time>.log`: everything the bot printed in one run; `runs/app/stop-*`: stop files.

## How it is safe to leave

- The password is written as one line to the bot's standard input, and the pipe is then closed
  (`ladder.py --password-stdin`). The venv's `python.exe` is a launcher that starts the real
  interpreter and hands it that pipe, so the password is in no process's environment, on no
  command line, and in no log, result or setting.
- The OpenAI key goes the same way, as the second line of that standard input (`ladder.py
  --key-stdin`). ladder.py keeps it in memory, puts it into the Authorization header of its
  requests to api.openai.com and nowhere else, and writes *** in its place in everything it
  prints or logs.
- The child Python is in a job object: when the window goes, however it goes, the child and
  everything it started on Windows go with it. The helpers inside WSL leave by themselves within
  20 s once their wsl.exe is gone (`bench/fp_bridge_server.py --parent-watch beat`); on a normal
  stop or "Stop now" ladder.py stops them at once. Known gap: the job is joined just after the
  launcher starts, so on a loaded laptop the real interpreter can be born outside it; a hard kill
  of the window then still ends the interpreter (the launcher's own job), and its WSL helpers
  leave within 20 s.

## Test options

`"PSBot Lookahead.exe" --test-server PORT --test-user NAME` plays on ws://127.0.0.1:PORT instead of
the public server (no password is needed or used; results go to `runs/ladder-test/`), and says
TEST in the title. `--test-foulplay-dir DIR` gives only the child FOULPLAY_DIR=DIR (a missing Foul
Play), `--test-memory-warn-gb N` sets the free-memory bar of the warning and of the check on
opening. Settings of a test run are kept apart, in `PSBot Lookahead.test.settings`.
`--test-think [FILE]` opens no window: it prints (or writes to FILE) what every stop of the thinking
slider passes to ladder.py in both search styles and both search modes (`Thinking.Table`; a copy is
`tests/fixtures/app_think_table.txt`, which `tests/test_search_player.py` checks against ladder.py).
`--test-replay FILE` handles the "@app" lines in FILE as if a run printed them (nothing is started
or sent), and with `--test-snapshot PNG` saves the window as a picture and closes: a way to see the
panels without playing.
