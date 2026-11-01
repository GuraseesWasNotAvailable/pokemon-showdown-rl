"""
Play the trained policy on a Pokemon Showdown ladder and print how it does.

    python ladder.py --games 20                       the public ladder, your account
    python ladder.py --history                        every game so far and the rating after each
    python ladder.py --graph                          the rating graph, in the browser
    python ladder.py --server local:8520 --games 6    a local server, for a test

The player is the one evaluate.py measures: the transformer alone, its best legal action each
turn (players.batched_policy_player). It plays --parallel games at the same time and waits
--think-ms before each move. Apart from its moves it sends the server three things: the ladder
search, the battle timer request at the start of each game (as PSBot does, so that an opponent
who stops moving cannot hold a game open; --no-timer turns it off), and leaving the room when a
game is over. After a lost connection, and only then, also the login again and joining its games'
rooms again (see "Losing the connection"). No chat, no settings.

With a search beside the network (search_player.py; the search runs in helper processes, inside
WSL on Windows, see bench/FOULPLAY_BRIDGE.md):

    python ladder.py --player mix                     the network and the search combined
    python ladder.py --player search                  the search's own choice
    python ladder.py --player predict                 mix, left only on a confident prediction of the
                                                      opponent's action (opponent_model.py; --predict-confidence,
                                                      --predict-gain, --predict-risk, --payoff-ms)
    python ladder.py --arms net,mix --games 60        every game gets one of the listed modes,
                                                      balanced, and its arm is written with its result
    python ladder.py --player mix-guard               mix with the guard against set-up moves and sleep
                                                      (search_player.MixSettings.guard; --guard turns it
                                                      on in every mix game)
    python ladder.py --arms mix,mix-counter           today's mix against mix with round 3's counter rules,
                                                      game by game: in a set-up position use the counter on
                                                      the team, and don't hit the boosted defence
                                                      (--setup-counter, --setup-other-side: one of them, or
                                                      both, in every mix game; off by default)

The three-arm A/B (launchers/laptop/Look Play AB.cmd, bench/ab_report.py):

    python ladder.py --arms net,mix,mix-deep --experiment ab3-net-mix-deep-024381 --parallel 3 --helpers 4

mix-deep is mix with its own search budget (search_player.SearchBudget, --deep-*): 1250 ms per
world on twice the sampled worlds, about 5 times mix's search, an answer within 7.5 s, and today's
budget for the rest of a game once the battle timer shows under 90 s or a decision took over 9 s.
--experiment tags the games in results.jsonl; the arms' balance and the report count that tag.

Settings per game (tools/search_sweep.py plays the app's thinking settings this way, all at once):

    python ladder.py --arm-settings settings.json --parallel 6 --games 30 --experiment sweep-1

settings.json is a list of named settings, [{"name": "2s-deeper", "args": ["--player", "mix-trust", "--search-ms",
"900", "--deadline", "2.7"]}, ...]. Every new game gets one (search_player.SettingPicker: always one of those with
the fewest games with a result in this experiment, and among those one with no game in progress), and plays with
that setting's --player, --search-ms, --deadline, --deep-* and --trust exactly as ladder.py started with that
setting's arguments alone would. Every other flag is the run's: a setting that gives one must give this run's
value, and a file whose settings disagree on one is refused, as is a setting whose search could not be played (the
same checks as the flags). --helpers is --parallel + 1 unless given. Each game's line in results.jsonl says its
"setting", the "setting_budget" its decisions are asked with, a deep setting's "setting_fallback", and how many
searched decisions had each ("budget_asked"). Without --arm-settings nothing of this is used.
--deep-workers W searches the deep arm's worlds W at a time, each in a process of Foul Play's own
pool (one core each) instead of one after the other in the battle's helper: the same search in
about 1/W of the time (the box: cloud/box_ladder.sh, run_box_ab.sh).
Every decision note (every arm) also has the network's own score of the position: v_exp and
v_std of the value head from the forward pass that made the move, phi (the reward's potential of
the position) and phi_scale, and win_est = (1 + v_exp + phi_scale x phi) / 2.

A model decides, and reading ahead (luna_player.py, search_player.py; the window app has all of it):

    python ladder.py --player sol                     OpenAI's gpt-6-sol picks the action (the Responses API; about
                                                      1 to 3 s and half a cent a decision; up to --sol-timeout 5 s)
    python ladder.py --arms luna,sol                  the two models, game by game
    python ladder.py --player mix-lines               today's bot that reads ahead: after the search its 3 favourite
                                                      actions are each played forward 5 turns (--lines-*), and mix's
                                                      pick gives way to one whose line ends --lines-margin 0.10 better
    python ladder.py --player luna-lines              the model decides with those lines in its brief
    python ladder.py --player sol-lines
    python ladder.py --luna-check --model sol         the check for Sol
    --key-stdin                                       the OpenAI key is the next line of standard input (after the
                                                      password's with --password-stdin): how the window app gives it
    python ladder.py --player luna-solo               the model ALONE: no search is started and the network's pick is
    python ladder.py --player sol-solo                never played. The model gets the position and the legal actions
                                                      with their facts and its choice is the order; when it gives none
                                                      a rule plays (the strongest attack, the healthiest switch-in).
                                                      Untested for strength: expect far below today's bot

Luna decides (luna_player.py; launchers/laptop/Luna Play.cmd):

    python ladder.py --luna-check                     one tiny real question to OpenAI's Decisions API (the key,
                                                      the endpoint, the answer's shape), then one of a game's
                                                      size (is it taken, how long does it take). Prints what came
                                                      back (never the key); exit code 0, or 5 with the reason
    python ladder.py --player luna                    Luna picks the action in every game
    python ladder.py --arms mix,luna --experiment luna-vs-mix-024381      Luna and today's bot, game by game

In a "luna" game mix decides as it does today, and then OpenAI's model gpt-6-luna is asked ONE
choice question: the battle in plain words, and every legal action with its facts, the network's
probability, the search's share and whether mix would play it. Its choice is played. Mix's pick
is played instead when no legal answer came within --luna-timeout s (1.2; never past --deadline),
on an HTTP error, a refusal, an answer of the wrong shape or a choice that is not legal; for 60 s
after 5 API failures in a row; and when the blunder guard overrules Luna (its pick under 1% of
the network and of the search, mix's over 60% of either; --luna-guard off turns it off). Each is
counted per game (results.jsonl: luna_asked, luna_answered, luna_played, luna_overruled,
luna_agreed_with_mix, luna_fallbacks by reason, the latency, the tokens and their cost) and
written in the decision's note ("luna", with the brief and the choices as they were sent unless
--no-luna-save-briefs). The run starts with the same check as --luna-check and plays nothing
when it fails (exit code 5); when the API refuses the key in the middle of a run no new games
are started, the games in progress are finished by mix, and the run ends with exit code 5.
The key comes from the environment variable OPENAI_API_KEY, else it is typed once at a hidden
prompt. It is kept in memory only: never in a file, a log, a note, a command line or the console
(errors print the HTTP status and the API's message with the key and the Authorization header
as ***, and the console log masks it as well).

Showdown's typing limit (OrderGuard): every message waits --order-spacing-ms (300) after the one
before it, across all games, and an order the server drops ("typing too quickly", in its battle
room) is sent again 1, 2 and 4 s later, while it is still the order for the current request, at
most 3 times; each timer line that says the server still waits for our order sends it once more
(up to 8 times). A notice with no room sends no order again (it was a search or a /leave).
Only one bot plays an account at a time: a second ladder.py (or the PSBot Lookahead app) on the
same account and server refuses to start (the account lock, as in the app).

Predictions (on by default with a search, --no-predictions turns them off): for every decision
of a searched game, one line with the opponent's three likeliest actions as opponent_model.py
reads them from the search's own view of the opponent's side (no extra search time), what we
play, and after the turn whether the read was right:

    [1408] T7 Venusaur: predict Sleep Powder (sleep) 58%, switch Lurantis 22%, Sludge Bomb 12% | we play Meteor Mash
    [1408] T7 read: wrong, it played Sludge Bomb (our 3rd guess)

[1408] is the game (the last digits of its number). Each line is also kept in the battle's file
as a note, and each game's line in results.jsonl counts its predictions (pred_made, pred_scored,
pred_top1_right, pred_top3_right). They change no move.

On the training box the bot runs in a tmux session started by cloud/ladder_box.sh (the Desktop's
Net Play, Net Stop and Net Elo call it): "mix" in every game, 3 games at a time on 4 helpers.

The search on the cloud box, the player here (launchers/laptop/Look Play Remote.cmd, cloud/remote_search.sh):

    python ladder.py --arms net,mix,mix-deep ... --parallel 6 --helpers 7 --search-remote vm
    python ladder.py ... --search-remote ubuntu@1.2.3.4 --search-key C:/Users/me/.ssh/box.pem

Showdown sees this machine's own connection and the network runs on its GPU; only Foul Play's
search helpers run on the box, each started over ssh (foulplay_bridge.RemoteTarget: no terminal,
BatchMode, keep-alives), on the box's Foul Play (~/foul-play) and its copy of this repository
(--search-remote-dir, ~/dev/ab-ladder; its helper file must be this one, or nothing starts).
"vm" is the box cloud/vm.sh talks to, its address looked up at the start (it changes at every
start). Nothing else changes: the same deadlines from the request (--deadline, --deep-deadline),
with the round trip measured when the helpers start (printed) and taken off every decision's time
on the box (a search that cannot be answered in time is not started, and the network's move is
played); the start refuses when a deadline cannot hold a search and that round trip. The round
trip is kept up to date (a probe every 2 s) and written with every searched decision and every
game ("search_at": "remote" or "wsl", "rtt_ms"). The helpers there stop when this run ends or
dies (its ssh sessions end; a connection that goes quiet stops them within 20 s), and close()
checks the box for any left over.

A search needs at least one helper more than --parallel (it refuses to start otherwise: with
fewer, games queue behind each other and their searches come too late). The helpers of a ladder
run are watched closely: one that answers nothing for 3 x --deadline (5 s at least) is replaced,
an idle one is pinged every 3 s so that a frozen one is found before a battle is put on it, and
a question that cannot be answered in time is given up at once. When the search failed in more
than --max-fallback-share of the last --health-window decisions it was asked, no new games are
started (the games in progress are finished): a bot whose helpers keep failing does not ladder
on as the network alone.

With --search-remote the link to the box is a condition of its own, not a failure of the search.
It is down while a helper has answered nothing for 4 s although it has work, while one whose ssh
dropped (or that the watchdog stopped) is being replaced, and for 5 s after a line the network
held up (foulplay_bridge.FoulPlayBridge.link_status). Then "search link down at HH:MM:SS (why)"
is printed, no game is started (a ladder search out is withdrawn), and the games in progress get
the network's moves at once. When it is back ("search link back at ..."), the decisions that fell
back while it was down are taken out of the --health-window (they never count for
--max-fallback-share) and games are started again. Only a link down for more than
--link-down-limit seconds in one go (180) stops the run, in one line ("STOPPING: the link to the
search helpers has been down since ..."); the games in progress are finished. Failures outside
the link's troubles (an error answer, a timeout on a link that held nothing up, a helper whose own
process failed on the box) are the search's, counted as before. With remote helpers a decision is
judged 7 s after it was made, when a link stall in it is known.

--player net (the default) is the network alone, as before: nothing is started and nothing is
sent to a search. --search-ms is the search time per decision and --helpers the number of helper
processes (one core each while they search; one more than --parallel keeps a search from waiting
behind another). The search is waited for until --deadline seconds after the request; when it
has not answered by then the network's move is played, and that is counted.

Every battle's messages, as they came and as they were sent, are kept in
runs/ladder/battles/<battle tag>.jsonl (battle rooms only: nothing about the login is in them).

Account. The name comes from --user or the PS_USERNAME environment variable. The password is
read from PS_PASSWORD, or typed at a prompt in this window when that is not set, or, with
--password-stdin, read as the first line of standard input (how the window app gives it: then it
is in no environment block at all). It is never printed, logged or written to a file, and
PS_PASSWORD is taken out of this process's environment as soon as it is read, so nothing started
from here inherits it. A local --no-security server needs no password (one given is dropped, and
nothing is sent to the login server).

One bot per account. A run takes a lock named after the server and the account (a named mutex
on Windows, a lock file under ~/.cache/psbot-ladder elsewhere, or $PSBOT_LOCK_DIR) and refuses to start when another
run holds it: two clients on one account would both get every battle and both send orders. The
lock goes with the process, however it ends.

The login (every one: the first and each after a lost connection, log_in_checked) is told in
one line per step, with nothing secret in it (never the password, never the login assertion):
the login server's answer ("the login server accepted it (status 200)"), then Showdown's
confirmation ("logged in as NAME"; a lock mark in front of the name is told too). What went wrong
is one line: the login server's refusal with its reason and HTTP status ("Wrong password."), a web
page where its JSON should be (a block on this address), no answer, or Showdown's own |nametaken|
or |popup|. Showdown gets --login-wait s (30) to confirm, with "waiting for Showdown to confirm
the login (N s)" every 5 s. A refused first login ends the run with exit code 4 and is not tried
again (start it again and type the password carefully); one that did not go through for another
reason ends it with exit code 2. --exit-file writes the reason in one line for run_box_ab.sh.

Stopping. When a file named ladder.stop appears next to this script it starts no more games,
finishes the ones in progress and stops. It also stops after --games games. Ctrl+C in the middle
of a game abandons that game, and the server then counts it as a loss on time: use the stop file.
--stop-now-file names a second file: when it appears the run stops at once, leaving the games in
progress (lost on time) and stopping the search helpers.

The window app. "PSBot Lookahead.exe" (app/PSBotLookahead.cs, built by build-app.cmd) runs this
script with --app-events and shows what it does: those "@app {json}" lines say when a game
starts, every order sent (with what the search did for it), every result and rating change.

Watching. Log in to the same account in an ordinary browser tab (without PSBot running there):
the games show up as your battles and can be watched live.

Losing the connection. A game is never given up because of something that is not Pokemon. When
the websocket closes (or poke-env stops listening, or a keep-alive ping, sent every 5 s, goes
unanswered for 10 s: a dead connection is noticed within about 17 s) the bot says so ("connection lost at HH:MM:SS, reconnecting (attempt 1)..."),
connects and logs in again the way it did at the start (the password, when there is one, is the
one it holds in memory; it is never written anywhere), retrying for --reconnect-for seconds (5
minutes: every second for the first 90 s, then 2, 4, 8 and 15 s apart), and starts no new game
meanwhile. It then joins every game in progress again (every battle not over, also one that began
a moment before the drop), and any
game the server says the account is in, and the server sends each one again from the start and
then its current request ("rejoined 2 battles", with the turn and the timer of each). The battle is
rebuilt from that, and the request is answered within the timer: decided as usual when it is a new
one, and when it is the one answered just before the connection went, not decided twice (the order
that went into the lost connection is sent again; nothing when the server says it has it). The
search cannot pick a battle up midway, so the rest of such a game is the network's, counted as
"rejoined" (not as a failure of the search). Showdown gives a player who disconnects a minute to
come back, counted from when the SERVER sees the drop and running alongside the turn timer (not on
top of it): a game in progress is lost about 60 s after the drop, whatever its timer says, minus
the time the bot took to notice. Hence the fast pings and the retry every second. A connection
that goes again while the games are being joined is made again at once. When it gives up, each
game still in progress gets a results line with no result ("abandoned") and ladder.py exits with
code 4; so it does at once when a login after the drop is refused with a reason (the same password
is not tried again every second). bench/ladder_faults.py makes all of this happen on a local server (the connection cut,
kicked by the server or stalled; the helpers killed or frozen).

The search helpers going away (WSL restarting, say) does not stop a game either: a decision whose
helper is not up is not waited for, the network's move is played at once (counted as "down", a
failure for the rule below), the helpers are started again as soon as they can run, and the bot
says so ("search helpers down at HH:MM:SS: playing the network's moves until they are back", then
"search helpers back").

Console log. Everything printed also goes to runs/ladder/console-<local date>.log (next to the
results file), appended and flushed line by line, with the time on each line: what happened can be
read afterwards. --no-console-log turns it off. Never the password.

Results go to runs/ladder/results.jsonl, one line per game, with the rating before and after.
The rating after each game is printed as it is played, and --history prints all of it.
runs/ladder/elo.html is the same as a graph (ladder_graph.py): it is rewritten after every game
and reloads itself in the browser. --graph opens it and keeps it up to date from the results
file, whichever version of this script is doing the playing.
"""

from __future__ import annotations

import argparse
import asyncio
import calendar
import copy
import getpass
import glob
import html
import inspect
import json
import os
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs", "ladder", "results.jsonl")
GRAPH = os.path.join(HERE, "runs", "ladder", "elo.html")
BATTLES = os.path.join(HERE, "runs", "ladder", "battles")
RATING_RE = re.compile(r"([^<>|]*?)'s rating: (\d+) &rarr; <strong>(\d+)</strong>")
TRAIL_SHOWN = 14
PLAYER_MODES = ("net", "search", "mix", "mix-veto", "mix-pool", "mix-shortlist", "predict", "mix-guard",   # search_player.MODES
                "mix-fix", "mix-defer", "mix-fixdefer", "mix-fixdefer-long", "mix-fix-long",    # search_player.ARM_FIXES
                "mix-blend4", "mix-blend8", "mix-counter", "mix-trust",
                "mix-deep",                                                                     # search_player.DEEP_ARM
                "luna",                                                                         # search_player.LUNA_ARM
                "sol",                                                                          # search_player.SOL_ARM
                "mix-lines", "luna-lines", "sol-lines",                                         # search_player.LINES_ARMS
                "luna-solo", "sol-solo",                                                        # search_player.SOLO_ARMS
                "fable-solo", "opus-solo", "sonnet-solo", "haiku-solo")                         # search_player.CLAUDE_SOLO_ARMS
# The models an arm can ask, in the order they are named: OpenAI's (luna_player.MODELS), then
# Anthropic's (claude_player.MODELS).
OPENAI_MODELS = ("luna", "sol")
CLAUDE_MODELS = ("fable", "opus", "sonnet", "haiku")
CLAUDE_LABELS = {"fable": "Claude Fable 5.1", "opus": "Claude Opus 5.5", "sonnet": "Claude Sonnet 5", "haiku": "Claude Haiku 4.5"}
LUNA_FAILED = 5           # exit code: Luna cannot be used (no key, the check failed, an address the key must not go to)
LOW_ITERATIONS = 60_000   # below this the search is weaker than anything measured as good (bench/FOULPLAY_BRIDGE.md)
FAST_RETRY_FOR = 90.0     # s after a lost connection during which a new one is tried every second (see reconnect)
MAX_HELPERS = 12          # --helpers: 6 games at a time (Showdown's own cap per account) need 7
MAX_DEEP_WORKERS = 16     # --deep-workers
HELPER_RESTART_GAP_S = 600.0   # --helper-restarts: at most one restart of the search helpers in this many seconds
APP_EVENTS = False        # --app-events: also print "@app {json}" lines for the window app (app/PSBotLookahead.cs)


def app_event(kind: str, **fields: Any) -> None:
    """
    One line for the window app: "@app " and a JSON object whose "ev" is kind. Only with
    --app-events. Never the password: nothing that comes from the login is ever put in one.
    """
    if not APP_EVENTS:
        return
    try:
        fields["ev"] = kind
        print("@app " + json.dumps(fields, default=str), flush=True)
    except Exception:   # a note for the app must never disturb a game
        pass


def attach_net_view(player: Any) -> None:
    """
    --app-net-view: a "net" event for every move the network makes (net_viz.py), with the turn and
    the 12 Pokemon in the observation's order (own team, then the foe's as seen), for the window's
    attention and neuron panels.
    """
    from net_viz import NetViz

    def species(team: Any) -> List[str]:
        return [str(getattr(m, "species", "") or "") for m in list(team.values())[:6]]

    def active(team: Any) -> int:
        """The slot of the Pokemon in battle, -1 for none."""
        for i, m in enumerate(list(team.values())[:6]):
            if getattr(m, "active", False):
                return i
        return -1

    def action_name(index: int, b: Any) -> str:
        """An action index as the window shows it: the move (+ Tera) or the switch-in."""
        try:
            target, kw = player.assembler.map_index_to_order(int(index), b)
            if getattr(target, "species", None):
                return "switch " + str(target.species)
            name = str(getattr(target, "id", "") or target)
            return name + (" + Tera" if kw.get("terastallize") else "")
        except Exception:
            return "action %d" % int(index)

    def sink(tag: str, summary: Dict[str, Any]) -> None:
        b = player.battles.get(tag)
        own = species(b.team) if b is not None else []
        foe = species(b.opponent_team) if b is not None else []
        own_active = active(b.team) if b is not None else -1
        foe_active = active(b.opponent_team) if b is not None else -1
        if b is not None and summary.get("acts"):
            summary = dict(summary, act_names=[action_name(a, b) for a in summary["acts"]])
        app_event("net", battle=tag, turn=int(getattr(b, "turn", 0) or 0), own=own, foe=foe,
                  own_active=own_active, foe_active=foe_active, **summary)

    player.net_viz = NetViz(sink)


class BattleRecorder:
    """
    Keeps what a battle room sent and what was sent to it: one file per battle, one JSON object
    per line, {"t": seconds, "dir": "in" | "out" | "note", "data": ...}. "in" is a websocket
    message of the room without its first line (the room's name), "out" what the player sent to
    the room, "note" something the player says about the game (its arm, a decision's record).

    Only rooms whose name starts with "battle-" are written, so nothing of the login (which goes
    to the lobby) can end up here. It never raises: a full disk must not disturb a game. A file
    that is already there (a local server numbers its battles from 1 again after a restart) is
    left alone and the new battle gets <battle tag>.2.jsonl, .3 and so on.

    The files are written by a thread of their own. write() is called from the player's event
    loop for every message, and a disk that is slow for a moment must not hold up a move: it
    only puts the line in a queue. flush() waits until everything queued so far is on disk.
    """

    def __init__(self, folder: str):
        import queue
        import threading

        self.folder = folder
        self.paths: Dict[str, str] = {}
        self.files: Dict[str, Any] = {}
        self.failed = 0
        self.seen: set = set()                    # rooms a line was queued for (the caller's side)
        self._queue: Any = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="battle-recorder", daemon=True)
        self._thread.start()

    def _path(self, room: str) -> str:
        path = self.paths.get(room)
        if path is None:
            os.makedirs(self.folder, exist_ok=True)
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", room)
            path = os.path.join(self.folder, safe + ".jsonl")
            n = 1
            while os.path.exists(path):
                n += 1
                path = os.path.join(self.folder, f"{safe}.{n}.jsonl")
            self.paths[room] = path
        return path

    def write(self, room: str, direction: str, data: Any) -> None:
        if not isinstance(room, str) or not room.startswith("battle-"):
            return
        self.seen.add(room)
        self._queue.put(("line", room, {"t": round(time.time(), 3), "dir": direction, "data": data}))

    def close(self, room: str) -> None:
        """The battle is over. Its file is closed once its queued lines are written; a late line opens it again."""
        self._queue.put(("close", room, None))

    def flush(self, timeout: float = 10.0) -> bool:
        """True when everything queued before this call has been written."""
        import threading

        done = threading.Event()
        self._queue.put(("mark", "", done))
        return done.wait(timeout)

    def close_all(self) -> None:
        self._queue.put(("close_all", "", None))
        self.flush()

    def _run(self) -> None:
        while True:
            kind, room, payload = self._queue.get()
            try:
                if kind == "line":
                    f = self.files.get(room)
                    if f is None:
                        f = self.files[room] = open(self._path(room), "a", encoding="utf-8")
                    f.write(json.dumps(payload) + "\n")
                    if self._queue.empty():
                        f.flush()
                elif kind == "close":
                    f = self.files.pop(room, None)
                    if f is not None:
                        f.close()
                elif kind == "close_all":
                    for name in list(self.files):
                        self.files.pop(name).close()
                elif kind == "mark":
                    for f in list(self.files.values()):
                        f.flush()
                    payload.set()
            except Exception:
                self.failed += 1
                if kind == "mark":
                    payload.set()


class ConsoleLog:
    """
    Everything the bot prints, on stdout and on stderr (poke-env's and the search helpers' warnings
    as well), also goes to <folder>/console-<local date>.log: appended, one line at a time with the
    local time in front, flushed at every line. A new file starts at midnight. What is shown in
    the window is not changed.

    The password and the login assertions are never printed. Should one ever be in a line all the
    same (an exception's text, say), it is replaced by *** before the line is written. Writing never raises: a full disk must
    not disturb a game.

    `more` are further texts that must never be shown or written (the OpenAI key of the Luna arm):
    each is put as *** in the file, and in the window as well (hide).
    """

    def __init__(self, folder: str, secret: Optional[str] = None, more: Optional[List[str]] = None):
        import threading

        self.folder = folder
        self._secret = secret if secret and len(secret) >= 3 else None
        self._more = [m for m in (more or []) if m and len(m) >= 6]
        self._lock = threading.Lock()
        self._date: Optional[str] = None
        self._file: Any = None
        self._partial: Dict[str, str] = {}
        self.failed = 0

    def path(self, date: Optional[str] = None) -> str:
        return os.path.join(self.folder, f"console-{date or time.strftime('%Y-%m-%d')}.log")

    def hide(self, text: str) -> str:
        """The text with the further secrets (the API key) put as ***."""
        for secret in self._more:
            if secret in text:
                text = text.replace(secret, "***")
        return text

    def line(self, text: str) -> None:
        """One line into the file only (the run's header, say)."""
        with self._lock:
            self._write(text)

    def _write(self, text: str) -> None:
        try:
            date = time.strftime("%Y-%m-%d")
            if date != self._date or self._file is None:
                if self._file is not None:
                    self._file.close()
                os.makedirs(self.folder, exist_ok=True)
                self._file = open(self.path(date), "a", encoding="utf-8")
                self._date = date
            if self._secret and self._secret in text:
                text = text.replace(self._secret, "***")
            text = self.hide(text)
            for secret in SECRET_TEXTS:          # the login assertions (never printed: a second net)
                if secret in text:
                    text = text.replace(secret, "***")
            self._file.write(time.strftime("%H:%M:%S ") + text.rstrip("\r") + "\n")
            self._file.flush()
        except Exception:
            self.failed += 1

    def feed(self, name: str, data: str) -> None:
        if not data:
            return
        with self._lock:
            text = self._partial.pop(name, "") + data
            *whole, rest = text.split("\n")
            for piece in whole:
                self._write(piece)
            if rest:
                self._partial[name] = rest

    def wrap(self, stream: Any, name: str) -> Any:
        log = self

        class Tee:
            def write(self, data: str) -> int:
                if log._more and isinstance(data, str):
                    data = log.hide(data)           # the API key never reaches the window either
                n = stream.write(data)
                try:
                    log.feed(name, data if isinstance(data, str) else str(data))
                except Exception:
                    pass
                return n

            def flush(self) -> None:
                stream.flush()

            def __getattr__(self, attr: str) -> Any:   # isatty, fileno, encoding and the rest: the stream's own
                return getattr(stream, attr)

        return Tee()


class ConsoleLines:
    """
    Lines for the window, printed by a thread of their own in the order they were given: the
    predictions (about two lines a decision) and the game results. Both come from the player's
    event loop, the predictions before the order is sent. In the old Windows console (conhost
    with QuickEdit) a click or a selection in the window holds every write to it until Esc or a
    right click, and a print on the event loop would hold the move and the websocket's
    keep-alive pings with it (review of 2026-10-07, finding 1). Here only this thread waits, and
    the lines come out once the window is let go. At most MAX_QUEUED lines wait; more are
    dropped and counted. Never raises.
    """

    MAX_QUEUED = 5000

    def __init__(self) -> None:
        import queue
        import threading

        self.dropped = 0
        self._queue: Any = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="console-lines", daemon=True)
        self._thread.start()

    def say(self, text: str) -> None:
        if self._queue.qsize() >= self.MAX_QUEUED:
            self.dropped += 1
            return
        self._queue.put(("line", str(text)))

    def flush(self, timeout: float = 5.0) -> bool:
        """True when every line given before this call has been printed (False: the window is held)."""
        import threading

        done = threading.Event()
        self._queue.put(("mark", done))
        return done.wait(timeout)

    def _run(self) -> None:
        while True:
            kind, payload = self._queue.get()
            try:
                if kind == "line":
                    sys.stdout.write(payload + "\n")
                    if self._queue.empty():
                        sys.stdout.flush()
                else:
                    sys.stdout.flush()
            except Exception:
                pass
            finally:
                if kind == "mark":
                    payload.set()


def mix_settings(args: Any) -> Any:
    """search_player.MixSettings from the flags: the rule, the guard and the set-up fixes (all off unless asked for)."""
    from search_player import MixSettings

    fix = bool(getattr(args, "veto_fix", False))
    return MixSettings(rule=args.mix_rule, guard=bool(getattr(args, "guard", False)), trust=float(getattr(args, "trust", 0.0) or 0.0),
                       no_third=fix, tera_own=fix, sure_attack=float(getattr(args, "sure_attack", 0.7)) if fix else 0.0,
                       setup_defer=bool(getattr(args, "setup_defer", False)),
                       setup_switch_sure=float(getattr(args, "setup_switch_sure", 0.0) or 0.0),
                       setup_switch_trust=float(getattr(args, "setup_switch_trust", 0.0) or 0.0),
                       setup_net_trust=float(getattr(args, "setup_net_trust", 0.0) or 0.0),
                       setup_blend=float(getattr(args, "setup_blend", 0.0) or 0.0),
                       setup_keep=float(getattr(args, "setup_keep", 0.0) or 0.0),
                       guard_status=float(getattr(args, "guard_sucker", 0.0) or 0.0),
                       setup_counter=bool(getattr(args, "setup_counter", False)),
                       setup_other_side=bool(getattr(args, "setup_other_side", False)))


def think_longer(args: Any) -> Any:
    """search_player.ThinkLonger from --think-longer MS and its settings, or None when it is 0 (the default)."""
    from search_player import ThinkLonger

    ms = int(getattr(args, "think_longer", 0) or 0)
    if ms <= 0:
        return None
    return ThinkLonger(ms=ms, read=float(getattr(args, "think_longer_read", 0.4)), boost=int(getattr(args, "think_longer_boost", 1)),
                       threads=max(1, int(getattr(args, "think_longer_threads", 2))))


def setup_fix_text(args: Any) -> str:
    """", veto fix on, set-up turns: the search's best, think longer 2000 ms" for the start line ("" when all are off)."""
    out = []
    if getattr(args, "veto_fix", False):
        out.append("veto fix on (sure attack %.2f)" % float(getattr(args, "sure_attack", 0.7)))
    if getattr(args, "setup_defer", False):
        sure = float(getattr(args, "setup_switch_sure", 0.0) or 0.0)
        out.append("set-up turns: the search's best" + (" unless the network switches at %.2f" % sure if sure > 0 else ""))
    if int(getattr(args, "think_longer", 0) or 0) > 0:
        out.append("think longer on set-up turns %d ms" % int(args.think_longer))
    for flag, text in (("setup_switch_trust", "set-up turns: the network's switch at %.2f is not vetoed"),
                       ("setup_net_trust", "set-up turns: the network's action at %.2f is played"),
                       ("setup_blend", "set-up turns: blend, search weight %.2f"),
                       ("setup_keep", "set-up turns: veto keep %.2f"),
                       ("guard_sucker", "guard: no Sucker Punch into a status read of %.2f")):
        value = float(getattr(args, flag, 0.0) or 0.0)
        if value > 0:
            out.append(text % value)
    if getattr(args, "setup_counter", False):
        out.append("set-up turns: use the counter")
    if getattr(args, "setup_other_side", False):
        out.append("don't hit the boosted defence")
    return "".join(", " + text for text in out)


def experiment_tag(args: Any) -> Optional[str]:
    """An --arms run's tag in results.jsonl ("experiment"): --experiment, else its arms joined by commas (None without arms).
    With --arm-settings: --experiment, else "settings:" and the settings' names joined by commas."""
    if getattr(args, "arm_settings", None):
        return str(getattr(args, "experiment", None) or "settings:" + ",".join(s.name for s in args.arm_settings))
    if not getattr(args, "arms", None):
        return None
    return str(getattr(args, "experiment", None) or ",".join(args.arms))


def deep_budget(args: Any) -> Any:
    """search_player.SearchBudget of the deep arm from the --deep-* flags (ValueError when its deadline is too short)."""
    from search_player import SearchBudget

    return SearchBudget(search_ms=int(args.deep_search_ms), worlds_x=int(args.deep_worlds), threads=int(args.deep_threads),
                        deadline=float(args.deep_deadline), min_time_left=float(args.deep_min_time_left),
                        max_decision_s=float(args.deep_max_decision_s), workers=int(getattr(args, "deep_workers", 1) or 1))


def lines_deadline(args: Any) -> float:
    """A lines arm's search deadline: --deadline, and what the helper reckons for the lines (a quarter over --lines-ms) with 0.2 s to spare."""
    return round(float(args.deadline) + 1.25 * int(args.lines_ms) / 1000.0 + 0.2, 2)


def arm_budgets(args: Any, modes: List[str]) -> Dict[str, Any]:
    """
    {arm: search_player.SearchBudget} for the arms of this run that have a budget of their own:
    the deep arm (--deep-*), the lines arms (--search-ms, then --lines-ms of lines; the deadline
    holds both) and the model arms that get seconds of their own for the answer (Sol: --sol-decide-s;
    Luna with lines: --luna-timeout). ValueError when a deadline is too short for its search.
    """
    from search_player import DECIDERS, DEEP_ARM, LINES_ARMS, SOL_ARM, SearchBudget

    out: Dict[str, Any] = {}
    if DEEP_ARM in modes:
        out[DEEP_ARM] = deep_budget(args)
    for arm in modes:
        model = DECIDERS.get(arm)
        decide_s = float(args.sol_decide_s) if model == "sol" else (float(args.luna_timeout) + 0.1 if model == "luna" else 0.0)
        slow = 12.0 if model == "sol" else 9.0
        if arm in LINES_ARMS:
            out[arm] = SearchBudget(search_ms=int(args.search_ms), worlds_x=1, threads=1, deadline=lines_deadline(args),
                                    decide_s=decide_s, max_decision_s=slow, lines_ms=int(args.lines_ms), lines_k=int(args.lines_k),
                                    lines_turns=int(args.lines_turns), lines_worlds=int(args.lines_worlds),
                                    lines_threads=int(args.lines_threads))
        elif arm == SOL_ARM:
            out[arm] = SearchBudget(search_ms=int(args.search_ms), worlds_x=1, threads=1, deadline=float(args.deadline),
                                    decide_s=decide_s, max_decision_s=slow)
    return out


def decider_models(args: Any) -> List[str]:
    """The models this run's arms ask ("luna", "sol", in that order); [] when no arm is a model's."""
    from search_player import DECIDERS

    modes = list(args.arms) if getattr(args, "arms", None) else [getattr(args, "player", "net")]
    return [name for name in OPENAI_MODELS + CLAUDE_MODELS if any(DECIDERS.get(mode) == name for mode in modes)]


def search_problems(args: Any, rtt_s: float = 0.0) -> List[str]:
    """
    Why the search settings cannot be played, without starting anything ([] when they can): a
    deadline a search could never meet, too few helpers, the deep arm's budget. make_player stops
    on the first; the launcher test reads them. rtt_s: the round trip to remote helpers (measured
    when they start; 0 here or in WSL), which every deadline has to hold as well.
    """
    from search_player import minimum_deadline, split_mode

    modes = list(args.arms) if args.arms else [args.player]
    problems: List[str] = []
    if not any(split_mode(mode)[0] != "net" for mode in modes):
        return problems
    trip = (f" and the round trip to the search helpers ({1000.0 * rtt_s:.0f} ms)") if rtt_s > 0 else ""
    if args.deadline < minimum_deadline(args.search_ms) + rtt_s:
        problems.append(f"--search-ms {args.search_ms}{trip} needs --deadline {minimum_deadline(args.search_ms) + rtt_s:.2f} or more: "
                        f"a search that cannot end by the deadline is never started")
    if "predict" in modes and args.deadline < minimum_deadline(args.search_ms) + args.payoff_ms / 1000.0 + rtt_s:
        problems.append(f"--search-ms {args.search_ms} with --payoff-ms {args.payoff_ms:.0f}{trip} needs --deadline "
                        f"{minimum_deadline(args.search_ms) + args.payoff_ms / 1000.0 + rtt_s:.2f} or more")
    if args.helpers < args.parallel + 1:
        # Fewer helpers than games and the games queue behind each other in them: at 6
        # games on 4 helpers a third of the decisions were the network's alone, each
        # played just before the deadline (review of 2026-10-06, run s3).
        problems.append(f"--helpers {args.helpers} with --parallel {args.parallel}: a search needs at least one "
                        f"helper more than games at a time (--helpers {args.parallel + 1} or --parallel {args.helpers - 1})")
    if float(getattr(args, "trust", 0.0) or 0.0) > 0:
        if args.mix_rule != "veto":
            problems.append(f"--trust {args.trust:g} works with the veto rule only (--mix-rule veto, not {args.mix_rule})")
        # search plays the search's own pick (no guard), and a predict game's record would not say it was guarded.
        unguarded = [mode for mode in modes if split_mode(mode)[0] in ("search", "predict")]
        if unguarded:
            problems.append(f"--trust {args.trust:g} guards the mix modes only, not {', '.join(unguarded)}")
    if "luna" in modes:
        # Luna is asked once the search has answered: the deadline has to hold both.
        least = minimum_deadline(args.search_ms) + rtt_s + LUNA_LEAST_S
        if args.deadline < least:
            problems.append(f"--deadline {args.deadline} leaves Luna no time after a search of {args.search_ms} ms{trip}: "
                            f"the Luna arm needs --deadline {least:.2f} or more")
    if "mix-deep" in modes:
        try:
            budget = deep_budget(args)
            if budget.deadline >= 10.0:
                problems.append(f"--deep-deadline {budget.deadline}: a decision of 10 s or more costs two ticks of the battle timer "
                                f"(5 s each) where a turn gives 10 s back; keep it under 10 s")
            if budget.deadline >= budget.max_decision_s:
                problems.append(f"--deep-deadline {budget.deadline:g} is not under --deep-max-decision-s {budget.max_decision_s:g}: a "
                                f"decision that runs to its deadline would send the game back to today's budget")
            least = minimum_deadline(budget.search_ms, budget.worlds_x, budget.workers) + rtt_s
            if rtt_s > 0 and budget.deadline < least:
                problems.append(f"the deep arm: a search of {budget.search_ms} ms on {budget.worlds_x}x the worlds{trip} needs "
                                f"--deep-deadline {least:.2f} or more")
        except ValueError as exc:
            problems.append(f"the deep arm: {exc} (--deep-search-ms, --deep-worlds, --deep-deadline)")
    from search_player import LINES_ARMS, SOL_ARM

    if any(mode in LINES_ARMS or mode == SOL_ARM for mode in modes):
        try:
            for arm, budget in arm_budgets(args, [m for m in modes if m in LINES_ARMS or m == SOL_ARM]).items():
                whole = float(budget.deadline) + float(budget.decide_s)
                if whole >= 10.0:
                    problems.append(f"the arm {arm}: a decision may take {whole:.1f} s (the search{' and its lines' if budget.lines_ms else ''} "
                                    f"within {budget.deadline:g} s, then {budget.decide_s:g} s for the model); 10 s or more costs two "
                                    f"ticks of the battle timer where a turn gives 10 s back (--deadline, --lines-ms, --sol-decide-s)")
                least = minimum_deadline(budget.search_ms, budget.worlds_x, budget.workers) + budget.lines_s() + rtt_s
                if rtt_s > 0 and budget.deadline < least:
                    problems.append(f"the arm {arm}: its search{trip} needs a deadline of {least:.2f} s or more "
                                    f"(it has {budget.deadline:g}: --deadline, --lines-ms)")
        except ValueError as exc:
            problems.append(f"the lines and Sol arms: {exc} (--search-ms, --deadline, --lines-ms)")
    return problems


LUNA_LEAST_S = 0.4        # of the deadline, at least this much for Luna's answer after the search's


# ---------------------------------------------------------------------------------------------
# Settings per game (--arm-settings; search_player.GameSetting and SettingPicker)
# ---------------------------------------------------------------------------------------------

# The flags a setting gives the games it is played in: the mode, today's search (--search-ms within
# --deadline: what a mode without a budget of its own searches, and what one with a budget falls back
# to), the deep arm's budget and the guard. A game has them as ladder.py started with that setting's
# arguments alone would: a flag the setting leaves out at ladder.py's default, never this run's.
SETTING_FLAGS = ("player", "search_ms", "deadline", "deep_search_ms", "deep_worlds", "deep_threads", "deep_workers",
                 "deep_deadline", "deep_min_time_left", "deep_max_decision_s", "trust")
# Not taken from a setting: the helpers are the run's (--parallel + 1 unless --helpers is given); a
# setting's (the app's: 3) is for the games it plays at once by itself.
SETTING_IGNORED = ("helpers",)
SETTING_NEVER = ("arms", "arm_settings_file")       # a setting is one game's settings, not a list of them
SETTING_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,39}")


class ArmSetting:
    """
    One setting of --arm-settings. name; argv: its arguments as in the file; args: this run's
    arguments with the setting's own SETTING_FLAGS (what search_problems and arm_budgets read for
    it); game: the search_player.GameSetting its games are given.
    """

    def __init__(self, name: str, argv: List[str], args: argparse.Namespace, game: Any = None):
        self.name, self.argv, self.args, self.game = name, list(argv), args, game

    def text(self) -> str:
        """'mix-trust, 900 ms within 2.7 s, trust 0.03' for the start line."""
        from search_player import ARM_FIXES, split_mode

        g = self.game
        own = g.own_budget()
        if own is not None:
            how = (f"{own.search_ms} ms per world on {own.worlds_x}x the worlds within {own.deadline:g} s "
                   f"(after a fall back {g.search_ms} ms within {g.deadline:g} s)")
        elif split_mode(g.mode)[0] == "net":
            how = "no search"
        else:
            how = f"{g.search_ms} ms within {g.deadline:g} s"
        trust = float(ARM_FIXES.get(g.mode, ({}, False))[0].get("trust", g.trust))     # an ARM_FIXES arm's own trust
        return f"{g.mode}, {how}" + (f", trust {trust:g}" if trust > 0 else "")


def flag_of(dest: str) -> str:
    """'--mix-rule' for the parser's 'mix_rule'."""
    for action in build_parser()._actions:
        if action.dest == dest and action.option_strings:
            return action.option_strings[0]
    return "--" + dest.replace("_", "-")


def given_flags(argv: List[str]) -> Dict[str, Any]:
    """The flags a command line gives itself, {dest: value} before tidy_args, whatever the value; ValueError for one ladder.py has not."""
    ap = build_parser(_SettingParser)
    for action in ap._actions:
        action.default = argparse.SUPPRESS
    return vars(ap.parse_args(list(argv)))


def load_arm_settings(args: argparse.Namespace, argv: Optional[List[str]] = None, data: Any = None) -> List[ArmSetting]:
    """
    The settings of --arm-settings (args.arm_settings_file, or data: the file's list as read), checked:
    a list of {"name", "args"} with different names; each setting's arguments read by ladder.py's own
    parser and tidy_args; its SETTING_FLAGS its own (not on the command line); every other flag the
    run's (a setting that gives one gives this run's value, and every setting the same, a setting that
    leaves it out counted at its default); every setting's search checked as search_problems checks
    the flags. --helpers becomes --parallel + 1 unless it is on the command line. SystemExit with the
    reason otherwise. argv: this run's command line (None: sys.argv[1:]).
    """
    where = f"--arm-settings {args.arm_settings_file}" if getattr(args, "arm_settings_file", None) else "--arm-settings"
    shape = 'a JSON list of settings, each {"name": "2s-deeper", "args": ["--player", "mix-trust", "--search-ms", "900", ...]}'
    if data is None:
        try:
            with open(args.arm_settings_file, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"{where}: cannot be read ({exc})")
    if not isinstance(data, list) or not data:
        raise SystemExit(f"{where}: not {shape}")
    if args.arms:
        raise SystemExit("--arm-settings and --arms cannot go together: every setting gives its games their mode")
    run_given = given_flags(sys.argv[1:] if argv is None else argv)
    per_game = [flag_of(d) for d in SETTING_FLAGS if d in run_given]
    if per_game:
        raise SystemExit(f"--arm-settings: {', '.join(per_game)} {'is' if len(per_game) == 1 else 'are'} given per game by the "
                         f"settings: leave {'it' if len(per_game) == 1 else 'them'} off the command line")
    from search_player import DECIDERS, SOLO_ARMS, GameSetting

    read: List[tuple] = []
    for i, item in enumerate(data, 1):
        if not (isinstance(item, dict) and isinstance(item.get("name"), str) and isinstance(item.get("args"), list)
                and all(isinstance(a, str) for a in item["args"])):
            raise SystemExit(f"{where}: setting {i} is not a {{\"name\": ..., \"args\": [...]}} ({shape})")
        name = item["name"].strip()
        if not SETTING_NAME.fullmatch(name):
            raise SystemExit(f"{where}: setting {i}: a name is up to 40 letters, digits and . _ + -, not {item['name']!r}")
        if any(name == row[0] for row in read):
            raise SystemExit(f"{where}: two settings are named {name}")
        tokens = list(item["args"])
        try:
            own = given_flags(tokens)
            full = tidy_args(build_parser(_SettingParser).parse_args(tokens))
        except (ValueError, SystemExit) as exc:
            raise SystemExit(f"{where}: setting {name}: {exc}")
        never = [flag_of(d) for d in SETTING_NEVER if d in own]
        if never:
            raise SystemExit(f"{where}: setting {name}: {', '.join(never)} cannot be in a setting")
        if full.player == "predict" or full.player in DECIDERS or full.player in SOLO_ARMS:
            raise SystemExit(f"{where}: setting {name}: --player {full.player} cannot be a setting (predict and the model arms "
                             f"have settings made once for the whole run): play it with --player or --arms")
        read.append((name, tokens, own, full))
    # Every other flag is the run's, the same for every game.
    for dest in sorted({d for row in read for d in row[2]} - set(SETTING_FLAGS) - set(SETTING_IGNORED)):
        values = [(row[0], getattr(row[3], dest)) for row in read]
        flag = flag_of(dest)
        if any(v != values[0][1] for _n, v in values):
            raise SystemExit(f"{where}: the settings disagree on {flag} (" + "; ".join(f"{n}: {v}" for n, v in values)
                             + "): it is one for the whole run, not per game, so every setting has to give it the same")
        if getattr(args, dest) != values[0][1]:
            raise SystemExit(f"{where}: the settings give {flag} {values[0][1]} and this run {getattr(args, dest)}: it is one for "
                             f"the whole run, not per game; give it on the command line too, or take it out of the settings")
    if "helpers" not in run_given:
        args.helpers = max(1, min(MAX_HELPERS, args.parallel + 1))
    out: List[ArmSetting] = []
    for name, tokens, _own, full in read:
        ns = copy.copy(args)
        for dest in SETTING_FLAGS:
            setattr(ns, dest, getattr(full, dest))
        ns.arm_settings = None
        out.append(ArmSetting(name, tokens, ns))
    problems = settings_problems(out)
    if problems:
        raise SystemExit(f"{where}: {problems[0]}")
    for s in out:
        try:
            budget = arm_budgets(s.args, [s.args.player]).get(s.args.player)
            s.game = GameSetting(name=s.name, mode=s.args.player, search_ms=int(s.args.search_ms),
                                 deadline=float(s.args.deadline), trust=float(s.args.trust), budget=budget)
        except ValueError as exc:
            raise SystemExit(f"{where}: setting {s.name}: {exc}")
    return out


def settings_problems(settings: Optional[List[ArmSetting]], rtt_s: float = 0.0) -> List[str]:
    """search_problems of every setting (its own flags, this run's helpers and the rest), each problem once, a setting's with its name."""
    out: List[str] = []
    for s in settings or []:
        for problem in search_problems(s.args, rtt_s):
            text = problem if problem.startswith("--helpers ") else f"setting {s.name}: {problem}"
            if text not in out:
                out.append(text)
    return out


def run_problems(args: Any, rtt_s: float = 0.0) -> List[str]:
    """search_problems of this run: of its flags, or with --arm-settings of every setting."""
    if getattr(args, "arm_settings", None):
        return settings_problems(args.arm_settings, rtt_s)
    return search_problems(args, rtt_s)


def run_modes(args: Any) -> List[str]:
    """The modes of this run's games: --arms, the settings' (each once, in the file's order), else --player."""
    if getattr(args, "arm_settings", None):
        modes: List[str] = []
        for s in args.arm_settings:
            if s.game.mode not in modes:
                modes.append(s.game.mode)
        return modes
    return list(args.arms) if args.arms else [args.player]


def run_deadline(args: Any) -> float:
    """--deadline, or with --arm-settings the longest of the settings' (the search helpers' patience, as each setting alone has it)."""
    if getattr(args, "arm_settings", None):
        return max(float(s.args.deadline) for s in args.arm_settings)
    return float(args.deadline)


def setting_counts(games: List[Dict[str, Any]], checkpoint: str, experiment: Optional[str]) -> Dict[str, int]:
    """
    Games per setting among the recorded games of one checkpoint and experiment (--arm-settings) that
    have a result: where a new run's balance starts. A game whose result was not seen counts for
    nothing, so a run started again plays its setting once more.
    """
    counts: Dict[str, int] = {}
    for d in games:
        if d.get("experiment") != experiment or d.get("checkpoint") != checkpoint:
            continue
        if d.get("setting") and d.get("won") in (True, False):
            counts[d["setting"]] = counts.get(d["setting"], 0) + 1
    return counts


def wants_luna(args: Any) -> bool:
    """Whether this run has an arm a model decides (luna, sol, luna-lines, sol-lines: --player, or among --arms)."""
    return bool(decider_models(args))


class SpendCap:
    """
    The run's spending on the models' API (--max-model-usd): every decision's own cost is added
    (the record's "luna" "cost_usd": the model's prices on the API's token counts), and once the
    sum reaches the cap every model is turned off for the rest of the run, said once. The games in
    progress are finished by mix and no new game is started, as when the API refuses the key.
    cap 0: no cap. Luna costs about $0.0002 a decision and Sol about $0.005: $20 is some 4,000
    decisions of Sol's, about 120 games.
    """

    def __init__(self, cap_usd: float):
        self.cap = max(0.0, float(cap_usd or 0.0))
        self.spent = 0.0
        self.hit = False

    def add(self, record: Dict[str, Any], deciders: Dict[str, Any], say: Callable[[str], None]) -> None:
        model = record.get("luna")
        cost = model.get("cost_usd") if isinstance(model, dict) else None
        if not isinstance(cost, (int, float)) or isinstance(cost, bool) or cost <= 0:
            return
        self.spent += float(cost)
        if self.cap <= 0 or self.hit or self.spent < self.cap:
            return
        self.hit = True
        why = f"this run's spending cap of ${self.cap:.2f} was reached (${self.spent:.2f} spent; --max-model-usd)"
        for who in deciders.values():
            if not who.client.disabled:
                who.client.off_reason = "spend_cap"
                who.client.disabled = why
        say(f"{' and '.join(sorted(who.label for who in deciders.values())) or 'The model'} OFF for the rest of this run: {why}. "
            f"Today's bot (mix) plays the games in progress, and no new games are started.")


def model_label(who: Any) -> str:
    """ "Luna" or "Sol" for a decision's or a game's model ("luna", "sol", the API's model id, or nothing: Luna, the first one). """
    text = str(who or "").lower()
    if text.endswith("-solo"):              # an arm's name: "luna-solo" has "sol" in it, and is Luna
        text = text[:-len("-solo")]
    for name in CLAUDE_MODELS:
        if name in text:
            return CLAUDE_LABELS[name]
    return "Sol" if "sol" in text else "Luna"


def luna_text(rows: List[Dict[str, Any]]) -> str:
    """
    ", Luna: asked 412 of 430 decisions with a choice, answered 398, ..." over the result lines of
    games of the Luna arm ("" for other arms): luna_player.counts_text on their sums.
    """
    rows = [d for d in rows if "luna_decisions" in d]
    if not rows:
        return ""
    try:
        import luna_player as lp

        total = lp.new_counts()
        for d in rows:
            for key in ("luna_decisions", "luna_asked", "luna_answered", "luna_played", "luna_overruled", "luna_agreed_with_mix",
                        "luna_changed", "luna_tokens", "luna_output_tokens"):
                total[key] += int(d.get(key) or 0)
            total["luna_cost"] += float(d.get("luna_cost_usd") or 0.0)
            for why, k in (d.get("luna_fallbacks") or {}).items():
                total["luna_fallbacks"][why] = total["luna_fallbacks"].get(why, 0) + int(k or 0)
            if d.get("luna_latency_ms_mean") is not None and d.get("luna_asked"):
                # the mean over the game's answers and errors that came back, weighted by its questions
                total["luna_latency_n"] += int(d["luna_asked"])
                total["luna_latency_ms"] += float(d["luna_latency_ms_mean"]) * int(d["luna_asked"])
                total["luna_latency_max"] = max(total["luna_latency_max"], float(d.get("luna_latency_ms_max") or 0.0))
        models = {model_label(d.get("luna_model")) for d in rows}
        return ", " + lp.counts_text(total, "the model" if len(models) > 1 else models.pop())
    except Exception:
        return ""


def luna_settings_text(args: Any) -> str:
    """One line about the Luna arm for the run's header (never the key)."""
    url = str(getattr(args, "luna_url", "") or "")
    where = "OpenAI" if "api.openai.com" in url else f"a TEST server ({url})"
    return (f"Luna: model {args.luna_model} at {where}, an answer within {args.luna_timeout:g} s and never past the "
            f"{args.deadline:g} s deadline, blunder guard {args.luna_guard}, briefs "
            f"{'kept in the battle files' if args.luna_save_briefs else 'not kept'}")


def sol_settings_text(args: Any, client: Any = None) -> str:
    """One line about the Sol arms for the run's header (never the key)."""
    url = str(getattr(args, "sol_url", "") or "")
    where = "OpenAI" if "api.openai.com" in url else f"a TEST server ({url})"
    effort = getattr(client, "effort", None) or args.sol_effort
    return (f"Sol: model {args.sol_model} at {where} (the Responses API, reasoning effort {effort}), an answer within "
            f"{args.sol_timeout:g} s, asked once the search has answered and given up to {args.sol_decide_s:g} s, "
            f"blunder guard {args.luna_guard}, briefs {'kept in the battle files' if args.luna_save_briefs else 'not kept'}"
            + (f"; about half a cent a decision, off for the run at ${args.max_model_usd:g}" if getattr(args, "max_model_usd", 0) else ""))


def claude_settings_text(args: Any, name: str, client: Any = None) -> str:
    """One line about a Claude arm for the run's header (never the key)."""
    url = str(getattr(args, "claude_url", "") or "")
    where = "Anthropic" if "api.anthropic.com" in url else f"a TEST server ({url})"
    if client is None:
        return f"{CLAUDE_LABELS[name]}: no key was given, so its games are played by the fallback rule"
    effort = getattr(client, "effort", "") or "none (the model does not take one)"
    return (f"{client.label}: model {client.model} at {where} (the Messages API, effort {effort}), aimed at under 10 s a move "
            f"and waited for up to {float(args.claude_timeout):g} s, then the fallback rule plays; briefs "
            f"{'kept in the battle files' if args.luna_save_briefs else 'not kept'}"
            + (f"; off for the run at ${args.max_model_usd:g}" if getattr(args, "max_model_usd", 0) else ""))


def lines_settings_text(args: Any) -> str:
    """One line about the lines arms for the run's header."""
    return (f"read ahead: after the search, its {args.lines_k} favourite actions are each played forward {args.lines_turns} turns in the "
            f"{args.lines_worlds} likeliest sampled sets ({args.lines_ms} ms in all on {args.lines_threads} threads; the search and the "
            f"lines within {lines_deadline(args):g} s); mix-lines changes mix's pick for a line that ends {args.lines_margin:g} or more "
            f"better and in no set worse")


def arm_counts(games: List[Dict[str, Any]], checkpoint: str, experiment: Optional[str] = None) -> Dict[str, int]:
    """
    Games per arm among the recorded games of one checkpoint: where a new run's balance starts.
    experiment (an --arms run: its arms joined by commas): only the games recorded with that same
    "experiment", so the games an ordinary run played ("mix" alone, hundreds of them) do not count
    for an A/B's control and the new arm is not given every game until it catches up.
    """
    counts: Dict[str, int] = {}
    for d in games:
        if experiment is not None and d.get("experiment") != experiment:
            continue
        if d.get("arm") and d.get("checkpoint") == checkpoint:
            counts[d["arm"]] = counts.get(d["arm"], 0) + 1
    return counts


def arm_fields(summary: Optional[Dict[str, Any]], role: Optional[str], search_ms: Optional[int],
               mix_rule: Optional[str]) -> Dict[str, Any]:
    """
    What a game's line in results.jsonl says about its arm: everything search_player's
    game_summary() counted ("arm", "decisions", "searched", "fallbacks", "overruled", "tera",
    "tera_turn" and the rest), the side it played, the search time when a search was running
    (None otherwise) and the rule behind a plain "mix".
    """
    fields = dict(summary or {})
    fields["role"] = role
    fields["search_ms"] = search_ms
    budget = fields.get("budget")
    if search_ms is not None and isinstance(budget, dict) and budget.get("ms"):
        fields["search_ms"] = int(budget["ms"])      # an arm with its own budget (the deep arm): its time per world
    fields["mix_rule"] = mix_rule if str(fields.get("arm")) in ("mix", "predict", "mix-deep", "luna", "sol", "mix-lines", "luna-lines",
                                                                    "sol-lines") else None
    return fields


def arms_text(games: List[Dict[str, Any]], key: Optional[Callable[[Dict[str, Any]], str]] = None) -> str:
    """One line per arm (ladder_graph.arm_of: a guarded game of another arm is apart; or per key(game), such as its setting):
    its games, its record, and how the search did in them."""
    from ladder_graph import arm_of

    arms: Dict[str, List[Dict[str, Any]]] = {}
    for d in games:
        if d.get("arm"):
            arms.setdefault((key or arm_of)(d), []).append(d)
    lines = []
    for arm in sorted(arms):
        rows = arms[arm]
        wins = sum(1 for d in rows if d.get("won"))
        decisions = sum(d.get("decisions") or 0 for d in rows)
        searched = sum(d.get("searched") or 0 for d in rows)
        fallbacks = sum(d.get("fallbacks") or 0 for d in rows)
        overruled = sum(d.get("overruled") or 0 for d in rows)
        tera = [d for d in rows if d.get("tera")]
        text = f"  {arm:<14} {len(rows):3d} games, {wins}-{len(rows) - wins} ({100.0 * wins / len(rows):.0f}%)"
        text += f", Tera in {len(tera)}"
        if searched or fallbacks:
            text += (f", search answered {searched} of {decisions} decisions, {fallbacks} fell back to the network, "
                     f"the network was overruled in {overruled}")
        text += predictions_text(rows)
        fired = sum(d.get("guard_fired") or 0 for d in rows)
        if fired:
            text += f", guard fired {fired} times and changed the move {sum(d.get('guard_changed') or 0 for d in rows)} times"
        text += counter_rules_text(rows)
        text += luna_text(rows)
        lines.append(text)
    return "\n".join(lines)


def counter_rules_text(rows: List[Dict[str, Any]]) -> str:
    """
    ", use the counter: looked 12, played 3, switched 1, kept for KO 2, mix had one 4; boosted defence: hit 5, swapped 2" over
    the result lines of an arm with round 3's counter rules ("" for an arm without them).
    """
    from search_player import COUNTER_COUNTS

    if not any(name in d for d in rows for name in COUNTER_COUNTS):
        return ""
    n = {name: sum(int(d.get(name) or 0) for d in rows) for name in COUNTER_COUNTS}
    return (f", use the counter: looked {n['counter_looked']}, played {n['counter_played']}, switched {n['counter_switched']}, "
            f"kept for KO {n['counter_ko_kept']}, mix had one {n['counter_had']}; boosted defence: hit {n['other_side_looked']}, "
            f"swapped {n['other_side']}")


# ---------------------------------------------------------------------------
# The login
# ---------------------------------------------------------------------------

LOGIN_REQUEST_TIMEOUT = 10.0   # s for the login server's answer (poke-env's own)
LOGIN_WATCH_S = 60.0           # s after the name is sent during which Showdown's popups and raw lines about it are told
# A |raw| line during the login is told when its text has one of these as a word: lock, proxy,
# datacenter, #hostfilter, name, password (not "name" in "tournament", not an HTML attribute).
LOGIN_WORDS_RE = re.compile(r"\b(?:un)?lock(?:ed|s)?\b|\bprox(?:y|ies)\b|\bdata ?cent(?:er|re)s?\b|#hostfilter|\b(?:user)?names?\b"
                            r"|\bpasswords?\b", re.I)
SECRET_TEXTS: List[str] = []   # this run's login assertions (in memory only): never printed, masked in the console log


def one_line(text: Any, limit: int = 240) -> str:
    """Showdown's text (HTML, popup line breaks) as one plain line of at most `limit` characters."""
    text = html.unescape(re.sub(r"<[^>]*>", " ", str(text or "").replace("||", " / ")))
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def masked(text: str, client: Any = None) -> str:
    """The text with the password (the client's) and every login assertion of this run put as ***."""
    password = getattr(getattr(client, "account_configuration", None), "password", None)
    for secret in [password] + SECRET_TEXTS:
        if secret and len(secret) >= 3 and secret in text:
            text = text.replace(secret, "***")
    return text


def read_login_answer(status: int, text: str) -> tuple:
    """
    The login server's answer to act=login, as (assertion, None, None) when it can be sent to
    Showdown, else (None, kind, line): kind "refused" when the login server says no (its JSON has
    actionsuccess false, an actionerror, or an assertion that starts with ";;" followed by the
    reason), "failed" when the answer is not one at all (a web page, no assertion). The line is
    one plain line with the HTTP status and the reason, never the assertion.
    """
    body = (text or "").strip()
    if body.startswith("]"):
        body = body[1:]
    try:
        data = json.loads(body)
    except ValueError:
        data = None
    if not isinstance(data, dict):
        if data is None and re.search(r"<\s*(!doctype|html|head|body|title|div|p)\b", body[:2000], re.I):
            return None, "failed", f"the login server answered with a web page, status {status} (likely a block on this address)"
        if data is None:
            return None, "failed", f"the login server's answer is not JSON ({len(body)} characters), status {status}"
        return None, "failed", f"the login server's answer has no assertion, status {status}"
    assertion = data.get("assertion")
    reason: Optional[str] = None
    if isinstance(assertion, str) and assertion.startswith(";;"):
        reason = assertion[2:]
    elif data.get("actionerror"):
        reason = str(data.get("actionerror"))
    elif data.get("actionsuccess") is False:
        reason = str(data.get("error") or "") or "no reason given (actionsuccess false)"
    if reason is not None:
        return None, "refused", f"the login server refused the login (status {status}): {one_line(reason, 200) or 'no reason given'}"
    if assertion == ";":
        return None, "refused", (f"the login server refused the login (status {status}): the name is registered "
                                 f"and it did not take the password")
    if not isinstance(assertion, str) or not assertion.strip() or assertion.startswith(";"):
        return None, "failed", f"the login server's answer has no assertion, status {status}"
    return assertion, None, None


class LoginState:
    """
    How the logins of one client went (the first one and each one after a lost connection), for
    the waits in drive() and reconnect(). Only lines and times: never the password or an assertion.
    """

    def __init__(self) -> None:
        self.attempts = 0
        self.generation = 0
        self.ever_confirmed = False
        self.challstr_at: Optional[float] = None   # this attempt: the server's challenge came
        self.sent_at: Optional[float] = None       # the name (/trn) went to Showdown
        self.confirmed_at: Optional[float] = None  # Showdown's |updateuser| with the name
        self.failed: Optional[str] = None          # this attempt failed (no answer, a web page): the line
        self.failed_at: Optional[float] = None
        self.refused: Optional[str] = None         # refused with a reason: the line; no login is tried again
        self.refused_kind: Optional[str] = None    # "password" (the login server) or "name" (Showdown's |nametaken|)
        self.identity: Optional[str] = None        # the name of the last |updateuser|, as Showdown wrote it
        self.watch_until = 0.0
        self.told: Dict[str, float] = {}           # line -> when it was printed (the same line once in 30 s)

    def new_attempt(self) -> int:
        self.attempts += 1
        self.generation += 1
        self.challstr_at = time.time()
        self.sent_at = self.confirmed_at = self.failed = self.failed_at = None
        self.watch_until = time.time() + LOGIN_REQUEST_TIMEOUT + LOGIN_WATCH_S
        return self.generation

    def fail(self, line: str) -> None:
        self.failed, self.failed_at = line, time.time()

    def refuse(self, line: str, kind: str) -> None:
        if self.refused is None:
            self.refused, self.refused_kind = line, kind


def login_state(client: Any) -> LoginState:
    state = getattr(client, "_ladder_login", None)
    if state is None:
        state = LoginState()
        client._ladder_login = state
    return state


def _post_login(url: str, data: Dict[str, str], timeout: float) -> tuple:
    import requests

    answer = requests.post(url, data=data, timeout=timeout)
    return answer.status_code, answer.text


def _say_login(client: Any, line: str) -> None:
    text = f"{hms()} {masked(line, client)}"
    try:
        print(text, flush=True)
    except UnicodeEncodeError:          # a window or a pipe that cannot show the lock mark, say
        print(text.encode("ascii", "replace").decode("ascii"), flush=True)


async def log_in_checked(client: Any, message: str) -> None:
    """
    Every login of the bot, on the server's |challstr| (at the start and after every lost
    connection): the login server's answer is read before anything is sent to Showdown, and what
    went wrong is told in ONE line, with nothing secret in it.

    The request is poke-env's (act=login, the name, the password, the challenge; none without a
    password), asked from a thread so the connection's pings go on meanwhile. Then:
      - a refusal with a reason (a JSON answer with actionsuccess false, an actionerror, or an
        assertion ";;" followed by the reason, "Wrong password." say): that line, and no login
        is tried again with this password (LoginState.refused; drive() and reconnect() stop);
      - a web page (a Cloudflare or proxy page), no answer in LOGIN_REQUEST_TIMEOUT s, an answer
        without an assertion: that line, this attempt has failed (LoginState.failed);
      - otherwise the name goes to Showdown (/trn, straight to this connection's websocket: never
        through a logger), and its |updateuser| with the name confirms it (note_updateuser).
    Not poke-env's own log_in: it sends whatever assertion it got, and then asserts the login
    within 5 s in a task nobody waits for (a traceback, and the login never confirmed).
    """
    state = login_state(client)
    generation = state.new_attempt()
    if state.refused:
        _say_login(client, f"login not tried again: {state.refused}")
        return
    websocket = getattr(client, "websocket", None)
    account = client.account_configuration
    password = account.password
    again = " (the login is tried again on the next connection)" if state.ever_confirmed else ""
    try:
        if password:
            parts = message.split("\n", 1)[0].split("|")
            if len(parts) < 4:
                state.fail("the server's challenge (|challstr|) has no key")
                _say_login(client, f"login failed: {state.failed}{again}")
                return
            data = {"act": "login", "name": account.username, "pass": password, "challstr": parts[2] + "%7C" + parts[3]}
            loop = asyncio.get_running_loop()
            status, text = await loop.run_in_executor(None, _post_login, client.server_configuration.authentication_url,
                                                      data, LOGIN_REQUEST_TIMEOUT)
            del data
            assertion, kind, line = read_login_answer(status, text)
            del text
            if kind == "refused":
                state.refuse(line, "password")
                _say_login(client, f"login failed: {line}")
                return
            if kind is not None:
                state.fail(line)
                _say_login(client, f"login failed: {line}{again}")
                return
            if assertion not in SECRET_TEXTS:
                SECRET_TEXTS.append(assertion)
                del SECRET_TEXTS[:-8]
            how = f"the login server accepted it (status {status}), "
        else:
            assertion, how = "", "no password (a local server), "
        if generation != state.generation:
            return                     # a newer connection's login has begun: this one's answer is not sent
        if websocket is None:
            raise ConnectionError("no connection to send the name on")
        await websocket.send(f"|/trn {client.username},0,{assertion}")
        del assertion
        state.sent_at = time.time()
        state.watch_until = state.sent_at + LOGIN_WATCH_S
        _say_login(client, f"login: {how}name {client.username} sent to Showdown, waiting for its confirmation")
    except asyncio.CancelledError as exc:
        client.logger.critical("CancelledError intercepted: %s", exc)     # as poke-env's own handler does
    except Exception as exc:     # noqa: BLE001 - every failure of the login request: one line, not a traceback
        text = masked(" ".join(str(exc).split()), client)
        state.fail(f"login request failed: {type(exc).__name__}" + (f": {text[:160]}" if text else ""))
        _say_login(client, state.failed + again)


def note_updateuser(client: Any, line: str) -> None:
    """
    Showdown's |updateuser|IDENTITY|NAMED|AVATAR|SETTINGS: the login is confirmed when NAMED is 1
    and the name is this account's, whatever mark is in front of it. poke-env takes only " name" or
    " name@!", so a name with a lock mark (‽, as Showdown shows every connection it has locked, its
    proxy filter #hostfilter on a datacenter address among them) was never taken as logged in.
    The mark is told in one line; ladder games are allowed when locked (Showdown's ladders.ts only
    drops nicknames), chat is not, and the bot sends none.
    """
    state = login_state(client)
    parts = line.split("|")
    ident = parts[2] if len(parts) > 2 else ""
    named = parts[3].strip() if len(parts) > 3 else ""
    name = ident[:-2] if ident.endswith("@!") else ident
    mark = name[0] if name and not name[0].isalnum() else ""
    name = name[1:] if mark else name
    state.identity = (mark.strip() + name).strip() or None
    if named != "1" or to_id(name) != to_id(client.username):
        return
    try:
        client.logged_in.set()
    except Exception:
        pass
    if state.confirmed_at is not None:
        return
    state.confirmed_at = time.time()
    state.ever_confirmed = True
    state.watch_until = max(state.watch_until, state.confirmed_at + 10.0)
    after = f" {state.confirmed_at - state.sent_at:.1f} s after the name was sent" if state.sent_at else ""
    _say_login(client, f"logged in as {name}: Showdown confirmed it{after}")
    if mark == "‽":
        _say_login(client, f"note: Showdown marks {name} as locked on this connection (‽ before the name). From a datacenter "
                           f"address like this box's that is most likely its proxy filter (#hostfilter): ladder games are "
                           f"allowed, chat is not (the bot sends none)")
    elif mark == "!":
        _say_login(client, f"note: Showdown marks {name} as muted on this connection (! before the name): ladder games are allowed")


def note_server_word(client: Any, line: str) -> bool:
    """
    A |popup|, |nametaken| or |raw| line from Showdown, told in one line when it bears on the login:
    every |nametaken| (the name refused: no login is tried again while this login is not confirmed)
    and every |popup|, and a |raw| line during the login (LOGIN_WATCH_S after the name went) that
    mentions a lock, a proxy, a datacenter, #hostfilter, the name or the password. True: the line is
    handled here and not passed on (poke-env raises on |nametaken|: a traceback).
    """
    state = login_state(client)
    kind = line.split("|", 2)[1] if line.count("|") >= 2 else ""
    now = time.time()
    if kind == "nametaken":
        parts = line.split("|", 3)
        why = one_line(parts[3] if len(parts) > 3 else "", 200) or "no reason given"
        text = f"Showdown refused the name {parts[2] if len(parts) > 2 and parts[2] else client.username} (nametaken): {why}"
        if state.confirmed_at is None:
            state.refuse(text, "name")
    elif kind == "popup":
        text = f"Showdown says (popup): {one_line(line[len('|popup|'):], 300)}"
    elif kind == "raw":
        said = one_line(line[len("|raw|"):], 300)
        if now > state.watch_until or not LOGIN_WORDS_RE.search(one_line(line[len("|raw|"):], 5000)):
            return False
        text = f"Showdown says during the login: {said}"
    else:
        return False
    if now - state.told.get(text, 0.0) >= 30.0:
        state.told[text] = now
        _say_login(client, text)
    return kind == "nametaken"


async def first_login(client: Any, args: Any, user: str) -> Optional[int]:
    """
    The first login, waited for: None once Showdown has confirmed it, else the exit code after ONE
    line that says why. The connection, the server's challenge and the login server's answer get
    --login-timeout s; once the name is sent, Showdown's confirmation gets --login-wait s (a line
    every 5 s meanwhile). 4: refused with a reason (the login server's, "Wrong password." say, or
    Showdown's |nametaken|), not tried again; 2: not logged in for another reason (no answer, a web
    page, no confirmation).
    """
    state = login_state(client)
    timeout = float(getattr(args, "login_timeout", 45))
    wait = float(getattr(args, "login_wait", 30.0))
    began = time.time()
    told = 0
    while not client.logged_in.is_set():
        now = time.time()
        if state.refused or state.failed:
            line = state.refused or state.failed
            password = state.refused_kind == "password"
            code, kind = (4, "password" if password else "login") if state.refused else (2, "login")
            break
        listening = getattr(client, "_listening_coroutine", None)
        if listening is not None and listening.done():
            line, code, kind, password = "the connection to the server ended before the login was confirmed", 2, "login", False
            break
        if state.sent_at is None:
            if now - began >= timeout:
                stage = ("no connection to the server" if getattr(client, "websocket", None) is None else
                         "the server sent no challenge" if state.challstr_at is None else "the login request did not finish")
                line, code, kind, password = f"not logged in within {timeout:.0f} s ({stage})", 2, "login", False
                break
        else:
            waited = now - state.sent_at
            if waited >= wait:
                line = (f"Showdown did not confirm the login within {wait:.0f} s"
                        + (f" (the last name it gave this connection: {state.identity})" if state.identity else ""))
                code, kind, password = 2, "login", False
                break
            while waited >= told + 5:
                told += 5
                _say_login(client, f"waiting for Showdown to confirm the login ({told} s)")
        await asyncio.sleep(0.1)
    else:
        return None
    _say_login(client, f"LOGIN FAILED for {user}: {line.rstrip('.')}. Nothing was played. Start it again"
                       + (" and type the password carefully." if password else "."))
    try:
        args._login_refused = line.rstrip(".")      # the window app's error line (run: app_event)
    except Exception:
        pass
    write_exit_note(args, kind, line, client)
    try:
        await asyncio.wait_for(client.websocket.close(), timeout=5.0)
    except Exception:
        pass
    return code


def write_exit_note(args: Any, kind: str, text: str, client: Any = None) -> None:
    """--exit-file: one line "KIND: why ladder.py ended" (KIND password, login, connection or done), for run_box_ab.sh."""
    path = getattr(args, "exit_file", None)
    if not path:
        return
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(f"{kind}: {masked(one_line(text, 400), client)}\n")
        os.replace(tmp, path)
    except OSError:
        pass


def predictions_text(rows: List[Dict[str, Any]]) -> str:
    """", predictions: likeliest right 14 of 25 (56%), in the top 3 21 (84%)" over some result lines ("" without predictions)."""
    scored = sum(d.get("pred_scored") or 0 for d in rows)
    if not scored:
        return ""
    top1 = sum(d.get("pred_top1_right") or 0 for d in rows)
    top3 = sum(d.get("pred_top3_right") or 0 for d in rows)
    return f", predictions: likeliest right {top1} of {scored} ({100.0 * top1 / scored:.0f}%), in the top 3 {top3} ({100.0 * top3 / scored:.0f}%)"


class SearchHealth:
    """
    How often the search failed lately. Over the last `window` decisions in which it should have
    answered (not "single" or "revive", where it is not asked), the share that fell back to the
    network because of the search itself: no answer in time ("timeout"), not started in time
    ("expired"), a failed helper ("error") or one that was not up ("down"). A bot whose helpers
    keep failing otherwise plays on as the network alone, quietly, for as many games as it was given.

    With helpers on another machine (settle_s > 0), a failure while the link to them was in
    trouble is the link's, not the search's: helpers unreachable or being replaced, or a line the
    network held up (FoulPlayBridge.link_spans). link_spans() takes such failures out of the
    window (link_failed counts them); the link has a rule of its own in ladder.py
    (--link-down-limit). A decision is judged only settle_s after it was made, because the link's
    trouble is known a few seconds after it began: until then it is not in the window.
    """

    FAILED = ("timeout", "expired", "error", "down")
    NOT_ASKED = ("single", "revive", "off", "rejoined")

    def __init__(self, window: int = 50, settle_s: float = 0.0):
        from collections import deque

        self.window = max(1, int(window))
        self.settle_s = max(0.0, float(settle_s))
        # [asked at, decided at (time.time()), failed]; more than the window while decisions settle.
        self.recent: Any = deque(maxlen=self.window if self.settle_s <= 0 else 4 * self.window + 200)
        self.total = 0
        self.failed = 0
        self.link_failed = 0

    def note(self, record: Dict[str, Any], now: Optional[float] = None) -> None:
        try:
            if str(record.get("mode", "net")) == "net":
                return
            reason = record.get("reason") or ""
            if reason in self.NOT_ASKED:
                return
            bad = reason in self.FAILED
            now = time.time() if now is None else float(now)
            self.recent.append([now - float(record.get("ms") or 0.0) / 1000.0, now, bad])
            self.total += 1
            self.failed += bad
        except Exception:   # bookkeeping must never disturb a game
            pass

    def link_spans(self, spans: List[Any]) -> int:
        """
        The failures whose decision overlapped one of these spans of link trouble ((start, end or
        None while it lasts), time.time()) are the link's: taken out of the window. Returns how many.
        """
        if not spans:
            return 0
        try:
            def link(entry: List[Any]) -> bool:
                return any(entry[0] <= (end if end is not None else float("inf")) and entry[1] >= start for start, end in spans)

            gone = [entry for entry in self.recent if entry[2] and link(entry)]
            if gone:
                ids = {id(entry) for entry in gone}
                kept = [entry for entry in self.recent if id(entry) not in ids]
                self.recent.clear()
                self.recent.extend(kept)
                self.failed -= len(gone)
                self.link_failed += len(gone)
            return len(gone)
        except Exception:
            return 0

    def _judged(self, now: Optional[float] = None) -> List[List[Any]]:
        """The last `window` decisions made at least settle_s ago."""
        entries = list(self.recent)
        if self.settle_s > 0:
            cut = (time.time() if now is None else float(now)) - self.settle_s
            entries = [entry for entry in entries if entry[1] <= cut]
        return entries[-self.window:]

    def share(self, now: Optional[float] = None) -> Optional[float]:
        """The failed share of the last `window` decisions, None until there are that many."""
        entries = self._judged(now)
        if len(entries) < self.window:
            return None
        return sum(1 for entry in entries if entry[2]) / float(len(entries))

    def text(self, now: Optional[float] = None) -> str:
        entries = self._judged(now)
        return (f"the search failed in {sum(1 for entry in entries if entry[2])} of the last {len(entries)} decisions it was asked, "
                f"{self.failed} of {self.total} in this run"
                + (f" (and {self.link_failed} while the link to the helpers was down, not counted)" if self.link_failed else ""))

    def reset(self) -> None:
        """A fresh window (after the helpers were restarted): the rule judges only what comes next. The run's totals stay."""
        self.recent.clear()


THROTTLE_NOTICE = "message-throttle-notice"
# The notice itself, as the server writes it (users.ts, User.chat): a whole line of its own. Only a
# line that starts so is a notice; chat or a private message that contains the words is not.
THROTTLE_LINE = '|raw|<strong class="message-throttle-notice">'
# The server's refusals of an order it does not need (room-battle.ts, RoomBattle.choose; sim side.ts).
LATE_ERRORS = ("[Invalid choice] Sorry, too late", "[Invalid choice] There's nothing to choose", "[Invalid choice] Can't undo")
# Lines of a battle room that say nothing about whether the server has our order: the timer, chat,
# people coming and going, the rating box. Any other line after an order (the turn's events, a new
# request, an error, the end) is the server reacting to it.
NOT_A_REACTION = frozenset(("", "t:", "inactive", "inactiveoff", "raw", "html", "uhtml", "uhtmlchange", "c", "c:", "chat",
                            "j", "J", "join", "l", "L", "leave", "n", "N", "name", "b", "B", "notify", "title", "hidelines",
                            "spectator", "unlink", "tempnotify", "tempnotifyoff"))
TIMER_LINE_RE = re.compile(r"^(.+?) has (\d+) seconds? left")


class OrderGuard:
    """
    Showdown's typing limit, between the player and the websocket (pokemon-showdown server/users.ts,
    User.chat): a user's messages closer together than 600 ms are queued (100 ms for a trusted
    account), and one that finds the queue full is dropped with

        |raw|<strong class="message-throttle-notice">Your message was not sent because you've been typing too quickly.</strong>

    sent to the room of the dropped message (users.ts: connection.sendTo(room, ...), which puts
    ">battle-..." in front for every room but the lobby). Every order goes to its battle room, so a
    dropped order's notice always comes in that battle's room (as in the lost game below); a notice
    without a room is about a message to no room (/search, /utm, /leave, /cancelsearch), never an
    order. A dropped order loses the game on time (battle-gen9randombattle-2694716418, 2026-10-08:
    two games' orders 1.6 s apart, the forced switch dropped, the notice in its room, no answer for
    150 s). So:

    (a) every outgoing message waits until spacing_s after the one before it, across all games
        (one asyncio lock; gate()). Never longer than max_wait_s from when it asked, so an order is
        not held for long whatever queues ahead of it. The wait of every order is counted
        (stats, per game: order_waits, order_wait_ms_total, order_wait_ms_max).
    (b) after a notice in a battle room (a line that starts THROTTLE_LINE: chat cannot fake it),
        that battle's last order is sent again with the request's id ("/choose ...|rqid": the
        server refuses a copy for an older request instead of taking it as the answer to the next
        one), resend_after_s later, then twice and four times that (plus up to jitter_s: the
        server's queue drains one message per 600 ms), at most max_resends times per request.
        Only while it is still the order for the current request (the request id the same, no
        newer order, the game not over) and the server has not reacted to it (NOT_A_REACTION).
        These are checked when the copy is due, and again after it has waited for the spacing
        (a new request can come meanwhile): a copy that went stale is not sent.
    (c) the timer's own word: "|inactive|<us> has N seconds left" in a battle whose order went out
        more than 2 s before and has had no reaction means the server is still waiting for it
        (the tick charges only a player whose order is not in). Each such line sends one copy,
        timer_after_s later, by the same rules, up to max_timer_resends per request: a count of its
        own, so a game is not lost on time while the timer says the server waits (the lines come
        30 s apart, then 5 s apart under 20 s: they cannot flood). This also catches a drop
        without a notice.
    (d) a notice without a room sends no order again. It is counted once, told in one console
        line, and kept in no_room_notice_at: the drive loop asks for a ladder game again when a
        search it sent just before was dropped (instead of waiting two minutes).
    A copy goes straight to the websocket after the guard's own spacing (send: no second gate),
    and is not the battle's order on record (the rejoin's copy stays the order poke-env sent).
    Each copy is one console line and a "throttle_resend" note. A copy the server did not need is
    answered "[Invalid choice] Sorry, too late" (or "There's nothing to choose", "Can't undo"):
    late_errors_to_ignore() lets one such error per copy sent in the last late_window_s pass by,
    whatever the request is by then (poke-env would decide again and send one order too many).
    """

    def __init__(self, send: Any, *, spacing_s: float = 0.3, max_wait_s: float = 1.5, resend_after_s: float = 1.0,
                 jitter_s: float = 0.3, max_resends: int = 3, max_timer_resends: int = 8, timer_after_s: float = 0.1,
                 late_window_s: float = 10.0, me: str = "", current_rqid: Any = None, finished: Any = None,
                 say: Any = None, note: Any = None, clock: Any = time.monotonic, rng: Any = None):
        import random

        self._send = send                       # how a copy goes out: (message, room, rqid), straight to the websocket
        self.spacing_s = max(0.0, float(spacing_s))
        self.max_wait_s = max(0.0, float(max_wait_s))
        self.resend_after_s = max(0.0, float(resend_after_s))
        self.jitter_s = max(0.0, float(jitter_s))
        self.max_resends = max(0, int(max_resends))
        self.max_timer_resends = max(0, int(max_timer_resends))
        self.timer_after_s = max(0.0, float(timer_after_s))
        self.late_window_s = max(0.0, float(late_window_s))
        self.me = to_id(me)
        self.current_rqid = current_rqid or (lambda room: None)
        self.finished = finished or (lambda room: False)
        self.say = say or (lambda text: print(text, flush=True))
        self.note = note or (lambda room, data: None)
        self.clock = clock
        self.rng = rng or random.Random()
        self._lock: Any = None
        self._last = -1e9
        self.pending: Dict[str, Dict[str, Any]] = {}     # battle room -> its last order: rqid, message, sent, reacted, copies
        self._copies: Dict[str, List[float]] = {}        # battle room -> when copies went out (one late error each)
        self.no_room_notice_at: Optional[float] = None   # when the last notice without a room came (the clock's time)
        self.no_room_notice_wall: Optional[float] = None # the same, time.time()
        self.games: Dict[str, Dict[str, float]] = {}     # battle room -> its counts, for its results line
        self.stats: Dict[str, Any] = {"orders": 0, "waits": 0, "wait_ms_total": 0.0, "wait_ms_max": 0.0, "notices": 0,
                                      "notices_without_room": 0, "timer_lines": 0, "resends": 0, "timer_resends": 0,
                                      "late_ignored": 0, "not_resent": {}}
        self._tasks: set = set()

    # -- (a) spacing -------------------------------------------------------------------------------

    @staticmethod
    def is_order(message: Any, room: Any) -> bool:
        return isinstance(room, str) and room.startswith("battle-") and isinstance(message, str) and message.startswith("/choose")

    async def gate(self, message: Any, room: Any) -> float:
        """Waits until spacing_s after the last message that went out (max_wait_s at most from now); the seconds waited."""
        entered = self.clock()
        if self.spacing_s <= 0:
            return 0.0
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            now = self.clock()
            wait = min(self._last + self.spacing_s - now, self.max_wait_s - (now - entered))
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = self.clock()
        waited = max(0.0, self.clock() - entered)
        if self.is_order(message, room):
            self.stats["orders"] += 1
            ms = waited * 1000.0
            if ms >= 1.0:
                self.stats["waits"] += 1
                self.stats["wait_ms_total"] += ms
                self.stats["wait_ms_max"] = max(self.stats["wait_ms_max"], ms)
                game = self._game(room)
                game["order_waits"] += 1
                game["order_wait_ms_total"] += ms
                game["order_wait_ms_max"] = max(game["order_wait_ms_max"], ms)
        return waited

    # -- (b) sending an order again ------------------------------------------------------------------

    def _game(self, room: str) -> Dict[str, float]:
        return self.games.setdefault(room, {"throttle_notices": 0, "throttle_resends": 0, "order_waits": 0,
                                            "order_wait_ms_total": 0.0, "order_wait_ms_max": 0.0})

    def game_counts(self, room: str) -> Dict[str, Any]:
        """The counts of one battle for its results line (and forgets it)."""
        game = self.games.pop(room, None)
        if game is None:
            self._game(room)
            game = self.games.pop(room)
        self.pending.pop(room, None)
        self._copies.pop(room, None)
        return {k: (round(v, 1) if isinstance(v, float) else v) for k, v in game.items()}

    def sent(self, room: str, rqid: Any, message: str) -> None:
        """An order went out (poke-env's, or the rejoin's copy): it is the one to send again, for this request."""
        old = self.pending.get(room)
        same = old is not None and old["rqid"] == rqid
        # "/choose move 1|7" (an order sent again after a rejoin carries the request's id): the order alone.
        self.pending[room] = {"rqid": rqid, "message": str(message).split("|", 1)[0], "sent": self.clock(), "reacted": False,
                              "resends": old["resends"] if same else 0, "timer_resends": old["timer_resends"] if same else 0,
                              "scheduled": False}

    def late_errors_to_ignore(self, room: str, count: int) -> int:
        """
        Of `count` "too late" errors (LATE_ERRORS) that just came in a battle room, how many answer a
        copy the guard sent there in the last late_window_s (one each, oldest first): those are no
        news, and poke-env must not decide again on them.
        """
        now = self.clock()
        copies = [t for t in self._copies.get(room, ()) if now - t <= self.late_window_s]
        take = max(0, min(int(count), len(copies)))
        self._copies[room] = copies[take:]
        self.stats["late_ignored"] += take
        return take

    def expects_error(self, room: str) -> bool:
        """A copy was sent to this room lately and its "too late" has not come yet (late_errors_to_ignore)."""
        now = self.clock()
        return any(now - t <= self.late_window_s for t in self._copies.get(room, ()))

    def on_frame(self, message: Any) -> None:
        """Every websocket message from the server, before poke-env reads it. Never raises."""
        try:
            if not isinstance(message, str):
                return
            lines = message.split("\n")
            head = lines[0]
            room = head[1:].strip() if head.startswith(">") else ""
            body = lines[1:] if room else lines
            notices = sum(1 for line in body if line.startswith(THROTTLE_LINE))
            if notices:
                self.stats["notices"] += notices
                if room.startswith("battle-"):
                    self._game(room)["throttle_notices"] += notices
                    self.note(room, {"throttle_notice": {"pending": self._describe(room)}})
                    self._schedule(room, "the server's typing limit dropped it", timer=False)
                else:
                    # A message to no room was dropped (/search, /utm, /leave, /cancelsearch): never an
                    # order, which goes to its battle room. No order is sent again; the drive loop asks for
                    # a game again if it was the search.
                    self.stats["notices_without_room"] += notices
                    self.no_room_notice_at = self.clock()
                    self.no_room_notice_wall = time.time()      # the drive loop's clock (search_dropped)
                    self.say(f"{hms()} the server's typing limit dropped a message to no room (a ladder search or a /leave): "
                             f"no order is sent again; a search just sent is asked again")
                body = [line for line in body if not line.startswith(THROTTLE_LINE)]
            if not room.startswith("battle-"):
                return
            p = self.pending.get(room)
            if p is None:
                return
            for line in body:
                parts = line.split("|")
                kind = parts[1] if len(parts) > 1 else ""
                if kind == "inactive" and len(parts) > 2 and not p["reacted"]:
                    found = TIMER_LINE_RE.match("|".join(parts[2:]))
                    if found and self.me and to_id(found.group(1)) == self.me and self.clock() - p["sent"] >= 2.0:
                        # The tick charges only a player whose order is not in: the server is waiting for ours.
                        self.stats["timer_lines"] += 1
                        self._schedule(room, "the timer says the server is still waiting for it", timer=True)
                elif kind not in NOT_A_REACTION:
                    p["reacted"] = True
        except Exception:       # bookkeeping must never disturb a game
            pass

    def _describe(self, room: str) -> Optional[Dict[str, Any]]:
        p = self.pending.get(room)
        if p is None:
            return None
        return {"rqid": p["rqid"], "order": p["message"], "ago_s": round(self.clock() - p["sent"], 2), "reacted": p["reacted"]}

    def _skip(self, why: str) -> None:
        self.stats["not_resent"][why] = self.stats["not_resent"].get(why, 0) + 1

    def _why_not(self, room: str, p: Dict[str, Any], timer: bool) -> Optional[str]:
        """Why this copy must not go (now), or None."""
        if self.pending.get(room) is not p:
            return "a newer order"
        if p["reacted"]:
            return "the server reacted"
        if p["rqid"] is None or self.current_rqid(room) != p["rqid"]:
            return "a new request"
        if self.finished(room):
            return "the game is over"
        if timer and p["timer_resends"] >= self.max_timer_resends:
            return "sent again %d times for the timer already" % self.max_timer_resends
        if not timer and p["resends"] >= self.max_resends:
            return "sent again %d times already" % self.max_resends
        return None

    def _schedule(self, room: str, why: str, timer: bool = False) -> None:
        p = self.pending.get(room)
        if p is None:
            return self._skip("no order")
        if p["reacted"]:
            return self._skip("the server reacted")
        if timer and p["timer_resends"] >= self.max_timer_resends:
            return self._skip("sent again %d times for the timer already" % self.max_timer_resends)
        if not timer and p["resends"] >= self.max_resends:
            return self._skip("sent again %d times already" % self.max_resends)
        if p["scheduled"]:
            return None             # a copy is on its way already
        p["scheduled"] = True
        if timer:
            delay = self.timer_after_s
        else:
            # 1, 2, 4 x resend_after_s: the server's queue drains one message per 600 ms.
            delay = self.resend_after_s * (2 ** p["resends"]) + self.jitter_s * self.rng.random()
        task = asyncio.ensure_future(self._resend_later(room, p, why, timer, delay))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return None

    async def _resend_later(self, room: str, p: Dict[str, Any], why: str, timer: bool = False, delay: float = 0.0) -> None:
        try:
            await asyncio.sleep(delay)
            why_not = self._why_not(room, p, timer)
            if why_not is None:
                await self.gate(p["message"], room)           # the spacing, as every message
                why_not = self._why_not(room, p, timer)      # again: a new request may have come while it waited
                if why_not is not None:
                    self.stats["stale_after_spacing"] = self.stats.get("stale_after_spacing", 0) + 1
            p["scheduled"] = False
            if why_not is not None:
                return self._skip(why_not)
            # From here to the websocket nothing waits: the copy goes for the request just checked.
            if timer:
                p["timer_resends"] += 1
                self.stats["timer_resends"] += 1
            else:
                p["resends"] += 1
            self.stats["resends"] += 1
            self._copies.setdefault(room, []).append(self.clock())
            self._game(room)["throttle_resends"] += 1
            ago = self.clock() - p["sent"]
            short = re.findall(r"-(\d+)", room)
            count = (f"{p['timer_resends']} of {self.max_timer_resends} for the timer" if timer
                     else f"{p['resends']} of {self.max_resends}")
            self.say(f"{hms()} [{short[0][-4:] if short else room[-4:]}] order not taken ({why}): sent again "
                     f"({count}, {ago:.1f} s after it went out): {p['message']}")
            self.note(room, {"throttle_resend": {"why": why, "rqid": p["rqid"], "order": p["message"],
                                                 "attempt": p["resends"] + p["timer_resends"], "timer": bool(timer),
                                                 "after_s": round(ago, 2)}})
            await self._send(p["message"], room, str(p["rqid"]))
        except asyncio.CancelledError:
            p["scheduled"] = False
            raise
        except Exception:       # a copy that could not go: the timer line or the next notice tries again
            p["scheduled"] = False
            self._skip("the copy could not be sent")
        return None

    def summary(self) -> str:
        s = self.stats
        text = (f"orders: {s['orders']} sent, {s['waits']} waited for the {self.spacing_s * 1000:.0f} ms spacing "
                f"({s['wait_ms_total'] / max(1, s['waits']):.0f} ms on average, {s['wait_ms_max']:.0f} ms at most); "
                f"typing-limit notices {s['notices']} ({s['notices_without_room']} without a room), timer lines that "
                f"showed an order not in {s['timer_lines']}, orders sent again {s['resends']} "
                f"({s['timer_resends']} of them for the timer), the server's \"too late\" for a copy let pass {s['late_ignored']}")
        if s["not_resent"]:
            text += " (not sent again: " + ", ".join(f"{k} {v}" for k, v in sorted(s["not_resent"].items())) + ")"
        return text


def read_password_stdin() -> Optional[str]:
    """
    The first line of standard input, as UTF-8 (--password-stdin: the window app writes the
    password there and closes the pipe). None when nothing was given.
    """
    stream = getattr(sys.stdin, "buffer", None)
    if stream is None:
        return None
    try:
        raw = stream.readline()
    except (OSError, ValueError):
        return None
    if raw.startswith(b"\xef\xbb\xbf"):   # a byte-order mark, should a writer add one
        raw = raw[3:]
    text = raw.rstrip(b"\r\n").decode("utf-8", errors="replace")
    del raw
    return text or None


def hms(t: Optional[float] = None) -> str:
    """The local time of day, for the lines about the connection and the search helpers."""
    return time.strftime("%H:%M:%S", time.localtime(time.time() if t is None else t))


def search_was_dropped(notice_at: Optional[float], search_sent: Optional[float], now: float,
                       searching: Optional[List[str]], fmt: str) -> bool:
    """
    The ladder search was lost to the typing limit: a notice with no room (OrderGuard) came after the
    last search went out, 2 s or more ago, and the server has not said since that this account is
    searching this format (|updatesearch|). All times time.time().
    """
    return (notice_at is not None and search_sent is not None and notice_at >= search_sent and now - notice_at >= 2.0
            and fmt not in (searching or []))


_ACCOUNT_LOCK: Any = None   # held for the life of this process (take_account_lock)


def take_account_lock(server: str, user_id: str) -> Optional[str]:
    """
    Take the lock for one account on one server: None when this process now holds it, else why
    not. Windows: a named mutex (not inherited, gone with the process). Elsewhere: flock on a file
    under ~/.cache/psbot-ladder (the descriptor is not inherited either). The same name and the
    same code as the PSBot Lookahead app's ladder.py (psbot-local), so each refuses while the
    other plays the account.
    """
    global _ACCOUNT_LOCK
    name = "PSBotLadder." + re.sub(r"[^A-Za-z0-9._-]", "_", f"{server}.{user_id}")
    taken = f"another bot is already playing as {user_id} on {server}: stop it first (two bots on one account fight over the same games)"
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateMutexW.restype = wintypes.HANDLE
        k32.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
        k32.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = k32.CreateMutexW(None, False, "Local\\" + name)
        err = ctypes.get_last_error()
        if not handle:
            return f"could not take the account lock {name} (Windows error {err})"
        if err == 183:   # ERROR_ALREADY_EXISTS: another process holds a handle to it
            k32.CloseHandle(handle)
            return taken
        _ACCOUNT_LOCK = handle
        return None
    import fcntl
    folder = os.environ.get("PSBOT_LOCK_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "psbot-ladder")
    try:
        os.makedirs(folder, exist_ok=True)
        f = open(os.path.join(folder, name + ".lock"), "a+")
    except OSError as exc:
        return f"could not take the account lock {name}: {exc}"
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return taken
    _ACCOUNT_LOCK = f
    return None


def to_id(name: Optional[str]) -> str:
    """Showdown's user id: lower case, letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def newest_checkpoint(folder: str) -> Optional[str]:
    files = glob.glob(os.path.join(folder, "learner_update_*.pt"))
    return max(files, key=os.path.getmtime) if files else None


def own_rating(text: str, me: str) -> Optional[tuple]:
    """(before, after) from the server's rating lines in text, for the account with id me."""
    loose = None
    for m in RATING_RE.finditer(text):
        name = to_id(html.unescape(m.group(1)))
        if name == me:
            return int(m.group(2)), int(m.group(3))
        if me and name.endswith(me) and loose is None:
            loose = (int(m.group(2)), int(m.group(3)))
    return loose


# ---------------------------------------------------------------------------
# The rating history kept in results.jsonl
# ---------------------------------------------------------------------------

def load_games(path: str, server: str, user_id: Optional[str]) -> List[Dict[str, Any]]:
    """The recorded games on one server, oldest first. Lines written before the account was
    recorded count for every account."""
    games: List[Dict[str, Any]] = []
    if not os.path.isfile(path):
        return games
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("server") != server:
                continue
            if user_id and d.get("user") not in (None, user_id):
                continue
            games.append(d)
    return games


def ratings_of(games: List[Dict[str, Any]]) -> List[int]:
    """The rating after each game, with the rating before the first one in front."""
    trail: List[int] = []
    for d in games:
        before, after = d.get("rating_before"), d.get("rating_after")
        if after is None:
            continue
        if not trail and before is not None:
            trail.append(int(before))
        trail.append(int(after))
    return trail


def trail_text(trail: List[int]) -> str:
    shown = trail[-TRAIL_SHOWN:]
    return ("... " if len(trail) > len(shown) else "") + " > ".join(str(r) for r in shown)


def summary_text(games: List[Dict[str, Any]]) -> str:
    wins = sum(1 for d in games if d.get("won"))
    unknown = sum(1 for d in games if d.get("won") is None)   # the bot was away when it ended (lost_track, abandoned)
    trail = ratings_of(games)
    text = f"{len(games)} games, {wins} wins, {len(games) - wins - unknown} losses" + (f", {unknown} with no result seen" if unknown else "")
    if trail:
        text += f", rating {trail[0]} -> {trail[-1]} (peak {max(trail)})"
    return text


def print_history(path: str, server: str, user_id: Optional[str]) -> int:
    games = load_games(path, server, user_id)
    where = "the public ladder" if server == "public" else server
    if not games:
        print(f"no games recorded on {where} yet")
        return 0
    print(f"Elo history on {where}" + (f", account {user_id}" if user_id else "") + f": {summary_text(games)}\n")
    rated = [int(d["rating_after"]) for d in games if d.get("rating_after") is not None]
    low, high = (min(rated), max(rated)) if rated else (0, 0)
    width = 24
    print(f"{'#':>4}  {'when':<12}  {'':<4}  {'turns':>5}  {'opponent':<26}  {'rating':>6}  {'change':>6}")
    for i, d in enumerate(games, 1):
        try:
            when = time.strftime("%d %b %H:%M", time.localtime(calendar.timegm(time.strptime(d["time"], "%Y-%m-%dT%H:%M:%SZ"))))
        except (KeyError, ValueError):
            when = "?"
        opp = str(d.get("opponent") or "?")[:18]
        if d.get("opponent_rating"):
            opp += f" ({d['opponent_rating']})"
        before, after = d.get("rating_before"), d.get("rating_after")
        if after is None:
            rating, change, bar = "?", "", ""
        else:
            rating = str(after)
            change = f"{int(after) - int(before):+d}" if before is not None else ""
            bar = "#" * (1 + (round((width - 1) * (int(after) - low) / (high - low)) if high > low else 0))
        print(f"{i:>4}  {when:<12}  {'WIN' if d.get('won') else 'LOSS':<4}  {str(d.get('turns', '?')):>5}  {opp:<26}  {rating:>6}  {change:>6}  {bar}")
    trail = ratings_of(games)
    if trail:
        print(f"\nelo: {trail_text(trail)}")
    return 0


def write_graph(server: str, user_id: Optional[str]) -> Optional[str]:
    """Rewrite the graph page. None when it could not be written: the graph must never
    disturb a game."""
    try:
        import ladder_graph

        return ladder_graph.write_page(load_games(OUT, server, user_id), GRAPH, server, user_id)
    except Exception:
        return None


def show_graph(server: str, user_id: Optional[str], once: bool) -> int:
    """Write the graph, open it in the browser and keep it up to date until the window is closed."""
    if user_id is None:
        games = load_games(OUT, server, None)
        user_id = games[-1].get("user") if games else None   # the account that played last
    if not load_games(OUT, server, user_id):
        print("no games recorded on " + ("the public ladder" if server == "public" else server) + " yet")
        return 0
    path = write_graph(server, user_id)
    if path is None:
        print("the graph could not be written")
        return 1
    print(f"graph: {path}")
    if once:
        return 0
    try:
        os.startfile(path)   # the default browser, on Windows
    except AttributeError:
        import webbrowser

        webbrowser.open("file://" + path)
    print("It is open in your browser and reloads itself.")
    print("Leave this window open to keep it up to date, and close it when you are done.")
    stamp = os.path.getmtime(OUT)
    try:
        while True:
            time.sleep(4.0)
            now = os.path.getmtime(OUT) if os.path.isfile(OUT) else stamp
            if now != stamp and write_graph(server, user_id) is not None:
                stamp = now
    except KeyboardInterrupt:
        return 0


# ---------------------------------------------------------------------------
# The player
# ---------------------------------------------------------------------------

def make_player(args: argparse.Namespace, user: str, password: Optional[str]) -> Any:
    from poke_env import AccountConfiguration, ServerConfiguration, ShowdownServerConfiguration

    from hybrid import HybridConfig
    from players import audited, batched_policy_player

    if args.server == "public":
        server = ShowdownServerConfiguration
    else:
        host, _, port = args.server.partition(":")
        if host != "local" or not port.isdigit():
            raise SystemExit("--server is 'public' or 'local:PORT'")
        server = ServerConfiguration(f"ws://127.0.0.1:{port}/showdown/websocket", "https://play.pokemonshowdown.com/action.php?")

    hcfg = HybridConfig(checkpoint=args.checkpoint, beta=0.0, veto_gap=None, policy_weight=1.0,
                        greedy=True, device=args.device, random_init=False)
    me = to_id(user)
    seen: Dict[str, Dict[str, Any]] = {}
    recorder = BattleRecorder(args.battles_dir) if args.save_battles else None
    # Remote helpers: a decision is judged a few seconds after it was made, once a link stall in it
    # is known (the bridge takes a helper that answers nothing for its timeout, max(4, --deadline), for
    # down; a probe goes every 2 s): such failures are the link's (SearchHealth, --link-down-limit).
    health = SearchHealth(args.health_window,
                          settle_s=(max(4.0, run_deadline(args)) + 3.0) if getattr(args, "search_remote", None) else 0.0)
    console = ConsoleLines()
    last_decision: Dict[str, Dict[str, Any]] = {}   # battle tag -> what the search did for its latest order (--app-events)

    # The network alone, as before, unless a search, arms or settings per game were asked for.
    modes = run_modes(args)
    settings = getattr(args, "arm_settings", None)       # --arm-settings: each game's setting (None: off)
    with_search = bool(args.arms) or args.player != "net" or bool(settings)
    extra: Dict[str, Any] = {}
    bridge = None
    picker = None
    setting_picker = None
    if with_search:
        import foulplay_bridge as fb
        from search_player import ArmPicker, MixSettings, minimum_deadline, search_player, split_mode

        if any(split_mode(mode)[0] != "net" for mode in modes):
            problems = run_problems(args)
            if problems:
                raise SystemExit(problems[0])
            target = None
            if getattr(args, "search_remote", None):
                # The helpers on another machine over ssh (foulplay_bridge.RemoteTarget); WSL is not used.
                try:
                    target = fb.remote_target(args.search_remote, key=args.search_key, ssh=args.search_ssh,
                                              known_hosts=args.search_known_hosts, code_dir=args.search_remote_dir)
                except fb.FoulPlayBridgeError as exc:
                    raise SystemExit(f"the search box: {exc}")
                print(f"search helpers on {target.host} over ssh (the player and the network stay here)", flush=True)
            ok, why = fb.available(remote=target)
            if not ok:
                raise SystemExit("the search cannot be used: " + why)
            print(f"search: {why}")
            print(f"starting {args.helpers} search helpers ...", flush=True)
            app_event("phase", text=f"starting {args.helpers} search helpers in WSL")
            # Live play: a frozen helper is replaced within seconds, also one with no battle (idle
            # pings), a question that cannot be answered in time fails at once, and a helper is
            # never started from the event loop (review of 2026-10-06, findings 1 to 3).
            bridge = fb.FoulPlayBridge(helpers=args.helpers, search_ms=args.search_ms, timeout=max(4.0, run_deadline(args)),
                                       hang_timeout=max(5.0, 3.0 * run_deadline(args)), idle_ping=3.0,
                                       early_expire=True, background_spawn=True, fail_when_down=True, remote=target)
            try:
                bridge.start(timeout=180.0)   # before logging in: the helpers take seconds to start, WSL up to a minute
            except Exception as exc:
                bridge.close()
                raise SystemExit(f"the search helpers did not start: {exc}")
            if target is not None:
                trip = bridge.rtt_start or {}
                if not trip.get("probes"):
                    bridge.close()
                    raise SystemExit(f"the search helpers on {target.host} started but did not answer their probes")
                print(f"round trip to the search helpers: {trip['rtt_ms']:.0f} ms (fastest {trip['min_ms']:.0f}, 90th "
                      f"{trip['p90_ms']:.0f}, slowest {trip['max_ms']:.0f}; {trip['probes']} probes), kept up to date and taken "
                      f"off each decision's time on the box", flush=True)
                problems = run_problems(args, rtt_s=float(trip["p90_ms"]) / 1000.0)
                if problems:
                    bridge.close()
                    raise SystemExit(problems[0])
        if args.arms:
            earlier = arm_counts(load_games(OUT, args.server, me), os.path.basename(args.checkpoint), experiment=experiment_tag(args))
            picker = ArmPicker(modes, counts=earlier)
        elif settings:
            from search_player import SettingPicker

            earlier = setting_counts(load_games(OUT, args.server, me), os.path.basename(args.checkpoint), experiment_tag(args))
            setting_picker = SettingPicker([s.game for s in settings], counts=earlier)

        live = bool(getattr(args, "predictions", False))
        show = bool(getattr(args, "print_predictions", live))
        from search_player import short_tag

        spend = SpendCap(getattr(args, "max_model_usd", 0.0))
        run_deciders: Dict[str, Any] = {}       # the model arms' deciders (filled below): what the spending cap turns off

        def luna_line(record: Dict[str, Any]) -> Optional[str]:
            """What the model (Luna, Sol) did at this decision when it was not simply mix's own pick (None otherwise)."""
            luna = record.get("luna")
            if not isinstance(luna, dict):
                return None
            mix_text, choice_text = luna.get("mix_text") or "mix's pick", luna.get("choice_text") or "?"
            head = f"[{short_tag(record.get('battle') or '')}] T{record.get('turn')} {model_label(luna.get('who'))}"
            if luna.get("solo"):
                # The model alone: every answer is its own move; only a move it did not make is worth a line.
                if luna.get("fell_back"):
                    return f"{head} gave no move ({luna.get('why')}): the fallback rule's {mix_text} is played"
                return None
            if luna.get("overruled"):
                return f"{head} wanted {choice_text}: overruled by the blunder guard, mix's {mix_text} is played"
            if luna.get("choice") is not None and luna.get("choice") != luna.get("mix"):
                conf = luna.get("confidence")
                return (f"{head} plays {choice_text} instead of mix's {mix_text}"
                        + (f" (confidence {100.0 * float(conf):.0f}%)" if isinstance(conf, (int, float)) else "")
                        + (f": {luna['reason']}" if luna.get("reason") else ""))
            if luna.get("fell_back") and luna.get("asked"):
                return f"{head} gave no answer ({luna.get('why')}): mix's {mix_text} is played"
            return None

        def lines_line(record: Dict[str, Any]) -> Optional[str]:
            """The lines read at this decision, in one line (None without lines)."""
            lines = record.get("lines")
            if not isinstance(lines, dict) or not lines.get("text"):
                return None
            text = f"[{short_tag(record.get('battle') or '')}] T{record.get('turn')} read ahead: {lines['text']}"
            if lines.get("changed"):
                text += f"  -> played instead of mix's {lines.get('mix_text') or 'pick'} (+{float(lines.get('gain') or 0.0):.2f})"
            return text

        def on_decision(record: Dict[str, Any]) -> None:
            battle = record.get("battle") or ""
            pred = record.get("predict")
            fell = record.get("deep_fallback")
            if isinstance(fell, dict):
                console.say(f"{hms()} [{short_tag(battle)}] {record.get('mode')}: today's search budget for the rest of this game "
                            f"({fell.get('why')}, turn {fell.get('turn')})")
            if bridge is not None and record.get("asked"):
                # Where this decision was searched, and the round trip to it then (remote: the
                # helpers' probes; None here or in WSL). A copy: the player keeps its own record.
                record = dict(record, search_at=bridge.where, rtt_ms=bridge.round_trip_ms())
            if isinstance(pred, dict):
                # A copy: "outcome" is added to the record's own dict after the turn, while the
                # recorder's thread may still be writing this line.
                record = dict(record, predict=dict(pred))
            if recorder is not None:
                recorder.write(battle, "note", {"decision": record})
            health.note(record)
            if APP_EVENTS:
                try:
                    last_decision[record.get("battle") or ""] = {
                        "rqid": record.get("rqid"), "source": record.get("source"), "reason": record.get("reason") or "",
                        "searched": record.get("search") is not None, "overruled": bool(record.get("net_overruled")),
                        "iterations": record.get("iterations"), "ms": record.get("ms")}
                    model = record.get("luna")
                    if isinstance(model, dict):
                        # by: who decided ("Luna", "Sol"); changed: its pick was not mix's own; fell: mix played, and why
                        last_decision[record.get("battle") or ""].update(
                            by=model_label(model.get("served_by") or model.get("who")),
                            changed=bool(record.get("source") in OPENAI_MODELS + CLAUDE_MODELS
                                                                           and model.get("choice") != model.get("mix")),
                            fell=(model.get("why") or "") if model.get("fell_back") else "", solo=bool(model.get("solo")))
                    if str(record.get("arm")).endswith("-solo") and int(record.get("legal") or 0) <= 1:
                        last_decision[record.get("battle") or ""]["reason"] = "single"      # the one legal action: nobody chose
                    if isinstance(record.get("lines"), dict):
                        last_decision[record.get("battle") or ""].update(
                            lines=bool(record["lines"].get("values")), lines_changed=bool(record["lines"].get("changed")))
                except Exception:
                    pass
            try:
                spend.add(record, run_deciders, lambda text: console.say(f"{hms()} {text}"))
            except Exception:       # the cap's bookkeeping must never disturb a game
                pass
            try:
                told = lines_line(record)
                if told and show:
                    console.say(told)
                told = luna_line(record)
                if told:
                    console.say(told)
            except Exception:       # a line for the window must never disturb a game
                pass
            line = pred.get("line") if isinstance(pred, dict) and not pred.get("repeat") else None
            if line:
                if recorder is not None:
                    recorder.write(battle, "note", {"prediction": {
                        "turn": record.get("turn"), "line": line, "top": pred.get("top"), "p_switch": pred.get("p_switch"),
                        "played": record.get("played"), "mode": record.get("mode"), "guard": record.get("guard")}})
                if show:
                    console.say(f"[{short_tag(battle)}] {line}")

        def on_outcome(record: Dict[str, Any]) -> None:
            battle = record.get("battle") or ""
            info = record.get("predict") or {}
            outcome = info.get("outcome") or {}
            if recorder is not None:
                recorder.write(battle, "note", {"outcome": record})
            if show and outcome.get("text") and info.get("line"):
                console.say(f"[{short_tag(battle)}] T{record.get('turn')} {outcome['text']}")

        extra = dict(mode=modes[0], mode_picker=picker, bridge=bridge, search_ms=args.search_ms, deadline=args.deadline,
                     mix=mix_settings(args), think_longer=think_longer(args), on_decision=on_decision,
                     live_predictions=live, on_outcome=on_outcome)
        budgets = arm_budgets(args, modes) if not settings else {}     # a setting's games have the setting's own
        if budgets:
            extra["budgets"] = budgets
        if setting_picker is not None:
            extra["setting_picker"] = setting_picker
        if decider_models(args):
            # The deciders of the model arms (made in main(): the key was read and checked there).
            # Their own lines (5 failures in a row, the key refused) come out with the others.
            deciders = dict(getattr(args, "_deciders", None) or {})
            if getattr(args, "_luna", None) is not None:
                deciders.setdefault("luna", args._luna)
            for who in deciders.values():
                who.client.say = lambda text: console.say(f"{hms()} {text}")
            run_deciders.update(deciders)
            extra["deciders"] = deciders
            extra["luna"] = deciders.get("luna")
        if "mix-lines" in modes:
            from search_player import LinesRule

            extra["lines_rule"] = LinesRule(margin=float(args.lines_margin))
        if "predict" in modes:
            from opponent_model import PredictSettings

            extra.update(predict=PredictSettings(min_confidence=args.predict_confidence, min_gain=args.predict_gain,
                                                 max_risk=args.predict_risk, payoff_ms=args.payoff_ms,
                                                 once_bitten=args.predict_once_bitten))
        base = audited(search_player())
    else:
        base = audited(batched_policy_player())
    guard_box: Dict[str, Any] = {}      # "guard": the OrderGuard, made once the client is there (below)

    class LadderPlayer(base):
        """The evaluated policy player, plus a pause before each move, a note of each game's
        rating change, and what it takes to pick its games up again after a lost connection."""

        def __init__(self, *a: Any, **kw: Any):
            # Before the base class (make_player now connects only after it, with start_listening=False).
            self._rq: Dict[str, Any] = {}            # battle tag -> rqid of its last request
            self._sent: Dict[str, tuple] = {}        # battle tag -> (rqid, message) of the last order sent, or tried
            self._resent: Dict[str, tuple] = {}      # battle tag -> (rqid, connection) of the last order sent again
            self._deciding: Dict[str, Any] = {}      # battle tag -> rqid of the decision being made
            self._last_turn: Dict[str, int] = {}     # battle tag -> turn of its last decision
            self._rejoining: set = set()             # replay received, the first request after it not seen yet
            self.replayed: Dict[str, float] = {}     # battle tag -> when the server last sent it from |init|
            self.rooms_gone: Dict[str, str] = {}     # battle tag -> why its room could not be joined again
            self.server_games: set = set()           # the battles the server says this account is in (|updatesearch|)
            self.server_searching: List[str] = []    # the formats it says this account searches
            self.rejoined_battles: set = set()       # battles picked up again after a lost connection
            super().__init__(*a, **kw)

        async def choose_move(self, battle):
            began = time.monotonic()
            tag = battle.battle_tag
            rqid = (getattr(battle, "last_request", None) or {}).get("rqid")
            self._deciding[tag] = rqid
            self._last_turn[tag] = getattr(battle, "turn", 0)
            try:
                order = super().choose_move(battle)
                if inspect.isawaitable(order):
                    order = await order
                if args.think_ms > 0:
                    pause = args.think_ms / 1000.0
                    if with_search:
                        pause -= time.monotonic() - began   # a search has already taken its time
                    if pause > 0:
                        await asyncio.sleep(pause)
                return order
            finally:
                if self._deciding.get(tag) == rqid:
                    self._deciding.pop(tag, None)

        # -- a battle sent again after its room was joined again ---------------------------------

        def _start_over(self, room: str) -> None:
            """
            The server is sending this battle again from |init| (its room was joined again after a
            lost connection). poke-env would parse it into the battle it has, every line a second
            time: it gets a new Battle instead, built from the replay alone. The battle's slot among
            the games at a time stays taken (it never ended), and the network keeps its history of
            the game (one step per decision: a request is never decided twice, see _held_request).
            The search lets the battle go (search_player.battle_rejoined).
            """
            from poke_env.battle.battle import Battle
            from poke_env.data import GenData

            self._battles[room] = Battle(battle_tag=room, username=self.username, logger=self.logger,
                                         gen=GenData.from_format(self._format).gen, save_replays=self._save_replays)
            events = getattr(self, "_events", None)
            if isinstance(events, dict):
                events.pop(room, None)
            retries = getattr(self, "_audit_retries", None)
            if isinstance(retries, dict):
                retries.pop(room, None)
            self._rejoining.add(room)
            self.rejoined_battles.add(room)
            hook = getattr(self, "battle_rejoined", None)
            if hook is not None:
                hook(room)
            if recorder is not None:
                recorder.write(room, "note", {"rejoined": {"turn_before": self._last_turn.get(room),
                                                           "rqid_before": self._rq.get(room)}})

        def mark_away(self, room: str) -> None:
            """
            A game the server started while the bot was away, found at the reconnect. The bot first
            sees it in the middle, from a replay that has none of its earlier requests, which Foul
            Play cannot follow: every search of it failed, and those failures stopped the run
            (2026-10-15). It is played as a rejoined battle instead: the network's moves, counted as
            "rejoined", not as failures of the search.
            """
            if room in self.rejoined_battles:
                return
            self.rejoined_battles.add(room)
            hook = getattr(self, "battle_rejoined", None)
            if hook is not None:
                hook(room)
            # No note in its record here: the first one written is the battle's header (recorder.seen).
            # Its results line says "rejoined", as for any battle in rejoined_battles.
            print(f"{room}: started while the bot was away: the network plays it (the search cannot pick it up midway)", flush=True)

        def _trim_events(self, room: str, split_messages: List[List[str]]) -> None:
            """After a replay: the protocol lines the network has not seen yet are those after the
            turn of its last decision, not the whole battle (hybrid.HybridPlayer's per-decision events)."""
            events = getattr(self, "_events", None)
            if not isinstance(events, dict):
                return
            lines = split_messages[1:]
            turn = self._last_turn.get(room)
            start = 0
            if turn is not None:
                for i, parts in enumerate(lines):
                    if len(parts) > 2 and parts[1] == "turn" and parts[2].strip() == str(turn):
                        start = i + 1
            events[room] = [parts for parts in lines[start:]
                            if len(parts) > 1 and parts[0] == "" and parts[1] not in ("t:", "request", "turn")]

        def _held_request(self, room: str, split_messages: List[List[str]]) -> Optional[tuple]:
            """
            The first request after a replay. None: a new request, decided as usual. Otherwise
            (the messages without it, the request, the order to send again or None, why): the
            request is one the bot already answered, or is answering, and must not be decided
            twice. The server adds |sentchoice| when it has the order; when it has not, the order
            that went into the lost connection is sent again, exactly as it was.
            """
            lines = split_messages[1:]
            parts = next((p for p in lines if len(p) > 2 and p[1] == "request"), None)
            if parts is None or not "|".join(parts[2:]).strip():
                return None
            request = json.loads("|".join(parts[2:]))
            rqid = request.get("rqid")
            self._rejoining.discard(room)
            resend = None
            if any(len(p) > 1 and p[1] == "sentchoice" for p in lines):
                why = "the server has its order already"
            elif rqid is not None and self._deciding.get(room) == rqid:
                why = "its decision is still being made, and is sent when it is"
            elif rqid is not None and self._sent.get(room, (None, None))[0] == rqid:
                # Sent again WITH the request's id: the server may have it already without saying
                # so (seen when the connection went at the moment the order was sent: no
                # |sentchoice|, and the copy was refused "too late"). A copy without the id would
                # then be taken as the answer to the NEXT request, and every order after it would
                # land one request late. With the id the server refuses a copy it does not need.
                resend = self._sent[room][1].split("|", 1)[0] + f"|{rqid}"
                why = "the order sent before the connection was lost is sent again, with the request's id: " + resend
                # The same battle sent twice on one connection (the server's own join at the login
                # and ours): the copy goes once per connection, not once per replay.
                conn = id(getattr(self.ps_client, "websocket", None))
                if self._resent.get(room) == (rqid, conn):
                    resend, why = None, "the order for it was sent again already on this connection"
                else:
                    self._resent[room] = (rqid, conn)
            else:
                return None
            rest = [split_messages[0]] + [p for p in lines if not (len(p) > 1 and p[1] in ("request", "sentchoice"))]
            return rest, request, resend, why

        def _room_gone(self, room: str, text: str) -> None:
            """The room could not be joined again (it no longer exists): the game ended while the bot was away."""
            self.rooms_gone[room] = text
            battle = self._battles.get(room)
            if battle is None or battle.finished:
                return
            battle._finished = True
            try:
                self._battle_count_queue.get_nowait()   # its slot among the games at a time
                self._battle_count_queue.task_done()
            except Exception:
                pass
            try:
                self._battle_finished_callback(battle)
            except Exception:
                pass

        async def _handle_battle_message(self, split_messages):  # noqa: D401
            room = ""
            replay = False
            held = None
            recorded = False
            try:
                head = split_messages[0][0] if split_messages and split_messages[0] else ""
                if head.startswith(">"):
                    room = head[1:].strip()
                    for parts in split_messages[1:]:
                        if len(parts) > 2 and parts[1] == "raw":
                            rating = own_rating("|".join(parts[2:]), me)
                            if rating:
                                seen.setdefault(room, {})["rating"] = rating
                        elif len(parts) > 2 and parts[1] == "inactive":
                            text = "|".join(parts[2:]).strip()
                            if "timer is ON" in text:
                                seen.setdefault(room, {})["timer"] = True
                            elif text.startswith("Time left:") or ("reconnect" in text and to_id(text).startswith(me)):
                                seen.setdefault(room, {})["time_left"] = text   # shown when the battle is joined again
                        elif len(parts) > 2 and parts[1] in ("-message", "message"):
                            # How a game ended when it was not played out: a forfeit, or a loss on time
                            # ("<name> lost due to inactivity."), and by whom (the A/B report counts them per arm).
                            text = "|".join(parts[2:]).strip()
                            for how, tail in (("forfeit", " forfeited."), ("inactivity", " lost due to inactivity.")):
                                if text.endswith(tail):
                                    seen.setdefault(room, {})["ending"] = {"how": how, "by": "us" if to_id(text[: -len(tail)]) == me else "them"}
                        elif len(parts) > 2 and parts[1] == "request" and parts[2].strip():
                            try:
                                self._rq[room] = json.loads("|".join(parts[2:])).get("rqid")
                            except ValueError:
                                pass
                    if room.startswith("battle-"):
                        kinds = {parts[1] for parts in split_messages[1:] if len(parts) > 1}
                        guard = guard_box.get("guard")
                        if "error" in kinds:
                            # "Too late": the server has an order for this request already, or has
                            # moved on to the next one, which comes. poke-env would decide again and
                            # send one more order, which the server takes as the answer to the next
                            # request: from then on every order lands one request late.
                            # "There's nothing to choose" (and "Can't undo"): the server had the copy
                            # sent again already and waits for the opponent; deciding again would be
                            # the same mistake. In a battle picked up again after a lost connection,
                            # every such error; otherwise one per copy the typing-limit guard sent to
                            # this room lately (OrderGuard.late_errors_to_ignore), whatever the request
                            # is by then.
                            late = [p for p in split_messages[1:] if len(p) > 2 and p[1] == "error"
                                    and p[2].startswith(LATE_ERRORS)]
                            if late and room not in self.rejoined_battles:
                                take = guard.late_errors_to_ignore(room, len(late)) if guard is not None else 0
                                late = late[:take]
                            if late:
                                if recorder is not None:
                                    recorder.write(room, "in", "\n".join("|".join(parts) for parts in split_messages[1:]))
                                    recorder.write(room, "note", {"too_late_ignored": len(late)})
                                    recorded = True
                                split_messages = [p for p in split_messages if not any(p is q for q in late)]
                                if len(split_messages) <= 1:
                                    return None
                                kinds = {parts[1] for parts in split_messages[1:] if len(parts) > 1}
                        if "noinit" in kinds:
                            text = next(("|".join(p[2:]) for p in split_messages[1:] if len(p) > 1 and p[1] == "noinit"), "")
                            if recorder is not None:
                                recorder.write(room, "in", "\n".join("|".join(parts) for parts in split_messages[1:]))
                            self._room_gone(room, text)
                            return None     # nothing for poke-env, which does not know this message
                        if "init" in kinds:
                            if room in self._battles and self._battles[room].finished:
                                return None     # a game that is over and reported: poke-env must not end it twice
                            self.replayed[room] = time.time()
                            if room in self._battles:
                                replay = True
                                self._start_over(room)
                            elif kinds & {"start", "turn"}:
                                # A battle the bot never saw, sent whole in one message: the server put the
                                # bot back into it after a lost connection (a game that began meanwhile).
                                # A new game's |init| comes alone, its first lines after it. Foul Play
                                # cannot follow a battle from such a replay: every search of it failed
                                # and stopped the run (2026-10-15, 11:46).
                                self.mark_away(room)
                        if room in self._rejoining and "request" in kinds:
                            held = self._held_request(room, split_messages)
                    if recorder is not None and room.startswith("battle-") and not recorded:
                        if room not in recorder.seen:
                            mode_of = getattr(self, "battle_mode", None)
                            arm = mode_of(room) if mode_of else None
                            budgets = getattr(self, "budgets", None) or {}
                            note = {"battle": room, "user": me, "server": args.server,
                                    "checkpoint": os.path.basename(args.checkpoint), "arm": arm,
                                    "experiment": experiment_tag(args),
                                    # how the decision notes' v_exp and phi make a win estimate
                                    "value_settings": getattr(getattr(self, "policy", None), "value_settings", None),
                                    "budget": budgets[arm].as_dict() if arm in budgets else None}
                            setting = self.battle_setting(room) if hasattr(self, "battle_setting") else None
                            if setting is not None:
                                # --arm-settings: the game's setting and its own budget, not the player's
                                own = setting.own_budget()
                                note.update(setting=setting.name, budget=own.as_dict() if own is not None else None,
                                            setting_budget=setting.asks())
                            recorder.write(room, "note", note)
                        recorder.write(room, "in", "\n".join("|".join(parts) for parts in split_messages[1:]))
            except Exception:  # a note for the printout must never disturb the game
                held = None
            result = await super()._handle_battle_message(held[0] if held else split_messages)
            if replay:
                try:
                    self._trim_events(room, split_messages)
                except Exception:
                    pass
            if held:
                _rest, request, resend, why = held
                battle = self._battles.get(room)
                if battle is not None:
                    battle.parse_request(request, self._strict_battle_tracking)   # the state, no decision
                print(f"{room}: the request after the rejoin (rqid {request.get('rqid')}) is not decided again: {why}", flush=True)
                if recorder is not None:
                    recorder.write(room, "note", {"held_request": {"rqid": request.get("rqid"), "why": why}})
                if resend:
                    await self.ps_client.send_message(resend, room)
            return result

    # After a keep-alive ping goes unanswered the websocket library waits close_timeout (10 s) for
    # the server to close the connection, which a dead one never does: 25 s from the drop to
    # "connection lost" with pings 5 + 10 (box run F1). poke-env's listen() gives connect() no
    # close_timeout, so its module gets a connect() that does. Nothing else of it changes.
    try:
        import poke_env.ps_client.ps_client as _psc

        class _WebsocketsWithCloseTimeout:
            def __init__(self, module: Any):
                self._module = module

            def __getattr__(self, name: str) -> Any:
                return getattr(self._module, name)

            def connect(self, *a: Any, **kw: Any) -> Any:
                kw.setdefault("close_timeout", args.close_timeout)
                return self._module.connect(*a, **kw)

        if not isinstance(_psc.ws, _WebsocketsWithCloseTimeout):
            _psc.ws = _WebsocketsWithCloseTimeout(_psc.ws)
    except Exception as exc:
        print(f"note: the websocket's close timeout stays at the library's default ({type(exc).__name__}: {exc})")

    app_event("phase", text="loading the network " + os.path.basename(args.checkpoint))
    try:
        player = LadderPlayer(
            hybrid=hcfg,
            account_configuration=AccountConfiguration(user, password),
            server_configuration=server,
            battle_format=args.format,
            max_concurrent_battles=args.parallel,
            start_timer_on_battle_start=args.timer,
            log_level=40,
            # A connection that dies without closing (a network adapter gone, as when WSL restarts)
            # is noticed within --ping-interval + --ping-timeout s, not poke-env's 20 + 20: the
            # server's minute for a game in progress starts when IT sees the drop (review of
            # 2026-10-07, finding 2: 45 s went by unnoticed). Websocket pings, not Showdown messages.
            ping_interval=args.ping_interval,
            ping_timeout=args.ping_timeout,
            # Not yet: the network loads after poke-env's Player.__init__ (seconds on the GPU), and the
            # server's challenge came meanwhile, so the first login went through poke-env's own
            # handler (box, 2026-10-09). The connection is made once watch_messages is in place, below.
            start_listening=False,
            **extra,
        )
    except BaseException:
        if bridge is not None:
            bridge.close()
        raise
    if getattr(player, "policy", None) is None:
        raise SystemExit("no network was loaded")
    if APP_EVENTS and getattr(args, "app_net_view", False) and hasattr(player, "_forward_batch"):
        attach_net_view(player)
    player._ladder_seen = seen
    player._ladder_health = health
    player._ladder_recorder = recorder
    player._ladder_console = console
    player._ladder_bridge = bridge
    player._ladder_modes = modes if with_search else None
    if setting_picker is not None:
        # The games in progress, for the picker's choice among the settings played least (one with no game in progress).
        setting_picker.in_play = lambda: [tag for tag, b in list(player.battles.items()) if not getattr(b, "finished", False)]

    # A lost connection is told in one line ("connection lost at ..."): poke-env's own report of
    # it, from listen(), is kept to one line as well instead of a screenful of traceback.
    import logging

    class _OneLine(logging.Filter):
        def __init__(self) -> None:
            super().__init__()
            self.told: Dict[str, float] = {}   # line -> when it was last let through

        def filter(self, record: logging.LogRecord) -> bool:
            if record.funcName == "listen" and record.exc_info:
                exc = record.exc_info[1]
                text = f"the websocket ended: {type(exc).__name__}: {exc}"
                record.msg, record.args = text, None
                record.exc_info = record.exc_text = None
                # Reconnecting every second: the same refusal is told once, not once a second.
                if time.time() - self.told.get(text, 0.0) < 30.0:
                    return False
                self.told[text] = time.time()
            return True

    try:
        player.ps_client.logger.addFilter(_OneLine())
    except Exception:
        pass

    # poke-env leaves the room the moment the game is won or lost, and the server posts the
    # rating change a moment after that. Stay until the rating has been read (12 s at most).
    client = player.ps_client
    send = client.send_message
    leaving: set = set()

    def guard_note(room: str, data: Dict[str, Any]) -> None:
        if recorder is not None:
            recorder.write(room, "note", data)

    def finished(room: str) -> bool:
        b = player.battles.get(room)
        return b is None or bool(getattr(b, "finished", False)) or room in getattr(player, "rooms_gone", {})

    async def send_message(message: str, room: str = "", message_2: Optional[str] = None):
        if isinstance(message, str) and message.startswith("/leave battle-"):
            task = asyncio.ensure_future(leave_later(message.split(" ", 1)[1].strip()))
            leaving.add(task)
            task.add_done_callback(leaving.discard)
            return None
        waited = 0.0
        try:
            # Showdown's typing limit: every message waits for the spacing after the one before (OrderGuard.gate).
            waited = await guard.gate(message, room)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        try:
            if OrderGuard.is_order(message, room):
                # The order for this battle's current request, kept until the next one: when it goes
                # into a connection that is lost, it is sent again after the room is joined again,
                # and when the typing limit drops it, it is sent again (OrderGuard).
                player._sent[room] = (player._rq.get(room), message)
                guard.sent(room, player._rq.get(room), message)
                if waited >= 0.001 and recorder is not None:
                    recorder.write(room, "note", {"order_wait_ms": round(waited * 1000.0, 1)})
            if recorder is not None and room:
                recorder.write(room, "out", message if not message_2 else f"{message}|{message_2}")   # battle rooms only
        except Exception:
            pass
        if APP_EVENTS and isinstance(room, str) and room.startswith("battle-") and isinstance(message, str) \
                and message.startswith("/choose "):
            try:
                b = player.battles.get(room)
                fields: Dict[str, Any] = {"battle": room, "turn": int(getattr(b, "turn", 0) or 0),
                                          "choice": message[len("/choose "):].split("|", 1)[0].strip()}
                decision = last_decision.pop(room, None)
                if decision is not None:
                    fields.update(decision)
                app_event("move", **fields)
            except Exception:
                pass
        try:
            return await send(message, room, message_2)
        except Exception as exc:
            # The connection is gone (the drive loop is reconnecting). Nothing sent may stop a game:
            # an order is sent again after the rejoin, a ladder search is asked again.
            word = message.split(" ", 1)[0] if isinstance(message, str) else "?"
            print(f"{hms()} {room or 'lobby'}: {word} not sent ({type(exc).__name__}: the connection is down)"
                  + (", it is sent again once the battle is joined again" if word == "/choose" else ""), flush=True)
            if recorder is not None and room:
                recorder.write(room, "note", {"not_sent": message, "error": type(exc).__name__})
            return None

    async def send_copy(message: str, room: str, rqid: str) -> None:
        """
        An order sent again for the typing limit, once OrderGuard has waited for the spacing and checked
        that it is still the order for the current request: straight to the websocket, nothing awaited
        before it goes. Into the battle file, but not the battle's order on record (player._sent and
        the guard's pending: the copy is the same order, with the request id it was made for), and not
        told to the search a second time (search_player's _order_sent skips what choose_move noted).
        """
        noted = getattr(player, "_noted", None)
        try:
            if isinstance(noted, dict) and player._searched_battle(room):
                noted[room] = message
        except Exception:
            pass
        try:
            if recorder is not None:
                recorder.write(room, "out", f"{message}|{rqid}")
        except Exception:
            pass
        await send(message, room, rqid)

    guard = OrderGuard(send_copy, spacing_s=args.order_spacing_ms / 1000.0, max_wait_s=args.order_max_wait_s,
                       resend_after_s=args.resend_after_s, jitter_s=0.3, max_resends=args.max_resends,
                       max_timer_resends=args.max_timer_resends, me=me,
                       current_rqid=lambda room: player._rq.get(room), finished=finished, say=console.say, note=guard_note)
    guard_box["guard"] = guard
    player._order_guard = guard

    async def leave_later(tag: str) -> None:
        t0 = time.time()
        while time.time() - t0 < 12.0 and "rating" not in seen.get(tag, {}):
            await asyncio.sleep(0.2)
        try:
            await guard.gate(f"/leave {tag}", "")
            await send(f"/leave {tag}")
        except Exception:
            pass

    client.send_message = send_message

    watch_messages(client, player, guard)
    # Now the connection: every login, the first one too, goes through log_in_checked.
    client._listening_coroutine = asyncio.run_coroutine_threadsafe(client.listen(), client.loop)
    return player


def watch_messages(client: Any, player: Any, guard: Optional[OrderGuard] = None) -> None:
    """
    The client's messages pass through here before poke-env reads them. What the server says
    about this account after every login (|updatesearch|): the battles it is in (to join again
    after a lost connection, also one that started while the bot was away) and whether it is
    still searching; poke-env reads nothing of it. And the login itself, every one of them (the
    first and each after a reconnect; make_player installs this before the client connects):
    |challstr| goes to log_in_checked, |updateuser| to note_updateuser (the confirmation), and
    |popup|, |nametaken| and |raw| to note_server_word (told in one line when they bear on the
    login). Every message also goes to the OrderGuard first (the typing limit's notice, the
    server's reactions to orders).
    """
    handle = client._handle_message
    login_state(client)

    async def handle_message(message: str):
        if guard is not None:
            guard.on_frame(message)
        if not isinstance(message, str):
            return await handle(message)
        if message.startswith("|updatesearch|"):
            try:
                data = json.loads(message.split("|", 2)[2])
                player.server_games = {room for room in (data.get("games") or {}) if str(room).startswith("battle-")}
                player.server_searching = list(data.get("searching") or [])
            except Exception:
                pass
        if message.startswith("|challstr|"):
            return await log_in_checked(client, message)
        if not message.startswith(">"):        # not a room's: the global messages, where the login's are
            handled = False
            for line in message.split("\n"):
                try:
                    if line.startswith("|updateuser|"):
                        note_updateuser(client, line)
                    elif line.startswith(("|popup|", "|nametaken|", "|raw|")):
                        handled = note_server_word(client, line) or handled
                except Exception:
                    pass
            if handled:
                return None
        return await handle(message)

    client._handle_message = handle_message


# ---------------------------------------------------------------------------
# Playing
# ---------------------------------------------------------------------------

async def drive(player: Any, args: argparse.Namespace, user: str) -> int:
    """Search, play and report until --games games are done or the stop file appears.
    Runs in poke-env's own event loop, next to the player."""
    client = player.ps_client
    me = to_id(user)
    ckpt = os.path.basename(args.checkpoint)
    app_event("phase", text=f"logging in as {user}")
    refused = await first_login(client, args, user)
    if refused is not None:
        app_event("error", text=f"could not log in as {user}: " + (getattr(args, "_login_refused", None)
                                                                   or "check the account name and the password"))
        return refused
    login = login_state(client)
    login_refused: Optional[str] = None     # a login after a reconnect refused with a reason: the run ends

    earlier = load_games(OUT, args.server, me)
    earlier_trail = ratings_of(earlier)
    app_event("start", checkpoint=ckpt, device=str(player.policy.device), user=user, server=args.server,
              parallel=args.parallel, games=args.games, player=args.player, arms=args.arms,
              search_ms=args.search_ms if getattr(player, "_ladder_bridge", None) is not None else None,
              helpers=args.helpers if getattr(player, "_ladder_bridge", None) is not None else None,
              deadline=args.deadline, earlier_games=len(earlier),
              earlier_rating=earlier_trail[-1] if earlier_trail else None)
    print(f"network: {ckpt} on {player.policy.device}   account: {user}   server: {args.server}   format: {args.format}")
    print(f"up to {args.parallel} games at a time, {args.think_ms} ms before each move, battle timer {'on' if args.timer else 'off'}")
    modes = getattr(player, "_ladder_modes", None)
    bridge = getattr(player, "_ladder_bridge", None)
    recorder = getattr(player, "_ladder_recorder", None)
    if modes:
        how = (("arms " + ", ".join(modes) + " (one per game, balanced; experiment " + str(experiment_tag(args)) + ")")
               if args.arms else ("player " + modes[0]))
        settings = getattr(args, "arm_settings", None)
        if settings:
            how = (f"{len(settings)} settings, one per game (balanced: the fewest games first, one with no game in progress "
                   f"where it can; experiment {experiment_tag(args)}):\n"
                   + "\n".join(f"  {s.name}: {s.text()}" for s in settings) + "\n")
        if "mix-deep" in modes and not settings:
            b = deep_budget(args)
            how += (f"; mix-deep searches {b.search_ms} ms per world on {b.worlds_x}x the worlds"
                    + (f", {b.threads} threads" if b.threads > 1 else "")
                    + (f", {b.workers} worlds at a time" if b.workers > 1 else "") + f", answers within {b.deadline} s")
        where = ""
        if bridge is not None:
            where = (f" on {bridge.remote_host} (round trip {bridge.round_trip_ms() or 0:.0f} ms)" if bridge.remote_host
                     else (" inside WSL" if bridge.where == "wsl" else " here"))
        search = (f"   search {args.search_ms} ms on {args.helpers} helpers{where}, answer within {args.deadline} s of the request, "
                  if not settings else f"search on {args.helpers} helpers{where}, each game's budget and trust its setting's, ")
        print(f"{how}" + (search + f"mix rule {args.mix_rule}" + (", guard on" if getattr(args, "guard", False) else "")
                          + (f", guarded: only moves the network gives {args.trust:g} or more"
                             if float(getattr(args, "trust", 0.0) or 0.0) > 0 and not settings else "")
                          + setup_fix_text(args)
                          + (", predictions shown" if getattr(args, "predictions", False) else "") if bridge is not None else ""))
    luna = getattr(player, "luna", None)
    deciders: Dict[str, Any] = dict(getattr(player, "deciders", None) or {})
    asked_models = decider_models(args) if modes else []
    if "luna" in asked_models:
        print(luna_settings_text(args) if luna is not None else "Luna: no key was given, so the luna games are played by mix")
    if "sol" in asked_models:
        print(sol_settings_text(args, deciders["sol"].client) if "sol" in deciders
              else "Sol: no key was given, so the sol games are played by mix")
    if modes and any(m in ("mix-lines", "luna-lines", "sol-lines") for m in modes):
        print(lines_settings_text(args))
    for name in CLAUDE_MODELS:
        if name in asked_models:
            print(claude_settings_text(args, name, deciders[name].client if name in deciders else None))
    if modes and any(m.endswith("-solo") for m in modes):
        print("the model ALONE (" + ", ".join(m for m in modes if m.endswith("-solo")) + "): no search, and the network's pick is never played. The model's choice is the "
              "order; when it gives none a rule plays (the strongest attack by power and type, the healthiest switch-in). "
              "Untested for strength: expect it far below today's bot.")
    if earlier:
        print(f"earlier on this account: {summary_text(earlier)}")
    print(f"to stop: create the file {args.stop_file} (no new games, the ones in progress are finished)\n", flush=True)

    seen = player._ladder_seen
    announce: Dict[str, int] = {}        # battle tag -> its number, until the app has been told of it
    active: Dict[str, float] = {}        # battle tag -> when it started
    done_at: Dict[str, float] = {}       # battle tag -> when it was first seen finished
    known: set = set()
    trail: List[int] = []                # this run's ratings; --history has all of them
    played: List[Dict[str, Any]] = []    # this run's result lines, when games have arms
    started = wins = losses = 0
    first_rating = last_rating = None
    searching_since: Optional[float] = None
    abandoned: Optional[List[str]] = None   # the battles in progress when the bot gave up reconnecting
    search_sent: Optional[float] = None
    cancel_at: Optional[float] = None
    stopping = False
    health = getattr(player, "_ladder_health", None)
    console = getattr(player, "_ladder_console", None)
    say = console.say if console is not None else (lambda text: print(text, flush=True))
    # The link to search helpers on another machine (watch_search): while it is down no game is
    # started; down longer than --link-down-limit seconds and the run starts no more.
    remote_search = bool(bridge is not None and getattr(bridge, "remote_host", None))
    link_limit = float(getattr(args, "link_down_limit", 180.0) or 0.0)
    link: Dict[str, Any] = {"down_at": None, "why": "", "times": 0, "down_s": 0.0}
    helper_restarts: Dict[str, Any] = {"n": 0, "at": -1e18}   # --helper-restarts: the search helpers started again by the health rule

    def search_dropped(now: float) -> bool:
        guard = getattr(player, "_order_guard", None)
        return search_was_dropped(getattr(guard, "no_room_notice_wall", None), search_sent, now,
                                  getattr(player, "server_searching", None), args.format)
    try:
        from search_player import short_tag
    except Exception:
        def short_tag(tag: str) -> str:
            return str(tag or "?")[-4:]
    # The results file and the graph are written by a thread of their own, one game after the
    # other: rewriting the graph reads every recorded game, which must not hold up the moves of
    # the games still being played (review of 2026-10-06, finding 8b).
    import concurrent.futures

    writer = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="ladder-results")

    def write_result(line: Dict[str, Any]) -> None:
        try:
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(line) + "\n")
        except OSError as exc:
            print(f"could not write {OUT}: {exc}", flush=True)
        write_graph(args.server, me)

    def report(tag: str, b: Any) -> None:
        nonlocal wins, losses, first_rating, last_rating
        gone = getattr(player, "rooms_gone", {}).get(tag)
        won = bool(b.won)
        if gone is None:
            wins += won
            losses += (not won)
        rating = seen.get(tag, {}).get("rating")
        if rating:
            first_rating = first_rating if first_rating is not None else rating[0]
            last_rating = rating[1]
            if not trail:
                trail.append(rating[0])
            trail.append(rating[1])
        opp = getattr(b, "opponent_username", None) or "?"
        line = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "battle": tag, "won": won, "user": me,
            "turns": b.turn, "opponent": opp, "opponent_rating": getattr(b, "opponent_rating", None),
            "rating_before": rating[0] if rating else getattr(b, "rating", None), "rating_after": rating[1] if rating else None,
            "checkpoint": ckpt, "seconds": round(done_at[tag] - active[tag], 1), "server": args.server,
            "parallel": args.parallel, "timer": bool(seen.get(tag, {}).get("timer")),
        }
        if tag in getattr(player, "rejoined_battles", ()):
            line["rejoined"] = True            # picked up again after a lost connection
        ending = seen.get(tag, {}).get("ending")
        if ending:
            line["ending"], line["ending_by"] = ending.get("how"), ending.get("by")   # a forfeit or a loss on time, and whose
        guard = getattr(player, "_order_guard", None)
        if guard is not None:
            try:
                line.update(guard.game_counts(tag))     # the typing limit: notices, orders sent again, the spacing's waits
            except Exception:
                pass
        if gone is not None:
            line["won"] = None                 # the room was gone when the bot came back: no result seen
            line["lost_track"] = gone
        arm = ""
        if modes:
            # The game's arm and what the search did in it (search_player.game_summary).
            try:
                summary = player.game_summary(tag) or {}
                ms = args.search_ms if bridge is not None else None
                if ms is not None and isinstance(summary.get("setting_budget"), dict):
                    ms = int(summary["setting_budget"]["ms"])     # --arm-settings: the game's setting's time per world
                line.update(arm_fields(summary, getattr(b, "player_role", None), ms, args.mix_rule))
                if args.arms or getattr(args, "arm_settings", None):
                    line["experiment"] = experiment_tag(args)       # an A/B's games: arm_counts, the scan's --experiment, ab_report
                if bridge is not None:
                    # Where the search ran ("remote": another machine over ssh, "wsl", "local"), and the
                    # round trip to it: measured when the helpers started, and now (the probes).
                    line["search_at"] = bridge.where
                    if bridge.remote_host:
                        line["search_host"] = bridge.remote_host.split("@")[-1]
                        line["rtt_ms"] = (bridge.rtt_start or {}).get("rtt_ms")
                        line["rtt_ms_now"] = bridge.round_trip_ms()
                played.append(line)
                arm = f"   [{summary.get('arm')}" + (f", search {summary.get('searched')}/{summary.get('decisions')}"
                                                     if summary.get("asked") or summary.get("fallbacks") else "")
                if summary.get("pred_scored"):
                    arm += (f", likeliest right {summary.get('pred_top1_right')}/{summary.get('pred_scored')}"
                            f", top 3 {summary.get('pred_top3_right')}/{summary.get('pred_scored')}")
                if summary.get("guard_fired"):
                    arm += f", guard {summary.get('guard_fired')} ({summary.get('guard_changed')} changed)"
                if summary.get("lines_read") is not None:
                    arm += f", lines read {summary.get('lines_read')}" + (
                        f" ({summary.get('lines_changed')} changed the move)" if summary.get("arm") == "mix-lines" else "")
                if summary.get("luna_decisions") is not None:
                    fell = sum((summary.get("luna_fallbacks") or {}).values())
                    arm += (f", {model_label(summary.get('luna_model') or summary.get('arm'))} answered "
                            f"{summary.get('luna_answered')}/{summary.get('luna_decisions')}"
                            f", changed {summary.get('luna_changed')}"
                            + (f", overruled {summary.get('luna_overruled')}" if summary.get("luna_overruled") else "")
                            + (f", mix instead {fell}" if fell else ""))
                arm += "]"
                player.forget_decisions(tag)
            except Exception:
                pass
        writer.submit(write_result, dict(line))
        if recorder is not None:
            recorder.write(tag, "note", {"result": line})
            recorder.close(tag)
        app_event("game_end", battle=tag, won=won, turns=b.turn, opponent=opp, opponent_rating=line["opponent_rating"],
                  rating_before=line["rating_before"], rating_after=line["rating_after"], wins=wins, losses=losses,
                  seconds=line["seconds"], arm=line.get("arm"), searched=line.get("searched"), decisions=line.get("decisions"),
                  fallbacks=line.get("fallbacks"), overruled=line.get("overruled"),
                  decision_ms=line.get("decision_ms_mean"), iterations=line.get("iterations_per_decision"))
        change = f"   rating {rating[0]} -> {rating[1]} ({rating[1] - rating[0]:+d})" if rating else ""
        if gone is not None:
            say(f"{tag}: the game ended while the bot was away (its room was gone: {gone}); no result seen, not counted")
            return
        # The game's tag first, as on its prediction lines: with games at a time, the way to
        # tell which lines were this game's (review of 2026-10-07, finding 5).
        say(f"[{short_tag(tag)}] game {wins + losses:3d}: {'WIN ' if won else 'LOSS'} in {b.turn:2d} turns vs {opp}"
            f"{' (' + str(line['opponent_rating']) + ')' if line['opponent_rating'] else ''}{change}"
            f"   record {wins}-{losses}{arm}")
        if trail:
            say(f"                 elo: {trail_text(trail)}   (peak {max(trail)})")
        if health is not None and health.total and (wins + losses) % 5 == 0:
            say(f"                 search: {health.text()}"
                + (f", round trip to {bridge.remote_host} {bridge.round_trip_ms() or 0:.0f} ms" if bridge is not None and bridge.remote_host else ""))

    async def reconnect() -> bool:
        """
        The connection is gone (the websocket closed, or poke-env stopped listening). Connect and
        log in again, the same way as at the start (poke-env's listen(): the password, when there
        is one, is the one held in memory since the start), for --reconnect-for seconds: every
        second for the first 90 s, then 2, 4, 8 and 15 s apart. Then join every battle that is not
        finished again, and any other the server says this account is in: the server sends each of
        them again from |init| and then its current request, which the bot answers (LadderPlayer).
        A connection that goes again before that is done is made again at once. Nothing new is
        started meanwhile. False: no connection within --reconnect-for (the battles still in
        progress are in `abandoned`).
        """
        nonlocal searching_since, search_sent, cancel_at, abandoned, login_refused
        lost = time.time()
        # Every battle poke-env has that is not over, also one that began a moment before the drop
        # and is not among the drive loop's games yet (review of 2026-10-07, finding 3).
        unfinished = lambda: sorted(t for t, b in list(player.battles.items()) if not b.finished)   # noqa: E731
        playing_now = unfinished()
        print(f"connection lost at {hms(lost)}, reconnecting (attempt 1)..."
              + (f" {len(playing_now)} game{'s' if len(playing_now) != 1 else ''} in progress" if playing_now else ""), flush=True)
        attempt, delay = 0, 1.0
        while True:
            client.logged_in.clear()
            while True:
                attempt += 1
                since = time.time() - lost
                # A game in progress has about 60 s from the moment the SERVER saw the drop (its
                # disconnect timer runs alongside the turn timer, not on top of it): for the first
                # 90 s a connection is tried every second (a refused one costs nothing; the login
                # is only sent once a socket is open), only then less often (review, finding 1).
                if attempt > 1 and (attempt <= 3 or attempt % 10 == 0 or since >= FAST_RETRY_FOR):
                    print(f"reconnecting (attempt {attempt}, {since:.0f} s after the connection was lost)...", flush=True)
                connecting = time.time()
                task = asyncio.ensure_future(client.listen())
                client._listening_coroutine = task
                # Until logged in, the connection gone, this attempt's login failed (log_in_checked:
                # no answer, a web page), or min(30, --login-timeout) s.
                limit = connecting + min(30.0, float(args.login_timeout))
                while (time.time() < limit and not task.done() and not client.logged_in.is_set() and not login.refused
                       and not (login.failed_at is not None and login.failed_at >= connecting)):
                    await asyncio.sleep(0.05)
                if client.logged_in.is_set() and not task.done():
                    break
                if login.refused:
                    # Refused with a reason (the login server's, or Showdown's |nametaken|): not tried
                    # again with the same password, every second for minutes.
                    playing_now = unfinished()
                    login_refused = login.refused
                    print(f"{hms()} the login after the lost connection was refused: {login.refused}. Not trying again: stopping"
                          + (f" ({len(playing_now)} game{'s' if len(playing_now) != 1 else ''} in progress abandoned: "
                             f"{', '.join(playing_now)})" if playing_now else ""), flush=True)
                    abandoned = playing_now
                    try:
                        await asyncio.wait_for(client.websocket.close(), timeout=5.0)
                    except Exception:
                        pass
                    if not task.done():
                        task.cancel()
                    return False
                if not task.done():
                    # Connected but not logged in in time: closed, and tried again.
                    try:
                        await asyncio.wait_for(client.websocket.close(), timeout=5.0)
                    except Exception:
                        pass
                    if not task.done():
                        task.cancel()
                delay = 1.0 if time.time() - lost < FAST_RETRY_FOR else min(15.0, max(2.0, delay * 2.0))
                if time.time() - lost + delay > args.reconnect_for:
                    playing_now = unfinished()
                    print(f"no connection to the server within {args.reconnect_for:.0f} s of losing it ({attempt} attempts): stopping"
                          + (f" ({len(playing_now)} game{'s' if len(playing_now) != 1 else ''} in progress abandoned: "
                             f"{', '.join(playing_now)})" if playing_now else ""), flush=True)
                    abandoned = playing_now
                    return False
                await asyncio.sleep(delay)
            print(f"reconnected and logged in at {hms()}, {time.time() - lost:.1f} s after the connection was lost "
                  f"({attempt} attempt{'s' if attempt != 1 else ''})", flush=True)

            # The server's word on this account comes with the login (|updatesearch|): wait a moment
            # for it. The new connection is watched all along: when it goes too, back to connecting
            # at once, not seconds spent waiting on a dead socket (review, finding 4).
            until = time.time() + 1.0
            while time.time() < until and not task.done():
                await asyncio.sleep(0.05)
            prefix = f"battle-{args.format}-"
            rooms = set(unfinished())
            rooms |= {g for g in getattr(player, "server_games", ())    # also one that started while the bot was away
                      if g.startswith(prefix) and not (g in player.battles and player.battles[g].finished)}
            for g in sorted(rooms):
                if g not in player.battles:
                    player.mark_away(g)
            for attempt_join in (1, 2):
                if task.done():
                    break
                waiting = sorted(r for r in rooms if player.replayed.get(r, 0.0) < connecting and r not in player.rooms_gone)
                if not waiting:
                    break
                for room in waiting:
                    await client.send_message(f"/join {room}")
                until = time.time() + 8.0
                while (time.time() < until and not task.done()
                       and any(player.replayed.get(r, 0.0) < connecting and r not in player.rooms_gone for r in waiting)):
                    await asyncio.sleep(0.1)
            if task.done():
                print(f"connection lost again at {hms()}, {time.time() - lost:.1f} s after the first loss: reconnecting", flush=True)
                continue
            break
        back = sorted(r for r in rooms if player.replayed.get(r, 0.0) >= connecting)
        gone = sorted(r for r in rooms if r in player.rooms_gone)
        missing = sorted(r for r in rooms if r not in back and r not in gone)
        print(f"rejoined {len(back)} battle{'s' if len(back) != 1 else ''} at {hms()}, {time.time() - lost:.1f} s after the connection was lost"
              + (f"; {len(gone)} had ended (room gone)" if gone else "")
              + (f"; NOT ANSWERING: {', '.join(missing)}" if missing else ""), flush=True)
        for room in back:
            b = player.battles.get(room)
            left = seen.get(room, {}).get("time_left")
            print(f"  {room}: turn {getattr(b, 'turn', '?')}" + (", over" if getattr(b, "finished", False) else "")
                  + (f", timer: {left}" if left else ""), flush=True)
        # A ladder search the server still holds goes on; one it dropped is asked again by the loop.
        if args.format in getattr(player, "server_searching", []):
            searching_since = search_sent = time.time()
        else:
            searching_since = search_sent = None
        cancel_at = None
        return True

    async def watch_search() -> None:
        """The search helpers all gone (WSL restarting, say) and back again, told as it happens: a
        task of its own, so that it is also told while the connection is being made again. Remote
        helpers: the link to them down and back (FoulPlayBridge.link_status; the main loop starts
        no game meanwhile), and the decisions that failed while it was down taken out of the
        search's health (SearchHealth.link_spans)."""
        down_at: Optional[float] = None
        while True:
            await asyncio.sleep(0.2)
            try:
                now = time.time()
                if remote_search:
                    status = bridge.link_status()
                    if health is not None:
                        health.link_spans(bridge.link_spans())
                    if status is not None and link["down_at"] is None:
                        link["down_at"], link["why"] = now, str(status.get("why") or "")
                        link["times"] += 1
                        print(f"search link down at {hms(now)} ({link['why']}): playing the network's moves, "
                              f"no new games until it is back", flush=True)
                    elif status is None and link["down_at"] is not None:
                        print(f"search link back at {hms(now)}, {now - link['down_at']:.0f} s after it went down: "
                              f"{bridge.helpers_up()} of {args.helpers} helpers up"
                              + ("" if stopping else ", starting games again"), flush=True)
                        link["down_s"] += now - link["down_at"]
                        link["down_at"] = None
                    continue
                up = bridge.helpers_up()
                if up == 0 and down_at is None:
                    down_at = now
                    print(f"search helpers down at {hms(now)}: playing the network's moves until they are back", flush=True)
                elif up > 0 and down_at is not None:
                    print(f"search helpers back at {hms(now)}: {up} of {args.helpers} up, {now - down_at:.0f} s after they went down",
                          flush=True)
                    down_at = None
            except Exception:
                pass

    watcher = asyncio.ensure_future(watch_search()) if bridge is not None else None
    while True:
        now = time.time()
        listening = getattr(client, "_listening_coroutine", None)
        if listening is not None and listening.done():
            if (any(t not in done_at for t in active) or any(not b.finished for b in list(player.battles.values()))
                    or not (stopping or started >= args.games)):
                if not await reconnect():
                    # Given up: each game still in progress gets a line of its own in the results
                    # (no result seen), and the run ends with an error (review, finding 6). A game
                    # that had ended and waited for its rating line is reported as it is.
                    for tag in list(active):
                        if player.battles[tag].finished:
                            done_at.setdefault(tag, time.time())
                            report(tag, player.battles[tag])
                            del active[tag]
                    for tag in abandoned or []:
                        b = player.battles.get(tag)
                        line = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "battle": tag, "won": None,
                                "user": me, "turns": getattr(b, "turn", None),
                                "opponent": getattr(b, "opponent_username", None) or "?", "checkpoint": ckpt,
                                "server": args.server, "parallel": args.parallel,
                                "abandoned": f"no connection to the server within {args.reconnect_for:.0f} s"}
                        if tag in getattr(player, "rejoined_battles", ()):
                            line["rejoined"] = True
                        writer.submit(write_result, dict(line))
                        if recorder is not None:
                            recorder.write(tag, "note", {"result": line})
                    app_event("error", text=f"the connection to the server was lost and did not come back within "
                                            f"{args.reconnect_for:.0f} s")
                    break
                continue
            # Nothing to pick up again and nothing to start: the games that wait for their rating
            # line are reported (without it) and the run ends as usual.

        for tag in list(player.battles):
            if tag not in known:
                known.add(tag)
                active[tag] = now
                started += 1
                searching_since = search_sent = None
                announce[tag] = started

        # The app hears of a game once its opponent is known (a moment after the room opens).
        for tag in list(announce):
            b = player.battles.get(tag)
            if b is None or getattr(b, "opponent_username", None) or now - active.get(tag, now) > 5.0 or b.finished:
                app_event("game_start", battle=tag, n=announce.pop(tag),
                          opponent=getattr(b, "opponent_username", None) if b is not None else None,
                          rating=getattr(b, "rating", None) if b is not None else None,
                          opponent_rating=getattr(b, "opponent_rating", None) if b is not None else None)

        if args.stop_now_file and os.path.exists(args.stop_now_file):
            # Stop at once: the games in progress are left, and the server counts them as lost on time.
            try:
                os.replace(args.stop_now_file, args.stop_now_file + ".used")
            except OSError:
                pass
            playing = sum(1 for t in active if t not in done_at)
            print("stop-now file found: stopping at once" + (f", leaving {playing} games in progress (the server counts them "
                                                             f"as lost on time)" if playing else ""), flush=True)
            app_event("stopping", how="now", playing=playing)
            if searching_since is not None:
                try:
                    await client.send_message("/cancelsearch")
                except Exception:
                    pass
            break

        for tag in list(active):
            b = player.battles[tag]
            if b.finished:
                first = done_at.setdefault(tag, now)
                if "rating" in seen.get(tag, {}) or now - first >= args.rating_wait:
                    report(tag, b)
                    del active[tag]
            elif now - active[tag] > args.game_timeout:
                print(f"{tag}: no result after {args.game_timeout} s, giving it up and stopping")
                del active[tag]
                stopping = True

        if not stopping and os.path.exists(args.stop_file):
            stopping = True
            os.replace(args.stop_file, args.stop_file + ".used")   # so the next run does not stop at once
            playing = sum(1 for t in active if t not in done_at)
            print("stop file found: no new games" + (f", finishing the {playing} in progress" if playing else ""), flush=True)
            app_event("stopping", how="after", playing=playing)
        failed_share = health.share() if (health is not None and modes and args.max_fallback_share > 0) else None
        if (not stopping and failed_share is not None and failed_share > args.max_fallback_share and bridge is not None
                and not remote_search and helper_restarts["n"] < args.helper_restarts
                and now - helper_restarts["at"] > HELPER_RESTART_GAP_S):
            # Helpers that are up but fail every question (the network gone and back, say): start them
            # again once before giving up. Their battles are rebuilt from the lines kept for them, the
            # window starts over, and the rule below stops the run if they keep failing after this.
            helper_restarts["n"] += 1
            helper_restarts["at"] = now
            stopped = bridge.restart_helpers(health.text())
            print(f"search failing: {health.text()} ({100.0 * failed_share:.0f}%). Restarting the {stopped} search helpers "
                  f"(restart {helper_restarts['n']} of at most {args.helper_restarts}) and playing on.", flush=True)
            app_event("phase", text=f"search failing: restarting the search helpers (restart {helper_restarts['n']})")
            health.reset()
            failed_share = None
        if not stopping and failed_share is not None and failed_share > args.max_fallback_share:
            # Playing on would be the network alone under the search's name. Stop starting games.
            stopping = True
            playing = sum(1 for t in active if t not in done_at)
            print(f"STOPPING: {health.text()} ({100.0 * failed_share:.0f}%, more than {100.0 * args.max_fallback_share:.0f}%). "
                  f"No new games" + (f", finishing the {playing} in progress" if playing else "")
                  + ". The search helpers are failing: see the warnings above.", flush=True)
            app_event("stopping", how="health", playing=playing, text=health.text())
        off = next((who for who in deciders.values() if who.client.disabled), None) if asked_models else None
        if not stopping and off is not None:
            # The API refused the key (or the model) in the middle of the run: every game of that model
            # from here on would be mix under its name. The games in progress are finished by mix.
            stopping = True
            playing = sum(1 for t in active if t not in done_at)
            print(f"STOPPING: {off.label} is off ({off.client.disabled}). No new games"
                  + (f", finishing the {playing} in progress with mix's moves" if playing else "") + ".", flush=True)
            app_event("stopping", how="model", playing=playing, text=f"{off.label} is off: {off.client.disabled}")
        link_down = link["down_at"] is not None
        if not stopping and link_down and link_limit > 0 and now - link["down_at"] > link_limit:
            # The games in progress go on with the network's moves; none is given up.
            stopping = True
            playing = sum(1 for t in active if t not in done_at)
            print(f"STOPPING: the link to the search helpers has been down since {hms(link['down_at'])}, more than "
                  f"{link_limit:.0f} s ({link['why']}). No new games"
                  + (f", finishing the {playing} in progress with the network's moves" if playing else "") + ".", flush=True)
        if (stopping or link_down) and searching_since is not None and cancel_at is None:
            await client.send_message("/cancelsearch")   # no new game while the search link is down
            cancel_at = now
        if cancel_at is not None and searching_since is not None and now - cancel_at > 3.0:
            searching_since = search_sent = None   # the search was withdrawn before a game was found
        if not stopping and not link_down and cancel_at is not None and searching_since is None:
            cancel_at = None   # withdrawn while the link was down, which is back: the next one can be withdrawn too

        # One search at a time: the server refuses a second one in the same format. A search
        # is asked again after two minutes in case it was lost.
        playing = sum(1 for t in active if t not in done_at)
        if not stopping and not link_down and playing < args.parallel:
            if searching_since is None and started < args.games:
                await client.search_ladder_game(player._format, player.get_next_team())
                searching_since = search_sent = now
                app_event("searching", started=started, games=args.games)
            elif searching_since is not None and now - search_sent > 120.0:
                await client.search_ladder_game(player._format, player.get_next_team())
                search_sent = now
            elif searching_since is not None and search_dropped(now):
                # The typing limit dropped a message to no room right after the search went out, and the
                # server does not say it is searching: the search was lost, asked again now, not in two minutes.
                say(f"{hms()} the ladder search was dropped by the typing limit: asked again")
                await client.search_ladder_game(player._format, player.get_next_team())
                search_sent = now

        if not active and searching_since is None and (stopping or started >= args.games):
            break
        await asyncio.sleep(0.2)

    if watcher is not None:
        watcher.cancel()
    writer.shutdown(wait=False)
    await asyncio.get_running_loop().run_in_executor(None, writer.shutdown, True)   # every result is on disk
    if console is not None:
        # The last games' lines before the summary (a window still held: up to 30 s, then on).
        await asyncio.get_running_loop().run_in_executor(None, console.flush, 30.0)
        if console.dropped:
            print(f"{console.dropped} lines were not shown (the window was held too long): the predictions are in the battle files, the results in {OUT}")
    rejected = getattr(player, "n_rejected", None)
    total = wins + losses
    print(f"\nfinished: {wins} wins, {losses} losses" + (f" ({100.0 * wins / total:.0f}%)" if total else "")
          + (f", rating {first_rating} -> {last_rating}" if last_rating is not None else "")
          + (f", orders rejected by the server: {rejected}" if rejected else ""))
    guard = getattr(player, "_order_guard", None)
    if guard is not None:
        try:
            print(guard.summary())
        except Exception:
            pass
    app_event("finished", wins=wins, losses=losses, first_rating=first_rating, last_rating=last_rating,
              abandoned=sum(1 for t in active if t not in done_at))
    if modes:
        if played:
            print("by arm:\n" + arms_text(played))
            if getattr(args, "arm_settings", None):
                print("by setting:\n" + arms_text(played, key=lambda d: str(d.get("setting") or "(none)")))
            told = predictions_text(played)
            if told:
                print("predictions of the opponent's action" + told[len(", predictions"):])
        try:
            text = player.search_summary()
            if text:
                print(text)
            audit = getattr(player, "audit", None) or {}
            print(f"orders the server refused as invalid: {audit.get('invalid_choices', 0)}, decisions that raised: "
                  f"{audit.get('exceptions', 0)}, random orders forced: {audit.get('forced_random', 0)}")
            endings = list((getattr(player, "audit_endings", None) or {}).values())
            print(f"games the server ended for inactivity: {endings.count('inactivity')}, by a forfeit: {endings.count('forfeit')}")
            if bridge is not None:
                stats = bridge.stats()
                recent = stats.get("recent_iterations_per_decision", 0)
                print(f"search helpers: {stats.get('decisions', 0)} decisions answered, {recent} iterations per decision lately, "
                      f"{stats.get('expired', 0)} not started for lack of time, {stats.get('helper_exits', 0)} helper exits"
                      + (f". BELOW {LOW_ITERATIONS}: the machine is too busy for this search time" if 0 < recent < LOW_ITERATIONS else ""))
                if remote_search:
                    down_s = link["down_s"] + ((time.time() - link["down_at"]) if link["down_at"] is not None else 0.0)
                    print(f"search link: down {link['times']} time{'s' if link['times'] != 1 else ''}, {down_s:.0f} s in all; "
                          f"{getattr(health, 'link_failed', 0)} decisions fell back to the network meanwhile (not counted as "
                          f"search failures); {stats.get('link_late', 0)} lines held up by the network")
        except Exception:
            pass
    if recorder is not None:
        recorder.close_all()
        if recorder.failed:
            print(f"{recorder.failed} lines could not be written to {recorder.folder}")
    lost_games = (f"{len(abandoned)} game{'s' if len(abandoned) != 1 else ''} abandoned, recorded with no result"
                  if abandoned is not None else "")
    if login_refused is not None:
        print(f"stopped: the login after a lost connection was refused ({login_refused.rstrip('.')}); {lost_games}")
        write_exit_note(args, "password" if login.refused_kind == "password" else "login",
                        f"the login after a lost connection was refused: {login_refused.rstrip('.')}; {lost_games}", client)
    elif abandoned is not None:
        print(f"stopped without a connection to the server: {lost_games}")
        write_exit_note(args, "connection", f"no connection to the server within {args.reconnect_for:.0f} s of losing it; "
                                            f"{lost_games}", client)
    elif asked_models and any(who.client.disabled for who in deciders.values()):
        # The API refused the key in the middle of the run: the games were finished by mix, and the run says so.
        off = next(who for who in deciders.values() if who.client.disabled)
        print(f"stopped early: {off.label} was turned off in the middle of the run ({off.client.disabled}); "
              f"{started} of up to {args.games} games were started")
        write_exit_note(args, "luna", f"{off.label} was turned off in the middle of the run: {off.client.disabled}; "
                                      f"{wins} wins, {losses} losses, {started} of up to {args.games} games started", client)
        sys.stdout.flush()
        return LUNA_FAILED
    else:
        write_exit_note(args, "done", f"{wins} wins, {losses} losses, {started} of up to {args.games} games started", client)
    sys.stdout.flush()
    return 4 if abandoned is not None else 0


async def run(args: argparse.Namespace, user: str, password: Optional[str]) -> int:
    player = make_player(args, user, password)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    from poke_env.concurrency import handle_threaded_coroutines

    try:
        return await handle_threaded_coroutines(drive(player, args, user), player.ps_client.loop)
    finally:
        bridge = getattr(player, "_ladder_bridge", None)
        if bridge is not None:
            bridge.close()   # stops the search helpers (inside WSL, or on the box over ssh, as well)
        closing = list((getattr(player, "deciders", None) or {}).values())
        if getattr(player, "luna", None) is not None and getattr(player, "luna") not in closing:
            closing.append(player.luna)
        for who in closing:
            who.client.close()


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    global OUT, GRAPH, BATTLES, APP_EVENTS
    # The password is read once and taken out of this process's environment at once, so that
    # nothing it starts (the search helpers' wsl.exe, a graph) inherits it.
    env_password = os.environ.pop("PS_PASSWORD", None)
    APP_EVENTS = bool(args.app_events)
    if args.out:
        OUT = os.path.abspath(args.out)
        GRAPH = os.path.join(os.path.dirname(OUT), "elo.html")
        BATTLES = os.path.join(os.path.dirname(OUT), "battles")
    args.battles_dir = os.path.abspath(args.battles_dir) if args.battles_dir else BATTLES
    if args.exit_file:
        args.exit_file = os.path.abspath(args.exit_file)

    user = args.user or os.environ.get("PS_USERNAME")
    if args.history:
        return print_history(OUT, args.server, to_id(user) if user else None)
    if args.graph:
        return show_graph(args.server, to_id(user) if user else None, args.once)
    if args.luna_check:
        return luna_check_only(args)

    if args.checkpoint is None:
        args.checkpoint = newest_checkpoint(os.path.join(HERE, "runs", "vm", "checkpoints"))
    if not args.checkpoint or not os.path.isfile(args.checkpoint):
        raise SystemExit("no checkpoint found: give --checkpoint")
    if not user:
        raise SystemExit("give the account name with --user or PS_USERNAME")
    if args.password_stdin:
        # The window app: the password is the first line of standard input (the API key of a model
        # arm the next one, --key-stdin), so it is read before anything else is.
        stdin_password = read_password_stdin()
        if stdin_password:
            env_password = stdin_password
        del stdin_password
    # One bot per account: the window app (PSBot Lookahead.exe) and a command line run the same
    # ladder.py, so two of them cannot play the same account at once (two bots answer the same games).
    refused = take_account_lock(args.server, to_id(user))
    if refused is not None:
        print(refused, flush=True)
        app_event("error", text=refused)
        write_exit_note(args, "lock", refused)
        return 3
    args._luna = None
    if wants_luna(args):
        # Before the Showdown password and before anything is started: no key, or a key that does
        # not work, and nothing is played.
        try:
            code = luna_ready(args)
        except KeyboardInterrupt:
            return stopped_by_ctrl_c(args)
        if code:
            return code
    password: Optional[str] = None
    try:
        if args.server == "public":
            password = env_password
            if not password and not args.password_stdin:
                password = getpass.getpass(f"Showdown password for {user} (not shown, not stored): ")
            if not password:
                write_exit_note(args, "password", "no password was typed")
                app_event("error", text="a password is needed for the public server")
                raise SystemExit("a password is needed for the public server")
        else:
            if env_password:
                print("password: given, but a local server needs none: it is not used and nothing is sent to the login server")
            if args.ask_password:
                typed = getpass.getpass(f"Showdown password for {user} (not shown, not stored; a local server needs none, so it is dropped): ")
                print(f"password prompt answered ({'something' if typed else 'nothing'} typed); a local server: not used")
                del typed
    except KeyboardInterrupt:
        return stopped_by_ctrl_c(args)
    del env_password
    if args.stop_now_file:
        args.stop_now_file = os.path.abspath(args.stop_now_file)
    args.stop_file = os.path.abspath(args.stop_file)
    os.chdir(HERE)
    if args.console_log:
        console = ConsoleLog(os.path.dirname(OUT), secret=password, more=luna_secrets())
        console.line("=" * 20 + f" ladder.py started (pid {os.getpid()}): " + " ".join(sys.argv[1:] if argv is None else argv))
        sys.stdout = console.wrap(sys.stdout, "out")
        sys.stderr = console.wrap(sys.stderr, "err")
        print(f"console log: {console.path()}", flush=True)
    try:
        return asyncio.run(run(args, user, password))
    except KeyboardInterrupt:
        return stopped_by_ctrl_c(args)


def luna_secrets() -> List[str]:
    """The texts of the Luna arm that must never be shown or written (the API key), for the console log."""
    try:
        import luna_player as lp

        return list(lp._SECRETS)
    except Exception:
        return []


def openai_key(args: argparse.Namespace) -> str:
    """
    The OpenAI key: with --key-stdin the next line of standard input (the window app writes it
    after the password's line; an empty line means OPENAI_API_KEY), else OPENAI_API_KEY, else typed
    once at a hidden prompt. The variable is taken out of this process's environment at once, so
    that nothing this run starts (the search helpers) inherits it. The key is returned to the
    caller and kept nowhere else. luna_player.LunaError when there is no usable key.
    """
    import luna_player as lp

    try:
        if getattr(args, "key_stdin", False):
            typed = read_password_stdin()
            key = lp.read_key(environ={lp.KEY_ENV: typed} if typed else dict(os.environ), ask=lambda prompt: "")
            del typed
        else:
            key = lp.read_key()
    finally:
        os.environ.pop(lp.KEY_ENV, None)
    return key


def anthropic_key(args: argparse.Namespace, from_stdin: bool) -> str:
    """
    The Anthropic key, as openai_key gives OpenAI's: with from_stdin the next line of standard
    input (the window app writes it after the password's line; an empty line means
    ANTHROPIC_API_KEY), else ANTHROPIC_API_KEY, else typed once at a hidden prompt (never under
    --key-stdin, where nobody is at a terminal). The variable is taken out of this process's
    environment at once. luna_player.LunaError when there is no usable key.
    """
    import claude_player as cp

    try:
        if from_stdin:
            typed = read_password_stdin()
            key = cp.read_key(environ={cp.KEY_ENV: typed} if typed else dict(os.environ), ask=lambda prompt: "")
            del typed
        elif getattr(args, "key_stdin", False):
            key = cp.read_key(ask=lambda prompt: "")
        else:
            key = cp.read_key()
    finally:
        os.environ.pop(cp.KEY_ENV, None)
    return key


def model_keys(args: argparse.Namespace, models: List[str]) -> Dict[str, str]:
    """
    {"openai": key, "anthropic": key} for the companies whose models this run asks. Under
    --key-stdin there is one line: it is the key of the one company the run needs, and when it
    needs both it is OpenAI's (Anthropic's then comes from ANTHROPIC_API_KEY).
    """
    keys: Dict[str, str] = {}
    openai = any(name in OPENAI_MODELS for name in models)
    if openai:
        keys["openai"] = openai_key(args)
    if any(name in CLAUDE_MODELS for name in models):
        keys["anthropic"] = anthropic_key(args, from_stdin=bool(getattr(args, "key_stdin", False)) and not openai)
    return keys


def model_timeout(args: argparse.Namespace, name: str) -> float:
    """The seconds a game gives one answer of this model."""
    if name in CLAUDE_MODELS:
        return float(args.claude_timeout)
    return float(args.sol_timeout if name == "sol" else args.luna_timeout)


def model_client(args: argparse.Namespace, name: str, key: str) -> Any:
    """
    The client of one model with the key: "luna" (the Decisions API, --luna-model at --luna-url),
    "sol" (the Responses API, --sol-model at --sol-url with --sol-effort) or one of Claude's
    (claude_player: the Messages API at --claude-url with --claude-effort). luna_player.LunaError
    when the address is one the key must not be sent to.
    """
    import luna_player as lp
    from dataclasses import replace as _replace

    if name in CLAUDE_MODELS:
        import claude_player as cp

        spec = _replace(cp.MODELS[name], url=str(args.claude_url), timeout=float(args.claude_timeout))
        return cp.ClaudeClient(key, spec, effort=str(args.claude_effort) if spec.effort else "")
    if name == "sol":
        spec = _replace(lp.MODELS["sol"], model=str(args.sol_model), url=str(args.sol_url), effort=str(args.sol_effort),
                        timeout=float(args.sol_timeout))
    else:
        spec = _replace(lp.MODELS["luna"], model=str(args.luna_model), url=str(args.luna_url), timeout=float(args.luna_timeout))
    return lp.LunaClient(key, url=spec.url, model=spec.model, api=spec.api, effort=spec.effort, label=spec.label, spec=spec)


def luna_client(args: argparse.Namespace) -> Any:
    """Luna's client (the Decisions API) with the key: see openai_key and model_client."""
    return model_client(args, "luna", openai_key(args))


def luna_check_only(args: argparse.Namespace) -> int:
    """--luna-check: one tiny real question, what came back (never the key), exit code 0 or LUNA_FAILED."""
    import luna_player as lp

    name = str(getattr(args, "model", "luna") or "luna")
    try:
        client = model_client(args, name, model_keys(args, [name])["anthropic" if name in CLAUDE_MODELS else "openai"])
    except lp.LunaError as exc:
        print(f"{model_label(name).upper()} CHECK FAILED: {exc.text}", flush=True)
        return LUNA_FAILED
    except KeyboardInterrupt:
        print("\nstopped with Ctrl+C", flush=True)
        return 130
    try:
        ok = lp.luna_check(client, timeout=45.0 if name in CLAUDE_MODELS else 15.0, game_timeout=model_timeout(args, name),
                           out=lambda text: print(lp.redact(text), flush=True)) == 0
    finally:
        client.close()
    return 0 if ok else LUNA_FAILED


def luna_ready(args: argparse.Namespace) -> int:
    """
    A run with the Luna arm: the key, the same check as --luna-check (unless --luna-skip-check), and
    the decider put in args._luna for make_player. 0, or LUNA_FAILED after one line that says why
    (nothing was played).
    """
    import luna_player as lp

    models = decider_models(args) or ["luna"]
    names = " and ".join(model_label(name) for name in models)
    made: Dict[str, Any] = {}

    def failed(text: str, note: str) -> int:
        for who in made.values():
            who.client.close()
        print(text, flush=True)
        app_event("error", text=note)
        write_exit_note(args, "luna", note)
        return LUNA_FAILED

    try:
        keys = model_keys(args, models)
    except lp.LunaError as exc:
        return failed(f"{names} cannot be used: {exc.text}. Nothing was played.", f"{names} cannot be used: {exc.text}")
    try:
        for name in models:
            label = model_label(name)
            timeout = model_timeout(args, name)
            claude = name in CLAUDE_MODELS
            try:
                client = model_client(args, name, keys["anthropic" if claude else "openai"])
            except lp.LunaError as exc:
                return failed(f"{label} cannot be used: {exc.text}. Nothing was played.", f"{label} cannot be used: {exc.text}")
            made[name] = lp.Luna(client, lp.LunaSettings(timeout=timeout, guard=args.luna_guard == "on",
                                                         save_briefs=bool(args.luna_save_briefs)), spec=client.spec)
            if not args.luna_skip_check:
                app_event("phase", text=f"checking the {'Anthropic' if claude else 'OpenAI'} key with {label}")
                if lp.luna_check(client, timeout=45.0 if claude else 15.0, game_timeout=timeout,
                                 out=lambda text: print(lp.redact(text), flush=True)) != 0:
                    return failed(f"{label} cannot be used (the check failed, see the line above). Nothing was played.",
                                  f"{label}'s check failed: " + str(client.last_error or "see the console"))
    finally:
        del keys
    args._deciders = made
    args._luna = made.get("luna")
    return 0


def stopped_by_ctrl_c(args: argparse.Namespace) -> int:
    """
    Ctrl+C (at the password prompt, during the login or in a game): one line and exit code 130,
    not a traceback with exit code 1, and --exit-file says so ("stopped: ..."), so run_box_ab.sh's
    last line says Ctrl+C and not "an error" (box, 2026-10-09: the login wait was cut with Ctrl+C).
    asyncio.run has already cancelled the run (run()'s finally stops the search helpers).
    """
    print(f"\n{hms()} stopped with Ctrl+C (a game in progress is abandoned: the server counts it as lost on time; "
          f"the stop file finishes the games instead)", flush=True)
    write_exit_note(args, "stopped", "it was stopped with Ctrl+C")
    return 130


class _SettingParser(argparse.ArgumentParser):
    """ladder.py's parser for one setting of --arm-settings: an error is a ValueError (for the file's message), never an exit."""

    def error(self, message: str):
        raise ValueError(message)

    def exit(self, status: int = 0, message: Optional[str] = None):
        raise ValueError((message or "").strip() or "not a setting (--help?)")


def build_parser(cls: type = argparse.ArgumentParser) -> argparse.ArgumentParser:
    """ladder.py's command line (parse_args reads it; --arm-settings reads every setting's arguments with it)."""
    ap = cls(description="Play the trained policy on a Showdown ladder.")
    ap.add_argument("--server", default="public", help="'public' (play.pokemonshowdown.com) or 'local:PORT'")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--parallel", type=int, default=3, help="games played at the same time (1 to 8)")
    ap.add_argument("--format", default="gen9randombattle")
    ap.add_argument("--checkpoint", default=None, help="default: the newest file in runs/vm/checkpoints")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda when available)")
    ap.add_argument("--user", default=None, help="account name (default: PS_USERNAME)")
    ap.add_argument("--history", action="store_true", help="print every recorded game and the rating after each, then exit")
    ap.add_argument("--graph", action="store_true", help="open the rating graph in the browser and keep it up to date")
    ap.add_argument("--once", action="store_true", help="with --graph: write the page and exit, without opening it")
    ap.add_argument("--timer", action=argparse.BooleanOptionalAction, default=True,
                    help="ask for the battle timer at the start of each game (default: on)")
    ap.add_argument("--think-ms", type=int, default=400, help="pause before each move, in milliseconds (the network itself answers in well under a second)")
    ap.add_argument("--game-timeout", type=int, default=3600, help="give up waiting for one game after this many seconds")
    ap.add_argument("--rating-wait", type=float, default=8.0, help="seconds to wait for the server's rating line after a game")
    ap.add_argument("--login-timeout", type=int, default=45,
                    help="seconds for the first login's connection, the server's challenge and the login server's answer")
    ap.add_argument("--login-wait", type=float, default=30.0,
                    help="seconds Showdown gets to confirm the first login once the name is sent (a line every 5 s meanwhile)")
    ap.add_argument("--exit-file", default=None,
                    help="write one line there when the run ends: 'KIND: why' (KIND password, login, connection or done; "
                         "nothing secret), for run_box_ab.sh")
    ap.add_argument("--reconnect-for", type=float, default=300.0,
                    help="when the connection to the server is lost: seconds to keep trying to connect again "
                         "(at least 120; the games in progress are joined again once it is back)")
    ap.add_argument("--ping-interval", type=float, default=5.0,
                    help="seconds between websocket keep-alive pings (a dead connection is noticed within interval + timeout)")
    ap.add_argument("--ping-timeout", type=float, default=10.0, help="seconds a keep-alive ping may go unanswered")
    ap.add_argument("--close-timeout", type=float, default=2.0,
                    help="seconds the websocket waits for the server to close a connection it gave up on (the library's own: 10)")
    ap.add_argument("--console-log", action=argparse.BooleanOptionalAction, default=True,
                    help="also write everything printed to console-<local date>.log next to the results file, "
                         "with the time on each line (default: on)")
    ap.add_argument("--stop-file", default=os.path.join(HERE, "ladder.stop"))
    ap.add_argument("--stop-now-file", default=None,
                    help="when this file appears, stop at once: the games in progress are left (lost on time), the helpers stopped")
    ap.add_argument("--app-events", action="store_true",
                    help="also print '@app {json}' lines: what the window app (PSBot Lookahead.exe) shows")
    ap.add_argument("--app-net-view", action="store_true",
                    help="with --app-events: also a 'net' line for every move the network makes, what it looked at "
                         "and which neurons fired (net_viz.py), for the window's network panels")
    ap.add_argument("--player", default="net", choices=PLAYER_MODES,
                    help="net: the network alone (the default, as before). search: the search's own choice. "
                         "mix: the two combined (--mix-rule). mix-veto, mix-pool, mix-shortlist: that rule. "
                         "predict: mix, left only on a confident prediction of the opponent's action (opponent_model.py)")
    ap.add_argument("--arms", default=None,
                    help="e.g. net,mix: every new game is played in one of these modes, balanced, and its arm is written with its result")
    ap.add_argument("--search-ms", type=int, default=500, help="search time per decision (an answer takes about 2.3 times this)")
    ap.add_argument("--helpers", type=int, default=4,
                    help="search helper processes, one core each while searching (1 to %d; one more than --parallel)" % MAX_HELPERS)
    ap.add_argument("--deadline", type=float, default=2.5,
                    help="seconds from the request to the order: a search that has not answered by then is not waited for")
    ap.add_argument("--search-remote", default=None, metavar="TARGET",
                    help="run the search helpers on another machine over ssh: vm (the box cloud/vm.sh talks to, its address "
                         "looked up at the start) or user@host. The player and the network stay here")
    ap.add_argument("--search-key", default=None, help="with --search-remote user@host: the private key (ssh -i)")
    ap.add_argument("--search-ssh", default=None,
                    help="with --search-remote: the ssh program (default: Git's ssh on Windows, the one cloud/vm.sh uses; else ssh)")
    ap.add_argument("--search-known-hosts", default=None, help="with --search-remote: ssh's known-hosts file")
    ap.add_argument("--search-remote-dir", default="~/dev/ab-ladder",
                    help="with --search-remote: that machine's copy of this repository (its bench/fp_bridge_server.py must be this one)")
    ap.add_argument("--mix-rule", default="veto", choices=("veto", "pool", "shortlist"), help="what 'mix' means (search_player.mix)")
    ap.add_argument("--predictions", action=argparse.BooleanOptionalAction, default=True,
                    help="with a search: print the opponent's three likeliest actions for every decision and whether the read was "
                         "right, keep them in the battle files and count them per game (default: on; they change no move)")
    ap.add_argument("--guard", action="store_true",
                    help="mix with the veto rule: the guard against set-up moves and sleep in every game (search_player.MixSettings.guard; "
                         "off by default, --arms mix,mix-guard compares the two)")
    ap.add_argument("--trust", type=float, default=0.0, metavar="P",
                    help="mix with the veto rule: the guarded search in every game, as the arm mix-trust plays it (P 0.03): the search "
                         "picks only among the moves the network gives P or more, and its likeliest (search_player.MixSettings.trust; "
                         "0: off, the default). With --player mix-deep it guards the deep budget: the app's guarded mode, searched wider")
    # The set-up fixes (search_player.MixSettings, bench/notes/setup_fix.md): all off by default.
    ap.add_argument("--veto-fix", action="store_true",
                    help="mix with the veto rule: never an action neither side picked (the search's best when it turned away from "
                         "the network's), a set-up move of ours favoured by the search does not veto an attack the network gives "
                         "--sure-attack or more that is estimated to knock the opponent out (not at full HP, not through a known Assault Vest, "
                         "Eviolite or an ability that could stop it), Tera only when one side picked the Tera form or the search "
                         "vetoed the plain form and not the Tera form")
    ap.add_argument("--sure-attack", type=float, default=0.70,
                    help="with --veto-fix: the network's least probability for its attack in KO range to stand against our own set-up move")
    ap.add_argument("--setup-defer", action="store_true",
                    help="mix with the veto rule: in a set-up position (opponent at +2, boosted after a set-up move, or a set-up read "
                         "of 40%% or more) play the search's best (off by default)")
    ap.add_argument("--setup-switch-sure", type=float, default=0.0,
                    help="with --setup-defer: a switch the network gives at least this is played instead (0: never)")
    # Round 2's set-up rules (search_player.MixSettings; bench/setup_puzzles/ROUND2.md on setupfix): all off by default,
    # each acts only in a set-up position (opponent at +2, boosted after a set-up move, or a set-up read of 40%% or more).
    ap.add_argument("--setup-switch-trust", type=float, default=0.0, metavar="P",
                    help="mix with the veto rule: in a set-up position the network's likeliest action, when it is a switch it gives "
                         "at least P, is never vetoed (0: off, the default)")
    ap.add_argument("--setup-net-trust", type=float, default=0.0, metavar="P",
                    help="mix with the veto rule: in a set-up position the network's likeliest action of any kind is played when it "
                         "gives it at least P (0: off, the default)")
    ap.add_argument("--setup-blend", type=float, default=0.0, metavar="LAMBDA",
                    help="mix with the veto rule: in a set-up position play the largest log p(network) + LAMBDA x the search's share "
                         "over its largest share (0: off, the default)")
    ap.add_argument("--setup-keep", type=float, default=0.0, metavar="K",
                    help="mix with the veto rule: in a set-up position the veto keeps actions with K x the search's largest share "
                         "(in place of 0.25; 0: off, the default)")
    ap.add_argument("--guard-sucker", type=float, default=0.0, metavar="P",
                    help="with --guard: no Sucker Punch when the read puts P or more on a status move (0: the guard's 0.40, the default)")
    # Round 3's counter rules (search_player.MixSettings.setup_counter, setup_other_side; branch counterfix): off by
    # default; the arm "mix-counter" has both (--arms mix,mix-counter alternates it with today's mix).
    ap.add_argument("--setup-counter", action="store_true",
                    help="mix: in a set-up position (opponent at +2, boosted after a set-up move, a set-up read of 50%% or more, or "
                         "+2 in any stat) play our counter that can work (phazing, Haze, Clear Smog, Encore right after a set-up "
                         "move, Taunt before the boost, a sleep move, Perish Song), else switch to the healthiest benched Pokemon "
                         "(50%% HP or more, not two-hit KO'd) holding one or Unaware; not when an attack is estimated to knock it out")
    ap.add_argument("--setup-other-side", action="store_true",
                    help="mix: an attack into a defence the opponent has boosted gives way to an attack on the other defence "
                         "estimated to do at least 0.8 times as much")
    ap.add_argument("--think-longer", type=int, default=0, metavar="MS",
                    help="with a search: search this many ms on set-up turns (opponent boosted or set up last turn, or the "
                         "ordinary answer reads a set-up move at --think-longer-read or more); 0: off (the default)")
    ap.add_argument("--think-longer-read", type=float, default=0.40,
                    help="with --think-longer: the set-up read of the ordinary answer that asks the search again for longer")
    ap.add_argument("--think-longer-boost", type=int, default=1,
                    help="with --think-longer: the opponent's stat stage (attack, defences, speed) that asks for the longer search at once")
    ap.add_argument("--think-longer-threads", type=int, default=2,
                    help="with --think-longer: the engine's threads for the long search when a helper is free (1: never a second)")
    ap.add_argument("--predict-confidence", type=float, default=0.60,
                    help="predict: the mix choice is left only when the opponent's action is predicted with at least this probability")
    ap.add_argument("--predict-gain", type=float, default=0.08,
                    help="predict: and the other action is worth at least this much more against the prediction (0 to 0.5, the engine's scale)")
    ap.add_argument("--predict-risk", type=float, default=0.08,
                    help="predict: and loses at most this much against the opponent's second most likely action")
    ap.add_argument("--predict-once-bitten", action=argparse.BooleanOptionalAction, default=True,
                    help="predict: a read that was wrong against this Pokemon in this battle is not acted on again (default: on)")
    ap.add_argument("--payoff-ms", type=float, default=400.0,
                    help="predict: time on top of the search for the table of values per (our action, the opponent's action)")
    ap.add_argument("--save-battles", action=argparse.BooleanOptionalAction, default=True,
                    help="keep every battle's messages in runs/ladder/battles (default: on)")
    ap.add_argument("--battles-dir", default=None, help="where the battles' messages go (default: battles/ next to the results file)")
    ap.add_argument("--out", default=None, help="the results file (default: runs/ladder/results.jsonl); the graph is written next to it")
    ap.add_argument("--max-fallback-share", type=float, default=0.10,
                    help="with a search: start no more games once the search failed (timeout, expired, error) in more than "
                         "this share of the last --health-window decisions it was asked (0: never)")
    ap.add_argument("--health-window", type=int, default=50)
    ap.add_argument("--helper-restarts", type=int, default=3,
                    help="with a search: when the rule of --max-fallback-share trips, start the search helpers again and "
                         "play on, up to this many times a run and at most once in %d s, before it stops the run (0: stop at once, "
                         "as before)" % int(HELPER_RESTART_GAP_S))
    ap.add_argument("--link-down-limit", type=float, default=180.0,
                    help="with --search-remote: start no more games once the link to the search helpers has been down "
                         "for this many seconds in one go (0: never). While it is down no game is started, the games in "
                         "progress get the network's moves, and the decisions that failed meanwhile do not count for "
                         "--max-fallback-share")
    ap.add_argument("--password-stdin", action="store_true",
                    help="read the password as the first line of standard input (the window app does this), not PS_PASSWORD")
    ap.add_argument("--ask-password", action="store_true",
                    help="ask for the password at the prompt also for a local server, and drop it (a test of the prompt: "
                         "a local server needs none, and nothing is sent to the login server)")
    # A/B runs: the tag of the experiment, and the deep arm's own search budget (search_player.SearchBudget).
    ap.add_argument("--experiment", default=None,
                    help="with --arms: the tag this run's games carry in results.jsonl (\"experiment\"); the balance of the arms "
                         "and bench/ab_report.py count the games of this tag only (default: the arms joined by commas)")
    ap.add_argument("--deep-search-ms", type=int, default=1250,
                    help="the arm mix-deep: search time per sampled world (half early in a game, when twice as many are sampled)")
    ap.add_argument("--deep-worlds", type=int, default=2,
                    help="the arm mix-deep: this many times the sampled opponent teams (Foul Play's parallelism; searched one "
                         "after the other, so each one multiplies the time)")
    ap.add_argument("--deep-threads", type=int, default=1, help="the arm mix-deep: poke-engine threads per world (1: one core, as mix)")
    ap.add_argument("--deep-workers", type=int, default=1,
                    help="the arm mix-deep: its worlds searched this many at a time, each in a process of Foul Play's own pool "
                         "forked from the helper, one core each (1: one after the other in the helper; the same search and "
                         "iterations either way, W at a time takes about 1/W of the time)")
    ap.add_argument("--deep-deadline", type=float, default=7.5,
                    help="the arm mix-deep: seconds from the request to the order (under 10: one tick of the battle timer at most)")
    ap.add_argument("--deep-min-time-left", type=float, default=90.0,
                    help="the arm mix-deep: with fewer seconds left on the battle timer, today's budget for the rest of the game")
    ap.add_argument("--deep-max-decision-s", type=float, default=9.0,
                    help="the arm mix-deep: after a decision slower than this, today's budget for the rest of the game")
    # The Luna arm (luna_player.py): OpenAI's Decisions API picks the action. The key is never a flag:
    # OPENAI_API_KEY, or the hidden prompt.
    ap.add_argument("--luna-check", action="store_true",
                    help="ask OpenAI's Decisions API one tiny real question (is 2+2=4, and a 3-choice) to check the key, "
                         "the endpoint and the answer's shape, then a question of a game's size, print what came back "
                         "(never the key) and exit (0, or 5)")
    ap.add_argument("--luna-timeout", type=float, default=1.2,
                    help="the Luna arm: seconds for the API's answer (and never past --deadline); without one mix's pick is played")
    ap.add_argument("--luna-guard", default="on", choices=("on", "off"),
                    help="the Luna arm: the blunder guard. on (the default): Luna's pick is overruled when it has under 1%% of "
                         "the network's probability and under 1%% of the search's share while mix's pick has over 60%% of either")
    ap.add_argument("--luna-model", default="gpt-6-luna", help="the Luna arm: the model (the only one the API has today)")
    ap.add_argument("--luna-url", default="https://api.openai.com/v1/decisions",
                    help="the Luna arm: the endpoint. OpenAI's, or a test server on this machine (http://127.0.0.1:PORT/v1/decisions): "
                         "the key is sent nowhere else")
    ap.add_argument("--luna-save-briefs", action=argparse.BooleanOptionalAction, default=True,
                    help="the Luna arm: keep the brief and the choices of every question in the decision's note (default: on)")
    ap.add_argument("--model", default="luna", choices=OPENAI_MODELS + CLAUDE_MODELS, help="with --luna-check: the model to check")
    # Claude alone (claude_player.MODELS: the arms fable-solo, opus-solo, sonnet-solo, haiku-solo).
    # The key: ANTHROPIC_API_KEY, the --key-stdin line, or the hidden prompt.
    ap.add_argument("--claude-effort", default="low", choices=("low", "medium", "high"),
                    help="the Claude arms: how much the model thinks (output_config.effort). low keeps a move under about 10 s")
    ap.add_argument("--claude-timeout", type=float, default=25.0,
                    help="the Claude arms: seconds waited for the API's answer; after them the fallback rule plays (not a target: "
                         "the effort is what keeps a move short)")
    ap.add_argument("--claude-url", default="https://api.anthropic.com", help="the Claude arms: the API's address (tests: a local fake)")
    ap.add_argument("--max-model-usd", type=float, default=20.0,
                    help="the model arms: once the run's decisions have cost this many dollars at the API's prices, the models "
                         "are turned off for the rest of the run (mix finishes the games in progress, no new game starts). 0: no cap")
    ap.add_argument("--key-stdin", action="store_true",
                    help="read the model's API key as the next line of standard input (after the password's with --password-stdin; "
                         "an empty line: the environment variable). OpenAI's for Luna and Sol, Anthropic's for the Claude arms. "
                         "The window app gives it this way")
    # The Sol arms (luna_player.MODELS["sol"]): the same question through the Responses API.
    ap.add_argument("--sol-model", default="gpt-6-sol", help="the Sol arms: the model")
    ap.add_argument("--sol-url", default="https://api.openai.com/v1/responses",
                    help="the Sol arms: the endpoint. OpenAI's, or a test server on this machine: the key is sent nowhere else")
    ap.add_argument("--sol-timeout", type=float, default=5.0, help="the Sol arms: seconds for the API's answer; without one mix's pick is played")
    ap.add_argument("--sol-decide-s", type=float, default=5.0,
                    help="the Sol arms: seconds a decision gets for Sol on top of the search's deadline (the answer is not "
                         "waited for longer than this, nor than --sol-timeout)")
    ap.add_argument("--sol-effort", default="none", choices=("none", "low", "medium", "high"),
                    help="the Sol arms: the API's reasoning effort (none: the fastest; when the API refuses it, low is used)")
    # The lines arms (search_player.LINES_ARMS; bench/fp_bridge_server.py "lines").
    ap.add_argument("--lines-ms", type=int, default=1500, help="the lines arms: time for the lines after the search, all of them together")
    ap.add_argument("--lines-k", type=int, default=3, help="the lines arms: how many of the search's favourite actions get a line (1 to 6)")
    ap.add_argument("--lines-turns", type=int, default=5, help="the lines arms: whole turns a line is played forward (1 to 8)")
    ap.add_argument("--lines-worlds", type=int, default=2, help="the lines arms: the likeliest sampled sets of the opponent a line is read in (1 to 4)")
    ap.add_argument("--lines-threads", type=int, default=4,
                    help="the lines arms: the engine's threads for each of a line's searches (1 to 8; the lines of one decision "
                         "hold this many cores for about --lines-ms)")
    ap.add_argument("--lines-margin", type=float, default=0.10,
                    help="mix-lines: mix's pick gives way to an action whose line ends this much better (the search's result, 0 to 1) "
                         "and in no sampled world worse")
    ap.add_argument("--luna-skip-check", action="store_true",
                    help="the Luna arm: do not ask the check question before the first game (tests with a server that fails on purpose)")
    # Showdown's typing limit (OrderGuard).
    ap.add_argument("--order-spacing-ms", type=float, default=300.0,
                    help="every message to the server waits this long after the one before it, across all games (0: off)")
    ap.add_argument("--order-max-wait-s", type=float, default=1.5, help="an order never waits longer than this for the spacing")
    ap.add_argument("--resend-after-s", type=float, default=1.0,
                    help="an order the typing limit dropped is sent again this long after the notice, then twice and four "
                         "times that (plus up to 0.3 s)")
    ap.add_argument("--max-resends", type=int, default=3,
                    help="after the typing limit's notices, an order is sent again at most this many times per request")
    ap.add_argument("--max-timer-resends", type=int, default=8,
                    help="and once for each timer line that says the server still waits for it, at most this many times "
                         "per request (a count of its own: the lines come 30 s apart, 5 s apart under 20 s)")
    # Settings per game (ArmSetting, search_player.GameSetting and SettingPicker): off unless given.
    ap.add_argument("--arm-settings", dest="arm_settings_file", default=None, metavar="FILE",
                    help="a JSON list of named settings, [{\"name\": \"2s-deeper\", \"args\": [\"--player\", \"mix-trust\", "
                         "\"--search-ms\", \"900\", ...]}, ...]: every new game gets one of them, balanced (the fewest games "
                         "first, one with no game in progress where it can), and plays with its mode, search budget and "
                         "--trust exactly as ladder.py started with that setting's arguments alone would. Every other flag is "
                         "the run's; a setting that gives one must give this run's value. Not with --arms")
    return ap


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """The command line, checked and tidied (the arms as a list, the settings of --arm-settings read). Nothing is started."""
    args = tidy_args(build_parser().parse_args(argv))
    args.arm_settings = load_arm_settings(args, argv) if args.arm_settings_file else None
    return args


def tidy_args(args: argparse.Namespace) -> argparse.Namespace:
    """parse_args' checks and clamps of a parsed command line (also of each setting's arguments, --arm-settings)."""
    args.parallel = max(1, min(8, args.parallel))
    args.reconnect_for = max(120.0, float(args.reconnect_for))
    args.helpers = max(1, min(MAX_HELPERS, args.helpers))
    args.deep_workers = max(1, min(MAX_DEEP_WORKERS, args.deep_workers))
    if args.arms:
        args.arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        bad = [a for a in args.arms if a not in PLAYER_MODES]
        if bad or len(set(args.arms)) != len(args.arms) or not args.arms:
            raise SystemExit("--arms is a list of different modes out of: " + ", ".join(PLAYER_MODES))
    if args.experiment is not None and not str(args.experiment).strip():
        raise SystemExit("--experiment needs a tag")
    if not (0.05 <= float(args.luna_timeout) <= 30.0):
        raise SystemExit("--luna-timeout is between 0.05 and 30 seconds")
    if not (0.05 <= float(args.sol_timeout) <= 30.0) or not (0.05 <= float(args.sol_decide_s) <= 30.0):
        raise SystemExit("--sol-timeout and --sol-decide-s are between 0.05 and 30 seconds")
    if not (0.0 <= float(args.trust) < 1.0):
        raise SystemExit("--trust is a probability from 0 (off) up to 1, such as the arm mix-trust's 0.03")
    if not 1.0 <= float(args.claude_timeout) <= 120.0:
        raise SystemExit("--claude-timeout is between 1 and 120 seconds")
    if not (1 <= args.lines_k <= 6 and 1 <= args.lines_turns <= 8 and 1 <= args.lines_worlds <= 4 and 100 <= args.lines_ms <= 20000
            and 1 <= args.lines_threads <= 8):
        raise SystemExit("--lines-k is 1 to 6, --lines-turns 1 to 8, --lines-worlds 1 to 4, --lines-ms 100 to 20000, --lines-threads 1 to 8")
    return args


if __name__ == "__main__":
    sys.exit(main())
