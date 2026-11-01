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

On the training box the bot runs in a tmux session started by cloud/ladder_box.sh (the Desktop's
Net Play, Net Stop and Net Elo call it): "mix" in every game, 3 games at a time on 4 helpers.

A search needs at least one helper more than --parallel (it refuses to start otherwise: with
fewer, games queue behind each other and their searches come too late). The helpers of a ladder
run are watched closely: one that answers nothing for 3 x --deadline (5 s at least) is replaced,
an idle one is pinged every 3 s so that a frozen one is found before a battle is put on it, and
a question that cannot be answered in time is given up at once. When the search failed in more
than --max-fallback-share of the last --health-window decisions it was asked, no new games are
started (the games in progress are finished): a bot whose helpers keep failing does not ladder
on as the network alone.

--player net (the default) is the network alone, as before: nothing is started and nothing is
sent to a search. --search-ms is the search time per decision and --helpers the number of helper
processes (one core each while they search; one more than --parallel keeps a search from waiting
behind another). The search is waited for until --deadline seconds after the request; when it
has not answered by then the network's move is played, and that is counted.

Every battle's messages, as they came and as they were sent, are kept in
runs/ladder/battles/<battle tag>.jsonl (battle rooms only: nothing about the login is in them).

Account. The name comes from --user or the PS_USERNAME environment variable. The password is
read from PS_PASSWORD, or typed at a prompt in this window when that is not set. It is never
printed, logged or written to a file. A local --no-security server needs no password.

Stopping. When a file named ladder.stop appears next to this script it starts no more games,
finishes the ones in progress and stops. It also stops after --games games. Ctrl+C in the middle
of a game abandons that game, and the server then counts it as a loss on time: use the stop file.

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
code 4. bench/ladder_faults.py makes all of this happen on a local server (the connection cut,
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
import getpass
import glob
import html
import inspect
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs", "ladder", "results.jsonl")
GRAPH = os.path.join(HERE, "runs", "ladder", "elo.html")
BATTLES = os.path.join(HERE, "runs", "ladder", "battles")
RATING_RE = re.compile(r"([^<>|]*?)'s rating: (\d+) &rarr; <strong>(\d+)</strong>")
TRAIL_SHOWN = 14
PLAYER_MODES = ("net", "search", "mix", "mix-veto", "mix-pool", "mix-shortlist", "predict")   # search_player.MODES
LOW_ITERATIONS = 60_000   # below this the search is weaker than anything measured as good (bench/FOULPLAY_BRIDGE.md)
FAST_RETRY_FOR = 90.0     # s after a lost connection during which a new one is tried every second (see reconnect)


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

    The password is never printed. Should it ever be in a line all the same (an exception's text,
    say), it is replaced by *** before the line is written. Writing never raises: a full disk must
    not disturb a game.
    """

    def __init__(self, folder: str, secret: Optional[str] = None):
        import threading

        self.folder = folder
        self._secret = secret if secret and len(secret) >= 3 else None
        self._lock = threading.Lock()
        self._date: Optional[str] = None
        self._file: Any = None
        self._partial: Dict[str, str] = {}
        self.failed = 0

    def path(self, date: Optional[str] = None) -> str:
        return os.path.join(self.folder, f"console-{date or time.strftime('%Y-%m-%d')}.log")

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


def arm_counts(games: List[Dict[str, Any]], checkpoint: str) -> Dict[str, int]:
    """Games per arm among the recorded games of one checkpoint: where a new run's balance starts."""
    counts: Dict[str, int] = {}
    for d in games:
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
    fields["mix_rule"] = mix_rule if str(fields.get("arm")) in ("mix", "predict") else None
    return fields


def arms_text(games: List[Dict[str, Any]]) -> str:
    """One line per arm: its games, its record, and how the search did in them."""
    arms: Dict[str, List[Dict[str, Any]]] = {}
    for d in games:
        if d.get("arm"):
            arms.setdefault(d["arm"], []).append(d)
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
        lines.append(text)
    return "\n".join(lines)


class SearchHealth:
    """
    How often the search failed lately. Over the last `window` decisions in which it should have
    answered (not "single" or "revive", where it is not asked), the share that fell back to the
    network because of the search itself: no answer in time ("timeout"), not started in time
    ("expired") or a failed helper ("error"). A bot whose helpers keep failing otherwise plays on
    as the network alone, quietly, for as many games as it was given.
    """

    FAILED = ("timeout", "expired", "error", "down")
    NOT_ASKED = ("single", "revive", "off", "rejoined")

    def __init__(self, window: int = 50):
        from collections import deque

        self.window = max(1, int(window))
        self.recent: Any = deque(maxlen=self.window)
        self.total = 0
        self.failed = 0

    def note(self, record: Dict[str, Any]) -> None:
        try:
            if str(record.get("mode", "net")) == "net":
                return
            reason = record.get("reason") or ""
            if reason in self.NOT_ASKED:
                return
            bad = reason in self.FAILED
            self.recent.append(bad)
            self.total += 1
            self.failed += bad
        except Exception:   # bookkeeping must never disturb a game
            pass

    def share(self) -> Optional[float]:
        """The failed share of the last `window` decisions, None until there are that many."""
        if len(self.recent) < self.window:
            return None
        return sum(self.recent) / float(len(self.recent))

    def text(self) -> str:
        recent = list(self.recent)
        return (f"the search failed in {sum(recent)} of the last {len(recent)} decisions it was asked, "
                f"{self.failed} of {self.total} in this run")


def hms(t: Optional[float] = None) -> str:
    """The local time of day, for the lines about the connection and the search helpers."""
    return time.strftime("%H:%M:%S", time.localtime(time.time() if t is None else t))


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
    health = SearchHealth(args.health_window)

    # The network alone, as before, unless a search or arms were asked for.
    modes = list(args.arms) if args.arms else [args.player]
    with_search = bool(args.arms) or args.player != "net"
    extra: Dict[str, Any] = {}
    bridge = None
    picker = None
    if with_search:
        import foulplay_bridge as fb
        from search_player import ArmPicker, MixSettings, minimum_deadline, search_player, split_mode

        if any(split_mode(mode)[0] != "net" for mode in modes):
            if args.deadline < minimum_deadline(args.search_ms):
                raise SystemExit(f"--search-ms {args.search_ms} needs --deadline {minimum_deadline(args.search_ms):.2f} or more: "
                                 f"a search that cannot end by the deadline is never started")
            if "predict" in modes and args.deadline < minimum_deadline(args.search_ms) + args.payoff_ms / 1000.0:
                raise SystemExit(f"--search-ms {args.search_ms} with --payoff-ms {args.payoff_ms:.0f} needs --deadline "
                                 f"{minimum_deadline(args.search_ms) + args.payoff_ms / 1000.0:.2f} or more")
            if args.helpers < args.parallel + 1:
                # Fewer helpers than games and the games queue behind each other in them: at 6
                # games on 4 helpers a third of the decisions were the network's alone, each
                # played just before the deadline (review of 2026-10-06, run s3).
                raise SystemExit(f"--helpers {args.helpers} with --parallel {args.parallel}: a search needs at least one "
                                 f"helper more than games at a time (--helpers {args.parallel + 1} or --parallel {args.helpers - 1})")
            ok, why = fb.available()
            if not ok:
                raise SystemExit("the search cannot be used: " + why)
            print(f"search: {why}")
            print(f"starting {args.helpers} search helpers ...", flush=True)
            # Live play: a frozen helper is replaced within seconds, also one with no battle (idle
            # pings), a question that cannot be answered in time fails at once, and a helper is
            # never started from the event loop (review of 2026-10-06, findings 1 to 3).
            bridge = fb.FoulPlayBridge(helpers=args.helpers, search_ms=args.search_ms, timeout=max(4.0, args.deadline),
                                       hang_timeout=max(5.0, 3.0 * args.deadline), idle_ping=3.0,
                                       early_expire=True, background_spawn=True, fail_when_down=True)
            try:
                bridge.start(timeout=180.0)   # before logging in: the helpers take seconds to start, WSL up to a minute
            except Exception as exc:
                bridge.close()
                raise SystemExit(f"the search helpers did not start: {exc}")
        if args.arms:
            earlier = arm_counts(load_games(OUT, args.server, me), os.path.basename(args.checkpoint))
            picker = ArmPicker(modes, counts=earlier)

        def on_decision(record: Dict[str, Any]) -> None:
            if recorder is not None:
                recorder.write(record.get("battle") or "", "note", {"decision": record})
            health.note(record)

        extra = dict(mode=modes[0], mode_picker=picker, bridge=bridge, search_ms=args.search_ms, deadline=args.deadline,
                     mix=MixSettings(rule=args.mix_rule), on_decision=on_decision)
        if "predict" in modes:
            from opponent_model import PredictSettings

            def on_outcome(record: Dict[str, Any]) -> None:
                if recorder is not None:
                    recorder.write(record.get("battle") or "", "note", {"outcome": record})

            extra.update(predict=PredictSettings(min_confidence=args.predict_confidence, min_gain=args.predict_gain,
                                                 max_risk=args.predict_risk, payoff_ms=args.payoff_ms,
                                                 once_bitten=args.predict_once_bitten),
                         on_outcome=on_outcome)
        base = audited(search_player())
    else:
        base = audited(batched_policy_player())

    class LadderPlayer(base):
        """The evaluated policy player, plus a pause before each move, a note of each game's
        rating change, and what it takes to pick its games up again after a lost connection."""

        def __init__(self, *a: Any, **kw: Any):
            # Before the base class: Player.__init__ starts listening.
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
                        elif len(parts) > 2 and parts[1] == "request" and parts[2].strip():
                            try:
                                self._rq[room] = json.loads("|".join(parts[2:])).get("rqid")
                            except ValueError:
                                pass
                    if room.startswith("battle-"):
                        kinds = {parts[1] for parts in split_messages[1:] if len(parts) > 1}
                        if "error" in kinds and room in self.rejoined_battles:
                            # "Too late": the server has an order for this request already, or has
                            # moved on to the next one, which comes. poke-env would decide again and
                            # send one more order, which the server takes as the answer to the next
                            # request: from then on every order lands one request late.
                            # "There's nothing to choose": the server had the copy sent again already
                            # and waits for the opponent; deciding again would be the same mistake.
                            late = [p for p in split_messages[1:] if len(p) > 2 and p[1] == "error"
                                    and (p[2].startswith("[Invalid choice] Sorry, too late")
                                         or p[2].startswith("[Invalid choice] There's nothing to choose"))]
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
                        if room in self._rejoining and "request" in kinds:
                            held = self._held_request(room, split_messages)
                    if recorder is not None and room.startswith("battle-") and not recorded:
                        if room not in recorder.seen:
                            mode_of = getattr(self, "battle_mode", None)
                            recorder.write(room, "note", {"battle": room, "user": me, "server": args.server,
                                                          "checkpoint": os.path.basename(args.checkpoint),
                                                          "arm": mode_of(room) if mode_of else None})
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
            **extra,
        )
    except BaseException:
        if bridge is not None:
            bridge.close()
        raise
    if getattr(player, "policy", None) is None:
        raise SystemExit("no network was loaded")
    player._ladder_seen = seen
    player._ladder_health = health
    player._ladder_recorder = recorder
    player._ladder_bridge = bridge
    player._ladder_modes = modes if with_search else None

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

    async def leave_later(tag: str) -> None:
        t0 = time.time()
        while time.time() - t0 < 12.0 and "rating" not in seen.get(tag, {}):
            await asyncio.sleep(0.2)
        try:
            await send(f"/leave {tag}")
        except Exception:
            pass

    async def send_message(message: str, room: str = "", message_2: Optional[str] = None):
        if isinstance(message, str) and message.startswith("/leave battle-"):
            task = asyncio.ensure_future(leave_later(message.split(" ", 1)[1].strip()))
            leaving.add(task)
            task.add_done_callback(leaving.discard)
            return None
        try:
            if isinstance(room, str) and room.startswith("battle-") and isinstance(message, str) and message.startswith("/choose"):
                # The order for this battle's current request, kept until the next one: when it goes
                # into a connection that is lost, it is sent again after the room is joined again.
                player._sent[room] = (player._rq.get(room), message)
            if recorder is not None and room:
                recorder.write(room, "out", message if not message_2 else f"{message}|{message_2}")   # battle rooms only
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

    client.send_message = send_message

    # What the server says about this account after every login: the battles it is in (to join
    # again after a lost connection, also one that started while the bot was away) and whether
    # it is still searching. poke-env reads nothing of it.
    handle = client._handle_message

    async def handle_message(message: str):
        if isinstance(message, str) and message.startswith("|updatesearch|"):
            try:
                data = json.loads(message.split("|", 2)[2])
                player.server_games = {room for room in (data.get("games") or {}) if str(room).startswith("battle-")}
                player.server_searching = list(data.get("searching") or [])
            except Exception:
                pass
        return await handle(message)

    client._handle_message = handle_message
    return player


# ---------------------------------------------------------------------------
# Playing
# ---------------------------------------------------------------------------

async def drive(player: Any, args: argparse.Namespace, user: str) -> int:
    """Search, play and report until --games games are done or the stop file appears.
    Runs in poke-env's own event loop, next to the player."""
    client = player.ps_client
    me = to_id(user)
    ckpt = os.path.basename(args.checkpoint)
    try:
        await asyncio.wait_for(client.logged_in.wait(), timeout=args.login_timeout)
    except asyncio.TimeoutError:
        print(f"could not log in as {user} within {args.login_timeout} s: check the account name and the password")
        return 2

    earlier = load_games(OUT, args.server, me)
    print(f"network: {ckpt} on {player.policy.device}   account: {user}   server: {args.server}   format: {args.format}")
    print(f"up to {args.parallel} games at a time, {args.think_ms} ms before each move, battle timer {'on' if args.timer else 'off'}")
    modes = getattr(player, "_ladder_modes", None)
    bridge = getattr(player, "_ladder_bridge", None)
    recorder = getattr(player, "_ladder_recorder", None)
    if modes:
        how = ("arms " + ", ".join(modes) + " (one per game, balanced)") if args.arms else ("player " + modes[0])
        print(f"{how}" + (f"   search {args.search_ms} ms on {args.helpers} helpers, answer within {args.deadline} s of the request, "
                          f"mix rule {args.mix_rule}" if bridge is not None else ""))
    if earlier:
        print(f"earlier on this account: {summary_text(earlier)}")
    print(f"to stop: create the file {args.stop_file} (no new games, the ones in progress are finished)\n", flush=True)

    seen = player._ladder_seen
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
        if gone is not None:
            line["won"] = None                 # the room was gone when the bot came back: no result seen
            line["lost_track"] = gone
        arm = ""
        if modes:
            # The game's arm and what the search did in it (search_player.game_summary).
            try:
                summary = player.game_summary(tag) or {}
                line.update(arm_fields(summary, getattr(b, "player_role", None),
                                       args.search_ms if bridge is not None else None, args.mix_rule))
                played.append(line)
                arm = f"   [{summary.get('arm')}" + (f", search {summary.get('searched')}/{summary.get('decisions')}"
                                                     if summary.get("asked") or summary.get("fallbacks") else "") + "]"
                player.forget_decisions(tag)
            except Exception:
                pass
        writer.submit(write_result, dict(line))
        if recorder is not None:
            recorder.write(tag, "note", {"result": line})
            recorder.close(tag)
        change = f"   rating {rating[0]} -> {rating[1]} ({rating[1] - rating[0]:+d})" if rating else ""
        if gone is not None:
            print(f"{tag}: the game ended while the bot was away (its room was gone: {gone}); no result seen, not counted", flush=True)
            return
        print(f"game {wins + losses:3d}: {'WIN ' if won else 'LOSS'} in {b.turn:2d} turns vs {opp}"
              f"{' (' + str(line['opponent_rating']) + ')' if line['opponent_rating'] else ''}{change}"
              f"   record {wins}-{losses}{arm}")
        if trail:
            print(f"          elo: {trail_text(trail)}   (peak {max(trail)})")
        if health is not None and health.total and (wins + losses) % 5 == 0:
            print(f"          search: {health.text()}")
        sys.stdout.flush()

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
        nonlocal searching_since, search_sent, cancel_at, abandoned
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
                login = asyncio.ensure_future(client.logged_in.wait())
                await asyncio.wait({task, login}, timeout=min(30.0, float(args.login_timeout)),
                                   return_when=asyncio.FIRST_COMPLETED)
                if client.logged_in.is_set() and not task.done():
                    break
                login.cancel()
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
        task of its own, so that it is also told while the connection is being made again."""
        down_at: Optional[float] = None
        while True:
            await asyncio.sleep(0.2)
            try:
                up = bridge.helpers_up()
                now = time.time()
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
        failed_share = health.share() if (health is not None and modes and args.max_fallback_share > 0) else None
        if not stopping and failed_share is not None and failed_share > args.max_fallback_share:
            # Playing on would be the network alone under the search's name. Stop starting games.
            stopping = True
            playing = sum(1 for t in active if t not in done_at)
            print(f"STOPPING: {health.text()} ({100.0 * failed_share:.0f}%, more than {100.0 * args.max_fallback_share:.0f}%). "
                  f"No new games" + (f", finishing the {playing} in progress" if playing else "")
                  + ". The search helpers are failing: see the warnings above.", flush=True)
        if stopping and searching_since is not None and cancel_at is None:
            await client.send_message("/cancelsearch")
            cancel_at = now
        if cancel_at is not None and searching_since is not None and now - cancel_at > 3.0:
            searching_since = search_sent = None   # the search was withdrawn before a game was found

        # One search at a time: the server refuses a second one in the same format. A search
        # is asked again after two minutes in case it was lost.
        playing = sum(1 for t in active if t not in done_at)
        if not stopping and playing < args.parallel:
            if searching_since is None and started < args.games:
                await client.search_ladder_game(player._format, player.get_next_team())
                searching_since = search_sent = now
            elif searching_since is not None and now - search_sent > 120.0:
                await client.search_ladder_game(player._format, player.get_next_team())
                search_sent = now

        if not active and searching_since is None and (stopping or started >= args.games):
            break
        await asyncio.sleep(0.2)

    if watcher is not None:
        watcher.cancel()
    writer.shutdown(wait=False)
    await asyncio.get_running_loop().run_in_executor(None, writer.shutdown, True)   # every result is on disk
    rejected = getattr(player, "n_rejected", None)
    total = wins + losses
    print(f"\nfinished: {wins} wins, {losses} losses" + (f" ({100.0 * wins / total:.0f}%)" if total else "")
          + (f", rating {first_rating} -> {last_rating}" if last_rating is not None else "")
          + (f", orders rejected by the server: {rejected}" if rejected else ""))
    if modes:
        if played:
            print("by arm:\n" + arms_text(played))
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
        except Exception:
            pass
    if recorder is not None:
        recorder.close_all()
        if recorder.failed:
            print(f"{recorder.failed} lines could not be written to {recorder.folder}")
    if abandoned is not None:
        print(f"stopped without a connection to the server: {len(abandoned)} game{'s' if len(abandoned) != 1 else ''} "
              f"abandoned, recorded with no result")
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
            bridge.close()   # stops the search helpers (inside WSL as well)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Play the trained policy on a Showdown ladder.")
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
    ap.add_argument("--login-timeout", type=int, default=45)
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
    ap.add_argument("--player", default="net", choices=PLAYER_MODES,
                    help="net: the network alone (the default, as before). search: the search's own choice. "
                         "mix: the two combined (--mix-rule). mix-veto, mix-pool, mix-shortlist: that rule. "
                         "predict: mix, left only on a confident prediction of the opponent's action (opponent_model.py)")
    ap.add_argument("--arms", default=None,
                    help="e.g. net,mix: every new game is played in one of these modes, balanced, and its arm is written with its result")
    ap.add_argument("--search-ms", type=int, default=500, help="search time per decision (an answer takes about 2.3 times this)")
    ap.add_argument("--helpers", type=int, default=4, help="search helper processes, one core each while searching (1 to 6)")
    ap.add_argument("--deadline", type=float, default=2.5,
                    help="seconds from the request to the order: a search that has not answered by then is not waited for")
    ap.add_argument("--mix-rule", default="veto", choices=("veto", "pool", "shortlist"), help="what 'mix' means (search_player.mix)")
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
    ap.add_argument("--ask-password", action="store_true",
                    help="ask for the password at the prompt also for a local server, and drop it (a test of the prompt: "
                         "a local server needs none, and nothing is sent to the login server)")
    args = ap.parse_args(argv)
    args.parallel = max(1, min(8, args.parallel))
    args.reconnect_for = max(120.0, float(args.reconnect_for))
    args.helpers = max(1, min(6, args.helpers))
    if args.arms:
        args.arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        bad = [a for a in args.arms if a not in PLAYER_MODES]
        if bad or len(set(args.arms)) != len(args.arms) or not args.arms:
            raise SystemExit("--arms is a list of different modes out of: " + ", ".join(PLAYER_MODES))
    global OUT, GRAPH, BATTLES
    if args.out:
        OUT = os.path.abspath(args.out)
        GRAPH = os.path.join(os.path.dirname(OUT), "elo.html")
        BATTLES = os.path.join(os.path.dirname(OUT), "battles")
    args.battles_dir = os.path.abspath(args.battles_dir) if args.battles_dir else BATTLES

    user = args.user or os.environ.get("PS_USERNAME")
    if args.history:
        return print_history(OUT, args.server, to_id(user) if user else None)
    if args.graph:
        return show_graph(args.server, to_id(user) if user else None, args.once)

    if args.checkpoint is None:
        args.checkpoint = newest_checkpoint(os.path.join(HERE, "runs", "vm", "checkpoints"))
    if not args.checkpoint or not os.path.isfile(args.checkpoint):
        raise SystemExit("no checkpoint found: give --checkpoint")
    if not user:
        raise SystemExit("give the account name with --user or PS_USERNAME")
    password: Optional[str] = None
    if args.server == "public":
        password = os.environ.get("PS_PASSWORD") or getpass.getpass(f"Showdown password for {user} (not shown, not stored): ")
        if not password:
            raise SystemExit("a password is needed for the public server")
    elif args.ask_password:
        typed = getpass.getpass(f"Showdown password for {user} (not shown, not stored; a local server needs none, so it is dropped): ")
        print(f"password prompt answered ({'something' if typed else 'nothing'} typed); a local server: not used")
        del typed
    os.chdir(HERE)
    if args.console_log:
        console = ConsoleLog(os.path.dirname(OUT), secret=password)
        console.line("=" * 20 + f" ladder.py started (pid {os.getpid()}): " + " ".join(sys.argv[1:] if argv is None else argv))
        sys.stdout = console.wrap(sys.stdout, "out")
        sys.stderr = console.wrap(sys.stderr, "err")
        print(f"console log: {console.path()}", flush=True)
    return asyncio.run(run(args, user, password))


if __name__ == "__main__":
    sys.exit(main())
