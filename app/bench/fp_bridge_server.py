"""
Foul Play as an adviser: one long-lived process that follows any number of battles with Foul
Play's own code and answers "what would you play here, and what did your search see?".

It is started by foulplay_bridge.py, with FOUL PLAY's virtualenv python (this file imports Foul
Play's modules and poke-engine, and nothing of this repository):

    <foul-play>/.venv/bin/python -u bench/fp_bridge_server.py --foulplay-dir <foul-play>

JSON lines on stdin, JSON lines on stdout. stderr is the log. One message per line:

    {"op":"ping","id":1}
    {"op":"feed","key":K,"user":U,"room":R,"data":D}      no reply
        One websocket message of the battle room R, as the player U received it: the lines after
        the ">room" header, joined with "\n". Every message, in order, requests included.
    {"op":"decide","id":2,"key":K,"ms":100,"rqid":7,"deadline":81234.5}
        Foul Play's move for the request it was fed last. "rqid" is optional: when given it must be
        the request Foul Play is looking at, or the answer is an error instead of a stale move.
        "deadline" is optional: the time, on time.monotonic() (the clock the "ping" answer shows),
        after which nobody waits for this answer. A decision that could not be made before it
        (about 2.5 x ms) is not searched: the answer is {"ok":false,"expired":true} at once.
    {"op":"chose","key":K,"choice":"move 2 terastallize"}  no reply
        What the player really sent. Foul Play reasons about the opponent's speed, item and set
        from the move it believes it selected, so it has to be told when that was not its advice.
    {"op":"end","key":K}                                   no reply: forget the battle
    {"op":"sync","id":3}        answered when everything before it has been processed
    {"op":"beat"}               no reply: "the bridge is still there" (see --parent-watch beat)
    {"op":"stats","id":4}       battles held, decisions served, memory
    {"op":"exit"}

Every answer to a message with an "id" also carries "got": this process's time.monotonic() when
it read that line, and "came": when the last line it had to wait for came in (that line, or one
before it in the same burst; a line read without waiting was there by then at the earliest). A
bridge on another machine tells from these and "clock" whether the network held a line up on its
way here or on its way back (foulplay_bridge: a link stall, not a failure of the search).

A decision answers with:

    {"id":2,"ok":true,"key":K,
     "choice":"move 2 terastallize",     the Showdown choice for the move Foul Play picked
     "best":"move 2 terastallize",       the option with the largest share of the search
     "fp_choice":"earthquake-tera",      the same move in Foul Play's own words
     "fp_message":"/choose move earthquake terastallize",   what Foul Play would have sent
     "options":[{"choice":"move 1","fp":"stealthrock","visits":412,"score":0.431,
                 "total_score":177.6,"policy":0.012,"considered":false}, ...],
         every root action of the search: visits and total score summed over the sampled worlds,
         score = total_score / visits (the mean playout result, 0 to 1, from Foul Play's side),
         policy = the share Foul Play itself computes (mean over worlds of visits / iterations)
         and draws its move from, considered = within 75% of the largest share, which is the set
         Foul Play draws from. Two options can carry the same choice: see "tera_dropped".
     "worlds":4,"iterations":57000,"iterations_per_world":[...],"search_ms":50,
     "ms":241.7,"turn":5,"rqid":7,"force_switch":false,
     "tera_dropped":false}   true: Foul Play asked for Terastallization where the request does not
                             offer it (a Pokemon locked into a move). The server plays that as the
                             plain move, and "choice" says the plain move. "fp_message" is literal.
    or {"id":2,"ok":false,"key":K,"error":"..."}, with "revive":true for the request that asks
       whom Revival Blessing revives and "expired":true for a decision past its deadline.

Since protocol version 2 a decision also carries the OPPONENT's side of the same search:

     "opponent":{"active":"garchomp","revealed":["rotomwash",...],
        "options":[{"name":"move earthquake","fp":"earthquake","share":0.41,"visits":9120,
                    "score":0.44,"worlds":2,"seen":true}, ...],
            every root option the search gave the opponent, by name ("move earthquake",
            "move earthquake terastallize", "switch garchomp", "none" when the opponent has no
            move to make): share = mean over the worlds of visits / iterations, score = mean
            playout result FROM THE OPPONENT'S SIDE (its total over the options is the
            iterations minus ours), worlds = in how many sampled worlds it was an option,
            seen = false for a switch to a Pokemon the opponent has not shown (a sampled guess).
        "worlds":[{"chance":0.5,"iterations":29000,
                   "active":{"species":"garchomp","moves":[...],"item":"...","ability":"...",
                             "tera_type":"...","hp":212,"max_hp":212,...},
                   "reserve":["rotomwash",...],
                   "options":[{"name":...,"fp":...,"visits":...,"share":...,"score":...}]}, ...]}
            per sampled world: the complete set that was drawn for the opponent's active Pokemon,
            the team it was given, and its options in that world.

and three optional fields of "decide" ask for more than Foul Play itself does:

    "iterations":N   every world is searched for exactly N iterations, whatever the clock says
                     (poke-engine's own argument; "ms" is then not used by the search).
    "threads":N      poke-engine's own thread count for the search of one world (default 1).
    "parallelism":P  Foul Play's --search-parallelism for this decision (default 1): it samples P
                     times as many worlds (4P early in a game at half "ms", 2P later at "ms"),
                     searched here one after the other, so the decision takes about P times as long.
    "workers":W      the worlds of this decision are searched W at a time, each in a process of
                     Foul Play's own pool (ProcessPoolExecutor, forked from this helper for the
                     decision, Foul Play's search function, one core each) instead of one after the
                     other here: the same search of every world, in about 1/W of the time. Default 1.
                     The answer then has "workers":W.
    "payoff_ms":B, "payoff":"engine"|"full"
                     after the decision, B ms more (all worlds together) for a table of values per
                     (our option, opponent's option), answered as
     "payoff":{"mode":"engine","budget_ms":B,"ms":203.1,"unit":"engine evaluation, larger is better for us",
        "worlds":[{"chance":0.5,"depth":2,"ours":["earthquake",...],"choices":["move 1",...],
                   "theirs":["move stompingtantrum",...],"matrix":[[-141.9,null,...],...],
                   "cells":84,"known":31,"ms":101.0}, ...]}
            "engine": poke-engine's iterative deepening expectiminimax on the world's state, its
            matrix as it is. The search prunes (alpha-beta), so cells it did not need are null.
            "full": every pair is played out with generate_instructions (every outcome, weighted
            by its chance) and each resulting state is searched 1 ply: no cell is pruned, depth
            is 2, and cells are null only where the budget ran out (pairs are visited in the
            order of how much the Monte Carlo search visited them).
    "lines":{"k":3,"turns":5,"ms":1500,"worlds":2,"threads":4}
                     after the decision, about "ms" more for LINES: our k options the search
                     favours most, each played forward "turns" whole turns in the "worlds"
                     likeliest sampled worlds. A step: the engine's likeliest outcome of the two
                     actions is applied, the new position is searched (ms / (worlds x k x
                     (turns + 2)) each, on "threads" threads of the engine's own, default 1), and
                     both sides play what that search visited most.
                     Answered as
     "lines":{"k":3,"turns":5,"step_ms":35.7,"ms":1490.2,"candidates":["move 1","switch 3",...],
        "worlds":[{"world":0,"chance":0.5,"lines":[
            {"choice":"move 1","fp":"earthquake","turns":5,"end":0.61,"value":0.58,
             "first_step":[{"p":0.1,"value":0.31}],
             "steps":[{"ours":"move earthquake","theirs":"move outrage","p":0.9,
                       "after":{"ours":{"active":"landorustherian","hp":41,"left":5,"team_hp":0.62},
                                "theirs":{"active":"garchomp","hp":22,"left":4,"team_hp":0.48}}}, ...]}]}]}
            "end": the search's mean result for our side at the line's last position (0 lost, 1
            won); "value": the same with the first step's other outcomes of 5% or more (a miss,
            a critical hit) each searched once and weighed in by its chance; "forced" marks a
            step that is a forced switch and not a turn; "cut" a line the time ran out on.

How Foul Play's code is used, and what is replaced in memory (its checkout is never edited):

  * State tracking is Foul Play's own battle loop, fp.run_battle.pokemon_battle, run once per
    battle and unchanged. It is given an object in place of its websocket client whose
    receive_message() yields the next fed message, so the loop is advanced by hand, one message
    at a time, with no event loop and no thread. Whatever Foul Play does with a message on the
    real server it does here, including the start of a battle and the private requests.
  * Inside that loop, the call that picks a move (async_pick_move) is replaced by one that only
    notes "Foul Play would move now". The search then runs when the player asks ("decide"), and
    it is Foul Play's original async_pick_move that runs it: copy the battle, bring it up to
    date from the request, find_best_move, format_decision.
  * fp.search.main.ProcessPoolExecutor is replaced by an executor that runs the search in this
    process (--pool inline, the default). Foul Play forks a fresh worker for every decision; the
    worker calls the same function with the same arguments. --pool fork keeps Foul Play's pool,
    and a decision with "workers" gets that pool with W processes (forked from this helper after
    the decision's settings are made, so they search exactly as the inline search would).
  * fp.search.main.select_move_from_mcts_results and fp.modes.base.find_best_move are wrapped to
    keep what they were given (the search results of every world, the prepared battle). They
    return what the originals return.
  * FoulPlayConfig.username is set to the battle's player before every step, and
    search_time_ms / parallelism before every decision: Foul Play keeps them in a module-level
    object because one Foul Play process normally plays one battle as one user.

The search holds the interpreter lock, so one process is one core, and decisions for the
battles of one process are answered one after the other.

Only random battle formats are supported (gen9randombattle is the one that was tested).
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import math
import os
import signal
import sys
import threading
import time
import traceback
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

PROTOCOL_VERSION = 2
DEFAULT_FORMAT = "gen9randombattle"
_TOMBSTONES = 4096          # finished battles remembered, so that their late lines are dropped
_FLAG_WORDS = {"terastallize": "terastallize", "mega": "mega", "ultra": "ultra", "dynamax": "dynamax", "zmove": "zmove"}


# =============================================================================================
# Process plumbing: the protocol channel, and not outliving the parent
# =============================================================================================

def _take_stdout() -> Any:
    """
    Keeps the real stdout for the protocol and points fd 1 at stderr, so that nothing Foul Play,
    a library or the Rust engine prints can end up between two JSON lines.
    """
    sys.stdout.flush()
    channel = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1, newline="\n")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return channel


def _own_group() -> None:
    """
    Makes this process the leader of a process group of its own, so that it and whatever it forks
    (--pool fork) can be stopped together by that group's id. The bridge does this itself with
    start_new_session on Linux; a helper started through wsl.exe from Windows has to ask for it.
    """
    try:
        if os.getpgrp() != os.getpid():
            os.setpgid(0, 0)
    except (OSError, AttributeError):
        pass


BEAT_TIMEOUT_S = float(os.environ.get("FP_BRIDGE_BEAT_TIMEOUT") or 20.0)   # --parent-watch beat: an idle helper that hears
                           # nothing for this long leaves (the tests set a shorter one)
_ALIVE = {"last": time.monotonic(), "busy": True}   # when the last line was dealt with; busy while one is (or while loading)


def _watch_parent(mode: Optional[str] = None) -> None:
    """
    Exits when the process that started this one is gone (it may have been killed outright).
    The parent's PID comes in FP_BRIDGE_PARENT_PID.

    mode "beat" is for a helper started through wsl.exe by a Windows process. That process's PID
    means nothing in here, and the helper's stdin is carried by wsl.exe: when the Windows process
    dies, wsl.exe closes it and the helper leaves, but when wsl.exe is killed along with it
    (measured: taskkill of the tree) the Linux side of the pipe stays open and nothing tells the
    helper. So the bridge sends a line every 2 seconds, and a helper that is not working and has
    heard nothing for BEAT_TIMEOUT_S leaves. Its parent in WSL, the relay of that wsl.exe, is
    watched as well.

    The same mode serves a helper started over ssh from another machine (foulplay_bridge's
    RemoteTarget): its parent is the ssh session here, which goes when the connection ends; a
    connection that goes quiet without ending (the other machine lost its network or its power)
    sends no more lines, and the helper leaves BEAT_TIMEOUT_S after its last one (the bridge's
    clock probes come every 2 s).
    """
    expected = os.environ.get("FP_BRIDGE_PARENT_PID", "")
    if os.name == "nt":
        return
    beat = mode == "beat"
    if mode in ("ppid", "beat"):
        expected = str(os.getppid())
    if not expected.isdigit():
        return
    parent = int(expected)

    def watch() -> None:
        while True:
            time.sleep(1.0)
            silent = beat and not _ALIVE["busy"] and time.monotonic() - _ALIVE["last"] > BEAT_TIMEOUT_S
            if os.getppid() != parent or silent:
                try:
                    if os.getpgrp() == os.getpid():
                        os.killpg(os.getpid(), signal.SIGKILL)   # this process and anything it forked
                finally:
                    os._exit(113)

    threading.Thread(target=watch, name="parent-watch", daemon=True).start()


def _rss_mb() -> float:
    try:
        with open("/proc/self/statm", "r") as fh:
            pages = int(fh.read().split()[1])
        return round(pages * os.sysconf("SC_PAGE_SIZE") / 1e6, 1)
    except Exception:
        try:
            import resource

            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
        except Exception:
            return -1.0


def _peak_rss_mb() -> float:
    """The largest resident size this process has had (VmHWM), in MB; -1 when it cannot be read."""
    try:
        with open("/proc/self/status", "r") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return round(int(line.split()[1]) * 1024 / 1e6, 1)
    except Exception:
        pass
    try:
        import resource

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except Exception:
        return -1.0


def _read_commit(fp_dir: str) -> str:
    try:
        git_dir = os.path.join(fp_dir, ".git")
        with open(os.path.join(git_dir, "HEAD"), "r", encoding="utf-8") as fh:
            head = fh.read().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(":", 1)[1].strip()
        path = os.path.join(git_dir, *ref.split("/"))
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read().strip()
    except Exception:
        pass
    return "unknown"


# =============================================================================================
# The pieces that stand in for Foul Play's websocket and its process pool
# =============================================================================================

class _NextMessage:
    """What the fed socket waits on. The driver sends the next message into the coroutine."""

    __slots__ = ()

    def __await__(self):
        message = yield self
        return message


class _FedSocket:
    """Stands in for fp.websocket_client.PSWebsocketClient. Nothing leaves this process."""

    def __init__(self) -> None:
        self.sent = 0
        self.last_sent: Optional[Tuple[str, List[str]]] = None

    async def receive_message(self) -> str:
        return await _NextMessage()

    async def send_message(self, room: str, message_list: List[str]) -> None:
        self.sent += 1
        self.last_sent = (room, list(message_list))

    async def leave_battle(self, battle_tag: str) -> None:
        return None

    async def save_replay(self, battle_tag: str) -> None:
        return None


class _InlineExecutor:
    """
    Stands in for the ProcessPoolExecutor in fp.search.main.find_best_move: submit() runs the
    search here and now. Foul Play submits one search per sampled world and reads the results
    after the pool has shut down, so the order of work is the same as with its one worker.
    """

    def __init__(self, max_workers: Optional[int] = None, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> "_InlineExecutor":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> "concurrent.futures.Future":
        future: "concurrent.futures.Future" = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # a Rust panic arrives as a BaseException
            future.set_exception(exc)
        return future


FORK_S_PER_WORKER = 0.03     # starting one process of a decision's pool, and getting its answer back (measured on the box)


def _fork_context() -> Any:
    """
    multiprocessing's "fork" start method, for a decision's own pool ("workers"). The helper has
    one more thread (the parent watch, which only sleeps and reads its parent's PID), so Python
    3.12's warning about forking a process with threads is not shown: a forked worker only runs
    Foul Play's search function on a state string and sends its result back.
    """
    import multiprocessing
    import warnings

    warnings.filterwarnings("ignore", message=r".*use of fork\(\) may lead to deadlocks", category=DeprecationWarning)
    return multiprocessing.get_context("fork")


# =============================================================================================
# One battle
# =============================================================================================

class _Session:
    NEW, RUNNING, FINISHED, FAILED = "new", "running", "finished", "failed"

    def __init__(self, key: str, user: str, room: str) -> None:
        self.key = key
        self.user = user
        self.room = room
        self.socket = _FedSocket()
        self.coro: Any = None
        self.state = self.NEW
        self.battle: Any = None             # Foul Play's Battle, once its loop has asked for a move
        self.pick_rqid: Any = None          # the request Foul Play's loop asked a move for
        self.prepared: Any = None           # the battle as prepared for the decision on prepared_rqid
        self.prepared_rqid: Any = None
        self.error: Optional[str] = None
        self.winner: Optional[str] = None
        self.messages = 0
        self.decisions = 0


class _Server:
    def __init__(self, fp_dir: str, battle_format: str, pool: str, log_level: str) -> None:
        self.fp_dir = fp_dir
        self.battle_format = battle_format
        self.pool = pool
        self.sessions: Dict[str, _Session] = {}
        self.gone: "OrderedDict[str, str]" = OrderedDict()   # key -> why it is no longer held
        self.current: Optional[_Session] = None
        self.capture: Dict[str, Any] = {}
        self.loop = asyncio.new_event_loop()
        self.started = time.time()
        self.counters = {"decisions": 0, "decide_errors": 0, "expired": 0, "failed_battles": 0, "finished_battles": 0,
                         "fed": 0, "dropped": 0, "notes": 0, "notes_unmapped": 0, "notes_revive": 0, "revive_requests": 0, "rejoined": 0, "decide_ms": 0.0,
                         "payoffs": 0, "payoff_ms": 0.0, "payoff_errors": 0,
                         "lines": 0, "lines_ms": 0.0, "line_errors": 0}
        self.fixed_iterations = 0     # > 0 for the decision being made: exactly this many iterations per world
        self.search_threads = 0       # > 0 for the decision being made: poke-engine's thread count
        self.search_workers = 0       # > 1 for the decision being made: its worlds are searched this many at a time
        self._load_foul_play(log_level)

    # ---- Foul Play ---------------------------------------------------------------------------

    def _load_foul_play(self, log_level: str) -> None:
        import logging

        logging.basicConfig(stream=sys.stderr, level=getattr(logging, log_level.upper(), logging.WARNING),
                            format="%(levelname)-8s %(message)s")

        from fp.config import FoulPlayConfig

        # Foul Play's own configuration code, with the arguments a real run would be given.
        argv, sys.argv = sys.argv, [
            "run.py", "--websocket-uri", "none", "--ps-username", "fpbridge",
            "--bot-mode", "accept_challenge", "--pokemon-format", self.battle_format,
            "--search-time-ms", "100", "--search-parallelism", "1", "--log-level", log_level.upper(),
        ]
        try:
            FoulPlayConfig.configure()
        finally:
            sys.argv = argv

        from fp import constants
        from fp.data.mods.apply_mods import apply_mods

        if FoulPlayConfig.format_spec.battle_type != constants.BattleType.RANDOM_BATTLE:
            raise SystemExit("fp_bridge_server: only random battle formats are supported, not %s" % self.battle_format)
        apply_mods(FoulPlayConfig.format_spec)

        # Foul Play checks after every battle that it has not changed its own data tables, and
        # stops if it has (fp.main.check_dictionaries_are_unmodified). Here a changed table would
        # reach every battle of this process, so the same check is made, see _check_data().
        import fp.data as fp_data

        self.fp_data = fp_data
        self.original_pokedex = deepcopy(fp_data.pokedex)
        self.original_moves = deepcopy(fp_data.all_move_json)

        import fp.battle.helpers as fp_helpers
        import fp.battle.state as fp_state
        import fp.modes.base as modes_base
        import fp.modes.random_battle as modes_random
        import fp.run_battle as run_battle
        import fp.search.main as search_main

        self.config = FoulPlayConfig
        self.constants = constants
        self.run_battle = run_battle
        self.fp_state = fp_state
        self.normalize_name = fp_helpers.normalize_name

        # 1. Inside Foul Play's battle loop, picking a move becomes "note that a move is due".
        self.original_pick = modes_base.async_pick_move
        server = self

        async def foul_play_wants_a_move(battle: Any) -> List[str]:
            session = server.current
            if session is not None:
                session.battle = battle
                session.pick_rqid = battle.rqid
                session.prepared = None
            return ["", str(battle.rqid)]

        modes_random.async_pick_move = foul_play_wants_a_move
        run_battle.async_pick_move = foul_play_wants_a_move

        # 2. Keep what the decision code was given and what it produced.
        original_find = modes_base.find_best_move
        original_select = search_main.select_move_from_mcts_results

        def find_best_move(battle: Any) -> str:
            server.capture["prepared"] = battle
            return original_find(battle)

        def select_move_from_mcts_results(mcts_results: Any) -> str:
            server.capture["results"] = mcts_results
            choice = original_select(mcts_results)
            server.capture["choice"] = choice
            return choice

        modes_base.find_best_move = find_best_move
        search_main.select_move_from_mcts_results = select_move_from_mcts_results

        # 3. The search runs in this process unless Foul Play's own pool was asked for (--pool fork),
        #    or the decision asks for "workers": then Foul Play's own pool with that many processes,
        #    forked from this one now, so that they search with this decision's settings.
        foul_plays_pool = search_main.ProcessPoolExecutor

        def pool_of_this_decision(max_workers: Optional[int] = None, *args: Any, **kwargs: Any) -> Any:
            if server.search_workers > 1:
                return foul_plays_pool(max_workers=server.search_workers, mp_context=_fork_context())
            if server.pool == "inline":
                return _InlineExecutor()
            return foul_plays_pool(max_workers, *args, **kwargs)

        search_main.ProcessPoolExecutor = pool_of_this_decision

        # 4. Keep every sampled world: Foul Play's Battle for it and the engine state made of it.
        #    The opponent's set in each world and the payoff table are read from these.
        original_to_state = search_main.battle_to_poke_engine_state

        def battle_to_poke_engine_state(world: Any) -> Any:
            state = original_to_state(world)
            server.capture.setdefault("worlds", []).append((world, state))
            return state

        search_main.battle_to_poke_engine_state = battle_to_poke_engine_state

        # 5. The search call, as Foul Play makes it unless a decision asks for an exact number of
        #    iterations or for threads (both are poke-engine's own arguments).
        original_mcts = search_main.monte_carlo_tree_search

        def monte_carlo_tree_search(state: Any, duration_ms: int = 1000, iterations: int = 0, threads: int = 1) -> Any:
            return original_mcts(state, duration_ms, server.fixed_iterations or iterations, server.search_threads or threads)

        search_main.monte_carlo_tree_search = monte_carlo_tree_search

        import poke_engine

        self.poke_engine = poke_engine

        self.engine_version = getattr(poke_engine, "__version__", None)
        if self.engine_version is None:
            try:
                import importlib.metadata as metadata

                self.engine_version = metadata.version("poke-engine")
            except Exception:
                self.engine_version = "unknown"

    # ---- following a battle ------------------------------------------------------------------

    def _advance(self, session: _Session, message: Optional[str]) -> None:
        """Runs Foul Play's battle loop for this battle until it waits for the next message."""
        self.config.username = session.user
        self.current = session
        try:
            session.coro.send(message)
        except StopIteration as stop:
            session.state = _Session.FINISHED
            session.winner = stop.value
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            session.state = _Session.FAILED
            session.error = "Foul Play's battle loop stopped on message %d: %s: %s" % (
                session.messages, type(exc).__name__, str(exc)[:300])
            self.counters["failed_battles"] += 1
            sys.stderr.write("fp_bridge_server: %s (%s)\n%s\n" % (session.key, session.error, traceback.format_exc(limit=6)))
        finally:
            self.current = None

    def feed(self, key: str, user: str, room: str, data: str) -> None:
        if data.startswith("|init|") and (key in self.gone or key in self.sessions):
            # The player joined the room again and is sent the whole battle once more. Foul Play
            # cannot pick a battle up midway (its loop expects a battle to begin), and feeding the
            # old state the same lines twice would corrupt it without a sound. So the battle starts
            # over here, never reaches a first request, and every decision for it is a plain error.
            if key in self.sessions:
                self._forget(key, "the room was joined again")
            self.gone.pop(key, None)
            self.counters["rejoined"] += 1
        if key in self.gone:
            self.counters["dropped"] += 1
            return
        session = self.sessions.get(key)
        if session is None:
            session = self.sessions[key] = _Session(key, user, room)
            session.coro = self.run_battle.pokemon_battle(session.socket, self.battle_format, None)
            self._advance(session, None)         # up to the first receive_message()
            if session.state == _Session.NEW:
                session.state = _Session.RUNNING
        if session.state != _Session.RUNNING:
            self.counters["dropped"] += 1
            return
        session.messages += 1
        self.counters["fed"] += 1
        self._advance(session, ">" + room + "\n" + data)
        if session.state == _Session.RUNNING and session.battle is not None:
            battle = session.battle
            if battle.user.name is not None and battle.user.name == battle.opponent.name:
                # Foul Play reads the opponent's name out of the room title by removing its own
                # name from it. With one name contained in the other it picks the wrong side.
                session.state = _Session.FAILED
                session.error = ("Foul Play took %s for both sides: the player's name %r must not be part of the "
                                 "opponent's name or the other way round" % (battle.user.name, session.user))
                self.counters["failed_battles"] += 1
        if session.state == _Session.FINISHED:
            self.counters["finished_battles"] += 1
            self._forget(key, "the battle is over")
            self._check_data()

    def _check_data(self) -> None:
        """As Foul Play does after each battle: its data tables must be as they were loaded."""
        if self.fp_data.pokedex != self.original_pokedex or self.fp_data.all_move_json != self.original_moves:
            # Foul Play exits here. So does this process: the bridge starts a new one, with fresh
            # tables, and rebuilds the battles that were in progress.
            sys.stderr.write("fp_bridge_server: Foul Play's pokedex or move table was changed in memory. Stopping.\n")
            sys.stderr.flush()
            os._exit(86)

    def _forget(self, key: str, why: str) -> None:
        session = self.sessions.pop(key, None)
        if session is not None and session.coro is not None:
            try:
                session.coro.close()
            except BaseException:
                pass
        self.gone[key] = why
        self.gone.move_to_end(key)
        while len(self.gone) > _TOMBSTONES:
            self.gone.popitem(last=False)

    def end(self, key: str) -> None:
        self._forget(key, "the battle was ended by the player")

    # ---- names --------------------------------------------------------------------------------

    def _prepared(self, session: _Session) -> Any:
        """The battle as Foul Play prepares it for a decision: a copy, updated from the request."""
        battle = session.battle
        if session.prepared is not None and session.prepared_rqid == battle.rqid:
            return session.prepared
        copy = deepcopy(battle)                       # the first two statements of async_pick_move
        if not copy.team_preview:
            copy.user.update_from_request_json(copy.request_json)
        session.prepared, session.prepared_rqid = copy, battle.rqid
        return copy

    def _slot_names(self, prepared: Any) -> Tuple[List[Optional[str]], Dict[int, str]]:
        """Foul Play's name for the move in every request slot, and for the Pokemon at every index."""
        request = prepared.request_json or {}
        moves: List[Optional[str]] = []
        active = request.get("active") or []
        for move in (active[0].get("moves") or []) if active else []:
            name = move.get("id") or ""
            if name == self.constants.HIDDEN_POWER:
                name = self.normalize_name(move.get("move") or name)
            try:
                moves.append(self.fp_state.Move(name).name)      # Foul Play's own spelling
            except KeyError:
                moves.append(None)
        switches = {int(p.index): str(p.name) for p in prepared.user.reserve if p.index is not None}
        return moves, switches

    def _choice_of(self, prepared: Any, names: Tuple[List[Optional[str]], Dict[int, str]], fp_name: str) -> Optional[str]:
        """"earthquake-tera" or "switch garchomp" as a Showdown choice with slot numbers."""
        moves, switches = names
        if fp_name.startswith("switch "):
            target = fp_name[len("switch "):]
            for index, name in switches.items():
                if name == target:
                    return "switch %d" % index
            return None
        flag = ""
        base = fp_name
        for suffix, word in (("-tera", " terastallize"), ("-mega", " mega")):
            if base.endswith(suffix):
                base, flag = base[: -len(suffix)], word
                break
        if base in moves:
            return "move %d%s" % (moves.index(base) + 1, flag)
        try:
            move = prepared.user.active.get_move(base)           # Foul Play's own loose matching
        except Exception:
            move = None
        if move is not None and move.name in moves:
            return "move %d%s" % (moves.index(move.name) + 1, flag)
        return None

    def _choice_of_message(self, prepared: Any, names: Tuple[List[Optional[str]], Dict[int, str]], message: str) -> Optional[str]:
        """Foul Play's "/choose move earthquake terastallize" or "/switch 3" with slot numbers."""
        parts = message.split()
        if len(parts) == 2 and parts[0] == "/switch" and parts[1].isdigit():
            return "switch %s" % parts[1]
        if len(parts) >= 3 and parts[0] == "/choose" and parts[1] == "move":
            base = self._choice_of(prepared, names, parts[2])
            if base is None:
                return None
            flags = [_FLAG_WORDS[p] for p in parts[3:] if p in _FLAG_WORDS]
            return " ".join([base] + flags)
        return None

    def _fp_name_of(self, session: _Session, choice: str) -> Optional[str]:
        """The reverse: "move 2 terastallize" in Foul Play's words, for last_selected_move."""
        prepared = self._prepared(session)
        moves, switches = self._slot_names(prepared)
        request = prepared.request_json or {}
        parts = (choice or "").split()
        if parts and parts[0] == "default":
            # what the server does with "/choose default": the first option it would accept
            if request.get("forceSwitch"):
                for i, mon in enumerate((request.get("side") or {}).get("pokemon") or []):
                    if not mon.get("active") and not str(mon.get("condition", "")).endswith(" fnt") and (i + 1) in switches:
                        return "switch " + switches[i + 1]
                return None
            active = request.get("active") or []
            for i, move in enumerate((active[0].get("moves") or []) if active else []):
                if not move.get("disabled") and i < len(moves) and moves[i]:
                    return moves[i]
            return None
        if len(parts) < 2 or not parts[1].isdigit():
            return None
        index = int(parts[1])
        if parts[0] == "switch":
            name = switches.get(index)
            return None if name is None else "switch " + name
        if parts[0] == "move" and 1 <= index <= len(moves):
            return moves[index - 1]
        return None

    @staticmethod
    def _is_revive_request(battle: Any) -> bool:
        """The request after Revival Blessing: a forced "switch" whose targets are the fainted Pokemon."""
        request = battle.request_json or {}
        if not request.get("forceSwitch"):
            return False
        return any(mon.get("reviving") for mon in (request.get("side") or {}).get("pokemon") or [])

    def chose(self, key: str, choice: str) -> None:
        session = self.sessions.get(key)
        if session is None or session.state != _Session.RUNNING or session.battle is None:
            return
        self.counters["notes"] += 1
        battle = session.battle
        if self._is_revive_request(battle):
            # Whom to revive is sent as "switch N", but nothing enters the field and no switch line
            # follows. Recorded as a selected switch it stops Foul Play's speed inference on the
            # next turn (KeyError in check_speed_ranges: seen in 2 of 12 recorded battles). The
            # move it selected stays Revival Blessing.
            self.counters["notes_revive"] += 1
            return
        try:
            self.config.username = session.user
            name = self._fp_name_of(session, choice)
            if name is None:
                self.counters["notes_unmapped"] += 1
                return
            # As async_pick_move records its own pick: who moved, the move without flags, the turn.
            battle.user.last_selected_move = self.fp_state.LastUsedMove(battle.user.active.name, name, battle.turn)
        except Exception as exc:
            self.counters["notes_unmapped"] += 1
            sys.stderr.write("fp_bridge_server: could not note %r for %s: %r\n" % (choice, key, exc))

    # ---- deciding -----------------------------------------------------------------------------

    @staticmethod
    def search_seconds(ms: int, parallelism: int = 1, workers: int = 1) -> float:
        """
        What one decision takes: Foul Play searches 4 worlds at half `ms` or 2 at `ms`, and
        prepares them before (measured: 54, 118, 226 and 446 ms at 20, 50, 100 and 200 ms).
        With "parallelism" p it samples p times as many worlds (Foul Play's search_params), and
        here they are searched one after the other (the inline pool): p times as long.
        With "workers" w they are searched w at a time: ceil(2p / w) rounds of `ms` (or
        ceil(4p / w) of half), plus starting the w processes. foulplay_bridge.search_seconds and
        search_player.minimum_deadline make the same sum.
        """
        ms, p, w = max(2, int(ms)), max(1, int(parallelism or 1)), max(1, int(workers or 1))
        if w <= 1:
            return 2.5 * ms * p / 1000.0 + 0.02
        rounds = max(math.ceil(2 * p / w), math.ceil(4 * p / w) / 2.0)
        return 1.25 * ms * rounds / 1000.0 + 0.02 + FORK_S_PER_WORKER * w

    # ---- the opponent's side of the search, and the payoff table --------------------------------

    @staticmethod
    def _opponent_name(fp_name: str) -> str:
        """poke-engine's name of an option of the opponent, in words: "move earthquake terastallize", "switch garchomp", "none"."""
        if fp_name.startswith("switch "):
            return fp_name
        if fp_name in ("No Move", "none", ""):
            return "none"
        for suffix, word in (("-tera", " terastallize"), ("-mega", " mega")):
            if fp_name.endswith(suffix):
                return "move " + fp_name[: -len(suffix)] + word
        return "move " + fp_name

    @staticmethod
    def _plain(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return str(getattr(value, "name", value))

    def _set_of(self, pokemon: Any) -> Dict[str, Any]:
        """A Pokemon as it was sampled for a world: everything the search assumed about it."""
        out: Dict[str, Any] = {"species": str(pokemon.name)}
        for name in ("level", "hp", "max_hp", "status", "item", "ability", "tera_type", "terastallized"):
            out[name] = self._plain(getattr(pokemon, name, None))
        out["moves"] = [str(getattr(m, "name", m)) for m in (getattr(pokemon, "moves", None) or [])]
        return out

    def _opponent(self, prepared: Any, results: List[Any], worlds: List[Any]) -> Dict[str, Any]:
        """Side two of every world's search: the search's own prediction of the opponent."""
        revealed = [str(p.name) for p in (prepared.opponent.reserve or [])]
        active = prepared.opponent.active
        table: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        per_world: List[Dict[str, Any]] = []
        for index, (result, chance, _index) in enumerate(results):
            total = int(result.total_visits)
            options: List[Dict[str, Any]] = []
            for option in result.side_two:
                name = self._opponent_name(option.move_choice)
                share = (option.visits / total) if total else 0.0
                options.append({"name": name, "fp": option.move_choice, "visits": int(option.visits), "share": round(share, 6),
                                "score": round(option.total_score / option.visits, 5) if option.visits else None})
                row = table.get(name)
                if row is None:
                    row = table[name] = {"name": name, "fp": option.move_choice, "visits": 0, "total_score": 0.0, "share": 0.0, "worlds": 0}
                row["visits"] += int(option.visits)
                row["total_score"] += float(option.total_score)
                row["share"] += float(chance) * share
                row["worlds"] += 1
            world: Dict[str, Any] = {"chance": round(float(chance), 6), "iterations": total, "options": options}
            if index < len(worlds):
                sampled = worlds[index][0]
                try:
                    if sampled.opponent.active is not None:
                        world["active"] = self._set_of(sampled.opponent.active)
                    world["reserve"] = [str(p.name) for p in (sampled.opponent.reserve or [])]
                except Exception as exc:
                    world["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
            per_world.append(world)
        options_out: List[Dict[str, Any]] = []
        for row in table.values():
            entry = {"name": row["name"], "fp": row["fp"], "share": round(row["share"], 6), "visits": row["visits"],
                     "score": round(row["total_score"] / row["visits"], 5) if row["visits"] else None, "worlds": row["worlds"]}
            if row["name"].startswith("switch "):
                entry["seen"] = row["name"][len("switch "):] in revealed
            options_out.append(entry)
        options_out.sort(key=lambda o: -o["share"])
        return {"active": None if active is None else str(active.name), "revealed": revealed,
                "options": options_out, "worlds": per_world}

    @staticmethod
    def _engine_argument(fp_name: str) -> str:
        """An option as generate_instructions wants it: the Pokemon's name for a switch, "none" for no move."""
        if fp_name.startswith("switch "):
            return fp_name[len("switch "):]
        return "none" if fp_name == "No Move" else fp_name

    @staticmethod
    def _maximin(result: Any) -> Optional[float]:
        """The value of an expectiminimax result for side one: its best row by its worst known cell."""
        n1, n2 = len(result.side_one), len(result.side_two)
        best: Optional[float] = None
        for i in range(n1):
            worst: Optional[float] = None
            for value in result.matrix[i * n2:(i + 1) * n2]:
                if value is None or value != value:      # pruned: this row was already known to be worse
                    continue
                if worst is None or value < worst:
                    worst = value
            if worst is not None and (best is None or worst > best):
                best = worst
        return best

    def _payoff_engine(self, text: str, share_ms: float) -> Dict[str, Any]:
        engine = self.poke_engine
        result = engine.iterative_deepening_expectiminimax(engine.State.from_string(text), max(1, int(share_ms)))
        n2 = len(result.side_two)
        rows = [[None if (v is None or v != v) else round(float(v), 2) for v in result.matrix[i * n2:(i + 1) * n2]]
                for i in range(len(result.side_one))]
        return {"depth": int(result.depth_searched), "ours": list(result.side_one), "theirs_fp": list(result.side_two), "matrix": rows}

    def _payoff_full(self, text: str, result: Any, share_ms: float) -> Dict[str, Any]:
        engine = self.poke_engine
        ours = [o.move_choice for o in result.side_one]
        theirs = [o.move_choice for o in result.side_two]
        root = engine.State.from_string(text)
        rows: List[List[Optional[float]]] = [[None] * len(theirs) for _ in ours]
        deadline = time.perf_counter() + share_ms / 1000.0
        # The pairs the Monte Carlo search visited most come first: under a short budget the cells
        # that are there are the ones that matter.
        order = sorted(((i, j) for i in range(len(ours)) for j in range(len(theirs))),
                       key=lambda ij: -(result.side_one[ij[0]].visits + 1) * (result.side_two[ij[1]].visits + 1))
        searches, failed, depth = 0, 0, None
        for i, j in order:
            if time.perf_counter() >= deadline:
                break
            try:
                total, weight = 0.0, 0.0
                for outcome in engine.generate_instructions(root, self._engine_argument(ours[i]), self._engine_argument(theirs[j])):
                    child = root.apply_instructions(outcome)
                    below = engine.iterative_deepening_expectiminimax(child, 1)
                    searches += 1
                    value = self._maximin(below)
                    if value is None:
                        continue
                    depth = below.depth_searched if depth is None else min(depth, below.depth_searched)
                    total += float(outcome.percentage) * value
                    weight += float(outcome.percentage)
                if weight > 0:
                    rows[i][j] = round(total / weight, 2)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:      # a Rust panic arrives as a BaseException: that cell stays unknown
                failed += 1
        return {"depth": 1 + int(depth or 0), "ours": ours, "theirs_fp": theirs, "matrix": rows, "searches": searches, "failed": failed}

    def _payoff(self, prepared: Any, names: Any, results: List[Any], worlds: List[Any], budget_ms: float, mode: str,
                as_offered: Any) -> Dict[str, Any]:
        """A table of values per (our option, opponent's option) for every sampled world."""
        started = time.perf_counter()
        mode = "full" if mode == "full" else "engine"
        out: List[Dict[str, Any]] = []
        count = max(1, min(len(worlds), len(results)))
        for index in range(min(len(worlds), len(results))):
            result, chance, _index = results[index]
            began = time.perf_counter()
            # What is left of the budget, shared among the worlds still to do.
            left = budget_ms - (began - started) * 1000.0
            share_ms = max(1.0, left / (count - index))
            entry: Dict[str, Any] = {"chance": round(float(chance), 6)}
            try:
                text = worlds[index][1].to_string()
                table = self._payoff_full(text, result, share_ms) if mode == "full" else self._payoff_engine(text, share_ms)
                entry.update(table)
                entry["choices"] = [as_offered(self._choice_of(prepared, names, name)) for name in table["ours"]]
                entry["theirs"] = [self._opponent_name(name) for name in entry.pop("theirs_fp")]
                entry["cells"] = sum(len(row) for row in table["matrix"])
                entry["known"] = sum(1 for row in table["matrix"] for value in row if value is not None)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:
                self.counters["payoff_errors"] += 1
                entry["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
            entry["ms"] = round((time.perf_counter() - began) * 1000.0, 2)
            out.append(entry)
        elapsed = (time.perf_counter() - started) * 1000.0
        self.counters["payoffs"] += 1
        self.counters["payoff_ms"] += elapsed
        return {"mode": mode, "budget_ms": budget_ms, "ms": round(elapsed, 2),
                "unit": "engine evaluation, larger is better for us", "worlds": out}

    # ---- lines: the search's favourite actions, each played forward a few turns -----------------

    _LINE_STATUS = {"BURN": "brn", "SLEEP": "slp", "FREEZE": "frz", "PARALYZE": "par", "POISON": "psn", "TOXIC": "tox"}

    @classmethod
    def _snapshot(cls, text: str) -> Dict[str, Any]:
        """
        What a line shows of an engine state (State.to_string(); side one is ours): per side the
        active Pokemon with its HP in percent, status and boosts, how many Pokemon are left and
        the share of the team's total HP that is, and whether the side has to switch now.
        """
        out: Dict[str, Any] = {}
        for who, side_text in zip(("ours", "theirs"), text.split("/")[:2]):
            f = side_text.split("=")
            mons: List[Tuple[str, int, int, str]] = []
            for k in range(6):
                p = f[k].split(",")
                mons.append((p[0].lower(), int(p[6]), int(p[7]), cls._LINE_STATUS.get(p[18].upper(), "")))
            index = int(f[6])
            real = [m for m in mons if m[0] not in ("", "none")]
            total = sum(m[2] for m in real)
            name, hp, maxhp, status = mons[index]
            side: Dict[str, Any] = {
                "active": name, "hp": int(math.ceil(100.0 * max(0, hp) / maxhp)) if maxhp > 0 else 0,
                "left": sum(1 for m in real if m[1] > 0),
                "team_hp": round(sum(max(0, min(m[1], m[2])) for m in real) / total, 3) if total else 0.0,
            }
            if status and hp > 0:
                side["status"] = status
            boosts = {stat: int(f[11 + i]) for i, stat in enumerate(("atk", "def", "spa", "spd", "spe")) if int(f[11 + i])}
            if boosts and hp > 0:
                side["boosts"] = boosts
            if f[22] == "true":
                side["must_switch"] = True
            out[who] = side
        return out

    def _line_value(self, state: Any, snap: Dict[str, Any], step_ms: float, threads: int = 1) -> Tuple[Optional[float], Any]:
        """(the search's mean result for our side at this state, 0 lost to 1 won; its result) after step_ms of search on `threads` threads. A side with nobody left: 0 or 1, no search."""
        if snap["ours"]["left"] <= 0:
            return 0.0, None
        if snap["theirs"]["left"] <= 0:
            return 1.0, None
        result = self.poke_engine.monte_carlo_tree_search(state, max(2, int(step_ms)), 0, max(1, int(threads)))
        visits = sum(int(o.visits) for o in result.side_one)
        if not visits:
            return None, result
        return sum(float(o.total_score) for o in result.side_one) / visits, result

    def _line_from(self, root: Any, forced: bool, ours: str, theirs: str, turns: int, step_ms: float, deadline: float,
                   threads: int = 1) -> Dict[str, Any]:
        """
        One line: from the root we play `ours` and the opponent `theirs`; of the outcomes the
        engine gives that pair the likeliest one is followed, the position is searched for
        step_ms, both sides play the action that search visited most, and so on until `turns`
        whole turns are played (a forced switch after a faint is a step, not a turn), a side has
        nobody left or the deadline is reached. The value of the line is the search's mean
        result at its last position. The first step's other outcomes (a miss, a critical hit, a
        secondary effect: every one with 5% or more) are each searched once and weighed in by
        their chance, so that a move that misses a third of the time is not valued as a sure hit.
        """
        engine = self.poke_engine
        state = root
        steps: List[Dict[str, Any]] = []
        value: Optional[float] = None
        played, searches, cut = 0, 0, False
        others: List[Tuple[float, Optional[float]]] = []
        main_share = 1.0
        for step in range(turns + 6):
            outcomes = list(engine.generate_instructions(state, self._engine_argument(ours), self._engine_argument(theirs)))
            if not outcomes:
                break
            total = sum(float(o.percentage) for o in outcomes) or 1.0
            outcomes.sort(key=lambda o: -float(o.percentage))
            child = state.apply_instructions(outcomes[0])
            snap = self._snapshot(child.to_string())
            share = float(outcomes[0].percentage) / total
            entry: Dict[str, Any] = {"ours": self._opponent_name(ours), "theirs": self._opponent_name(theirs),
                                     "p": round(share, 3), "after": snap}
            if forced:
                entry["forced"] = True
            else:
                played += 1
            steps.append(entry)
            if step == 0:
                main_share = share
                for other in outcomes[1:4]:
                    part = float(other.percentage) / total
                    if part < 0.05 or time.perf_counter() >= deadline:
                        continue
                    side = state.apply_instructions(other)
                    worth, _result = self._line_value(side, self._snapshot(side.to_string()), step_ms, threads)
                    searches += 1
                    others.append((part, worth))
            state = child
            value, result = self._line_value(state, snap, step_ms, threads)
            if result is None:          # a side has nobody left: the game is over on this line
                break
            searches += 1
            if played >= turns:
                break
            if time.perf_counter() >= deadline:
                cut = True
                break
            if not result.side_one or not result.side_two:
                break
            ours = max(result.side_one, key=lambda o: o.visits).move_choice
            theirs = max(result.side_two, key=lambda o: o.visits).move_choice
            forced = bool(snap["ours"].get("must_switch") or snap["theirs"].get("must_switch"))
        out: Dict[str, Any] = {"steps": steps, "turns": played, "searches": searches,
                               "end": None if value is None else round(value, 4)}
        if cut:
            out["cut"] = True
        # The line's value: its end, and the first step's other outcomes by their chance.
        known = [(main_share, value)] + others
        weight = sum(p for p, v in known if v is not None)
        out["value"] = round(sum(p * v for p, v in known if v is not None) / weight, 4) if weight > 0 and value is not None else None
        if others:
            out["first_step"] = [{"p": round(p, 3), "value": None if v is None else round(v, 4)} for p, v in others]
        return out

    def _lines(self, prepared: Any, names: Any, results: List[Any], worlds: List[Any], table: Any, spec: Dict[str, Any],
               as_offered: Any) -> Dict[str, Any]:
        """
        "lines" of a decision: our `k` options the search favours most (its policy over all the
        sampled worlds), each played forward `turns` turns in the `worlds` likeliest sampled
        worlds (_line_from), within about `ms` in all. Per world and option: the steps, the end
        value and the value with the first step's other outcomes weighed in.
        """
        started = time.perf_counter()
        k = max(1, min(6, int(spec.get("k") or 3)))
        turns = max(1, min(8, int(spec.get("turns") or 5)))
        budget_ms = max(50.0, min(20000.0, float(spec.get("ms") or 1500.0)))
        most = max(1, min(4, int(spec.get("worlds") or 2)))
        threads = max(1, min(8, int(spec.get("threads") or 1)))        # poke-engine's own threads for each of the lines' searches
        ranked = sorted(table.values(), key=lambda row: -row["policy"])
        candidates = [row["fp"] for row in ranked[:k] if row["policy"] > 0]
        count = min(len(worlds), len(results))
        order = sorted(range(count), key=lambda i: -float(results[i][1]))[:most]
        # A line is turns + 1 searches or so, and the first step's other outcomes one each.
        step_ms = max(2.0, budget_ms / max(1, len(order) * max(1, len(candidates)) * (turns + 2)))
        deadline = started + 1.25 * budget_ms / 1000.0
        out: List[Dict[str, Any]] = []
        for index in order:
            result, chance, _index = results[index]
            entry: Dict[str, Any] = {"world": index, "chance": round(float(chance), 6), "lines": []}
            try:
                text = worlds[index][1].to_string()
                root = self.poke_engine.State.from_string(text)
                before = self._snapshot(text)
                forced = bool(before["ours"].get("must_switch") or before["theirs"].get("must_switch"))
                theirs = max(result.side_two, key=lambda o: o.visits).move_choice if result.side_two else "No Move"
                offered = {o.move_choice for o in result.side_one}
                for fp_name in candidates:
                    if fp_name not in offered or time.perf_counter() >= deadline:
                        continue
                    began = time.perf_counter()
                    try:
                        line = self._line_from(root, forced, fp_name, theirs, turns, step_ms, deadline, threads)
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except BaseException as exc:      # a Rust panic arrives as a BaseException: this line is left out
                        self.counters["line_errors"] += 1
                        line = {"steps": [], "turns": 0, "value": None, "end": None,
                                "error": "%s: %s" % (type(exc).__name__, str(exc)[:160])}
                    line["fp"] = fp_name
                    line["choice"] = as_offered(self._choice_of(prepared, names, fp_name))
                    line["ms"] = round((time.perf_counter() - began) * 1000.0, 1)
                    entry["lines"].append(line)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:
                self.counters["line_errors"] += 1
                entry["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
            out.append(entry)
        elapsed = (time.perf_counter() - started) * 1000.0
        self.counters["lines"] += 1
        self.counters["lines_ms"] += elapsed
        return {"k": k, "turns": turns, "budget_ms": budget_ms, "step_ms": round(step_ms, 1), "ms": round(elapsed, 2), "threads": threads,
                "candidates": [as_offered(self._choice_of(prepared, names, fp_name)) for fp_name in candidates], "worlds": out}

    def decide(self, key: str, ms: int, parallelism: int = 1, rqid: Any = None, fixed_iterations: int = 0, threads: int = 0,
               payoff_ms: float = 0.0, payoff_mode: str = "engine", workers: int = 0,
               lines: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        session = self.sessions.get(key)
        if session is None:
            return {"ok": False, "error": self.gone.get(key) or "unknown battle: nothing was fed for it"}
        if session.state == _Session.FAILED:
            return {"ok": False, "error": session.error or "the battle failed", "failed": True}
        battle = session.battle
        if battle is None:
            return {"ok": False, "error": "the battle has not reached its first request (%d messages fed)" % session.messages}
        if session.pick_rqid != battle.rqid or battle.wait:
            return {"ok": False, "error": "Foul Play has no move to make for request %s (a wait request)" % battle.rqid}
        if rqid is not None and str(rqid) != str(battle.rqid):
            return {"ok": False, "error": "asked about request %s, Foul Play is at request %s" % (rqid, battle.rqid)}
        if self._is_revive_request(battle):
            # Foul Play's engine does not know the second half of Revival Blessing. It takes this
            # request for a forced switch and answers with a LIVING Pokemon, which the server
            # refuses (12 of 12 recorded cases), or divides by zero when none lives. No answer is
            # the honest one: the player picks whom to revive.
            self.counters["revive_requests"] += 1
            return {"ok": False, "revive": True,
                    "error": "Foul Play has no answer for a Revival Blessing request (whom to revive): its engine does not know that half of the move"}

        self.config.username = session.user
        self.config.search_time_ms = max(2, int(ms))   # Foul Play halves it early in a game: never 0 ms
        self.config.parallelism = max(1, int(parallelism))
        self.fixed_iterations = max(0, int(fixed_iterations or 0))
        self.search_threads = max(0, int(threads or 0))
        self.search_workers = max(0, int(workers or 0))
        self.capture.clear()
        started = time.perf_counter()
        # Foul Play's own decision function, as its battle loop would have called it right here.
        try:
            message, message_rqid = self.loop.run_until_complete(self.original_pick(battle))
        finally:
            self.fixed_iterations = 0
            self.search_threads = 0
            self.search_workers = 0
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        prepared = self.capture.get("prepared")
        results = self.capture.get("results") or []
        fp_choice = self.capture.get("choice")
        session.prepared, session.prepared_rqid = prepared, battle.rqid
        names = self._slot_names(prepared)

        # The same sum Foul Play makes in select_move_from_mcts_results, kept per option.
        table: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        iterations: List[int] = []
        for result, chance, _index in results:
            iterations.append(int(result.total_visits))
            for option in result.side_one:
                row = table.get(option.move_choice)
                if row is None:
                    row = table[option.move_choice] = {"fp": option.move_choice, "visits": 0, "total_score": 0.0, "policy": 0.0}
                row["visits"] += int(option.visits)
                row["total_score"] += float(option.total_score)
                if result.total_visits:
                    row["policy"] += float(chance) * (option.visits / result.total_visits)
        # Foul Play's engine offers Terastallization whenever its side has not used it. The server
        # does not while the Pokemon is locked into a move (Outrage, the second turn of a charge
        # move): the request then has no "canTerastallize", and the server plays Foul Play's
        # "move outrage terastallize" as plain Outrage (sim/side.ts, chooseMove, the lockedMove
        # branch returns before it looks at the modifier). So that is what such a choice means.
        request = prepared.request_json or {}
        active = request.get("active") or []
        offers_tera = bool(active and active[0].get("canTerastallize"))

        def as_offered(choice: Optional[str]) -> Optional[str]:
            if choice and not offers_tera and choice.endswith(" terastallize"):
                return choice[: -len(" terastallize")]
            return choice

        top = max((row["policy"] for row in table.values()), default=0.0)
        options: List[Dict[str, Any]] = []
        share: Dict[Optional[str], float] = {}
        for row in table.values():
            literal = self._choice_of(prepared, names, row["fp"])
            choice = as_offered(literal)
            share[choice] = share.get(choice, 0.0) + row["policy"]
            option = {
                "choice": choice,
                "fp": row["fp"],
                "visits": row["visits"],
                "total_score": round(row["total_score"], 3),
                "score": round(row["total_score"] / row["visits"], 5) if row["visits"] else None,
                "policy": round(row["policy"], 6),
                "considered": bool(top > 0 and row["policy"] >= top * 0.75),
            }
            if choice != literal:
                option["tera_dropped"] = True
            options.append(option)
        best = max(share, key=lambda c: share[c]) if share else None
        literal_choice = self._choice_of_message(prepared, names, message)
        chosen = as_offered(literal_choice)

        session.decisions += 1
        self.counters["decisions"] += 1
        self.counters["decide_ms"] += elapsed_ms
        num_worlds, per_world_ms = len(results), None
        try:
            _n, per_world_ms = battle.mode.search_params(prepared)
        except Exception:
            pass
        worlds = self.capture.get("worlds") or []
        opponent: Optional[Dict[str, Any]] = None
        try:
            opponent = self._opponent(prepared, results, worlds)
        except Exception as exc:    # the decision itself is Foul Play's and stands
            sys.stderr.write("fp_bridge_server: the opponent's side of %s could not be read: %r\n" % (key, exc))
        payoff: Optional[Dict[str, Any]] = None
        if payoff_ms and payoff_ms > 0:
            payoff = self._payoff(prepared, names, results, worlds, float(payoff_ms), payoff_mode, as_offered)
        lines_out: Optional[Dict[str, Any]] = None
        if lines:
            try:
                lines_out = self._lines(prepared, names, results, worlds, table, lines, as_offered)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:    # the decision itself is Foul Play's and stands
                self.counters["line_errors"] += 1
                lines_out = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:200]), "worlds": []}
        reply = {
            "ok": True,
            "choice": chosen,
            "tera_dropped": bool(chosen != literal_choice),
            "best": best,
            "fp_choice": fp_choice,
            "fp_message": message,
            "options": options,
            "worlds": num_worlds,
            "iterations": sum(iterations),
            "iterations_per_world": iterations,
            "search_ms": per_world_ms,
            "ms": round(elapsed_ms, 2),
            "turn": battle.turn,
            "rqid": battle.rqid,
            "force_switch": bool(battle.force_switch),
        }
        if opponent is not None:
            reply["opponent"] = opponent
        if payoff is not None:
            reply["payoff"] = payoff
        if lines_out is not None:
            reply["lines"] = lines_out
        if fixed_iterations:
            reply["iterations_asked"] = int(fixed_iterations)
        if threads:
            reply["threads"] = int(threads)
        if int(parallelism or 1) > 1:
            reply["parallelism"] = int(parallelism)
        if int(workers or 0) > 1:
            reply["workers"] = int(workers)
        return reply

    # ---- protocol -----------------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        states: Dict[str, int] = {}
        for session in self.sessions.values():
            states[session.state] = states.get(session.state, 0) + 1
        out: Dict[str, Any] = dict(self.counters)
        out["decide_ms"] = round(out["decide_ms"], 1)
        out.update({"battles": len(self.sessions), "states": states, "remembered_gone": len(self.gone),
                    "rss_mb": _rss_mb(), "peak_rss_mb": _peak_rss_mb(), "uptime_s": round(time.time() - self.started, 1),
                    "pid": os.getpid()})
        return out

    def handle(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        op = message.get("op")
        if op == "feed":
            self.feed(str(message["key"]), str(message.get("user") or ""), str(message["room"]), str(message.get("data") or ""))
            return None
        if op == "chose":
            self.chose(str(message["key"]), str(message.get("choice") or ""))
            return None
        if op == "end":
            self.end(str(message["key"]))
            return None
        if op == "decide":
            key = str(message["key"])
            ms = int(message.get("ms") or 100)
            deadline = message.get("deadline")
            payoff_ms = float(message.get("payoff_ms") or 0.0)
            parallelism = max(1, int(message.get("parallelism") or 1))
            workers = max(0, int(message.get("workers") or 0))
            lines = message.get("lines") if isinstance(message.get("lines"), dict) else None
            lines_s = 1.25 * max(0.0, float((lines or {}).get("ms") or 0.0)) / 1000.0
            if deadline is not None and (time.monotonic() + self.search_seconds(ms, parallelism, workers) + payoff_ms / 1000.0
                                         + lines_s > float(deadline)):
                # The player that asked has given up, or will have before a search could end: it
                # has moved on with another move. Searching now would only make every question
                # behind this one late as well, so that under overload nothing is answered in time.
                self.counters["expired"] += 1
                return {"ok": False, "expired": True, "key": key, "clock": time.monotonic(),
                        "error": "not searched: nobody waits for this decision any more (it waited too long behind others in its helper)"}
            try:
                reply = self.decide(key, ms, parallelism, message.get("rqid"),
                                    fixed_iterations=int(message.get("iterations") or 0), threads=int(message.get("threads") or 0),
                                    payoff_ms=payoff_ms, payoff_mode=str(message.get("payoff") or "engine"), workers=workers,
                                    lines=lines)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:
                sys.stderr.write("fp_bridge_server: decision for %s failed\n%s\n" % (key, traceback.format_exc(limit=8)))
                reply = {"ok": False, "error": "Foul Play's decision failed: %s: %s" % (type(exc).__name__, str(exc)[:300])}
            if not reply.get("ok"):
                self.counters["decide_errors"] += 1
            reply["key"] = key
            reply["clock"] = time.monotonic()     # the caller keeps its clock and this one in step (see "ping")
            return reply
        if op == "ping":
            # "clock": the caller checks that both read the same clock before it sends deadlines.
            return {"ok": True, "version": PROTOCOL_VERSION, "pid": os.getpid(), "format": self.battle_format, "clock": time.monotonic(),
                    "pool": self.pool, "foulplay_dir": self.fp_dir, "foulplay_commit": _read_commit(self.fp_dir),
                    "poke_engine": self.engine_version, "python": sys.version.split()[0], "rss_mb": _rss_mb(),
                    "ppid": os.getppid() if hasattr(os, "getppid") else None,
                    "pgid": os.getpgrp() if hasattr(os, "getpgrp") else None}
        if op == "sync":
            return {"ok": True}
        if op == "stats":
            return dict(self.stats(), ok=True)
        if op == "exit":
            raise SystemExit(0)
        return {"ok": False, "error": "unknown op %r" % (op,)}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Foul Play as an adviser, JSON lines on stdin and stdout.")
    parser.add_argument("--foulplay-dir", default=os.environ.get("FOULPLAY_DIR") or "~/foul-play")
    parser.add_argument("--format", default=DEFAULT_FORMAT, dest="battle_format")
    parser.add_argument("--pool", choices=("inline", "fork"), default=os.environ.get("FP_BRIDGE_POOL") or "inline",
                        help="inline: search in this process. fork: Foul Play's own process pool, a fork per decision.")
    parser.add_argument("--log-level", default=os.environ.get("FP_BRIDGE_LOG_LEVEL") or "WARNING")
    parser.add_argument("--parent-watch", choices=("env", "ppid", "beat"), default="env",
                        help="env: the parent is FP_BRIDGE_PARENT_PID. ppid: the parent is whoever it is at the start. "
                             "beat: that, and leave when no line has come for %d s while idle (a helper started "
                             "through wsl.exe from Windows: the bridge sends a line every 2 s)." % int(BEAT_TIMEOUT_S))
    parser.add_argument("--own-group", action="store_true", help="become the leader of a process group of its own")
    parser.add_argument("--tag", default="", help="not used here: a mark on the command line by which the bridge finds this process")
    args = parser.parse_args(argv)

    channel = _take_stdout()
    if args.own_group:
        _own_group()
    _watch_parent(args.parent_watch)
    fp_dir = os.path.abspath(os.path.expanduser(args.foulplay_dir))
    if not os.path.isdir(os.path.join(fp_dir, "fp")):
        channel.write(json.dumps({"id": 0, "ok": False, "fatal": "%s is not a Foul Play checkout" % fp_dir}) + "\n")
        return 2
    os.chdir(fp_dir)
    sys.path[0] = fp_dir      # sys.path[0] is this file's folder: Foul Play must not see the repository's modules
    try:
        server = _Server(fp_dir, args.battle_format, args.pool, args.log_level)
    except SystemExit as exc:
        channel.write(json.dumps({"id": 0, "ok": False, "fatal": str(exc)}) + "\n")
        return 2
    except BaseException as exc:
        traceback.print_exc()
        channel.write(json.dumps({"id": 0, "ok": False, "fatal": "could not load Foul Play: %r" % (exc,)}) + "\n")
        return 2

    stdin = sys.stdin
    try:
        stdin.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    _ALIVE["last"], _ALIVE["busy"] = time.monotonic(), False      # loaded: from here on silence means nobody is there
    came = time.monotonic()
    for line in stdin:
        got = time.monotonic()
        if got - _ALIVE["last"] > 0.002:
            came = got      # it was waiting for input: this line came in now (one read without waiting came no earlier)
        _ALIVE["busy"] = True
        line = line.strip()
        if not line:
            _ALIVE["last"], _ALIVE["busy"] = time.monotonic(), False
            continue
        try:
            message = json.loads(line)
        except ValueError:
            _ALIVE["last"], _ALIVE["busy"] = time.monotonic(), False
            continue
        try:
            reply = server.handle(message) if message.get("op") != "beat" else None
        except SystemExit:
            break
        except BaseException as exc:
            sys.stderr.write("fp_bridge_server: %r failed\n%s\n" % (message.get("op"), traceback.format_exc(limit=8)))
            reply = {"ok": False, "error": "%s: %s" % (type(exc).__name__, str(exc)[:300])}
        if message.get("id") is not None:
            if reply is None:
                reply = {"ok": True}
            reply["id"] = message["id"]
            reply["got"], reply["came"] = got, came
            channel.write(json.dumps(reply, separators=(",", ":")) + "\n")
            channel.flush()
        _ALIVE["last"], _ALIVE["busy"] = time.monotonic(), False
    return 0


if __name__ == "__main__":
    sys.exit(main())
