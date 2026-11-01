"""
Foul Play (https://github.com/pmariglia/foul-play) as a benchmark opponent for gen9randombattle.

    from bench import foulplay

    ok, why = foulplay.available()            # (True, "foul-play <commit> at <dir>") or (False, reason)
    result = await foulplay.run_matches(player, 100, "127.0.0.1", 9000,
                                        parallel=8, search_time_ms=200, log_dir="runs/main/eval/logs")

`player` is a poke-env 0.15 Player that is already connected to ws://host:port. run_matches starts
`parallel` Foul Play processes. Each one logs in under its own name and challenges the player; the
player accepts, so up to `parallel` games run at once against the one player object. Everything
that was started is stopped before the function returns. bench/FOULPLAY.md has the reasoning and
the measured numbers, bench/install_foulplay.sh installs Foul Play.

From a shell, with the training venv (it needs poke-env):

    python bench/foulplay.py                              # is Foul Play installed and usable?
    python bench/foulplay.py --play 8 --port 9000         # 8 games, poke-env's RandomPlayer against it
    python bench/foulplay.py --play 4 --port 9000 --fixed-iterations 20000   # every world searched 20,000 times

How the pieces fit:

  * Foul Play's guest login asks play.pokemonshowdown.com for an assertion over HTTPS. A local
    server started with --no-security accepts an empty assertion, which is what poke-env sends.
    So each process is started through this file (`--child`), which replaces that one login
    method in memory and then runs Foul Play unchanged. The Foul Play checkout is never edited.
  * Foul Play challenges and the player accepts. The other direction cannot run games in
    parallel: Showdown lets a user have one outgoing challenge at a time, and Foul Play drops a
    challenge that arrives while it is still leaving its previous battle.
  * With fixed_iterations=N (evaluate.py --foulplay-iterations N) every sampled world is searched
    for exactly N iterations, whatever the clock says: poke-engine's own `iterations` argument, the
    path bench/fp_bridge_server.py already takes for a decision that asks for a number of
    iterations. The child replaces fp.search.main.monte_carlo_tree_search in memory (see
    _fix_iterations); the checkout is still never edited. Foul Play then plays at the same
    strength on an idle machine and on a loaded one, at the price of the time. Its search time
    still decides how many worlds it samples (search_params), not how long each is searched.
    Every "Iterations" line of its log is checked against N, and the dict says how many were not.
  * The result is counted from the player's own battle objects, for the battles that ran to
    their natural end while their Foul Play process was being watched. A game that had to be
    stopped (process died, never challenged, ran past the timeout) is an error, not a win.
    Foul Play's own log ("Winner: ...") is compared with that count and the dict says if they agree.

This module imports nothing outside the standard library at import time, because the same file
also runs inside Foul Play's virtualenv, where poke-env is not installed.
"""

from __future__ import annotations

import asyncio
import atexit
import os
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple

__all__ = ["available", "run_matches", "foulplay_dir", "PINNED_COMMIT", "BATTLE_FORMAT"]

# Foul Play main as of 2026-10-03. bench/install_foulplay.sh installs this commit.
PINNED_COMMIT = "8f541f80f91f960603673796559f796b10ccf0fb"
BATTLE_FORMAT = "gen9randombattle"
DEFAULT_DIR = "~/foul-play"

_POLL_S = 0.2            # how often processes, logs and battles are looked at
_STAGGER_S = 0.4         # gap between process starts: Showdown refuses a 4th pending challenge
_LOGIN_WAIT_S = 30.0     # how long to wait for the player to be logged in
_ACCEPT_WAIT_S = 10.0    # challenge accepted -> the battle exists on the player's side
_EXIT_GRACE_S = 15.0     # last game over -> the process should have exited by itself
_ABORT_WAIT_S = 10.0     # /forfeit sent -> the player has seen the battle end
_SETTLE_WAIT_S = 3.0     # process gone -> the player has processed the end of its last game
_CHILD_LOGIN_TIMEOUT_S = 30.0

_THIS_FILE = os.path.abspath(__file__)


def _to_id(name: Any) -> str:
    """Showdown's user id: lower case, letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# =============================================================================================
# The child side. This part runs inside Foul Play's virtualenv:
#     <foul-play>/.venv/bin/python -u bench/foulplay.py --child <foul-play dir> <run.py arguments>
# =============================================================================================

async def _local_login(self: Any) -> str:
    """
    Replaces fp.websocket_client.PSWebsocketClient.login for a server started with --no-security.

    Foul Play's own guest login posts the challenge string to play.pokemonshowdown.com and sends
    back the signed assertion, then sleeps three seconds. That needs the internet, fails for any
    name somebody has registered, and proves nothing to a server that does not check. This sends
    the empty assertion such a server accepts and waits for the server to confirm the name.
    """
    import logging

    log = logging.getLogger("psppo.foulplay")
    want = _to_id(self.username)
    deadline = time.monotonic() + _CHILD_LOGIN_TIMEOUT_S
    sent = False
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            raise RuntimeError("login as %s timed out" % self.username)
        try:
            message = await asyncio.wait_for(self.receive_message(), left)
        except asyncio.TimeoutError:
            raise RuntimeError("login as %s timed out" % self.username)
        for line in str(message).split("\n"):
            parts = line.split("|")
            if len(parts) < 2:
                continue
            if parts[1] == "challstr" and not sent:
                await self.send_message("", ["/trn " + self.username + ",0,"])
                sent = True
            elif parts[1] == "updateuser" and sent and len(parts) > 2:
                # "| Name|1|avatar|{settings}", or " Name@!" when the user is away
                if _to_id(parts[2].split("@")[0]) == want:
                    # Foul Play's log formatter prints record.msg as it is: no %-arguments.
                    log.info("Logged in as {} without an assertion (local server)".format(self.username))
                    return self.username
            elif parts[1] == "nametaken":
                raise RuntimeError("the server refused the name %s: %s" % (self.username, line))


def _watch_parent() -> None:
    """
    Stops this process, and the search workers it forked, when the process that started it is
    gone. Without this a benchmark killed with SIGKILL would leave Foul Play challenging nobody.
    """
    expected = os.environ.get("FOULPLAY_PARENT_PID", "")
    if os.name == "nt" or not expected.isdigit():
        return
    parent = int(expected)

    def watch() -> None:
        while True:
            time.sleep(1.0)
            if os.getppid() != parent:
                try:
                    if os.getpgrp() == os.getpid():
                        os.killpg(os.getpid(), signal.SIGKILL)  # this process and its workers
                finally:
                    os._exit(113)

    threading.Thread(target=watch, name="parent-watch", daemon=True).start()


def _report_usage() -> None:
    """One line for the parent: CPU and peak memory of this process and of its search workers."""
    try:
        import resource

        me = resource.getrusage(resource.RUSAGE_SELF)
        workers = resource.getrusage(resource.RUSAGE_CHILDREN)
        print("INFO     Usage: cpu_s={:.1f} worker_cpu_s={:.1f} rss_mb={} worker_rss_mb={}".format(
            me.ru_utime + me.ru_stime, workers.ru_utime + workers.ru_stime,
            me.ru_maxrss // 1024, workers.ru_maxrss // 1024), flush=True)
    except Exception:
        pass


_FIXED_FLAG = "--fixed-iterations"     # the child's own option, right after the Foul Play directory


def _fix_iterations(n: int) -> None:
    """
    Every sampled world is searched for exactly n iterations, whatever the clock says.

    The same replacement bench/fp_bridge_server.py makes (its monte_carlo_tree_search wrapper, for
    a decision that asks for a number of iterations): Foul Play's call goes through with
    poke-engine's own `iterations` argument set, and poke-engine then ignores the duration. Foul
    Play searches its worlds in processes forked from this one, which inherit the replaced
    function. Where the default start method is not fork (Python 3.14 and later on Linux), fork is
    asked for, or the workers would import the original. The parent checks every "Iterations" line
    of the log against n either way.
    """
    import multiprocessing

    import fp.search.main as search_main

    if (sys.platform.startswith("linux") and multiprocessing.get_start_method(allow_none=True) is None
            and "fork" in multiprocessing.get_all_start_methods()):
        multiprocessing.set_start_method("fork")
    original = search_main.monte_carlo_tree_search

    def monte_carlo_tree_search(state: Any, duration_ms: int = 1000, iterations: int = 0, threads: int = 1) -> Any:
        return original(state, duration_ms, n, threads)

    search_main.monte_carlo_tree_search = monte_carlo_tree_search
    # Foul Play's logging is set up later, in run_foul_play: this line goes to the log as it is.
    print("INFO     Fixed iterations: {} per sampled world (search time ignored)".format(n), flush=True)


def _child_main(argv: List[str]) -> int:
    if not argv:
        print("usage: foulplay.py --child <foul-play dir> [--fixed-iterations N] <run.py arguments>")
        return 2
    fp_dir = os.path.abspath(argv[0])
    rest = list(argv[1:])
    fixed = 0
    if rest[:1] == [_FIXED_FLAG]:
        try:
            fixed = int(rest[1])
        except (IndexError, ValueError):
            print("usage: foulplay.py --child <foul-play dir> [--fixed-iterations N] <run.py arguments>")
            return 2
        rest = rest[2:]
    _watch_parent()
    os.chdir(fp_dir)
    # sys.path[0] is this file's folder. Foul Play must not see the benchmark's modules.
    sys.path[0] = fp_dir
    sys.argv = [os.path.join(fp_dir, "run.py")] + rest
    try:
        import fp.websocket_client as websocket_client

        websocket_client.PSWebsocketClient.login = _local_login
        if fixed > 0:
            _fix_iterations(fixed)
        from fp.main import run_foul_play

        asyncio.run(run_foul_play())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        return 1
    finally:
        _report_usage()
    return 0


# =============================================================================================
# The parent side: finding the installation
# =============================================================================================

def foulplay_dir() -> str:
    """$FOULPLAY_DIR, or ~/foul-play."""
    return os.path.abspath(os.path.expanduser(os.environ.get("FOULPLAY_DIR") or DEFAULT_DIR))


def _venv_python(fp_dir: str) -> Optional[str]:
    override = os.environ.get("FOULPLAY_PYTHON")
    if override:
        return override if os.path.isfile(override) else None
    for rel in ((".venv", "bin", "python"), ("venv", "bin", "python"),
                (".venv", "Scripts", "python.exe"), ("venv", "Scripts", "python.exe")):
        candidate = os.path.join(fp_dir, *rel)
        if os.path.isfile(candidate):
            return candidate
    return None


def _read_commit(fp_dir: str) -> str:
    """The checked-out commit, read from .git without running git. "unknown" when it cannot be read."""
    sha = re.compile(r"[0-9a-f]{40}")
    try:
        git_dir = os.path.join(fp_dir, ".git")
        with open(os.path.join(git_dir, "HEAD"), "r", encoding="utf-8") as fh:
            head = fh.read().strip()
        if not head.startswith("ref:"):
            return head if sha.fullmatch(head) else "unknown"
        ref = head.split(":", 1)[1].strip()
        ref_file = os.path.join(git_dir, *ref.split("/"))
        if os.path.isfile(ref_file):
            with open(ref_file, "r", encoding="utf-8") as fh:
                value = fh.read().strip()
            return value if sha.fullmatch(value) else "unknown"
        packed = os.path.join(git_dir, "packed-refs")
        if os.path.isfile(packed):
            with open(packed, "r", encoding="utf-8") as fh:
                for line in fh:
                    bits = line.strip().split(" ")
                    if len(bits) == 2 and bits[1] == ref and sha.fullmatch(bits[0]):
                        return bits[0]
    except Exception:
        pass
    return "unknown"


def _child_env() -> Dict[str, str]:
    env = dict(os.environ)
    # The caller's Python settings belong to the training venv, not to Foul Play's.
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV", "__PYVENV_LAUNCHER__"):
        env.pop(key, None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["FOULPLAY_PARENT_PID"] = str(os.getpid())
    return env


def _locate() -> Tuple[bool, str, Dict[str, str]]:
    fp_dir = foulplay_dir()
    hint = "run bench/install_foulplay.sh, or set FOULPLAY_DIR"
    if not os.path.isdir(fp_dir):
        return False, "Foul Play is not installed: %s does not exist (%s)" % (fp_dir, hint), {}
    if not (os.path.isfile(os.path.join(fp_dir, "run.py")) and os.path.isdir(os.path.join(fp_dir, "fp"))):
        return False, "%s is not a Foul Play checkout: no run.py or fp/ (%s)" % (fp_dir, hint), {}
    python = _venv_python(fp_dir)
    if python is None:
        return False, "Foul Play at %s has no virtualenv (.venv/bin/python): run bench/install_foulplay.sh" % fp_dir, {}
    sets = os.path.join(fp_dir, "fp", "data", "pkmn_sets_cache", BATTLE_FORMAT + ".json")
    if not os.path.isfile(sets) or os.path.getsize(sets) < 10_000:
        # Without it Foul Play downloads the sets in its first battle, and caches an empty list
        # when that fails. It then plays without knowing any set, which is not the benchmark.
        return False, "Foul Play at %s has no %s set data (%s): run bench/install_foulplay.sh" % (fp_dir, BATTLE_FORMAT, sets), {}
    try:
        done = subprocess.run(
            [python, "-c", "import poke_engine, websockets, requests, dateutil"],
            cwd=fp_dir, env=_child_env(), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return False, "Foul Play's virtualenv at %s did not answer within 120 s" % fp_dir, {}
    except OSError as exc:
        return False, "Foul Play's virtualenv at %s cannot be run: %s" % (fp_dir, exc), {}
    if done.returncode != 0:
        lines = (done.stdout or b"").decode("utf-8", "replace").strip().splitlines()
        last = lines[-1] if lines else "exit code %d" % done.returncode
        return False, "Foul Play's virtualenv at %s cannot import its requirements (%s): run bench/install_foulplay.sh" % (fp_dir, last), {}
    commit = _read_commit(fp_dir)
    message = "foul-play %s at %s" % (commit, fp_dir)
    if commit != PINNED_COMMIT:
        message += " (not the pinned commit %s)" % PINNED_COMMIT[:12]
    return True, message, {"dir": fp_dir, "python": python, "commit": commit}


def available() -> Tuple[bool, str]:
    """
    (True, "foul-play <commit> at <dir>") when Foul Play can be run, else (False, reason).
    Looks in $FOULPLAY_DIR, default ~/foul-play. Never raises.
    """
    try:
        ok, message, _info = _locate()
        return ok, message
    except Exception as exc:  # belt and braces: this function must not raise
        return False, "could not check for Foul Play: %r" % (exc,)


# =============================================================================================
# The parent side: processes
# =============================================================================================

_LIVE: Set[Any] = set()  # every Popen this module started and has not yet seen dead


def _kill_tree(popen: Any) -> None:
    """
    Stops one process this module started, and everything it started, by its PID.

    Linux: the process was started as the leader of its own process group, so the group is
    exactly Foul Play and the search workers it forked. Windows: taskkill /PID <pid> /T /F.
    """
    try:
        if os.name == "nt":
            if popen.poll() is None:
                subprocess.run(["taskkill", "/PID", str(popen.pid), "/T", "/F"],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=30)
        else:
            try:
                os.killpg(popen.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                if popen.poll() is None:
                    popen.kill()
    except Exception:
        try:
            if popen.poll() is None:
                popen.kill()
        except Exception:
            pass
    try:
        popen.wait(timeout=10)
    except Exception:
        pass
    if popen.poll() is not None:
        _LIVE.discard(popen)


def _kill_everything() -> None:
    for popen in list(_LIVE):
        _kill_tree(popen)


atexit.register(_kill_everything)

_RE_WINNER = re.compile(r"^INFO\s+Winner: (.*)$")
_RE_ITERATIONS = re.compile(r"^INFO\s+Iterations \d+: (\d+)$")
_RE_SEARCH = re.compile(r"^INFO\s+Searching for a move using MCTS")
_RE_FIXED = re.compile(r"^INFO\s+Fixed iterations: (\d+) ")

# poke-engine searches in batches of this many iterations and looks at its limit (time or count)
# between them: a count it is given is reached in whole batches, and a root where Foul Play has a
# single option (a forced switch) ends after the first batch, in both modes.
ENGINE_BATCH = 1000
_RE_USAGE = re.compile(r"^INFO\s+Usage: cpu_s=([\d.]+) worker_cpu_s=([\d.]+) rss_mb=(\d+) worker_rss_mb=(\d+)$")


class _Proc:
    """One Foul Play process and what is known about it."""

    def __init__(self, slot: int, attempt: int, name: str, run_count: int, popen: Any, log_path: str,
                 fixed_iterations: int = 0):
        self.slot = slot
        self.attempt = attempt
        self.name = name
        self.run_count = run_count
        self.popen = popen
        self.log_path = log_path
        self.started = time.monotonic()
        self.ended: Optional[float] = None
        self.idle_since = self.started        # spawn, or the end of its last battle
        self.pending = False                  # a challenge was accepted, the battle is not there yet
        self.games_started = 0                # battles of the player against this process
        self.current: Optional[str] = None    # battle tag being played
        self.battle_started = 0.0
        self.results: List[Tuple[str, Optional[bool], float]] = []  # (tag, player won, seconds)
        # from its log
        self._log_pos = 0
        self._log_rest = ""
        self._in_traceback = False
        self.exception: Optional[str] = None  # last line of the last traceback
        self.winners: List[str] = []
        self.searches = 0
        self.iterations_sum = 0
        self.iterations_n = 0
        # with fixed iterations: what was asked, what the process said it applied, the smallest
        # and largest count it logged, and how many of its "Iterations" lines were not the number
        self.fixed_iterations = int(fixed_iterations)
        self.fixed_said: Optional[int] = None
        self.iterations_min: Optional[int] = None
        self.iterations_max: Optional[int] = None
        self.iterations_off = 0
        self.iterations_one_option = 0
        self.rss_mb: Optional[int] = None         # peak, the process itself
        self.worker_rss_mb: Optional[int] = None  # peak, its largest search worker
        # from /proc
        self.cpu_seconds: Optional[float] = None

    @property
    def open_game(self) -> bool:
        return self.pending or self.current is not None

    def read_log(self) -> None:
        try:
            with open(self.log_path, "rb") as fh:
                fh.seek(self._log_pos)
                data = fh.read()
        except OSError:
            return
        if not data:
            return
        self._log_pos += len(data)
        text = self._log_rest + data.decode("utf-8", "replace")
        lines = text.split("\n")
        self._log_rest = lines.pop()
        for line in lines:
            self.parse_line(line.rstrip("\r"))

    def parse_line(self, line: str) -> None:
        if not line:
            return
        if "Traceback (most recent call last)" in line:
            self._in_traceback = True
            return
        if self._in_traceback:
            if line.startswith((" ", "\t")):
                return  # a frame
            self._in_traceback = False
            self.exception = line[:300]
            return
        match = _RE_ITERATIONS.match(line)
        if match:
            value = int(match.group(1))
            self.iterations_sum += value
            self.iterations_n += 1
            if self.fixed_iterations:
                if value == ENGINE_BATCH and self.fixed_iterations > ENGINE_BATCH:
                    # a root with one option for Foul Play: poke-engine stops after one batch
                    self.iterations_one_option += 1
                    return
                self.iterations_min = value if self.iterations_min is None else min(self.iterations_min, value)
                self.iterations_max = value if self.iterations_max is None else max(self.iterations_max, value)
                if value != self.fixed_iterations:
                    self.iterations_off += 1
            return
        if self.fixed_iterations:
            match = _RE_FIXED.match(line)
            if match:
                self.fixed_said = int(match.group(1))
                return
        if _RE_SEARCH.match(line):
            self.searches += 1
            return
        match = _RE_WINNER.match(line)
        if match:
            self.winners.append(match.group(1).strip())
            return
        match = _RE_USAGE.match(line)
        if match:
            self.cpu_seconds = float(match.group(1)) + float(match.group(2))
            self.rss_mb = int(match.group(3))
            self.worker_rss_mb = int(match.group(4))

    def sample_proc(self) -> None:
        """
        Linux: CPU seconds so far of the process and the search workers it has reaped. A process
        that exits by itself reports the exact figure in its last log line (see _report_usage);
        this covers the ones that are stopped.
        """
        if not sys.platform.startswith("linux"):
            return
        try:
            with open("/proc/%d/stat" % self.popen.pid, "rb") as fh:
                fields = fh.read().decode("ascii", "replace").rsplit(")", 1)[1].split()
            ticks = sum(int(fields[i]) for i in (11, 12, 13, 14))  # utime stime cutime cstime
            self.cpu_seconds = ticks / float(os.sysconf("SC_CLK_TCK"))
        except Exception:
            pass

    def why(self) -> str:
        if self.exception:
            return self.exception
        return "no traceback in %s" % os.path.basename(self.log_path)


def _split_games(n_games: int, slots: int) -> List[int]:
    """n_games over `slots` processes, as evenly as it goes. Slots with no game are left out."""
    if n_games <= 0 or slots <= 0:
        return []
    slots = min(slots, n_games)
    base, extra = divmod(n_games, slots)
    return [base + (1 if i < extra else 0) for i in range(slots)]


def _default_game_timeout(search_time_ms: int) -> float:
    """
    Seconds one game may take before it is stopped. Foul Play spends a little over twice its
    search time on a decision, and 240 decisions is far beyond a normal game (19 to 30 on
    average in testing, 99 in the longest). Four minutes at least, which stays under the five
    minutes evaluate.py waits before it calls a whole match stalled.
    """
    return max(240.0, 240.0 * (2.0 * search_time_ms / 1000.0 + 0.5))


def _empty_result(n_games: int, parallel: int, search_time_ms: int) -> Dict[str, Any]:
    return {
        "games": 0, "wins": 0, "losses": 0, "ties": 0, "errors": max(0, int(n_games)),
        "seconds": 0.0, "foulplay_commit": None,
        "search_time_ms": int(search_time_ms), "parallel": int(parallel),
        "requested": max(0, int(n_games)),
    }


class _Run:
    """State of one run_matches call. Lives on the player's event loop."""

    def __init__(self, player: Any, info: Dict[str, str], host: str, port: int, search_time_ms: int,
                 search_parallelism: int, log_dir: str, battle_format: str,
                 game_timeout_s: float, start_timeout_s: float, max_restarts: int, fixed_iterations: int = 0):
        self.player = player
        self.player_name = str(player.username)
        self.info = info
        self.uri = "ws://%s:%d/showdown/websocket" % (host, int(port))
        self.search_time_ms = int(search_time_ms)
        self.search_parallelism = int(search_parallelism)
        self.log_dir = log_dir
        self.battle_format = battle_format
        self.game_timeout_s = float(game_timeout_s)
        self.start_timeout_s = float(start_timeout_s)
        self.max_restarts = int(max_restarts)
        self.fixed_iterations = max(0, int(fixed_iterations or 0))
        self.tag = secrets.token_hex(3)
        self.procs: List[_Proc] = []
        self.by_id: Dict[str, _Proc] = {}
        self.before: Set[str] = set(player.battles.keys())   # battles that are not ours
        self.bound: Dict[str, Optional[_Proc]] = {}           # new battle tag -> its process
        self.aborted: Set[str] = set()                        # battles this run forfeited
        self.permits = 0                                      # battle-semaphore permits taken
        self.ignored_challenges = 0                           # from a dead or busy process
        self.failures: List[str] = []
        self.notes: List[str] = []

    # ---- processes ----
    def command(self, name: str, run_count: int) -> List[str]:
        fixed = [_FIXED_FLAG, str(self.fixed_iterations)] if self.fixed_iterations else []
        return [
            self.info["python"], "-u", _THIS_FILE, "--child", self.info["dir"], *fixed,
            "--websocket-uri", self.uri,
            "--ps-username", name,
            "--bot-mode", "challenge_user",
            "--user-to-challenge", self.player_name,
            "--pokemon-format", self.battle_format,
            "--search-time-ms", str(self.search_time_ms),
            "--search-parallelism", str(self.search_parallelism),
            "--run-count", str(run_count),
            "--log-level", "INFO",
        ]

    def spawn(self, slot: int, attempt: int, run_count: int) -> _Proc:
        name = "FP%ss%dr%d" % (self.tag, slot, attempt)
        log_path = os.path.join(self.log_dir, "foulplay_%s_s%02d_r%d.log" % (self.tag, slot, attempt))
        kwargs: Dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True  # its own process group: see _kill_tree
        with open(log_path, "ab") as log:
            popen = subprocess.Popen(self.command(name, run_count), cwd=self.info["dir"], env=_child_env(),
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **kwargs)
        _LIVE.add(popen)
        proc = _Proc(slot, attempt, name, run_count, popen, log_path, self.fixed_iterations)
        self.procs.append(proc)
        self.by_id[_to_id(name)] = proc
        return proc

    def stop(self, proc: _Proc) -> None:
        proc.sample_proc()
        _kill_tree(proc.popen)
        if proc.ended is None:
            proc.ended = time.monotonic()
        proc.read_log()

    # ---- battles ----
    def scan(self) -> None:
        """Ties each new battle of the player to the process it is played against."""
        now = time.monotonic()
        for tag, battle in list(self.player.battles.items()):
            if tag in self.before or tag in self.bound:
                continue
            opponent = getattr(battle, "opponent_username", None)
            if not opponent:
                continue  # the title line has not been parsed yet
            proc = self.by_id.get(_to_id(opponent))
            self.bound[tag] = proc
            if proc is not None:
                self.settle(proc)
                proc.pending = False
                proc.games_started += 1
                proc.current = tag
                proc.battle_started = now

    def settle(self, proc: _Proc) -> None:
        """Records the current battle of `proc` when it has ended by itself."""
        tag = proc.current
        if tag is None:
            return
        battle = self.player.battles.get(tag)
        if battle is None or not battle.finished:
            return
        now = time.monotonic()
        proc.current = None
        proc.idle_since = now
        if tag not in self.aborted:
            proc.results.append((tag, battle.won, now - proc.battle_started))

    async def abort(self, proc: Optional[_Proc], tag: str) -> None:
        """Ends a battle that cannot finish, by forfeiting it. It will not be counted."""
        battle = self.player.battles.get(tag)
        if battle is not None and not battle.finished:
            self.aborted.add(tag)
            try:
                await self.player.ps_client.send_message("/forfeit", tag)
                deadline = time.monotonic() + _ABORT_WAIT_S
                while not battle.finished and time.monotonic() < deadline:
                    await asyncio.sleep(0.05)
            except Exception as exc:
                self.notes.append("could not forfeit %s: %r" % (tag, exc))
        if proc is not None and proc.current == tag:
            proc.current = None
            proc.idle_since = time.monotonic()

    async def abort_leftover(self, proc: _Proc) -> None:
        """After a process is gone: end the battle it left, also one that was only just accepted."""
        deadline = time.monotonic() + 2.0
        while proc.pending and proc.current is None and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
            self.scan()
        self.scan()
        proc.pending = False
        if proc.current is not None:
            await self.abort(proc, proc.current)


async def _acceptor(run: _Run) -> None:
    """
    Accepts the challenges of this run's processes, one at a time, as poke-env's own
    Player._accept_challenges does, but with a time limit on every step and without accepting
    a second challenge from a process that is already in a game.
    """
    player = run.player
    queue = player._challenge_queue
    semaphore = player._battle_semaphore
    while True:
        challenger = _to_id(await queue.get())
        proc = run.by_id.get(challenger)
        if proc is None:
            continue
        # Foul Play challenges again within milliseconds of a game ending. Bring the books up
        # to date here instead of waiting for the next look of the process's supervisor,
        # or that challenge would be mistaken for one sent in the middle of a game.
        run.scan()
        run.settle(proc)
        if proc.popen.poll() is not None or proc.open_game or proc.games_started >= proc.run_count:
            run.ignored_challenges += 1
            continue
        # Marked before the accept is sent, not after: the battle can be there, and be tied to
        # the process by a supervisor (which clears the mark), before this coroutine runs again.
        proc.pending = True
        try:
            team = player.get_next_team() if hasattr(player, "get_next_team") else None
            await player.ps_client.accept_challenge(proc.name, team)
        except Exception as exc:
            proc.pending = False
            run.notes.append("accepting %s failed: %r" % (proc.name, exc))
            continue
        try:
            # Released by poke-env when the battle object is created. Taking it keeps the
            # player's own bookkeeping in step, as if it had accepted the challenge itself.
            await asyncio.wait_for(semaphore.acquire(), _ACCEPT_WAIT_S)
            run.permits += 1
        except asyncio.TimeoutError:
            proc.pending = False
            run.notes.append("%s was accepted but no battle started" % proc.name)


async def _supervise(run: _Run, proc: _Proc) -> Optional[str]:
    """Watches one process until it is gone. Returns None when it played all its games."""
    while True:
        run.scan()
        run.settle(proc)
        proc.read_log()
        proc.sample_proc()
        now = time.monotonic()
        done = len(proc.results)
        code = proc.popen.poll()
        if code is not None:
            proc.ended = now
            _kill_tree(proc.popen)  # the workers it may have left behind
            proc.read_log()
            # The process is gone, which says nothing yet about its last game: Foul Play leaves
            # as soon as it has seen the game end, and the player may still have the final
            # message waiting. Let the player catch up before the game is called lost, and
            # wait longer when Foul Play's own log says the game did end.
            saw_the_end = len(proc.winners) > len(proc.results)
            deadline = time.monotonic() + (5 * _SETTLE_WAIT_S if saw_the_end else _SETTLE_WAIT_S)
            while proc.open_game and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
                run.scan()
                run.settle(proc)
            done = len(proc.results)
            if proc.current is not None:
                return "exited with code %s during %s (%s)" % (code, proc.current, proc.why())
            if done >= proc.run_count:
                return None
            return "exited with code %s after %d of %d games (%s)" % (code, done, proc.run_count, proc.why())
        if done >= proc.run_count and not proc.open_game:
            if now - proc.idle_since > _EXIT_GRACE_S:
                run.stop(proc)  # played everything but did not leave
                return None
        elif proc.current is not None:
            if now - proc.battle_started > run.game_timeout_s:
                tag = proc.current
                run.stop(proc)
                return "%s was not over after %.0f s" % (tag, run.game_timeout_s)
        elif now - proc.idle_since > run.start_timeout_s:
            run.stop(proc)
            return "no battle started within %.0f s (%s)" % (run.start_timeout_s, proc.why())
        await asyncio.sleep(_POLL_S)


async def _slot(run: _Run, slot: int, games: int, delay: float) -> None:
    """Plays `games` games in sequence, restarting Foul Play when a process fails."""
    if delay > 0:
        await asyncio.sleep(delay)
    remaining = games
    attempt = 0
    while remaining > 0:
        try:
            proc = run.spawn(slot, attempt, remaining)
        except Exception as exc:
            run.failures.append("slot %d: could not start Foul Play: %r" % (slot, exc))
            return
        try:
            reason = await _supervise(run, proc)
            if reason is not None:
                await run.abort_leftover(proc)
        except BaseException:
            # cancelled (overall timeout, or the caller went away): leave nothing running.
            # _cleanup ends the battle.
            run.stop(proc)
            raise
        remaining -= len(proc.results)
        if reason is None:
            return
        run.failures.append("%s: %s" % (proc.name, reason))
        remaining -= 1          # the game that was being set up or played is an error
        attempt += 1
        if attempt > run.max_restarts:
            if remaining > 0:
                run.failures.append("slot %d: gave up after %d failed processes, %d games not played"
                                    % (slot, attempt, remaining))
            return


def _tally(run: _Run, result: Dict[str, Any], n_games: int) -> None:
    wins = losses = ties = 0
    log_wins = log_losses = log_ties = 0
    player_id = _to_id(run.player_name)
    game_seconds: List[float] = []
    for proc in run.procs:
        for _tag, won, seconds in proc.results:
            game_seconds.append(seconds)
            if won is True:
                wins += 1
            elif won is False:
                losses += 1
            else:
                ties += 1
        for winner in proc.winners:
            if _to_id(winner) == player_id:
                log_wins += 1
            elif winner == proc.name:
                log_losses += 1
            else:
                log_ties += 1  # Foul Play prints "Winner: None" for a tie
    games = wins + losses + ties
    result.update({
        "games": games, "wins": wins, "losses": losses, "ties": ties,
        "errors": max(0, int(n_games) - games),
    })
    # Foul Play prints the winner of every game it saw end. A process is always stopped before
    # one of its games is forfeited, so its log should hold exactly the games counted here.
    result["foulplay_log"] = {"wins": log_wins, "losses": log_losses, "ties": log_ties}
    result["log_agrees"] = bool(log_wins == wins and log_losses == losses and log_ties == ties)
    if game_seconds:
        result["game_seconds_mean"] = round(sum(game_seconds) / len(game_seconds), 2)
        result["game_seconds_max"] = round(max(game_seconds), 2)
    searches = sum(p.searches for p in run.procs)
    iterations_n = sum(p.iterations_n for p in run.procs)
    if iterations_n:
        # per searched state. Foul Play's search is limited by time, so on a busy machine this
        # number falls and the opponent is weaker: compare it between evaluations.
        result["foulplay_mean_iterations"] = int(sum(p.iterations_sum for p in run.procs) / iterations_n)
    if searches:
        result["foulplay_decisions"] = searches
    if run.fixed_iterations:
        # Each world searched is logged twice (by the search worker and by the main process):
        # these count lines. Every one should read exactly the number asked for.
        lows = [p.iterations_min for p in run.procs if p.iterations_min is not None]
        highs = [p.iterations_max for p in run.procs if p.iterations_max is not None]
        if lows:
            result["iterations_min"] = min(lows)
            result["iterations_max"] = max(highs)
        result["iterations_off"] = sum(p.iterations_off for p in run.procs)
        result["iterations_one_option"] = sum(p.iterations_one_option for p in run.procs)
        silent = [p.name for p in run.procs if p.games_started and p.fixed_said != run.fixed_iterations]
        if silent:
            result["iterations_unconfirmed"] = silent[:20]
    cpu = [(p.cpu_seconds, (p.ended or time.monotonic()) - p.started) for p in run.procs if p.cpu_seconds is not None]
    if cpu:
        alive = sum(wall for _c, wall in cpu)
        result["foulplay_cpu_seconds"] = round(sum(c for c, _w in cpu), 1)
        if alive > 0:
            result["foulplay_cores_per_process"] = round(sum(c for c, _w in cpu) / alive, 2)
    rss = [p.rss_mb + run.search_parallelism * (p.worker_rss_mb or 0) for p in run.procs if p.rss_mb is not None]
    if rss:
        result["foulplay_rss_mb_max"] = max(rss)  # one process with its search workers, peak
    result["restarts"] = sum(1 for p in run.procs if p.attempt > 0)
    if run.ignored_challenges:
        result["ignored_challenges"] = run.ignored_challenges
    if run.failures:
        result["failures"] = run.failures[:50]
    if run.notes:
        result["notes"] = run.notes[:50]


async def _cleanup(run: _Run) -> None:
    """Stops every process, ends every battle of this run that is still open, squares the player."""
    for proc in run.procs:
        if proc.popen.poll() is None or proc.popen in _LIVE:
            run.stop(proc)
    try:
        run.scan()
        aborts = []
        for tag, battle in list(run.player.battles.items()):
            if tag in run.before or battle.finished:
                continue
            proc = run.bound.get(tag)
            opponent = getattr(battle, "opponent_username", None)
            if proc is not None or not opponent or _to_id(opponent) in run.by_id:
                aborts.append(run.abort(proc, tag))
        if aborts:  # all at once: the wait for each is bounded, and so is the wait for all
            await asyncio.gather(*aborts, return_exceptions=True)
        # poke-env releases one battle-semaphore permit per battle it creates. Take back any
        # this run did not take, so a later battle_against on the same player counts right.
        created = sum(1 for proc in run.bound.values() if proc is not None)
        semaphore = run.player._battle_semaphore
        while run.permits < created and not semaphore.locked():
            await semaphore.acquire()
            run.permits += 1
    except Exception as exc:
        run.notes.append("cleanup: %r" % (exc,))


async def _run_on_player_loop(player: Any, n_games: int, host: str, port: int, parallel: int,
                              search_time_ms: int, log_dir: Optional[str], search_parallelism: int,
                              game_timeout_s: Optional[float], overall_timeout_s: Optional[float],
                              start_timeout_s: float, max_restarts: int, battle_format: str,
                              fixed_iterations: int = 0) -> Dict[str, Any]:
    t0 = time.monotonic()
    result = _empty_result(n_games, parallel, search_time_ms)
    result["search_parallelism"] = int(search_parallelism)
    fixed_iterations = max(0, int(fixed_iterations or 0))
    if fixed_iterations:
        result["fixed_iterations"] = fixed_iterations
        if fixed_iterations % ENGINE_BATCH:
            result["seconds"] = 0.0
            result["error"] = ("fixed_iterations %d is not a multiple of %d: poke-engine would search %d"
                               % (fixed_iterations, ENGINE_BATCH, -(-fixed_iterations // ENGINE_BATCH) * ENGINE_BATCH))
            return result

    def failed(reason: str) -> Dict[str, Any]:
        result["error"] = reason
        result["seconds"] = round(time.monotonic() - t0, 2)
        return result

    if n_games <= 0:
        return result
    try:
        # In a thread: the check starts a Python process, and this loop also serves the player.
        ok, message, info = await asyncio.get_running_loop().run_in_executor(None, _locate)
    except Exception as exc:
        return failed("could not check for Foul Play: %r" % (exc,))
    if not ok:
        return failed(message)
    result["foulplay_commit"] = info["commit"]
    result["foulplay"] = message

    for attribute in ("ps_client", "username", "battles", "_challenge_queue", "_battle_semaphore"):
        if not hasattr(player, attribute):
            return failed("the player has no %s: run_matches needs a poke-env 0.15 Player" % attribute)
    player_format = getattr(player, "format", battle_format)
    if player_format != battle_format:
        return failed("the player plays %s, the benchmark is %s" % (player_format, battle_format))
    if any(bad in str(player.username) for bad in ("vs.", "|", ",")):
        return failed("the player's name %r would confuse Foul Play's parsing" % (player.username,))
    try:
        await asyncio.wait_for(player.ps_client.logged_in.wait(), _LOGIN_WAIT_S)
    except asyncio.TimeoutError:
        return failed("the player was not logged in after %.0f s" % _LOGIN_WAIT_S)

    notes: List[str] = []
    slots = max(1, min(int(parallel), int(n_games)))
    limit = int(getattr(player, "_max_concurrent_battles", 0) or 0)
    if 0 < limit < slots:
        # A battle past the player's limit would start on the server while the player ignores
        # it, with Foul Play's timer running against the player.
        notes.append("parallel lowered from %d to the player's max_concurrent_battles=%d" % (slots, limit))
        slots = limit
    result["parallel"] = slots

    if log_dir is None:
        log_dir = tempfile.mkdtemp(prefix="foulplay_")
    os.makedirs(log_dir, exist_ok=True)
    result["log_dir"] = os.path.abspath(log_dir)

    if game_timeout_s is None:
        game_timeout_s = _default_game_timeout(search_time_ms)
    shares = _split_games(n_games, slots)
    if overall_timeout_s is None:
        # every slot playing its games one after the other at the per-game limit, plus slack
        overall_timeout_s = max(shares) * float(game_timeout_s) + 120.0
    result["game_timeout_s"] = float(game_timeout_s)
    result["overall_timeout_s"] = float(overall_timeout_s)

    run = _Run(player, info, host, port, search_time_ms, search_parallelism, result["log_dir"],
               battle_format, game_timeout_s, start_timeout_s, max_restarts, fixed_iterations)
    run.notes.extend(notes)
    acceptor = asyncio.ensure_future(_acceptor(run))
    tasks = [asyncio.ensure_future(_slot(run, i, games, i * _STAGGER_S)) for i, games in enumerate(shares)]
    try:
        _done, pending = await asyncio.wait(tasks, timeout=overall_timeout_s)
        if pending:
            run.failures.append("stopped after the overall timeout of %.0f s" % overall_timeout_s)
            result["timed_out"] = True
        for task in tasks:
            if task.done() and not task.cancelled() and task.exception() is not None:
                run.failures.append("slot task failed: %r" % (task.exception(),))
    finally:
        acceptor.cancel()
        for task in tasks:
            task.cancel()
        await asyncio.gather(acceptor, *tasks, return_exceptions=True)
        await _cleanup(run)
    _tally(run, result, n_games)
    result["seconds"] = round(time.monotonic() - t0, 2)
    if result["games"] and result["seconds"] > 0:
        result["games_per_minute"] = round(60.0 * result["games"] / result["seconds"], 2)
    return result


async def run_matches(player: Any, n_games: int, host: str, port: int, *,
                      parallel: int = 4, search_time_ms: int = 200, log_dir: Optional[str] = None,
                      search_parallelism: int = 1, game_timeout_s: Optional[float] = None,
                      overall_timeout_s: Optional[float] = None, start_timeout_s: float = 60.0,
                      max_restarts: int = 2, battle_format: str = BATTLE_FORMAT,
                      fixed_iterations: int = 0) -> Dict[str, Any]:
    """
    Plays `n_games` games of gen9randombattle between `player` and Foul Play.

    player            a poke-env 0.15 Player (any subclass) connected to ws://host:port. Do not use
                      it for anything else until this returns.
    parallel          Foul Play processes, which is also the number of games at once. Lowered to
                      the player's max_concurrent_battles when that is smaller.
    search_time_ms    Foul Play's --search-time-ms. A decision takes about twice this.
    log_dir           where each process's output goes (foulplay_<run>_sNN_rN.log). A new
                      temporary directory when None.
    search_parallelism  Foul Play's --search-parallelism: CPU cores per process, and the
                      multiplier on how many possible opponent teams it searches.
    game_timeout_s    a game not over after this long is stopped and counted as an error
                      (default: 240 s, more for searches longer than 250 ms).
    overall_timeout_s everything is stopped after this long (default: the per-game limit times
                      the games of the busiest process, plus two minutes).
    start_timeout_s   a process with no battle this long after it started, or after its last
                      game, is stopped.
    max_restarts      how often one slot may replace a failed process. Each failure costs one
                      game, counted as an error; the rest of its games go to the replacement.
    fixed_iterations  0 (default): Foul Play searches each sampled world for its search time.
                      N > 0: for exactly N iterations instead, whatever the clock says (the
                      search time still sets how many worlds it samples). The dict then also has
                      fixed_iterations (N), iterations_min and iterations_max (over every
                      "Iterations" line of the logs), iterations_off (lines that were not N:
                      should be 0) and, when a process that played never said it applied N,
                      iterations_unconfirmed (their names). N must be a multiple of
                      ENGINE_BATCH (1,000): poke-engine searches in whole batches. A search
                      where Foul Play had one option (a forced switch) ends after one batch in
                      either mode: those lines are counted in iterations_one_option and left
                      out of the minimum and maximum.

    Returns a dict, counted from the player's side:
        games    games that ran to their end  (wins + losses + ties)
        errors   games that did not           (games + errors == n_games)
        seconds, foulplay_commit, search_time_ms, parallel (the value used)
    plus requested, log_agrees, foulplay_log, games_per_minute, game_seconds_mean,
    foulplay_mean_iterations, foulplay_decisions, foulplay_cores_per_process,
    foulplay_rss_mb_max, restarts, log_dir, and, when something went wrong, error, failures,
    notes, timed_out.

    Never raises for a failure of Foul Play or the server: those are errors in the dict.
    Read the result from the dict, not from player.n_won_battles: a game this function had to
    stop is forfeited so that the player is free again, and shows there as a loss.
    """
    try:
        loop = player.ps_client.loop
    except Exception:
        result = _empty_result(n_games, parallel, search_time_ms)
        result["error"] = "the player has no ps_client.loop: run_matches needs a poke-env 0.15 Player"
        return result
    coroutine = _run_on_player_loop(player, int(n_games), host, int(port), int(parallel), int(search_time_ms),
                                    log_dir, int(search_parallelism), game_timeout_s, overall_timeout_s,
                                    float(start_timeout_s), int(max_restarts), battle_format, int(fixed_iterations or 0))
    try:
        running: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        return await coroutine
    # poke-env keeps its players on their own event loop, in their own thread. The queue and the
    # semaphore used here belong to that loop, so the whole run lives there.
    future = asyncio.run_coroutine_threadsafe(coroutine, loop)
    try:
        return await asyncio.wrap_future(future)
    except asyncio.CancelledError:
        future.cancel()
        raise


# =============================================================================================
# Command line: a check of the installation, and a short real match
# =============================================================================================

def _cli(argv: List[str]) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Foul Play as a benchmark opponent. With no option: is it installed?")
    parser.add_argument("--play", type=int, default=0, metavar="N",
                        help="play N games against it with a poke-env baseline player (needs poke-env and a running server)")
    parser.add_argument("--agent", choices=("random", "maxpower", "simple"), default="random")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--search-time-ms", type=int, default=200)
    parser.add_argument("--search-parallelism", type=int, default=1)
    parser.add_argument("--fixed-iterations", type=int, default=0, metavar="N",
                        help="search every sampled world for exactly N iterations (a multiple of 1,000) instead of --search-time-ms")
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args(argv)

    ok, message = available()
    print(message)
    if not ok:
        return 1
    if args.play <= 0:
        return 0

    from poke_env import AccountConfiguration, ServerConfiguration
    from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

    async def play() -> Dict[str, Any]:
        cls = {"random": RandomPlayer, "maxpower": MaxBasePowerPlayer, "simple": SimpleHeuristicsPlayer}[args.agent]
        player = cls(
            account_configuration=AccountConfiguration("%s%s" % (args.agent, secrets.token_hex(3)), None),
            server_configuration=ServerConfiguration("ws://%s:%d/showdown/websocket" % (args.host, args.port),
                                                     "https://play.pokemonshowdown.com/action.php?"),
            battle_format=BATTLE_FORMAT, max_concurrent_battles=max(1, args.parallel), log_level=40,
        )
        return await run_matches(player, args.play, args.host, args.port, parallel=args.parallel,
                                 search_time_ms=args.search_time_ms, search_parallelism=args.search_parallelism,
                                 log_dir=args.log_dir, fixed_iterations=args.fixed_iterations)

    result = asyncio.run(play())
    print(json.dumps(result, indent=1, sort_keys=True))
    games = result["games"]
    print("%s won %d of %d against Foul Play at %d ms (%d errors, %.0f s)" % (
        args.agent, result["wins"], games, result["search_time_ms"], result["errors"], result["seconds"]))
    return 0 if result["errors"] == 0 and games > 0 else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        sys.exit(_child_main(sys.argv[2:]))
    sys.exit(_cli(sys.argv[1:]))
