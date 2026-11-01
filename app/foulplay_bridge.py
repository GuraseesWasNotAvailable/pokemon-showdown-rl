"""
Bridge between poke-env and Foul Play (https://github.com/pmariglia/foul-play).

Foul Play is a search bot: for every decision it samples complete teams the opponent could have
and runs a Monte Carlo tree search on each. bench/foulplay.py runs it as a separate player. This
module asks it for advice instead: for a battle that OUR poke-env player is playing, what would
Foul Play play in this exact position, and what did its search see for every legal action.

    bridge = FoulPlayBridge(helpers=4, search_ms=100)      # 4 helper processes, one core each
    class MyPlayer(FoulPlayMixin, Player): ...             # forwards the battle's lines by itself
    advice = await player.foulplay_advice(battle)          # .choice, .action, .policy[14], .mask[14]
    order, advice = await player.foulplay_decision(battle) # the same as a poke-env order
    order, action, advice = await player.foulplay_teacher(battle)   # for a player that LEARNS from it:
                                                           # action is None when there is no label

The helpers are bench/fp_bridge_server.py processes, started with FOUL PLAY's own virtualenv
python. Each one follows its battles with Foul Play's own battle loop and decides with Foul
Play's own decision function (that file says exactly how). This file is the other end:

  * a pool of helpers. A search holds its process for its whole duration, so one helper is one
    core. A battle is pinned to the helper with the fewest battles when its first line arrives.
  * nothing here blocks the caller. Lines are put on a queue per helper and written by that
    helper's own thread; a decision is a future, awaited from any event loop.
  * a question carries the time until which its answer is of use. A helper that is asked more
    than it can answer does not search for callers that have given up: it answers as many as
    it can, in time, and says "expired" at once for the rest.
  * a helper that dies or hangs is replaced, and the battles it held are rebuilt in the new one
    from the lines and choices stored here. Decisions that were waiting are asked again.
  * a battle that is forgotten (the mixin does it when the battle ends, the bridge itself when
    the player is taken out of the room) takes its stored lines with it, so memory follows the
    battles in progress and not the battles played.
  * every helper is the leader of its own process group and is stopped by that group's id, on
    close() and at interpreter exit. A helper also stops by itself when this process is gone.

Foul Play's strength comes from the ITERATIONS of its search and its setting is a TIME, so the
same setting is a weaker player on slower or shared cores, with no error anywhere:

    python foulplay_bridge.py calibrate --helpers 4       # iterations per decision here, now
    teacher_problems(asked, unanswered, decisions, iterations_per_decision)   # when a run should stop

Foul Play is found in FOULPLAY_DIR (default ~/foul-play), its python in <dir>/.venv/bin/python or
FOULPLAY_PYTHON, as bench/foulplay.py does. bench/install_foulplay.sh installs it.

On Windows, where Foul Play does not build, the helpers are started INSIDE WSL (wsl.exe running
Foul Play's python on bench/fp_bridge_server.py, the same JSON lines over its stdin and stdout),
so a player that runs on Windows with the GPU gets the same advice. FOULPLAY_DIR is then a path
inside WSL (looked for in ~/psppo-bench/foul-play and ~/foul-play when it is not set);
FOULPLAY_WSL=0 switches this off, FOULPLAY_WSL_DISTRO names a distribution other than the default.
See "Windows" in bench/FOULPLAY_BRIDGE.md.

On ANOTHER machine, over ssh (remote=RemoteTarget("ubuntu@1.2.3.4", key=...), ladder.py
--search-remote): the same helper command is started on that machine by `ssh -T` (no terminal,
BatchMode, keep-alives every 5 s), with its Foul Play (~/foul-play) and its copy of this repository
(~/dev/ab-ladder), whose bench/fp_bridge_server.py must be this one (locate_remote() compares the
two). The player stays here; only the search runs there. The extra network round trip is measured
when the helpers start (measure_round_trip) and kept up to date by a clock probe every 2 s, which is
also the line the helper's --parent-watch beat waits for: a decision's deadline is given on the
helper's own clock less the answer's way back, and a question that cannot be answered in time is
not searched (the caller plays its own move). A helper there leaves when its ssh session ends (its
parent goes), when the connection goes quiet for 20 s, and is stopped by the tag on its command
line (pkill over ssh) on close() and at exit, so nothing is left running there. See "Remote" in
bench/FOULPLAY_BRIDGE.md.

A decision also carries the opponent's side of the same search (advice.opponent: its options by
name with their share over the sampled worlds, advice.opponent_worlds: the set sampled for its
active Pokemon in each world), and on request a table of values per (our action, its option):
advice(..., payoff_ms=200) -> advice.payoff_table[14][len(advice.payoff_theirs)].

Nothing here imports torch, ray, numpy or poke-env, so it is safe to use from rollout workers.
bench/FOULPLAY_BRIDGE.md has the design notes and the measured numbers, and
bench/fp_bridge_replay.py checks, exactly, that the helper tracks a battle as Foul Play does.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import gzip
import hashlib
import itertools
import json
import logging
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

from psbot_bridge import choice_to_order
from psbot_bridge import order_to_choice as _psbot_order_to_choice

logger = logging.getLogger(__name__)

__all__ = [
    "FoulPlayBridge", "FoulPlayBridgeError", "FoulPlayMixin", "Advice", "Option",
    "available", "locate", "shared_bridge", "close_shared", "legal_mask", "order_to_index",
    "index_to_target", "choice_to_index", "fill_actions", "order_to_choice",
    "teacher_problems", "calibrate", "load_trace", "OpponentOption", "wsl_path", "wsl_leftovers",
    "RemoteTarget", "remote_target", "locate_remote", "remote_leftovers", "remote_echo_round_trip", "find_ssh",
    "ACTION_DIM", "PINNED_COMMIT", "BATTLE_FORMAT",
    "FALLBACK_FLOOR", "FALLBACK_SHARE", "TARGET_ITERATIONS", "MIN_ITERATIONS", "MIN_ITERATIONS_SAMPLE",
]

REPO_ROOT = Path(__file__).resolve().parent
SERVER_PY = REPO_ROOT / "bench" / "fp_bridge_server.py"

# Foul Play main as of 2026-10-03, the commit bench/install_foulplay.sh installs.
PINNED_COMMIT = "8f541f80f91f960603673796559f796b10ccf0fb"
BATTLE_FORMAT = "gen9randombattle"
DEFAULT_DIR = "~/foul-play"
DEFAULT_SEARCH_MS = 100
ACTION_DIM = 14   # 0-3 moves, 4-9 switches, 10-13 the same moves with Terastallization (obs_assembler.py)

WSL_DIRS = ("~/psppo-bench/foul-play", DEFAULT_DIR)   # where Foul Play is looked for inside WSL, in this order
WSL_GRACE_S = 8.0           # how long a helper inside WSL is given to leave after its stdin was closed (it takes
                            # about 3 s for wsl.exe to end on the loaded laptop; one more wsl.exe to stop it costs 2 to 9 s)
WSL_BEAT_S = 2.0            # a helper inside WSL is sent a line this often, and leaves when it hears nothing for 20 s
_GONE_KEPT = 20000          # ended battles remembered, so that their late lines are not taken for new battles
_CLOCK_TOLERANCE_S = 5.0    # a helper whose clock differs from ours by more is sent no deadlines
_RECENT_DECISIONS = 5000    # decisions behind stats()["recent_iterations_per_decision"]

# Helpers on another machine, over ssh (RemoteTarget).
REMOTE_CODE_DIR = "~/dev/ab-ladder"  # that machine's copy of this repository: its bench/fp_bridge_server.py is the helper
REMOTE_FP_DIRS = ("~/foul-play", "~/psppo-bench/foul-play")   # where Foul Play is looked for there, in this order
REMOTE_GRACE_S = 4.0        # how long remote helpers are given to leave after their stdin was closed: the end of input
                            # crosses the network, and a search in progress (up to about 2 s) is finished first
REMOTE_PROBE_S = 2.0        # a remote helper is sent a clock probe this often: the round trip, its clock, and the
                            # line its --parent-watch beat waits for (it leaves after 20 s with none)
REMOTE_PROBE_STALE_S = 10.0  # a probe unanswered this long no longer holds the next one back
REMOTE_MARGIN_S = 0.05      # kept on top of the measured way back in a remote helper's deadline
REMOTE_WINDOW = 30          # probes behind a remote helper's round trip and clock difference (a minute of them)
REMOTE_START_PROBES = 8     # probes per helper when the helpers start (measure_round_trip)
REMOTE_KILL_TIMEOUT_S = 40.0  # one ssh that stops leftover helpers there (it connects in 1 to 3 s; ConnectTimeout 15 s)
# The link to remote helpers (link_status, link_spans): its troubles are told apart from failures of the search.
REMOTE_LATE_S = 0.4         # a line the network held up this much longer than half the start's round trip, either way, is a stall
REMOTE_LATE_HOLD_S = 5.0    # the link counts as down this long after such a line came in
LINK_SPANS_KEPT_S = 900.0   # link troubles are remembered this long after they ended

# When a run that learns from Foul Play, or trains against it, is no longer what it says it is.
# See teacher_problems() and "Choosing the settings" in bench/FOULPLAY_BRIDGE.md.
FALLBACK_FLOOR = 50                    # this many decisions without Foul Play's answer are tolerated in any case,
FALLBACK_SHARE = 0.02                  # and beyond that this share of the decisions asked
TARGET_ITERATIONS = 60_000             # search iterations per decision to set the search time for (the laptop's 50 ms)
MIN_ITERATIONS = 45_000                # below this the search is weaker than anything that was measured as good enough
MIN_ITERATIONS_SAMPLE = 2000           # decisions before the iterations are judged


FORK_S_PER_WORKER = 0.03               # a decision's own pool ("workers"): starting one process (bench/fp_bridge_server.py)


def search_seconds(ms: int, parallelism: int = 1, workers: int = 1) -> float:
    """
    What a helper reckons one decision takes (bench/fp_bridge_server.py, _Server.search_seconds,
    the same sum): 2.5 x ms x parallelism + 20 ms when the worlds are searched one after the
    other; with workers w, ceil(2p / w) rounds of ms (or ceil(4p / w) of half) x 1.25, plus
    starting the w processes.
    """
    ms, p, w = max(2, int(ms)), max(1, int(parallelism or 1)), max(1, int(workers or 1))
    if w <= 1:
        return 2.5 * ms * p / 1000.0 + 0.02
    rounds = max(-(-2 * p // w), -(-4 * p // w) / 2.0)
    return 1.25 * ms * rounds / 1000.0 + 0.02 + FORK_S_PER_WORKER * w


def _clock() -> float:
    """
    Seconds on the clock deadlines are given on. time.monotonic() is one clock for every process
    of a machine on Linux, Windows and macOS and never jumps, unlike the wall clock. A helper
    shows its reading when it starts, and is sent deadlines only when it reads the same.
    """
    return time.monotonic()


class FoulPlayBridgeError(RuntimeError):
    revive = False    # True: the request asks whom Revival Blessing revives, which Foul Play cannot answer
    expired = False   # True: not searched, the answer could not have come before the caller stopped waiting
    down = False      # True: not asked, the battle's helper is not up (starting, hung, or its slot left down)


# =============================================================================================
# Finding the installation
# =============================================================================================

def foulplay_dir(path: Optional[str] = None) -> str:
    """The given path, or $FOULPLAY_DIR, or ~/foul-play."""
    return os.path.abspath(os.path.expanduser(path or os.environ.get("FOULPLAY_DIR") or DEFAULT_DIR))


def _venv_python(fp_dir: str, python: Optional[str] = None) -> Optional[str]:
    override = python or os.environ.get("FOULPLAY_PYTHON")
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


def _child_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = dict(os.environ)
    # The caller's Python settings belong to the training venv, not to Foul Play's.
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV", "__PYVENV_LAUNCHER__"):
        env.pop(key, None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["FP_BRIDGE_PARENT_PID"] = str(os.getpid())
    if extra:
        env.update(extra)
    return env


# ---- Windows: Foul Play lives in WSL --------------------------------------------------------

def _wsl_wanted(fp_dir: Optional[str] = None) -> bool:
    """
    True when this process should reach Foul Play through wsl.exe: on Windows, unless
    FOULPLAY_WSL=0, when the directory asked for is a path inside WSL ("/..." or a "~/..." that
    is no folder on the Windows side) or when none is given and there is no native checkout.
    A Windows path keeps the native behaviour, as on Linux.
    """
    if os.name != "nt":
        return False
    flag = (os.environ.get("FOULPLAY_WSL") or "auto").strip().lower()
    if flag in ("0", "no", "off", "false"):
        return False
    if flag in ("1", "yes", "on", "true"):
        return True
    given = fp_dir or os.environ.get("FOULPLAY_DIR")
    if not given:
        return not os.path.isdir(os.path.expanduser(DEFAULT_DIR))
    if given.startswith("/"):
        return True
    return given.startswith("~") and not os.path.isdir(os.path.expanduser(given))


def _wsl_base() -> List[str]:
    """wsl.exe and the distribution, the start of every command that runs something inside WSL."""
    exe = os.environ.get("FOULPLAY_WSL_EXE") or shutil.which("wsl.exe") or os.path.join(
        os.environ.get("SystemRoot") or r"C:\Windows", "System32", "wsl.exe")
    distro = (os.environ.get("FOULPLAY_WSL_DISTRO") or "").strip()
    return [exe, "-d", distro] if distro else [exe]


def wsl_path(path: str) -> str:
    """A Windows path as WSL sees it: C:\\a\\b -> /mnt/c/a/b, \\\\wsl.localhost\\Ubuntu\\home\\x -> /home/x. Others unchanged."""
    text = str(path)
    if text.startswith("/") or text.startswith("~"):
        return text
    unc = re.match(r"^[\\/]{2}(?:wsl\.localhost|wsl\$)[\\/][^\\/]+(.*)$", text, re.IGNORECASE)
    if unc:
        return unc.group(1).replace("\\", "/") or "/"
    text = os.path.abspath(text)
    drive, rest = os.path.splitdrive(text)
    if len(drive) == 2 and drive[1] == ":":
        return "/mnt/" + drive[0].lower() + rest.replace("\\", "/")
    return text.replace("\\", "/")


def _wsl_text(raw: Optional[bytes]) -> str:
    """Output of wsl.exe: UTF-8 from the Linux side, UTF-16 when wsl.exe itself complains."""
    raw = raw or b""
    if raw.count(b"\x00") > len(raw) // 4:
        return raw.decode("utf-16-le", "replace").replace("\ufeff", "")
    return raw.decode("utf-8", "replace")


def _wsl_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


# One sh script, read from stdin, that says what is installed inside WSL. Its arguments: Foul
# Play's python ("" = look in the checkout), "1" to try the imports, the format, then the
# directories to look in. The answers are key=value lines.
_WSL_PROBE = r'''
py="$1"; check="$2"; fmt="$3"; shift 3
fp=""; first=""
for d in "$@"; do
  case "$d" in "~/"*) d="$HOME/${d#??}";; "~") d="$HOME";; esac
  [ -n "$first" ] || first="$d"
  if [ -d "$d" ]; then fp="$d"; break; fi
done
echo "first=$first"
echo "dir=$fp"
[ -n "$fp" ] || exit 0
if [ -f "$fp/run.py" ] && [ -d "$fp/fp" ]; then echo "checkout=1"; fi
if [ -n "$py" ]; then
  case "$py" in "~/"*) py="$HOME/${py#??}";; esac
  [ -f "$py" ] || py=""
else
  for c in .venv/bin/python venv/bin/python; do
    if [ -f "$fp/$c" ]; then py="$fp/$c"; break; fi
  done
fi
echo "python=$py"
sets="$fp/fp/data/pkmn_sets_cache/$fmt.json"
echo "sets=$sets"
if [ -f "$sets" ]; then echo "sets_bytes=$(wc -c < "$sets" | tr -d ' ')"; fi
head=$(cat "$fp/.git/HEAD" 2>/dev/null)
case "$head" in
  "ref: "*) ref="${head#ref: }"
     if [ -f "$fp/.git/$ref" ]; then head=$(cat "$fp/.git/$ref");
     else head=$(grep " $ref\$" "$fp/.git/packed-refs" 2>/dev/null | cut -d" " -f1 | head -n 1); fi;;
esac
echo "commit=$head"
if [ "$check" = 1 ] && [ -n "$py" ]; then
  out=$(cd "$fp" && "$py" -c "import poke_engine, fp.run_battle, fp.modes.base, fp.search.main" 2>&1)
  echo "import=$?"
  echo "import_error=$(printf "%s" "$out" | tail -n 1)"
fi
'''


def _locate_wsl(fp_dir: Optional[str], python: Optional[str], check_imports: bool) -> Tuple[bool, str, Dict[str, str]]:
    """locate() for a Windows process: the same checks, made inside WSL by one sh script."""
    base = _wsl_base()
    given = fp_dir or os.environ.get("FOULPLAY_DIR")
    candidates = [given] if given else list(WSL_DIRS)
    py = python or os.environ.get("FOULPLAY_PYTHON") or ""
    cache_key = ("wsl:" + "|".join(base[1:] + candidates), py)
    if check_imports and cache_key in _LOCATE_CACHE:
        return _LOCATE_CACHE[cache_key]
    hint = "install it there with bench/install_foulplay.sh, or set FOULPLAY_DIR to its path inside WSL"
    command = base + ["--exec", "sh", "-s", "--", py, "1" if check_imports else "0", BATTLE_FORMAT] + candidates
    try:
        done = subprocess.run(command, input=_WSL_PROBE.replace("\r\n", "\n").encode("utf-8"), stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=180, creationflags=_wsl_flags())
    except subprocess.TimeoutExpired:
        return False, "WSL did not answer within 180 s (%s)" % " ".join(base), {}
    except OSError as exc:
        return False, "Foul Play does not build on Windows and wsl.exe could not be run (%s): %s" % (base[0], exc), {}
    seen: Dict[str, str] = {}
    for line in _wsl_text(done.stdout).splitlines():
        key, eq, value = line.partition("=")
        if eq:
            seen[key.strip()] = value.strip()
    if done.returncode != 0 or "first" not in seen:
        words = (_wsl_text(done.stderr) + " " + _wsl_text(done.stdout)).split()
        return False, "Foul Play does not build on Windows and WSL could not be asked for it (%s): %s" % (
            " ".join(base), " ".join(words)[:300] or "exit code %s" % done.returncode), {}
    where = "in WSL" + ((" " + base[2]) if len(base) > 2 else "")
    result = _probe_verdict(seen, where, candidates, hint, check_imports, {"wsl": "1"})
    if check_imports:
        _LOCATE_CACHE[cache_key] = result
    return result


def _probe_verdict(seen: Dict[str, str], where: str, candidates: List[str], hint: str, check_imports: bool,
                   extra: Dict[str, str]) -> Tuple[bool, str, Dict[str, str]]:
    """What _WSL_PROBE's answers (seen) say about an installation inside WSL or on another machine (where: "in WSL", "on user@host")."""
    found = seen.get("dir") or ""
    if not found:
        return False, "Foul Play is not installed %s: %s does not exist (%s)" % (where, " or ".join(candidates), hint), {}
    if seen.get("checkout") != "1":
        return False, "%s %s is not a Foul Play checkout: no run.py or fp/ (%s)" % (found, where, hint), {}
    if not seen.get("python"):
        return False, "Foul Play at %s %s has no virtualenv (.venv/bin/python): run bench/install_foulplay.sh there" % (found, where), {}
    if int(seen.get("sets_bytes") or 0) < 10_000:
        return False, "Foul Play at %s %s has no %s set data (%s): run bench/install_foulplay.sh there" % (
            found, where, BATTLE_FORMAT, seen.get("sets")), {}
    commit = seen.get("commit") or ""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        commit = "unknown"
    message = "foul-play %s at %s %s" % (commit, found, where)
    if commit != PINNED_COMMIT:
        message += " (not the pinned commit %s)" % PINNED_COMMIT[:12]
    info = dict({"dir": found, "python": seen["python"], "commit": commit}, **extra)
    if not check_imports:
        return True, message, info
    if seen.get("import") != "0":
        return (False, "Foul Play's virtualenv at %s %s cannot import Foul Play and poke-engine (%s): run bench/install_foulplay.sh there" % (
            found, where, seen.get("import_error") or "exit code %s" % seen.get("import")), {})
    return True, message, info


_WSL_TOKEN = "fpb.%d.%s." % (os.getpid(), os.urandom(4).hex())   # on the command line of every helper this process starts in WSL
_wsl_started = False    # True once this process has started a helper inside WSL


def _wsl_run(arguments: List[str], timeout: float = 30.0) -> Tuple[int, str]:
    """Runs one program inside WSL without a shell. (exit code, its output); (-1, reason) when it could not be run."""
    try:
        done = subprocess.run(_wsl_base() + ["--exec"] + arguments, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=timeout, creationflags=_wsl_flags())
        return done.returncode, _wsl_text(done.stdout)
    except Exception as exc:
        return -1, repr(exc)


def _wsl_kill(tag: str) -> None:
    """Stops, inside WSL, every helper whose command line carries this tag (or begins a tag with it)."""
    if tag:
        _wsl_run(["pkill", "-KILL", "-f", "fp_bridge_server.py.*--tag " + tag], timeout=30.0)


def wsl_leftovers(tag: Optional[str] = None) -> List[int]:
    """
    PIDs, inside WSL, of the helpers started by this process that are still running (tag: of any
    process whose tag begins with it, "fpb." for every helper of every process). For checks:
    after close() and after the death of a parent this must come back empty.
    """
    code, out = _wsl_run(["pgrep", "-f", "fp_bridge_server.py.*--tag " + (tag if tag is not None else _WSL_TOKEN)])
    return [int(word) for word in out.split() if word.isdigit()] if code in (0, 1) else []


# ---- Another machine, over ssh ----------------------------------------------------------------

def _git_roots() -> List[str]:
    """Where Git for Windows may be installed: next to the git.exe on PATH, and the usual places."""
    roots: List[str] = []
    git = shutil.which("git")
    if git:
        here = os.path.dirname(os.path.abspath(git))          # ...\Git\cmd, ...\Git\bin or ...\Git\mingw64\bin
        roots += [os.path.dirname(here), os.path.dirname(os.path.dirname(here))]
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramW6432"), os.environ.get("LOCALAPPDATA")):
        if base:
            roots += [os.path.join(base, "Git"), os.path.join(base, "Programs", "Git")]
    out: List[str] = []
    for root in roots:
        if root and root not in out:
            out.append(root)
    return out


def find_ssh() -> List[str]:
    """
    The ssh program, as a command: $FOULPLAY_SSH, else on Windows Git's own ssh (the one cloud/vm.sh
    runs, so the key, its permissions and the known hosts that work there work here), else the
    system's ssh (Windows' own OpenSSH, or ssh on PATH).
    """
    given = os.environ.get("FOULPLAY_SSH")
    if given:
        return [given]
    if os.name == "nt":
        for root in _git_roots():
            candidate = os.path.join(root, "usr", "bin", "ssh.exe")
            if os.path.isfile(candidate):
                return [candidate]
        system = os.path.join(os.environ.get("SystemRoot") or r"C:\Windows", "System32", "OpenSSH", "ssh.exe")
        if os.path.isfile(system):
            return [system]
    return [shutil.which("ssh") or "ssh"]


def find_bash() -> str:
    """Git's bash on Windows (never WSL's bash.exe), $FOULPLAY_BASH when set, bash elsewhere."""
    given = os.environ.get("FOULPLAY_BASH")
    if given:
        return given
    if os.name == "nt":
        for root in _git_roots():
            candidate = os.path.join(root, "bin", "bash.exe")
            if os.path.isfile(candidate):
                return candidate
        raise FoulPlayBridgeError("Git Bash was not found (set FOULPLAY_BASH to its bash.exe)")
    return shutil.which("bash") or "bash"


def _ssh_path(path: str) -> str:
    """A file name for ssh's command line: forward slashes (Git's ssh and Windows' OpenSSH both read C:/a/b)."""
    return str(path).replace("\\", "/") if os.name == "nt" else str(path)


def _sh_word(word: str) -> str:
    """One word of a command line for the remote shell. A leading ~/ stays unquoted, so the shell there expands it."""
    word = str(word)
    if word.startswith("~/"):
        rest = word[2:]
        return "~/" + (shlex.quote(rest) if rest else "")
    return shlex.quote(word)


@dataclass
class RemoteTarget:
    """
    Another machine that runs the search helpers, reached by `ssh [options] host <command>`.

    host         user@address. A cloud box's address changes at every start: look it up each run
                 (remote_target("vm") asks cloud/vm.sh).
    key          the private key (ssh -i), None for ssh's own choice
    ssh          the ssh program: a path, or a command as a list (the tests give a fake); default find_ssh()
    known_hosts  ssh's known-hosts file (default ssh's own). A host seen for the first time is
                 accepted and remembered (StrictHostKeyChecking=accept-new), a changed key is refused.
    code_dir     that machine's copy of this repository: its bench/fp_bridge_server.py is the helper
    fp_dir, python  Foul Play there and its python (default: REMOTE_FP_DIRS, the checkout's .venv)
    port, connect_timeout, alive_interval, alive_count: ssh's own settings. A connection that
    answers nothing for alive_interval x alive_count seconds ends the helper's ssh, and the bridge
    starts another (its battles rebuilt there).
    """
    host: str
    key: Optional[str] = None
    ssh: Any = None
    known_hosts: Optional[str] = None
    code_dir: str = REMOTE_CODE_DIR
    fp_dir: Optional[str] = None
    python: Optional[str] = None
    port: Optional[int] = None
    connect_timeout: int = 15
    alive_interval: int = 5
    alive_count: int = 3

    def argv(self) -> List[str]:
        """ssh and its options, up to and with the host: the remote command line follows as ONE argument."""
        if isinstance(self.ssh, (list, tuple)):
            program = [str(word) for word in self.ssh]
        else:
            program = [str(self.ssh)] if self.ssh else find_ssh()
        out = program + ["-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=%d" % int(self.connect_timeout),
                         "-o", "ServerAliveInterval=%d" % int(self.alive_interval),
                         "-o", "ServerAliveCountMax=%d" % int(self.alive_count),
                         "-o", "StrictHostKeyChecking=accept-new", "-o", "LogLevel=ERROR",
                         "-o", "ForwardAgent=no", "-o", "ForwardX11=no", "-o", "ClearAllForwardings=yes"]
        if self.known_hosts:
            out += ["-o", "UserKnownHostsFile=" + _ssh_path(self.known_hosts)]
        if self.key:
            out += ["-i", _ssh_path(self.key), "-o", "IdentitiesOnly=yes"]
        if self.port:
            out += ["-p", str(int(self.port))]
        return out + [self.host]

    @staticmethod
    def line(words: List[str]) -> str:
        """The remote command line: one string, which ssh hands to the login shell there."""
        return " ".join(_sh_word(word) for word in words)

    def server_path(self) -> str:
        return self.code_dir.rstrip("/") + "/bench/fp_bridge_server.py"


def _parse_ssh_config(text: str) -> Dict[str, str]:
    """`ssh -G` output (lower-case keyword, a space, the value) as a dict: the first value of each keyword."""
    seen: Dict[str, str] = {}
    for line in text.splitlines():
        key, _, value = line.strip().partition(" ")
        if key and value and key.lower() not in seen:
            seen[key.lower()] = value.strip()
    return seen


def default_vm_sh() -> str:
    """cloud/vm.sh with the box's settings (its .state): $PSPPO_VM_SH, this checkout's own, or ~/ps-ppo's."""
    given = os.environ.get("PSPPO_VM_SH")
    if given:
        return given
    own = REPO_ROOT / "cloud" / "vm.sh"
    if (REPO_ROOT / "cloud" / ".state").is_dir() and own.is_file():
        return str(own)
    return os.path.join(os.path.expanduser("~"), "ps-ppo", "cloud", "vm.sh")


def remote_target(spec: str, key: Optional[str] = None, ssh: Any = None, known_hosts: Optional[str] = None,
                  code_dir: Optional[str] = None, vm_sh: Optional[str] = None, timeout: float = 120.0) -> RemoteTarget:
    """
    A RemoteTarget from what a command line says: "user@host" (key, known_hosts as given), or "vm":
    the box cloud/vm.sh talks to, looked up NOW (its address changes at every start). For "vm" the
    address, the user, the key and the known-hosts file are what `vm.sh ssh -G` shows (ssh's own
    resolved settings, without connecting: no secret is read here, only file names), so this is the
    same box, key and host check as `bash cloud/vm.sh ssh`. Raises FoulPlayBridgeError saying why not.
    """
    spec = str(spec or "").strip()
    if not spec:
        raise FoulPlayBridgeError("no remote target given (user@host, or vm)")
    code_dir = code_dir or REMOTE_CODE_DIR
    if spec != "vm":
        if any(c.isspace() for c in spec) or spec.startswith("-"):
            raise FoulPlayBridgeError("the remote target %r is not user@host" % spec)
        return RemoteTarget(host=spec, key=key, ssh=ssh, known_hosts=known_hosts, code_dir=code_dir)
    script = vm_sh or default_vm_sh()
    if not os.path.isfile(script):
        raise FoulPlayBridgeError("cloud/vm.sh was not found at %s (set PSPPO_VM_SH, or give user@host)" % script)
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    try:
        done = subprocess.run([find_bash(), script, "ssh", "-G"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=timeout, env=env, creationflags=_wsl_flags())
    except subprocess.TimeoutExpired:
        raise FoulPlayBridgeError("cloud/vm.sh did not say where the box is within %.0f s" % timeout) from None
    except OSError as exc:
        raise FoulPlayBridgeError("cloud/vm.sh could not be run: %s" % exc) from None
    seen = _parse_ssh_config(done.stdout.decode("utf-8", "replace"))
    if done.returncode != 0 or not seen.get("hostname"):
        words = (done.stderr.decode("utf-8", "replace") + " " + done.stdout.decode("utf-8", "replace")).split()
        raise FoulPlayBridgeError("cloud/vm.sh did not give the box's address (is it running?): %s" % (
            " ".join(words)[:300] or "exit code %d" % done.returncode))
    host = "%s@%s" % (seen.get("user") or "ubuntu", seen["hostname"])
    files = (seen.get("userknownhostsfile") or "").split()
    found_key = seen.get("identityfile")
    return RemoteTarget(host=host, key=key or (_from_msys(found_key) if found_key else None), ssh=ssh,
                        known_hosts=known_hosts or (_from_msys(files[0]) if files else None), code_dir=code_dir,
                        port=int(seen["port"]) if (seen.get("port") or "22") != "22" else None)


def _from_msys(path: str) -> str:
    """Git Bash's /c/Users/x as C:/Users/x on Windows (for any ssh); anything else unchanged."""
    match = re.match(r"^/([A-Za-z])(/.*)?$", path or "")
    if os.name == "nt" and match:
        return match.group(1).upper() + ":" + (match.group(2) or "/")
    return path


def _server_sha() -> str:
    """sha256 of this repository's bench/fp_bridge_server.py, line ends as LF (a Windows checkout has CRLF)."""
    try:
        return hashlib.sha256(SERVER_PY.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    except OSError:
        return "missing"


# The same questions as _WSL_PROBE, asked on the other machine, and before them whether the helper
# file there is this one. Arguments: the helper file, then _WSL_PROBE's.
_REMOTE_PROBE = r'''
server="$1"; shift
case "$server" in "~/"*) server="$HOME/${server#??}";; esac
echo "server_path=$server"
if [ -f "$server" ]; then echo "server=1"; echo "server_sha=$(tr -d '\r' < "$server" | sha256sum | cut -c1-64)"; fi
echo "cpus=$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null)"
''' + _WSL_PROBE


def _remote_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)   # ssh gets no console of its own (and no Ctrl+C from ours)


def locate_remote(target: RemoteTarget, fp_dir: Optional[str] = None, python: Optional[str] = None,
                  check_imports: bool = True, timeout: float = 120.0) -> Tuple[bool, str, Dict[str, str]]:
    """
    locate() on another machine, by one ssh that runs a sh script there: Foul Play, its virtualenv,
    its set data, its imports, and the helper file in target.code_dir, which must be this
    repository's (compared by hash, line ends aside). (ok, message, info) with "remote": "1",
    "server" (the helper there), "cpus".
    """
    given = fp_dir or target.fp_dir
    candidates = [given] if given else list(REMOTE_FP_DIRS)
    py = python or target.python or ""
    cache_key = ("remote:%s|%s|%s" % (target.host, target.code_dir, "|".join(candidates)), py)
    if check_imports and cache_key in _LOCATE_CACHE:
        return _LOCATE_CACHE[cache_key]
    server = target.server_path()
    hint = "install it there with bench/install_foulplay.sh"
    command = target.argv() + [target.line(["sh", "-s", "--", server, py, "1" if check_imports else "0", BATTLE_FORMAT] + candidates)]
    try:
        done = subprocess.run(command, input=_REMOTE_PROBE.replace("\r\n", "\n").encode("utf-8"), stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=timeout, creationflags=_remote_flags())
    except subprocess.TimeoutExpired:
        return False, "%s did not answer over ssh within %.0f s" % (target.host, timeout), {}
    except OSError as exc:
        return False, "ssh could not be run (%s): %s" % (" ".join(command[:1]), exc), {}
    seen: Dict[str, str] = {}
    for line in done.stdout.decode("utf-8", "replace").splitlines():
        key, eq, value = line.partition("=")
        if eq:
            seen[key.strip()] = value.strip()
    if done.returncode != 0 or "first" not in seen:
        words = done.stderr.decode("utf-8", "replace").split() or done.stdout.decode("utf-8", "replace").split()
        return False, "%s could not be asked over ssh (exit code %d%s): %s" % (
            target.host, done.returncode, ", ssh itself failed" if done.returncode == 255 else "",
            " ".join(words)[-300:] or "no output"), {}
    where = "on " + target.host
    path = seen.get("server_path") or server
    if seen.get("server") != "1":
        return False, ("the search helper %s is not on %s: copy this repository to %s there "
                       "(bash cloud/remote_search.sh does)" % (path, target.host, target.code_dir)), {}
    if seen.get("server_sha") != _server_sha():
        return False, ("the search helper %s on %s is not this repository's (another version): copy this repository "
                       "to %s there first (bash cloud/remote_search.sh does it when the copies differ)" % (path, target.host, target.code_dir)), {}
    result = _probe_verdict(seen, where, candidates, hint, check_imports,
                            {"remote": "1", "server": path, "cpus": seen.get("cpus") or "?"})
    if result[0]:
        result = (True, result[1] + " (%s cores there)" % (seen.get("cpus") or "?"), result[2])
    if check_imports:
        _LOCATE_CACHE[cache_key] = result
    return result


_REMOTE_TOKEN = "fpr.%d.%s." % (os.getpid(), os.urandom(4).hex())   # on the command line of every helper this process starts over ssh
_remote_started: Dict[str, RemoteTarget] = {}    # host -> target, for every target this process has started a helper on


def _remote_pattern(tag: str) -> str:
    """
    pkill/pgrep's pattern for the helpers whose tag begins with `tag`. "[-]-tag fpr[.]..." matches
    "--tag fpr.123." on a helper's command line but not itself: the shell that runs pkill (whose
    command line carries the pattern) is never one of the processes found.
    """
    return "[-]-tag " + re.sub(r"([.])", r"[\1]", tag)


def _remote_run(target: RemoteTarget, words: List[str], timeout: float) -> Tuple[int, str]:
    """One command on the other machine. (exit code, output); (-1, reason) when ssh could not be run or timed out."""
    try:
        done = subprocess.run(target.argv() + [target.line(words)], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=timeout, creationflags=_remote_flags())
        return done.returncode, done.stdout.decode("utf-8", "replace")
    except Exception as exc:
        return -1, repr(exc)


def _remote_kill(target: RemoteTarget, tag: str, timeout: float = REMOTE_KILL_TIMEOUT_S) -> bool:
    """Stops, on the other machine, every helper whose tag begins with `tag` (and what it forked). True when that was done."""
    if not tag:
        return True
    code, out = _remote_run(target, ["pkill", "-KILL", "-f", _remote_pattern(tag)], timeout)
    if code in (0, 1):    # 1: nothing was left
        return True
    logger.warning("could not stop leftover Foul Play helpers on %s (%s): they stop by themselves within 20 s of "
                   "silence", target.host, " ".join(out.split())[:200])
    return False


def _remote_kill_later(target: RemoteTarget, tag: str) -> None:
    """_remote_kill on a thread of its own: an ssh takes seconds, and the watchdog must not wait for it."""
    threading.Thread(target=_remote_kill, args=(target, tag), name="fp-remote-kill", daemon=True).start()


def remote_leftovers(target: RemoteTarget, tag: Optional[str] = None, timeout: float = 60.0) -> Optional[List[int]]:
    """
    PIDs, on the other machine, of helpers started by this process that still run (tag: of any
    process whose tag begins with it, "fpr." for every remote helper of every process). None when
    the machine could not be asked. After close() this must come back empty.
    """
    code, out = _remote_run(target, ["pgrep", "-f", _remote_pattern(tag if tag is not None else _REMOTE_TOKEN)], timeout)
    if code not in (0, 1):
        return None
    return [int(word) for word in out.split() if word.isdigit()]


def remote_echo_round_trip(target: RemoteTarget, pings: int = 10, gap_s: float = 0.1, timeout: float = 60.0) -> Dict[str, Any]:
    """
    The network's own round trip to the other machine, through ssh as the helpers use it: one ssh
    running `cat` there, a short line written every gap_s and timed until it comes back. Nothing
    else runs there. {"ok", "rtt_ms" (median), "min_ms", "p90_ms", "max_ms", "pings", "error"}.
    """
    out: Dict[str, Any] = {"ok": False, "pings": 0}
    try:
        proc = subprocess.Popen(target.argv() + [target.line(["exec", "cat"])], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, creationflags=_remote_flags(), **({} if os.name == "nt" else {"start_new_session": True}))
    except OSError as exc:
        out["error"] = "ssh could not be run: %s" % exc
        return out
    lines: "queue.SimpleQueue[Optional[bytes]]" = queue.SimpleQueue()

    def read() -> None:
        try:
            for raw in proc.stdout:
                lines.put(raw)
        except Exception:
            pass
        lines.put(None)

    threading.Thread(target=read, name="fp-echo", daemon=True).start()
    times: List[float] = []
    pairs: List[float] = []
    try:
        # The first line also waits for the connection to be made: not counted.
        for n in range(int(pings) + 1):
            sent = time.monotonic()
            proc.stdin.write(b"echo %d\n" % n)
            proc.stdin.flush()
            try:
                raw = lines.get(timeout=timeout if n == 0 else 15.0)
            except queue.Empty:
                out["error"] = "no echo within %.0f s" % (timeout if n == 0 else 15.0)
                break
            if raw is None:
                out["error"] = "ssh ended: %s" % " ".join(proc.stderr.read().decode("utf-8", "replace").split())[:300]
                break
            if n:
                times.append(time.monotonic() - sent)
            time.sleep(gap_s)
        # Two lines written one right after the other, as a battle's lines and its question often
        # are: the second one's echo shows whether a small write waits for the first one's
        # acknowledgement on the way (Nagle's algorithm: one round trip more). About one round
        # trip: it does not.
        for n in range(5 if len(times) == int(pings) else 0):
            sent = time.monotonic()
            proc.stdin.write(b"first %d\n" % n)
            proc.stdin.flush()
            proc.stdin.write(b"second %d\n" % n)
            proc.stdin.flush()
            got = 0
            while got < 2:
                raw = lines.get(timeout=15.0)
                if raw is None:
                    raise OSError("ssh ended")
                got += 1
            pairs.append(time.monotonic() - sent)
            time.sleep(gap_s)
    except (OSError, queue.Empty) as exc:
        out["error"] = out.get("error") or "ssh ended: %r" % (exc,)
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=15)
        except Exception:
            _kill_tree(proc)
    if times:
        ms = sorted(1000.0 * t for t in times)
        out.update(ok=len(times) == int(pings) and "error" not in out, pings=len(ms), rtt_ms=round(_median(ms), 1),
                   min_ms=round(ms[0], 1), p90_ms=round(ms[min(len(ms) - 1, int(0.9 * len(ms)))], 1), max_ms=round(ms[-1], 1))
    if pairs:
        out["pair_ms"] = round(_median([1000.0 * t for t in pairs]), 1)
    return out


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return 0.0
    return ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


_LOCATE_CACHE: Dict[Tuple[str, str], Tuple[bool, str, Dict[str, str]]] = {}


def locate(fp_dir: Optional[str] = None, python: Optional[str] = None,
           check_imports: bool = True, remote: Optional[RemoteTarget] = None) -> Tuple[bool, str, Dict[str, str]]:
    """
    (ok, message, {"dir", "python", "commit"}). The import check starts a Python process once per installation.
    On Windows the installation is looked for inside WSL (see _wsl_wanted) and the answer carries "wsl": "1".
    With remote, on that machine over ssh (locate_remote; WSL is not used): "remote": "1".
    """
    if remote is not None:
        return locate_remote(remote, fp_dir, python, check_imports)
    if _wsl_wanted(fp_dir):
        return _locate_wsl(fp_dir, python, check_imports)
    fp_dir = foulplay_dir(fp_dir)
    hint = "run bench/install_foulplay.sh, or set FOULPLAY_DIR"
    if not SERVER_PY.is_file():
        return False, "the helper %s is missing from this repository" % SERVER_PY, {}
    if not os.path.isdir(fp_dir):
        return False, "Foul Play is not installed: %s does not exist (%s)" % (fp_dir, hint), {}
    if not (os.path.isfile(os.path.join(fp_dir, "run.py")) and os.path.isdir(os.path.join(fp_dir, "fp"))):
        return False, "%s is not a Foul Play checkout: no run.py or fp/ (%s)" % (fp_dir, hint), {}
    py = _venv_python(fp_dir, python)
    if py is None:
        return False, "Foul Play at %s has no virtualenv (.venv/bin/python): run bench/install_foulplay.sh" % fp_dir, {}
    sets = os.path.join(fp_dir, "fp", "data", "pkmn_sets_cache", BATTLE_FORMAT + ".json")
    if not os.path.isfile(sets) or os.path.getsize(sets) < 10_000:
        # Without it Foul Play downloads the sets in its first battle and caches an empty list when
        # that fails. It would then advise without knowing any set.
        return False, "Foul Play at %s has no %s set data (%s): run bench/install_foulplay.sh" % (fp_dir, BATTLE_FORMAT, sets), {}
    commit = _read_commit(fp_dir)
    message = "foul-play %s at %s" % (commit, fp_dir)
    if commit != PINNED_COMMIT:
        message += " (not the pinned commit %s)" % PINNED_COMMIT[:12]
    info = {"dir": fp_dir, "python": py, "commit": commit}
    if not check_imports:
        return True, message, info
    cached = _LOCATE_CACHE.get((fp_dir, py))
    if cached is not None:
        return cached
    try:
        done = subprocess.run([py, "-c", "import poke_engine, fp.run_battle, fp.modes.base, fp.search.main"],
                              cwd=fp_dir, env=_child_env(), stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    except subprocess.TimeoutExpired:
        return False, "Foul Play's virtualenv at %s did not answer within 120 s" % fp_dir, {}
    except OSError as exc:
        return False, "Foul Play's virtualenv at %s cannot be run: %s" % (fp_dir, exc), {}
    if done.returncode != 0:
        lines = (done.stdout or b"").decode("utf-8", "replace").strip().splitlines()
        last = lines[-1] if lines else "exit code %d" % done.returncode
        result: Tuple[bool, str, Dict[str, str]] = (
            False, "Foul Play's virtualenv at %s cannot import Foul Play and poke-engine (%s): run bench/install_foulplay.sh" % (fp_dir, last), {})
    else:
        result = (True, message, info)
    _LOCATE_CACHE[(fp_dir, py)] = result
    return result


def available(fp_dir: Optional[str] = None, python: Optional[str] = None,
              remote: Optional[RemoteTarget] = None) -> Tuple[bool, str]:
    """
    (True, "foul-play <commit> at <dir>") when the bridge can be used, else (False, reason).
    Looks in $FOULPLAY_DIR, default ~/foul-play; with remote, on that machine. Never raises.
    """
    try:
        ok, message, _info = locate(fp_dir, python, remote=remote)
        return ok, message
    except Exception as exc:  # this function must not raise
        return False, "could not check for Foul Play: %r" % (exc,)


# =============================================================================================
# What a decision looks like
# =============================================================================================

@dataclass
class Option:
    """One root action of Foul Play's search, summed over the sampled worlds."""
    choice: Optional[str]      # Showdown choice, "move 2 terastallize" or "switch 4". None: no such slot in the request
    fp: str                    # Foul Play's own name for it: "earthquake-tera", "switch garchomp"
    visits: int                # times the search tried it
    score: Optional[float]     # mean playout result from Foul Play's side, 0 to 1
    total_score: float
    policy: float              # Foul Play's share for it: mean over worlds of visits / iterations
    considered: bool = False   # within 75% of the largest share: the set Foul Play draws its move from


@dataclass
class OpponentOption:
    """One root option of the OPPONENT in Foul Play's search, over the sampled worlds."""
    name: str                  # "move earthquake", "move earthquake terastallize", "switch garchomp", "none"
    fp: str                    # poke-engine's own name for it
    share: float               # mean over the worlds of visits / iterations: the search's prediction
    visits: int
    score: Optional[float]     # mean playout result from the OPPONENT's side, 0 to 1
    worlds: int = 0            # in how many of the sampled worlds it was an option at all
    seen: Optional[bool] = None   # switches only: False when the Pokemon has not been shown (a sampled guess)


@dataclass
class Advice:
    """Foul Play's answer for one request."""
    choice: Optional[str]              # the move Foul Play picked (it draws among the considered options)
    best: Optional[str]                # the option with the largest share
    fp_choice: str = ""
    fp_message: str = ""               # what Foul Play itself would have sent, e.g. "/choose move earthquake terastallize"
    options: List[Option] = field(default_factory=list)
    worlds: int = 0                    # opponent teams sampled and searched
    iterations: int = 0                # search iterations, all worlds together
    iterations_per_world: List[int] = field(default_factory=list)
    search_ms: Optional[int] = None    # search time given to each world
    ms: float = 0.0                    # time inside the helper for this decision
    wall_ms: float = 0.0               # asked -> answered, queueing included
    turn: int = 0
    rqid: Any = None
    force_switch: bool = False
    tera_dropped: bool = False         # Foul Play asked for Terastallization where the request does not offer it
                                       # (locked into a move). The server plays the plain move, and so does `choice`.
    helper: int = -1
    # Filled by fill_actions(), which needs the poke-env battle: the same numbers on our action indices.
    action: Optional[int] = None       # index of `choice`, None when it is not a legal action of ours
    best_action: Optional[int] = None
    mask: List[int] = field(default_factory=list)                 # 14 entries, 1 = legal (create_action_mask)
    visits: List[int] = field(default_factory=list)               # 14 entries
    policy: List[float] = field(default_factory=list)             # 14 entries, sums to 1 when every option maps
    score: List[Optional[float]] = field(default_factory=list)    # 14 entries, None where the search has no visit
    unmapped: List[Option] = field(default_factory=list)          # searched options with no legal index of ours
    # The opponent's side of the same search: what the search itself expects the opponent to do.
    opponent: List[OpponentOption] = field(default_factory=list)  # largest share first
    opponent_active: Optional[str] = None                         # the opponent's active Pokemon, as Foul Play names it
    opponent_revealed: List[str] = field(default_factory=list)    # its Pokemon shown so far, the active one excluded
    opponent_worlds: List[Dict[str, Any]] = field(default_factory=list)   # per sampled world: "chance", "iterations",
                                       # "active" (the set drawn for its active Pokemon: species, moves, item, ability,
                                       # tera_type, hp, max_hp ...), "reserve" (the team it was given), "options"
    # Only when asked for (payoff_ms): values per (our option, the opponent's option).
    payoff: Optional[Dict[str, Any]] = field(default=None, repr=False)   # as the helper sent it, per world, our side by choice
    # Filled by fill_actions() from it: the same on our 14 actions.
    payoff_theirs: List[str] = field(default_factory=list)        # the opponent's options, by name: the columns
    payoff_table: List[List[Optional[float]]] = field(default_factory=list)   # [14][columns]: mean over the worlds that
                                       # have the cell, weighted by the world's chance. None: no world has it (not an
                                       # action, not an option there, pruned by the search, or out of budget)
    payoff_known: List[List[float]] = field(default_factory=list)  # [14][columns]: the share of the worlds behind each cell
    payoff_depth: List[int] = field(default_factory=list)          # plies searched, per world
    payoff_worlds: List[Dict[str, Any]] = field(default_factory=list)   # per world: "theirs", "table" [14][its columns], "depth", "chance"
    # Only when asked for (lines): the search's favourite options of ours, each played forward a few turns.
    lines: Optional[Dict[str, Any]] = field(default=None, repr=False)    # as the helper sent it (bench/fp_bridge_server.py "lines")
    # Filled by fill_actions() from it: per action of ours that has a line, {"value": the mean over the worlds
    # read (weighted by the world's chance) of the line's value, 0 lost to 1 won; "end": the same of its end
    # position alone; "turns": the fewest whole turns a world's line reached; "worlds": [{"world", "chance",
    # "value", "end", "turns", "steps", "cut"}], the likeliest world first}. Empty: no lines.
    lines_by_action: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    iterations_asked: int = 0          # > 0: every world was searched for exactly this many iterations
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)  # the helper's answer as it came


def _advice_from(reply: Dict[str, Any], wall_ms: float, helper: int) -> Advice:
    options = [
        Option(choice=o.get("choice"), fp=str(o.get("fp") or ""), visits=int(o.get("visits") or 0),
               score=o.get("score"), total_score=float(o.get("total_score") or 0.0),
               policy=float(o.get("policy") or 0.0), considered=bool(o.get("considered")))
        for o in reply.get("options") or []
    ]
    theirs = reply.get("opponent") or {}
    opponent = [
        OpponentOption(name=str(o.get("name") or ""), fp=str(o.get("fp") or ""), share=float(o.get("share") or 0.0),
                       visits=int(o.get("visits") or 0), score=o.get("score"), worlds=int(o.get("worlds") or 0), seen=o.get("seen"))
        for o in theirs.get("options") or []
    ]
    return Advice(
        opponent=opponent, opponent_active=theirs.get("active"), opponent_revealed=list(theirs.get("revealed") or []),
        opponent_worlds=list(theirs.get("worlds") or []), payoff=reply.get("payoff"),
        lines=reply.get("lines") if isinstance(reply.get("lines"), dict) else None,
        iterations_asked=int(reply.get("iterations_asked") or 0),
        choice=reply.get("choice"), best=reply.get("best"), fp_choice=str(reply.get("fp_choice") or ""),
        fp_message=str(reply.get("fp_message") or ""), options=options, worlds=int(reply.get("worlds") or 0),
        iterations=int(reply.get("iterations") or 0), iterations_per_world=list(reply.get("iterations_per_world") or []),
        search_ms=reply.get("search_ms"), ms=float(reply.get("ms") or 0.0), wall_ms=round(wall_ms, 2),
        turn=int(reply.get("turn") or 0), rqid=reply.get("rqid"), force_switch=bool(reply.get("force_switch")),
        tera_dropped=bool(reply.get("tera_dropped")), helper=helper, raw=reply,
    )


# =============================================================================================
# Our 14 actions. These three functions are ObservationAssembler.create_action_mask,
# map_order_to_index and map_index_to_order (obs_assembler.py) without numpy: worker.py turns the
# teacher's order into an action index with map_order_to_index and plays map_index_to_order of it.
# tests/test_foulplay_bridge.py holds them against the assembler.
# =============================================================================================

def legal_mask(battle: Any) -> List[int]:
    mask = [0] * ACTION_DIM
    active = battle.active_pokemon
    if not active:
        return mask
    if battle.available_moves:
        move_ids = [m.id for m in active.moves.values()]
        for move in battle.available_moves:
            if move.id in move_ids:
                idx = move_ids.index(move.id)
                if idx < 4:
                    mask[idx] = 1
                    if battle.can_tera:
                        mask[idx + 10] = 1
            elif move.id == "struggle":
                mask[0] = 1
        if not any(mask[:4]):
            mask[0] = 1
    if battle.available_switches:
        from switch_names import switch_name_taken
        team = list(battle.team.values())
        for mon in battle.available_switches:
            for idx, member in enumerate(team):
                if member is mon:
                    if idx < 6 and not switch_name_taken(active, mon):
                        mask[4 + idx] = 1
                    break
    if not any(mask):
        mask[0] = 1
    return mask


def order_to_index(order: Any, battle: Any) -> int:
    if not order:
        return 0
    choice = getattr(order, "order", None)
    if not choice:
        return 0
    if hasattr(choice, "id"):                     # a move
        active = battle.active_pokemon
        if not active:
            return 0
        move_ids = [m.id for m in active.moves.values()]
        if choice.id in move_ids:
            slot = move_ids.index(choice.id)
            return slot + 10 if getattr(order, "terastallize", False) else slot
        return 0
    if hasattr(choice, "current_hp"):             # a Pokemon: a switch
        for idx, member in enumerate(battle.team.values()):
            if member is choice:
                return 4 + idx
        return 4
    return 0


def index_to_target(index: int, battle: Any) -> Tuple[Any, bool]:
    """What worker.py would play for an action index: (Move or Pokemon or None, terastallize)."""
    active = battle.active_pokemon
    if not active:
        return None, False
    if 0 <= index <= 3 or 10 <= index <= 13:
        move_ids = [m.id for m in active.moves.values()]
        slot = index if index <= 3 else index - 10
        if slot < len(move_ids):
            wanted = move_ids[slot]
            for move in battle.available_moves:
                if move.id == wanted:
                    return move, index >= 10
                if move.id == "struggle" and slot == 0:
                    return move, False
        if battle.available_moves:
            return battle.available_moves[0], False
        if battle.available_switches:
            return battle.available_switches[0], False
        return None, False
    if 4 <= index <= 9:
        team = list(battle.team.values())
        if index - 4 < len(team) and team[index - 4]:
            return team[index - 4], False
    return None, False


def order_to_choice(battle: Any, order: Any) -> str:
    """A poke-env order as a Showdown choice with slot numbers. "default" for anything else."""
    try:
        if isinstance(getattr(order, "order", None), str):
            return "default"
        return _psbot_order_to_choice(battle, order) or "default"
    except Exception:
        return "default"


def choice_to_index(player: Any, battle: Any, choice: Optional[str],
                    index_of: Optional[Callable[[Any, Any], int]] = None,
                    mask: Optional[List[int]] = None) -> Optional[int]:
    """
    Our action index for a Showdown choice, or None when the choice is not one of our legal actions.
    It is only an index when playing that index plays this very choice: the same move with the
    same Terastallization, or the same Pokemon. Anything the 14 actions cannot say is None.
    """
    if not choice:
        return None
    order = choice_to_order(player, battle, choice)
    if order is None:
        return None
    index = int((index_of or order_to_index)(order, battle))
    if mask is None:
        mask = legal_mask(battle)
    if not (0 <= index < ACTION_DIM) or not mask[index]:
        return None
    target, tera = index_to_target(index, battle)
    wanted = getattr(order, "order", None)
    if target is None or wanted is None:
        return None
    if hasattr(wanted, "current_hp"):
        return index if target is wanted else None
    same_move = getattr(target, "id", None) == getattr(wanted, "id", object())
    return index if same_move and bool(tera) == bool(getattr(order, "terastallize", False)) else None


def fill_actions(player: Any, battle: Any, advice: Advice,
                 index_of: Optional[Callable[[Any, Any], int]] = None,
                 mask: Optional[List[int]] = None) -> Advice:
    """
    Puts the search statistics of `advice` on our 14 action indices. Returns the same object.
    Options that mean the same choice are added up (the helper turns a Terastallization the
    request does not offer into the plain move: see tera_dropped).
    """
    mask = [int(bool(x)) for x in (mask if mask is not None else legal_mask(battle))]
    visits = [0] * ACTION_DIM
    policy = [0.0] * ACTION_DIM
    totals = [0.0] * ACTION_DIM
    unmapped: List[Option] = []
    cache: Dict[Optional[str], Optional[int]] = {}

    def index(choice: Optional[str]) -> Optional[int]:
        if choice not in cache:
            cache[choice] = choice_to_index(player, battle, choice, index_of, mask)
        return cache[choice]

    for option in advice.options:
        idx = index(option.choice)
        if idx is None:
            unmapped.append(option)
            continue
        visits[idx] += option.visits
        policy[idx] += option.policy
        totals[idx] += option.total_score
    advice.mask = mask
    advice.visits = visits
    advice.policy = policy
    advice.score = [round(totals[i] / visits[i], 5) if visits[i] else None for i in range(ACTION_DIM)]
    advice.unmapped = unmapped
    advice.action = index(advice.choice)
    advice.best_action = max(range(ACTION_DIM), key=lambda i: policy[i]) if any(v > 0 for v in policy) else None
    if advice.payoff:
        _fill_payoff(advice, index)
    if advice.lines:
        _fill_lines(advice, index)
    return advice


def _fill_lines(advice: Advice, index: Callable[[Optional[str]], Optional[int]]) -> None:
    """
    The lines of the sampled worlds on our 14 actions (Advice.lines_by_action). An action's value
    is the mean of its lines' values over the worlds that have one, weighted by the world's
    chance. A line without a value (an error, the time ran out before its first search) is left out.
    """
    table: Dict[int, Dict[str, Any]] = {}
    for world in (advice.lines or {}).get("worlds") or []:
        if not isinstance(world, dict):
            continue
        chance = float(world.get("chance") or 0.0)
        for line in world.get("lines") or []:
            if not isinstance(line, dict) or line.get("value") is None:
                continue
            idx = index(line.get("choice"))
            if idx is None:
                continue
            row = table.setdefault(idx, {"worlds": []})
            if any(w["world"] == world.get("world") for w in row["worlds"]):
                continue                 # two options of ours that are the same action: the first one stands
            row["worlds"].append({"world": world.get("world"), "chance": chance, "value": float(line["value"]),
                                  "end": line.get("end"), "turns": int(line.get("turns") or 0),
                                  "steps": list(line.get("steps") or []), "cut": bool(line.get("cut"))})
    for row in table.values():
        row["worlds"].sort(key=lambda w: -w["chance"])
        total = sum(w["chance"] for w in row["worlds"])
        weights = [(w["chance"] / total) if total > 0 else 1.0 / len(row["worlds"]) for w in row["worlds"]]
        row["value"] = round(sum(k * w["value"] for k, w in zip(weights, row["worlds"])), 4)
        ends = [(k, float(w["end"])) for k, w in zip(weights, row["worlds"]) if w["end"] is not None]
        row["end"] = round(sum(k * e for k, e in ends) / sum(k for k, _e in ends), 4) if ends and sum(k for k, _e in ends) > 0 else None
        row["turns"] = min(w["turns"] for w in row["worlds"])
    advice.lines_by_action = table


def _fill_payoff(advice: Advice, index: Callable[[Optional[str]], Optional[int]]) -> None:
    """
    The payoff tables of the sampled worlds on our 14 actions: one table per world, and their
    mean. A cell is the mean over the worlds that have a value for it, weighted by the world's
    chance; payoff_known says how much of the weight that was. Two options of ours that are the
    same action (see tera_dropped) share a row: the first value found stands.
    """
    columns: List[str] = []
    per_world: List[Dict[str, Any]] = []
    for world in (advice.payoff or {}).get("worlds") or []:
        theirs = [str(name) for name in world.get("theirs") or []]
        table: List[List[Optional[float]]] = [[None] * len(theirs) for _ in range(ACTION_DIM)]
        for choice, row in zip(world.get("choices") or [], world.get("matrix") or []):
            idx = index(choice)
            if idx is None:
                continue
            for j, value in enumerate(row[:len(theirs)]):
                if value is not None and table[idx][j] is None:
                    table[idx][j] = float(value)
        for name in theirs:
            if name not in columns:
                columns.append(name)
        per_world.append({"theirs": theirs, "table": table, "depth": int(world.get("depth") or 0),
                          "chance": float(world.get("chance") or 0.0), "error": world.get("error")})
    total = sum(w["chance"] for w in per_world) or 1.0
    mean: List[List[Optional[float]]] = [[None] * len(columns) for _ in range(ACTION_DIM)]
    known = [[0.0] * len(columns) for _ in range(ACTION_DIM)]
    for a in range(ACTION_DIM):
        for c, name in enumerate(columns):
            value, weight = 0.0, 0.0
            for world in per_world:
                if name in world["theirs"]:
                    cell = world["table"][a][world["theirs"].index(name)]
                    if cell is not None:
                        value += world["chance"] * cell
                        weight += world["chance"]
            if weight > 0:
                mean[a][c] = round(value / weight, 3)
                known[a][c] = round(weight / total, 4)
    advice.payoff_theirs = columns
    advice.payoff_table = mean
    advice.payoff_known = known
    advice.payoff_depth = [w["depth"] for w in per_world]
    advice.payoff_worlds = per_world


# =============================================================================================
# One helper process
# =============================================================================================

_LIVE: Set[Any] = set()  # every Popen this module started and has not yet seen dead


def _kill_tree(popen: Any) -> None:
    """
    Stops one helper, and anything it started, by its PID (its process group on Linux). For a
    helper on another machine that is its ssh here; the helper there is then stopped by its tag,
    by an ssh of its own on a thread of its own, when its ssh was still running or failed as ssh
    (exit code 255: the connection went, the helper there may not have noticed yet).
    """
    remote = getattr(popen, "fp_remote", None)
    if remote is not None:
        running = popen.poll() is None
        if running or popen.returncode == 255:
            popen.fp_remote = None
            _remote_kill_later(remote[0], remote[1])
    try:
        if os.name == "nt":
            if popen.poll() is None:
                subprocess.run(["taskkill", "/PID", str(popen.pid), "/T", "/F"], stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
                # That was wsl.exe. The helper is a Linux process inside WSL and is stopped there,
                # by the tag on its command line (it would leave by itself once its stdin closes,
                # but not while the engine is searching).
                _wsl_kill(getattr(popen, "fp_wsl_tag", ""))
        else:
            # Only while the helper has not been reaped: once it has, its PID may be another
            # process's (the PIDs of a busy machine wrap within hours), and a stray SIGKILL to a
            # process group on a box shared with a trainer is not a risk worth taking. An
            # "inline" helper starts no processes of its own, so nothing is left behind.
            try:
                if popen.poll() is None:
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
    live = list(_LIVE)
    tagged = [p for p in live if getattr(p, "fp_wsl_tag", "") and p.poll() is None]
    for popen in tagged:
        popen.fp_wsl_tag = ""       # stopped inside WSL below, all of them with one call
    remote = [p for p in live if getattr(p, "fp_remote", None) is not None]
    hosts = {p.fp_remote[0].host: p.fp_remote[0] for p in remote}
    for popen in remote:
        popen.fp_remote = None      # stopped on their machine below, one call per machine
    for popen in live:
        _kill_tree(popen)
    if tagged:
        _wsl_kill(_WSL_TOKEN)
    for target in hosts.values():
        _remote_kill(target, _REMOTE_TOKEN)


@dataclass
class _Pending:
    rid: int
    message: Dict[str, Any]
    future: "concurrent.futures.Future"
    kind: str
    key: Optional[str]
    sent: float
    attempts: int = 1
    clean: bool = False     # a probe sent while its helper had no decision to make: its round trip is the network's


class _Helper:
    """One helper process: a queue and a thread that writes it, a thread that reads the answers."""

    def __init__(self, bridge: "FoulPlayBridge", index: int, generation: int):
        self.bridge = bridge
        self.index = index
        self.generation = generation
        self.proc: Any = None
        self.pending: Dict[int, _Pending] = {}
        self.plock = threading.Lock()
        self.outbox: "queue.SimpleQueue[Optional[bytes]]" = queue.SimpleQueue()
        self.stderr_tail: Deque[str] = deque(maxlen=30)
        self.info: Dict[str, Any] = {}
        self.fatal: Optional[str] = None
        self.exited = False
        self.replies = 0
        self.started = time.monotonic()
        self.last_reply = self.started
        self.clock_ok = False   # True once its first answer showed that it reads the clock we read
        self.clock_offset = 0.0  # its clock minus ours: 0 for a helper of this machine, kept up to date for one inside WSL
        self.tag = ""           # on its command line inside WSL or on the other machine: what it is found and stopped by there
        self.last_beat = self.started
        # A helper on another machine (bridge._remote): the network between us, from its clock probes.
        self.samples: Deque[Tuple[float, float, bool]] = deque(maxlen=REMOTE_WINDOW)   # (round trip s, clock offset, clean)
        self.rtt_s: Optional[float] = None     # the round trip now: the median of the clean probes in the window
        self.return_s = 0.0                    # what a deadline keeps for the answer's way back: rtt/2 + REMOTE_MARGIN_S
        self.probe_sent: Optional[float] = None  # an unanswered probe went out at this time
        self.hung = False                      # stopped by the watchdog: it had work and answered nothing

    def start(self) -> None:
        kwargs: Dict[str, Any] = {}
        command = list(self.bridge._command)
        remote = self.bridge._remote
        env = _child_env(self.bridge._extra_env)
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            if self.bridge._wsl:
                global _wsl_started
                _wsl_started = True
                kwargs["creationflags"] |= _wsl_flags()
                self.tag = "%s%d.%d.%d." % (_WSL_TOKEN, self.bridge._serial, self.index, self.generation)
                command += ["--tag", self.tag]
            if remote is not None:
                kwargs["creationflags"] |= _remote_flags()
        else:
            kwargs["start_new_session"] = True   # its own process group: see _kill_tree
        if remote is not None:
            # ssh with ONE argument after the host: the helper's command line there, with its tag
            # (what it is found and stopped by there) and its environment (none of ours reaches it).
            self.tag = "%s%d.%d.%d." % (_REMOTE_TOKEN, self.bridge._serial, self.index, self.generation)
            command.append(remote.line(["exec", "env"] + self.bridge._remote_env + self.bridge._remote_words + ["--tag", self.tag]))
            env = _child_env()
            _remote_started[remote.host] = remote
        self.proc = subprocess.Popen(command, cwd=self.bridge._cwd, env=env,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
        self.proc.fp_wsl_tag = self.tag if self.bridge._wsl else ""
        self.proc.fp_remote = (remote, self.tag) if remote is not None else None
        _LIVE.add(self.proc)
        name = "fp-helper-%d.%d" % (self.index, self.generation)
        threading.Thread(target=self._write_loop, name=name + "-w", daemon=True).start()
        threading.Thread(target=self._read_loop, name=name + "-r", daemon=True).start()
        threading.Thread(target=self._stderr_loop, name=name + "-e", daemon=True).start()

    @property
    def alive(self) -> bool:
        return self.proc is not None and not self.exited and self.proc.poll() is None

    def post(self, message: Dict[str, Any]) -> None:
        """Queued for the writer thread. Never blocks, whatever the helper is doing."""
        self.outbox.put((json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8"))

    def ask(self, pending: _Pending) -> None:
        with self.plock:
            self.pending[pending.rid] = pending
        self.post(dict(pending.message, id=pending.rid))

    def take_pending(self) -> List[_Pending]:
        with self.plock:
            taken, self.pending = list(self.pending.values()), {}
        return taken

    def greeted(self, reply: Dict[str, Any]) -> None:
        """Its answer to a ping: who it is, and whether its clock is ours (see _clock)."""
        self.info.update(reply)
        clock = reply.get("clock")
        if self.bridge._remote is not None:
            # Another machine's clock. This answer came after Foul Play loaded, so its timing says
            # nothing about the network: the probes measure both (probed), deadlines wait for them.
            return
        if self.bridge._wsl and isinstance(clock, (int, float)):
            # A helper inside WSL reads the clock of another kernel. The difference is kept
            # instead (see heard()): its deadlines are given on its own clock.
            self.heard(clock)
            return
        self.clock_ok = isinstance(clock, (int, float)) and abs(float(clock) - _clock()) < _CLOCK_TOLERANCE_S
        if not self.clock_ok:
            logger.warning("Foul Play helper %d does not show the clock this process reads (%r): its decisions get no "
                           "deadline, so a decision nobody waits for any more is still searched", self.index, clock)

    def heard(self, clock: Any) -> None:
        """
        An answer of a helper inside WSL shows its clock, read as the answer was written. That
        minus our clock now is the difference between the two, short by the few milliseconds the
        answer travelled, so a deadline is at most that much early. Taken from every answer: the
        two clocks do not stay in step when the laptop sleeps.
        """
        if self.bridge._wsl and isinstance(clock, (int, float)):
            self.clock_offset = float(clock) - _clock()
            self.clock_ok = True

    def probed(self, sent: float, clock: Any, clean: bool) -> Optional[float]:
        """
        A clock probe of a remote helper came back: sent at `sent`, answered with the helper's
        clock. The round trip is now minus sent; the helper read its clock about half way, so its
        clock minus ours is clock - (sent + now) / 2, with an error of at most half that round trip
        (and less the more even the two ways are). Of the last REMOTE_WINDOW probes, the offset of
        the one with the shortest round trip is used (the best measured, within a minute: two
        clocks drift apart by far less than a millisecond in that time), and the round trip is the
        median of those sent while the helper had no decision to make (clean: not queued behind a
        search). Returns this probe's round trip.
        """
        if self.probe_sent is not None and sent >= self.probe_sent:
            self.probe_sent = None
        if not isinstance(clock, (int, float)):
            return None
        now = _clock()
        rtt = max(0.0, now - sent)
        self.samples.append((rtt, float(clock) - (sent + now) / 2.0, bool(clean)))
        self.clock_offset = min(self.samples, key=lambda s: s[0])[1]
        trips = [s[0] for s in self.samples if s[2]]
        if trips:
            self.rtt_s = _median(trips)
        elif self.rtt_s is None:
            self.rtt_s = _median([s[0] for s in self.samples])
        self.return_s = (self.rtt_s or 0.0) / 2.0 + REMOTE_MARGIN_S
        self.clock_ok = True
        return rtt

    @property
    def transit_s(self) -> float:
        """What the network adds to a decision of a remote helper: there and back, plus the margin. 0 here or in WSL."""
        if self.bridge._remote is None or self.rtt_s is None:
            return 0.0
        return self.rtt_s + REMOTE_MARGIN_S

    def decisions_waiting(self) -> int:
        """Decisions asked of this helper that are not answered yet and that somebody still waits for."""
        with self.plock:
            return sum(1 for p in self.pending.values() if p.kind == "decide" and not p.future.done())

    def idle(self) -> bool:
        """No decision in its queue at all (also none whose caller gave up): a probe sent now is not held up by a search."""
        with self.plock:
            return not any(p.kind == "decide" for p in self.pending.values())

    def stuck(self, now: float, hang_timeout: float, start_timeout: float) -> Optional[_Pending]:
        """
        The oldest unanswered question, when this helper has answered nothing for too long although
        it has work. A long queue is not a hang: a healthy helper answers something every decision.
        """
        with self.plock:
            if not self.pending:
                return None
            oldest = min(self.pending.values(), key=lambda p: p.sent)
        if not self.replies:
            limit = start_timeout                      # still loading Foul Play
        else:
            asked_ms = (float(oldest.message.get("ms") or 0.0) * max(1, int(oldest.message.get("parallelism") or 1))
                        + float(oldest.message.get("payoff_ms") or 0.0)
                        + float((oldest.message.get("lines") or {}).get("ms") or 0.0))
            limit = max(hang_timeout, 5.0 * asked_ms / 1000.0)   # a very long search is not a hang either
        return oldest if now - max(self.last_reply, oldest.sent) > limit else None

    def _write_loop(self) -> None:
        stdin = self.proc.stdin
        try:
            while True:
                item = self.outbox.get()
                if item is None:
                    break
                stdin.write(item)
                if self.outbox.empty():
                    stdin.flush()
        except (OSError, ValueError):
            pass   # the helper is gone: the reader thread sees the end of its output and reports it
        try:
            stdin.close()
        except Exception:
            pass

    def _read_loop(self) -> None:
        try:
            for raw in self.proc.stdout:
                try:
                    message = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(message, dict):
                    continue
                if message.get("fatal"):
                    self.fatal = str(message["fatal"])
                    continue
                with self.plock:
                    pending = self.pending.pop(message.get("id"), None)
                self.replies += 1
                self.last_reply = time.monotonic()
                if pending is not None:
                    self.bridge._deliver(self, pending, message)
        except Exception as exc:  # a broken pipe object on interpreter shutdown, for one
            logger.debug("helper %d reader stopped: %r", self.index, exc)
        try:
            self.proc.wait(timeout=10)
        except Exception:
            pass
        self.exited = True
        self.outbox.put(None)
        self.bridge._helper_exited(self)

    def _stderr_loop(self) -> None:
        try:
            for raw in self.proc.stderr:
                line = raw.decode("utf-8", "replace").rstrip()
                if line:
                    self.stderr_tail.append(line)
                    logger.debug("helper %d: %s", self.index, line)
        except Exception:
            pass

    def why(self) -> str:
        if self.fatal:
            return self.fatal
        tail = [line for line in self.stderr_tail if line.strip()]
        code = self.proc.poll() if self.proc is not None else None
        if code is not None and code < 0:
            return "killed by signal %d" % -code
        return "exit code %s%s" % (code, (", last log line: " + tail[-1][:300]) if tail else "")

    def stop(self, grace: float = 0.5) -> None:
        """Asks the helper to leave, then stops its process group whatever it answered."""
        if self.proc is None:
            return
        if self.bridge._wsl:
            grace = max(grace, WSL_GRACE_S)   # through wsl.exe the end of stdin takes seconds to arrive, and to come back
        self.outbox.put(None)   # the writer closes stdin, on which the helper exits
        deadline = time.monotonic() + grace
        while self.proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        _kill_tree(self.proc)


# =============================================================================================
# The pool
# =============================================================================================

_BRIDGE_SERIAL = itertools.count(1)


class _BattleRecord:
    __slots__ = ("helper", "user", "room", "log", "nbytes", "fed", "last")

    def __init__(self, helper: int, user: str, room: str):
        self.helper = helper
        self.user = user
        self.room = room
        self.log: List[Tuple[str, str]] = []   # ("feed", data) and ("chose", choice), in order
        self.nbytes = 0
        self.fed = 0
        self.last = time.monotonic()


class FoulPlayBridge:
    """
    A pool of Foul Play helpers.

    helpers        processes, which is cores while they search.
    search_ms      Foul Play's --search-time-ms, used when a call does not give one. Foul Play
                   searches 4 sampled teams at half this early in a game and 2 at the full time
                   later, so a decision takes about twice this plus 15 to 40 ms.
    fp_dir, python where Foul Play is. Default: $FOULPLAY_DIR or ~/foul-play, and its .venv.
    command        the helper's full command line, instead of the real one (the tests use it).
    pool           "inline" (search in the helper) or "fork" (Foul Play's own process pool).
    timeout        how long advice() waits before it gives up, in seconds. The helper is told: a
                   decision that could not be answered before then is not searched when its turn
                   comes (it would only make the ones behind it late as well).
    hang_timeout   a helper that has work and answers nothing for this long is stopped and replaced.
    max_restarts   a helper that had to be replaced more often than this within restart_window
                   seconds is left down for cooldown seconds; its battles get errors meanwhile.
    max_battles    battles whose lines are kept at once. Beyond it the one unused longest is dropped.

    For live play (ladder.py), all off by default:
    early_expire   a question with a ttl that cannot be answered in time, given the decisions
                   already waiting in its helper, fails at once as "expired" instead of waiting
                   for its turn there (the caller plays its own move at once, not at the deadline).
    idle_ping      seconds: a helper with nothing to do is asked a sync this often, so a frozen
                   helper with no battle is found (and replaced, after hang_timeout) before a
                   battle is put on it.
    background_spawn  a helper that has to be started again is started by a thread of its own:
                   the process is never created by the caller of feed(), note() or advice_future()
                   (the player's event loop) nor with the lock held. What it is sent meanwhile
                   waits in its queue. A slot left down (it kept failing) is started again by the
                   watchdog as soon as its cooldown is over, not only when a battle asks for it.
    fail_when_down a question whose helper is not up (still starting, not answering anything for
                   `timeout` although it has work, or its slot left down) fails at once with
                   .down, instead of waiting in the queue of a helper that cannot answer it in
                   time: the caller plays its own move at once (WSL gone or restarting, ladder.py).

    remote         a RemoteTarget: the helpers run on that machine, started over ssh (see the top of
                   this file). fp_dir and python are then paths there, and `command` (the tests) is
                   the helper's command line THERE, run through the target's ssh. start() also
                   measures the round trip (rtt_start); every decision's deadline and early_expire
                   account for it, and stats() shows it per helper.
    """

    def __init__(self, helpers: int = 1, search_ms: int = DEFAULT_SEARCH_MS, *, fp_dir: Optional[str] = None,
                 python: Optional[str] = None, command: Optional[List[str]] = None, pool: str = "inline",
                 timeout: float = 60.0, hang_timeout: float = 60.0, start_timeout: float = 60.0,
                 max_restarts: int = 5, restart_window: float = 60.0, cooldown: float = 30.0,
                 max_battles: int = 4096, max_battle_bytes: int = 64 << 20,
                 battle_format: str = BATTLE_FORMAT, env: Optional[Dict[str, str]] = None,
                 early_expire: bool = False, idle_ping: Optional[float] = None, background_spawn: bool = False,
                 fail_when_down: bool = False, remote: Optional[RemoteTarget] = None):
        if helpers < 1:
            raise ValueError("helpers must be at least 1")
        self.search_ms = int(search_ms)
        self.timeout = float(timeout)
        self.early_expire = bool(early_expire)
        self.idle_ping = float(idle_ping) if idle_ping else None
        self.background_spawn = bool(background_spawn)
        self.fail_when_down = bool(fail_when_down)
        self.hang_timeout = float(hang_timeout)
        self.start_timeout = float(start_timeout)
        self.max_restarts = int(max_restarts)
        self.restart_window = float(restart_window)
        self.cooldown = float(cooldown)
        self.max_battles = int(max_battles)
        self.max_battle_bytes = int(max_battle_bytes)
        self.battle_format = battle_format
        self.pool = pool
        self._extra_env = dict(env or {})
        self._fp_dir = fp_dir
        self._python = python
        self._remote = remote
        # Remote: the helper's command line on the other machine (without "exec env" and its tag), and its environment.
        self._remote_words: Optional[List[str]] = list(command) if (command and remote is not None) else None
        self._remote_env: List[str] = []
        self.rtt_start: Optional[Dict[str, Any]] = None   # remote: the round trip measured when the helpers started
        self._command: Optional[List[str]] = list(command) if (command and remote is None) else None
        self._cwd: Optional[str] = None
        self._wsl = False            # True: the helpers are started inside WSL through wsl.exe (a Windows process)
        self._serial = next(_BRIDGE_SERIAL)
        self.install: str = ""
        self._lock = threading.RLock()
        self._ids = itertools.count(1)
        self._n = int(helpers)
        self._helpers: List[Optional[_Helper]] = [None] * self._n
        self._generation = [0] * self._n
        self._exits: List[Deque[float]] = [deque() for _ in range(self._n)]
        self._down_until = [0.0] * self._n
        self._down_why = [""] * self._n
        self._decide_s: List[Optional[float]] = [None] * self._n   # what one decision takes in each helper, a running mean
        self._recent: Deque[int] = deque(maxlen=_RECENT_DECISIONS)   # iterations of the last decisions answered
        self._recent_lock = threading.Lock()                          # written by the helpers' reader threads, read by stats()
        self._battles: Dict[str, _BattleRecord] = {}
        self._gone: "OrderedDict[str, None]" = OrderedDict()
        self._closed = False
        self._watchdog: Optional[threading.Thread] = None
        self._wake = threading.Event()
        self.counters: Dict[str, int] = {k: 0 for k in (
            "decisions", "decision_errors", "timeouts", "expired", "helper_exits", "helper_starts", "hangs",
            "rebuilt_battles", "refed_lines", "asked_again", "fed", "notes", "late_lines", "forgotten", "left", "evicted",
            "truncated", "rejoined", "iterations", "helper_ms", "expired_early", "idle_pings", "not_up",
            "slot_restarts", "link_late", "link_downs")}
        # Remote: the link's troubles (link_status, link_spans). A slot's trouble is open from when its
        # helper went quiet until a helper answers in it again; a held-up line is a closed span.
        self._link_lock = threading.Lock()
        self._link_spans: Deque[List[Any]] = deque()               # [start, end or None, why], time.time()
        self._slot_span: List[Optional[List[Any]]] = [None] * self._n   # a slot's open span (also in _link_spans)
        self._slot_cause: List[Optional[Tuple[str, float]]] = [None] * self._n   # its last exit: ("link"|"helper", quiet since)
        self._late_until = 0.0                                       # monotonic: a held-up line came in lately
        self._late_since = 0.0                                       # time.time(): the first of those lines held up
        self._late_why = ""
        _BRIDGES.add(self)

    # ---- lifecycle ---------------------------------------------------------------------------

    @property
    def helpers(self) -> int:
        return self._n

    @property
    def where(self) -> str:
        """Where the search runs: "remote" (another machine, over ssh), "wsl" (inside WSL) or "local"."""
        return "remote" if self._remote is not None else ("wsl" if self._wsl else "local")

    @property
    def remote_host(self) -> Optional[str]:
        return self._remote.host if self._remote is not None else None

    def round_trip_ms(self) -> Optional[float]:
        """Remote: the round trip now, the median over the helpers of each one's (from its probes). None otherwise."""
        with self._lock:
            trips = [h.rtt_s for h in self._helpers if h is not None and not h.exited and h.rtt_s is not None]
        return round(1000.0 * _median(trips), 1) if trips else None

    def _resolve_command(self) -> None:
        if self._command is not None:
            return
        if self._remote is not None:
            # Another machine: ssh, its options and the host; each helper adds its command line there
            # (_Helper.start). Settings go on that command line through env(1): nothing of this
            # process's environment crosses ssh. It watches its parent there (the ssh session) and
            # the beat (the clock probes), and leads a process group of its own.
            self._remote_env = ["PYTHONIOENCODING=utf-8", "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1"]
            self._remote_env += ["%s=%s" % (key, value) for key, value in self._extra_env.items()]
            if self._remote_words is None:
                ok, message, info = locate_remote(self._remote, self._fp_dir, self._python)
                if not ok:
                    raise FoulPlayBridgeError(message)
                self.install = message
                self._remote_words = [info["python"], "-u", info["server"], "--foulplay-dir", info["dir"],
                                      "--format", self.battle_format, "--pool", self.pool, "--parent-watch", "beat", "--own-group"]
            elif not self.install:
                self.install = "the given helper command on %s" % self._remote.host
            self._cwd = None
            self._command = self._remote.argv()
            return
        ok, message, info = locate(self._fp_dir, self._python)
        if not ok:
            raise FoulPlayBridgeError(message)
        self.install = message
        if info.get("wsl"):
            # A Windows process: the helper is Foul Play's python INSIDE WSL, started by wsl.exe
            # without a shell. Its stdin and stdout are this process's pipes, carried by wsl.exe.
            # Settings go on the command line (the Windows environment does not reach it), the
            # extra environment through env(1). It watches its own parent, the relay of that
            # wsl.exe, and leads a process group of its own.
            self._wsl = True
            self._cwd = None
            settings = ["PYTHONIOENCODING=utf-8", "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1"]
            settings += ["%s=%s" % (key, value) for key, value in self._extra_env.items()]
            self._command = _wsl_base() + ["--exec", "/usr/bin/env"] + settings + [
                info["python"], "-u", wsl_path(str(SERVER_PY)), "--foulplay-dir", info["dir"],
                "--format", self.battle_format, "--pool", self.pool, "--parent-watch", "beat", "--own-group"]
            return
        self._cwd = info["dir"]
        self._command = [info["python"], "-u", str(SERVER_PY), "--foulplay-dir", info["dir"],
                         "--format", self.battle_format, "--pool", self.pool]

    def _spawn(self, index: int, now_here: bool = False) -> _Helper:
        """
        Starts the helper in slot `index` and asks it who it is. Does not wait. Lock held.
        With background_spawn (and not now_here) the process itself is created by a thread of
        its own, after the lock is released: the helper is in its slot at once, and whatever is
        posted to it meanwhile waits in its queue until its writer thread starts.
        """
        if _exiting:
            raise FoulPlayBridgeError("the interpreter is exiting: no Foul Play helper is started now")
        self._resolve_command()
        self._generation[index] += 1
        helper = _Helper(self, index, self._generation[index])
        if self.background_spawn and not now_here:
            threading.Thread(target=self._start_helper, args=(helper,), name="fp-spawn-%d" % index, daemon=True).start()
        else:
            try:
                helper.start()
            except OSError as exc:
                raise FoulPlayBridgeError("could not start the Foul Play helper %r: %s" % (self._command, exc)) from exc
        self._helpers[index] = helper
        self.counters["helper_starts"] += 1
        now = time.monotonic()
        hello: "concurrent.futures.Future" = concurrent.futures.Future()
        helper.ask(_Pending(next(self._ids), {"op": "ping"}, hello, "ping", None, now))   # answered through _Helper.greeted
        helper.info["hello"] = hello
        if self._watchdog is None:
            self._watchdog = threading.Thread(target=self._watch, name="fp-bridge-watchdog", daemon=True)
            self._watchdog.start()
        return helper

    def _start_helper(self, helper: _Helper) -> None:
        """background_spawn: creates the helper's process, outside the lock (see _spawn)."""
        try:
            with self._lock:
                gone = self._closed or self._helpers[helper.index] is not helper or _exiting
            if not gone:
                helper.start()
        except Exception as exc:  # treated as the helper's exit: replaced, or its slot left down
            helper.fatal = "could not start the Foul Play helper %r: %s" % (self._command, exc)
            helper.exited = True
            self._helper_exited(helper)
            return
        with self._lock:
            gone = self._closed or self._helpers[helper.index] is not helper
        if gone and helper.proc is not None:
            helper.stop(grace=0.0)   # the bridge was closed (or the slot taken) while the process was being created
        elif gone:
            for pending in helper.take_pending():
                self._fail(pending, "the Foul Play bridge was closed")

    def _helper(self, index: int, now_here: bool = False) -> _Helper:
        """The helper in a slot, started if need be. Lock held. Raises while the slot is down."""
        if self._closed:
            raise FoulPlayBridgeError("the Foul Play bridge is closed")
        helper = self._helpers[index]
        if helper is not None:
            # Also when its process has just ended: _helper_exited() is waiting for the lock held
            # here, and will take over whatever is asked of this helper in the meantime.
            return helper
        now = time.monotonic()
        if now < self._down_until[index]:
            error = FoulPlayBridgeError("Foul Play helper %d is down for %.0f s more: %s" % (
                index, self._down_until[index] - now, self._down_why[index]))
            error.down = True
            raise error
        helper = self._spawn(index, now_here)
        self._rebuild(helper)
        return helper

    def start(self, wait: bool = True, timeout: Optional[float] = None) -> "FoulPlayBridge":
        """Starts every helper. With wait, returns when all have answered, or raises saying why not."""
        with self._lock:
            started = [self._helper(i, now_here=True) for i in range(self._n)]
        if wait:
            deadline = time.monotonic() + (timeout if timeout is not None else self.start_timeout)
            for helper in started:
                try:
                    helper.info["hello"].result(timeout=max(0.1, deadline - time.monotonic()))
                except concurrent.futures.TimeoutError:
                    raise FoulPlayBridgeError("Foul Play helper %d did not answer within %.0f s (%s)" % (
                        helper.index, self.start_timeout, helper.why())) from None
                except Exception as exc:
                    raise FoulPlayBridgeError("Foul Play helper %d did not start: %s" % (helper.index, exc)) from None
            if self._remote is not None:
                self.rtt_start = self.measure_round_trip()
        return self

    def measure_round_trip(self, rounds: int = REMOTE_START_PROBES, timeout: float = 15.0) -> Dict[str, Any]:
        """
        Remote helpers: `rounds` clock probes to every helper that is up, one round after the other
        (each helper has nothing else to do then, so each probe's time is the network's). Sets
        every helper's round trip and clock difference (_Helper.probed) and returns, over all
        probes: {"rtt_ms" (median), "min_ms", "p90_ms", "max_ms", "probes", "helpers": [median per
        helper, ms]}. {} when there is no remote helper.
        """
        if self._remote is None:
            return {}
        per: Dict[int, List[float]] = {}
        for _ in range(max(1, int(rounds))):
            asked = []
            with self._lock:
                for helper in self._helpers:
                    if helper is None or helper.exited or not helper.replies:
                        continue
                    future: "concurrent.futures.Future" = concurrent.futures.Future()
                    helper.ask(_Pending(next(self._ids), {"op": "ping"}, future, "probe", None, time.monotonic(), clean=True))
                    asked.append((helper, future))
            if not asked:
                break
            for helper, future in asked:
                try:
                    reply = future.result(timeout=timeout)
                except Exception:
                    continue
                trip = reply.get("_rtt_s") if isinstance(reply, dict) else None
                if isinstance(trip, (int, float)):
                    per.setdefault(helper.index, []).append(1000.0 * float(trip))
        every = sorted(ms for trips in per.values() for ms in trips)
        if not every:
            return {"probes": 0}
        return {"rtt_ms": round(_median(every), 1), "min_ms": round(every[0], 1),
                "p90_ms": round(every[min(len(every) - 1, int(0.9 * len(every)))], 1), "max_ms": round(every[-1], 1),
                "probes": len(every), "helpers": [round(_median(per[i]), 1) for i in sorted(per)]}

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            helpers = [h for h in self._helpers if h is not None]
            self._helpers = [None] * self._n
            self._battles.clear()
        self._wake.set()
        if self._wsl:
            # Every helper is asked to leave at once, and they are waited for together. What is
            # still there after that (a helper in the middle of a search does not see its stdin
            # close) is stopped inside WSL, all of it with one call: a wsl.exe costs seconds.
            for helper in helpers:
                helper.outbox.put(None)
            deadline = time.monotonic() + WSL_GRACE_S
            while time.monotonic() < deadline and any(h.proc is not None and h.proc.poll() is None for h in helpers):
                time.sleep(0.05)
            left = [h for h in helpers if h.proc is not None and h.proc.poll() is None]
            for helper in left:
                helper.proc.fp_wsl_tag = ""
            for helper in left:
                _kill_tree(helper.proc)
            if left:
                _wsl_kill("%s%d." % (_WSL_TOKEN, self._serial))
        if self._remote is not None and self.counters["helper_starts"]:
            # The same on the other machine: every stdin closed at once (ssh carries the end of
            # input, and the helper leaves after the search it may be in), waited for together; the
            # ssh still running after that is stopped here, and then whatever of this bridge is
            # still running there (also of helpers replaced earlier, whose connection went quiet)
            # is stopped by its tag, all of it with one ssh.
            for helper in helpers:
                helper.outbox.put(None)
            deadline = time.monotonic() + REMOTE_GRACE_S
            while time.monotonic() < deadline and any(h.proc is not None and h.proc.poll() is None for h in helpers):
                time.sleep(0.05)
            for helper in helpers:
                if helper.proc is not None:
                    helper.proc.fp_remote = None
                    if helper.proc.poll() is None:
                        _kill_tree(helper.proc)
            _remote_kill(self._remote, "%s%d." % (_REMOTE_TOKEN, self._serial))
        for helper in helpers:
            helper.stop()
            for pending in helper.take_pending():
                self._fail(pending, "the Foul Play bridge was closed")
        _BRIDGES.discard(self)

    def __enter__(self) -> "FoulPlayBridge":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- answers and failures (called from helper threads) -------------------------------------

    @staticmethod
    def _fail(pending: _Pending, reason: str) -> None:
        if not pending.future.done():
            try:
                pending.future.set_exception(FoulPlayBridgeError(reason))
            except concurrent.futures.InvalidStateError:
                pass

    def _deliver(self, helper: _Helper, pending: _Pending, reply: Dict[str, Any]) -> None:
        if self._remote is not None:
            try:
                self._transit(helper, pending, reply)   # with the clock difference as it was when the line was written
            except Exception as exc:
                logger.debug("link bookkeeping failed: %r", exc)
        if pending.kind == "decide":
            if reply.get("ok"):
                # What a decision takes in this helper, whoever still waits for this one. With the
                # questions waiting in it (stats) that says how far behind the helper is.
                took = float(reply.get("ms") or 0.0) / 1000.0
                mean = self._decide_s[helper.index]
                self._decide_s[helper.index] = took if mean is None else mean + 0.1 * (took - mean)
            elif reply.get("expired"):
                self.counters["expired"] += 1   # a search the helper did not run for nobody
            helper.heard(reply.get("clock"))
        elif pending.kind == "ping" and reply.get("ok"):
            helper.greeted(reply)               # before anybody sees the answer: start() waits for it
            if self._remote is not None and not helper.samples:
                # A remote helper that has just loaded: its clock at once, so that its first
                # decisions get deadlines. Not clean: its battles' lines may still be ahead of it.
                self._probe(helper, clean=False)
        elif pending.kind == "probe":
            trip = helper.probed(pending.sent, reply.get("clock"), pending.clean)
            reply = dict(reply, _rtt_s=trip)
        if pending.future.done():   # the caller stopped waiting
            return
        try:
            if pending.kind != "decide":
                pending.future.set_result(reply)
            elif reply.get("ok"):
                advice = _advice_from(reply, (time.monotonic() - pending.sent) * 1000.0, helper.index)
                self.counters["decisions"] += 1
                # The search is limited by time, so on a machine with no free core it does fewer
                # iterations and advises worse without anything else looking different. These
                # sums are there to be watched and acted on (teacher_problems): iterations /
                # decisions must stay where it was when the search time was chosen.
                self.counters["iterations"] += advice.iterations
                self.counters["helper_ms"] += int(round(advice.ms))
                with self._recent_lock:
                    self._recent.append(advice.iterations)
                pending.future.set_result(advice)
            else:
                self.counters["decision_errors"] += 1
                error = FoulPlayBridgeError(str(reply.get("error") or "Foul Play gave no decision"))
                error.revive = bool(reply.get("revive"))
                error.expired = bool(reply.get("expired"))
                pending.future.set_exception(error)
        except concurrent.futures.InvalidStateError:
            pass

    def _helper_exited(self, helper: _Helper) -> None:
        """The process of `helper` is gone. Replace it, rebuild its battles, ask again what was waiting."""
        if helper.proc is not None:
            _kill_tree(helper.proc)   # whatever it may have left in its process group
        with self._lock:
            # Taken under the lock: every question is asked under it, so none can slip in behind.
            waiting = helper.take_pending()
            index = helper.index
            if self._closed or self._helpers[index] is not helper:
                for pending in waiting:
                    self._fail(pending, "the Foul Play helper was stopped")
                return
            reason = helper.why()
            self.counters["helper_exits"] += 1
            if self._remote is not None:
                self._exit_cause(helper, waiting)
            self._helpers[index] = None
            now = time.monotonic()
            exits = self._exits[index]
            exits.append(now)
            while exits and now - exits[0] > self.restart_window:
                exits.popleft()
            if len(exits) > self.max_restarts:
                self._down_until[index] = now + self.cooldown
                self._down_why[index] = reason
                exits.clear()
                logger.error("Foul Play helper %d keeps failing (%s): leaving it down for %.0f s", index, reason, self.cooldown)
                for pending in waiting:
                    self._fail(pending, "the Foul Play helper keeps failing: %s" % reason)
                return
            logger.warning("Foul Play helper %d (pid %s) is gone: %s. Starting another and rebuilding its battles.",
                           index, getattr(helper.proc, "pid", "?"), reason)
            try:
                fresh = self._spawn(index)
            except FoulPlayBridgeError as exc:
                self._down_until[index] = now + self.cooldown
                self._down_why[index] = str(exc)
                for pending in waiting:
                    self._fail(pending, str(exc))
                return
            self._rebuild(fresh)
            for pending in waiting:
                if pending.kind == "decide" and pending.attempts < 2 and pending.key in self._battles and not pending.future.done():
                    # Once more, after the battle's lines: the helper may have died of something else.
                    pending.attempts += 1
                    pending.rid = next(self._ids)
                    self.counters["asked_again"] += 1
                    fresh.ask(pending)
                else:
                    self._fail(pending, "the Foul Play helper stopped while answering: %s" % reason)

    def _rebuild(self, helper: _Helper) -> None:
        """Feeds a new helper everything its battles have seen and chosen so far. Lock held."""
        for key, record in self._battles.items():
            if record.helper != helper.index or not record.log:
                continue
            for kind, payload in record.log:
                if kind == "feed":
                    helper.post({"op": "feed", "key": key, "user": record.user, "room": record.room, "data": payload})
                else:
                    helper.post({"op": "chose", "key": key, "choice": payload})
            # A sign of life after every battle: rebuilding many battles takes a while (about
            # 0.2 s each), and a helper that answers nothing for hang_timeout is taken for hung.
            beat: "concurrent.futures.Future" = concurrent.futures.Future()
            helper.ask(_Pending(next(self._ids), {"op": "sync"}, beat, "sync", None, time.monotonic()))
            self.counters["rebuilt_battles"] += 1
            self.counters["refed_lines"] += len(record.log)

    def _watch(self) -> None:
        """Stops a helper that has work and has answered nothing for too long. Its reader does the rest."""
        while not self._closed:
            self._wake.wait(0.5)
            if self._closed:
                return
            if self.background_spawn:
                # A slot left down after repeated failures (WSL gone, say) is started again the
                # moment its cooldown is over: the search is back as soon as the helpers can run,
                # not when a battle next happens to ask. The process is created by a thread of
                # its own (_spawn), so this costs the watchdog nothing.
                with self._lock:
                    for index in range(self._n):
                        if (not self._closed and self._helpers[index] is None and not _exiting
                                and time.monotonic() >= self._down_until[index] > 0.0):
                            try:
                                self._helper(index)
                                self.counters["slot_restarts"] += 1
                            except FoulPlayBridgeError as exc:
                                logger.debug("slot %d not started again yet: %s", index, exc)
            now = time.monotonic()
            with self._lock:
                helpers = [h for h in self._helpers if h is not None and not h.exited]
            for helper in helpers:
                if self._wsl and now - helper.last_beat >= WSL_BEAT_S:
                    helper.last_beat = now
                    helper.post({"op": "beat"})     # "this process is still there": see --parent-watch beat in the helper
                if (self._remote is not None and helper.replies and helper.alive and now - helper.last_beat >= REMOTE_PROBE_S
                        and (helper.probe_sent is None or now - helper.probe_sent > REMOTE_PROBE_STALE_S)):
                    # A clock probe: the round trip and the clock difference kept up to date, and the
                    # line the helper's --parent-watch beat waits for. One at a time per helper.
                    helper.last_beat = now
                    self._probe(helper)
                if (self.idle_ping and helper.replies and helper.alive and now - helper.last_reply >= self.idle_ping
                        and not helper.pending):
                    # Nothing asked of it, so stuck() would never look at it: ask it something. A
                    # frozen helper then has work and no answer, and is replaced after hang_timeout.
                    with self._lock:
                        if self._helpers[helper.index] is helper and not helper.pending:
                            ping: "concurrent.futures.Future" = concurrent.futures.Future()
                            helper.ask(_Pending(next(self._ids), {"op": "sync"}, ping, "sync", None, now))
                            self.counters["idle_pings"] += 1
                overdue = helper.stuck(now, self.hang_timeout, self.start_timeout)
                if overdue is not None and helper.proc is not None and helper.proc.poll() is None:
                    self.counters["hangs"] += 1
                    helper.hung = True
                    helper.fatal = helper.fatal or "no answer to anything for %.0f s, a %s request waiting" % (
                        now - max(helper.last_reply, overdue.sent), overdue.kind)
                    logger.warning("Foul Play helper %d (pid %s) gave %s: stopping it", helper.index, helper.proc.pid, helper.fatal)
                    _kill_tree(helper.proc)
            try:
                self._link_tick()
            except Exception as exc:   # bookkeeping: never in the way of the watchdog's own work
                logger.debug("link bookkeeping failed: %r", exc)

    # ---- the link to remote helpers ------------------------------------------------------------

    @staticmethod
    def _wall(mono: float) -> float:
        """A time.monotonic() as time.time()."""
        return time.time() - (time.monotonic() - mono)

    def _one_way_s(self, helper: _Helper) -> float:
        """Half the round trip as measured when the helpers started (the link's usual way), else as now."""
        start = self.rtt_start or {}
        if isinstance(start.get("rtt_ms"), (int, float)):
            return float(start["rtt_ms"]) / 2000.0
        return (helper.rtt_s or 0.3) / 2.0

    def _transit(self, helper: _Helper, pending: _Pending, reply: Dict[str, Any]) -> None:
        """
        A remote helper's answer just came in: did the network hold it up on its way back, or the
        question on its way there, by REMOTE_LATE_S more than usual? Then that time is a link stall
        (link_spans), and the link counts as down for REMOTE_LATE_HOLD_S (link_status). The way
        back: now minus when the helper wrote it ("clock"). The way there: when the helper had last
        waited for a line ("came", at or before this question's arrival) minus when it was sent; a
        question that waited behind a search is not taken for held up.
        """
        if not helper.clock_ok:
            return
        arrival = time.monotonic()
        limit = self._one_way_s(helper) + REMOTE_LATE_S
        start: Optional[float] = None
        end: Optional[float] = None
        clock, came = reply.get("clock"), reply.get("came")
        if isinstance(clock, (int, float)):
            written = float(clock) - helper.clock_offset
            if arrival - written > limit:
                start, end = written, arrival
        if isinstance(came, (int, float)) and came > 0:
            there = float(came) - helper.clock_offset
            if there - pending.sent > limit:
                start = pending.sent if start is None else min(start, pending.sent)
                end = there if end is None else max(end, there)
        if start is None or end is None:
            return
        why = "the network held a line of helper %d up for %.1f s" % (helper.index, end - start)
        with self._link_lock:
            now = time.monotonic()
            if now >= self._late_until:
                self._late_since = self._wall(start)
            else:
                self._late_since = min(self._late_since, self._wall(start))
            self._late_until = now + REMOTE_LATE_HOLD_S
            self._late_why = why
            self._link_spans.append([self._wall(start), self._wall(end), why])
            self.counters["link_late"] += 1

    def _exit_cause(self, helper: _Helper, waiting: List[_Pending]) -> None:
        """
        A remote helper's process ended (lock held): was it the link or the helper? The link: the
        watchdog stopped it (it answered nothing), ssh lost its connection (exit code 255), or it
        left by itself after its parent went quiet (0). The helper: Foul Play's process there failed
        by itself (any other exit code). Kept for the slot until a helper answers in it again.
        """
        code = helper.proc.poll() if helper.proc is not None else None
        link = helper.hung or code is None or code < 0 or code in (0, 255)
        quiet = max([helper.last_reply] + [p.sent for p in waiting]) if waiting else time.monotonic()
        self._slot_cause[helper.index] = ("link" if link else "helper", quiet)
        if link:
            # Down from now, not from the watchdog's next look: a new helper may be up before it.
            self._slot_trouble(helper.index, "helper %d is being replaced after its link dropped" % helper.index, quiet)

    def _slot_trouble(self, index: int, why: Optional[str], since: float) -> None:
        """Opens (why: a line) or closes (why None) a slot's span of link trouble. since: monotonic."""
        with self._link_lock:
            span = self._slot_span[index]
            if why is not None:
                if span is None:
                    span = [self._wall(since), None, why]
                    self._slot_span[index] = span
                    self._link_spans.append(span)
                    self.counters["link_downs"] += 1
                else:
                    span[2] = why
            elif span is not None:
                span[1] = time.time()
                self._slot_span[index] = None

    def _link_tick(self) -> None:
        """
        Remote: opens and closes each slot's link trouble (link_status, link_spans). A slot is in
        trouble from when its helper went quiet: one that has work and has answered nothing for
        `timeout`, or the replacement (starting, or the slot left down) of one whose link dropped,
        until a helper answers in it again. A slot whose helper failed by itself is not: that is
        the search failing, counted as before.
        """
        if self._remote is None or self._closed:
            return
        for index in range(self._n):
            with self._lock:   # one slot at a time, never between a helper's exit and its replacement (_helper_exited)
                if self._closed:
                    return
                now = time.monotonic()
                helper = self._helpers[index]
                why: Optional[str] = None
                since = now
                if helper is not None and self._up(helper, now):
                    self._slot_cause[index] = None
                elif helper is not None and helper.alive and helper.replies:
                    oldest = helper.stuck(now, self.timeout, self.start_timeout)
                    if oldest is not None:
                        since = max(helper.last_reply, oldest.sent)
                    why = "helper %d has answered nothing for %.0f s" % (index, now - since)
                else:
                    cause = self._slot_cause[index]
                    if cause is not None and cause[0] == "link":
                        since = cause[1]
                        why = "helper %d is being replaced after its link dropped" % index
                self._slot_trouble(index, why, since)
        with self._link_lock:
            old = time.time() - LINK_SPANS_KEPT_S
            while self._link_spans and self._link_spans[0][1] is not None and self._link_spans[0][1] < old:
                self._link_spans.popleft()

    def link_status(self) -> Optional[Dict[str, Any]]:
        """
        Remote: None while the link to the helpers is fine, else {"since": time.time() it went,
        "why": one line, "slots": helpers in trouble}: a helper that has answered nothing for
        `timeout` although it has work, a helper being replaced after its link dropped (also its
        slot left down), or a line the network held up within the last REMOTE_LATE_HOLD_S. Not a
        helper whose own process failed (the search's failure). None here and in WSL.
        """
        if self._remote is None or self._closed:
            return None
        with self._link_lock:
            spans = [s for s in self._slot_span if s is not None]
            late = time.monotonic() < self._late_until
            if not spans and not late:
                return None
            starts = [s[0] for s in spans] + ([self._late_since] if late else [])
            whys = [s[2] for s in spans] + ([self._late_why] if late and not spans else [])
            return {"since": min(starts), "why": "; ".join(whys[:2]) + ("; and %d more" % (len(whys) - 2) if len(whys) > 2 else ""),
                    "slots": len(spans)}

    def link_spans(self) -> List[Tuple[float, Optional[float]]]:
        """Remote: the link's troubles of the last LINK_SPANS_KEPT_S, (start, end or None while it lasts), time.time(). [] otherwise."""
        with self._link_lock:
            return [(s[0], s[1]) for s in self._link_spans]

    def _probe(self, helper: _Helper, clean: Optional[bool] = None) -> None:
        """Sends a remote helper a clock probe (a ping; answered through _Helper.probed). clean: default, no decision in its queue."""
        with self._lock:
            if self._closed or self._helpers[helper.index] is not helper:
                return
            sent = time.monotonic()
            helper.probe_sent = sent
            probe: "concurrent.futures.Future" = concurrent.futures.Future()
            helper.ask(_Pending(next(self._ids), {"op": "ping"}, probe, "probe", None, sent,
                                clean=helper.idle() if clean is None else bool(clean)))

    # ---- battles -----------------------------------------------------------------------------

    def _up(self, helper: Optional[_Helper], now: float) -> bool:
        """
        Can this helper answer a question soon: its process runs, it has answered its first
        question (Foul Play is loaded), and it has not gone `timeout` without answering anything
        while it had work (frozen: the watchdog replaces it after hang_timeout).
        """
        return (helper is not None and helper.alive and helper.replies > 0
                and helper.stuck(now, self.timeout, self.start_timeout) is None)

    def helpers_up(self) -> int:
        """How many helpers can answer now (see _up). 0: the search is down (WSL gone, say)."""
        with self._lock:
            if self._closed:
                return 0
            now = time.monotonic()
            return sum(1 for helper in self._helpers if self._up(helper, now))

    def restart_helpers(self, why: str) -> int:
        """
        Every running helper stopped, as the watchdog stops a frozen one: _helper_exited starts its
        slot again and rebuilds its battles from their stored lines. For helpers that are up but
        failing every question (after the network went away and came back, say). Questions waiting
        are asked again of the new helper. Returns how many were stopped. Never raises.
        """
        stopped = 0
        try:
            with self._lock:
                if self._closed:
                    return 0
                helpers = [h for h in self._helpers if h is not None and not h.exited and h.proc is not None]
            for helper in helpers:
                if helper.proc.poll() is None:
                    helper.fatal = helper.fatal or ("restarted: " + why)
                    self.counters["restarts_asked"] = self.counters.get("restarts_asked", 0) + 1
                    _kill_tree(helper.proc)
                    stopped += 1
        except Exception as exc:   # a restart that could not be made leaves the helpers as they were
            logger.warning("the search helpers could not be restarted: %s", exc)
        return stopped

    def _least_loaded(self) -> int:
        load = [0] * self._n
        for record in self._battles.values():
            load[record.helper] += 1
        now = time.monotonic()
        return min(range(self._n), key=lambda i: (now < self._down_until[i], load[i], i))

    def _helper_or_none(self, index: int) -> Optional[_Helper]:
        """
        The battle's helper, or None while its slot is down (the stored log brings it up to date
        when it is back). Call it BEFORE adding to the battle's log: starting a helper replays the log.
        """
        try:
            return self._helper(index)
        except FoulPlayBridgeError as exc:
            logger.debug("not sent to the Foul Play helper: %s", exc)
            return None

    @staticmethod
    def _missed_lines(record: "_BattleRecord", replay: str) -> Optional[str]:
        """
        The lines of `replay` (the whole battle, sent again from |init| after the room was joined
        again) that come after the last "|turn|N" line the battle was fed: what it missed. "" when
        it missed nothing, None when that turn is not in the replay or no turn was fed (then the
        battle cannot be resumed). A turn's lines come in one message, so none is half fed.
        """
        last = None
        for kind, payload in reversed(record.log):
            if kind != "feed":
                continue
            for line in reversed(payload.split("\n")):
                if line.startswith("|turn|"):
                    last = line.strip()
                    break
            if last is not None:
                break
        if last is None:
            return None
        lines = replay.split("\n")
        at = next((i for i in range(len(lines) - 1, -1, -1) if lines[i].strip() == last), None)
        if at is None:
            return None
        return "\n".join(lines[at + 1:]).strip("\n")

    def knows(self, key: str) -> bool:
        """Whether the battle is followed here (fed and not let go): one joined again can be resumed."""
        with self._lock:
            return key in self._battles

    def feed(self, key: str, user: str, room: str, data: str) -> None:
        """One websocket message of a battle room, as the player received it. Never blocks, never raises."""
        if not data:
            return
        with self._lock:
            if self._closed:
                return
            if data == "|deinit" or "|deinit" in data.split("\n"):
                # The player is out of the room: it left, or the room was closed under it (the
                # training server closes every battle room after five minutes). No line of the
                # battle will come any more and nobody reports it as finished, so it is forgotten
                # here. Without this its lines here and its state in the helper stay for good.
                if key in self._battles:
                    self.counters["left"] += 1
                    self._drop(key)
                elif key in self._gone:
                    self.counters["late_lines"] += 1
                return
            tail = self._missed_lines(self._battles[key], data) if data.startswith("|init|") and key in self._battles else None
            if tail is not None:
                # The player joined the room again after a lost connection and is sent the whole
                # battle once more. The battle is kept: only the lines after the last turn it was
                # fed go on, so the search resumes where it was (the lines it already has are not
                # fed twice, the ones it missed are not lost).
                self.counters["resumed"] = self.counters.get("resumed", 0) + 1
                if not tail:
                    return
                data = tail
            elif data.startswith("|init|") and (key in self._gone or key in self._battles):
                # The player joined the room again and is sent the whole battle once more, and
                # where it left off cannot be found in it (no turn fed yet, or the battle was
                # already let go). Foul Play cannot pick a battle up midway, and an old state
                # must not be fed the same lines twice: start the battle over. The new one never
                # reaches a first request, so its decisions are errors, not guesses.
                if key in self._battles:
                    self._drop(key)
                self._gone.pop(key, None)
                self.counters["rejoined"] += 1
            if key in self._gone:
                self.counters["late_lines"] += 1
                return
            record = self._battles.get(key)
            if record is None:
                while len(self._battles) >= self.max_battles:
                    oldest = min(self._battles, key=lambda k: self._battles[k].last)
                    self.counters["evicted"] += 1
                    self._drop(oldest)
                record = self._battles[key] = _BattleRecord(self._least_loaded(), user, room)
            helper = self._helper_or_none(record.helper)
            if record.nbytes + len(data) > self.max_battle_bytes:
                # Far beyond any real battle. Keep following it, but it can no longer be rebuilt.
                if record.log:
                    self.counters["truncated"] += 1
                    record.log = []
            else:
                record.log.append(("feed", data))
                record.nbytes += len(data)
            record.fed += 1
            record.last = time.monotonic()
            self.counters["fed"] += 1
            if helper is not None:
                helper.post({"op": "feed", "key": key, "user": user, "room": room, "data": data})

    def note(self, key: str, choice: str) -> None:
        """What the player really sent for the request it was last asked about."""
        with self._lock:
            record = self._battles.get(key)
            if record is None or self._closed:
                return
            helper = self._helper_or_none(record.helper)
            if record.log:
                record.log.append(("chose", choice))
            self.counters["notes"] += 1
            if helper is not None:
                helper.post({"op": "chose", "key": key, "choice": choice})

    def _drop(self, key: str) -> None:
        record = self._battles.pop(key, None)
        self._gone[key] = None
        self._gone.move_to_end(key)
        while len(self._gone) > _GONE_KEPT:
            self._gone.popitem(last=False)
        if record is not None:
            helper = self._helpers[record.helper]
            if helper is not None and not helper.exited:
                helper.post({"op": "end", "key": key})

    def forget(self, key: str) -> None:
        """The battle is over: drop its stored lines here and its state in the helper."""
        with self._lock:
            if self._closed or (key not in self._battles and key in self._gone):
                return
            self.counters["forgotten"] += 1
            self._drop(key)

    def knows(self, key: str) -> bool:
        with self._lock:
            return key in self._battles

    # ---- decisions ---------------------------------------------------------------------------

    def advice_future(self, key: str, search_ms: Optional[int] = None, rqid: Any = None,
                      ttl: Optional[float] = None, *, payoff_ms: Optional[float] = None, payoff: Optional[str] = None,
                      iterations: Optional[int] = None, threads: Optional[int] = None,
                      parallelism: Optional[int] = None, workers: Optional[int] = None,
                      lines: Optional[Dict[str, Any]] = None) -> "concurrent.futures.Future":
        """
        Asks for Foul Play's decision in a battle. Returns at once with a future that becomes an
        Advice or raises FoulPlayBridgeError. The question is queued behind the lines fed so far.

        ttl: for how many seconds from now the answer is of use (advice() and advice_sync() pass
        their timeout). The helper is told, and when the question's turn comes it does not start
        a search that could not end in time: it answers "expired" at once (FoulPlayBridgeError
        with .expired). A caller that gives up tells nobody, so without this its question was
        still searched in full; once a helper was more than a timeout behind, every answer was
        late, each still cost a search, and nothing was answered in time any more. With it a
        helper that is asked more than it can answer keeps answering as many as it can, in time.
        Without ttl every question is searched, however long it waited.

        More than Foul Play itself does, all optional:
        payoff_ms   after the decision, this many ms more (all worlds together) for a table of
                    values per (our option, the opponent's option): advice.payoff, and after
                    fill_actions() advice.payoff_table on our 14 actions. payoff="engine" (the
                    default) is poke-engine's expectiminimax matrix, which leaves out the cells
                    its pruning did not need; payoff="full" computes every cell 2 plies deep
                    and needs about 0.5 s per world.
        iterations  every world is searched for exactly this many iterations, whatever the clock
                    says: the same strength on a loaded machine, at the price of the time.
        threads     poke-engine's own thread count for the search of one world.
        parallelism Foul Play's --search-parallelism for this decision: P times as many sampled
                    worlds (the opponent teams it could have), each searched for search_ms (half
                    early in a game), one after the other in the helper: about P times the time.
        workers     the worlds of this decision searched this many at a time, each in a process
                    of Foul Play's own pool forked from the helper (one core each): the same search
                    of every world in about 1/workers of the time (search_seconds).
        lines       {"k": 3, "turns": 5, "ms": 1500, "worlds": 2, "threads": 4}: after the decision, about ms
                    more for our k options the search favours most, each played forward `turns`
                    turns in the `worlds` likeliest sampled worlds (the search picks both sides'
                    actions at every step): advice.lines, and after fill_actions()
                    advice.lines_by_action on our 14 actions.

        (A question is NOT refused here, before it is queued, however far behind its helper is.
        That was tried: players that are told at once play their fallback move at once, their
        battles run ahead, and the lines of those battles alone then keep the helper busy. A
        caller that waits for its turn in the helper is what keeps the two in step.)
        """
        future: "concurrent.futures.Future" = concurrent.futures.Future()
        with self._lock:
            try:
                if self._closed:
                    raise FoulPlayBridgeError("the Foul Play bridge is closed")
                record = self._battles.get(key)
                if record is None:
                    raise FoulPlayBridgeError("no lines were fed for battle %s%s" % (key, " (it has ended)" if key in self._gone else ""))
                helper = self._helper(record.helper)
                record.last = time.monotonic()
                if self.fail_when_down and not self._up(helper, time.monotonic()):
                    # Starting (Foul Play loading, battles being rebuilt behind it), or frozen. The
                    # question would wait in its queue until the caller stops waiting: say so now.
                    self.counters["not_up"] += 1
                    error = FoulPlayBridgeError("Foul Play helper %d is not up (%s)" % (
                        record.helper, "starting" if helper.alive and not helper.replies else
                        ("not answering" if helper.alive else "gone")))
                    error.down = True
                    raise error
                message = {"op": "decide", "key": key, "ms": int(search_ms or self.search_ms)}
                if rqid is not None:
                    message["rqid"] = rqid
                now = time.monotonic()
                if payoff_ms:
                    message["payoff_ms"] = float(payoff_ms)
                    if payoff:
                        message["payoff"] = str(payoff)
                if iterations:
                    message["iterations"] = int(iterations)
                if threads:
                    message["threads"] = int(threads)
                worlds_x = max(1, int(parallelism or 1))
                if worlds_x > 1:
                    message["parallelism"] = worlds_x
                n_workers = max(1, int(workers or 1))
                if n_workers > 1:
                    message["workers"] = n_workers
                lines_ms = 0.0
                if lines:
                    message["lines"] = {"k": int(lines.get("k") or 3), "turns": int(lines.get("turns") or 5),
                                        "ms": float(lines.get("ms") or 1500.0), "worlds": int(lines.get("worlds") or 2),
                                        "threads": int(lines.get("threads") or 1)}
                    lines_ms = message["lines"]["ms"]
                if ttl is not None and helper.clock_ok:
                    # On the helper's clock. A remote helper's answer still has its way back to make:
                    # return_s (half the round trip and a margin) comes off (0 here and in WSL).
                    message["deadline"] = round(_clock() + helper.clock_offset + ttl - helper.return_s, 3)
                if self.early_expire and ttl is not None:
                    # The decisions waiting in this helper go first, each about what one takes
                    # there, and this one needs what the helper itself reckons for a search
                    # (search_seconds: 2.5 x ms + 20 ms, less with workers). When that cannot end
                    # within ttl the helper would only say "expired" when it got to the question,
                    # just before the caller stops waiting: say it now, and the caller plays at once.
                    ahead = helper.decisions_waiting()
                    each = self._decide_s[helper.index]
                    if each is None:
                        each = 2.3 * message["ms"] / 1000.0 + 0.04
                    need = (ahead * each + search_seconds(message["ms"], worlds_x, n_workers) + 2.5 * float(payoff_ms or 0.0) / 1000.0
                            + 1.25 * lines_ms / 1000.0
                            + helper.transit_s)    # a remote helper: there and back over the network
                    if ahead and need > ttl:
                        self.counters["expired"] += 1
                        self.counters["expired_early"] += 1
                        self.counters["decision_errors"] += 1
                        error = FoulPlayBridgeError("not asked: %d decisions wait in Foul Play helper %d, about %.2f s "
                                                    "with this one, %.2f s left" % (ahead, helper.index, need, ttl))
                        error.expired = True
                        future.set_exception(error)
                        return future
                helper.ask(_Pending(next(self._ids), message, future, "decide", key, now))
            except FoulPlayBridgeError as exc:
                self.counters["decision_errors"] += 1
                future.set_exception(exc)
        return future

    async def advice(self, key: str, search_ms: Optional[int] = None, rqid: Any = None,
                     timeout: Optional[float] = None, **more: Any) -> Advice:
        """Foul Play's decision. The event loop keeps running while the helper searches. more: see advice_future()."""
        limit = self.timeout if timeout is None else timeout
        future = self.advice_future(key, search_ms, rqid, ttl=limit, **more)   # queued now, before anything fed later
        try:
            return await asyncio.wait_for(asyncio.wrap_future(future), timeout=limit)
        except asyncio.TimeoutError:
            self.counters["timeouts"] += 1
            raise FoulPlayBridgeError("Foul Play did not answer for %s within %.1f s" % (key, limit)) from None

    def advice_sync(self, key: str, search_ms: Optional[int] = None, rqid: Any = None,
                    timeout: Optional[float] = None, **more: Any) -> Advice:
        limit = self.timeout if timeout is None else timeout
        try:
            return self.advice_future(key, search_ms, rqid, ttl=limit, **more).result(timeout=limit)
        except concurrent.futures.TimeoutError:
            self.counters["timeouts"] += 1
            raise FoulPlayBridgeError("Foul Play did not answer for %s within %.1f s" % (key, limit)) from None

    # ---- bookkeeping -------------------------------------------------------------------------

    def _ask_all(self, message: Dict[str, Any], timeout: float) -> List[Optional[Dict[str, Any]]]:
        futures: List[Optional["concurrent.futures.Future"]] = []
        with self._lock:
            for helper in self._helpers:
                if helper is None or helper.exited:
                    futures.append(None)
                    continue
                future: "concurrent.futures.Future" = concurrent.futures.Future()
                now = time.monotonic()
                helper.ask(_Pending(next(self._ids), dict(message), future, str(message.get("op")), None, now))
                futures.append(future)
        out: List[Optional[Dict[str, Any]]] = []
        deadline = time.monotonic() + timeout
        for future in futures:
            if future is None:
                out.append(None)          # no helper is running in this slot
                continue
            try:
                out.append(future.result(timeout=max(0.05, deadline - time.monotonic())))
            except Exception as exc:
                out.append({"ok": False, "error": str(exc) or type(exc).__name__})
        return out

    def sync(self, timeout: float = 30.0) -> bool:
        """True when every running helper has processed everything sent to it so far."""
        return all(reply is None or reply.get("ok") for reply in self._ask_all({"op": "sync"}, timeout))

    def helper_stats(self, timeout: float = 30.0) -> List[Optional[Dict[str, Any]]]:
        """What each helper says about itself (battles held, decisions, memory). None for a slot with no helper running."""
        return [reply if reply and reply.get("ok") else None for reply in self._ask_all({"op": "stats"}, timeout)]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            load = [0] * self._n
            stored = 0
            for record in self._battles.values():
                load[record.helper] += 1
                stored += record.nbytes
            helpers = []
            for i, helper in enumerate(self._helpers):
                each = self._decide_s[i]
                helpers.append({
                    "index": i, "alive": bool(helper and helper.alive), "pid": getattr(getattr(helper, "proc", None), "pid", None),
                    # Through WSL "pid" is the wsl.exe that carries the pipes; the helper itself is this PID inside WSL.
                    "helper_pid": helper.info.get("pid") if helper else None, "wsl": self._wsl,
                    "generation": self._generation[i], "battles": load[i],
                    "waiting": len(helper.pending) if helper else 0, "queued": helper.outbox.qsize() if helper else 0,
                    "decisions_waiting": helper.decisions_waiting() if helper else 0,
                    "down": time.monotonic() < self._down_until[i],
                    "deadlines": bool(helper and helper.clock_ok),
                    "decide_ms": None if each is None else round(each * 1000.0, 1),
                    "rtt_ms": round(1000.0 * helper.rtt_s, 1) if helper is not None and helper.rtt_s is not None else None,
                })
            out: Dict[str, Any] = dict(self.counters)
            with self._recent_lock:
                recent = list(self._recent)
            trips = [h["rtt_ms"] for h in helpers if h["rtt_ms"] is not None]
            out.update({"battles": len(self._battles), "stored_bytes": stored, "remembered_gone": len(self._gone),
                        "helpers": helpers, "search_ms": self.search_ms, "where": self.where, "remote_host": self.remote_host,
                        "rtt_ms": round(_median(trips), 1) if trips else None,
                        "rtt_start_ms": (self.rtt_start or {}).get("rtt_ms"),
                        # How hard Foul Play searched: over every decision answered, and over the last
                        # _RECENT_DECISIONS of them (what a run should be judged by: see teacher_problems).
                        "iterations_per_decision": self.counters["iterations"] // max(1, self.counters["decisions"]),
                        "recent_decisions": len(recent),
                        "recent_iterations_per_decision": (sum(recent) // len(recent)) if recent else 0})
            return out


_BRIDGES: Set[FoulPlayBridge] = set()


_exiting = False   # set at interpreter exit: from then on nothing is started


def _close_bridges() -> None:
    """At interpreter exit: every helper is stopped, and no bridge or helper is started any more."""
    global _exiting
    _exiting = True
    for bridge in list(_BRIDGES):
        try:
            bridge.close()
        except Exception:
            pass
    _kill_everything()


atexit.register(_close_bridges)

_shared: Optional[FoulPlayBridge] = None
_shared_lock = threading.Lock()


def shared_bridge(helpers: Optional[int] = None, search_ms: Optional[int] = None, **kwargs: Any) -> FoulPlayBridge:
    """
    The process-wide bridge, made on first use. The first call decides the settings: the arguments,
    or FP_BRIDGE_HELPERS (default 1) and FP_BRIDGE_SEARCH_MS (default 100). A later call with other
    numbers gets the bridge that exists and a line in the log.
    """
    global _shared
    with _shared_lock:
        if _shared is None or _shared._closed:
            if _exiting:
                raise FoulPlayBridgeError("the interpreter is exiting: no Foul Play bridge is made now")
            n = int(helpers or os.environ.get("FP_BRIDGE_HELPERS") or 1)
            ms = int(search_ms or os.environ.get("FP_BRIDGE_SEARCH_MS") or DEFAULT_SEARCH_MS)
            _shared = FoulPlayBridge(helpers=n, search_ms=ms, **kwargs)
        elif helpers and int(helpers) != _shared.helpers:
            logger.warning("the shared Foul Play bridge already runs %d helpers, not %s", _shared.helpers, helpers)
        return _shared


def close_shared() -> None:
    global _shared
    with _shared_lock:
        bridge, _shared = _shared, None
    if bridge is not None:
        bridge.close()


# =============================================================================================
# Is a run still what it says it is
# =============================================================================================

def teacher_problems(asked: int, unanswered: int, decisions: int = 0, iterations_per_decision: float = 0.0,
                     min_iterations: int = MIN_ITERATIONS) -> List[str]:
    """
    Why a run that learns from Foul Play, or trains against it, should stop. [] while it is fine.

    asked         decisions Foul Play was asked for (the sum of the players' foulplay_counts["asked"])
    unanswered    of those, the ones another bot had to move for (no answer in time, a helper down,
                  a battle Foul Play could not follow, Revival Blessing). Such a step must not be
                  recorded as Foul Play's label; beyond max(FALLBACK_FLOOR, FALLBACK_SHARE of asked)
                  the games themselves are no longer Foul Play's.
    decisions, iterations_per_decision
                  stats()["recent_decisions"] and ["recent_iterations_per_decision"]. Foul Play's
                  search is limited by TIME: helpers that do not get a core each search less and
                  advise worse, with no error anywhere. Below min_iterations the search is weaker
                  than anything measured as good enough (bench/FOULPLAY_BRIDGE.md). 0 switches
                  this check off; it waits for MIN_ITERATIONS_SAMPLE decisions.
    """
    problems: List[str] = []
    allowed = max(float(FALLBACK_FLOOR), FALLBACK_SHARE * asked)
    if unanswered > allowed:
        problems.append(
            "Foul Play gave no usable answer for %d of %d decisions (%.1f%%, more than the %d allowed): another bot moved "
            "instead. Look for helpers that are down, behind (too many battles per helper) or starved of CPU." % (
                unanswered, asked, 100.0 * unanswered / max(1, asked), int(allowed)))
    if min_iterations and decisions >= MIN_ITERATIONS_SAMPLE and iterations_per_decision < min_iterations:
        problems.append(
            "Foul Play searched only %d iterations per decision over its last %d decisions, under the floor of %d: the helpers "
            "do not get the CPU they need, so its advice is weaker than what was measured. Raise foulplay_search_ms, or run "
            "fewer helpers and workers than there are cores (python foulplay_bridge.py calibrate measures it)." % (
                int(iterations_per_decision), decisions, min_iterations))
    return problems


# =============================================================================================
# Player mixin
# =============================================================================================

class FoulPlayMixin:
    """
    Gives any poke-env Player access to Foul Play's opinion on its own battles.

    Put it before the Player base class and switch it on:

        class MyPlayer(FoulPlayMixin, Player):
            foulplay_enabled = True

    It is off by default, unlike PSBotMixin: a player class that merely has the mixin must not
    start helper processes. Set the flag before Player.__init__ runs (in the class body, or in
    __init__ before calling super), because the player starts listening inside it. While it is
    off nothing is forwarded and no helper is started.

    Whoever plays a move that is not the one foulplay_decision() returned has to call
    foulplay_note(battle, order) with the order really sent: Foul Play draws conclusions about the
    opponent from the move it believes it made.

    A player that LEARNS from Foul Play asks foulplay_teacher(): it says, with an action index
    or None, whether there is a label on this request. A move the player makes because Foul Play
    had no answer is another bot's move and must not be recorded as Foul Play's.
    """

    foulplay_enabled: bool = False
    foulplay_search_ms: Optional[int] = None     # None: the bridge's own setting
    foulplay_helpers: Optional[int] = None       # only used when this player makes the shared bridge
    foulplay_timeout: Optional[float] = None     # None: the bridge's own setting
    foulplay_pick: str = "choice"                # "choice": the move Foul Play drew. "best": the largest share
    _foulplay_bridge: Optional[FoulPlayBridge] = None

    @property
    def foulplay(self) -> FoulPlayBridge:
        bridge = self._foulplay_bridge
        if bridge is None or (bridge._closed and self.__dict__.get("_foulplay_shared", False)):
            # None yet, or the process-wide one was closed (close_shared) and there is a new one.
            # A bridge that was GIVEN to this player is kept also when it is closed: its calls then
            # do nothing or fail, they do not quietly start another pool of helpers.
            bridge = self._foulplay_bridge = shared_bridge(self.foulplay_helpers, self.foulplay_search_ms)
            self.__dict__["_foulplay_shared"] = True
        return bridge

    @foulplay.setter
    def foulplay(self, bridge: Optional[FoulPlayBridge]) -> None:
        self._foulplay_bridge = bridge
        self.__dict__["_foulplay_shared"] = False

    @property
    def foulplay_counts(self) -> Dict[str, int]:
        counts = self.__dict__.get("_foulplay_counts")
        if counts is None:
            counts = self.__dict__["_foulplay_counts"] = {"asked": 0, "answered": 0, "errors": 0, "illegal": 0,
                                                          "tera_dropped": 0, "revive": 0}
            self.__dict__["foulplay_problems"] = []   # the last few errors and illegal answers, for the log
        return counts

    def _foulplay_problem(self, text: str) -> None:
        self.foulplay_counts
        problems = self.__dict__["foulplay_problems"]
        problems.append(text[:400])
        del problems[:-20]

    def _foulplay_key(self, battle_tag: str) -> str:
        # Both sides of a self-play game can live in one process, so the tag alone is not unique.
        return "%s|%s" % (self.username, battle_tag)   # type: ignore[attr-defined]

    async def _handle_battle_message(self, split_messages: List[List[str]]):
        # Forward first: poke-env calls choose_move from inside this handler, and the helper has
        # to have been sent the same lines by then.
        if self.foulplay_enabled and split_messages and split_messages[0] and split_messages[0][0].startswith(">"):
            room = split_messages[0][0][1:].strip()
            if room.startswith("battle-"):
                data = "\n".join("|".join(parts) for parts in split_messages[1:])
                if data:
                    try:
                        self.foulplay.feed(self._foulplay_key(room), self.username, room, data)   # type: ignore[attr-defined]
                    except Exception as exc:   # not installed, for one: keep playing, say so once a minute
                        now = time.monotonic()
                        if not _exiting and now - self.__dict__.get("_foulplay_warned", -1e9) > 60.0:
                            self.__dict__["_foulplay_warned"] = now
                            logger.warning("Foul Play bridge unavailable: %s", exc)
        await super()._handle_battle_message(split_messages)   # type: ignore[misc]

    def _battle_finished_callback(self, battle: Any):
        if self.foulplay_enabled:
            try:
                self.foulplay.forget(self._foulplay_key(battle.battle_tag))
            except Exception:
                pass
        super()._battle_finished_callback(battle)   # type: ignore[misc]

    def _foulplay_fill(self, battle: Any, advice: Advice) -> Advice:
        """
        The statistics on our action indices. A player with an ObservationAssembler (worker.py's)
        gets the assembler's own mask and order-to-index function, so the numbers sit exactly
        where its trajectories put the teacher's action. Any other player gets the copies here.
        """
        index_of, mask = self._foulplay_index_tools(battle)
        return fill_actions(self, battle, advice, index_of=index_of, mask=mask)

    def _foulplay_index_tools(self, battle: Any) -> Tuple[Optional[Callable[[Any, Any], int]], Optional[List[int]]]:
        assembler = getattr(self, "assembler", None)
        index_of = getattr(assembler, "map_order_to_index", None)
        mask = None
        make_mask = getattr(assembler, "create_action_mask", None)
        if make_mask is not None:
            mask = [int(x) for x in make_mask(battle, list(battle.team.values()))]
        return index_of, mask

    async def foulplay_advice(self, battle: Any, search_ms: Optional[int] = None, **more: Any) -> Advice:
        """
        Foul Play's choice and its search statistics for the current request, on our action indices,
        with the opponent's side of the search (advice.opponent). more: payoff_ms, payoff, iterations,
        threads, as FoulPlayBridge.advice_future() takes them.
        """
        request = getattr(battle, "last_request", None) or {}
        advice = await self.foulplay.advice(self._foulplay_key(battle.battle_tag),
                                            search_ms=search_ms or self.foulplay_search_ms,
                                            rqid=request.get("rqid"), timeout=self.foulplay_timeout, **more)
        return self._foulplay_fill(battle, advice)

    def foulplay_note(self, battle: Any, order: Any) -> None:
        """Tells Foul Play which order was really sent, so that what it infers from it stays right."""
        if not self.foulplay_enabled:
            return
        try:
            self.foulplay.note(self._foulplay_key(battle.battle_tag), order_to_choice(battle, order))
        except Exception:
            pass

    async def foulplay_decision(self, battle: Any, search_ms: Optional[int] = None,
                                pick: Optional[str] = None) -> Tuple[Optional[Any], Optional[Advice]]:
        """
        (order, advice): Foul Play's move as a poke-env order, already noted as played, and the
        advice it came from. The order is None when the bridge failed, when the move is not legal
        for poke-env, or when the request asks whom Revival Blessing revives (Foul Play has no
        answer to that); the advice is None when there was no answer.
        """
        counts = self.foulplay_counts
        counts["asked"] += 1
        try:
            advice = await self.foulplay_advice(battle, search_ms)
        except asyncio.CancelledError:
            raise
        except Exception as exc:   # helper down, timeout, a state Foul Play could not follow
            if getattr(exc, "revive", False):
                # Not a failure of the bridge: Foul Play has no answer to "whom to revive".
                counts["revive"] += 1
                return None, None
            counts["errors"] += 1
            self._foulplay_problem("%s turn %s: %s" % (battle.battle_tag, getattr(battle, "turn", "?"), exc))
            now = time.monotonic()
            if now - self.__dict__.get("_foulplay_logged", -1e9) > 1.0:   # a bridge that is down fails every decision
                self.__dict__["_foulplay_logged"] = now
                logger.warning("Foul Play advice failed in %s: %s (%d failures of this player so far)",
                               battle.battle_tag, exc, counts["errors"])
            return None, None
        counts["answered"] += 1
        if advice.tera_dropped:
            counts["tera_dropped"] += 1
        choice = advice.best if (pick or self.foulplay_pick) == "best" else advice.choice
        order = choice_to_order(self, battle, choice) if choice else None
        if order is None:
            counts["illegal"] += 1
            self._foulplay_problem("%s turn %s: Foul Play sent %r (%r), not a legal order here" % (
                battle.battle_tag, getattr(battle, "turn", "?"), advice.fp_message, choice))
            return None, advice
        self.foulplay_note(battle, order)
        return order, advice

    async def foulplay_order(self, battle: Any, search_ms: Optional[int] = None) -> Optional[Any]:
        """Foul Play's move as a poke-env order, or None when there is no legal answer."""
        order, _advice = await self.foulplay_decision(battle, search_ms)
        return order

    async def foulplay_teacher(self, battle: Any, search_ms: Optional[int] = None,
                               pick: Optional[str] = None) -> Tuple[Optional[Any], Optional[int], Optional[Advice]]:
        """
        For a player that LEARNS from Foul Play: (order, action, advice).

        `action` is the index among our 14 actions that plays exactly `order`, Foul Play's move,
        and it is the only thing that may be recorded as Foul Play's label for this step.

        (None, None, advice or None) means there is no label on this request: no answer in time, a
        helper down, a battle Foul Play's code could not follow, an answer poke-env does not hold
        legal, a Revival Blessing request, or a move our 14 actions cannot say. The caller then
        plays something of its own, says so with foulplay_note(), counts it, and does NOT record
        that step as a teacher's step: it is another bot's move, and a student must not be taught
        that it is Foul Play's. teacher_problems() says when there are too many of them.
        """
        order, advice = await self.foulplay_decision(battle, search_ms, pick)
        if order is None or advice is None:
            return None, None, advice
        choice = advice.best if (pick or self.foulplay_pick) == "best" else advice.choice
        index_of, mask = self._foulplay_index_tools(battle)
        action = choice_to_index(self, battle, choice, index_of, mask)
        if action is None:
            self._foulplay_problem("%s turn %s: our 14 actions cannot say Foul Play's %r (%r)" % (
                battle.battle_tag, getattr(battle, "turn", "?"), advice.fp_message, choice))
            return None, None, advice
        return order, action, advice


# =============================================================================================
# Measuring the search on this machine
# =============================================================================================

DEFAULT_TRACE = REPO_ROOT / "tests" / "fixtures" / "foulplay_bridge_trace.jsonl.gz"


def load_trace(path: Optional[str] = None) -> List[List[Dict[str, Any]]]:
    """
    The recorded battles of a trace file (default: the test fixture, seven real battles), each a
    list of events: {"t": "msg", "room", "data"} as the player received them, {"t": "order",
    "rqid", "choice"} where it moved, and the player's "username" in the last event.
    """
    battles: List[List[Dict[str, Any]]] = []
    current: Optional[List[Dict[str, Any]]] = None
    with gzip.open(str(path or DEFAULT_TRACE), "rt", encoding="utf-8") as fh:
        for line in fh:
            event = json.loads(line)
            if event["t"] == "msg" and event["data"].startswith("|init|"):
                current = []
                battles.append(current)
            if current is not None:
                current.append(event)
    return battles


def calibrate(ms_list: Tuple[int, ...] = (20, 35, 50, 75, 100), helpers: int = 1, target: int = TARGET_ITERATIONS,
              trace: Optional[str] = None, battles: Optional[int] = None, bridge: Optional[FoulPlayBridge] = None,
              **bridge_kwargs: Any) -> Dict[str, Any]:
    """
    How hard does Foul Play search on THIS machine, as loaded as it is right now?

    Foul Play's strength comes from the iterations of its search, and its setting is a time. The
    same milliseconds are fewer iterations on slower cores and on cores that are shared, with no
    error anywhere. So the search time is chosen by iterations, where and how the run will happen:
    every decision point of the recorded battles is asked at each search time in ms_list, with
    `helpers` helpers working at the same time (give the number the run will have, and run it
    while the rest of the run's load is there).

    Returns {"rows": [{"ms", "decisions", "iterations_per_decision", "decision_ms",
    "decisions_per_second"}], "helpers", "target", "search_ms", "problems"}: search_ms is the
    shortest search time that reaches `target` iterations per decision, on the straight line
    between the two measured times around it, or None when the longest time measured does not.
    """
    ms_list = tuple(sorted({int(ms) for ms in ms_list}))
    if not ms_list:
        raise ValueError("no search time to measure")
    recorded = load_trace(trace)
    if battles:
        recorded = recorded[:battles]
    if not recorded:
        raise FoulPlayBridgeError("the trace %s holds no battle" % (trace or DEFAULT_TRACE))
    own = bridge is None
    if bridge is None:
        bridge = FoulPlayBridge(helpers=helpers, **bridge_kwargs)
    totals = {ms: {"decisions": 0, "iterations": 0, "helper_ms": 0.0} for ms in ms_list}
    lock = threading.Lock()
    problems: List[str] = []

    def stream(n: int) -> None:
        # One battle at a time for every helper, each starting with another one of the battles.
        for b in range(len(recorded)):
            events = recorded[(b + n) % len(recorded)]
            user = str(events[-1].get("username") or "calibrate")
            room = str(next(e["room"] for e in events if e["t"] == "msg"))
            key = "%s|%s#calibrate%d" % (user, room, n)
            try:
                for event in events:
                    if event["t"] == "msg":
                        bridge.feed(key, user, room, event["data"])
                    elif event["t"] == "order":
                        for ms in ms_list:
                            try:
                                advice = bridge.advice_future(key, search_ms=ms, rqid=event.get("rqid")).result(timeout=300.0)
                            except Exception as exc:
                                if not getattr(exc, "revive", False):     # the one request Foul Play has no answer to
                                    with lock:
                                        problems.append(str(exc) or type(exc).__name__)
                                break
                            with lock:
                                row = totals[ms]
                                row["decisions"] += 1
                                row["iterations"] += advice.iterations
                                row["helper_ms"] += advice.ms
                        bridge.note(key, str(event.get("choice") or "default"))
            finally:
                bridge.forget(key)

    try:
        bridge.start()
        threads = [threading.Thread(target=stream, args=(n,), name="fp-calibrate-%d" % n, daemon=True) for n in range(bridge.helpers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        if own:
            bridge.close()

    rows = []
    for ms in ms_list:
        row = totals[ms]
        n = int(row["decisions"])
        each = row["helper_ms"] / n if n else 0.0
        rows.append({"ms": ms, "decisions": n, "iterations_per_decision": int(row["iterations"] // n) if n else 0,
                     "decision_ms": round(each, 1), "decisions_per_second": round(1000.0 / each, 2) if each else 0.0})
    search_ms: Optional[int] = None
    for i, row in enumerate(rows):
        if row["decisions"] and row["iterations_per_decision"] >= target:
            search_ms = row["ms"]
            if i > 0 and rows[i - 1]["decisions"] and row["iterations_per_decision"] > rows[i - 1]["iterations_per_decision"]:
                low = rows[i - 1]
                share = (target - low["iterations_per_decision"]) / float(row["iterations_per_decision"] - low["iterations_per_decision"])
                search_ms = int(low["ms"] + max(0.0, min(1.0, share)) * (row["ms"] - low["ms"]) + 0.999)
            break
    return {"rows": rows, "helpers": bridge.helpers, "target": int(target), "search_ms": search_ms, "problems": problems[:20]}


def _remote_check(target: RemoteTarget, pings: int = 10) -> int:
    """
    `foulplay_bridge.py remote-check`: the network's round trip through ssh (an echo, nothing else
    runs there), then one helper started there, probed `pings` times and stopped, and nothing of
    it left running there. Exit code 0 when all of that went, with "rtt_ms=N" (the helper's round
    trip, the one ladder.py accounts for) as the last line.
    """
    echo = remote_echo_round_trip(target, pings)
    if not echo.get("ok"):
        print("the round trip through ssh to %s could not be measured: %s" % (target.host, echo.get("error") or "?"))
        return 1
    print("network round trip through ssh to %s: %.0f ms (fastest %.0f, 90th %.0f, slowest %.0f; %d lines)%s" % (
        target.host, echo["rtt_ms"], echo["min_ms"], echo["p90_ms"], echo["max_ms"], echo["pings"],
        "; two lines back to back: %.0f ms" % echo["pair_ms"] if "pair_ms" in echo else ""))
    code = 0
    trip: Dict[str, Any] = {}
    bridge = FoulPlayBridge(helpers=1, remote=target, timeout=10.0)
    try:
        began = time.monotonic()
        bridge.start(timeout=120.0)
        loaded = time.monotonic() - began
        trip = bridge.measure_round_trip(rounds=max(1, int(pings)))
        info = bridge._helpers[0].info if bridge._helpers[0] is not None else {}
        if not trip.get("probes"):
            print("one helper started there but did not answer its probes")
            code = 1
        else:
            print("one helper started there in %.1f s (pid %s there, Foul Play %s, poke-engine %s, python %s) and probed %d times: "
                  "round trip %.0f ms (fastest %.0f, 90th %.0f, slowest %.0f)" % (
                      loaded, info.get("pid"), str(info.get("foulplay_commit") or "?")[:12], info.get("poke_engine"), info.get("python"),
                      trip["probes"], trip["rtt_ms"], trip["min_ms"], trip["p90_ms"], trip["max_ms"]))
    except Exception as exc:
        print("the helper did not start there: %s" % exc)
        code = 1
    finally:
        bridge.close()
    left = remote_leftovers(target)
    if left is None:
        print("could not ask %s whether anything of this check is left running" % target.host)
        code = code or 1
    elif left:
        print("NOT STOPPED on %s: %d process(es) of this check (pid %s)" % (target.host, len(left), " ".join(map(str, left))))
        code = 1
    else:
        print("stopped: nothing of this check is left running on %s" % target.host)
    if code == 0:
        print("rtt_ms=%d" % int(round(trip["rtt_ms"])))
    return code


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Foul Play bridge: is it there, and how hard does it search here.")
    where = argparse.ArgumentParser(add_help=False)
    where.add_argument("--remote", default=None,
                       help="the helpers on another machine over ssh: user@host, or vm (the box cloud/vm.sh talks to, looked up now)")
    where.add_argument("--key", default=None, help="with --remote user@host: the private key (ssh -i)")
    where.add_argument("--ssh", default=None, help="with --remote: the ssh program (default: Git's ssh on Windows, else ssh)")
    where.add_argument("--known-hosts", default=None, help="with --remote: ssh's known-hosts file")
    where.add_argument("--remote-dir", default=REMOTE_CODE_DIR, help="with --remote: that machine's copy of this repository")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("available", parents=[where], help="say whether Foul Play can be used from here (or on --remote)")
    cal = sub.add_parser("calibrate", parents=[where],
                         help="measure iterations per decision at several search times, on this machine (or --remote) as loaded as it is")
    cal.add_argument("--ms", default="20,35,50,75,100", help="search times to measure, comma separated")
    cal.add_argument("--helpers", type=int, default=1, help="helpers working at the same time: the number the run will have")
    cal.add_argument("--target", type=int, default=TARGET_ITERATIONS, help="iterations per decision to reach")
    cal.add_argument("--trace", default=None, help="recorded battles (default: the test fixture)")
    cal.add_argument("--battles", type=int, default=None, help="use only the first N recorded battles")
    chk = sub.add_parser("remote-check", parents=[where],
                         help="--remote: Foul Play and the helper there, the network's round trip through ssh, one helper started, "
                              "probed and stopped, and nothing of it left running there")
    chk.add_argument("--pings", type=int, default=10, help="echo lines through ssh, and probes of the helper")
    args = parser.parse_args(argv)

    target: Optional[RemoteTarget] = None
    if args.remote:
        try:
            target = remote_target(args.remote, key=args.key, ssh=args.ssh, known_hosts=args.known_hosts, code_dir=args.remote_dir)
        except FoulPlayBridgeError as exc:
            print(exc)
            return 1
    elif args.command == "remote-check":
        print("remote-check needs --remote")
        return 2
    ok, text = available(remote=target)
    print(text)
    if args.command == "available" or not ok:
        return 0 if ok else 1
    if args.command == "remote-check":
        return _remote_check(target, args.pings)
    result = calibrate(tuple(int(x) for x in args.ms.split(",") if x.strip()), helpers=args.helpers, target=args.target,
                       trace=args.trace, battles=args.battles, remote=target)
    print("%d helpers at once, every decision point of the recorded battles at each search time:" % result["helpers"])
    print("  search_ms  decisions  iterations/decision  ms/decision  decisions/s per helper")
    for row in result["rows"]:
        print("  %9d  %9d  %19d  %11.1f  %22.2f" % (row["ms"], row["decisions"], row["iterations_per_decision"],
                                                      row["decision_ms"], row["decisions_per_second"]))
    for problem in result["problems"]:
        print("  problem: %s" % problem)
    if result["search_ms"] is None:
        print("No search time measured reaches %d iterations per decision: measure longer ones (--ms)." % result["target"])
        return 2
    print("foulplay_search_ms = %d reaches %d iterations per decision here (a run that falls under %d should stop)." % (
        result["search_ms"], result["target"], MIN_ITERATIONS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
