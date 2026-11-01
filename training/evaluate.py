#!/usr/bin/env python
"""
Measures whether the policy is getting stronger.

One evaluation plays the policy stored in a checkpoint against fixed opponents, on a Showdown
server of its own, appends one line per opponent to RUN/eval/results.jsonl and rewrites
RUN/eval/REPORT.md:

    python evaluate.py --run-dir RUN --checkpoint newest --opponents random,maxpower,simple,psbot \\
                       --games 200 --port 9000 --server-dir /home/ubuntu/pokemon-showdown

Watch mode does that for a whole training run and never stops:

    python evaluate.py --run-dir RUN --watch --config RUN_CONFIG.json

It evaluates the newest checkpoint when the last evaluation is `every_hours` old, once more when
RUN/state.json changes phase, and a last time when the phase is "done", after which it writes
RUN/eval/final.done. Each evaluation is a separate process, so a crash or a hang in one costs that
evaluation and nothing else. What is already in results.jsonl is never played again.

What is measured. The policy plays alone (the transformer, no PSBot guard) and takes its best
legal action each turn, as `play.py --agent policy` does; --sample draws the action from its
probabilities instead. The choice is written in every line. Opponents:

    random     poke-env RandomPlayer                      the floor: a working policy beats it
    maxpower   poke-env MaxBasePowerPlayer
    simple     poke-env SimpleHeuristicsPlayer, stat fix  PSBot beats it in about 91% of games
    psbot      the hand-written bot the policy imitates   50% means level with the teacher
    foulplay   pmariglia's search bot (bench/foulplay.py) skipped with a note when not installed

A line of results.jsonl:

    time, checkpoint, update, phase, total_steps, opponent, games, wins, losses, ties, errors,
    winrate, ci_low, ci_high, seconds, sample, and more (see result_line)

`games` are the games that ran to their natural end, and wins + losses + ties == games. A game
that did not (never finished, forfeited, timed out) is in `errors` and is not a win or a loss.
`winrate` counts a tie as half a win; the interval is the 95% Wilson interval. A line with
games == 0 has winrate 0.0, the interval 0 to 1 and "valid": false. Such a line is also what an
opponent that could not be played leaves behind, with the reason in "notes": Foul Play where it
is not installed, or every opponent when the checkpoint's tensors do not fit the network.

The report compares checkpoints in order of training (phase, then decisions), not in the order
they were measured, and says "better" or "worse" only at p < 0.01 (VERDICT_P).

Files, all under RUN/eval/:  results.jsonl, REPORT.md, final.done, in_progress.json, watch.json,
hybrid_results.jsonl (only with --also-hybrid), logs/.

Exit codes of one evaluation: 0 lines were written (errors, if any, are in them), 2 bad
arguments, 3 no usable checkpoint (missing, still being written, unreadable, or replaced while it
was being evaluated), 4 no server on the port and none could be started, 5 another watcher is
running on this run directory, 1 anything else.

The training servers are never touched: only --port is used, a server is started there only when
nothing is listening and --server-dir is given, and only a server started here is stopped here.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import math
import os
import re
import secrets
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BUILTIN_OPPONENTS: Tuple[str, ...] = ("random", "maxpower", "simple", "psbot")
OPPONENTS: Tuple[str, ...] = BUILTIN_OPPONENTS + ("foulplay",)
FOULPLAY_MODULE = "bench.foulplay"
# --foulplay-iterations must be a multiple of this: poke-engine searches in batches of 1,000
# iterations (bench/foulplay.py ENGINE_BATCH; this file imports nothing of that module up front).
FOULPLAY_ITERATION_BATCH = 1000
CKPT_RE = re.compile(r"^learner_update_(\d+)\.pt$")
PHASE_ORDER: Tuple[str, ...] = ("imitation", "warmup", "ppo")
# "Better" and "worse" are said below this two-sided p. At 0.05, with two comparisons per opponent
# and four opponents, a policy that never changes got a verdict in about one report in three.
VERDICT_P = 0.01
# How many evaluations are taken together on each side of the trend comparison.
TREND_POOL = 3

EXIT_OK, EXIT_CRASH, EXIT_USAGE, EXIT_NO_CHECKPOINT, EXIT_NO_SERVER, EXIT_LOCKED = 0, 1, 2, 3, 4, 5

CORE_KEYS: Tuple[str, ...] = (
    "time", "checkpoint", "update", "phase", "total_steps", "opponent", "games", "wins", "losses",
    "ties", "errors", "winrate", "ci_low", "ci_high", "seconds", "sample",
)


# =============================================================================================
# Small helpers
# =============================================================================================

def ascii_text(text: Any) -> str:
    """The console here can be cp1252 and a log can be anything: keep our own output ASCII."""
    return str(text).encode("ascii", "replace").decode("ascii")


def say(*parts: Any) -> None:
    print(ascii_text(" ".join(str(p) for p in parts)), flush=True)


def utc_iso(ts: Optional[float] = None) -> str:
    when = time.time() if ts is None else ts
    return datetime.datetime.fromtimestamp(when, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(text: Any) -> Optional[float]:
    """Epoch seconds of an ISO 8601 time, or None."""
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.strip()
    if t.endswith(("Z", "z")):
        t = t[:-1] + "+00:00"
    try:
        d = datetime.datetime.fromisoformat(t)
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=datetime.timezone.utc)
    return d.timestamp()


def duration(seconds: float) -> str:
    """45 s, 12 min, 2.5 h."""
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60.0:.0f} min"
    return f"{seconds / 3600.0:.1f} h"


def read_json(path: Any) -> Optional[Any]:
    """The parsed file, or None when it is missing, half-written or not JSON."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_jsonl(path: Any) -> List[Dict[str, Any]]:
    """Every line that parses to an object. A broken line (a write cut short) is skipped."""
    out: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
    except OSError:
        pass
    return out


def append_jsonl(path: Any, record: Dict[str, Any]) -> None:
    """One line, written with a single write call so that two writers cannot interleave."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = json.dumps(record, separators=(", ", ": "), default=str) + "\n"
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(data)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def write_text_atomic(path: Any, text: str) -> None:
    """Replaces a file this program generates (REPORT.md, final.done, the markers) in one step."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows: someone has the file open for a moment
            if attempt == 4:
                raise
            time.sleep(0.2)


def write_json_atomic(path: Any, obj: Any) -> None:
    write_text_atomic(path, json.dumps(obj, indent=1, default=str) + "\n")


# =============================================================================================
# Statistics
# =============================================================================================

def wilson(successes: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for `successes` out of `n` (successes may be x.5: ties are half)."""
    if n <= 0:
        return 0.0, 1.0
    p = min(1.0, max(0.0, successes / n))
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def score(wins: int, ties: int, games: int) -> float:
    """Win rate with a tie as half a win. 0.0 when no game was played."""
    return (wins + 0.5 * ties) / games if games > 0 else 0.0


def two_proportion_p(p1: float, n1: int, p2: float, n2: int) -> float:
    """Two-sided p-value of the pooled two-proportion z-test. 1.0 when there is nothing to test."""
    if n1 <= 0 or n2 <= 0:
        return 1.0
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(max(0.0, pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2)))
    if se <= 0.0:
        return 1.0 if p1 == p2 else 0.0
    return math.erfc(abs(p2 - p1) / se / math.sqrt(2.0))


def noise_points(n: int, p: float = 0.5) -> float:
    """Half-width, in percentage points, of the 95% interval of one win rate from n games."""
    return 100.0 * 1.96 * math.sqrt(p * (1.0 - p) / n) if n > 0 else 100.0


def clear_points(n: int, p: Optional[float] = None) -> float:
    """
    How far apart, in percentage points, two win rates near 50% from n games each must be for the
    two-proportion test to give `p` (VERDICT_P when not given): the smallest difference compare()
    calls better or worse. Further from 50% a little less is enough.
    """
    if n <= 0:
        return 100.0
    z = statistics.NormalDist().inv_cdf(1.0 - (VERDICT_P if p is None else p) / 2.0)
    return 100.0 * z * math.sqrt(0.5 / n)


# =============================================================================================
# The run directory
# =============================================================================================

class RunPaths:
    def __init__(self, run_dir: Any):
        self.run = os.path.abspath(str(run_dir))
        self.checkpoints = os.path.join(self.run, "checkpoints")
        self.state = os.path.join(self.run, "state.json")
        self.metrics = os.path.join(self.run, "metrics.jsonl")
        self.eval = os.path.join(self.run, "eval")
        self.results = os.path.join(self.eval, "results.jsonl")
        self.hybrid_results = os.path.join(self.eval, "hybrid_results.jsonl")
        self.report = os.path.join(self.eval, "REPORT.md")
        self.final_done = os.path.join(self.eval, "final.done")
        self.in_progress = os.path.join(self.eval, "in_progress.json")
        self.watch_state = os.path.join(self.eval, "watch.json")
        self.logs = os.path.join(self.eval, "logs")


def state_phase(state: Any) -> Optional[str]:
    """The phase in state.json, or None when there is no usable state."""
    if isinstance(state, dict) and isinstance(state.get("phase"), str) and state["phase"].strip():
        return state["phase"].strip()
    return None


# =============================================================================================
# Checkpoints
# =============================================================================================

@dataclass(frozen=True)
class Ckpt:
    path: str
    name: str
    mtime: float
    size: int

    @property
    def key(self) -> Tuple[str, float]:
        """Identifies this version of the file. A later phase reuses the names of an earlier one."""
        return self.name, round(self.mtime, 3)


def stat_ckpt(path: Any) -> Optional[Ckpt]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return Ckpt(os.path.abspath(str(path)), os.path.basename(str(path)), float(st.st_mtime), int(st.st_size))


def list_checkpoints(ckpt_dir: Any) -> List[Ckpt]:
    """learner_update_NNNNNN.pt files, newest first by modification time."""
    out: List[Ckpt] = []
    try:
        names = os.listdir(ckpt_dir)
    except OSError:
        return out
    for name in names:
        if not CKPT_RE.match(name):
            continue
        c = stat_ckpt(os.path.join(str(ckpt_dir), name))
        if c is not None and os.path.isfile(c.path):
            out.append(c)
    out.sort(key=lambda c: (c.mtime, c.name), reverse=True)
    return out


def is_settled(ckpt: Ckpt, settle_s: float = 2.0, sleep: Callable[[float], None] = time.sleep) -> bool:
    """True when the file is not empty and neither its size nor its time moves over settle_s."""
    if ckpt.size <= 0:
        return False
    if settle_s > 0:
        sleep(settle_s)
    again = stat_ckpt(ckpt.path)
    return again is not None and again.size == ckpt.size and again.mtime == ckpt.mtime


def newest_checkpoint(
    ckpt_dir: Any,
    *,
    settle_s: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    loader: Optional[Callable[[str], Any]] = None,
    fall_back: bool = True,
    skip: Iterable[Tuple[str, float]] = (),
    notes: Optional[List[str]] = None,
) -> Optional[Ckpt]:
    """
    The newest checkpoint that can be used, or None.

    A file is used only when it has stopped changing (is_settled) and, if `loader` is given,
    when loader(path) does not raise (torch.load, for a file copied in from elsewhere: train.py
    itself writes to a .tmp name and renames, so its files appear complete).

    fall_back=True   a file that fails is skipped and the next older one is tried (one evaluation
                     asked for by hand: better an older checkpoint than none).
    fall_back=False  a file that fails means "not now": None, and the caller comes back later
                     (the watch loop: the newest one is the one that matters).
    `skip` holds Ckpt.key values to pass over (versions already given up on).
    """
    skip = set(skip)
    for c in list_checkpoints(ckpt_dir):
        if c.key in skip:
            continue
        problem = None
        if not is_settled(c, settle_s, sleep):
            problem = "is still being written"
        elif loader is not None:
            try:
                loader(c.path)
            except Exception as e:  # a half-copied or damaged file can fail in many ways
                problem = f"cannot be read ({type(e).__name__}: {str(e)[:120]})"
        if problem is None:
            return c
        if notes is not None:
            notes.append(f"{c.name} {problem}")
        if not fall_back:
            return None
    return None


def load_checkpoint_meta(path: str) -> Dict[str, Any]:
    """
    What a learner checkpoint says about itself (learner.py, _save_checkpoint): update_idx,
    total_steps, total_episodes, and the mode it was trained in. Raises when the file cannot be
    read or holds no model, which is how a half-written file shows.
    """
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"expected a dict in the checkpoint, got {type(ckpt).__name__}")
    weights = None
    for k in ("model", "state_dict", "net", "policy", "actor_critic"):
        v = ckpt.get(k)
        if isinstance(v, dict) and any(isinstance(x, torch.Tensor) for x in v.values()):
            weights = v
            break
    if weights is None and any(isinstance(x, torch.Tensor) for x in ckpt.values()):
        weights = ckpt
    if weights is None:
        raise ValueError("no model weights in the checkpoint")

    def as_int(v: Any) -> Optional[int]:
        return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    mode = None
    run_cfg = ckpt.get("run_cfg")
    if isinstance(run_cfg, dict) and isinstance(run_cfg.get("learner"), dict):
        mode = run_cfg["learner"].get("mode")
    meta = {
        "update": as_int(ckpt.get("update_idx")),
        "total_steps": as_int(ckpt.get("total_steps")),
        "total_episodes": as_int(ckpt.get("total_episodes")),
        "phase": mode if isinstance(mode, str) and mode else None,
        "tensors": len(weights),
        # Without the learner's auxiliary heads (SetHead, OppActionHead): the players never load them.
        "tensor_names": sorted(str(k) for k, v in weights.items() if isinstance(v, torch.Tensor)
                               and not str(k).startswith(("set_head.", "opp_action_head."))),
    }
    del ckpt, weights
    return meta


def weights_mismatch(network_names: Iterable[Any], checkpoint_names: Iterable[Any],
                     learned_names: Optional[Iterable[Any]] = None) -> Optional[str]:
    """
    None when the checkpoint has a tensor for every learned tensor of the network and nothing the
    network has no place for, else what is wrong, in words.

    hybrid.PolicyNet loads a checkpoint with strict=False: a tensor of the network that the file
    does not name keeps its untrained value, a tensor of the file the network has no place for is
    dropped, and nothing is raised. A file written by a wrapped or compiled module (every name
    prefixed), or by a model with a layer more or less, would then be played by a network that is
    partly or wholly untrained, under the checkpoint's name.

    `learned_names` are the network's parameters (the rest of `network_names` are buffers: values
    the network builds for itself, like the value head's support). A buffer that the file lacks
    is not untrained weights and is not a mismatch; without `learned_names` every tensor counts
    as learned. Names with "attn_mask" in them are left out on both sides, as learner.py leaves
    them out when it loads.
    """
    net = {str(n) for n in network_names if "attn_mask" not in str(n)}
    ckpt = {str(n) for n in checkpoint_names if "attn_mask" not in str(n)}
    learned = net if learned_names is None else {str(n) for n in learned_names} & net
    missing, unexpected = sorted(learned - ckpt), sorted(ckpt - net)
    parts = []
    if missing:
        parts.append(f"{len(missing)} of the network's {len(learned)} learned tensors are not in the file and would keep their "
                     f"untrained values (for example {missing[0]})")
    if unexpected:
        parts.append(f"{len(unexpected)} of the file's {len(ckpt)} tensors have no place in the network (for example {unexpected[0]})")
    return "; ".join(parts) or None


def steps_from_metrics(metrics_path: Any, update: Optional[int], phase: Optional[str]) -> Optional[int]:
    """total_steps of the given update from metrics.jsonl, for a checkpoint that does not store it."""
    if update is None:
        return None
    found = None
    for rec in read_jsonl(metrics_path):
        if rec.get("update") == update and (phase is None or rec.get("phase") in (None, phase)):
            v = rec.get("total_steps")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                found = int(v)
    return found


def run_decisions(state: Any, phase: Optional[str], total_steps: Optional[int]) -> Optional[int]:
    """
    Decisions of the whole run behind a checkpoint: the finished phases before its phase, from
    state.json, plus its own count (which restarts at zero in every phase). None when unknown.
    """
    if total_steps is None or phase not in PHASE_ORDER or not isinstance(state, dict):
        return None
    phases = state.get("phases")
    if not isinstance(phases, dict):
        return None
    total = int(total_steps)
    for name in PHASE_ORDER[:PHASE_ORDER.index(phase)]:
        info = phases.get(name)
        if not isinstance(info, dict):
            return None
        if info.get("status") == "skipped":
            continue
        d = info.get("decisions")
        if info.get("status") != "done" or not isinstance(d, (int, float)) or isinstance(d, bool):
            return None
        total += int(d)
    return total


def describe_checkpoint(ckpt: Ckpt, loaded: Optional[Dict[str, Any]], state: Any, metrics_path: Any) -> Dict[str, Any]:
    """
    update, phase and total_steps for the result lines: from the checkpoint itself when it stores
    them, otherwise the update from the file name, the phase from state.json and total_steps from
    metrics.jsonl.
    """
    loaded = loaded or {}
    m = CKPT_RE.match(ckpt.name)
    sources = []
    update = loaded.get("update")
    if update is None and m:
        update = int(m.group(1))
        sources.append("update from the file name")
    phase = loaded.get("phase")
    if phase is None:
        phase = state_phase(state)
        if phase == "done":
            phase = None
        if phase is not None:
            sources.append("phase from state.json")
    total_steps = loaded.get("total_steps")
    if total_steps is None:
        total_steps = steps_from_metrics(metrics_path, update, phase)
        if total_steps is not None:
            sources.append("total_steps from metrics.jsonl")
    return {
        "checkpoint": ckpt.name,
        "update": update,
        "phase": phase,
        "total_steps": total_steps,
        "total_episodes": loaded.get("total_episodes"),
        "run_decisions": run_decisions(state, phase, total_steps),
        "checkpoint_path": ckpt.path,
        "checkpoint_mtime": round(ckpt.mtime, 3),
        "checkpoint_size": ckpt.size,
        "meta_from": "checkpoint" if not sources else "checkpoint; " + ", ".join(sources),
    }


# =============================================================================================
# Result lines
# =============================================================================================

@dataclass
class Tally:
    """What one match against one opponent came to, from the policy's side."""
    requested: int
    wins: int = 0
    losses: int = 0
    ties: int = 0
    forfeits: int = 0                 # ended by a forfeit: not a win or a loss
    timeouts: int = 0                 # ended by the inactivity timer: not a win or a loss
    turns: int = 0                    # summed over the games that count
    seconds: float = 0.0
    errors_reported: Optional[int] = None   # the opponent's own runner counted the errors (foulplay)
    stop_reason: Optional[str] = None       # why the match was cut short, if it was
    notes: List[str] = field(default_factory=list)
    counters: Dict[str, int] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def errors(self) -> int:
        if self.errors_reported is not None:
            return max(0, int(self.errors_reported))
        return max(0, self.requested - self.games)

    def count(self, key: str, n: Any) -> None:
        try:
            self.counters[key] = self.counters.get(key, 0) + int(n or 0)
        except (TypeError, ValueError):
            pass

    def absorb(self, other: "Tally") -> None:
        """Adds a second attempt at the same match (after a server restart). `requested` stays."""
        for name in ("wins", "losses", "ties", "forfeits", "timeouts", "turns"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.seconds += other.seconds
        self.stop_reason = other.stop_reason
        self.notes.extend(other.notes)
        for k, v in other.counters.items():
            self.count(k, v)
        self.info.update(other.info)


PROBLEM_TEXT: Tuple[Tuple[str, str], ...] = (
    ("policy_invalid_orders", "orders of the policy were invalid and rejected by the server"),
    ("policy_fallbacks", "decisions of the policy mapped to no order and a random move was played"),
    ("policy_exceptions", "decisions of the policy raised and a random move was played"),
    ("policy_forced_random", "times a random move was forced after repeated rejections"),
    ("opponent_invalid_orders", "orders of the opponent were invalid and rejected by the server"),
    ("opponent_exceptions", "decisions of the opponent raised and a random move was played"),
    ("opponent_fallbacks", "decisions of the opponent fell back to a random move"),
    ("opponent_forced_random", "times the opponent was forced to a random move"),
)


def line_problems(line: Dict[str, Any]) -> List[str]:
    """Everything in a result line that a reader of the win rate should know about."""
    out: List[str] = []
    games = int(line.get("games") or 0)
    errors = int(line.get("errors") or 0)
    requested = int(line.get("requested") or (games + errors))
    if games == 0:
        out.append("no game ran to its end")
    if errors:
        parts = []
        for key, text in (("ended_by_forfeit", "forfeited"), ("ended_by_timeout", "timed out"), ("unfinished", "unfinished")):
            if line.get(key):
                parts.append(f"{line[key]} {text}")
        out.append(f"{errors} of {requested} games did not end normally" + (f" ({', '.join(parts)})" if parts else ""))
    for key, text in PROBLEM_TEXT:
        if line.get(key):
            out.append(f"{line[key]} {text}")
    if line.get("foulplay_log_agrees") is False:
        out.append("Foul Play's own log counts the games of this match differently")
    if line.get("foulplay_slow_search"):
        out.append(str(line["foulplay_slow_search"]))
    out.extend(fixed_iterations_problems(line))
    return out


FOULPLAY_KEYS: Tuple[Tuple[str, str], ...] = (
    # key in the dict bench/foulplay.py returns -> key at the top of the result line
    ("search_time_ms", "foulplay_search_time_ms"),
    ("parallel", "foulplay_parallel"),
    ("foulplay_commit", "foulplay_commit"),
    ("foulplay_mean_iterations", "foulplay_mean_iterations"),
    ("log_agrees", "foulplay_log_agrees"),
    # --foulplay-iterations N: N, and what the logs say was searched (bench/foulplay.py run_matches)
    ("fixed_iterations", "foulplay_iterations"),
    ("iterations_min", "foulplay_iterations_min"),
    ("iterations_max", "foulplay_iterations_max"),
    ("iterations_off", "foulplay_iterations_off"),
    ("iterations_one_option", "foulplay_iterations_one_option"),   # forced moves: one batch of 1,000, in either mode
)


def foulplay_value(line: Dict[str, Any], key: str) -> Any:
    """One of FOULPLAY_KEYS' values of a result line: from the top of the line, or, in a line
    written before they were copied there, from its "opponent_info"."""
    top = dict(FOULPLAY_KEYS)[key]
    if top in line:
        return line[top]
    info = line.get("opponent_info")
    return info.get(key) if isinstance(info, dict) else None


def fixed_iterations_of(line: Dict[str, Any]) -> int:
    """
    The N of --foulplay-iterations a Foul Play line was played at: every sampled world searched for
    exactly N iterations. 0 for a line played on Foul Play's search time, which is every line
    written before the option existed.
    """
    v = foulplay_value(line, "fixed_iterations")
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        return 0
    return int(v)


def fixed_iterations_problems(line: Dict[str, Any]) -> List[str]:
    """What a line played at fixed iterations must say when the logs do not show exactly N everywhere."""
    n = fixed_iterations_of(line)
    if not n or line.get("opponent") != "foulplay" or int(line.get("games") or 0) <= 0:
        return []
    off = foulplay_value(line, "iterations_off")
    low, high = foulplay_value(line, "iterations_min"), foulplay_value(line, "iterations_max")
    if isinstance(off, (int, float)) and not isinstance(off, bool) and off > 0:
        return [f"Foul Play was asked for exactly {n:,} iterations a world, but {int(off):,} of its searches logged another "
                f"count ({_num(low)} to {_num(high)}): this is not a fixed-iteration measurement"]
    if low is None or high is None:
        return [f"Foul Play was asked for exactly {n:,} iterations a world, and its logs show no search to check that by"]
    return []


def play_settings(line: Dict[str, Any]) -> Tuple[Any, ...]:
    """
    What must be equal for two lines against one opponent to be compared. For Foul Play that is
    how long it may search, how many of it ran at once and which version it was: a change in any
    of them moves the win rate without the policy having changed. Nothing for the other opponents.
    A line played at fixed iterations (--foulplay-iterations N) has N as a fourth value, so it is
    never compared or pooled with a line played at another N or on the search time alone.
    """
    if line.get("opponent") != "foulplay":
        return ()
    settings = tuple(foulplay_value(line, k) for k in ("search_time_ms", "parallel", "foulplay_commit"))
    n = fixed_iterations_of(line)
    return settings + (n,) if n else settings


def foulplay_slow_search(iterations: Any, search_time_ms: Any, earlier: Iterable[Dict[str, Any]],
                         fixed_iterations: int = 0) -> Optional[str]:
    """
    Foul Play searches for a fixed time, not a fixed number of iterations. On a machine that is
    busier than it was (training beside it, the evaluator at a lower priority) it gets through
    fewer iterations per decision and may play weaker, and the policy's win rate against it then
    moves with the load. A sentence when this match's mean iterations per decision are below half
    the median of the earlier matches at the same search time, else None. A match played at fixed
    iterations (fixed_iterations > 0) cannot be slow, and is never a match to compare with.
    """
    if fixed_iterations:
        return None
    if isinstance(iterations, bool) or not isinstance(iterations, (int, float)) or iterations <= 0:
        return None
    before = []
    for r in earlier:
        v = foulplay_value(r, "foulplay_mean_iterations")
        if (r.get("opponent") == "foulplay" and int(r.get("games") or 0) > 0 and foulplay_value(r, "search_time_ms") == search_time_ms
                and not fixed_iterations_of(r)
                and isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0):
            before.append(float(v))
    if not before:
        return None
    before.sort()
    middle = len(before) // 2
    median = before[middle] if len(before) % 2 else 0.5 * (before[middle - 1] + before[middle])
    if float(iterations) >= 0.5 * median:
        return None
    return (f"Foul Play searched {int(iterations):,} iterations a decision, less than half the median of {int(median):,} in the "
            f"{len(before)} earlier match{'' if len(before) == 1 else 'es'}: the machine was busier, so it may have played weaker")


def line_valid(line: Dict[str, Any]) -> bool:
    """
    False when the win rate should not be trusted: no games, more than a tenth of the games lost
    to errors, or a side that was not really playing (more than 1% of its decisions replaced by
    random moves).
    """
    games = int(line.get("games") or 0)
    errors = int(line.get("errors") or 0)
    if games <= 0:
        return False
    if errors > 0.10 * max(int(line.get("requested") or (games + errors)), 1):
        return False
    broken = sum(int(line.get(k) or 0) for k in ("policy_exceptions", "policy_forced_random"))
    if broken > 0.01 * max(int(line.get("policy_decisions") or 0), 1) and broken > 0:
        return False
    broken = sum(int(line.get(k) or 0) for k in ("opponent_exceptions", "opponent_forced_random", "opponent_fallbacks"))
    if broken > 0.01 * max(int(line.get("opponent_decisions") or 0), 1) and broken > 0:
        return False
    if fixed_iterations_problems(line):
        return False            # asked for N iterations a world and not shown to have searched exactly that
    return True


def result_line(
    meta: Dict[str, Any],
    opponent: str,
    tally: Tally,
    *,
    sample: bool,
    subject: str = "policy",
    extras: Optional[Dict[str, Any]] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """
    One line of results.jsonl. The contract keys come first; everything after `sample` is extra:

      subject, action ("best" or "sample"), valid, problems, requested, unfinished,
      ended_by_forfeit, ended_by_timeout, mean_turns, games_per_min,
      policy_decisions, policy_invalid_orders (the server rejected the order: a fault),
      policy_unavailable_orders (the server revealed a trap or a disabled move: normal play),
      policy_fallbacks, policy_exceptions, policy_forced_random (a random move was played instead),
      the same with opponent_ in front,
      total_episodes, run_decisions, checkpoint_path, checkpoint_mtime, checkpoint_size, meta_from,
      and whatever the caller adds (device, gpu_peak_mb: the most GPU memory torch held in the
      evaluation process during this match, without the CUDA context; concurrency, port, reason,
      started, state_phase, notes).
    """
    games = tally.games
    half_wins = tally.wins + 0.5 * tally.ties
    lo, hi = wilson(half_wins, games)
    line: Dict[str, Any] = {
        "time": utc_iso(now),
        "checkpoint": meta.get("checkpoint"),
        "update": meta.get("update"),
        "phase": meta.get("phase"),
        "total_steps": meta.get("total_steps"),
        "opponent": opponent,
        "games": games,
        "wins": tally.wins,
        "losses": tally.losses,
        "ties": tally.ties,
        "errors": tally.errors,
        "winrate": round(score(tally.wins, tally.ties, games), 6),
        "ci_low": round(lo, 6),
        "ci_high": round(hi, 6),
        "seconds": round(float(tally.seconds), 2),
        "sample": bool(sample),
        "subject": subject,
        "action": "sample" if sample else "best",
        "requested": tally.requested,
        "unfinished": max(0, tally.errors - tally.forfeits - tally.timeouts),
        "ended_by_forfeit": tally.forfeits,
        "ended_by_timeout": tally.timeouts,
        "mean_turns": round(tally.turns / games, 1) if games and tally.turns else None,
        "games_per_min": round(60.0 * games / tally.seconds, 1) if games and tally.seconds > 0 else None,
    }
    for k, v in tally.counters.items():
        line.setdefault(k, v)
    for k in ("total_episodes", "run_decisions", "checkpoint_path", "checkpoint_mtime", "checkpoint_size", "meta_from"):
        line[k] = meta.get(k)
    for k, v in (extras or {}).items():
        if k not in CORE_KEYS:
            line[k] = v
    if tally.info:
        line["opponent_info"] = tally.info
        if opponent == "foulplay":
            for key, top in FOULPLAY_KEYS:
                if key in tally.info:
                    line[top] = tally.info[key]
    notes = list(line.get("notes") or []) + [n for n in tally.notes if n]
    if tally.stop_reason:
        notes.append(f"match cut short: {tally.stop_reason}")
    if notes:
        line["notes"] = [ascii_text(n)[:300] for n in notes][:12]
    line["problems"] = line_problems(line)
    line["valid"] = line_valid(line)
    return json.loads(json.dumps(line, default=str))


def same_version(line: Dict[str, Any], ckpt: Ckpt) -> bool:
    """Is this result line about this version of the checkpoint file?"""
    if line.get("checkpoint") != ckpt.name:
        return False
    mt = line.get("checkpoint_mtime")
    if isinstance(mt, (int, float)) and not isinstance(mt, bool):
        return abs(float(mt) - ckpt.mtime) < 0.01
    return True  # a line without the time: the name is all there is to go on


def opponents_done(results: Sequence[Dict[str, Any]], ckpt: Ckpt, sample: bool = False) -> List[str]:
    """
    Opponents this version of the checkpoint has a result against, in this play mode. A line
    with no finished game (the server was down, the opponent broke) is not a result: that
    opponent is still to be played, and the watch loop limits how often it tries.
    """
    done: List[str] = []
    for r in results:
        if (bool(r.get("sample")) == bool(sample) and same_version(r, ckpt) and r.get("subject", "policy") == "policy"
                and int(r.get("games") or 0) > 0):
            name = r.get("opponent")
            if isinstance(name, str) and name not in done:
                done.append(name)
    return done


def line_time(line: Dict[str, Any]) -> Optional[float]:
    """When the evaluation a line belongs to started (or, failing that, when the line was written)."""
    return parse_iso(line.get("started")) or parse_iso(line.get("time"))


# =============================================================================================
# Report
# =============================================================================================

def usable(line: Dict[str, Any]) -> bool:
    return int(line.get("games") or 0) > 0 and line.get("valid", True) is not False


def training_order(line: Dict[str, Any]) -> Tuple[Any, ...]:
    """
    Sorts result lines by how far the policy in them was trained: the phase, then the decisions
    of that phase, then the checkpoint file's own time. When the line was written comes last.
    A checkpoint from early in the run that is measured late (the end-of-imitation copy, by hand,
    to have the baseline) must not become "now" and turn every verdict into "worse".
    A phase this run does not know sorts before the known ones.
    """
    phase = line.get("phase")
    rank = next((i for i, name in enumerate(PHASE_ORDER) if isinstance(phase, str) and phase.startswith(name)), -1)

    def number(key: str, default: float) -> float:
        v = line.get(key)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else default

    return rank, number("total_steps", -1.0), number("checkpoint_mtime", 0.0), parse_iso(line.get("time")) or 0.0


def compare(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Is `new` better or worse than `old`, or is there no clear change? "Clear" means the two 95%
    intervals do not overlap, or the pooled two-proportion test gives p < VERDICT_P (0.01).
    """
    n_old, n_new = int(old.get("games") or 0), int(new.get("games") or 0)
    if n_old <= 0 or n_new <= 0:
        return {"verdict": "no clear change", "delta": 0.0, "p": 1.0, "overlap": True, "clear": False}
    p_old, p_new = float(old["winrate"]), float(new["winrate"])
    overlap = not (float(new["ci_low"]) > float(old["ci_high"]) or float(new["ci_high"]) < float(old["ci_low"]))
    p = two_proportion_p(p_old, n_old, p_new, n_new)
    clear = (not overlap or p < VERDICT_P) and p_new != p_old
    verdict = "no clear change" if not clear else ("better" if p_new > p_old else "worse")
    return {"verdict": verdict, "delta": round(100.0 * (p_new - p_old), 1), "p": p, "overlap": overlap, "clear": clear}


def pool(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Several result lines taken together as one: the games added up, a tie still half a win.
    Raises ValueError for Foul Play lines played at different --foulplay-iterations (0, Foul
    Play's search time, counts as one more N): those measure different opponents.
    """
    fixed = sorted({fixed_iterations_of(r) for r in rows if r.get("opponent") == "foulplay"})
    if len(fixed) > 1:
        raise ValueError("refusing to pool Foul Play lines played at different fixed iterations: "
                         + ", ".join(f"{n:,}" if n else "its search time" for n in fixed))
    games = sum(int(r.get("games") or 0) for r in rows)
    half_wins = sum(int(r.get("wins") or 0) + 0.5 * int(r.get("ties") or 0) for r in rows)
    lo, hi = wilson(half_wins, games)
    return {"games": games, "winrate": half_wins / games if games else 0.0, "ci_low": lo, "ci_high": hi, "lines": len(rows)}


def opponent_verdict(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    For one opponent's lines, in any order: the usable line of the most trained checkpoint (see
    training_order) compared with the one before it and with the first, among lines played the
    same way (best action or sampled, and for Foul Play with the same settings: play_settings).

    "trend" is the last TREND_POOL of those taken together against the TREND_POOL before them
    (None with fewer than twice that many). One evaluation of 200 games sees a real gain of five
    points less than one time in ten at p < 0.01; three of them on each side see it far more
    often, so this is the comparison that shows a slow gain.
    "other_settings" counts the usable lines that were left out because they were played another
    way.
    """
    good = sorted((r for r in rows if usable(r)), key=training_order)
    if not good:
        return {"latest": None, "previous": None, "first": None, "vs_previous": None, "vs_first": None,
                "trend": None, "other_settings": 0}
    latest = good[-1]
    same = [r for r in good if bool(r.get("sample")) == bool(latest.get("sample")) and play_settings(r) == play_settings(latest)]
    previous = same[-2] if len(same) >= 2 else None
    first = same[0] if len(same) >= 2 else None
    trend = None
    if len(same) >= 2 * TREND_POOL:
        trend = compare(pool(same[-2 * TREND_POOL:-TREND_POOL]), pool(same[-TREND_POOL:]))
    return {
        "latest": latest,
        "previous": previous,
        "first": first,
        "vs_previous": compare(previous, latest) if previous is not None else None,
        "vs_first": compare(first, latest) if first is not None else None,
        "trend": trend,
        "other_settings": len(good) - len(same),
    }


def _pct(x: Any) -> str:
    return f"{100.0 * float(x):.1f}%"


def _p_text(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def _num(x: Any) -> str:
    return f"{int(x):,}" if isinstance(x, (int, float)) and not isinstance(x, bool) else "-"


def _where(row: Dict[str, Any]) -> str:
    upd = row.get("update")
    phase = row.get("phase")
    return f"update {upd if upd is not None else '?'}" + (f" ({phase})" if phase else "")


def _cmp_cell(c: Optional[Dict[str, Any]]) -> str:
    if c is None:
        return "-"
    return f"{c['verdict']} ({c['delta']:+.1f} points, {_p_text(c['p'])})"


def _cmp_sentence(c: Dict[str, Any]) -> str:
    how = "the intervals do not overlap" if not c["overlap"] else "the intervals overlap"
    return f"{c['verdict'].capitalize()}: {c['delta']:+.1f} points, {_p_text(c['p'])}, {how}."


def _why_nothing(row: Dict[str, Any]) -> List[str]:
    """For a line with no finished game: its notes, which hold the reason (the opponent is not
    installed, the checkpoint does not fit the network, the server would not start). The report
    prints them, because "no game ran to its end" alone says nothing about what to do."""
    if int(row.get("games") or 0) > 0:
        return []
    return [ascii_text(n)[:240] for n in (row.get("notes") or []) if isinstance(n, str) and n][:2]


def _table_rows(rows: Sequence[Dict[str, Any]], foulplay: bool = False) -> List[str]:
    """The history table of one opponent. With foulplay=True two more columns: what Foul Play was
    allowed (search time, processes at once) and what it managed (iterations a decision)."""
    out = [
        "| Measured (UTC) | Phase | Update | Decisions | Win rate | 95% interval | Games | W-L-T | Errors | Turns | "
        + ("Search | Iterations | " if foulplay else "") + "Checkpoint | Notes |",
        "|---|---|---:|---:|---:|---|---:|---|---:|---:|" + ("---|---:|" if foulplay else "") + "---|---|",
    ]
    for r in rows:
        games = int(r.get("games") or 0)
        notes = []
        if r.get("sample"):
            notes.append("sampled")
        if r.get("reason") in ("final", "phase_change"):
            notes.append("final" if r["reason"] == "final" else "phase change")
        if r.get("valid") is False:
            notes.append("NOT TRUSTED")
        notes.extend(str(p) for p in (r.get("problems") or []))
        notes.extend(_why_nothing(r))
        turns = r.get("mean_turns")
        when = str(r.get("time", "-"))[:16].replace("T", " ")
        update = r.get("update") if r.get("update") is not None else "-"
        rate = _pct(r["winrate"]) if games else "-"
        interval = "{:.1f} to {:.1f}".format(100 * r["ci_low"], 100 * r["ci_high"]) if games else "-"
        wlt = "{}-{}-{}".format(r.get("wins", 0), r.get("losses", 0), r.get("ties", 0))
        search = ""
        if foulplay:
            ms, procs = foulplay_value(r, "search_time_ms"), foulplay_value(r, "parallel")
            search = (f"{ms} ms" if ms is not None else "-") + (f" x{procs}" if procs is not None else "")
            if fixed_iterations_of(r):
                search += f", {fixed_iterations_of(r):,} iterations fixed"
            search = f"{search} | {_num(foulplay_value(r, 'foulplay_mean_iterations'))} | "
        out.append(
            f"| {when} | {r.get('phase') or '-'} | {update} | {_num(r.get('total_steps'))} | {rate} | {interval} | {games} | "
            f"{wlt} | {r.get('errors', 0)} | {turns if turns is not None else '-'} | {search}{r.get('checkpoint') or '-'} | "
            f"{ascii_text('; '.join(notes)) or '-'} |"
        )
    return out


def build_report(results: Sequence[Dict[str, Any]], hybrid: Sequence[Dict[str, Any]] = (),
                 run_dir: str = "", now: Optional[float] = None) -> str:
    """REPORT.md as text, from the lines of results.jsonl (and hybrid_results.jsonl)."""
    # In order of training, not of measuring: see training_order.
    rows = sorted((r for r in results if isinstance(r.get("opponent"), str) and "winrate" in r and "ci_low" in r and "ci_high" in r),
                  key=training_order)
    names = [o for o in OPPONENTS if any(r["opponent"] == o for r in rows)]
    names += sorted({r["opponent"] for r in rows} - set(names))
    by_opp = {o: [r for r in rows if r["opponent"] == o] for o in names}
    versions = {(r.get("checkpoint"), r.get("checkpoint_mtime")) for r in rows}
    out: List[str] = ["# Evaluation report", ""]
    if run_dir:
        out.append(f"Run directory: `{run_dir}`")
        out.append("")
    out.append(f"Written {utc_iso(now)}. {len(rows)} result line{'' if len(rows) == 1 else 's'} for "
               f"{len(versions)} checkpoint{'' if len(versions) == 1 else 's'}.")
    if not rows:
        out += ["", "No evaluation has finished yet.", ""]
        return "\n".join(out)
    last = rows[-1]
    out.append(f"Most trained checkpoint evaluated: `{last.get('checkpoint')}`, {_where(last)}, "
               f"{_num(last.get('total_steps'))} decisions in its phase.")
    out += [
        "",
        "The policy plays alone, without the PSBot guard, and takes its best legal action each turn. "
        "A row marked \"sampled\" drew the action from the policy's probabilities instead.",
        "",
        "## Is it getting stronger?",
        "",
        "\"Now\" is the most trained checkpoint that was evaluated, whenever it was measured. Checkpoints are compared in "
        "order of training: phase, then decisions.",
        "",
        f"| Opponent | Now | 95% interval | Games | Against the checkpoint before | Against the first | "
        f"The last {TREND_POOL} together against the {TREND_POOL} before |",
        "|---|---:|---|---:|---|---|---|",
    ]
    verdicts = {o: opponent_verdict(by_opp[o]) for o in names}
    comparisons = 0
    for o in names:
        v = verdicts[o]
        if v["latest"] is None:
            out.append(f"| {o} | - | - | 0 | no usable evaluation yet | - | - |")
            continue
        r = v["latest"]
        comparisons += sum(1 for c in (v["vs_previous"], v["vs_first"], v["trend"]) if c is not None)
        out.append(f"| {o} | {_pct(r['winrate'])} | {100 * r['ci_low']:.1f} to {100 * r['ci_high']:.1f} | {r['games']} | "
                   f"{_cmp_cell(v['vs_previous'])} | {_cmp_cell(v['vs_first'])} | {_cmp_cell(v['trend'])} |")
    out.append("")
    if comparisons:
        chance = 100.0 * (1.0 - (1.0 - VERDICT_P) ** comparisons)
        out.append(f"This table makes {comparisons} comparison{'' if comparisons == 1 else 's'}. Each says \"better\" or \"worse\" only "
                   f"at p < {VERDICT_P:g} (or when the two intervals do not overlap), so with a policy that has not changed at all "
                   f"at most about {max(1, round(chance))} report{'' if round(chance) <= 1 else 's'} in 100 still "
                   f"show{'s' if round(chance) <= 1 else ''} a verdict somewhere in it. \"No clear change\" is not \"no change\": "
                   f"see \"How much to trust this\" for how large a difference has to be before it is called.")
        out.append("")
    for o in names:
        v = verdicts[o]
        r = v["latest"]
        if r is None:
            out.append(f"- **{o}**: no usable evaluation yet.")
            continue
        text = f"- **{o}**: {_pct(r['winrate'])} at {_where(r)}."
        if v["previous"] is None:
            text += " One evaluation so far, nothing to compare with."
        else:
            text += f" The checkpoint before it, {_where(v['previous'])}: {_pct(v['previous']['winrate'])}. {_cmp_sentence(v['vs_previous'])}"
            if v["first"] is not v["previous"]:
                text += f" The first, {_where(v['first'])}: {_pct(v['first']['winrate'])}. {_cmp_sentence(v['vs_first'])}"
        if v["other_settings"]:
            text += (f" {v['other_settings']} other evaluation{'' if v['other_settings'] == 1 else 's'} against it "
                     + ("were" if v["other_settings"] != 1 else "was") + " played another way"
                     + (" (Foul Play's search time, its fixed iterations, its number of processes or its version differ)"
                        if o == "foulplay" else "")
                     + " and " + ("are" if v["other_settings"] != 1 else "is") + " not compared with this one.")
        newest_row = by_opp[o][-1]
        if newest_row is not r:
            text += (f" The line of the most trained checkpoint ({_where(newest_row)}) is not used: "
                     f"{'; '.join(newest_row.get('problems') or ['not valid'])}.")
        out.append(text)

    sizes = sorted(int(r["games"]) for r in rows if int(r.get("games") or 0) > 0)
    typical = sizes[len(sizes) // 2] if sizes else 0
    out += ["", "## How much to trust this", ""]
    if typical:
        out.append(f"- A typical row here is {typical} games. At that size a win rate near 50% is known to about plus or minus "
                   f"{noise_points(typical):.0f} points, and the difference between two rows to about plus or minus "
                   f"{noise_points(typical) * math.sqrt(2):.0f}. For comparison: 200 games give about plus or minus 7 points, "
                   f"800 games about 3.5.")
    out += [
        f"- \"Better\" or \"worse\" is said only when the two 95% intervals do not overlap, or a two-proportion test gives "
        f"p < {VERDICT_P:g}. About one comparison in a hundred passes that test by chance, and this report makes up to three "
        f"per opponent every time it is written. One verdict alone is weak evidence. The same direction against several "
        f"opponents, or in several evaluations in a row, is strong.",
    ]
    if typical:
        out.append(f"- The price of that caution: near a win rate of 50%, two rows of {typical} games must differ by about "
                   f"{clear_points(typical):.0f} points before the difference is called, so a smaller real gain mostly reads "
                   f"\"no clear change\". The last column of the table at the top takes {TREND_POOL} evaluations together on each "
                   f"side, where about {clear_points(TREND_POOL * typical):.0f} points are enough: that is the column that shows a slow gain.")
    out += [
        "- Rows are in order of training (phase, then decisions, then the checkpoint's own time). The first column of each "
        "table below says when the row was measured, which can be much later.",
        "- A tie counts as half a win. A game that did not run to its end (unfinished, forfeited, timed out) is not a win "
        "or a loss: it is in the Errors column. A row marked NOT TRUSTED is left out of the verdicts.",
        "- Reference points: PSBot, the imitation teacher, beats `simple` in about 91% of games (HYBRID.md), so that is "
        "what imitation can reach against `simple`. 50% against `psbot` means level with the teacher. `random` is the "
        "floor: a policy that does not beat it is broken.",
        "- Decisions are counted per phase. The count starts again at zero when the phase changes.",
        "",
    ]
    bad = sorted((r for r in rows if r.get("problems")), key=lambda r: (parse_iso(r.get("time")) or 0.0))
    if bad:
        out += ["## Problems seen", ""]
        for r in bad[-20:]:
            out.append(f"- {str(r.get('time', ''))[:16].replace('T', ' ')} {r['opponent']}, {_where(r)}: "
                       f"{ascii_text('; '.join([str(p) for p in r['problems']] + _why_nothing(r)))}")
        out.append("")
    for o in names:
        out += [f"## {o}", ""] + _table_rows(by_opp[o], foulplay=(o == "foulplay")) + [""]
    hyb = sorted((r for r in hybrid if isinstance(r.get("opponent"), str) and "winrate" in r and "ci_low" in r and "ci_high" in r),
                 key=training_order)
    if hyb:
        out += ["## Extra: the PSBot-guarded hybrid", "",
                "The same checkpoints played through hybrid.py (policy and PSBot together). Not part of the verdicts above.", ""]
        for o in [n for n in OPPONENTS if any(r["opponent"] == n for r in hyb)]:
            out += [f"### hybrid against {o}", ""] + _table_rows([r for r in hyb if r["opponent"] == o]) + [""]
    return "\n".join(out)


def write_report(paths: RunPaths) -> None:
    text = build_report(read_jsonl(paths.results), read_jsonl(paths.hybrid_results), paths.run)
    write_text_atomic(paths.report, text)


def print_table(lines: Sequence[Dict[str, Any]], meta: Dict[str, Any], sample: bool) -> None:
    say("")
    say(f"checkpoint {meta.get('checkpoint')}  phase {meta.get('phase')}  update {meta.get('update')}  "
        f"decisions {_num(meta.get('total_steps'))}  policy plays: {'sampled action' if sample else 'best legal action'}")
    say(f"{'subject':<7} {'opponent':<9} {'games':>5} {'W-L-T':>11} {'win rate':>8} {'95% interval':>14} {'errors':>6} "
        f"{'turns':>6} {'games/min':>9} {'invalid':>8} {'fallback':>8}  note")
    for r in lines:
        games = int(r.get("games") or 0)
        wlt = f"{r.get('wins', 0)}-{r.get('losses', 0)}-{r.get('ties', 0)}"
        rate = _pct(r["winrate"]) if games else "-"
        ci = f"{100 * r['ci_low']:.1f} to {100 * r['ci_high']:.1f}" if games else "-"
        fallback = sum(int(r.get(k) or 0) for k in ("policy_fallbacks", "policy_exceptions", "policy_forced_random"))
        note = "" if r.get("valid") else "NOT TRUSTED: " + "; ".join(r.get("problems") or [])
        say(f"{r.get('subject', 'policy'):<7} {r['opponent']:<9} {games:>5} {wlt:>11} {rate:>8} {ci:>14} {r.get('errors', 0):>6} "
            f"{r.get('mean_turns') if r.get('mean_turns') is not None else '-':>6} "
            f"{r.get('games_per_min') if r.get('games_per_min') is not None else '-':>9} "
            f"{r.get('policy_invalid_orders', 0):>8} {fallback:>8}  {note}")


# =============================================================================================
# Processes and the evaluation server
# =============================================================================================

def port_listening(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def detached_kwargs() -> Dict[str, Any]:
    """Start a child in its own group, so that a Ctrl+C meant for us is ours to pass on."""
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


class WinJob:
    """
    A Windows job object around one child and everything it starts (the same thing pipeline.py
    has, repeated here so that each of the two files runs on its own).

    On Windows a process tree is only a chain of parent ids, and "taskkill /T" walks that chain
    from a new process: it finds nothing once the parent has gone, and it can fail outright when
    the machine is short of memory. Seen on the laptop: an evaluation ended, its Showdown server
    stayed up on the evaluation port. A job has neither hole: what is started inside it stays in
    it, one call from this process ends it all, and it ends by itself if this process dies.
    """

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.CreateJobObjectW.restype = wintypes.HANDLE
        k.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        k.SetInformationJobObject.restype = wintypes.BOOL
        k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
        k.QueryInformationJobObject.restype = wintypes.BOOL
        k.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD, wintypes.LPVOID]
        k.AssignProcessToJobObject.restype = wintypes.BOOL
        k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k.TerminateJobObject.restype = wintypes.BOOL
        k.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k.CloseHandle.restype = wintypes.BOOL
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        self._k = k
        self.handle = k.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObject failed")

        class BasicLimits(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", ctypes.c_uint64 * 6),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        info = ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = 0x2000     # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):   # 9: extended limit information
            err = ctypes.get_last_error()
            self.close()
            raise OSError(err, "SetInformationJobObject failed")

    def assign(self, process_handle: int) -> None:
        if not self._k.AssignProcessToJobObject(self.handle, int(process_handle)):
            raise OSError(self._ctypes.get_last_error(), "AssignProcessToJobObject failed")

    def pids(self) -> List[int]:
        """The ids of the processes that are in the job now. Empty when it cannot be asked."""
        if not self.handle:
            return []
        ctypes = self._ctypes
        size = 4096
        for _ in range(4):
            class IdList(ctypes.Structure):
                _fields_ = [("NumberOfAssignedProcesses", ctypes.c_uint32), ("NumberOfProcessIdsInList", ctypes.c_uint32),
                            ("ProcessIdList", ctypes.c_size_t * size)]
            buf = IdList()
            if self._k.QueryInformationJobObject(self.handle, 3, ctypes.byref(buf), ctypes.sizeof(buf), None):   # 3: basic process id list
                return [int(buf.ProcessIdList[i]) for i in range(buf.NumberOfProcessIdsInList)]
            if ctypes.get_last_error() != 234:      # ERROR_MORE_DATA: more processes than fit
                return []
            size *= 8
        return []

    def terminate(self) -> bool:
        """Ends every process in the job, in one call. False when the call was refused."""
        return bool(self.handle) and bool(self._k.TerminateJobObject(self.handle, 1))

    def close(self) -> None:
        if self.handle:
            self._k.CloseHandle(self.handle)
            self.handle = None


def start_process(cmd: Sequence[str], **kwargs: Any) -> subprocess.Popen:
    """
    subprocess.Popen in a process group of its own and, on Windows, inside a job object of its
    own, which kill_tree() then uses. The job is kept on the returned object as `psppo_job`.
    """
    job = None
    if os.name == "nt":
        try:
            job = WinJob()
        except Exception:
            job = None                       # kill_tree() falls back to taskkill alone
    try:
        proc = subprocess.Popen(list(cmd), **kwargs, **detached_kwargs())
    except BaseException:
        if job is not None:
            job.close()
        raise
    if job is not None:
        try:
            job.assign(proc._handle)         # type: ignore[attr-defined]  at once, before it has started anything of its own
        except Exception:
            job.close()
            job = None
    proc.psppo_job = job                     # type: ignore[attr-defined]
    return proc


def kill_tree(proc: Optional[subprocess.Popen], grace: float = 10.0) -> List[int]:
    """
    Stops a process started here and what it started, by its own PID, never by name, and also
    when the process itself has already gone but something it started has not. Returns the ids
    of what is still alive afterwards (normally none; never raises).
    """
    if proc is None:
        return []
    job = getattr(proc, "psppo_job", None)
    if os.name == "nt":
        left: List[int] = []
        try:
            if job is not None:
                job.terminate()              # one call, nothing to walk, takes orphans too
            if proc.poll() is None:
                try:                         # taskkill as well: reaches what was started before the job held the root
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
                except Exception as e:
                    say(f"taskkill of pid {proc.pid} failed: {e!r}")
            try:
                proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                proc.kill()
            deadline = time.time() + min(max(grace, 2.0), 15.0)
            while job is not None and job.pids() and time.time() < deadline:
                time.sleep(0.2)
            left = job.pids() if job is not None else ([proc.pid] if proc.poll() is None else [])
            if left:
                say(f"{len(left)} processes under pid {proc.pid} are still alive after the kill: {left[:20]}")
        except Exception as e:               # stopping must never take the caller down with it
            say(f"could not stop pid {proc.pid} cleanly: {e!r}")
        finally:
            if job is not None:
                job.close()                  # the last handle of a KILL_ON_JOB_CLOSE job: ends what is still in it
                proc.psppo_job = None        # type: ignore[attr-defined]
        return left
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)  # started with start_new_session: its group id is its pid
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
        deadline = time.time() + 3.0
        while time.time() < deadline:
            try:
                os.killpg(proc.pid, 0)
            except (ProcessLookupError, PermissionError):
                break
            time.sleep(0.2)
    # The leader has gone, or would not go: whatever is left in its group is ended now.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=grace)
    except Exception:
        pass
    deadline = time.time() + 5.0
    while True:                              # SIGKILL is not instant: the group empties a moment later
        try:
            os.killpg(proc.pid, 0)
        except (ProcessLookupError, PermissionError):
            return []
        if time.time() > deadline:
            return [proc.pid]
        time.sleep(0.1)


def pid_alive(pid: Any) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(ok) and code.value == 259  # STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def pid_command_line(pid: Any, needle: str, proc_root: str = "/proc") -> Optional[bool]:
    """
    Does the command line of process `pid` contain `needle`? True or False where the system
    exposes it (Linux: /proc/PID/cmdline), None where it cannot be told (Windows, no /proc, no
    permission): the caller then has only the process id to go on.
    """
    try:
        with open(os.path.join(proc_root, str(int(pid)), "cmdline"), "rb") as f:
            raw = f.read()
    except (OSError, TypeError, ValueError):
        return None
    if not raw:
        return None            # a kernel thread or a process on its way out: nothing to read
    return needle.encode() in raw


class EvalServer:
    """
    The Showdown server the evaluation plays on. ensure() starts one only when nothing is
    listening on the port and a server directory was given; stop() stops only what ensure()
    started. Whatever else is on the machine (the training servers) is never looked at.
    """

    def __init__(self, server_dir: Optional[str], port: int, log_dir: str, host: str = "127.0.0.1", heap_mb: int = 2048):
        self.server_dir = os.path.abspath(server_dir) if server_dir else None
        self.port = int(port)
        self.host = host
        self.log_dir = log_dir
        self.heap_mb = int(heap_mb)
        self.proc: Optional[subprocess.Popen] = None
        self.starts = 0
        self._log = None

    @property
    def owned(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def listening(self) -> bool:
        return port_listening(self.host, self.port)

    def ensure(self, wait_s: float = 120.0) -> Tuple[bool, str]:
        if self.listening():
            return True, "a server is listening" + (" (started here)" if self.owned else "")
        where = f"{self.host}:{self.port}"
        if self.proc is not None:
            # Ours, and it is not answering: stop what is left of it before starting again.
            kill_tree(self.proc)
            self.proc = None
        if not self.server_dir:
            return False, f"nothing is listening on {where} and no server directory was given (--server-dir)"
        if self.host not in ("127.0.0.1", "localhost", "::1"):
            return False, f"nothing is listening on {where}, and a server can only be started on this machine"
        if not os.path.isfile(os.path.join(self.server_dir, "pokemon-showdown")):
            return False, f"{self.server_dir} is not a Pokemon Showdown checkout (no pokemon-showdown file)"
        if not os.path.isdir(os.path.join(self.server_dir, "dist")):
            return False, f"{self.server_dir} is not built (no dist/): run `node build` there once"
        node = os.environ.get("PSBOT_NODE") or shutil.which("node")
        if not node:
            return False, "node was not found on PATH"
        os.makedirs(self.log_dir, exist_ok=True)
        if self._log is None:
            self._log = open(os.path.join(self.log_dir, f"server_{self.port}.log"), "ab")
        self._log.write(f"\n=== {utc_iso()} starting on port {self.port} from {self.server_dir}\n".encode("ascii", "replace"))
        self._log.flush()
        # The same command start_servers.py uses. --skip-build: a build under running servers breaks them.
        cmd = [node, f"--max-old-space-size={self.heap_mb}", "pokemon-showdown", "start", "--no-security", "--skip-build",
               "--port", str(self.port)]
        self.proc = start_process(cmd, cwd=self.server_dir, stdin=subprocess.DEVNULL, stdout=self._log, stderr=self._log)
        self.starts += 1
        deadline = time.time() + wait_s
        while time.time() < deadline:
            if self.listening():
                return True, f"started a server on {where} (pid {self.proc.pid})"
            if self.proc.poll() is not None:
                code, self.proc = self.proc.returncode, None
                return False, f"the server exited with code {code} while starting (see {self.log_dir}/server_{self.port}.log)"
            time.sleep(0.25)
        kill_tree(self.proc)
        self.proc = None
        return False, f"the server did not listen on {where} within {wait_s:.0f}s"

    def stop(self) -> None:
        if self.proc is not None:
            left = kill_tree(self.proc)
            self.proc = None
            # A server of ours that stays up keeps the port, and every later evaluation would then
            # play on a server that nobody restarts: say so rather than leave it unnoticed.
            deadline = time.time() + 10.0
            while time.time() < deadline and self.listening():
                time.sleep(0.3)
            if left or self.listening():
                say(f"WARNING: the evaluation server on {self.host}:{self.port} could not be stopped completely "
                    f"(processes still alive: {left or 'unknown'}). Stop it by its process id.")
        if self._log is not None:
            try:
                self._log.close()
            except OSError:
                pass
            self._log = None


# =============================================================================================
# Playing
# =============================================================================================

SERVER_GONE = "the server went away"


@dataclass
class EvalContext:
    host: str
    port: int
    battle_format: str
    concurrency: int
    checkpoint: Ckpt
    sample: bool
    device: str                        # "cuda" or "cpu": what is asked for; falls to cpu if cuda fails
    batch: bool
    stall_seconds: float
    foulplay_parallel: int
    foulplay_search_ms: int
    foulplay_module: str
    foulplay_log_dir: str
    server: EvalServer
    notes: List[str] = field(default_factory=list)
    # the names of the tensors in the checkpoint file (load_checkpoint_meta), None when not known
    tensor_names: Optional[frozenset] = None
    # --foulplay-iterations: > 0 searches every sampled world for exactly this many iterations
    foulplay_iterations: int = 0

    def note(self, text: str) -> None:
        text = ascii_text(text)
        if text not in self.notes:
            self.notes.append(text)
            say("  note:", text)


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.messages.append(record.getMessage())
        except Exception:
            pass


def _is_cuda_failure(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "cuda" in text or "out of memory" in text or "cudnn" in text


def _warm_up(player: Any) -> None:
    """
    One forward pass on an empty observation, before any game. The first pass on a GPU takes
    seconds (it would be counted as game time), and it is where a GPU without free memory fails:
    better here, where the caller can still fall back to the CPU, than in the middle of a game.
    """
    import numpy as np

    policy = player.policy
    size = max(end for _start, end in policy.assembler.offsets.values())
    obs = np.zeros(size, dtype=np.float32)
    tag = "__warm_up__"
    try:
        if hasattr(player, "_forward_batch"):
            player._forward_batch([tag], obs[None, :])
            player.batch_stats.update({"forwards": 0, "rows": 0, "largest": 0})
        else:
            policy.logits(tag, obs)
    finally:
        policy.forget(tag)


def _make_subject(ctx: EvalContext, kind: str, concurrency: int) -> Any:
    """The player being measured: the checkpoint's policy ("policy") or the PSBot-guarded "hybrid"."""
    from play import build_player, server_configuration

    listener = _ListHandler()
    hybrid_log = logging.getLogger("hybrid")
    hybrid_log.addHandler(listener)
    try:
        devices = [ctx.device] if ctx.device == "cpu" else [ctx.device, "cpu"]
        for i, dev in enumerate(devices):
            try:
                player = build_player(
                    kind, server=server_configuration(ctx.host, ctx.port), tag=secrets.token_hex(3) + "a",
                    battle_format=ctx.battle_format, concurrency=concurrency, checkpoint=ctx.checkpoint.path,
                    sample=ctx.sample, device=dev, audit=True, batch=ctx.batch,
                )
                if getattr(player, "policy", None) is not None:
                    try:
                        _warm_up(player)
                    except Exception as e:
                        if _is_cuda_failure(e):
                            raise
                        ctx.note(f"the warm-up forward pass failed ({type(e).__name__}: {str(e)[:120]}); carrying on")
                break
            except Exception as e:
                if i == len(devices) - 1 or not _is_cuda_failure(e):
                    raise
                ctx.note(f"cuda failed ({type(e).__name__}: {str(e)[:120]}); the policy runs on cpu")
                ctx.device = "cpu"
    finally:
        hybrid_log.removeHandler(listener)
    # HybridPlayer turns into PSBot when it finds no network. That must never be measured as the policy.
    policy = getattr(player, "policy", None)
    if policy is None or not policy.path or not os.path.samefile(policy.path, ctx.checkpoint.path):
        raise RuntimeError(f"the {kind} player did not load {ctx.checkpoint.path}")
    # Nor must a network that took only some of its weights from the file, or none: hybrid.PolicyNet
    # loads with strict=False and only logs a warning (see weights_mismatch).
    problem = None
    net = getattr(policy, "net", None)
    if ctx.tensor_names is not None and net is not None and callable(getattr(net, "state_dict", None)):
        names = [str(n) for n in net.state_dict().keys()]
        learned = [n for n, _p in net.named_parameters()] if callable(getattr(net, "named_parameters", None)) else None
        problem = weights_mismatch(names, ctx.tensor_names, learned)
        built = sorted(n for n in set(names) - ctx.tensor_names - set(learned or names) if "attn_mask" not in n)
        if problem is None and built:
            ctx.note(f"{len(built)} buffer(s) of the network are not in {ctx.checkpoint.name} and keep the values the network builds "
                     f"for them (for example {built[0]})")
    else:
        # nothing to compare names with: the loader's own warning is all there is to go on
        problem = next((m for m in listener.messages if "missing and" in m and "unexpected" in m), None)
    if problem is not None:
        _release(player)
        raise RuntimeError(f"{ctx.checkpoint.name} does not fit the network, so it is not played (it would be measured with "
                           f"untrained weights under this checkpoint's name): {problem}")
    for message in listener.messages:
        ctx.note(f"while loading the model: {message}")
    return player


async def _wait_logged_in(players: Sequence[Any], timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if all(p.ps_client.logged_in.is_set() for p in players):
            return
        await asyncio.sleep(0.05)
    late = [p.username for p in players if not p.ps_client.logged_in.is_set()]
    raise RuntimeError(f"could not log in to the server within {timeout:.0f}s: {', '.join(late)}")


async def _disconnect(players: Sequence[Any]) -> None:
    for p in players:
        try:
            await asyncio.wait_for(p.ps_client.stop_listening(), timeout=5.0)
        except Exception:
            pass


def _gpu_peak_reset() -> None:
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _gpu_peak_mb() -> Optional[float]:
    """
    The most GPU memory torch held in this process since _gpu_peak_reset(), in MiB, or None when
    the GPU was not used. It is what the network and its batches take. The CUDA context every
    process on a GPU also needs (a few hundred MiB, more on Windows) is not in it: nvidia-smi shows
    that.
    """
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            return round(torch.cuda.max_memory_reserved() / (1024.0 * 1024.0), 1)
    except Exception:
        pass
    return None


def _release(player: Any) -> None:
    """Lets go of the network of a player that is finished, so the next one does not add to it."""
    try:
        player.policy = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _progress(player: Any) -> Tuple[int, int, int]:
    """(battles finished, turns played, battles started). Read from another thread: best effort."""
    for _ in range(3):
        try:
            battles = list(getattr(player, "battles", {}).values())
            return sum(1 for b in battles if b.finished), sum(int(b.turn or 0) for b in battles), len(battles)
        except RuntimeError:  # the dict changed while it was being read
            continue
    return -1, -1, -1


async def _watch_match(coro: Any, subject: Any, ctx: EvalContext, label: str, requested: int,
                       stall_s: float, cap_s: float) -> Optional[str]:
    """
    Runs the match and gives up on it when it stops moving. Returns None when it ran to the end,
    otherwise the reason it was cut short. An exception raised by the match itself is passed on.
    """
    task = asyncio.ensure_future(coro)
    t0 = last_change = last_print = time.time()
    last_sig: Any = None
    port_misses = 0
    reason: Optional[str] = None
    while True:
        done, _pending = await asyncio.wait({task}, timeout=1.0)
        if done:
            break
        now = time.time()
        sig = _progress(subject)
        if sig != last_sig:
            last_sig, last_change = sig, now
        port_misses = 0 if port_listening(ctx.host, ctx.port, timeout=2.0) else port_misses + 1
        if port_misses >= 3:
            reason = SERVER_GONE
        elif now - last_change > stall_s:
            reason = f"no progress for {stall_s:.0f}s"
        elif now - t0 > cap_s:
            reason = f"time limit of {cap_s:.0f}s reached"
        if reason:
            break
        if now - last_print >= 30.0:
            last_print = now
            say(f"  {label}: {max(sig[0], 0)} of {requested} games finished after {now - t0:.0f}s")
    if reason is None:
        exc = task.exception()
        if exc is not None:
            raise exc
        return None
    task.cancel()
    await asyncio.wait({task}, timeout=20.0)
    return reason


def _collect_battles(tally: Tally, subject: Any, skip: Iterable[str] = (), count_results: bool = True) -> None:
    """Counts the subject's battles. A forfeit or a timer loss is not a result."""
    skip = set(skip)
    endings = getattr(subject, "audit_endings", {})
    for tag, b in list(subject.battles.items()):
        if tag in skip or not b.finished:
            continue
        how = endings.get(tag)
        if how == "forfeit":
            tally.forfeits += 1
            continue
        if how == "inactivity":
            tally.timeouts += 1
            continue
        tally.turns += int(b.turn or 0)
        if not count_results:
            continue
        if b.won:
            tally.wins += 1
        elif b.lost:
            tally.losses += 1
        else:
            tally.ties += 1


def _collect_audit(tally: Tally, subject: Any, opponent: Any = None) -> None:
    a = getattr(subject, "audit", {})
    counts = getattr(subject, "counts", {})
    tally.count("policy_decisions", counts.get("decisions", a.get("requests", 0)))
    tally.count("policy_fallbacks", counts.get("fallbacks", 0))
    tally.count("policy_invalid_orders", a.get("invalid_choices", 0) + a.get("other_errors", 0))
    tally.count("policy_unavailable_orders", a.get("unavailable_choices", 0))
    tally.count("policy_exceptions", a.get("exceptions", 0))
    tally.count("policy_forced_random", a.get("forced_random", 0))
    stats = getattr(subject, "batch_stats", None)
    if stats:
        tally.count("policy_forward_passes", stats.get("forwards", 0))
    for text in list(getattr(subject, "audit_messages", []))[:3]:
        tally.notes.append(f"policy: {text}")
    if opponent is not None:
        o = getattr(opponent, "audit", {})
        tally.count("opponent_decisions", o.get("requests", 0))
        tally.count("opponent_invalid_orders", o.get("invalid_choices", 0) + o.get("other_errors", 0))
        tally.count("opponent_unavailable_orders", o.get("unavailable_choices", 0))
        tally.count("opponent_exceptions", o.get("exceptions", 0))
        tally.count("opponent_forced_random", o.get("forced_random", 0))
        tally.count("opponent_fallbacks", getattr(opponent, "psbot_fallbacks", 0))
        for text in list(getattr(opponent, "audit_messages", []))[:3]:
            tally.notes.append(f"opponent: {text}")


async def _play_builtin_once(ctx: EvalContext, subject_kind: str, opponent: str, n: int) -> Tally:
    from play import build_player, server_configuration

    part = Tally(requested=n)
    subject = _make_subject(ctx, subject_kind, ctx.concurrency)
    other = None
    try:
        other = build_player(opponent, server=server_configuration(ctx.host, ctx.port), tag=secrets.token_hex(3) + "b",
                             battle_format=ctx.battle_format, concurrency=ctx.concurrency, audit=True)
        await _wait_logged_in([subject, other])
        for p in (subject, other):
            if getattr(p, "psbot_enabled", False) and hasattr(type(p), "psbot"):
                # Starts the PSBot engine (a Node process) now, not inside the first game.
                await asyncio.get_running_loop().run_in_executor(None, lambda p=p: p.psbot)
        t0 = time.time()
        try:
            # A game of two bots takes a second or two. A minute each, on top of ten, is far out.
            part.stop_reason = await _watch_match(subject.battle_against(other, n_battles=n), subject, ctx,
                                                  f"{subject_kind} vs {opponent}", n, ctx.stall_seconds, 600.0 + 60.0 * n)
        except Exception as e:
            # poke-env raises out of the match when a connection breaks. The games that finished
            # before that still count; whether the server is gone decides what happens next.
            gone = True
            for _ in range(6):
                if port_listening(ctx.host, ctx.port, timeout=1.0):
                    gone = False
                    break
                await asyncio.sleep(0.5)
            part.stop_reason = SERVER_GONE if gone else f"the match failed ({type(e).__name__}: {str(e)[:160]})"
            say(f"  the match stopped: {type(e).__name__}: {str(e)[:160]}")
        finally:
            part.seconds = time.time() - t0
            _collect_battles(part, subject)
            _collect_audit(part, subject, other)
            part.info["device"] = str(getattr(getattr(subject, "policy", None), "device", ctx.device))
    finally:
        await _disconnect([p for p in (subject, other) if p is not None])
        _release(subject)
    return part


async def play_builtin(ctx: EvalContext, subject_kind: str, opponent: str, n: int) -> Tally:
    """n games against one of the built-in opponents. Restarts the server once if it dies."""
    total = Tally(requested=n)
    remaining = n
    for attempt in (1, 2):
        part = await _play_builtin_once(ctx, subject_kind, opponent, remaining)
        total.absorb(part)
        if part.stop_reason != SERVER_GONE or attempt == 2:
            break
        remaining = n - total.games
        ok, message = ctx.server.ensure()
        total.notes.append(f"the server went away during the match; {message}")
        say(f"  {SERVER_GONE}: {message}")
        if not ok or remaining <= 0:
            break
        total.stop_reason = None
    return total


def load_foulplay(module_name: str = FOULPLAY_MODULE) -> Tuple[Optional[Any], str]:
    """
    The Foul Play runner, if it can be used: (module, version text) or (None, why not).
    The module's contract:  available() -> (bool, text)   and
    async run_matches(player, n_games, host, port, *, parallel, search_time_ms, log_dir) -> dict
    (and the keyword fixed_iterations, passed only when --foulplay-iterations asks for it)
    """
    import importlib

    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        return None, f"{module_name} cannot be imported ({type(e).__name__}: {str(e)[:160]})"
    if not callable(getattr(mod, "available", None)) or not callable(getattr(mod, "run_matches", None)):
        return None, f"{module_name} has no available() and run_matches()"
    try:
        ok, text = mod.available()
    except Exception as e:
        return None, f"{module_name}.available() failed ({type(e).__name__}: {str(e)[:160]})"
    if not ok:
        return None, str(text)
    return mod, str(text)


def foulplay_tally(result: Any, n: int) -> Tally:
    """The dict run_matches returned, checked and turned into a Tally."""
    tally = Tally(requested=n)
    if not isinstance(result, dict):
        tally.notes.append(f"run_matches returned {type(result).__name__}, not a dict")
        return tally

    def as_count(key: str) -> int:
        v = result.get(key)
        return max(0, int(v)) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else 0

    tally.wins, tally.losses, tally.ties = as_count("wins"), as_count("losses"), as_count("ties")
    claimed = as_count("games")
    if claimed != tally.games:
        tally.notes.append(f"run_matches reported games={claimed} but wins+losses+ties={tally.games}; the three counts are used")
    # Its own error count, but never fewer than the games that are unaccounted for.
    tally.errors_reported = max(as_count("errors"), n - tally.games)
    v = result.get("seconds")
    tally.seconds = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0 else 0.0
    core = {"games", "wins", "losses", "ties", "errors", "seconds"}
    tally.info = json.loads(json.dumps({k: v for k, v in result.items() if k not in core}, default=str))
    if result.get("error"):
        tally.notes.append(f"foulplay: {result['error']}")
    return tally


async def play_foulplay(ctx: EvalContext, n: int, make_subject: Optional[Callable[[], Any]] = None) -> Tuple[Optional[Tally], str]:
    """
    n games against Foul Play through its runner. (None, why) when it is not installed: that is
    a skip, not a failure. `make_subject` replaces the policy player (tests).
    """
    mod, text = load_foulplay(ctx.foulplay_module)
    if mod is None:
        return None, text
    parallel = max(1, min(ctx.foulplay_parallel, n))
    subject = make_subject() if make_subject is not None else _make_subject(ctx, "policy", max(parallel, ctx.concurrency))
    tally = Tally(requested=n)
    try:
        if make_subject is None:
            await _wait_logged_in([subject])
        before = set(getattr(subject, "battles", {}).keys())
        t0 = time.time()
        # Its runner has its own limits per game and overall. This one is only a backstop behind
        # them: a search bot that has not moved in five minutes is not going to.
        more: Dict[str, Any] = {}
        if ctx.foulplay_iterations > 0:
            more["fixed_iterations"] = int(ctx.foulplay_iterations)
        coro = mod.run_matches(subject, n, ctx.host, ctx.port, parallel=parallel,
                               search_time_ms=ctx.foulplay_search_ms, log_dir=ctx.foulplay_log_dir, **more)
        result: Dict[str, Any] = {}

        async def run() -> None:
            result["value"] = await coro

        stop = await _watch_match(run(), subject, ctx, "policy vs foulplay", n, max(300.0, ctx.stall_seconds),
                                  1800.0 + 400.0 * n / parallel)
        if stop is None:
            tally = foulplay_tally(result.get("value"), n)
        else:
            tally.stop_reason = stop
        if tally.seconds <= 0:
            tally.seconds = time.time() - t0
        tally.info["foulplay"] = tally.info.get("foulplay") or text
        if hasattr(subject, "battles"):
            _collect_battles(tally, subject, skip=before, count_results=False)
            # Games its runner had to stop are forfeited by our side. They are its errors already.
            tally.forfeits = min(tally.forfeits, tally.errors)
            tally.timeouts = min(tally.timeouts, tally.errors - tally.forfeits)
        _collect_audit(tally, subject)
        tally.info["device"] = str(getattr(getattr(subject, "policy", None), "device", ctx.device))
    finally:
        if make_subject is None:
            await _disconnect([subject])
            _release(subject)
    return tally, text


# =============================================================================================
# One evaluation
# =============================================================================================

def parse_opponents(text: Any) -> Tuple[List[str], List[str]]:
    """(known opponents in the order given, unknown names)."""
    items = text if isinstance(text, (list, tuple)) else str(text or "").replace(";", ",").split(",")
    good: List[str] = []
    bad: List[str] = []
    for item in items:
        name = str(item).strip().lower()
        if not name:
            continue
        if name in OPPONENTS:
            if name not in good:
                good.append(name)
        elif name not in bad:
            bad.append(name)
    return good, bad


def resolve_device(wanted: Optional[str]) -> str:
    wanted = (wanted or "auto").strip().lower()
    if wanted == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda" if wanted in ("auto", "cuda") else wanted
    except Exception:
        pass
    return "cpu"


def _arm_exit_watchdog(code: int, seconds: float = 20.0) -> None:
    """poke-env stops its event loop at exit and can wait there forever. Do not wait with it."""
    def force() -> None:
        time.sleep(seconds)
        os._exit(code)
    threading.Thread(target=force, daemon=True).start()


def _close_psbot_engine() -> None:
    try:
        bridge = sys.modules.get("psbot_bridge")
        engine = getattr(bridge, "_shared", None) if bridge else None
        if engine is not None:
            engine.close()
    except Exception:
        pass


async def _run_evaluation(args: argparse.Namespace, paths: RunPaths, ctx: EvalContext, meta: Dict[str, Any],
                          opponents: List[str], common: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    lines: List[Dict[str, Any]] = []
    code = EXIT_OK
    subjects = ["policy"] + (["hybrid"] if args.also_hybrid else [])
    for subject_kind in subjects:
        target = paths.results if subject_kind == "policy" else paths.hybrid_results
        for opponent in opponents:
            if subject_kind == "hybrid" and opponent == "foulplay":
                continue
            n = args.foulplay_games if opponent == "foulplay" else args.games
            if n <= 0:
                continue
            now = stat_ckpt(ctx.checkpoint.path)
            if now is None or now.mtime != ctx.checkpoint.mtime or now.size != ctx.checkpoint.size:
                say(f"{ctx.checkpoint.name} was replaced or removed during the evaluation. Stopping: the lines written so far stand.")
                return lines, EXIT_NO_CHECKPOINT
            say(f"{subject_kind} vs {opponent}: {n} games on {ctx.host}:{ctx.port}")
            tally: Optional[Tally]
            _gpu_peak_reset()
            try:
                ok, message = ctx.server.ensure()
                if not ok:
                    raise RuntimeError(message)
                if opponent == "foulplay":
                    tally, text = await play_foulplay(ctx, n)
                    if tally is None:
                        # Not played is not an error of the games, but it must not pass in silence
                        # either: a line with no game and the reason, so that the report has a
                        # Foul Play section that says why it is empty and status.py counts it.
                        say(f"  foulplay skipped: {text}")
                        tally = Tally(requested=n, errors_reported=0, notes=[f"foulplay was not played: {text}"])
                else:
                    tally = await play_builtin(ctx, subject_kind, opponent, n)
            except Exception as e:
                say(f"  failed: {type(e).__name__}: {e}")
                traceback.print_exc()
                tally = Tally(requested=n, notes=[f"failed: {type(e).__name__}: {str(e)[:200]}"])
            # The player read the file again when it was built, after the check above. If the trainer
            # wrote the same name in between (a later phase reaching this update number, the save on
            # a stop), the games were played by other weights than the ones this line would name.
            now = stat_ckpt(ctx.checkpoint.path)
            if now is None or now.mtime != ctx.checkpoint.mtime or now.size != ctx.checkpoint.size:
                say(f"{ctx.checkpoint.name} was replaced or removed while {opponent} was being played. That match is not "
                    f"written down; the lines written before it stand.")
                return lines, EXIT_NO_CHECKPOINT
            extras = dict(common)
            if opponent == "foulplay":
                if ctx.foulplay_iterations > 0:
                    # in every line of such an evaluation, also one where Foul Play could not be played
                    extras["foulplay_iterations"] = int(ctx.foulplay_iterations)
                slow = foulplay_slow_search(tally.info.get("foulplay_mean_iterations"), tally.info.get("search_time_ms"),
                                            read_jsonl(target), ctx.foulplay_iterations)
                if slow:
                    extras["foulplay_slow_search"] = slow
            extras["device"] = tally.info.pop("device", ctx.device)
            extras["gpu_peak_mb"] = _gpu_peak_mb() if str(extras["device"]).startswith("cuda") else None
            extras["concurrency"] = ctx.foulplay_parallel if opponent == "foulplay" else ctx.concurrency
            extras["inference"] = "batched" if (ctx.batch and subject_kind == "policy") else "single"
            extras["notes"] = list(ctx.notes)
            line = result_line(meta, opponent, tally, sample=ctx.sample, subject=subject_kind, extras=extras)
            append_jsonl(target, line)
            lines.append(line)
            games = line["games"]
            say(f"  {line['wins']}-{line['losses']}-{line['ties']} of {games}"
                + (f", {_pct(line['winrate'])} ({100 * line['ci_low']:.1f} to {100 * line['ci_high']:.1f})" if games else "")
                + f", {line['errors']} errors, {line['seconds']:.0f}s"
                + (f", {line['games_per_min']} games/min" if line.get("games_per_min") else "")
                + (f", {line['mean_turns']} turns/game" if line.get("mean_turns") else ""))
            for problem in line["problems"]:
                say(f"  problem: {problem}")
    return lines, code


def run_once(args: argparse.Namespace) -> int:
    code = EXIT_CRASH
    try:
        code = _run_once(args)
        return code
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        _arm_exit_watchdog(code)


def _run_once(args: argparse.Namespace) -> int:
    paths = RunPaths(args.run_dir)
    opponents, unknown = parse_opponents(args.opponents if args.opponents is not None else ",".join(BUILTIN_OPPONENTS))
    if unknown:
        say(f"unknown opponents: {', '.join(unknown)}. Known: {', '.join(OPPONENTS)}")
        return EXIT_USAGE
    if not opponents:
        say("no opponents given")
        return EXIT_USAGE
    games = args.games if args.games is not None else 200
    args.games = games
    args.foulplay_games = args.foulplay_games if args.foulplay_games is not None else games
    port = args.port if args.port is not None else 9000
    concurrency = args.concurrency if args.concurrency is not None else 32
    if games < 1 or concurrency < 1 or not (0 < port < 65536):
        say("--games and --concurrency must be at least 1, --port a valid port")
        return EXIT_USAGE

    try:
        import torch  # noqa: F401
        import poke_env  # noqa: F401
    except ImportError as e:
        say(f"this interpreter ({sys.executable}) cannot run an evaluation: {e}. Use the Python that has torch and poke-env.")
        return EXIT_CRASH

    # -- the checkpoint -----------------------------------------------------------------------
    notes: List[str] = []
    wanted = args.checkpoint or "newest"
    if wanted == "newest":
        loaded: Dict[str, Dict[str, Any]] = {}

        def loader(path: str) -> None:
            loaded[path] = load_checkpoint_meta(path)

        ckpt = newest_checkpoint(paths.checkpoints, settle_s=args.settle_seconds, loader=loader, fall_back=True, notes=notes)
        for n in notes:
            say(f"skipped: {n}")
        if ckpt is None:
            say(f"no usable checkpoint in {paths.checkpoints}")
            return EXIT_NO_CHECKPOINT
        ckpt_meta = loaded[ckpt.path]
    else:
        ckpt = stat_ckpt(wanted)
        if ckpt is None or not os.path.isfile(ckpt.path):
            say(f"checkpoint not found: {wanted}")
            return EXIT_NO_CHECKPOINT
        if args.expect_mtime is not None and abs(ckpt.mtime - args.expect_mtime) >= 0.01:
            say(f"{ckpt.name} is not the file that was chosen (it was written again since). Not evaluating it under the old label.")
            return EXIT_NO_CHECKPOINT
        try:
            ckpt_meta = load_checkpoint_meta(ckpt.path)
        except Exception as e:
            say(f"{ckpt.name} cannot be read ({type(e).__name__}: {str(e)[:160]}). It may still be being written.")
            return EXIT_NO_CHECKPOINT
        again = stat_ckpt(ckpt.path)
        if again is None or again.mtime != ckpt.mtime or again.size != ckpt.size:
            say(f"{ckpt.name} changed while it was being read. Try again in a moment.")
            return EXIT_NO_CHECKPOINT
    state = read_json(paths.state)
    meta = describe_checkpoint(ckpt, ckpt_meta, state, paths.metrics)

    # -- the server ---------------------------------------------------------------------------
    os.makedirs(paths.logs, exist_ok=True)
    server = EvalServer(args.server_dir, port, paths.logs, host=args.host)
    lines: List[Dict[str, Any]] = []
    started = args.started or utc_iso()

    def on_term(*_a: Any) -> None:
        raise KeyboardInterrupt()

    try:
        signal.signal(signal.SIGTERM, on_term)
    except (ValueError, OSError):
        pass
    try:
        ok, message = server.ensure()
        say(f"server: {message}")
        if not ok:
            return EXIT_NO_SERVER
        ctx = EvalContext(
            host=args.host, port=port, battle_format=args.format, concurrency=concurrency, checkpoint=ckpt,
            sample=bool(args.sample), device=resolve_device(args.device), batch=not args.single_inference,
            stall_seconds=float(args.stall_seconds), foulplay_parallel=max(1, int(args.foulplay_parallel)),
            foulplay_search_ms=max(1, int(args.foulplay_search_ms)), foulplay_module=args.foulplay_module,
            foulplay_log_dir=os.path.join(paths.logs, "foulplay"), server=server,
            foulplay_iterations=max(0, int(getattr(args, "foulplay_iterations", 0) or 0)),
        )
        if ckpt_meta.get("tensor_names"):
            ctx.tensor_names = frozenset(ckpt_meta["tensor_names"])
        common = {
            "port": port, "format": args.format, "reason": args.reason or "manual", "started": started,
            "state_phase": args.state_phase if args.state_phase is not None else state_phase(state),
            "eval_id": f"{started}-{os.getpid()}",
        }
        say(f"evaluating {ckpt.name} ({meta['phase']}, update {meta['update']}, {_num(meta['total_steps'])} decisions) "
            f"on {ctx.device}; the policy plays its {'sampled' if ctx.sample else 'best legal'} action")
        write_json_atomic(paths.in_progress, {"status": "running", "checkpoint": ckpt.name, "started": started,
                                              "pid": os.getpid(), "opponents": opponents})
        try:
            lines, code = asyncio.run(_run_evaluation(args, paths, ctx, meta, opponents, common))
        finally:
            write_json_atomic(paths.in_progress, {"status": "idle", "checkpoint": ckpt.name, "started": started,
                                                  "finished": utc_iso(), "pid": os.getpid(), "lines": len(lines)})
        if lines:
            try:
                write_report(paths)
            except Exception as e:
                say(f"could not write {paths.report}: {type(e).__name__}: {e}")
            print_table(lines, meta, ctx.sample)
            say(f"\nresults: {paths.results}\nreport:  {paths.report}")
        return code
    except KeyboardInterrupt:
        say("interrupted")
        return EXIT_CRASH
    finally:
        _close_psbot_engine()
        server.stop()


# =============================================================================================
# Watch mode
# =============================================================================================

EVAL_DEFAULTS: Dict[str, Any] = {
    "every_hours": 3.0,            # at most one evaluation this often
    "games": 200,                  # per opponent
    "opponents": list(BUILTIN_OPPONENTS),
    "port": 9000,
    "host": "127.0.0.1",
    "concurrency": 32,             # battles at once
    "device": "auto",              # auto, cuda or cpu. cuda falls back to cpu if it fails
    "foulplay": {
        "games": 100,              # 0 turns Foul Play off
        "every_hours": 12.0,       # it rides along with an evaluation when this much time has passed
        "parallel": 4,             # Foul Play processes, one core each
        "search_time_ms": 200,
    },
    "sample": False,
    "also_hybrid": False,
    "format": "gen9randombattle",
    "server_dir": None,            # default: the top-level "showdown_dir" of the config
    "python": None,                # default: the top-level "python" of the config, then this interpreter
    "poll_seconds": 60.0,
    "settle_seconds": 2.0,
    "stall_seconds": 90.0,
    "timeout_minutes": 150.0,      # one evaluation is stopped after this
    "retry_minutes": 10.0,         # wait this long after a failed evaluation
    "max_attempts": 3,             # failed evaluations of one checkpoint before it is given up
}


def _as_number(v: Any, lo: float, hi: float = float("inf")) -> Optional[float]:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not (lo <= v <= hi):
        return None
    return float(v)


def eval_config(raw: Any, overrides: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], List[str]]:
    """
    The "eval" section of the run config with every key filled in, and what was wrong with it.
    `raw` is the whole parsed config file (or None). A bad value is replaced by its default and
    reported; an unknown key is reported and ignored. `overrides` (command line) win.
    """
    warnings: List[str] = []
    cfg = json.loads(json.dumps(EVAL_DEFAULTS))
    section: Dict[str, Any] = {}
    if isinstance(raw, dict):
        if isinstance(raw.get("eval"), dict):
            # A key that starts with "_" is a comment, as everywhere else in the run config (pipeline.py).
            section = {k: v for k, v in raw["eval"].items() if not str(k).startswith("_")}
            if isinstance(section.get("foulplay"), dict):
                section["foulplay"] = {k: v for k, v in section["foulplay"].items() if not str(k).startswith("_")}
        elif raw.get("eval") is not None:
            warnings.append("\"eval\" in the config is not an object: using the defaults")
        if isinstance(raw.get("showdown_dir"), str) and raw["showdown_dir"]:
            cfg["server_dir"] = raw["showdown_dir"]
        if isinstance(raw.get("python"), str) and raw["python"]:
            cfg["python"] = raw["python"]
    elif raw is not None:
        warnings.append("the config is not a JSON object: using the defaults")
    for k, v in (overrides or {}).items():
        if v is None:
            continue
        if k == "foulplay" and isinstance(v, dict):
            merged = dict(section["foulplay"]) if isinstance(section.get("foulplay"), dict) else {}
            merged.update({fk: fv for fk, fv in v.items() if fv is not None})
            if merged:
                section["foulplay"] = merged
        else:
            section[k] = v

    def number(key: str, lo: float, hi: float = float("inf"), integer: bool = False, where: Optional[Dict[str, Any]] = None,
               src: Optional[Dict[str, Any]] = None, label: Optional[str] = None) -> None:
        src = section if src is None else src
        where = cfg if where is None else where
        if key not in src:
            return
        v = _as_number(src[key], lo, hi)
        if v is None or (integer and v != int(v)):
            warnings.append(f"eval.{label or key}={src[key]!r} is not usable: keeping {where[key]!r}")
            return
        where[key] = int(v) if integer else v

    number("every_hours", 0.0)
    number("games", 1, 100000, integer=True)
    number("port", 1, 65535, integer=True)
    number("concurrency", 1, 1024, integer=True)
    number("poll_seconds", 0.05, 86400)
    number("settle_seconds", 0.0, 600)
    number("stall_seconds", 5.0, 86400)
    number("timeout_minutes", 1.0, 100000)
    number("retry_minutes", 0.0, 100000)
    number("max_attempts", 1, 1000, integer=True)
    if "opponents" in section:
        good, bad = parse_opponents(section["opponents"])
        if bad:
            warnings.append(f"eval.opponents: unknown {', '.join(bad)} (known: {', '.join(OPPONENTS)})")
        if good:
            cfg["opponents"] = good
        else:
            warnings.append("eval.opponents names no known opponent: using the defaults")
    if "device" in section:
        dev = str(section["device"]).strip().lower()
        if dev in ("auto", "cpu") or dev.startswith("cuda"):
            cfg["device"] = dev
        else:
            warnings.append(f"eval.device={section['device']!r} is not auto, cuda or cpu: keeping {cfg['device']!r}")
    for key in ("sample", "also_hybrid"):
        if key in section:
            if isinstance(section[key], bool):
                cfg[key] = section[key]
            else:
                warnings.append(f"eval.{key}={section[key]!r} is not true or false: keeping {cfg[key]!r}")
    for key in ("host", "format", "server_dir", "python"):
        if key in section and section[key] is not None:
            if isinstance(section[key], str) and section[key].strip():
                cfg[key] = section[key].strip()
            else:
                warnings.append(f"eval.{key}={section[key]!r} is not text: keeping {cfg[key]!r}")
    fp = section.get("foulplay")
    if isinstance(fp, dict):
        number("games", 0, 100000, integer=True, where=cfg["foulplay"], src=fp, label="foulplay.games")
        number("every_hours", 0.0, where=cfg["foulplay"], src=fp, label="foulplay.every_hours")
        number("parallel", 1, 256, integer=True, where=cfg["foulplay"], src=fp, label="foulplay.parallel")
        number("search_time_ms", 1, 600000, integer=True, where=cfg["foulplay"], src=fp, label="foulplay.search_time_ms")
        if "iterations" in fp:
            # Not among the defaults: a config without it plays on Foul Play's search time, as always.
            v = _as_number(fp["iterations"], 0, 1e9)
            if v is None or v != int(v) or int(v) % FOULPLAY_ITERATION_BATCH:
                warnings.append(f"eval.foulplay.iterations={fp['iterations']!r} is not usable (a whole multiple of "
                                f"{FOULPLAY_ITERATION_BATCH:,}): Foul Play keeps its search time")
            elif int(v) > 0:
                cfg["foulplay"]["iterations"] = int(v)
        for key in fp:
            if key not in cfg["foulplay"] and key != "iterations":
                warnings.append(f"eval.foulplay.{key} is not a known key")
    elif fp is False:
        cfg["foulplay"]["games"] = 0
    elif fp is not None and fp is not True:
        warnings.append("eval.foulplay is not an object: using its defaults")
    for key in section:
        if key not in cfg:
            warnings.append(f"eval.{key} is not a known key")
    # Foul Play is its own block with its own pace: "foulplay" in the opponents list only says it is wanted.
    if "foulplay" in cfg["opponents"]:
        cfg["opponents"] = [o for o in cfg["opponents"] if o != "foulplay"] or list(BUILTIN_OPPONENTS)
        if cfg["foulplay"]["games"] <= 0:
            cfg["foulplay"]["games"] = EVAL_DEFAULTS["foulplay"]["games"]
    return cfg, warnings


@dataclass(frozen=True)
class Snapshot:
    """Everything the decision looks at, read at one moment."""
    now: float
    phase: Optional[str]                    # state.json, None when there is none
    newest: Optional[Ckpt]                  # newest settled checkpoint, None when there is none yet
    n_checkpoints: int                      # checkpoint files that exist, settled or not
    results: Tuple[Dict[str, Any], ...]     # results.jsonl
    final_done: Optional[Dict[str, Any]]    # parsed final.done, None when it does not exist
    failures: Dict[Tuple[str, float], Tuple[int, float]] = field(default_factory=dict)  # Ckpt.key -> (count, last time)


@dataclass(frozen=True)
class Plan:
    action: str                             # "evaluate" or "mark_final"
    reason: str                             # first, interval, phase_change, resume, final
    checkpoint: Optional[Ckpt]
    opponents: Tuple[str, ...] = ()
    final: bool = False                     # write final.done when it has run
    ok: bool = True
    note: str = ""


def final_covers(final_done: Any, ckpt: Optional[Ckpt]) -> bool:
    """Does the final.done on disk belong to this version of the newest checkpoint?"""
    if not isinstance(final_done, dict):
        return False
    if ckpt is None:
        return final_done.get("checkpoint") is None
    mt = final_done.get("checkpoint_mtime")
    return final_done.get("checkpoint") == ckpt.name and isinstance(mt, (int, float)) and abs(float(mt) - ckpt.mtime) < 0.01


def decide(snap: Snapshot, cfg: Dict[str, Any]) -> Tuple[Optional[Plan], str]:
    """
    Should an evaluation start now? Returns (plan or None, the reason in words). No side effects.

      * nothing to do without a settled checkpoint, or when the newest one already has a line
        for every opponent (in the configured play mode);
      * the phase is "done": evaluate the newest checkpoint against everything it has not played
        yet, Foul Play included, whatever the clock says, then final.done;
      * a checkpoint with lines for some opponents but not all (the evaluation was interrupted):
        finish it now;
      * otherwise evaluate when there is no evaluation yet, when the phase in state.json differs
        from the phase recorded with the last evaluation, or when every_hours have passed since
        the last evaluation started;
      * Foul Play rides along with an evaluation when foulplay.every_hours have passed since its
        last line (or it has none);
      * a checkpoint whose evaluation failed is retried after retry_minutes, max_attempts times,
        whatever the interval says: lines with no finished game do not count as an evaluation.
    """
    base = [o for o in cfg["opponents"] if o != "foulplay"]
    fp = cfg["foulplay"]
    fp_on = int(fp["games"]) > 0
    sample = bool(cfg["sample"])
    done_phase = snap.phase == "done"
    final_done = snap.final_done if isinstance(snap.final_done, dict) else None

    if snap.newest is None:
        if snap.n_checkpoints > 0:
            return None, "the newest checkpoint is still being written"
        if done_phase and final_done is None:
            return Plan("mark_final", "final", None, final=True, ok=False,
                        note="the run is done and there is no checkpoint to evaluate"), "the run is done with no checkpoint"
        return None, "no checkpoint yet"

    ck = snap.newest
    done = opponents_done(snap.results, ck, sample)
    missing = [o for o in base if o not in done]
    final_needed = done_phase and not final_covers(final_done, ck)
    if done_phase and not final_needed:
        return None, "the run is done and its final evaluation is written"

    count, last_fail = snap.failures.get(ck.key, (0, 0.0))
    if count >= int(cfg["max_attempts"]):
        if final_needed:
            return Plan("mark_final", "final", ck, final=True, ok=False,
                        note=f"the final evaluation of {ck.name} failed {count} times"), "giving up on the final evaluation"
        return None, f"gave up on {ck.name} after {count} failed attempts; waiting for a newer checkpoint"
    if count and snap.now - last_fail < float(cfg["retry_minutes"]) * 60.0:
        wait = float(cfg["retry_minutes"]) * 60.0 - (snap.now - last_fail)
        return None, f"attempt {count} on {ck.name} failed; trying again in {duration(wait)}"

    if final_needed:
        opps = missing + (["foulplay"] if fp_on and "foulplay" not in done else [])
        if not opps:
            return Plan("mark_final", "final", ck, final=True), f"the run is done and {ck.name} is already evaluated"
        return Plan("evaluate", "final", ck, tuple(opps), final=True), f"the run is done: final evaluation of {ck.name}"

    if not missing:
        return None, f"{ck.name} is evaluated; waiting for a newer checkpoint"

    # "The last evaluation" is the last one that measured something. One in which no game finished
    # (the server would not start, the logins timed out) leaves lines too, with games 0: taking
    # their time would put the next attempt a whole every_hours away instead of retry_minutes.
    played = [r for r in snap.results if int(r.get("games") or 0) > 0]
    times = [t for t in (line_time(r) for r in played) if t is not None]
    last = max(times) if times else None
    fp_times = [t for t in (line_time(r) for r in snap.results if r.get("opponent") == "foulplay") if t is not None]
    fp_due = fp_on and "foulplay" not in done and (not fp_times or snap.now - max(fp_times) >= float(fp["every_hours"]) * 3600.0)
    last_phase = None
    if played:
        newest_line = max(played, key=lambda r: (line_time(r) or 0.0))
        last_phase = newest_line.get("state_phase")

    if any(o in done for o in base):
        reason, why = "resume", f"finishing the interrupted evaluation of {ck.name}"
    elif last is None:
        reason, why = "first", f"first evaluation: {ck.name}"
    elif snap.phase and isinstance(last_phase, str) and last_phase and snap.phase != last_phase:
        reason, why = "phase_change", f"the phase changed from {last_phase} to {snap.phase}: evaluating {ck.name}"
    elif snap.now - last >= float(cfg["every_hours"]) * 3600.0:
        reason, why = "interval", f"{duration(snap.now - last)} since the last evaluation: evaluating {ck.name}"
    else:
        wait = float(cfg["every_hours"]) * 3600.0 - (snap.now - last)
        return None, f"{ck.name} is next, in {duration(wait)}"
    return Plan("evaluate", reason, ck, tuple(missing + (["foulplay"] if fp_due else []))), why


class Watcher:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.paths = RunPaths(args.run_dir)
        self.failures: Dict[Tuple[str, float], Tuple[int, float]] = {}
        self.stop = threading.Event()
        self.child: Optional[subprocess.Popen] = None
        self.server: Optional[EvalServer] = None
        self.last_idle = ""
        self.last_idle_at = 0.0
        self.last_warnings: Tuple[str, ...] = ()
        self.status = "starting"
        self.started = utc_iso()
        self.evaluations = 0

    # -- output -------------------------------------------------------------------------------
    def log(self, text: str) -> None:
        line = f"{utc_iso()} {ascii_text(text)}"
        print(line, flush=True)
        if os.path.isdir(self.paths.run):
            try:
                os.makedirs(self.paths.logs, exist_ok=True)
                with open(os.path.join(self.paths.logs, "watch.log"), "a", encoding="utf-8", newline="\n") as f:
                    f.write(line + "\n")
            except OSError:
                pass

    def idle(self, why: str) -> None:
        """Says why nothing is happening, when that changes and every half hour."""
        self.status = why
        steady = re.sub(r"\d+(\.\d+)? (s|min|h)\b", "a while", why)
        if steady != self.last_idle or time.time() - self.last_idle_at >= 1800.0:
            self.last_idle, self.last_idle_at = steady, time.time()
            self.log(f"waiting: {why}")

    def heartbeat(self) -> None:
        if not os.path.isdir(self.paths.run):
            return
        try:
            write_json_atomic(self.paths.watch_state, {
                "pid": os.getpid(), "host": socket.gethostname(), "started": self.started, "heartbeat": utc_iso(),
                "heartbeat_ts": round(time.time(), 1), "status": self.status, "evaluations": self.evaluations,
                # status.py calls the heartbeat stale after five minutes, or three polls if that is longer
                "poll_seconds": getattr(self, "poll", None),
            })
        except OSError:
            pass

    def other_watcher(self) -> Optional[str]:
        """
        Another watcher on this run directory, if one is running: its heartbeat is fresh, it did
        not say it stopped, and its process is still there. A process id alone is weak evidence
        (after a reboot the same number belongs to something else within seconds), so where the
        system can say what the process is, it must be an evaluate.py.
        """
        st = read_json(self.paths.watch_state)
        if not isinstance(st, dict) or st.get("pid") == os.getpid() or st.get("host") not in (None, socket.gethostname()):
            return None
        if st.get("status") == "stopped":
            return None
        beat = st.get("heartbeat_ts")
        fresh = isinstance(beat, (int, float)) and time.time() - float(beat) < 600.0
        if fresh and pid_alive(st.get("pid")) and pid_command_line(st.get("pid"), "evaluate.py") is not False:
            return f"pid {st.get('pid')}, last heard {st.get('heartbeat')}"
        return None

    # -- config -------------------------------------------------------------------------------
    def config(self) -> Dict[str, Any]:
        a = self.args
        path = a.config
        if not path:
            state = read_json(self.paths.state)
            if isinstance(state, dict) and isinstance(state.get("config"), str):
                path = state["config"]
        raw = None
        problems: List[str] = []
        if path:
            raw = read_json(path)
            if raw is None:
                problems.append(f"the config {path} is missing or not JSON: using the defaults")
        overrides = {
            "every_hours": a.every_hours, "games": a.games, "port": a.port, "concurrency": a.concurrency,
            "device": a.device, "opponents": a.opponents, "server_dir": a.server_dir, "poll_seconds": a.poll_seconds,
            "host": a.host_override, "format": a.format_override, "stall_seconds": a.stall_override,
            "settle_seconds": a.settle_override, "python": a.python,
            "sample": True if a.sample else None, "also_hybrid": True if a.also_hybrid else None,
            "foulplay": {"games": a.foulplay_games, "parallel": a.foulplay_parallel_override,
                         "search_time_ms": a.foulplay_search_override,
                         "iterations": getattr(a, "foulplay_iterations_override", None)},
        }
        cfg, warnings = eval_config(raw, overrides)
        warnings = problems + warnings
        if tuple(warnings) != self.last_warnings:
            self.last_warnings = tuple(warnings)
            for w in warnings:
                self.log(f"config: {w}")
        return cfg

    # -- one look at the run ------------------------------------------------------------------
    def snapshot(self, cfg: Dict[str, Any]) -> Snapshot:
        phase = state_phase(read_json(self.paths.state))
        files = list_checkpoints(self.paths.checkpoints)
        newest = newest_checkpoint(self.paths.checkpoints, settle_s=float(cfg["settle_seconds"]), fall_back=False)
        return Snapshot(
            now=time.time(), phase=phase, newest=newest, n_checkpoints=len(files),
            results=tuple(read_jsonl(self.paths.results)),
            final_done=read_json(self.paths.final_done) if os.path.exists(self.paths.final_done) else None,
            failures=dict(self.failures),
        )

    def write_final(self, plan: Plan, snap: Snapshot) -> None:
        ck = plan.checkpoint
        lines = [r for r in read_jsonl(self.paths.results) if ck is not None and same_version(r, ck)]
        write_json_atomic(self.paths.final_done, {
            "time": utc_iso(),
            "ok": bool(plan.ok),
            "note": plan.note,
            "checkpoint": ck.name if ck else None,
            "checkpoint_mtime": round(ck.mtime, 3) if ck else None,
            "phase": snap.phase,
            "results": [{k: r.get(k) for k in ("opponent", "games", "wins", "losses", "ties", "errors", "winrate",
                                               "ci_low", "ci_high", "valid")} for r in lines],
        })
        self.log(f"wrote {self.paths.final_done}" + (f" ({plan.note})" if plan.note else ""))

    # -- running one evaluation ---------------------------------------------------------------
    def command(self, plan: Plan, cfg: Dict[str, Any], snap: Snapshot) -> List[str]:
        ck = plan.checkpoint
        assert ck is not None
        fp = cfg["foulplay"]
        python = cfg["python"] or sys.executable
        if os.path.exists(python):
            python = os.path.abspath(python)
        cmd = [
            python, str(REPO_ROOT / "evaluate.py"),
            "--run-dir", self.paths.run,
            "--checkpoint", ck.path, "--expect-mtime", repr(ck.mtime),
            "--opponents", ",".join(plan.opponents),
            "--games", str(cfg["games"]), "--port", str(cfg["port"]), "--host", str(cfg["host"]),
            "--concurrency", str(cfg["concurrency"]), "--device", str(cfg["device"]), "--format", str(cfg["format"]),
            "--stall-seconds", str(cfg["stall_seconds"]), "--settle-seconds", str(cfg["settle_seconds"]),
            "--foulplay-games", str(fp["games"]), "--foulplay-parallel", str(fp["parallel"]),
            "--foulplay-search-ms", str(fp["search_time_ms"]), "--foulplay-module", self.args.foulplay_module,
            "--reason", plan.reason, "--started", utc_iso(snap.now),
        ]
        if snap.phase:
            cmd += ["--state-phase", snap.phase]
        if cfg["server_dir"]:
            cmd += ["--server-dir", os.path.abspath(str(cfg["server_dir"]))]
        if cfg["sample"]:
            cmd.append("--sample")
        if cfg["also_hybrid"]:
            cmd.append("--also-hybrid")
        if self.args.single_inference:
            cmd.append("--single-inference")
        if int(fp.get("iterations") or 0) > 0:
            cmd += ["--foulplay-iterations", str(int(fp["iterations"]))]
        return cmd

    def run_child(self, plan: Plan, cfg: Dict[str, Any], snap: Snapshot) -> int:
        ck = plan.checkpoint
        assert ck is not None
        os.makedirs(self.paths.logs, exist_ok=True)
        log_path = os.path.join(self.paths.logs, f"eval_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(snap.now))}_{ck.name[:-3]}.log")
        cmd = self.command(plan, cfg, snap)
        env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
        timeout = float(cfg["timeout_minutes"]) * 60.0
        t0 = time.time()
        with open(log_path, "ab") as out:
            out.write((" ".join(cmd) + "\n").encode("utf-8", "replace"))
            out.flush()
            self.child = start_process(cmd, cwd=str(REPO_ROOT), stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, env=env)
            self.log(f"evaluation started (pid {self.child.pid}, log {log_path})")
            try:
                while self.child.poll() is None:
                    if self.stop.is_set():
                        self.log("stopping: ending the evaluation in progress")
                        kill_tree(self.child)
                        return -1
                    if time.time() - t0 > timeout:
                        self.log(f"the evaluation ran past timeout_minutes={cfg['timeout_minutes']:g}: stopping it")
                        kill_tree(self.child)
                        return -2
                    self.heartbeat()
                    self.stop.wait(1.0)
                return int(self.child.returncode)
            finally:
                kill_tree(self.child)
                self.child = None

    def step(self) -> None:
        cfg = self.config()
        self.poll = float(cfg["poll_seconds"])
        if not os.path.isdir(self.paths.run):
            self.idle(f"the run directory {self.paths.run} does not exist yet")
            return
        snap = self.snapshot(cfg)
        plan, why = decide(snap, cfg)
        if plan is None:
            self.idle(why)
            return
        if plan.action == "mark_final":
            self.log(why)
            self.write_final(plan, snap)
            return
        ck = plan.checkpoint
        assert ck is not None
        self.status = f"evaluating {ck.name} against {', '.join(plan.opponents)} ({plan.reason})"
        self.last_idle = ""
        self.log(f"{why}; opponents: {', '.join(plan.opponents)}")
        self.heartbeat()
        self.server = EvalServer(cfg["server_dir"], int(cfg["port"]), self.paths.logs, host=str(cfg["host"]))
        before = len(snap.results)
        t0 = time.time()
        code = EXIT_CRASH
        try:
            ok, message = self.server.ensure()
            self.log(f"server: {message}")
            code = self.run_child(plan, cfg, snap) if ok else EXIT_NO_SERVER
        except Exception:
            self.log("could not run the evaluation:\n" + traceback.format_exc())
        finally:
            self.server.stop()
            self.server = None
        if code == -1:
            return
        new = read_jsonl(self.paths.results)[before:]
        if new:
            try:  # the evaluation writes it too; this covers one that was cut short after some lines
                write_report(self.paths)
            except Exception as e:
                self.log(f"could not write {self.paths.report}: {type(e).__name__}: {e}")
        for r in new:
            if int(r.get("games") or 0) > 0:
                self.log(f"  {r.get('opponent')}: {_pct(r['winrate'])} ({100 * r['ci_low']:.1f} to {100 * r['ci_high']:.1f}), "
                         f"{r.get('wins')}-{r.get('losses')}-{r.get('ties')} of {r.get('games')}, {r.get('errors')} errors"
                         + ("" if r.get("valid", True) else "  NOT TRUSTED: " + "; ".join(r.get("problems") or [])))
            else:
                self.log(f"  {r.get('opponent')}: no game finished ({'; '.join(r.get('notes') or r.get('problems') or [])})")
        have = opponents_done(read_jsonl(self.paths.results), ck, bool(cfg["sample"]))
        left = [o for o in plan.opponents if o != "foulplay" and o not in have]
        if code == EXIT_OK and not left:
            self.evaluations += 1
            self.failures.pop(ck.key, None)
            self.log(f"evaluation of {ck.name} finished in {time.time() - t0:.0f}s, {len(new)} lines")
            if plan.final:
                self.write_final(plan, snap)
        elif code == EXIT_OK:
            count = self.failures.get(ck.key, (0, 0.0))[0] + 1
            self.failures[ck.key] = (count, time.time())
            self.log(f"evaluation of {ck.name} ran, but no game finished against {', '.join(left)}; "
                     f"attempt {count} of {cfg['max_attempts']}")
        else:
            count = self.failures.get(ck.key, (0, 0.0))[0] + 1
            self.failures[ck.key] = (count, time.time())
            what = {EXIT_NO_CHECKPOINT: "the checkpoint could not be used", EXIT_NO_SERVER: "no server",
                    -2: "it ran past the time limit"}.get(code, f"exit code {code}")
            self.log(f"evaluation of {ck.name} failed ({what}); attempt {count} of {cfg['max_attempts']}")

    def run(self) -> int:
        other = None if self.args.force else self.other_watcher()
        if other:
            say(f"another watcher is running on {self.paths.run} ({other}). Stop it first, or use --force.")
            return EXIT_LOCKED

        def on_signal(*_a: Any) -> None:
            self.stop.set()

        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            if hasattr(signal, name):
                try:
                    signal.signal(getattr(signal, name), on_signal)
                except (ValueError, OSError):
                    pass
        self.poll = 60.0
        self.log(f"watching {self.paths.run} (pid {os.getpid()})")
        try:
            while not self.stop.is_set():
                try:
                    self.step()
                except Exception:
                    self.log("the watch loop hit an error and carries on:\n" + traceback.format_exc())
                self.heartbeat()
                if self.args.once:
                    break
                self.stop.wait(max(0.05, self.poll))
        finally:
            kill_tree(self.child)
            if self.server is not None:
                self.server.stop()
            self.status = "stopped"
            self.heartbeat()
            self.log("watcher stopped")
        return EXIT_OK


# =============================================================================================
# Command line
# =============================================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="the run directory (checkpoints/, state.json, eval/)")
    ap.add_argument("--checkpoint", default=None, help="a checkpoint file, or 'newest' (default): the newest usable file in RUN/checkpoints")
    ap.add_argument("--opponents", default=None, help=f"comma separated, from: {', '.join(OPPONENTS)} (default: {', '.join(BUILTIN_OPPONENTS)})")
    ap.add_argument("--games", type=int, default=None, help="games per opponent (default 200)")
    ap.add_argument("--port", type=int, default=None, help="port of the evaluation server (default 9000). Never a training port")
    ap.add_argument("--host", dest="host_override", default=None, help="default 127.0.0.1")
    ap.add_argument("--server-dir", default=None, help="Pokemon Showdown checkout; a server is started from it when nothing listens on --port")
    ap.add_argument("--concurrency", type=int, default=None, help="battles at the same time (default 32)")
    ap.add_argument("--device", default=None, help="auto (default), cuda or cpu; cuda falls back to cpu when it fails")
    ap.add_argument("--sample", action="store_true", help="sample the policy's action instead of taking the best one")
    ap.add_argument("--also-hybrid", action="store_true", help="also play the PSBot-guarded hybrid; its lines go to eval/hybrid_results.jsonl")
    ap.add_argument("--single-inference", action="store_true", help="run the network once per decision instead of once for all waiting battles")
    ap.add_argument("--format", dest="format_override", default=None, help="default gen9randombattle")
    ap.add_argument("--foulplay-games", type=int, default=None, help="games against foulplay (default: --games; in watch mode the config)")
    ap.add_argument("--foulplay-parallel", dest="foulplay_parallel_override", type=int, default=None, help="Foul Play processes (default 4)")
    ap.add_argument("--foulplay-search-ms", dest="foulplay_search_override", type=int, default=None, help="Foul Play search time per decision (default 200)")
    ap.add_argument("--foulplay-iterations", dest="foulplay_iterations_override", type=int, default=None, metavar="N",
                    help="Foul Play searches every sampled world for exactly N iterations, whatever the clock says, instead of "
                         "for its search time: the same opponent on a loaded box and an idle one (bench/notes/FIXED_ITERATIONS.md). "
                         "Every result line says N, and the report never compares or pools lines of different N. "
                         "Default: off (its search time); in watch mode eval.foulplay.iterations")
    ap.add_argument("--foulplay-module", default=FOULPLAY_MODULE, help=argparse.SUPPRESS)
    ap.add_argument("--stall-seconds", dest="stall_override", type=float, default=None, help="give up on a match that has not moved for this long (default 90)")
    ap.add_argument("--settle-seconds", dest="settle_override", type=float, default=None, help="a checkpoint must be unchanged this long to be used (default 2)")
    ap.add_argument("--report-only", action="store_true", help="rewrite RUN/eval/REPORT.md from results.jsonl and exit")
    ap.add_argument("--check-config", metavar="CONFIG", default=None,
                    help="print what watch mode would read from this run config and what is wrong with its \"eval\" section, "
                         "then exit: 0 when nothing is wrong, 2 otherwise. Starts nothing")
    g = ap.add_argument_group("watch mode")
    g.add_argument("--watch", action="store_true", help="evaluate the run as it trains, forever")
    g.add_argument("--config", default=None, help="the run config (JSON); its \"eval\" section is read. Default: the config named in RUN/state.json")
    g.add_argument("--every-hours", type=float, default=None, help="overrides eval.every_hours")
    g.add_argument("--poll-seconds", type=float, default=None, help="overrides eval.poll_seconds")
    g.add_argument("--python", default=None, help="interpreter for the evaluations (default: the config's \"python\", then this one)")
    g.add_argument("--once", action="store_true", help="look once, evaluate if it is time, and exit")
    g.add_argument("--force", action="store_true", help="start even when another watcher seems to be running on this run directory")
    # set by the watcher on the evaluations it starts
    ap.add_argument("--expect-mtime", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--reason", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--started", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--state-phase", default=None, help=argparse.SUPPRESS)
    return ap


def check_config(path: str) -> int:
    """
    What watch mode would do with this run config, in words, and every complaint about its "eval"
    section. In watch mode an unknown key or a bad value is logged once and the default is used:
    this is the place to find a typo before the run starts.
    """
    raw = read_json(path)
    if raw is None:
        say(f"{path} is missing or not JSON")
        return EXIT_USAGE
    cfg, warnings = eval_config(raw, {})
    fp = cfg["foulplay"]
    say(f"evaluator settings from {path}:")
    say(f"  an evaluation at most every {cfg['every_hours']:g} h (and at every change of phase, and when the run is done): "
        f"{cfg['games']} games against each of {', '.join(cfg['opponents'])}")
    say("  Foul Play: " + (f"{fp['games']} games every {fp['every_hours']:g} h, {fp['parallel']} processes, {fp['search_time_ms']} ms a decision"
                           + (f", every sampled world searched exactly {fp['iterations']:,} iterations" if fp.get("iterations") else "")
                           if fp["games"] > 0 else "off"))
    say(f"  its own server on {cfg['host']}:{cfg['port']}, started from {cfg['server_dir'] or '(no server directory: a server must already be listening)'}")
    say(f"  {cfg['concurrency']} battles at once, device {cfg['device']}, the policy plays its {'sampled' if cfg['sample'] else 'best legal'} action"
        + (", the hybrid is played as well" if cfg["also_hybrid"] else ""))
    say(f"  interpreter {cfg['python'] or sys.executable}; looks every {cfg['poll_seconds']:g} s; one evaluation may take {cfg['timeout_minutes']:g} min")
    for w in warnings:
        say(f"WARNING: {w}")
    return EXIT_OK if not warnings else EXIT_USAGE


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check_config:
        return check_config(args.check_config)
    if args.report_only:
        paths = RunPaths(args.run_dir)
        write_report(paths)
        say(f"wrote {paths.report}")
        return EXIT_OK
    if args.watch:
        return Watcher(args).run()
    # one evaluation: fill in what watch mode would take from the config
    args.host = args.host_override or "127.0.0.1"
    args.format = args.format_override or "gen9randombattle"
    args.stall_seconds = args.stall_override if args.stall_override is not None else EVAL_DEFAULTS["stall_seconds"]
    args.settle_seconds = args.settle_override if args.settle_override is not None else EVAL_DEFAULTS["settle_seconds"]
    args.foulplay_parallel = args.foulplay_parallel_override if args.foulplay_parallel_override is not None else EVAL_DEFAULTS["foulplay"]["parallel"]
    args.foulplay_search_ms = args.foulplay_search_override if args.foulplay_search_override is not None else EVAL_DEFAULTS["foulplay"]["search_time_ms"]
    if args.foulplay_iterations_override is not None and (args.foulplay_iterations_override < 0
                                                          or args.foulplay_iterations_override % FOULPLAY_ITERATION_BATCH):
        say(f"--foulplay-iterations must be 0 (off) or a multiple of {FOULPLAY_ITERATION_BATCH:,}: poke-engine searches in "
            f"batches of {FOULPLAY_ITERATION_BATCH:,} and would round anything else up")
        return EXIT_USAGE
    args.foulplay_iterations = int(args.foulplay_iterations_override or 0)
    return run_once(args)


if __name__ == "__main__":
    sys.exit(main())
