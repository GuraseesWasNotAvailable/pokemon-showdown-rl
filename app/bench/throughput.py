#!/usr/bin/env python
"""
Measures how fast a training run really goes, and where the CPU and the GPU go while it does.

It runs train.py with the arguments after `--`, throws away the first --warmup-minutes (the
first minutes of a run are faster than what follows), measures for --minutes, and prints and
writes JSON with:

  * decisions a second, from the trainer's own telemetry (TRAIN total_steps), plus inference
    requests a second and the mean inference batch
  * learner updates a minute, seconds per update cycle, and an estimate of how long the learner
    is busy per update
  * GPU utilisation, memory and power, sampled from nvidia-smi
  * CPU seconds per wall second by role: Showdown main process, Showdown simulator processes,
    Showdown sockets processes, other Showdown children, rollout workers, inference actor,
    learner actor, driver, raylet and friends, PSBot engines. Also the busiest single process
    of each role (a rollout worker near 1.0 is pinned on one Python thread) and CPU
    milliseconds per decision.

Servers: `--servers start` starts its own through start_servers.py on --ports and stops them
afterwards; `--servers use` measures servers that are already running on --ports and never
signals them. It stops only processes it started, by PID, and their descendants.

    python bench/throughput.py --servers start --ps-dir /home/ubuntu/pokemon-showdown --ports 8000-8015 \
        --warmup-minutes 5 --minutes 10 --out /home/ubuntu/runs/bench/base.json -- \
        --mode ppo --battles 512 --rooms-per-pair 16 --ckpt-dir /home/ubuntu/runs/bench/ckpt

Give train.py a throwaway --ckpt-dir holding a COPY of the checkpoint to measure from: the run
writes checkpoints there, and speed depends on how the policy plays. Do not pass --minutes
after `--`: this script sets it (warmup + minutes + a margin) so the trainer stops by itself
and saves on the way out.

Other modes:
    --parse-log FILE     telemetry numbers from a saved train.py output, nothing is started
    --probe-servers      list the Showdown processes on --ports with their roles and CPU, read-only
    --self-test          checks of the parsers, the role classifier and the accounting
    --dry-run            print the commands and exit

Works on Linux and Windows. Needs psutil (it comes with ray).
"""
from __future__ import annotations

import argparse
import ast
import datetime as _dt
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BENCH_DIR)
IS_WINDOWS = os.name == "nt"

# --------------------------------------------------------------------------- small parsers


def parse_ports(spec: str) -> List[int]:
    """Same grammar as train.py and start_servers.py: 8000-8003 or 8000,8001,8005."""
    out: List[int] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_ASCTIME = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")
_TELEMETRY = re.compile(r"\[Telemetry\]\s*ACTIVE:\s*(-?\d+)\s*\|\s*INFER:\s*(\{.*?\})\s*\|\s*TRAIN:\s*(\{.*?\})\s*$")
_LEARNER = re.compile(r"\[learner\]\s+upd=(\d+)\s+(.*)$")
_KV = re.compile(r"([A-Za-z_]+)=([-+0-9.eE]+|nan|inf|-inf)")
_PID_PREFIX = re.compile(r"\((\w+)[^)]*pid=(\d+)[^)]*\)")
# The start of a line of train.py's own logger. Two writers share the trainer's stdout, and now
# and then the logger's line lands between the learner's text and its newline:
#   ... n_mb=128 sec=41.92026-10-05 09:48:48,489 [INFO] [Telemetry] ...
_LOG_STAMP = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \[[A-Z]+\] ")


def strip_ansi(line: str) -> str:
    return _ANSI.sub("", line)


def parse_asctime(line: str) -> Optional[float]:
    """Seconds since the epoch (local time) from a logging line of train.py, or None."""
    m = _ASCTIME.match(line)
    if not m:
        return None
    try:
        t = _dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return t.timestamp() + int(m.group(2)) / 1000.0


def parse_telemetry(line: str) -> Optional[Dict[str, Any]]:
    """
    One "[Telemetry] ACTIVE: n | INFER: {...} | TRAIN: {...}" line of train.py, or None.
    The two dicts are printed with str(dict), so they are read back with literal_eval.
    """
    line = strip_ansi(line).rstrip()
    m = _TELEMETRY.search(line)
    if not m:
        return None
    try:
        infer = ast.literal_eval(m.group(2))
        train = ast.literal_eval(m.group(3))
    except (ValueError, SyntaxError):
        return None
    if not isinstance(infer, dict) or not isinstance(train, dict) or "total_steps" not in train:
        return None
    return {
        "active": int(m.group(1)),
        "total_requests": int(infer.get("total_requests", 0) or 0),
        "avg_batch_size": float(infer.get("avg_batch_size", 0.0) or 0.0),
        "model_version": int(infer.get("model_version", -1)),
        "update": int(train.get("update", 0) or 0),
        "episodes": int(train.get("episodes", 0) or 0),
        "steps_in_dataset": int(train.get("steps_in_dataset", 0) or 0),
        "total_steps": int(train.get("total_steps", 0) or 0),
        "log_time": parse_asctime(line),
    }


def parse_learner(line: str) -> Optional[Dict[str, Any]]:
    """One "[learner] upd=N kl=... n_mb=..." line (Ray prefixes it with the actor and pid), or None."""
    line = strip_ansi(line).rstrip()
    m = _LEARNER.search(line)
    if not m:
        return None
    out: Dict[str, Any] = {"upd": int(m.group(1))}
    rest = m.group(2)
    glued = _LOG_STAMP.search(rest)
    if glued:
        rest = rest[:glued.start()]         # the last number would otherwise run into the timestamp
    for k, v in _KV.findall(rest):
        try:
            out[k] = float(v)
        except ValueError:
            pass
    pm = _PID_PREFIX.search(line)
    if pm:
        out["actor"] = pm.group(1)
        out["pid"] = int(pm.group(2))
    return out


# --------------------------------------------------------------------------- telemetry maths


def _round(x: Optional[float], n: int = 2) -> Optional[float]:
    return None if x is None else round(float(x), n)


def _rate(a: Dict[str, Any], b: Dict[str, Any], key: str) -> Optional[float]:
    dt = b["t"] - a["t"]
    if dt <= 0:
        return None
    return (b[key] - a[key]) / dt


def telemetry_metrics(samples: Sequence[Dict[str, Any]], learner: Sequence[Dict[str, Any]],
                      w0: float, w1: float, period: float = 5.0) -> Dict[str, Any]:
    """
    samples: telemetry dicts with "t" (seconds, any common clock). learner: learner dicts with "t".
    Window [w0, w1] on the same clock.

    total_steps only moves when the learner is free to ingest: during an update the workers'
    episodes wait, and they arrive in a burst when it ends. So the rate between two arbitrary
    lines is off by up to one update's worth of steps. Lines printed right after an update
    finished (the update counter went up) are all at the same point of that cycle, so the rate
    between the first and the last of those is the honest one. It is used when the window holds
    at least three of them, the plain end points otherwise.
    """
    inside = [s for s in samples if w0 <= s["t"] <= w1]
    out: Dict[str, Any] = {"samples_in_window": len(inside)}
    if len(inside) < 2:
        out["error"] = "fewer than two telemetry lines in the window"
        return out
    first, last = inside[0], inside[-1]
    span = last["t"] - first["t"]
    out["span_seconds"] = round(span, 2)
    out["decisions_per_second_endpoints"] = _round(_rate(first, last, "total_steps"))
    out["inference_requests_per_second"] = _round(_rate(first, last, "total_requests"))

    # lines where the update counter rose since the previous line (the previous may be before w0)
    by_t = sorted(samples, key=lambda s: s["t"])
    aligned = []
    prev = None
    for s in by_t:
        if prev is not None and s["update"] > prev["update"] and w0 <= s["t"] <= w1:
            aligned.append(s)
        prev = s
    out["update_completions_seen"] = len(aligned)
    if len(aligned) >= 3:
        a, b = aligned[0], aligned[-1]
        out["decisions_per_second"] = _round(_rate(a, b, "total_steps"))
        out["decisions_per_second_method"] = "between the first and last line that follows a finished update"
        out["aligned_span_seconds"] = round(b["t"] - a["t"], 2)
        ups = b["update"] - a["update"]
        if ups > 0 and b["t"] > a["t"]:
            out["updates_per_minute"] = _round(ups / (b["t"] - a["t"]) * 60.0)
            out["seconds_per_update_cycle"] = _round((b["t"] - a["t"]) / ups)
            out["decisions_per_update"] = _round((b["total_steps"] - a["total_steps"]) / ups, 1)
    else:
        out["decisions_per_second"] = out["decisions_per_second_endpoints"]
        out["decisions_per_second_method"] = "window end points (fewer than three finished updates in the window)"
        ups = last["update"] - first["update"]
        if ups > 0 and span > 0:
            out["updates_per_minute"] = _round(ups / span * 60.0)
            out["seconds_per_update_cycle"] = _round(span / ups)
            out["decisions_per_update"] = _round((last["total_steps"] - first["total_steps"]) / ups, 1)
    out["updates_in_window"] = last["update"] - first["update"]

    # The telemetry loop sleeps `period` seconds, then asks the learner for its stats, and the
    # learner cannot answer while an update runs. So a gap much longer than the period, with
    # the update counter going up across it, is (about) one update's running time.
    blocked: List[float] = []
    prev = None
    for s in by_t:
        if prev is not None and w0 <= s["t"] <= w1 and s["update"] > prev["update"]:
            gap = s["t"] - prev["t"]
            n = s["update"] - prev["update"]
            if gap > period * 1.5:
                blocked.append((gap - period) / n)
        prev = s
    if blocked:
        out["learner_blocked_seconds_per_update_estimate"] = _round(sum(blocked) / len(blocked))
        cyc = out.get("seconds_per_update_cycle")
        if cyc:
            out["learner_busy_share_estimate"] = _round(min(1.0, out["learner_blocked_seconds_per_update_estimate"] / cyc), 3)
        out["learner_blocked_note"] = ("from telemetry gaps: a lower bound, and blind to updates shorter than "
                                       f"the {period:.0f} s telemetry period")
    out["active_battle_sides_mean"] = _round(sum(s["active"] for s in inside) / len(inside), 1)
    bs = [s["avg_batch_size"] for s in inside if s["avg_batch_size"] > 0]
    if bs:
        out["inference_batch_mean"] = _round(sum(bs) / len(bs), 2)
        rps = out.get("inference_requests_per_second")
        if rps:
            out["inference_calls_per_second"] = _round(rps / (sum(bs) / len(bs)), 2)
    lw = [x for x in learner if w0 <= x["t"] <= w1]
    out["learner_lines_in_window"] = len(lw)
    if lw:
        mbs = [x["n_mb"] for x in lw if "n_mb" in x]
        if mbs:
            out["minibatches_per_update_mean"] = _round(sum(mbs) / len(mbs), 1)
        out["last_learner_line"] = {k: v for k, v in lw[-1].items() if k != "t"}
        if len(lw) >= 2:
            out["seconds_between_learner_lines_mean"] = _round((lw[-1]["t"] - lw[0]["t"]) / (len(lw) - 1))
        # The learner prints how long each update took (sec=). That is the exact figure the
        # estimate from telemetry gaps above stands in for with older trainers.
        secs = [x["sec"] for x in lw if "sec" in x]
        if secs:
            out["learner_seconds_per_update"] = _round(sum(secs) / len(secs))
            out["learner_seconds_per_update_min_max"] = [_round(min(secs)), _round(max(secs))]
            per_mb = [x["sec"] / x["n_mb"] for x in lw if "sec" in x and x.get("n_mb")]
            if per_mb:
                out["learner_seconds_per_minibatch"] = _round(sum(per_mb) / len(per_mb), 4)
            cyc = out.get("seconds_per_update_cycle") or out.get("seconds_between_learner_lines_mean")
            if cyc:
                out["learner_busy_share"] = _round(min(1.0, out["learner_seconds_per_update"] / cyc), 3)
        accs = [x["acc"] for x in lw if "acc" in x]
        if accs:
            out["imitation_accuracy_last"] = accs[-1]
    return out


# --------------------------------------------------------------------------- roles

ROLES = (
    "showdown_main", "showdown_simulator", "showdown_sockets", "showdown_other",
    "rollout_worker", "inference_actor", "learner_actor", "driver", "raylet_and_friends",
    "psbot_engine", "other",
)

_ACTOR_ROLE = {
    "RolloutWorker": "rollout_worker",
    "InferenceActor": "inference_actor",
    "LearnerActor": "learner_actor",
}
_RAY_TITLE = re.compile(r"ray::([A-Za-z_][A-Za-z0-9_]*)")
_RAY_INFRA = ("raylet", "gcs_server", "log_monitor.py", "dashboard", "runtime_env", "monitor.py", "plasma", "redis-server")


def _norm(parts: Sequence[str]) -> List[str]:
    return [str(p).replace("\\", "/") for p in parts if p is not None]


def classify(name: str, cmdline: Sequence[str], actor_by_pid: Optional[Dict[int, str]] = None,
             pid: Optional[int] = None, train_script: str = "train.py") -> str:
    """
    The role of one process from its executable name and command line.

    actor_by_pid comes from Ray's worker log files (see actor_names_from_ray_logs). It is what
    identifies actors on Windows, where a process cannot retitle itself: there every Ray worker
    is "python.exe .../default_worker.py ...". On Linux Ray retitles the process to
    "ray::RolloutWorker" (or "ray::RolloutWorker.run" while a method runs) and that is enough.
    """
    parts = _norm(cmdline)
    low = [p.lower() for p in parts]
    joined = " ".join(low)
    base = (name or "").lower()
    exe0 = os.path.basename(low[0]) if low else base

    # --- node: Showdown and the PSBot engine
    if base.startswith("node") or exe0.startswith("node"):
        scripts = [p for p in low[1:] if not p.startswith("-")]
        script = scripts[0] if scripts else ""
        if script.endswith("dist/server/room-battle.js"):
            return "showdown_simulator"
        if script.endswith("dist/server/sockets.js"):
            return "showdown_sockets"
        if "/dist/server/" in script or "/dist/lib/" in script or script.startswith("dist/server/") or script.startswith("dist/lib/"):
            return "showdown_other"
        if os.path.basename(script) == "pokemon-showdown":
            return "showdown_main"
        if script.endswith("engine.js"):
            return "psbot_engine"
        return "other"

    # --- Ray actors and workers
    m = _RAY_TITLE.search(" ".join(parts))
    if m:
        return _ACTOR_ROLE.get(m.group(1), "raylet_and_friends")
    if "default_worker.py" in joined:
        cls = (actor_by_pid or {}).get(pid) if pid is not None else None
        return _ACTOR_ROLE.get(cls or "", "raylet_and_friends")
    # --- the driver (on Windows the venv launcher and the interpreter both carry this command line)
    ts = train_script.replace("\\", "/").lower()
    ts_base = os.path.basename(ts)
    for p in low[1:4]:
        if p == ts or os.path.basename(p) == ts_base:
            return "driver"

    # --- Ray's helpers. Judged by the program, not by the whole command line: a trainer started
    #     with --ckpt-dir /x/ray/dashboard must not be mistaken for one.
    if base.startswith("raylet") or base.startswith("gcs_server") or exe0.startswith("raylet") or exe0.startswith("gcs_server"):
        return "raylet_and_friends"
    scripts = [p for p in low[1:] if not p.startswith("-")]
    script = scripts[0] if scripts else ""
    if "/ray/" in script and any(tok in script for tok in _RAY_INFRA):
        return "raylet_and_friends"
    return "other"


def read_actor_marker(lines: Iterable[str], limit: int = 40) -> Optional[str]:
    """Ray writes ':actor_name:Class' near the top of a worker's .out and .err files."""
    for i, line in enumerate(lines):
        if i >= limit:
            break
        if line.startswith(":actor_name:"):
            return line[len(":actor_name:"):].strip() or None
    return None


_WORKER_LOG = re.compile(r"^worker-[0-9a-f]+-[0-9a-f]+-(\d+)\.(out|err)$")


def actor_names_from_ray_logs(logs_dir: str, want: Optional[Iterable[int]] = None) -> Dict[int, str]:
    """pid -> actor class, from <session>/logs/worker-<id>-<job>-<pid>.out|.err."""
    out: Dict[int, str] = {}
    want_set = set(want) if want is not None else None
    try:
        names = os.listdir(logs_dir)
    except OSError:
        return out
    for fn in names:
        m = _WORKER_LOG.match(fn)
        if not m:
            continue
        pid = int(m.group(1))
        if pid in out or (want_set is not None and pid not in want_set):
            continue
        try:
            with open(os.path.join(logs_dir, fn), "r", encoding="utf-8", errors="replace") as fh:
                cls = read_actor_marker(fh)
        except OSError:
            cls = None
        if cls:
            out[pid] = cls
    return out


_SESSION = re.compile(r"(.*?[/\\]session_\d{4}-\d{2}-\d{2}_[0-9_\-]+?_(\d+))(?=[/\\]|$|\s|\")")


def session_dir_from_cmdlines(cmdlines: Iterable[Sequence[str]]) -> Optional[str]:
    """Ray's helper processes carry the session directory in their arguments."""
    for cmd in cmdlines:
        for arg in cmd or ():
            for piece in str(arg).split("="):
                m = _SESSION.match(piece)
                if m:
                    return m.group(1)
    return None


_SESSION_NAME = re.compile(r"^session_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_\d+_(\d+)$")


def session_started(name: str) -> Optional[float]:
    """When a Ray session directory was made, from its name (local time), or None for another kind of name."""
    m = _SESSION_NAME.match(name)
    if not m:
        return None
    try:
        return _dt.datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S").timestamp()
    except ValueError:
        return None


def pick_sessions(names: Iterable[str], pids: Iterable[int], newer_than: Optional[float] = None) -> List[str]:
    """
    The session directory names that end in one of the pids and, when newer_than is given, were
    made at or after it. Ray names the directory session_<date>_<time>_<micros>_<driver pid>,
    and old sessions stay on disk: without the time, any process of this run that happens to
    get the pid of an earlier day's driver would point at that day's logs.
    """
    want = {str(p) for p in pids}
    out = []
    for n in names:
        m = _SESSION_NAME.match(n)
        if not m or m.group(2) not in want:
            continue
        started = session_started(n)
        if newer_than is not None and (started is None or started < newer_than):
            continue
        out.append(n)
    return sorted(out, reverse=True)          # the newest first


def session_dirs_for_pids(pids: Iterable[int], roots: Optional[Sequence[str]] = None,
                          newer_than: Optional[float] = None) -> List[str]:
    """Session directories under Ray's temp folder that belong to one of the pids (see pick_sessions)."""
    roots = list(roots) if roots else [os.path.join(tempfile.gettempdir(), "ray"), "/tmp/ray"]
    pids = list(pids)
    found = []
    for root in roots:
        try:
            names = os.listdir(root)
        except OSError:
            continue
        found += [os.path.join(root, n) for n in pick_sessions(names, pids, newer_than)]
    return found


# --------------------------------------------------------------------------- process sampling


class Tracker:
    """
    Follows the process trees under a set of roots and accumulates their CPU time.

    A process is identified by (pid, create_time), so a recycled pid is never mistaken for a
    process this script saw before, and never signalled. Every process remembers which root it
    was found under: once a parent has exited its tree can no longer be walked.
    """

    def __init__(self, train_script: str = "train.py", ray_logs: Optional[str] = None):
        import psutil
        self.psutil = psutil
        self.roots: List[Tuple[Any, bool]] = []          # (psutil.Process, owned)
        self.known: Dict[Tuple[int, float], Dict[str, Any]] = {}
        self.samples: List[Dict[str, Any]] = []
        self.actor_by_pid: Dict[int, str] = {}
        self.ray_logs = ray_logs
        self.train_script = train_script
        self._ray_logs_tried = 0.0
        # A directory given by the caller or read off a Ray process's own command line is
        # final. One found by matching pids is kept only until a command line names another.
        self._ray_logs_final = bool(ray_logs)
        self._started = time.time()

    def add_root(self, pid: int, owned: bool) -> None:
        try:
            self.roots.append((self.psutil.Process(pid), owned))
        except self.psutil.Error:
            pass

    def _walk(self) -> List[Tuple[Any, bool, int]]:
        seen: Dict[int, Tuple[Any, bool, int]] = {}
        for root, owned in self.roots:
            try:
                procs = [root] + root.children(recursive=True)
            except self.psutil.Error:
                procs = []
            for p in procs:
                seen.setdefault(p.pid, (p, owned, root.pid))
        return list(seen.values())

    def _find_ray_logs(self) -> None:
        if self._ray_logs_final or time.time() - self._ray_logs_tried < 10.0:
            return
        self._ray_logs_tried = time.time()
        cmds, pids = [], []
        for info in self.known.values():
            cmds.append(info["cmdline"])
            pids.append(info["pid"])
        sd = session_dir_from_cmdlines(cmds)
        if sd and os.path.isdir(os.path.join(sd, "logs")):
            self._set_ray_logs(os.path.join(sd, "logs"))
            self._ray_logs_final = True
            return
        if self.ray_logs:
            return
        # Until a Ray process shows its session on its command line: a session directory named
        # after one of this run's pids, and made since this run began (a minute of slack for
        # clocks and for the directory being made before the process is first seen).
        for c in session_dirs_for_pids(pids, newer_than=self._started - 60.0):
            logs = os.path.join(c, "logs")
            if os.path.isdir(logs):
                self._set_ray_logs(logs)
                return

    def _set_ray_logs(self, logs: str) -> None:
        if self.ray_logs and os.path.normcase(os.path.abspath(self.ray_logs)) != os.path.normcase(os.path.abspath(logs)):
            self.actor_by_pid.clear()          # names read from another session's logs
        self.ray_logs = logs

    def sample(self) -> Dict[str, Any]:
        now = time.time()
        snap: Dict[Tuple[int, float], Tuple[float, int]] = {}
        for p, owned, root_pid in self._walk():
            try:
                with p.oneshot():
                    key = (p.pid, round(p.create_time(), 2))
                    ct = p.cpu_times()
                    cpu = float(ct.user + ct.system)
                    rss = int(p.memory_info().rss)
                    if key not in self.known:
                        try:
                            cmd = list(p.cmdline())
                        except self.psutil.Error:
                            cmd = []
                        self.known[key] = {"pid": p.pid, "create_time": key[1], "name": p.name(), "cmdline": cmd,
                                           "owned": owned, "proc": p, "role": None, "root_pid": root_pid}
                    elif self.known[key]["role"] in (None, "raylet_and_friends", "other") and not IS_WINDOWS:
                        # Linux: an actor's title only appears once the actor is created in the worker.
                        try:
                            self.known[key]["cmdline"] = list(p.cmdline())
                        except self.psutil.Error:
                            pass
            except self.psutil.Error:
                continue
            snap[key] = (cpu, rss)
        self._find_ray_logs()
        if self.ray_logs:
            unknown = [i["pid"] for i in self.known.values()
                       if i["pid"] not in self.actor_by_pid and "default_worker.py" in " ".join(i["cmdline"]).replace("\\", "/")]
            if unknown:
                self.actor_by_pid.update(actor_names_from_ray_logs(self.ray_logs, unknown))
        for info in self.known.values():
            info["role"] = classify(info["name"], info["cmdline"], self.actor_by_pid, info["pid"], self.train_script)
        s = {"t": now, "procs": snap}
        self.samples.append(s)
        return s

    def roles(self) -> Dict[Tuple[int, float], str]:
        return {k: (v["role"] or "other") for k, v in self.known.items()}


def cpu_by_role(samples: Sequence[Dict[str, Any]], roles: Dict[Any, str], w0: float, w1: float) -> Dict[str, Any]:
    """
    CPU seconds per wall second for each role over [w0, w1].

    samples: [{"t": seconds, "procs": {key: (cpu_seconds_total, rss_bytes)}}, ...] in time order.
    The CPU a process used between two samples is spread evenly over that interval and the
    part of the interval inside the window is counted. A process first seen at a sample has
    no earlier reading, so its first interval is not counted.
    """
    per_proc: Dict[Any, float] = {}
    rss_max: Dict[Any, int] = {}
    covered = 0.0
    for a, b in zip(samples, samples[1:]):
        t0, t1 = a["t"], b["t"]
        if t1 <= t0:
            continue
        lo, hi = max(t0, w0), min(t1, w1)
        if hi <= lo:
            continue
        frac = (hi - lo) / (t1 - t0)
        covered += hi - lo
        for key, (cpu1, rss1) in b["procs"].items():
            if key in a["procs"]:
                d = cpu1 - a["procs"][key][0]
                if d > 0:
                    per_proc[key] = per_proc.get(key, 0.0) + d * frac
            rss_max[key] = max(rss_max.get(key, 0), rss1)
    out: Dict[str, Any] = {"covered_seconds": round(covered, 2), "by_role": {}}
    if covered <= 0:
        return out
    agg: Dict[str, Dict[str, Any]] = {}
    for key in set(per_proc) | set(rss_max):
        role = roles.get(key, "other")
        g = agg.setdefault(role, {"processes": 0, "cpu": 0.0, "max_one": 0.0, "rss": 0})
        g["processes"] += 1
        c = per_proc.get(key, 0.0) / covered
        g["cpu"] += c
        g["max_one"] = max(g["max_one"], c)
        g["rss"] += rss_max.get(key, 0)
    total = 0.0
    for role in sorted(agg, key=lambda r: -agg[r]["cpu"]):
        g = agg[role]
        total += g["cpu"]
        out["by_role"][role] = {
            "processes": g["processes"],
            "cpu_seconds_per_wall_second": round(g["cpu"], 3),
            "busiest_single_process": round(g["max_one"], 3),
            "rss_mib": round(g["rss"] / 2**20, 1),
        }
    out["total_cpu_seconds_per_wall_second"] = round(total, 3)
    sd = [out["by_role"].get(r, {}).get("cpu_seconds_per_wall_second", 0.0) for r in ("showdown_sockets", "showdown_other")]
    out["showdown_children_other_than_simulator"] = round(sum(sd), 3)
    return out


def add_per_decision(cpu: Dict[str, Any], decisions_per_second: Optional[float]) -> None:
    if not decisions_per_second or decisions_per_second <= 0:
        return
    for g in cpu.get("by_role", {}).values():
        g["cpu_ms_per_decision"] = round(g["cpu_seconds_per_wall_second"] / decisions_per_second * 1000.0, 4)
    if "total_cpu_seconds_per_wall_second" in cpu:
        cpu["cores_per_1000_decisions_per_second"] = round(cpu["total_cpu_seconds_per_wall_second"] / decisions_per_second * 1000.0, 2)


# --------------------------------------------------------------------------- GPU


def _num(x: str) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def gpu_sample(runner: Callable[..., Any] = subprocess.run) -> Optional[Dict[str, Any]]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = runner([exe, "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw", "--format=csv,noheader,nounits"],
                   capture_output=True, text=True, timeout=15)
        rows = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
        gpus = []
        for ln in rows:
            f = [x.strip() for x in ln.split(",")]
            gpus.append({"util": _num(f[0]), "mem": _num(f[1]), "total": _num(f[2]), "power": _num(f[3]) if len(f) > 3 else None})
        apps: Dict[int, float] = {}
        r2 = runner([exe, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        for ln in r2.stdout.strip().splitlines():
            f = [x.strip() for x in ln.split(",")]
            if len(f) >= 2 and f[0].isdigit() and _num(f[1]) is not None:
                apps[int(f[0])] = float(_num(f[1]))
        return {"t": time.time(), "gpus": gpus, "apps": apps}
    except Exception:  # noqa: BLE001
        return None


def gpu_summary(samples: Sequence[Dict[str, Any]], w0: float, w1: float, role_by_pid: Dict[int, str]) -> Dict[str, Any]:
    inside = [s for s in samples if s and w0 <= s["t"] <= w1 and s["gpus"]]
    if not inside:
        return {"samples": 0}
    out: Dict[str, Any] = {"samples": len(inside), "gpus": []}
    n = max(len(s["gpus"]) for s in inside)
    for i in range(n):
        u = [s["gpus"][i]["util"] for s in inside if i < len(s["gpus"]) and s["gpus"][i]["util"] is not None]
        m = [s["gpus"][i]["mem"] for s in inside if i < len(s["gpus"]) and s["gpus"][i]["mem"] is not None]
        pw = [s["gpus"][i]["power"] for s in inside if i < len(s["gpus"]) and s["gpus"][i]["power"] is not None]
        tot = [s["gpus"][i]["total"] for s in inside if i < len(s["gpus"]) and s["gpus"][i]["total"] is not None]
        out["gpus"].append({
            "utilization_percent_mean": _round(sum(u) / len(u), 1) if u else None,
            "utilization_percent_min": min(u) if u else None,
            "utilization_percent_max": max(u) if u else None,
            "memory_used_mib_mean": _round(sum(m) / len(m), 0) if m else None,
            "memory_used_mib_max": max(m) if m else None,
            "memory_total_mib": tot[0] if tot else None,
            "power_watts_mean": _round(sum(pw) / len(pw), 1) if pw else None,
        })
    by_role: Dict[str, float] = {}
    for s in inside:
        cur: Dict[str, float] = {}
        for pid, mem in s["apps"].items():
            role = role_by_pid.get(pid)
            if role:
                cur[role] = cur.get(role, 0.0) + mem
        for role, mem in cur.items():
            by_role[role] = max(by_role.get(role, 0.0), mem)
    if by_role:
        out["memory_mib_max_by_role"] = {k: round(v, 0) for k, v in by_role.items()}
    else:
        out["memory_by_role_note"] = "nvidia-smi gave no per-process memory (normal on Windows)"
    return out


# --------------------------------------------------------------------------- starting and stopping


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_ports(ports: Sequence[int], seconds: float, is_open: Callable[[int], bool] = port_open,
                   alive: Callable[[], bool] = lambda: True, sleep: Callable[[float], None] = time.sleep,
                   clock: Callable[[], float] = time.time) -> List[int]:
    """Returns the ports still closed when the time ran out (empty list: all up)."""
    deadline = clock() + seconds
    pending = list(ports)
    while pending and clock() < deadline and alive():
        pending = [p for p in pending if not is_open(p)]
        if pending:
            sleep(1.0)
    return pending


def survivors(known: Sequence[Dict[str, Any]], is_same_and_alive: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
    """
    Of the given processes, the ones this script started (or that descend from one it
    started) and that are still running as the same process. Entries not marked owned are
    never returned, whatever their state: servers this script did not start are never signalled.
    """
    return [k for k in known if k.get("owned") and is_same_and_alive(k)]


def _same_and_alive(psutil_mod, info: Dict[str, Any]) -> bool:
    try:
        p = psutil_mod.Process(info["pid"])
        return p.is_running() and round(p.create_time(), 2) == info["create_time"] and p.status() != psutil_mod.STATUS_ZOMBIE
    except psutil_mod.Error:
        return False


def stop_owned(tracker: Tracker, popen: Optional[subprocess.Popen], gentle_signal: Optional[int], grace: float,
               log: Callable[[str], None]) -> List[int]:
    """
    Stops one started command and everything under it. Gentle first where the OS allows it
    (a signal to the root only), then, after `grace` seconds, a kill of exactly the processes
    recorded under it that are still the same processes. Returns the pids killed.
    """
    psutil_mod = tracker.psutil
    if popen is None:
        return []
    try:
        tracker.sample()            # last look at the tree while the parent links still exist
    except Exception:  # noqa: BLE001
        pass
    root_pid = popen.pid
    # Everything ever seen under this command, including processes whose parent has already
    # gone (after the parent exits the tree can no longer be walked, so it was remembered).
    mine: List[Dict[str, Any]] = [info for info in tracker.known.values() if info["owned"] and info.get("root_pid") == root_pid]
    if popen.poll() is None and gentle_signal is not None and not IS_WINDOWS:
        try:
            popen.send_signal(gentle_signal)
        except OSError:
            pass
    alive_check = lambda k: _same_and_alive(psutil_mod, k)  # noqa: E731
    deadline = time.time() + grace
    while time.time() < deadline:
        if popen.poll() is not None and not survivors(mine, alive_check):
            break
        time.sleep(0.5)
    killed: List[int] = []
    for info in reversed(survivors(mine, alive_check)):       # later-seen (deeper) first
        try:
            p = psutil_mod.Process(info["pid"])
            if round(p.create_time(), 2) == info["create_time"]:
                p.kill()
                killed.append(info["pid"])
        except psutil_mod.Error:
            pass
    if popen.poll() is None:
        try:
            popen.kill()
            killed.append(root_pid)
        except OSError:
            pass
    try:
        popen.wait(timeout=15)
    except Exception:  # noqa: BLE001
        pass
    if killed:
        log(f"killed {len(set(killed))} leftover process(es) by pid: {sorted(set(killed))}")
    return sorted(set(killed))


def find_running_servers(psutil_mod, ports: Sequence[int]) -> List[int]:
    """pids of Showdown main processes whose command line names one of the ports. Read-only."""
    want = {str(p) for p in ports}
    out = []
    for p in psutil_mod.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = p.info["cmdline"] or []
            if classify(p.info["name"] or "", cmd) != "showdown_main":
                continue
            toks = [str(c) for c in cmd]
            port = None
            for i, t in enumerate(toks):
                if t == "--port" and i + 1 < len(toks):
                    port = toks[i + 1]
                elif t.startswith("--port="):
                    port = t.split("=", 1)[1]
            if port is None:
                tail = [t for t in toks if t.isdigit()]
                port = tail[-1] if tail else None
            if port in want:
                out.append(p.info["pid"])
        except psutil_mod.Error:
            continue
    return sorted(out)


# --------------------------------------------------------------------------- the run


def split_passthrough(argv: Sequence[str]) -> Tuple[List[str], List[str]]:
    argv = list(argv)
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def passthrough_value(args: Sequence[str], flag: str) -> Optional[str]:
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def build_commands(opts: argparse.Namespace, train_args: List[str]) -> Dict[str, Any]:
    if passthrough_value(train_args, "--minutes") is not None:
        raise SystemExit("do not pass --minutes after `--`: this script sets it from --warmup-minutes and --minutes")
    pt_ports = passthrough_value(train_args, "--ports")
    ports_spec = opts.ports or pt_ports
    if opts.ports and pt_ports and parse_ports(opts.ports) != parse_ports(pt_ports):
        raise SystemExit(f"--ports {opts.ports} here and --ports {pt_ports} after `--` disagree")
    if opts.servers in ("start", "use") and not ports_spec:
        raise SystemExit("--ports is required (here or after `--`)")
    train_args = list(train_args)
    if ports_spec and pt_ports is None:
        train_args += ["--ports", ports_spec]
    total = float(opts.warmup_minutes) + float(opts.minutes) + float(opts.margin_minutes)
    train_args += ["--minutes", f"{total:g}"]
    stop_file = getattr(opts, "stop_file_path", None)
    if stop_file:
        if passthrough_value(train_args, "--stop-file") is not None:
            raise SystemExit("--stop-file after `--` and --train-stop-file here: give only one")
        train_args += ["--stop-file", stop_file]
    train_script = opts.train_script if os.path.isabs(opts.train_script) else os.path.join(opts.repo, opts.train_script)
    train_cmd = [opts.python, "-u", train_script] + train_args
    servers_cmd = None
    if opts.servers == "start":
        if not opts.ps_dir:
            raise SystemExit("--servers start needs --ps-dir (the Pokemon Showdown checkout)")
        script = opts.servers_script if os.path.isabs(opts.servers_script) else os.path.join(opts.repo, opts.servers_script)
        servers_cmd = [opts.python, "-u", script, "--dir", opts.ps_dir, "--ports", ports_spec]
    return {"train": train_cmd, "servers": servers_cmd, "ports": parse_ports(ports_spec) if ports_spec else [],
            "train_minutes": total, "train_script": train_script}


class LineReader(threading.Thread):
    """Copies a child's output to a file and keeps the telemetry and learner lines."""

    def __init__(self, stream, log_path: Optional[str], echo: bool):
        super().__init__(daemon=True)
        self.stream = stream
        self.log_path = log_path
        self.echo = echo
        self.telemetry: List[Dict[str, Any]] = []
        self.learner: List[Dict[str, Any]] = []
        self.tail: List[str] = []
        self.first_telemetry: Optional[float] = None
        self.lines = 0

    def run(self) -> None:
        fh = open(self.log_path, "a", encoding="utf-8", errors="replace") if self.log_path else None
        try:
            for raw in iter(self.stream.readline, b""):
                now = time.time()
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                self.lines += 1
                if fh:
                    fh.write(line + "\n")
                    fh.flush()
                t = parse_telemetry(line)
                if t is not None:
                    t["t"] = now
                    self.telemetry.append(t)
                    if self.first_telemetry is None:
                        self.first_telemetry = now
                    if self.echo:
                        print(f"  [{time.strftime('%H:%M:%S')}] steps={t['total_steps']} update={t['update']} "
                              f"requests={t['total_requests']} batch={t['avg_batch_size']}", flush=True)
                else:
                    lr = parse_learner(line)
                    if lr is not None:
                        lr["t"] = now
                        self.learner.append(lr)
                self.tail.append(strip_ansi(line)[:400])
                if len(self.tail) > 40:
                    del self.tail[0]
        finally:
            if fh:
                fh.close()


def host_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"platform": platform.platform(), "python": sys.version.split()[0]}
    try:
        import psutil
        info["logical_cpus"] = psutil.cpu_count(logical=True)
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["memory_total_gib"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:  # noqa: BLE001
        pass
    return info


def run(opts: argparse.Namespace, train_args: List[str]) -> int:
    import psutil

    out_path = os.path.abspath(opts.out) if opts.out else None
    if opts.log:
        log_path = os.path.abspath(opts.log)
    elif out_path:
        log_path = (out_path[:-5] if out_path.endswith(".json") else out_path) + ".log"
    else:
        log_path = os.path.abspath("throughput.log")
    # A file train.py watches for (its --stop-file): creating it asks for the same clean exit
    # as --minutes. It is the only polite way to stop the trainer early on Windows. The name is
    # new for every run and the file is never removed by this script.
    opts.stop_file_path = None
    if opts.train_stop_file:
        opts.stop_file_path = (log_path[:-4] if log_path.endswith(".log") else log_path) + f".stop.{os.getpid()}.{int(time.time())}"
    cmds = build_commands(opts, train_args)
    if opts.dry_run:
        print(json.dumps({"servers": cmds["servers"], "train": cmds["train"], "ports": cmds["ports"], "log": log_path, "out": out_path}, indent=2))
        return 0
    for pth in (out_path, log_path):
        if pth:
            os.makedirs(os.path.dirname(pth) or ".", exist_ok=True)

    def say(msg: str) -> None:
        print(f"[throughput {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    # A plain kill (SIGTERM) or a closed terminal (SIGHUP) would end this script without its
    # clean-up and leave the servers it started running. Treat both like Ctrl+C.
    killed_by_signal = {"got": False}

    def _as_interrupt(_signum, _frame):
        killed_by_signal["got"] = True
        raise KeyboardInterrupt()

    for sig_name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                signal.signal(sig, _as_interrupt)
            except (ValueError, OSError):
                pass

    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")
    env["PYTHONUNBUFFERED"] = "1"
    tracker = Tracker(train_script=cmds["train_script"], ray_logs=opts.ray_logs)
    servers_proc: Optional[subprocess.Popen] = None
    servers_log = None
    train_proc: Optional[subprocess.Popen] = None
    reader: Optional[LineReader] = None
    gpu_samples: List[Dict[str, Any]] = []
    sys_cpu: List[Tuple[float, float]] = []
    mem_free: List[Tuple[float, float]] = []
    killed: List[int] = []
    result: Dict[str, Any] = {
        "label": opts.label, "started": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "host": host_info(), "commands": {"train": cmds["train"], "servers": cmds["servers"], "servers_mode": opts.servers},
        "ports": cmds["ports"], "log": log_path,
    }
    rc = 0
    interrupted = False
    try:
        if cmds["servers"]:
            busy = [p for p in cmds["ports"] if port_open(p, timeout=0.5)] if opts.server_ready_seconds > 0 else []
            if busy:
                raise SystemExit(f"ports already in use, refusing to start servers on them: {busy}")
            say("starting servers: " + " ".join(cmds["servers"]))
            servers_log = open((log_path[:-4] if log_path.endswith(".log") else log_path) + ".servers.log", "ab")
            servers_proc = subprocess.Popen(cmds["servers"], cwd=opts.repo, stdout=servers_log, stderr=subprocess.STDOUT, env=env)
            tracker.add_root(servers_proc.pid, owned=True)
            if opts.server_ready_seconds > 0:
                closed = wait_for_ports(cmds["ports"], opts.server_ready_seconds, alive=lambda: servers_proc.poll() is None)
                if closed:
                    raise SystemExit(f"servers not accepting connections on {closed} after {opts.server_ready_seconds:.0f} s")
            tracker.sample()
            say(f"servers up on {len(cmds['ports'])} port(s)")
        elif opts.servers == "use":
            pids = find_running_servers(psutil, cmds["ports"])
            for pid in pids:
                tracker.add_root(pid, owned=False)
            say(f"using {len(pids)} running Showdown server(s) on the given ports (measured, never signalled)")
            result["servers_found"] = pids

        say("starting trainer: " + " ".join(cmds["train"]))
        train_proc = subprocess.Popen(cmds["train"], cwd=opts.repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        tracker.add_root(train_proc.pid, owned=True)
        reader = LineReader(train_proc.stdout, log_path, echo=not opts.quiet)
        reader.start()

        psutil.cpu_percent(interval=None)
        t_begin = time.time()
        w0 = w1 = None
        announced = set()
        hard_stop = t_begin + (cmds["train_minutes"] * 60.0) + opts.startup_seconds + opts.grace_seconds
        while True:
            time.sleep(max(0.2, float(opts.sample_seconds)))
            now = time.time()
            tracker.sample()
            g = gpu_sample()
            if g:
                gpu_samples.append(g)
            sys_cpu.append((now, psutil.cpu_percent(interval=None)))
            mem_free.append((now, psutil.virtual_memory().available / 2**20))
            if w0 is None and reader.first_telemetry is not None:
                w0 = reader.first_telemetry + opts.warmup_minutes * 60.0
                w1 = w0 + opts.minutes * 60.0
                say(f"first telemetry line seen; measuring from {time.strftime('%H:%M:%S', time.localtime(w0))} "
                    f"to {time.strftime('%H:%M:%S', time.localtime(w1))}")
            if w0 is not None and now >= w0 and "w0" not in announced:
                announced.add("w0")
                say("warmup over, measuring")
            if w1 is not None and now >= w1 + float(opts.sample_seconds) and "w1" not in announced:
                announced.add("w1")
                say("window complete, waiting for the trainer to save and exit")
            if train_proc.poll() is not None:
                break
            if reader.first_telemetry is None and now - t_begin > opts.startup_seconds:
                say(f"no telemetry after {opts.startup_seconds:.0f} s, giving up")
                rc = 4
                break
            if now > hard_stop:
                say("the trainer did not exit by itself in time, stopping it")
                rc = 5
                break
    except KeyboardInterrupt:
        interrupted = True
        rc = 130
        say("interrupted, stopping what was started")
    finally:
        if train_proc is not None:
            # A trainer that is still running is asked to stop (SIGINT) and gets time to save its
            # checkpoint. After Ctrl+C it already got that signal from the terminal itself, on
            # either OS. Windows has no way to ask a single process politely, so a trainer that
            # is stuck there (no Ctrl+C involved) is not waited for.
            asked = False
            if train_proc.poll() is None and opts.stop_file_path:
                try:
                    with open(opts.stop_file_path, "w", encoding="utf-8") as sf:
                        sf.write("stop requested by bench/throughput.py\n")
                    asked = True
                    say(f"asked the trainer to stop through {opts.stop_file_path}")
                except OSError as e:
                    say(f"could not create the stop file: {e!r}")
            if train_proc.poll() is not None:
                grace = 20.0
            elif interrupted or asked or not IS_WINDOWS:
                grace = 180.0
            else:
                grace = 5.0
            # Never ask twice. The trainer saves its checkpoint while it unwinds from the first
            # request, and a second SIGINT landing in the middle of that would abort the save.
            # After Ctrl+C in a terminal it has the signal already (same foreground group), and
            # after a stop file it is already leaving. A SIGTERM or SIGHUP sent to this script
            # reached only this script, so then the trainer still has to be told.
            already_told = asked or (interrupted and not killed_by_signal["got"])
            killed += stop_owned(tracker, train_proc, None if already_told else signal.SIGINT, grace, say)
        if servers_proc is not None:
            # start_servers.py stops its servers on SIGTERM. On Windows it cannot be asked, so
            # it and the node processes recorded under it are ended by pid.
            killed += stop_owned(tracker, servers_proc, signal.SIGTERM, 20.0 if not IS_WINDOWS else 2.0, say)
        if reader is not None:
            reader.join(timeout=10)
        if servers_log is not None:
            servers_log.close()

    # ---- numbers
    tel = reader.telemetry if reader else []
    lrn = reader.learner if reader else []
    if reader is None or reader.first_telemetry is None:
        result["error"] = "no telemetry line was ever printed"
        result["tail"] = reader.tail if reader else []
        rc = rc or 4
    else:
        w0 = reader.first_telemetry + opts.warmup_minutes * 60.0
        w1 = w0 + opts.minutes * 60.0
        last_t = tel[-1]["t"] if tel else w0
        complete = last_t >= w1 - 2 * float(opts.sample_seconds)
        w1_eff = min(w1, last_t)
        result["window"] = {
            "start": _dt.datetime.fromtimestamp(w0, _dt.timezone.utc).isoformat(timespec="seconds"),
            "end": _dt.datetime.fromtimestamp(w1_eff, _dt.timezone.utc).isoformat(timespec="seconds"),
            "seconds": round(max(w1_eff - w0, 0.0), 1), "warmup_minutes": opts.warmup_minutes, "minutes": opts.minutes,
            "complete": bool(complete),
        }
        result["telemetry"] = telemetry_metrics(tel, lrn, w0, w1_eff, period=5.0)
        result["telemetry_warmup"] = telemetry_metrics(tel, lrn, reader.first_telemetry, w0, period=5.0)
        cpu = cpu_by_role(tracker.samples, tracker.roles(), w0, w1_eff)
        add_per_decision(cpu, result["telemetry"].get("decisions_per_second"))
        cpu["logical_cpus"] = result["host"].get("logical_cpus")
        sc = [v for t, v in sys_cpu if w0 <= t <= w1_eff]
        if sc:
            cpu["system_cpu_percent_mean"] = round(sum(sc) / len(sc), 1)
            cpu["system_cpu_percent_max"] = round(max(sc), 1)
        mf = [v for t, v in mem_free if w0 <= t <= w1_eff]
        if mf:
            cpu["memory_available_mib_min"] = round(min(mf), 0)
        result["cpu"] = cpu
        role_by_pid = {k[0]: r for k, r in tracker.roles().items()}
        result["gpu"] = gpu_summary(gpu_samples, w0, w1_eff, role_by_pid)
        result["ray_logs"] = tracker.ray_logs
        result["actors_identified"] = {str(k): v for k, v in sorted(tracker.actor_by_pid.items())}
        if not complete:
            result["tail"] = reader.tail
            rc = rc or 3
    result["exit"] = {"train_returncode": train_proc.returncode if train_proc else None, "killed_pids": killed, "code": rc}
    result["finished"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print_summary(result)
    return rc


def print_summary(r: Dict[str, Any]) -> None:
    t = r.get("telemetry") or {}
    c = r.get("cpu") or {}
    g = (r.get("gpu") or {}).get("gpus") or []
    print("")
    print("=== throughput ===")
    if r.get("window"):
        print(f"window: {r['window']['seconds']} s after {r['window']['warmup_minutes']} min warmup"
              + ("" if r["window"]["complete"] else "  (INCOMPLETE: the trainer stopped early)"))
    if t.get("error"):
        print(f"telemetry: {t['error']} (a window shorter than a few update cycles cannot be measured)")
    print(f"decisions/s: {t.get('decisions_per_second')}   ({t.get('decisions_per_second_method')})")
    print(f"inference requests/s: {t.get('inference_requests_per_second')}   mean batch: {t.get('inference_batch_mean')}")
    print(f"updates/min: {t.get('updates_per_minute')}   s per update cycle: {t.get('seconds_per_update_cycle')}   "
          f"learner busy per update (est.): {t.get('learner_blocked_seconds_per_update_estimate')} s")
    if t.get("learner_seconds_per_update") is not None:
        print(f"s per update as the learner timed it: {t.get('learner_seconds_per_update')}   "
              f"per minibatch: {t.get('learner_seconds_per_minibatch')}   learner busy share: {t.get('learner_busy_share')}")
    for i, x in enumerate(g):
        print(f"GPU {i}: util mean {x.get('utilization_percent_mean')}% (min {x.get('utilization_percent_min')}, max {x.get('utilization_percent_max')}), "
              f"memory max {x.get('memory_used_mib_max')} of {x.get('memory_total_mib')} MiB, power {x.get('power_watts_mean')} W")
    if c.get("by_role"):
        print(f"{'role':<22}{'procs':>6}{'CPU s/s':>10}{'busiest':>9}{'ms/decision':>13}{'RSS MiB':>10}")
        for role, v in c["by_role"].items():
            per = v.get("cpu_ms_per_decision")
            per_s = f"{per:>13.4f}" if per is not None else f"{'-':>13}"
            print(f"{role:<22}{v['processes']:>6}{v['cpu_seconds_per_wall_second']:>10.3f}{v['busiest_single_process']:>9.3f}{per_s}{v['rss_mib']:>10.1f}")
        print(f"total CPU s/s: {c.get('total_cpu_seconds_per_wall_second')} of {c.get('logical_cpus')} logical CPUs; "
              f"cores per 1000 decisions/s: {c.get('cores_per_1000_decisions_per_second')}")


# --------------------------------------------------------------------------- offline modes


def parse_log_file(path: str, warmup_minutes: float, minutes: float) -> Dict[str, Any]:
    """Telemetry numbers from a saved train.py output, on the log's own clock."""
    tel: List[Dict[str, Any]] = []
    lrn: List[Dict[str, Any]] = []
    last_t: Optional[float] = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            ts = parse_asctime(strip_ansi(line))
            if ts is not None:
                last_t = ts
            t = parse_telemetry(line)
            if t is not None and t["log_time"] is not None:
                t["t"] = t["log_time"]
                tel.append(t)
                continue
            lr = parse_learner(line)
            if lr is not None and last_t is not None:
                lr["t"] = last_t            # these lines carry no time: the last stamped line before them
                lrn.append(lr)
    if not tel:
        return {"file": path, "error": "no telemetry lines"}
    t0 = tel[0]["t"]
    w0 = t0 + warmup_minutes * 60.0
    w1 = min(w0 + minutes * 60.0, tel[-1]["t"]) if minutes > 0 else tel[-1]["t"]
    return {
        "file": path, "telemetry_lines": len(tel), "learner_lines": len(lrn),
        "run_seconds": round(tel[-1]["t"] - t0, 1),
        "window_seconds": round(max(w1 - w0, 0.0), 1),
        "window": telemetry_metrics(tel, lrn, w0, w1),
        "whole_run": telemetry_metrics(tel, lrn, t0, tel[-1]["t"]),
        "learner_line_times_note": "offline, a learner line gets the time of the last stamped line before it",
    }


def probe_servers(ports: Sequence[int], seconds: float) -> Dict[str, Any]:
    """Read-only: the Showdown process trees on the given ports, their roles and their CPU."""
    import psutil
    tr = Tracker()
    pids = find_running_servers(psutil, ports)
    for pid in pids:
        tr.add_root(pid, owned=False)
    a = tr.sample()
    time.sleep(seconds)
    b = tr.sample()
    cpu = cpu_by_role(tr.samples, tr.roles(), a["t"], b["t"])
    rows = []
    for _key, info in sorted(tr.known.items()):
        rows.append({"pid": info["pid"], "role": info["role"], "owned": info["owned"], "cmd": " ".join(info["cmdline"])[-110:]})
    return {"main_pids": pids, "processes": rows, "cpu": cpu}


# --------------------------------------------------------------------------- stand-ins for the end-to-end test


def _fake_train(argv: Sequence[str]) -> int:
    """
    Stand-in for train.py used by the end-to-end test: prints telemetry in train.py's format at
    a known rate and keeps one child process busy. No Ray, no servers, no GPU.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=400.0)
    ap.add_argument("--steps-per-update", type=int, default=2000)
    ap.add_argument("--update-seconds", type=float, default=1.0)
    ap.add_argument("--period", type=float, default=0.5)
    ap.add_argument("--minutes", type=float, default=0.1)
    ap.add_argument("--ports", default=None)
    ap.add_argument("--burn", type=float, default=0.5, help="CPU share of the child")
    ap.add_argument("--orphan", action="store_true", help="leave the child running at exit, to test the clean-up")
    ap.add_argument("--stop-file", default=None, help="as in train.py: leave cleanly once this file exists")
    ap.add_argument("--silent", action="store_true", help="print no telemetry, to test the give-up path")
    a, _ = ap.parse_known_args(list(argv))
    burner = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--_fake-burn", str(a.burn)])
    start = time.time()
    update = 0
    in_dataset_from = 0.0
    try:
        while time.time() - start < a.minutes * 60.0:
            if a.stop_file and os.path.exists(a.stop_file):
                print(f"Stop file {a.stop_file} found. Shutting down.", flush=True)
                break
            if a.silent:
                time.sleep(a.period)
                continue
            now = time.time() - start
            steps = int(a.rate * now)
            pending = steps - int(in_dataset_from)
            if pending >= a.steps_per_update:
                time.sleep(a.update_seconds)       # the learner is busy: no telemetry meanwhile
                update += 1
                in_dataset_from = steps
                print(f"\x1b[36m(LearnerActor pid={os.getpid()})\x1b[0m [learner] upd={update} kl=0.0010 clip=0.050 ent=1.250 "
                      f"vloss=2.400 ploss=-0.001 loss=1.170 n_mb=128 sec={a.update_seconds:.1f}", flush=True)
                steps = int(a.rate * (time.time() - start))
            stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
            infer = {"total_requests": steps, "avg_batch_size": 28.5, "model_version": update}
            train = {"update": update, "episodes": steps // 30, "steps_in_dataset": steps - int(in_dataset_from), "total_steps": steps}
            print(f"{stamp} [INFO] [Telemetry] ACTIVE: 256 | INFER: {infer} | TRAIN: {train}", flush=True)
            time.sleep(a.period)
        print("Reached --minutes. Shutting down.", flush=True)
    finally:
        if not a.orphan:
            burner.kill()
            burner.wait()
    return 0


def _fake_burn(share: float) -> int:
    """Uses about `share` of one core until killed."""
    share = min(max(share, 0.0), 1.0)
    while True:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 0.05 * share:
            pass
        time.sleep(0.05 * (1.0 - share))


def _fake_servers(argv: Sequence[str]) -> int:
    """Stand-in for start_servers.py: two idle children, no ports, runs until stopped."""
    kids = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "--_fake-burn", "0.05"]) for _ in range(2)]
    try:
        while True:
            time.sleep(0.5)
    finally:
        for k in kids:
            k.kill()
    return 0


# --------------------------------------------------------------------------- self-test

_REAL_LINES = [
    # verbatim from a 12-minute PPO run on the laptop, 2026-10-05 (ANSI codes as Ray printed them)
    "2026-10-05 09:57:40,216 [INFO] [Telemetry] ACTIVE: 256 | INFER: {'total_requests': 5086, 'avg_batch_size': 28.54, 'model_version': 0} | TRAIN: {'update': 0, 'episodes': 6, 'steps_in_dataset': 98, 'total_steps': 98}",
    "\x1b[36m(LearnerActor pid=72908)\x1b[0m [learner] upd=2 kl=0.0016 clip=0.052 ent=1.261 vloss=2.356 ploss=-0.001 loss=1.152 n_mb=132",
    "2026-10-05 09:59:23,609 [INFO] [Telemetry] ACTIVE: 256 | INFER: {'total_requests': 52133, 'avg_batch_size': 28.62, 'model_version': 1} | TRAIN: {'update': 2, 'episodes': 949, 'steps_in_dataset': 1117, 'total_steps': 34562}",
    "[learner] upd=1 kl=0.0000 clip=0.000 ent=0.000 vloss=3.719 ploss=1.858 loss=3.718 n_mb=128",
    "2026-10-05 09:57:35,089 [INFO] Training started. Entering telemetry loop.",
    "\x1b[36m(RolloutWorker pid=39152)\x1b[0m Using blocking ray.get inside async actor. This blocks the event loop.",
    # verbatim from a PPO run of 2026-10-05 with the update's duration at the end of the line
    "\x1b[36m(LearnerActor pid=44296)\x1b[0m [learner] upd=34 kl=0.0060 clip=0.297 ent=1.274 vloss=2.030 ploss=0.004 loss=0.994 n_mb=32 sec=6.1",
    # the imitation form (sec and acc), with train.py's logger line landing before the newline
    "\x1b[36m(LearnerActor pid=5996)\x1b[0m [learner] upd=3 kl=0.0000 clip=0.000 ent=0.000 vloss=3.780 ploss=1.796 loss=3.686 n_mb=128 sec=41.9 acc=0.412"
    "2026-10-05 09:59:23,609 [INFO] [Telemetry] ACTIVE: 256 | INFER: {'total_requests': 52133, 'avg_batch_size': 28.62, 'model_version': 1} | TRAIN: {'update': 2, 'episodes': 949, 'steps_in_dataset': 1117, 'total_steps': 34562}",
]


def self_test(ray_logs: Optional[str] = None) -> int:
    fails: List[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(("ok   " if cond else "FAIL ") + name + (("  " + detail) if detail and not cond else ""))
        if not cond:
            fails.append(name)

    # 1 ports
    check("ports range", parse_ports("8000-8003") == [8000, 8001, 8002, 8003])
    check("ports list", parse_ports("8000, 8002,9000-9001") == [8000, 8002, 9000, 9001])

    # 2 real lines
    t = parse_telemetry(_REAL_LINES[0])
    check("telemetry line", t is not None and t["total_steps"] == 98 and t["total_requests"] == 5086 and t["active"] == 256
          and abs(t["avg_batch_size"] - 28.54) < 1e-9 and t["update"] == 0 and t["log_time"] is not None, str(t))
    t2 = parse_telemetry(_REAL_LINES[2])
    check("telemetry line 2", t2 is not None and t2["total_steps"] == 34562 and t2["update"] == 2 and t2["model_version"] == 1)
    check("telemetry clock", t is not None and t2 is not None and abs((t2["log_time"] - t["log_time"]) - 103.393) < 1e-3)
    l1 = parse_learner(_REAL_LINES[1])
    check("learner line with Ray prefix and colour", l1 is not None and l1["upd"] == 2 and l1["n_mb"] == 132.0 and l1.get("pid") == 72908
          and l1.get("actor") == "LearnerActor" and abs(l1["ploss"] + 0.001) < 1e-12, str(l1))
    l2 = parse_learner(_REAL_LINES[3])
    check("learner line bare", l2 is not None and l2["upd"] == 1 and abs(l2["vloss"] - 3.719) < 1e-12 and "pid" not in l2)
    check("other lines ignored", parse_telemetry(_REAL_LINES[4]) is None and parse_learner(_REAL_LINES[4]) is None
          and parse_telemetry(_REAL_LINES[5]) is None and parse_learner(_REAL_LINES[5]) is None)
    check("truncated telemetry ignored", parse_telemetry(_REAL_LINES[0][:120]) is None)
    l3 = parse_learner(_REAL_LINES[6])
    check("learner line with the update's seconds", l3 is not None and l3["upd"] == 34 and l3["n_mb"] == 32.0 and l3["sec"] == 6.1
          and "acc" not in l3 and l3.get("pid") == 44296, str(l3))
    l4 = parse_learner(_REAL_LINES[7])
    check("imitation line with accuracy, the logger's next line glued on behind it",
          l4 is not None and l4["upd"] == 3 and l4["sec"] == 41.9 and l4["acc"] == 0.412 and l4["n_mb"] == 128.0, str(l4))
    check("the glued telemetry is still read", (parse_telemetry(_REAL_LINES[7]) or {}).get("total_steps") == 34562)

    # 3 telemetry maths on a synthetic run: a steady 400 decisions/s, and a learner that starts an
    #   update whenever 16000 steps are waiting and is deaf for 30 s while it runs
    samples, learner = [], []
    tt, update, ingested = 0.0, 0, 0.0
    while tt < 900:
        true_steps = 400.0 * tt
        if true_steps - ingested >= 16000:
            start_steps = true_steps
            tt += 30.0                       # blocked: the next line only appears after the update
            update += 1
            learner.append({"t": tt, "upd": update, "n_mb": 128.0})
            samples.append({"t": tt, "active": 256, "total_requests": int(400 * (tt - 30)), "avg_batch_size": 29.0,
                            "update": update, "total_steps": int(start_steps), "model_version": update})
            ingested = start_steps
            tt += 5.0
            continue
        samples.append({"t": tt, "active": 256, "total_requests": int(true_steps), "avg_batch_size": 29.0,
                        "update": update, "total_steps": int(true_steps), "model_version": update})
        tt += 5.0
    m = telemetry_metrics(samples, learner, 300.0, 900.0)
    check("rate from update-aligned lines", m.get("decisions_per_second") is not None and abs(m["decisions_per_second"] - 400.0) < 8.0,
          str(m.get("decisions_per_second")) + " " + str(m.get("decisions_per_second_method")))
    check("update cycle", m.get("seconds_per_update_cycle") is not None and 35.0 < m["seconds_per_update_cycle"] < 50.0, str(m.get("seconds_per_update_cycle")))
    check("learner blocked estimate", m.get("learner_blocked_seconds_per_update_estimate") is not None
          and abs(m["learner_blocked_seconds_per_update_estimate"] - 30.0) < 6.0, str(m.get("learner_blocked_seconds_per_update_estimate")))
    check("too few lines", "error" in telemetry_metrics(samples[:1], [], 0.0, 1.0))

    # 4 the classifier, on command lines as psutil shows them
    W = "C:\\Program Files\\nodejs\\node.exe"
    PS = "C:\\Users\\you\\showdown-bot\\server\\dist\\server\\"
    DW = "C:\\Users\\you\\ps-ppo\\.venv\\Lib\\site-packages\\ray\\_private\\workers\\default_worker.py"
    cases = [
        ("node.exe", ["node", "--max-old-space-size=2048", "pokemon-showdown", "start", "--no-security", "--skip-build", "--port", "8000"], None, "showdown_main"),
        ("node.exe", [W, "--max-old-space-size=2048", PS + "room-battle.js"], None, "showdown_simulator"),
        ("node.exe", [W, "--max-old-space-size=2048", PS + "sockets.js", "start", "--no-security", "--skip-build", "--port", "8000"], None, "showdown_sockets"),
        ("node.exe", [W, "--max-old-space-size=2048", PS + "team-validator-async.js"], None, "showdown_other"),
        ("node.exe", [W, "--max-old-space-size=2048", PS + "chat-plugins\\datasearch.js"], None, "showdown_other"),
        ("node.exe", [W, "--max-old-space-size=2048", PS + "artemis\\remote.js"], None, "showdown_other"),
        ("node", ["node", "--max-old-space-size=2048", "pokemon-showdown", "start", "--no-security", "--skip-build", "--port", "8003"], None, "showdown_main"),
        ("node", ["/usr/bin/node", "--max-old-space-size=2048", "/home/ubuntu/pokemon-showdown/dist/server/room-battle.js"], None, "showdown_simulator"),
        ("node", ["/usr/bin/node", "--max-old-space-size=2048", "/home/ubuntu/pokemon-showdown/dist/server/sockets.js", "start", "--no-security", "--skip-build", "--port", "8003"], None, "showdown_sockets"),
        ("node", ["/usr/bin/node", "--max-old-space-size=2048", "/home/ubuntu/pokemon-showdown/dist/server/verifier.js"], None, "showdown_other"),
        ("node", ["/usr/bin/node", "/home/ubuntu/ps-ppo/heuristic/engine.js"], None, "psbot_engine"),
        ("node.exe", [W, "C:\\Users\\you\\ps-ppo\\heuristic\\engine.js"], None, "psbot_engine"),
        ("node", ["node", "/somewhere/else/app.js"], None, "other"),
        ("ray::RolloutWorker", ["ray::RolloutWorker.run", "", ""], None, "rollout_worker"),
        ("ray::RolloutWor", ["ray::RolloutWorker"], None, "rollout_worker"),
        ("ray::InferenceAc", ["ray::InferenceActor.infer_batch"], None, "inference_actor"),
        ("ray::LearnerActo", ["ray::LearnerActor"], None, "learner_actor"),
        ("ray::WeightStore", ["ray::WeightStore"], None, "raylet_and_friends"),
        ("ray::IDLE", ["ray::IDLE"], None, "raylet_and_friends"),
        ("raylet", ["/home/ubuntu/venv/lib/python3.12/site-packages/ray/core/src/ray/raylet/raylet", "--raylet_socket_name=/tmp/ray/session_2026-10-05_10-48-43_423715_42188/sockets/raylet"], None, "raylet_and_friends"),
        ("gcs_server", ["/home/ubuntu/venv/lib/python3.12/site-packages/ray/core/src/ray/gcs/gcs_server", "--log_dir=/tmp/ray/session_2026-10-05_10-48-43_423715_42188/logs"], None, "raylet_and_friends"),
        ("python", ["/home/ubuntu/venv/bin/python", "-u", "/home/ubuntu/venv/lib/python3.12/site-packages/ray/_private/log_monitor.py", "--logs-dir=/tmp/ray/x"], None, "raylet_and_friends"),
        ("python", ["/home/ubuntu/venv/bin/python", "-u", "/home/ubuntu/venv/lib/python3.12/site-packages/ray/dashboard/agent.py", "--node-ip-address=127.0.0.1"], None, "raylet_and_friends"),
        ("raylet.exe", ["C:\\Users\\you\\ps-ppo\\.venv\\Lib\\site-packages\\ray\\core\\src\\ray\\raylet\\raylet.exe", "--x"], None, "raylet_and_friends"),
        ("python.exe", ["C:\\Users\\you\\ps-ppo\\.venv\\Scripts\\python.exe", DW, "--node-ip-address=127.0.0.1"], 111, "rollout_worker"),
        ("python.exe", ["C:\\Python314\\python.exe", DW, "--node-ip-address=127.0.0.1"], 222, "inference_actor"),
        ("python.exe", ["C:\\Python314\\python.exe", DW], 333, "learner_actor"),
        ("python.exe", ["C:\\Python314\\python.exe", DW], 444, "raylet_and_friends"),
        ("python.exe", ["C:\\Python314\\python.exe", DW], 555, "raylet_and_friends"),
        ("python.exe", ["C:\\Users\\you\\ps-ppo\\.venv\\Scripts\\python.exe", "-u", "C:\\Users\\you\\ps-ppo\\train.py", "--mode", "ppo"], None, "driver"),
        ("python", ["/home/ubuntu/venv/bin/python", "-u", "/home/ubuntu/ps-ppo/train.py", "--mode", "ppo", "--ports", "8000-8015"], None, "driver"),
        ("python", ["/home/ubuntu/venv/bin/python", "-u", "/home/ubuntu/ps-ppo/train.py", "--ckpt-dir", "/home/ubuntu/ray/dashboard"], None, "driver"),
        ("python", ["/home/ubuntu/venv/bin/python", "start_servers.py", "--dir", "/home/ubuntu/pokemon-showdown", "--ports", "8000-8015"], None, "other"),
        ("bash", ["bash", "-c", "sleep 1"], None, "other"),
        ("node", [], None, "other"),
    ]
    amap = {111: "RolloutWorker", 222: "InferenceActor", 333: "LearnerActor", 444: "WeightStore"}
    bad = []
    for name, cmd, pid, want in cases:
        ts = "/home/ubuntu/ps-ppo/train.py" if "/home/ubuntu" in " ".join(cmd) else "C:\\Users\\you\\ps-ppo\\train.py"
        got = classify(name, cmd, amap, pid, ts)
        if got != want:
            bad.append((name, cmd[-1] if cmd else "", got, want))
    check(f"classifier ({len(cases)} command lines)", not bad, str(bad))

    # 5 Ray log markers
    check("actor marker", read_actor_marker([":job_id:01000000\n", ":actor_name:LearnerActor\n", "x\n"]) == "LearnerActor")
    check("actor marker missing", read_actor_marker([":job_id:01000000\n", "hello\n"]) is None)
    m1 = _WORKER_LOG.match("worker-4853a0f550488bbed5164184eb1febbdc3ceb97e37361019830cf590-ffffffff-66744.out")
    m2 = _WORKER_LOG.match("worker-4853a0f550488bbed5164184eb1febbdc3ceb97e37361019830cf590-01000000-66744.err")
    check("worker log name", bool(m1) and bool(m2) and m2.group(1) == "66744" and not _WORKER_LOG.match("python-core-worker-abc_66744.log"))
    sd = session_dir_from_cmdlines([["raylet.exe", "--log_dir=C:\\Users\\you\\AppData\\Local\\Temp\\ray\\session_2026-10-05_10-48-43_423715_42188\\logs"]])
    check("session dir from a Windows command line", sd is not None and sd.endswith("session_2026-10-05_10-48-43_423715_42188"), str(sd))
    sd = session_dir_from_cmdlines([["x"], ["/a/gcs_server", "--log_dir=/tmp/ray/session_2026-10-05_10-48-43_423715_42188/logs", "--y"]])
    check("session dir from a Linux command line", sd == "/tmp/ray/session_2026-10-05_10-48-43_423715_42188", str(sd))
    check("no session dir", session_dir_from_cmdlines([["python", "train.py"]]) is None)
    names = ["session_2026-10-05_11-52-49_868036_71820", "session_2026-10-05_16-23-07_768847_105468", "session_latest", "sockets"]
    t_run = _dt.datetime(2026, 10, 5, 16, 23, 0).timestamp()
    check("session time from its name", abs((session_started(names[1]) or 0) - (t_run + 7)) < 1e-6 and session_started("session_latest") is None)
    check("a pid that an old session is named after does not point at the old logs",
          pick_sessions(names, [71820, 105468, 4242], newer_than=t_run - 60) == [names[1]]
          and pick_sessions(names, [71820], newer_than=t_run - 60) == []
          and pick_sessions(names, [71820, 105468]) == [names[1], names[0]])
    if ray_logs and os.path.isdir(ray_logs):
        got = actor_names_from_ray_logs(ray_logs)
        kinds: Dict[str, int] = {}
        for v in got.values():
            kinds[v] = kinds.get(v, 0) + 1
        print("     actors in", ray_logs, "->", kinds)
        check("real Ray logs: one learner, one inference actor, some rollout workers",
              kinds.get("LearnerActor") == 1 and kinds.get("InferenceActor") == 1 and kinds.get("RolloutWorker", 0) >= 1, str(kinds))

    # 6 CPU accounting: one process at 0.5 core, one at 2.0 cores (threads), one born mid-way
    s = [{"t": 0.0, "procs": {("a", 1): (10.0, 100 << 20), ("b", 1): (5.0, 50 << 20)}},
         {"t": 10.0, "procs": {("a", 1): (15.0, 110 << 20), ("b", 1): (25.0, 50 << 20), ("c", 1): (1.0, 10 << 20)}},
         {"t": 20.0, "procs": {("a", 1): (20.0, 120 << 20), ("b", 1): (45.0, 50 << 20), ("c", 1): (11.0, 10 << 20)}}]
    roles = {("a", 1): "rollout_worker", ("b", 1): "learner_actor", ("c", 1): "rollout_worker"}
    c = cpu_by_role(s, roles, 0.0, 20.0)
    rw, la = c["by_role"]["rollout_worker"], c["by_role"]["learner_actor"]
    check("cpu by role", abs(rw["cpu_seconds_per_wall_second"] - 1.0) < 1e-9 and abs(la["cpu_seconds_per_wall_second"] - 2.0) < 1e-9
          and rw["processes"] == 2 and abs(rw["busiest_single_process"] - 0.5) < 1e-9 and abs(rw["rss_mib"] - 130.0) < 1e-6, str(c))
    c2 = cpu_by_role(s, roles, 5.0, 15.0)
    check("cpu by role, window inside samples", abs(c2["by_role"]["learner_actor"]["cpu_seconds_per_wall_second"] - 2.0) < 1e-9
          and abs(c2["covered_seconds"] - 10.0) < 1e-9, str(c2))
    add_per_decision(c, 500.0)
    check("cpu per decision", abs(c["by_role"]["learner_actor"]["cpu_ms_per_decision"] - 4.0) < 1e-9 and abs(c["cores_per_1000_decisions_per_second"] - 6.0) < 1e-9)
    check("cpu by role, empty", cpu_by_role([], {}, 0, 1)["by_role"] == {})

    # 7 which leftovers would be killed: only owned ones that are still the same process
    known = [{"pid": 1, "create_time": 1.0, "owned": True}, {"pid": 2, "create_time": 2.0, "owned": True},
             {"pid": 3, "create_time": 3.0, "owned": False}, {"pid": 4, "create_time": 4.0, "owned": True}]
    alive = {1, 3, 4}
    recycled = {4}                    # pid 4 now belongs to another process
    sv = survivors(known, lambda k: k["pid"] in alive and k["pid"] not in recycled)
    check("survivor selection never includes servers it did not start or recycled pids", [k["pid"] for k in sv] == [1], str(sv))

    # 8 command building
    ns = argparse.Namespace(ports="8000-8001", servers="start", ps_dir="/ps", python="py", repo="/repo", train_script="train.py",
                            servers_script="start_servers.py", warmup_minutes=5.0, minutes=10.0, margin_minutes=0.5)
    b = build_commands(ns, ["--mode", "ppo", "--battles", "64"])
    check("train command", b["train"][:3] == ["py", "-u", os.path.join("/repo", "train.py")] and b["train"][-4:] == ["--ports", "8000-8001", "--minutes", "15.5"], str(b["train"]))
    check("servers command", b["servers"] == ["py", "-u", os.path.join("/repo", "start_servers.py"), "--dir", "/ps", "--ports", "8000-8001"], str(b["servers"]))
    try:
        build_commands(ns, ["--minutes", "3"])
        check("--minutes after -- is refused", False)
    except SystemExit:
        check("--minutes after -- is refused", True)
    try:
        build_commands(ns, ["--ports", "9000-9001"])
        check("conflicting ports are refused", False)
    except SystemExit:
        check("conflicting ports are refused", True)
    ns2 = argparse.Namespace(**{**vars(ns), "ports": None, "servers": "use"})
    b2 = build_commands(ns2, ["--ports", "8000-8003"])
    check("ports taken from after --", b2["ports"] == [8000, 8001, 8002, 8003] and b2["train"].count("--ports") == 1 and b2["servers"] is None)
    check("split at --", split_passthrough(["--out", "x", "--", "--mode", "ppo"]) == (["--out", "x"], ["--mode", "ppo"]))

    # 9 waiting for ports, with a fake clock and fake sockets
    clock = {"t": 0.0}
    opened = {8000: 2.0, 8001: 4.0, 8002: 1e9}
    left = wait_for_ports([8000, 8001, 8002], 10.0, is_open=lambda p: clock["t"] >= opened[p],
                          sleep=lambda s_: clock.__setitem__("t", clock["t"] + s_), clock=lambda: clock["t"])
    check("wait for ports reports the ones that never opened", left == [8002], str(left))

    # 10 GPU summary
    gs = [{"t": 1.0, "gpus": [{"util": 90.0, "mem": 7000.0, "total": 46068.0, "power": 250.0}], "apps": {111: 3000.0, 222: 2000.0, 999: 50.0}},
          {"t": 2.0, "gpus": [{"util": 70.0, "mem": 7200.0, "total": 46068.0, "power": 230.0}], "apps": {111: 3100.0, 222: 2000.0}}]
    g = gpu_summary(gs, 0.0, 3.0, {111: "learner_actor", 222: "inference_actor"})
    check("gpu summary", g["gpus"][0]["utilization_percent_mean"] == 80.0 and g["gpus"][0]["memory_used_mib_max"] == 7200.0
          and g["memory_mib_max_by_role"] == {"learner_actor": 3100.0, "inference_actor": 2000.0}, str(g))
    check("gpu summary without samples", gpu_summary([], 0, 1, {}) == {"samples": 0})

    print("")
    print("ALL PASSED" if not fails else "FAILED: " + ", ".join(fails))
    return 0 if not fails else 1


# --------------------------------------------------------------------------- main


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--_fake-burn"]:
        return _fake_burn(float(argv[1]))
    own, train_args = split_passthrough(argv)
    if "--_fake-train" in own:           # only the child: in the parent the flag sits after `--`
        return _fake_train([a for a in own if a != "--_fake-train"])
    if os.environ.get("THROUGHPUT_FAKE_SERVERS") == "1" and "--dir" in own:
        return _fake_servers(own)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", default=sys.executable, help="interpreter for train.py and start_servers.py")
    ap.add_argument("--repo", default=REPO_ROOT, help="the repository (train.py runs from here)")
    ap.add_argument("--train-script", default="train.py")
    ap.add_argument("--servers-script", default="start_servers.py")
    ap.add_argument("--servers", choices=("start", "use", "none"), default="use",
                    help="start: start and stop my own servers; use: servers already running on --ports; none: do not look for servers")
    ap.add_argument("--ps-dir", default=None, help="Pokemon Showdown checkout, for --servers start")
    ap.add_argument("--ports", default=None, help="e.g. 8000-8015 (also given to train.py)")
    ap.add_argument("--warmup-minutes", type=float, default=5.0)
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--margin-minutes", type=float, default=0.5, help="extra time train.py runs past the window")
    ap.add_argument("--sample-seconds", type=float, default=5.0)
    ap.add_argument("--startup-seconds", type=float, default=600.0, help="give up if no telemetry appears in this time")
    ap.add_argument("--server-ready-seconds", type=float, default=120.0, help="0 skips the wait for the ports")
    ap.add_argument("--grace-seconds", type=float, default=300.0, help="time the trainer gets to save and exit after its --minutes")
    ap.add_argument("--train-stop-file", action="store_true",
                    help="give train.py a --stop-file and create it if the trainer has to be stopped early (needs a train.py that has --stop-file)")
    ap.add_argument("--ray-logs", default=None, help="Ray session logs directory, if it cannot be found from the process tree")
    ap.add_argument("--out", default=None, help="JSON result file")
    ap.add_argument("--log", default=None, help="copy of the trainer's output (default: next to --out)")
    ap.add_argument("--label", default="")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--parse-log", default=None, help="offline: numbers from a saved train.py output")
    ap.add_argument("--probe-servers", action="store_true", help="read-only look at the Showdown processes on --ports")
    ap.add_argument("--probe-seconds", type=float, default=5.0)
    ap.add_argument("--self-test", action="store_true")
    opts = ap.parse_args(own)

    if opts.self_test:
        return self_test(opts.ray_logs)
    if opts.parse_log:
        res = parse_log_file(opts.parse_log, opts.warmup_minutes, opts.minutes)
        if opts.out:
            with open(opts.out, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
        print(json.dumps(res, indent=2))
        return 0 if "error" not in res else 1
    if opts.probe_servers:
        if not opts.ports:
            raise SystemExit("--probe-servers needs --ports")
        print(json.dumps(probe_servers(parse_ports(opts.ports), opts.probe_seconds), indent=2))
        return 0
    return run(opts, train_args)


if __name__ == "__main__":
    sys.exit(main())
