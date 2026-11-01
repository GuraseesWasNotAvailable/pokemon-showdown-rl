#!/usr/bin/env python
"""
Unattended training supervisor. One long-lived process that takes a run from nothing to a
finished PPO run: imitation, then warmup, then PPO self-play, then "done".

    python pipeline.py --config cloud/run_vm.json --run-dir /home/ubuntu/runs/main
    python pipeline.py --config cloud/run_vm.json --run-dir RUN --dry-run     print the commands and rules
    python pipeline.py --run-dir RUN --stop                                   ask a running supervisor to stop cleanly
    python pipeline.py --config CFG --run-dir RUN --can-start                 exit 0 if a start would do anything
    python pipeline.py --run-dir RUN --clear-failed                           make a failed run startable again

What it does, in a loop, for the phase named in RUN/state.json:

  1. starts the Showdown servers (start_servers.py) and waits until every port answers. They are
     restarted before every trainer start: the spawning plugin keeps refilling battles for the
     players of the previous trainer for a long time.
  2. starts train.py for the phase with --ckpt-dir RUN/checkpoints and --stop-file RUN/trainer.stop.
     train.py resumes from the newest checkpoint by itself.
  3. copies the trainer's output to RUN/logs/trainer_NNNN_<phase>_<time>.log and reads it as it
     arrives: the "[Telemetry]" line (total_steps, update) and the "[learner] upd=" line (losses).
  4. ends the phase when its rule fires, restarts everything when the trainer stalls, stops
     updating or dies, and stops cleanly on SIGTERM / SIGINT / Ctrl+Break or RUN/pipeline.stop.

Stopping a trainer goes through the stop file, so train.py saves a checkpoint on the way out.
The one exception is a trainer whose weights are known or feared to be bad (it could not load the
newest checkpoint, its loss is not a number, the checkpoint it loaded is from a later phase): that
one is killed outright, whatever the reason for the stop, so that its weights never become the
newest checkpoint. Starting this program again on the same run directory picks the run up where
it stopped.

What makes it restart servers and trainer (each restart resumes from the newest checkpoint):

  stall             total_steps has not grown for watchdog.stall_minutes
  update_stall      no learner update for update_stall_minutes although decisions arrive (the
                    learner logs a failed update and skips it)
  trainer_exit      train.py left by itself (a rollout worker crashed, Ray would not start, ...)
  telemetry_errors  train.py cannot reach its learner or inference actor
  server_restart    a Showdown server died and start_servers.py restarted it: its players are gone
  nan_loss          the loss is NaN: terminated without a save, the checkpoints written since are
                    moved to checkpoints/quarantine, the restart uses the one before
  resume_failed     the newest checkpoint would not load: it is moved to checkpoints/quarantine
  start_failed      a server port is taken by something else, or the servers did not come up

It waits longer before each restart (watchdog.backoff_seconds) and gives up, marking the run
"failed", after more than max_restarts_per_hour failures in an hour, after max_fruitless_starts
starts in a row without a single decision, when a NaN comes back after a restart, when the
newest checkpoint belongs to a later phase than the one the state names, or when this program
itself hits more than max_restarts_per_hour internal errors in an hour. A failed run is not
started again until "--clear-failed". Planned restarts (watchdog.recycle_hours, or train.py's
own --minutes) are not failures.

A run that is marked failed, or stopped because the disk is full, leaves the machine up with
nothing running on it. With on_failed "shutdown" this program then waits
on_failed_grace_minutes (stopping it cancels that) and runs shutdown_command, the way on_done
"shutdown" does after the last phase. shutdown_command is tried again every
shutdown_retry_minutes for as long as it fails.

Files in the run directory:

  state.json              the supervisor's state, rewritten atomically (phase, status, counters)
  metrics.jsonl           one JSON object per learner update
  events.jsonl            one JSON object per supervisor event (starts, restarts, phase changes)
  checkpoints/            learner_update_NNNNNN.pt, written by train.py
  phase_final/            a copy (hard link where possible) of the last checkpoint of each
                          finished phase. A later phase restarts the update count at zero and
                          would otherwise overwrite a file of the same number.
  logs/                   pipeline.log, trainer_*.log (the trainer's output, byte for byte),
                          servers_*.log (what start_servers.py printed), used stop files
  logs/servers/           server_<port>.out and .err: each Showdown server's own output
  eval/                   written by evaluate.py; read here only to protect checkpoints from pruning

How decisions are counted. train.py's total_steps is saved in every checkpoint. A restart in
the same phase continues from the checkpoint's count; a change of phase (the checkpoint was
written in another mode) restarts it at zero. So the trainer's own counter IS the number of
decisions of the current phase behind the current weights, and this program takes it as that:
after a crash the count falls back to the last checkpoint together with the weights (the
difference is recorded as lost), and the overall count is the sum over finished phases plus the
current one.

The count this program holds is the last one the trainer reported, and the trainer reports
between learner updates only. So when a trainer is stopped, the checkpoint it saves on the way
out holds a few seconds' worth more than the count (900 to 3,900 decisions in the laptop runs):
the count of a finished phase is that last report, and a trainer that resumes the phase starts
by reporting the checkpoint's larger count, which is then taken. Likewise the "decisions_phase"
of a metrics record is the last report before the learner's line arrived, which can be one
update old when two updates follow each other without a report in between.

Exit codes: 0 stopped cleanly or done, 1 internal error, 2 bad configuration, 3 the run is marked
failed (too many restarts), 4 the disk is too full, 5 another supervisor holds the run directory.

Only the Python standard library is used, so this file runs with any Python 3.10 or later.
"""
from __future__ import annotations

import argparse
import ast
import collections
import copy
import datetime
import difflib
import json
import logging
import math
import os
import queue
import re
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

IS_WINDOWS = os.name == "nt"
HERE = os.path.dirname(os.path.abspath(__file__))

PHASES: Tuple[str, ...] = ("imitation", "warmup", "ppo")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2
EXIT_FAILED = 3
EXIT_DISK = 4
EXIT_LOCKED = 5

CKPT_RE = re.compile(r"^learner_update_(\d{6,})\.pt$")
CKPT_NAME_RE = re.compile(r"learner_update_\d{6,}\.pt")

log = logging.getLogger("pipeline")


# --------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------

def utc_iso(ts: Optional[float] = None) -> str:
    t = time.time() if ts is None else ts
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp(ts: Optional[float] = None) -> str:
    t = time.time() if ts is None else ts
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ascii_safe(text: Any) -> str:
    return str(text).encode("ascii", "replace").decode("ascii")


def atomic_write_json(path: str, obj: Any, retries: int = 40) -> None:
    """
    Write a temp file next to `path`, flush it to disk, then os.replace it over `path`: a reader
    sees the old file or the new one, never half of one. On Windows the replace fails while
    another process has the file open, so it is retried for about two seconds.
    """
    tmp = path + ".tmp"
    data = json.dumps(obj, indent=1)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            last = None
            break
        except PermissionError as e:  # a reader has it open (Windows)
            last = e
            time.sleep(0.05)
    if last is not None:
        raise last
    if not IS_WINDOWS:
        try:
            fd = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str) -> List[dict]:
    """Every line that parses to an object. A truncated or damaged line is skipped, not fatal."""
    out: List[dict] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
    except OSError:
        pass
    return out


class JsonlWriter:
    """Appends one JSON object per line. If the file ends in a torn line, starts on a new one."""

    def __init__(self, path: str, sync: bool = False):
        self.path = path
        self.sync = sync
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        needs_newline = False
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                if f.tell() > 0:
                    f.seek(-1, os.SEEK_END)
                    needs_newline = f.read(1) != b"\n"
        except OSError:
            pass
        self._f = open(path, "a", encoding="utf-8")
        if needs_newline:
            self._f.write("\n")
            self._f.flush()

    def write(self, obj: dict) -> None:
        self._f.write(json.dumps(obj, allow_nan=False) + "\n")
        self._f.flush()
        if self.sync:
            try:
                os.fsync(self._f.fileno())
            except OSError:
                pass

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


def parse_ports(spec: str) -> List[int]:
    out: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def ports_spec(ports: Sequence[int]) -> str:
    """8000,8001,8002 -> "8000-8002"; anything not contiguous stays a comma list."""
    ports = list(ports)
    if len(ports) > 1 and ports == list(range(ports[0], ports[0] + len(ports))):
        return f"{ports[0]}-{ports[-1]}"
    return ",".join(str(p) for p in ports)


def port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def quote_cmd(cmd: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(cmd)) if IS_WINDOWS else shlex.join(list(cmd))


def free_gb(path: str) -> float:
    return shutil.disk_usage(path).free / (1024.0 ** 3)


def unique_new_file(directory: str, stem: str, suffix: str) -> str:
    """Creates a new empty file and returns its path. Never reuses an existing name."""
    os.makedirs(directory, exist_ok=True)
    n = 0
    while True:
        name = f"{stem}{suffix}" if n == 0 else f"{stem}_{n}{suffix}"
        path = os.path.join(directory, name)
        try:
            with open(path, "xb"):
                pass
            return path
        except FileExistsError:
            n += 1


def move_aside(path: str, directory: str) -> Optional[str]:
    """Renames `path` into `directory` under a name that does not exist yet. Never deletes."""
    if not os.path.lexists(path):
        return None
    os.makedirs(directory, exist_ok=True)
    base = os.path.basename(path)
    n = 0
    while True:
        dst = os.path.join(directory, f"{base}.{utc_stamp()}" + (f".{n}" if n else ""))
        if not os.path.lexists(dst):
            try:
                os.rename(path, dst)
                return dst
            except FileNotFoundError:
                return None
            except FileExistsError:
                pass
        n += 1


# --------------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------------

class ConfigError(ValueError):
    pass


_REQUIRED = object()

# config key -> train.py flag. Everything train.py takes, except what this program must own:
# --mode (the phase's "mode"), --ports (the servers of the phase), --ckpt-dir and --stop-file
# (fixed by the run directory).
TRAIN_FLAGS: Dict[str, str] = {
    "battles": "--battles",
    "rooms_per_pair": "--rooms-per-pair",
    "teacher": "--teacher",
    "steps_per_update": "--steps-per-update",
    "minibatch": "--minibatch",
    "seq_len": "--seq-len",
    "grad_accum": "--grad-accum",
    "save_every": "--save-every",
    "shaping": "--shaping",
    "minutes": "--minutes",
}
_TRAIN_KINDS = {"battles": "int?", "rooms_per_pair": "int?", "teacher": "str?", "steps_per_update": "int?",
                "minibatch": "int?", "seq_len": "int?", "grad_accum": "int?", "save_every": "int?",
                "shaping": "str?", "minutes": "num?"}

SCHEMA_TRAIN = {k: (kind, None) for k, kind in _TRAIN_KINDS.items()}
SCHEMA_TRAIN_DEFAULTS = dict(SCHEMA_TRAIN, extra_train_args=("strlist", []))
SCHEMA_SERVERS = {
    "count": ("int", 4),
    "base_port": ("int", 8000),
    "ports": ("str?", None),
    "heap_mb": ("int", 2048),
    "host": ("str", "127.0.0.1"),
    "ready_timeout_seconds": ("num", 180),
    "settle_seconds": ("num", 2),
    "stop_timeout_seconds": ("num", 60),
    "port_free_timeout_seconds": ("num", 60),
    "restart_trainer_on_server_restart": ("bool", True),
    "command": ("strlist?", None),
}
SCHEMA_SERVERS_PHASE = {k: (SCHEMA_SERVERS[k][0].rstrip("?") + "?", None) for k in ("count", "base_port", "ports", "heap_mb")}
SCHEMA_TRAINER = {
    "command": ("strlist?", None),
    "env": ("strdict", {}),
    "stop_timeout_seconds": ("num", 420),
    "failure_stop_timeout_seconds": ("num", 90),
    "kill_timeout_seconds": ("num", 30),
}
SCHEMA_WATCHDOG = {
    "stall_minutes": ("num", 10),
    "startup_grace_minutes": ("num", 15),
    "update_stall_minutes": ("num", 30),
    "max_restarts_per_hour": ("int", 6),
    "backoff_seconds": ("numlist", [15, 60, 180, 600]),
    "poll_seconds": ("num", 1.0),
    "recycle_hours": ("num", 0),
    "max_quarantined_checkpoints": ("int", 2),
    "max_fruitless_starts": ("int", 5),
    "telemetry_error_minutes": ("num", 5),
    "nan_updates_limit": ("int", 3),
}
SCHEMA_DISK = {"min_free_gb": ("num", 5), "check_seconds": ("num", 60)}
SCHEMA_RETENTION = {
    "enabled": ("bool", False),
    "dry_run": ("bool", False),
    "keep_newest": ("int", 5),
    "milestone_every": ("int", 0),
    "keep_hours": ("num", 0),
    "check_minutes": ("num", 10),
}
SCHEMA_REPORTING = {"state_seconds": ("num", 5), "sample_seconds": ("num", 30), "progress_log_seconds": ("num", 60)}
SCHEMA_PLATEAU = {
    "metric": ("str", _REQUIRED),
    "window_decisions": ("int", _REQUIRED),
    "min_improvement_pct": ("num", _REQUIRED),
    "smooth_updates": ("int", 5),
}
_PHASE_COMMON = {
    "skip": ("bool", False),
    "mode": ("str?", None),
    "train": ("dict", {}),
    "extra_train_args": ("strlist", []),
    "servers": ("dict", {}),
    "recycle_hours": ("num?", None),
}
SCHEMA_PHASE = dict(_PHASE_COMMON, min_decisions=("int", 0), max_decisions=("int", 0), plateau=("dict?", None))
SCHEMA_PHASE_PPO = dict(_PHASE_COMMON, target_decisions=("int", 0))
SCHEMA_TOP = {
    "repo_dir": ("str?", None),
    "python": ("str?", None),
    "showdown_dir": ("str?", None),
    "servers": ("dict", {}),
    "trainer": ("dict", {}),
    "train_defaults": ("dict", {}),
    "phases": ("dict", {}),
    "watchdog": ("dict", {}),
    "disk": ("dict", {}),
    "retention": ("dict", {}),
    "reporting": ("dict", {}),
    "on_done": ("str", "idle"),
    "on_done_wait_minutes": ("num", 180),
    "on_done_grace_minutes": ("num", 5),
    # What to do once the run is marked failed, or stopped because the disk is full: nothing starts
    # it again by itself, so on a rented machine "idle" means paying for a GPU that does nothing.
    "on_failed": ("str", "idle"),
    "on_failed_grace_minutes": ("num", 30),
    "shutdown_command": ("strlist", ["sudo", "shutdown", "-h", "now"]),
    "shutdown_retry_minutes": ("num", 10),
    # Read by evaluate.py, not by this program: passed through unchecked.
    "eval": ("any", None),
}


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _coerce(where: str, value: Any, kind: str) -> Any:
    optional = kind.endswith("?")
    base = kind.rstrip("?")
    if value is None:
        if optional:
            return None
        raise ConfigError(f"{where} must not be null")
    if base == "any":
        return value
    if base == "str":
        if not isinstance(value, str):
            raise ConfigError(f"{where} must be a string")
        return value
    if base == "int":
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if not _is_int(value):
            raise ConfigError(f"{where} must be a whole number")
        return value
    if base == "num":
        if not _is_num(value):
            raise ConfigError(f"{where} must be a number")
        return value
    if base == "bool":
        if not isinstance(value, bool):
            raise ConfigError(f"{where} must be true or false")
        return value
    if base == "strlist":
        if not isinstance(value, list) or not all(isinstance(x, (str, int, float)) and not isinstance(x, bool) for x in value):
            raise ConfigError(f"{where} must be a list of strings")
        return [str(x) for x in value]
    if base == "numlist":
        if not isinstance(value, list) or not value or not all(_is_num(x) for x in value):
            raise ConfigError(f"{where} must be a non-empty list of numbers")
        return list(value)
    if base == "dict":
        if not isinstance(value, dict):
            raise ConfigError(f"{where} must be an object")
        return value
    if base == "strdict":
        if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
            raise ConfigError(f"{where} must be an object")
        return {k: str(v) for k, v in value.items() if not k.startswith("_")}
    raise AssertionError(kind)


def _section(where: str, raw: Any, schema: Dict[str, tuple]) -> Dict[str, Any]:
    """Checks one object of the config: an unknown key is an error, a missing one gets its default.
    Keys that start with "_" are comments."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be an object")
    out: Dict[str, Any] = {}
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        if key not in schema:
            near = difflib.get_close_matches(key, list(schema), n=1)
            hint = f" (did you mean '{near[0]}'?)" if near else ""
            raise ConfigError(f"unknown key '{key}' in {where}{hint}. Known keys: {', '.join(sorted(schema))}")
        out[key] = _coerce(f"{where}.{key}", value, schema[key][0])
    for key, (_kind, default) in schema.items():
        if key not in out:
            if default is _REQUIRED:
                raise ConfigError(f"{where}.{key} is required")
            out[key] = copy.deepcopy(default)
    return out


def load_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        raw = read_json(path)
    except OSError as e:
        raise ConfigError(f"cannot read {path}: {e}")
    except ValueError as e:
        raise ConfigError(f"{path} is not valid JSON: {e}")
    return validate_config(raw, overrides)


def validate_config(raw: Any, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = _section("config", raw, SCHEMA_TOP)
    cfg["repo_dir"] = os.path.abspath(cfg["repo_dir"] or HERE)
    cfg["python"] = cfg["python"] or sys.executable
    cfg["servers"] = _section("servers", cfg["servers"], SCHEMA_SERVERS)
    cfg["trainer"] = _section("trainer", cfg["trainer"], SCHEMA_TRAINER)
    cfg["train_defaults"] = _section("train_defaults", cfg["train_defaults"], SCHEMA_TRAIN_DEFAULTS)
    cfg["watchdog"] = _section("watchdog", cfg["watchdog"], SCHEMA_WATCHDOG)
    cfg["disk"] = _section("disk", cfg["disk"], SCHEMA_DISK)
    cfg["retention"] = _section("retention", cfg["retention"], SCHEMA_RETENTION)
    cfg["reporting"] = _section("reporting", cfg["reporting"], SCHEMA_REPORTING)
    for name, cmd in (overrides or {}).items():
        if cmd is not None:
            cfg[name]["command"] = _coerce(f"--{name}-command", cmd, "strlist")

    phases_raw = cfg["phases"]
    for key in phases_raw:
        if not key.startswith("_") and key not in PHASES:
            raise ConfigError(f"unknown phase '{key}' in phases. The phases are: {', '.join(PHASES)}")
    phases: Dict[str, Dict[str, Any]] = {}
    for name in PHASES:
        schema = SCHEMA_PHASE_PPO if name == "ppo" else SCHEMA_PHASE
        ph = _section(f"phases.{name}", phases_raw.get(name), schema)
        ph["mode"] = ph["mode"] or name
        ph["train"] = _section(f"phases.{name}.train", ph["train"], SCHEMA_TRAIN)
        ph["servers"] = _section(f"phases.{name}.servers", ph["servers"], SCHEMA_SERVERS_PHASE)
        if name == "ppo":
            if ph["target_decisions"] < 0:
                raise ConfigError("phases.ppo.target_decisions must be 0 (never) or more")
            ph["min_decisions"], ph["max_decisions"], ph["plateau"] = 0, ph["target_decisions"], None
        else:
            if ph["plateau"] is not None:
                ph["plateau"] = _section(f"phases.{name}.plateau", ph["plateau"], SCHEMA_PLATEAU)
                pl = ph["plateau"]
                if pl["window_decisions"] <= 0 or pl["smooth_updates"] < 1 or not pl["metric"]:
                    raise ConfigError(f"phases.{name}.plateau needs a metric, window_decisions > 0 and smooth_updates >= 1")
            if ph["min_decisions"] < 0 or ph["max_decisions"] < 0:
                raise ConfigError(f"phases.{name}: min_decisions and max_decisions must not be negative")
            if ph["max_decisions"] and ph["min_decisions"] > ph["max_decisions"]:
                raise ConfigError(f"phases.{name}: min_decisions is larger than max_decisions")
            if not ph["skip"] and ph["max_decisions"] == 0 and ph["plateau"] is None:
                raise ConfigError(f"phases.{name} would never end: give it max_decisions or a plateau rule (or \"skip\": true)")
        ports = phase_ports(cfg, ph)
        if not ports:
            raise ConfigError(f"phases.{name}: no server ports")
        if len(set(ports)) != len(ports) or any(p < 1 or p > 65535 for p in ports):
            raise ConfigError(f"phases.{name}: bad server ports {ports}")
        phases[name] = ph
    cfg["phases"] = phases
    active = [n for n in PHASES if not phases[n]["skip"]]
    modes = [phases[n]["mode"] for n in active]
    if len(set(modes)) != len(modes):
        # The trainer continues its counter when the mode is unchanged, so two phases with the
        # same mode could not be told apart.
        raise ConfigError(f"every phase needs its own mode, got {modes}")

    w = cfg["watchdog"]
    for key in ("stall_minutes", "startup_grace_minutes", "poll_seconds"):
        if w[key] <= 0:
            raise ConfigError(f"watchdog.{key} must be above 0")
    for key in ("update_stall_minutes", "recycle_hours", "max_restarts_per_hour", "max_quarantined_checkpoints",
                "max_fruitless_starts", "telemetry_error_minutes", "nan_updates_limit"):
        if w[key] < 0:
            raise ConfigError(f"watchdog.{key} must not be negative")
    if any(b < 0 for b in w["backoff_seconds"]):
        raise ConfigError("watchdog.backoff_seconds must not be negative")
    r = cfg["retention"]
    if r["keep_newest"] < 2:
        raise ConfigError("retention.keep_newest must be at least 2 (the newest is what a restart resumes from; "
                          "the one before it is the fallback when the newest is damaged)")
    if r["milestone_every"] < 0 or r["keep_hours"] < 0 or r["check_minutes"] <= 0:
        raise ConfigError("retention: milestone_every and keep_hours must not be negative, check_minutes must be above 0")
    if cfg["on_done"] not in ("idle", "shutdown"):
        raise ConfigError("on_done must be \"idle\" or \"shutdown\"")
    if cfg["on_failed"] not in ("idle", "shutdown"):
        raise ConfigError("on_failed must be \"idle\" or \"shutdown\"")
    for key in ("on_done", "on_failed"):
        if cfg[key] == "shutdown" and not cfg["shutdown_command"]:
            raise ConfigError(f"{key} is \"shutdown\" but shutdown_command is empty")
    if cfg["on_failed_grace_minutes"] < 0:
        raise ConfigError("on_failed_grace_minutes must not be negative")
    if cfg["shutdown_retry_minutes"] <= 0:
        raise ConfigError("shutdown_retry_minutes must be above 0")
    if cfg["servers"]["command"] is None and not cfg["showdown_dir"]:
        raise ConfigError("showdown_dir is required (the Pokemon Showdown checkout the servers run from)")
    if cfg["disk"]["min_free_gb"] < 0:
        raise ConfigError("disk.min_free_gb must not be negative")
    return cfg


def first_phase(cfg: Dict[str, Any]) -> str:
    for name in PHASES:
        if not cfg["phases"][name]["skip"]:
            return name
    return "done"


def next_phase(cfg: Dict[str, Any], phase: str) -> str:
    i = PHASES.index(phase)
    for name in PHASES[i + 1:]:
        if not cfg["phases"][name]["skip"]:
            return name
    return "done"


def phase_servers(cfg: Dict[str, Any], ph: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(cfg["servers"])
    for key, value in (ph.get("servers") or {}).items():
        if value is not None:
            merged[key] = value
            if key in ("count", "base_port"):
                # a phase that sets a count or a base port means a range, not the global list
                merged["ports"] = (ph["servers"].get("ports"))
    return merged


def phase_ports(cfg: Dict[str, Any], ph: Dict[str, Any]) -> List[int]:
    s = phase_servers(cfg, ph)
    if s.get("ports"):
        try:
            return parse_ports(s["ports"])
        except ValueError:
            raise ConfigError(f"bad ports spec: {s['ports']!r}")
    if s["count"] < 1:
        raise ConfigError("servers.count must be at least 1")
    return list(range(s["base_port"], s["base_port"] + s["count"]))


def _fill(tokens: Sequence[str], values: Dict[str, str]) -> List[str]:
    out = []
    for tok in tokens:
        for key, value in values.items():
            tok = tok.replace("{" + key + "}", value)
        out.append(tok)
    return out


class Paths:
    def __init__(self, run_dir: str):
        self.run = os.path.abspath(run_dir)
        self.checkpoints = os.path.join(self.run, "checkpoints")
        self.quarantine = os.path.join(self.checkpoints, "quarantine")
        self.phase_final = os.path.join(self.run, "phase_final")
        self.logs = os.path.join(self.run, "logs")
        self.server_logs = os.path.join(self.logs, "servers")
        self.stop_files = os.path.join(self.logs, "stop_files")
        self.state = os.path.join(self.run, "state.json")
        self.metrics = os.path.join(self.run, "metrics.jsonl")
        self.events = os.path.join(self.run, "events.jsonl")
        self.eval = os.path.join(self.run, "eval")
        self.final_done = os.path.join(self.eval, "final.done")
        self.trainer_stop = os.path.join(self.run, "trainer.stop")
        self.pipeline_stop = os.path.join(self.run, "pipeline.stop")
        self.lock = os.path.join(self.run, "pipeline.lock")
        self.pipeline_log = os.path.join(self.logs, "pipeline.log")

    def rel(self, path: str) -> str:
        try:
            return os.path.relpath(path, self.run).replace("\\", "/")
        except ValueError:
            return path


def _placeholders(cfg: Dict[str, Any], paths: Paths, phase: str, ports: Sequence[int], heap_mb: int) -> Dict[str, str]:
    return {
        "python": cfg["python"],
        "repo": cfg["repo_dir"],
        "showdown_dir": cfg["showdown_dir"] or "",
        "run_dir": paths.run,
        "phase": phase,
        "ports": ports_spec(ports),
        "heap_mb": str(heap_mb),
        "server_log_dir": paths.server_logs,
    }


def build_server_command(cfg: Dict[str, Any], paths: Paths, phase: str) -> List[str]:
    ph = cfg["phases"][phase]
    s = phase_servers(cfg, ph)
    ports = phase_ports(cfg, ph)
    # --log-dir: each Showdown server's own output (server_<port>.out and .err) stays with the run,
    # under RUN/logs/servers, and not in the Showdown checkout where the next run would add to it.
    base = s["command"] or ["{python}", "start_servers.py", "--dir", "{showdown_dir}", "--ports", "{ports}", "--heap-mb", "{heap_mb}",
                            "--log-dir", "{server_log_dir}"]
    return _fill(base, _placeholders(cfg, paths, phase, ports, s["heap_mb"]))


def phase_train_settings(cfg: Dict[str, Any], phase: str) -> Dict[str, Any]:
    ph = cfg["phases"][phase]
    merged: Dict[str, Any] = {}
    for key in TRAIN_FLAGS:
        value = ph["train"].get(key)
        if value is None:
            value = cfg["train_defaults"].get(key)
        if value is not None:
            merged[key] = value
    return merged


def build_trainer_command(cfg: Dict[str, Any], paths: Paths, phase: str) -> List[str]:
    ph = cfg["phases"][phase]
    ports = phase_ports(cfg, ph)
    values = _placeholders(cfg, paths, phase, ports, phase_servers(cfg, ph)["heap_mb"])
    cmd = _fill(cfg["trainer"]["command"] or ["{python}", "train.py"], values)
    cmd += ["--mode", ph["mode"], "--ports", ports_spec(ports)]
    for key, value in phase_train_settings(cfg, phase).items():
        cmd += [TRAIN_FLAGS[key], str(value)]
    cmd += ["--ckpt-dir", paths.checkpoints, "--stop-file", paths.trainer_stop]
    cmd += _fill(list(cfg["train_defaults"]["extra_train_args"]) + list(ph["extra_train_args"]), values)
    return cmd


def describe_rule(cfg: Dict[str, Any], phase: str) -> str:
    ph = cfg["phases"][phase]
    if ph["skip"]:
        return "skipped"
    if phase == "ppo":
        if ph["max_decisions"] == 0:
            return "never ends (target_decisions is 0)"
        return f"ends at {ph['max_decisions']:,} decisions"
    parts = []
    if ph["plateau"]:
        pl = ph["plateau"]
        parts.append(f"ends once it has {ph['min_decisions']:,} decisions and {pl['metric']} (mean of {pl['smooth_updates']} updates) "
                     f"improved by less than {pl['min_improvement_pct']}% over the last {pl['window_decisions']:,} decisions")
    if ph["max_decisions"]:
        parts.append(f"ends at {ph['max_decisions']:,} decisions at the latest")
    return "; ".join(parts)


# --------------------------------------------------------------------------------------------
# the trainer's output
# --------------------------------------------------------------------------------------------

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
# What Ray puts behind an actor's line when it held back copies of it from other processes:
#   ... instead. [repeated 5x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 ...)
# It is Ray's remark, not part of what the actor printed, so it is cut off before a record is read.
RAY_DEDUP_RE = re.compile(r"\s*\[repeated \d+x across cluster\].*$")
# the start of a line written by train.py's logger: "2026-10-05 09:48:48,489 [INFO] "
LOG_STAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \[[A-Z]+\] ")
RAY_PREFIX_RE = re.compile(r"^\((?P<actor>[A-Za-z_][\w.]*)(?:\s+pid=(?P<pid>\d+))?[^)]*\)\s?")
TELEMETRY_RE = re.compile(r"\[Telemetry\]\s+ACTIVE:\s*(?P<active>-?\d+)\s*\|(?P<rest>.*)$")
TRAIN_DICT_RE = re.compile(r"TRAIN:\s*(\{.*?\})")
INFER_DICT_RE = re.compile(r"INFER:\s*(\{.*?\})")
LEARNER_RE = re.compile(r"\[learner\]\s+upd=(?P<upd>\d+)\b(?P<rest>.*)$")
KV_RE = re.compile(r"(?<![\w.])(?P<k>[A-Za-z_]\w*)=(?P<v>[-+]?(?:nan|inf(?:inity)?|(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?))(?![\w.])", re.I)
HEADER_RE = re.compile(
    r"\bmode=(?P<mode>\S+)\s+teacher=(?P<teacher>\S+)\s+reward=(?P<reward>\S+)\s+ports=\[(?P<ports>[^\]]*)\]\s+"
    r"steps_per_update=(?P<spu>\d+)\s+minibatch=(?P<mb>\d+)x(?P<accum>\d+)\s+seq=(?P<seq>\d+)\s+ckpt=(?P<ckpt>.*)$")
RESUMED_RE = re.compile(r"\[learner\]\s+resumed from\s+(?P<path>.+)$")
RESUME_FAILED_RE = re.compile(r"\[learner\]\s+resume failed from\s+(?P<path>.+?\.pt):\s*(?P<err>.*)$")
PHASE_CHANGE_RE = re.compile(r"PHASE CHANGE DETECTED \((?P<old>\S+) -> (?P<new>\S+)\)")
UPDATE_FAILED_RE = re.compile(r"PPO Update failed:\s*(?P<err>.*)$")
FINAL_SAVED_RE = re.compile(r"Saved a final checkpoint:\s*(?P<path>.+)$")
STOP_ACK_RE = re.compile(r"Stop file .* found\. Shutting down\.")
_INT_FIELD_RE = {name: re.compile(r"'" + name + r"':\s*(-?\d+)") for name in ("update", "episodes", "steps_in_dataset", "total_steps")}


def _literal_dict(text: Optional[str]) -> Optional[dict]:
    if not text:
        return None
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and float(value).is_integer():
        return int(value)
    return None


def parse_trainer_events(line: str) -> List[Dict[str, Any]]:
    """
    Everything of use in one physical line of train.py's output, in order. Usually that is one
    record or none, but two threads write to the trainer's stdout (Ray printing what the actors
    print, and train.py's own logger), and now and then the logger's line lands between an
    actor's text and its newline:

        (LearnerActor pid=N) [learner] upd=6 ... loss=3.016 n_mb=342026-10-05 09:48:48,489 [INFO] [Telemetry] ...

    Such a line is cut in front of the logger's timestamp ("n_mb=34" | "2026-10-05 ...") and both
    halves are read.
    """
    text = ANSI_RE.sub("", line).rstrip("\r\n")
    cuts = [m.start() for m in LOG_STAMP_RE.finditer(text) if m.start() > 0]
    parts = [text[a:b] for a, b in zip([0] + cuts, cuts + [len(text)])]
    out = []
    for part in parts:
        ev = _parse_record(part)
        if ev is not None:
            out.append(ev)
    return out


def parse_trainer_line(line: str) -> Optional[Dict[str, Any]]:
    """The first record of a line, or None. See parse_trainer_events."""
    events = parse_trainer_events(line)
    return events[0] if events else None


def _parse_record(text: str) -> Optional[Dict[str, Any]]:
    """
    One record of train.py's output (its own log lines and the actor output Ray forwards, with or
    without Ray's "(LearnerActor pid=N)" prefix) -> {"kind": ..., ...}, or None for one this
    program does not use. The formats are train.py's _telemetry_loop and learner.py's
    "[learner] upd=" print.
    """
    actor = None
    m = RAY_PREFIX_RE.match(text)
    if m:
        actor = m.group("actor")
        text = RAY_DEDUP_RE.sub("", text[m.end():])

    m = TELEMETRY_RE.search(text)
    if m:
        rest = m.group("rest")
        td = TRAIN_DICT_RE.search(rest)
        train = _literal_dict(td.group(1) if td else None) or {}
        if not train:  # a dict this parser cannot evaluate: fall back to the plain numbers
            for name, rx in _INT_FIELD_RE.items():
                mm = rx.search(rest)
                if mm:
                    train[name] = int(mm.group(1))
        idm = INFER_DICT_RE.search(rest)
        return {
            "kind": "telemetry",
            "active": int(m.group("active")),
            "total_steps": _as_int(train.get("total_steps")),
            "update": _as_int(train.get("update")),
            "episodes": _as_int(train.get("episodes")),
            "steps_in_dataset": _as_int(train.get("steps_in_dataset")),
            "infer": _literal_dict(idm.group(1) if idm else None) or {},
        }

    m = LEARNER_RE.search(text)
    if m:
        values: Dict[str, Optional[float]] = {}
        nonfinite: List[str] = []
        for kv in KV_RE.finditer(m.group("rest")):
            try:
                v = float(kv.group("v"))
            except ValueError:
                continue
            key = kv.group("k")
            if math.isfinite(v):
                values[key] = v
            else:
                values[key] = None
                nonfinite.append(key)
        return {"kind": "learner_update", "update": int(m.group("upd")), "values": values, "nonfinite": nonfinite}

    m = RESUME_FAILED_RE.search(text)
    if m:
        return {"kind": "resume_failed", "path": m.group("path").strip(), "error": m.group("err").strip()}
    m = RESUMED_RE.search(text)
    if m:
        return {"kind": "resumed", "path": m.group("path").strip()}
    if "[learner] pushed init policy 0" in text:
        return {"kind": "init_policy"}
    m = PHASE_CHANGE_RE.search(text)
    if m:
        return {"kind": "phase_change", "old": m.group("old"), "new": m.group("new")}
    m = UPDATE_FAILED_RE.search(text)
    if m:
        return {"kind": "update_failed", "error": m.group("err").strip()}
    if "Learner training loop CRASHED" in text:
        return {"kind": "learner_crashed"}
    if "A RolloutWorker crashed in run()" in text:
        return {"kind": "worker_crashed", "detail": text.strip()[-300:]}
    m = FINAL_SAVED_RE.search(text)
    if m:
        return {"kind": "final_saved", "path": m.group("path").strip()}
    if "Could not save a final checkpoint" in text:
        return {"kind": "final_save_failed", "detail": text.strip()[-300:]}
    if "Reached --minutes" in text:
        return {"kind": "minutes_reached"}
    if STOP_ACK_RE.search(text):
        return {"kind": "stop_ack"}
    if "Telemetry loop encountered an error" in text:
        return {"kind": "telemetry_error", "detail": text.strip()[-300:]}
    if actor is None:
        m = HEADER_RE.search(text)
        if m:
            return {
                "kind": "header", "mode": m.group("mode"), "teacher": m.group("teacher"), "reward": m.group("reward"),
                "steps_per_update": int(m.group("spu")), "minibatch": int(m.group("mb")), "grad_accum": int(m.group("accum")),
                "seq_len": int(m.group("seq")), "ckpt_dir": m.group("ckpt").strip(),
            }
    return None


def path_basename(path: str) -> str:
    """The file name of a path printed by either OS (the learner joins with the local separator)."""
    return re.split(r"[\\/]", path.strip())[-1]


# --------------------------------------------------------------------------------------------
# phase rules
# --------------------------------------------------------------------------------------------

def plateau_check(history: Sequence[Tuple[int, Optional[float]]], window_decisions: int,
                  min_improvement_pct: float, smooth_updates: int = 5) -> Tuple[bool, Dict[str, Any]]:
    """
    Has the loss stopped improving?

    history: (decisions, loss) per learner update, oldest first. The loss "now" is the mean of the
    last `smooth_updates` updates; the loss "then" is the mean of the `smooth_updates` updates
    ending at the newest update that is at least `window_decisions` old. It fires when the
    improvement from then to now is below `min_improvement_pct` percent of the old value (a loss
    that got worse counts as no improvement). It never fires before the history covers the window.
    """
    pts = [(d, v) for d, v in history if v is not None and math.isfinite(v)]
    if len(pts) < 2:
        return False, {"why": "fewer than two updates"}
    k = max(1, int(smooth_updates))
    d_now = pts[-1][0]
    cutoff = d_now - int(window_decisions)
    old = [p for p in pts if p[0] <= cutoff]
    new = [p for p in pts if p[0] > cutoff]
    if not old or not new:
        return False, {"why": "the history does not cover the window yet", "decisions_now": d_now}
    old_vals = [v for _, v in old[-k:]]
    new_vals = [v for _, v in new[-k:]]
    s_old = sum(old_vals) / len(old_vals)
    s_new = sum(new_vals) / len(new_vals)
    improvement = (s_old - s_new) / max(abs(s_old), 1e-12) * 100.0
    detail = {"then": round(s_old, 6), "now": round(s_new, 6), "improvement_pct": round(improvement, 3),
              "decisions_now": d_now, "decisions_then": old[-1][0]}
    return improvement < float(min_improvement_pct), detail


def phase_end_reason(ph: Dict[str, Any], decisions: int, history: Sequence[Tuple[int, Optional[float]]]) -> Optional[str]:
    """The reason the phase is over, or None. `ph` is one validated entry of config["phases"]."""
    max_d = int(ph.get("max_decisions") or 0)
    if max_d > 0 and decisions >= max_d:
        return f"reached {max_d:,} decisions ({decisions:,})"
    pl = ph.get("plateau")
    if pl and decisions >= int(ph.get("min_decisions") or 0):
        fired, d = plateau_check(history, pl["window_decisions"], pl["min_improvement_pct"], pl["smooth_updates"])
        if fired:
            return (f"plateau: {pl['metric']} went from {d['then']:.4f} to {d['now']:.4f} "
                    f"({d['improvement_pct']:.2f}% < {pl['min_improvement_pct']}%) over the last {pl['window_decisions']:,} decisions")
    return None


# --------------------------------------------------------------------------------------------
# checkpoints and pruning
# --------------------------------------------------------------------------------------------

Ckpt = collections.namedtuple("Ckpt", "name update mtime size")


def list_checkpoints(ckpt_dir: str) -> List[Ckpt]:
    """learner_update_NNNNNN.pt files, oldest first by modification time."""
    out: List[Ckpt] = []
    try:
        names = os.listdir(ckpt_dir)
    except OSError:
        return out
    for name in names:
        m = CKPT_RE.match(name)
        if not m:
            continue
        try:
            st = os.stat(os.path.join(ckpt_dir, name))
        except OSError:
            continue
        if not stat.S_ISREG(st.st_mode):
            continue
        out.append(Ckpt(name, int(m.group(1)), st.st_mtime, st.st_size))
    out.sort(key=lambda c: (c.mtime, c.name))
    return out


def newest_checkpoint(ckpt_dir: str) -> Optional[Ckpt]:
    files = list_checkpoints(ckpt_dir)
    return files[-1] if files else None


def select_checkpoints_to_prune(files: Iterable[Any], keep_newest: int, milestone_every: int = 0,
                                protected: Iterable[str] = (), keep_seconds: float = 0.0,
                                now: Optional[float] = None) -> Tuple[Dict[str, str], List[str]]:
    """
    Which checkpoints may go. Pure: it looks at names and times only and touches no file.

    files: Ckpt tuples or (name, mtime) pairs. Returns (keep, remove): `keep` maps every name that
    stays to the reason, `remove` lists the rest, oldest first.

    Kept: the newest `keep_newest` by modification time (the update number restarts at every
    change of phase, so it does not order them), every `milestone_every`-th update number, every
    name in `protected` (the last checkpoint of each finished phase and everything an evaluation
    names), and everything younger than `keep_seconds`. A name that is not
    learner_update_NNNNNN.pt is never offered for removal.
    """
    now = time.time() if now is None else now
    protected = set(protected)
    entries: List[Tuple[str, int, float]] = []
    keep: Dict[str, str] = {}
    for f in files:
        name, mtime = (f.name, f.mtime) if hasattr(f, "name") else (f[0], f[1])
        m = CKPT_RE.match(name)
        if not m:
            keep[name] = "not a checkpoint name"
            continue
        entries.append((name, int(m.group(1)), float(mtime)))
    entries.sort(key=lambda e: (e[2], e[0]))
    newest = {e[0] for e in entries[-max(1, int(keep_newest)):]}
    remove: List[str] = []
    for name, update, mtime in entries:
        if name in newest:
            keep[name] = "newest"
        elif name in protected:
            keep[name] = "protected"
        elif milestone_every and milestone_every > 0 and update > 0 and update % milestone_every == 0:
            keep[name] = "milestone"
        elif keep_seconds and now - mtime < keep_seconds:
            keep[name] = "recent"
        else:
            remove.append(name)
    return keep, remove


def checkpoints_named_in_eval(eval_dir: str, max_bytes: int = 8 << 20) -> set:
    """
    Checkpoint file names an evaluation refers to: the "checkpoint" field of eval/results.jsonl,
    plus any learner_update_NNNNNN.pt mentioned in the other small text files directly under
    eval/ (a queue or an in-progress marker the evaluator may keep there).
    """
    names: set = set()
    for rec in read_jsonl(os.path.join(eval_dir, "results.jsonl")):
        value = rec.get("checkpoint")
        if isinstance(value, str) and value:
            names.add(path_basename(value))
    try:
        listing = os.listdir(eval_dir)
    except OSError:
        return names
    for fn in listing:
        if not fn.endswith((".json", ".jsonl", ".txt")):
            continue
        path = os.path.join(eval_dir, fn)
        try:
            if not os.path.isfile(path) or os.path.getsize(path) > max_bytes:
                continue
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                names.update(CKPT_NAME_RE.findall(f.read()))
        except OSError:
            continue
    return names


def remove_checkpoint_file(path: str, ckpt_dir: str, expect: Optional[Tuple[float, int]] = None) -> None:
    """
    The only place in this program that removes a file, reached only when retention.enabled is
    true and retention.dry_run is false. It refuses anything that is not a regular
    learner_update_NNNNNN.pt directly inside the checkpoint directory, and refuses everything when
    the environment variable PSPPO_NO_DELETE is set.

    `expect` is the (modification time, size) the file had when it was chosen. The trainer writes
    a checkpoint under a name a finished phase already used, so between the choice and this call
    an old file picked for removal can have become the newest checkpoint: a file that is no
    longer the one that was chosen is refused.
    """
    if os.environ.get("PSPPO_NO_DELETE"):
        raise PermissionError("PSPPO_NO_DELETE is set: not removing anything")
    real = os.path.realpath(path)
    if os.path.islink(path) or os.path.dirname(real) != os.path.realpath(ckpt_dir):
        raise ValueError(f"refusing to remove {path}: not directly inside {ckpt_dir}")
    if not CKPT_RE.match(os.path.basename(real)) or not os.path.isfile(real):
        raise ValueError(f"refusing to remove {path}: not a checkpoint file")
    if expect is not None:
        st = os.stat(real)
        if st.st_mtime != expect[0] or st.st_size != expect[1]:
            raise ValueError(f"refusing to remove {path}: it was written again since it was chosen")
    os.remove(real)


# --------------------------------------------------------------------------------------------
# what the supervisor knows about the run (no processes in here: every method takes `now`)
# --------------------------------------------------------------------------------------------

def new_state(cfg: Dict[str, Any], paths: Paths, config_path: str) -> Dict[str, Any]:
    now = time.time()
    phases = {}
    for name in PHASES:
        phases[name] = {
            "status": "skipped" if cfg["phases"][name]["skip"] else "pending",
            "decisions": 0, "updates": 0, "lost_decisions": 0, "trainer_starts": 0,
            "started": None, "started_ts": None, "ended": None, "end_reason": None,
            "final_checkpoint": None, "final_confirmed": None, "final_copy": None,
        }
    return {
        "version": 1,
        "run_dir": paths.run,
        "config": os.path.abspath(config_path) if config_path else None,
        "created": utc_iso(now),
        "phase": first_phase(cfg),
        "status": "starting",
        "status_reason": "",
        "updated": utc_iso(now),
        "updated_ts": now,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "phases": phases,
        "decisions_total": 0,
        "updates_total": 0,
        "collected_total": 0,
        "segment": 0,
        "trainer": None,
        "servers": None,
        "failures": [],
        "restarts_total": 0,
        "recycles_total": 0,
        "quarantined": [],
        "fruitless_starts": 0,
        "nan_failures": [],
        "samples": [],
        "last_events": [],
        "limits": {},
        "newest_checkpoint": None,
        "checkpoints": None,
        "done": None,
        "done_ts": None,
        "shutdown_issued": None,      # when the shutdown command last returned 0
        "shutdown_failed": None,      # the last failed attempt, until one succeeds
        "shutdown_planned": None,     # when the shutdown after a failed run is due (on_failed "shutdown")
        "supervisor_errors": [],      # times of this program's own internal errors, last hour
        "leftover": None,
    }


class Tracker:
    """
    Turns the trainer's lines into counters, metrics records and health verdicts. It owns the
    "phases", "trainer", "samples" and total fields of the state dict it is given.
    """

    GAP_GRACE_SECONDS = 60.0   # how long a learner line may trail the telemetry that counted its update
    SAMPLE_KEEP_SECONDS = 2 * 3600 + 600

    def __init__(self, cfg: Dict[str, Any], state: Dict[str, Any],
                 on_metric: Callable[[dict], None], on_event: Callable[[str, dict], None]):
        self.cfg = cfg
        self.state = state
        self.on_metric = on_metric
        self.on_event = on_event
        self.phase: Optional[str] = None
        self.hist: Dict[int, Tuple[int, Optional[float]]] = {}
        self.fatal: Optional[Tuple[str, str, bool]] = None   # (kind, reason, try a graceful stop first)
        self._first_sample = True
        self._last_seen: Optional[int] = None
        self._metrics_update: Optional[int] = None
        self._tele_update_ts: Optional[float] = None
        self._last_sample_ts = 0.0

    # -- phase and trainer lifecycle ---------------------------------------------------------
    @property
    def ph(self) -> Dict[str, Any]:
        return self.state["phases"][self.phase]

    @property
    def t(self) -> Optional[Dict[str, Any]]:
        return self.state.get("trainer")

    def begin_phase(self, phase: str, now: float, history: Optional[Iterable[dict]] = None) -> None:
        """Called when the supervisor starts or resumes work on `phase`. `history` is the phase's
        records from metrics.jsonl, so the plateau rule survives a restart of this program."""
        self.phase = phase
        ph = self.ph
        if ph["status"] != "running":
            ph["status"] = "running"
        if not ph.get("started"):
            ph["started"], ph["started_ts"] = utc_iso(now), now
        self.hist = {}
        plateau = self.cfg["phases"][phase].get("plateau")
        if plateau and history:
            for rec in history:
                u, d = _as_int(rec.get("update")), _as_int(rec.get("decisions_phase"))
                if rec.get("phase") != phase or u is None or d is None:
                    continue
                v = rec.get(plateau["metric"])
                self._hist_add(u, d, v if _is_num(v) else None)
        self._totals()

    def begin_trainer(self, pid: int, now: float, log_rel: str, expect_resume: bool) -> Dict[str, Any]:
        self.state["segment"] = int(self.state.get("segment") or 0) + 1
        self.ph["trainer_starts"] = int(self.ph.get("trainer_starts") or 0) + 1
        self.fatal = None
        self._first_sample = True
        self._last_seen = None
        self._metrics_update = None
        self._tele_update_ts = None
        self.state["trainer"] = {
            "segment": self.state["segment"], "phase": self.phase, "pid": pid,
            "started": utc_iso(now), "started_ts": now, "log": log_rel, "expect_resume": bool(expect_resume),
            "running": True, "exit_code": None,
            "total_steps": None, "update": None, "active": None, "steps_per_update": None, "mode": None,
            "telemetry_lines": 0, "last_telemetry_ts": None,
            "progress_seen": False, "last_progress_ts": now,
            # last_update_ts is the watchdog's clock: it starts at this trainer's first report, before
            # any update. updates_seen says whether an update has actually arrived since then.
            "last_update_ts": None, "steps_at_last_update": None, "updates_seen": 0,
            "update_failures": 0, "last_update_error": None,
            "telemetry_errors": 0, "telemetry_error_since": None, "last_telemetry_error": None,
            "nonfinite_updates": 0, "nan_in_a_row": 0, "first_nan_update": None, "last_loss_nonfinite": False,
            "resumed_from": None, "phase_change": None, "fresh_init": False, "resume_failed": None,
            "final_checkpoint": None, "final_save_failed": False, "reached_minutes": False, "stop_ack": False,
            "worker_crashed": False,
        }
        return self.state["trainer"]

    # -- lines --------------------------------------------------------------------------------
    def feed(self, line: str, now: float) -> List[Dict[str, Any]]:
        """Takes one line of the trainer's output. Returns the records found in it."""
        events = parse_trainer_events(line)
        if self.t is not None:
            for ev in events:
                self._apply(ev, now)
        return events

    def _apply(self, ev: Dict[str, Any], now: float) -> None:
        t = self.t
        kind = ev["kind"]
        if kind == "telemetry":
            self._on_telemetry(ev, now)
        elif kind == "learner_update":
            self._on_update(ev, now)
        elif kind == "header":
            t["steps_per_update"] = ev["steps_per_update"]
            t["mode"] = ev["mode"]
            want = self.cfg["phases"][self.phase]["mode"]
            if ev["mode"] != want:
                self.on_event("mode_mismatch", {"phase": self.phase, "expected": want, "trainer": ev["mode"]})
        elif kind == "resumed":
            t["resumed_from"] = path_basename(ev["path"])
        elif kind == "phase_change":
            t["phase_change"] = f"{ev['old']} -> {ev['new']}"
            # The trainer starts a phase over, with its counters at zero, on whatever the newest
            # checkpoint holds. From an earlier phase that is the normal step forward. From a
            # LATER one it means this program's idea of the phase is behind the checkpoints (its
            # state was lost): going on would train the earlier phases again on top of them.
            came_from = self._phase_of_mode(ev["old"])
            if came_from is not None and PHASES.index(came_from) > PHASES.index(self.phase) and self.fatal is None:
                self.fatal = ("phase_backwards",
                              f"the newest checkpoint was written in the {came_from} phase (mode {ev['old']}), which comes after "
                              f"{self.phase}: the supervisor's state is behind the checkpoints", False)
        elif kind == "resume_failed":
            t["resume_failed"] = path_basename(ev["path"])
            self.fatal = ("resume_failed", f"the trainer could not load {t['resume_failed']}: {ascii_safe(ev['error'])[:200]}", False)
        elif kind == "init_policy":
            t["fresh_init"] = True
            if t.get("expect_resume") and self.fatal is None:
                self.fatal = ("fresh_init", "the trainer started from untrained weights although checkpoints exist", False)
        elif kind == "update_failed":
            t["update_failures"] += 1
            t["last_update_error"] = ascii_safe(ev["error"])[:300]
            if t["update_failures"] in (1, 10, 100) or t["update_failures"] % 1000 == 0:
                self.on_event("update_failed", {"phase": self.phase, "count": t["update_failures"], "error": t["last_update_error"]})
        elif kind == "learner_crashed":
            if self.fatal is None:
                self.fatal = ("learner_crashed", "the learner's training loop crashed", True)
        elif kind == "worker_crashed":
            t["worker_crashed"] = True
        elif kind == "final_saved":
            t["final_checkpoint"] = path_basename(ev["path"])
        elif kind == "final_save_failed":
            t["final_save_failed"] = True
        elif kind == "minutes_reached":
            t["reached_minutes"] = True
        elif kind == "stop_ack":
            t["stop_ack"] = True
        elif kind == "telemetry_error":
            t["telemetry_errors"] += 1
            t["last_telemetry_error"] = ascii_safe(ev["detail"])[:300]
            if t["telemetry_error_since"] is None:
                t["telemetry_error_since"] = now

    def _phase_of_mode(self, mode: str) -> Optional[str]:
        """The phase whose trainer runs in `mode`, or None for a mode no phase of this config uses."""
        for name in PHASES:
            if self.cfg["phases"][name]["mode"] == mode:
                return name
        return None

    def _on_telemetry(self, ev: Dict[str, Any], now: float) -> None:
        t, ph = self.t, self.ph
        t["telemetry_lines"] += 1
        t["last_telemetry_ts"] = now
        t["telemetry_error_since"] = None
        t["active"] = ev.get("active")
        steps, upd = ev.get("total_steps"), ev.get("update")
        if steps is None:
            return
        grew = False
        if self._first_sample:
            # The first number a trainer reports is where its counter stands after resuming: the
            # checkpoint's count in the same phase, zero after a change of phase or with no
            # checkpoint of this phase. Below our count means work since that checkpoint is gone.
            self._first_sample = False
            before = int(ph.get("decisions") or 0)
            if steps < before:
                lost = before - steps
                ph["lost_decisions"] = int(ph.get("lost_decisions") or 0) + lost
                self.on_event("rollback", {"phase": self.phase, "from": before, "to": steps, "lost": lost,
                                           "resumed_from": t.get("resumed_from")})
                if upd is not None:
                    for u in [u for u in self.hist if u > upd]:
                        del self.hist[u]
            self._metrics_update = upd
            t["last_update_ts"] = now
            t["steps_at_last_update"] = steps
        else:
            delta = steps - int(self._last_seen or 0)
            if delta > 0:
                grew = True
                self.state["collected_total"] = int(self.state.get("collected_total") or 0) + delta
            elif delta < 0:
                ph["lost_decisions"] = int(ph.get("lost_decisions") or 0) - delta
                self.on_event("counter_reset", {"phase": self.phase, "from": self._last_seen, "to": steps})
        self._last_seen = steps
        if grew:
            t["progress_seen"] = True
            t["last_progress_ts"] = now
        if upd is not None:
            if t["update"] is not None and upd > t["update"]:
                t["last_update_ts"] = now
                t["steps_at_last_update"] = steps
                t["updates_seen"] = int(t.get("updates_seen") or 0) + (upd - t["update"])
                self._tele_update_ts = now
            # Never backwards inside one trainer: a telemetry line is put together from several
            # actors' answers and can be printed after the learner's line for the next update.
            if t["update"] is None or upd > t["update"]:
                t["update"] = upd
                ph["updates"] = upd
        t["total_steps"] = steps
        ph["decisions"] = steps
        self._totals()

    def _on_update(self, ev: Dict[str, Any], now: float) -> None:
        t, ph = self.t, self.ph
        u = ev["update"]
        late = self._metrics_update is not None and u <= self._metrics_update
        t["last_update_ts"] = now
        if t["total_steps"] is not None:
            t["steps_at_last_update"] = t["total_steps"]
        if t["update"] is None or u > t["update"]:
            t["updates_seen"] = int(t.get("updates_seen") or 0) + (1 if t["update"] is None else u - t["update"])
            t["update"] = u
            ph["updates"] = u
            self._totals()
        if ev["nonfinite"]:
            t["nonfinite_updates"] += 1
            if t["nonfinite_updates"] in (1, 10, 100):
                self.on_event("nonfinite_loss", {"phase": self.phase, "update": u, "fields": ev["nonfinite"]})
        t["last_loss_nonfinite"] = bool(ev["nonfinite"])
        # A total loss that is not a number means the weights are gone: the updates that follow
        # cannot repair them, and every checkpoint from here on would store them.
        if "loss" in ev["nonfinite"]:
            t["nan_in_a_row"] += 1
            if t["first_nan_update"] is None:
                t["first_nan_update"] = u
            limit = int(self.cfg["watchdog"]["nan_updates_limit"])
            if limit and t["nan_in_a_row"] >= limit and self.fatal is None:
                self.fatal = ("nan_loss", f"the loss was not a number in {t['nan_in_a_row']} updates in a row "
                                          f"(updates {t['first_nan_update']} to {u})", False)
        else:
            t["nan_in_a_row"], t["first_nan_update"] = 0, None
        rec = self._record(u, now)
        rec.update(ev["values"])
        if ev["nonfinite"]:
            rec["nonfinite"] = ev["nonfinite"]
        if late:
            rec["late"] = True
        self.on_metric(rec)
        if not late:
            self._metrics_update = u
        plateau = self.cfg["phases"][self.phase].get("plateau")
        if plateau:
            self._hist_add(u, rec["decisions_phase"], ev["values"].get(plateau["metric"]))

    def _record(self, update: int, now: float) -> Dict[str, Any]:
        t, ph = self.t, self.ph
        return {
            "time": utc_iso(now), "wall_time": round(now, 3),
            "phase": self.phase, "update": update,
            "total_steps": t["total_steps"] if t["total_steps"] is not None else ph["decisions"],
            "decisions_phase": ph["decisions"],
            "decisions_total": self.state["decisions_total"],
            "updates_total": self.state["updates_total"],
            "segment": t["segment"],
        }

    def _hist_add(self, update: int, decisions: int, value: Optional[float]) -> None:
        # An update number at or below one already held means the trainer went back to a
        # checkpoint: what was recorded past that point belongs to weights that no longer exist.
        for u in [u for u in self.hist if u >= update]:
            del self.hist[u]
        self.hist[update] = (decisions, value)

    def history(self) -> List[Tuple[int, Optional[float]]]:
        return [self.hist[u] for u in sorted(self.hist)]

    def _totals(self) -> None:
        phases = self.state["phases"]
        self.state["decisions_total"] = sum(int(phases[p].get("decisions") or 0) for p in PHASES)
        self.state["updates_total"] = sum(int(phases[p].get("updates") or 0) for p in PHASES)

    # -- time-driven --------------------------------------------------------------------------
    def tick(self, now: float) -> None:
        """Writes the metrics records of updates whose "[learner] upd=" line never arrived (the
        telemetry counted them), and samples the collected-decisions counter for the speed figures."""
        t = self.t
        if (t and t["update"] is not None and self._metrics_update is not None and t["update"] > self._metrics_update
                and self._tele_update_ts is not None and now - self._tele_update_ts > self.GAP_GRACE_SECONDS):
            for u in range(self._metrics_update + 1, t["update"] + 1):
                rec = self._record(u, now)
                rec["losses_missing"] = True
                self.on_metric(rec)
            self._metrics_update = t["update"]
        self.sample(now)

    def sample(self, now: float, force: bool = False) -> None:
        every = float(self.cfg["reporting"]["sample_seconds"])
        if not force and now - self._last_sample_ts < every:
            return
        self._last_sample_ts = now
        samples = self.state.setdefault("samples", [])
        samples.append([round(now, 1), int(self.state.get("collected_total") or 0)])
        cut = now - self.SAMPLE_KEEP_SECONDS
        while samples and samples[0][0] < cut:
            samples.pop(0)

    @property
    def synced(self) -> bool:
        """True once the running trainer has reported its counter at least once."""
        return not self._first_sample

    def phase_rule(self) -> Optional[str]:
        # Not before the trainer has reported: until then the count is the previous trainer's,
        # and after a crash the new one may have fallen back to an older checkpoint.
        if self.phase is None or self._first_sample:
            return None
        return phase_end_reason(self.cfg["phases"][self.phase], int(self.ph.get("decisions") or 0), self.history())

    def health(self, now: float) -> Optional[Tuple[str, str, bool]]:
        """None while the trainer looks healthy, else (kind, reason, try a graceful stop first)."""
        t = self.t
        if t is None:
            return None
        if self.fatal is not None:
            return self.fatal
        w = self.cfg["watchdog"]
        stall = float(w["stall_minutes"]) * 60.0
        allowed = stall if t["progress_seen"] else max(stall, float(w["startup_grace_minutes"]) * 60.0)
        idle = now - float(t["last_progress_ts"])
        if idle > allowed:
            what = "no growth in total_steps" if t["telemetry_lines"] else "no telemetry at all"
            return ("stall", f"{what} for {idle / 60.0:.1f} minutes (limit {allowed / 60.0:.1f})", True)
        tele_limit = float(w["telemetry_error_minutes"]) * 60.0
        if tele_limit > 0 and t.get("telemetry_error_since") is not None and now - float(t["telemetry_error_since"]) > tele_limit:
            # train.py cannot reach one of its actors (a dead learner or inference actor looks like this)
            return ("telemetry_errors", f"the trainer's telemetry has failed for {(now - t['telemetry_error_since']) / 60.0:.1f} minutes "
                                        f"({t['telemetry_errors']} errors; last: {t.get('last_telemetry_error')})", True)
        upd_stall = float(w["update_stall_minutes"]) * 60.0
        if upd_stall > 0 and t["last_update_ts"] is not None and t["total_steps"] is not None:
            since = now - float(t["last_update_ts"])
            grown = int(t["total_steps"]) - int(t["steps_at_last_update"] or 0)
            need = 2 * int(t["steps_per_update"]) if t.get("steps_per_update") else 1
            if since > upd_stall and grown >= need:
                extra = f"; last error: {t['last_update_error']}" if t.get("last_update_error") else ""
                return ("update_stall", f"no learner update for {since / 60.0:.1f} minutes while {grown:,} decisions arrived "
                                        f"({t['update_failures']} failed updates{extra})", True)
        return None


# --------------------------------------------------------------------------------------------
# child processes
# --------------------------------------------------------------------------------------------

def _spawn_kwargs() -> Dict[str, Any]:
    # Own process group / session: a Ctrl+C in the terminal reaches this program only, and the
    # whole tree under a child can be stopped by the id of the child we started.
    if IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


class WinJob:
    """
    A Windows job object around one child and everything it starts.

    On Windows a process tree is only a chain of parent ids: once a parent has gone, nothing ties
    its children to us any more, and "taskkill /T" (which walks that chain, in a new process) can
    fail outright when the machine is short of memory. Seen on the laptop: the servers' launcher
    was killed, both Showdown servers stayed up, and the supervisor exited "stopped" with the
    ports still open. A job does not have these holes. Every process started inside it stays in
    it, whatever happens to its parent, one call from this process ends them all, and they also
    end when this process dies without having said goodbye (KILL_ON_JOB_CLOSE): a trainer without
    a supervisor is of no use and would block the ports for the next start.

    Jobs nest (Windows 8 and later), so the jobs Python's venv launcher and Ray make for their own
    children go inside this one.
    """

    _EXTENDED_LIMIT_INFORMATION = 9
    _BASIC_PROCESS_ID_LIST = 3
    _KILL_ON_JOB_CLOSE = 0x2000

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
        info.BasicLimitInformation.LimitFlags = self._KILL_ON_JOB_CLOSE
        if not k.SetInformationJobObject(self.handle, self._EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)):
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
            if self._k.QueryInformationJobObject(self.handle, self._BASIC_PROCESS_ID_LIST, ctypes.byref(buf), ctypes.sizeof(buf), None):
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


class Child:
    """A process we started, with its output copied to a log file and queued line by line."""

    def __init__(self, name: str, cmd: Sequence[str], cwd: str, env: Dict[str, str], log_path: str):
        self.name = name
        self.cmd = list(cmd)
        self.log_path = log_path
        self.q: "queue.Queue[str]" = queue.Queue()
        self.eof = False
        self.exit_seen_ts: Optional[float] = None
        self.job: Optional[WinJob] = None
        if IS_WINDOWS:
            try:
                self.job = WinJob()
            except Exception as e:   # without it the tree is still stopped through taskkill, as before
                log.warning("no job object for %s (%r): its process tree will be stopped through taskkill only", name, e)
        try:
            self.proc = subprocess.Popen(self.cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **_spawn_kwargs())
        except BaseException:
            if self.job is not None:
                self.job.close()
            raise
        self.pid = self.proc.pid
        if self.job is not None:
            try:
                # At once, before the child has had the time to start anything of its own.
                self.job.assign(self.proc._handle)   # type: ignore[attr-defined]
            except Exception as e:
                log.warning("could not put %s (pid %s) into its job object (%r): taskkill only", name, self.pid, e)
                self.job.close()
                self.job = None
        self._thread = threading.Thread(target=self._pump, name=f"{name}-output", daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        # Reading must go on whatever happens to the log file. If it stopped, the child would
        # block on a full pipe a few minutes later and be taken for a stalled trainer, on every
        # restart. So a failed write (a full disk) costs the copy of that line and nothing else.
        f = None
        complained = False
        try:
            try:
                f = open(self.log_path, "ab")
            except Exception as e:
                complained = True
                self.q.put(f"[pipeline] cannot open {self.log_path} ({e!r}): the output of {self.name} is read but not kept\n")
            for raw in self.proc.stdout:  # type: ignore[union-attr]
                if f is not None:
                    try:
                        f.write(raw)
                        f.flush()
                    except Exception as e:
                        if not complained:
                            complained = True
                            self.q.put(f"[pipeline] could not write to {self.log_path} ({e!r}): the output of {self.name} is still read\n")
                self.q.put(raw.decode("utf-8", "replace"))
        except Exception as e:  # the reader must never take the supervisor down
            self.q.put(f"[pipeline] output reader stopped: {e!r}\n")
        finally:
            self.eof = True
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass

    def lines(self) -> List[str]:
        out: List[str] = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out

    def alive(self) -> bool:
        return self.proc.poll() is None

    def returncode(self) -> Optional[int]:
        return self.proc.poll()

    def output_done(self, patience: float = 10.0) -> bool:
        """True once the process has exited and its output has been read to the end (or, when
        something else still holds the pipe open, `patience` seconds after the exit)."""
        if self.alive():
            return False
        if self.eof:
            return True
        if self.exit_seen_ts is None:
            self.exit_seen_ts = time.time()
        return time.time() - self.exit_seen_ts > patience

    def leftover(self) -> List[int]:
        """Process ids that are still alive under this child (Windows: in its job; elsewhere: the
        leader's own id when its process group still has a member)."""
        if IS_WINDOWS:
            pids = self.job.pids() if self.job is not None else []
            if not pids and self.alive():
                pids = [self.pid]
            return pids
        try:
            os.killpg(self.pid, 0)
        except (ProcessLookupError, PermissionError):
            return []
        return [self.pid]

    def kill_tree(self, grace: float = 10.0, hard: bool = False) -> List[int]:
        """
        Stops the process and everything under it, by the id of the process we started, also when
        that process has already gone. Returns the ids of what is still alive afterwards: normally
        none.

        hard=True gives the process no chance to do anything on its way out. On POSIX the default
        is SIGTERM first, and Ray turns a SIGTERM to train.py into a SystemExit: train.py then
        saves a final checkpoint of whatever its learner holds. That must not happen when what it
        holds is known to be bad (untrained weights after a failed resume, NaN weights), so those
        stops go straight to SIGKILL. On Windows the kill never was anything else.
        """
        if IS_WINDOWS:
            return self._kill_tree_windows(grace)
        self._kill_tree_posix(grace, hard)
        left = self.leftover()
        deadline = time.time() + 5.0
        while left and time.time() < deadline:   # SIGKILL is not instant: the group empties a moment later
            time.sleep(0.1)
            left = self.leftover()
        if left:
            log.error("the process group of %s (pid %s) still has members after SIGKILL", self.name, self.pid)
        return left

    def _kill_tree_windows(self, grace: float) -> List[int]:
        # 1. The job: one call from this process, nothing to walk, and it takes the processes whose
        #    parent has already gone.
        if self.job is not None and not self.job.terminate() and self.job.pids():
            log.warning("TerminateJobObject for %s (pid %s) was refused", self.name, self.pid)
        # 2. taskkill as well while the root is there: it reaches a process that was started before
        #    the root was in the job, and it is all there is when no job could be made.
        if self.alive():
            try:
                r = subprocess.run(["taskkill", "/PID", str(self.pid), "/T", "/F"], stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
                if r.returncode != 0 and self.alive():
                    log.warning("taskkill of %s (pid %s) returned %s: %s", self.name, self.pid, r.returncode,
                                ascii_safe((r.stderr or b"").decode("utf-8", "replace")).strip()[:200])
            except Exception as e:
                log.warning("taskkill of %s (pid %s) failed: %r", self.name, self.pid, e)
        try:
            self.proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            try:
                self.proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                log.error("%s (pid %s) is still alive after kill", self.name, self.pid)
        # 3. Look again. Ending a process is not instant, so give it a moment, and ask once more.
        deadline = time.time() + min(max(grace, 2.0), 15.0)
        again = False
        while self.job is not None and self.job.pids() and time.time() < deadline:
            if not again and time.time() > deadline - min(max(grace, 2.0), 15.0) / 2.0:
                again = True
                self.job.terminate()
            time.sleep(0.2)
        left = self.leftover()
        if left:
            log.error("%d processes under %s (pid %s) are still alive after the kill: %s. Stop them by hand: taskkill /PID <pid> /T /F",
                      len(left), self.name, self.pid, left[:20])
        if self.job is not None:
            # The last handle on a KILL_ON_JOB_CLOSE job: closing it ends whatever is still in it.
            self.job.close()
            self.job = None
        return left

    def _kill_tree_posix(self, grace: float, hard: bool = False) -> None:
        steps = ((signal.SIGKILL, 5.0),) if hard else ((signal.SIGTERM, grace), (signal.SIGKILL, 5.0))
        for sig, wait in steps:
            try:
                os.killpg(self.pid, sig)   # the child leads its own session, so its group id is its pid
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self.proc.wait(timeout=wait)
            except subprocess.TimeoutExpired:
                continue
            if sig == signal.SIGTERM:
                # the leader is gone; make sure nothing it started is left holding a port or the GPU
                try:
                    os.killpg(self.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            break

    def join_output(self, timeout: float = 5.0) -> None:
        self._thread.join(timeout=timeout)


class RunLock:
    """An OS file lock on RUN/pipeline.lock, so two supervisors can never drive one run. The
    lock dies with the process: no stale lock file to clean up."""

    def __init__(self, path: str):
        self.path = path
        self._f = None

    def acquire(self) -> bool:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        f = open(self.path, "a+b")
        try:
            if IS_WINDOWS:
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            return False
        self._f = f
        return True

    def release(self) -> None:
        if self._f is None:
            return
        try:
            if IS_WINDOWS:
                import msvcrt
                self._f.seek(0)
                msvcrt.locking(self._f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._f.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            self._f.close()
        finally:
            self._f = None


# --------------------------------------------------------------------------------------------
# the supervisor
# --------------------------------------------------------------------------------------------

class StartError(Exception):
    pass


class StopRequested(Exception):
    pass


Outcome = collections.namedtuple("Outcome", "kind what reason graceful")


class Supervisor:
    def __init__(self, cfg: Dict[str, Any], run_dir: str, config_path: str = ""):
        self.cfg = cfg
        self.paths = Paths(run_dir)
        self.config_path = config_path
        self.state: Dict[str, Any] = {}
        self.tracker: Optional[Tracker] = None
        self.trainer: Optional[Child] = None
        self.servers: Optional[Child] = None
        self.lock = RunLock(self.paths.lock)
        self._metrics: Optional[JsonlWriter] = None
        self._events: Optional[JsonlWriter] = None
        self._stop_reason: Optional[str] = None
        self._stop_count = 0
        self._last_state_write = 0.0
        self._last_progress_log = 0.0
        self._last_disk_check = 0.0
        self._last_retention = 0.0
        self._last_ckpt_scan = 0.0
        self._last_prune_report: Optional[Tuple[str, ...]] = None
        self._server_restart_seen: Optional[str] = None

    # -- plumbing -----------------------------------------------------------------------------
    def event(self, name: str, detail: Optional[dict] = None) -> None:
        now = time.time()
        rec = {"time": utc_iso(now), "ts": round(now, 3), "event": name, "phase": self.state.get("phase")}
        if detail:
            rec.update({k: v for k, v in detail.items() if k not in ("time", "ts", "event")})
        if self._events is not None:
            try:
                self._events.write(rec)
            except Exception as e:
                log.error("could not write the event %s: %r", name, e)
        tail = self.state.setdefault("last_events", [])
        tail.append(rec)
        del tail[:-30]
        log.info("event %s %s", name, ascii_safe(json.dumps(detail, sort_keys=True)) if detail else "")

    def _metric(self, rec: dict) -> None:
        if self._metrics is not None:
            try:
                self._metrics.write(rec)
            except Exception as e:
                log.error("could not write a metrics record: %r", e)

    def save_state(self, force: bool = True) -> None:
        now = time.time()
        if not force and now - self._last_state_write < float(self.cfg["reporting"]["state_seconds"]):
            return
        self._last_state_write = now
        self.state["updated"], self.state["updated_ts"] = utc_iso(now), now
        self.state["pid"] = os.getpid()
        try:
            atomic_write_json(self.paths.state, self.state)
        except Exception as e:
            log.error("could not write state.json (it will be retried): %r", e)

    def set_status(self, status: str, reason: str = "") -> None:
        if self.state.get("status") != status or self.state.get("status_reason") != reason:
            self.state["status"], self.state["status_reason"] = status, reason
            log.info("status %s%s", status, f": {ascii_safe(reason)}" if reason else "")
        self.save_state()

    def request_stop(self, reason: str) -> None:
        self._stop_count += 1
        if self._stop_reason is None:
            self._stop_reason = reason

    def _on_signal(self, signum, _frame) -> None:
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        self.request_stop(f"signal {name}")

    def _install_signals(self) -> None:
        names = ["SIGINT", "SIGTERM"] + (["SIGBREAK"] if IS_WINDOWS else ["SIGHUP"])
        for name in names:
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, self._on_signal)
            except (ValueError, OSError):
                pass

    def _check_stop_file(self) -> None:
        if os.path.exists(self.paths.pipeline_stop):
            moved = move_aside(self.paths.pipeline_stop, self.paths.stop_files)
            self.request_stop(f"stop file {self.paths.rel(self.paths.pipeline_stop)}" + (f" (moved to {self.paths.rel(moved)})" if moved else ""))

    def stopping(self) -> bool:
        if self._stop_reason is None:
            self._check_stop_file()
        return self._stop_reason is not None

    def sleep(self, seconds: float) -> bool:
        """Sleeps in small steps. Returns True if a stop was requested meanwhile."""
        deadline = time.time() + max(0.0, seconds)
        step = min(1.0, float(self.cfg["watchdog"]["poll_seconds"]))
        while True:
            if self.stopping():
                return True
            left = deadline - time.time()
            if left <= 0:
                return False
            time.sleep(min(step, left))
            self.save_state(force=False)

    # -- startup ------------------------------------------------------------------------------
    def _load_state(self) -> None:
        st = None
        if os.path.exists(self.paths.state):
            try:
                st = read_json(self.paths.state)
                if not isinstance(st, dict) or st.get("phase") not in PHASES + ("done",) or not isinstance(st.get("phases"), dict):
                    raise ValueError("not a state file")
            except Exception as e:
                moved = move_aside(self.paths.state, self.paths.logs)
                log.error("state.json could not be used (%r); moved to %s, rebuilding from events.jsonl", e, moved)
                st = self._state_from_events()
        else:
            # No state.json. If this run directory has a history (the file was moved aside and the
            # process died before writing a new one, or it was removed by hand), the phase comes
            # from events.jsonl: starting at the first phase would train it again, and every
            # phase after it, on top of the checkpoints of a later one.
            st = self._state_from_events()
            if st is not None:
                log.error("no state.json, but events.jsonl has the history of this run: rebuilt the state from it (phase %s)", st["phase"])
        fresh = new_state(self.cfg, self.paths, self.config_path)
        if st is None:
            self.state = fresh
            existing = list_checkpoints(self.paths.checkpoints)
            if existing:
                log.warning("no state.json but %d checkpoints already exist in %s: the trainer will resume from the newest",
                            len(existing), self.paths.checkpoints)
            return
        # keep what is there, add fields a newer version of this file introduced
        for key, value in fresh.items():
            st.setdefault(key, value)
        for name in PHASES:
            cur = st["phases"].setdefault(name, fresh["phases"][name])
            for key, value in fresh["phases"][name].items():
                cur.setdefault(key, value)
        st["run_dir"], st["config"], st["host"] = fresh["run_dir"], fresh["config"], fresh["host"]
        self.state = st

    def _state_from_events(self) -> Optional[Dict[str, Any]]:
        events = read_jsonl(self.paths.events)
        if not events:
            return None
        st = new_state(self.cfg, self.paths, self.config_path)
        for ev in events:
            name = ev.get("event")
            if name == "phase_start" and ev.get("name") in PHASES:
                st["phase"] = ev["name"]
                st["phases"][ev["name"]]["status"] = "running"
            elif name == "phase_end" and ev.get("name") in PHASES:
                ph = st["phases"][ev["name"]]
                ph.update({"status": "done", "decisions": int(ev.get("decisions") or 0), "updates": int(ev.get("updates") or 0),
                           "end_reason": ev.get("reason"), "final_checkpoint": ev.get("final_checkpoint"),
                           "final_copy": ev.get("final_copy"), "ended": ev.get("time")})
            elif name == "done":
                st["phase"] = "done"
                if _is_num(ev.get("ts")):
                    st["done_ts"], st["done"] = float(ev["ts"]), ev.get("time")
            elif name == "shutdown":
                # Without this a finished run whose state was lost would wait for the final
                # evaluation and shut the machine down a second time.
                st["shutdown_issued"] = ev.get("time") or utc_iso()
            elif name == "shutdown_failed":
                st["shutdown_issued"] = None
            elif name == "failed":
                st["status"], st["status_reason"] = "failed", str(ev.get("reason") or "")
            elif name == "failed_cleared":
                st["status"], st["status_reason"] = "stopped", ""
        st["status"] = st["status"] if st["status"] == "failed" else "stopped"
        return st

    def _limits(self) -> Dict[str, Any]:
        w, c = self.cfg["watchdog"], self.cfg
        return {
            "stall_minutes": w["stall_minutes"], "startup_grace_minutes": w["startup_grace_minutes"],
            "update_stall_minutes": w["update_stall_minutes"], "max_restarts_per_hour": w["max_restarts_per_hour"],
            "min_free_gb": c["disk"]["min_free_gb"], "state_seconds": c["reporting"]["state_seconds"],
            "on_done": c["on_done"], "on_failed": c["on_failed"],
            "phases": {n: {"min_decisions": c["phases"][n]["min_decisions"], "max_decisions": c["phases"][n]["max_decisions"],
                           "plateau": c["phases"][n]["plateau"], "skip": c["phases"][n]["skip"]} for n in PHASES},
        }

    def _setup_logging(self) -> None:
        fmt = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
        fmt.converter = time.gmtime
        log.setLevel(logging.INFO)
        log.propagate = False
        for h in list(log.handlers):
            log.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(errors="backslashreplace")  # type: ignore[union-attr]
            except Exception:
                pass
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(sh)
        fh = logging.FileHandler(self.paths.pipeline_log, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)

    def run(self) -> int:
        for d in (self.paths.run, self.paths.logs, self.paths.checkpoints, self.paths.eval):
            os.makedirs(d, exist_ok=True)
        self._setup_logging()
        if not self.lock.acquire():
            log.error("another pipeline.py already holds %s. Not starting a second one.", self.paths.lock)
            return EXIT_LOCKED
        code = EXIT_ERROR
        try:
            self._install_signals()
            # A stop request from before this process existed is not for this process.
            stale = move_aside(self.paths.pipeline_stop, self.paths.stop_files)
            if stale:
                log.warning("an old stop request was lying in the run directory; moved to %s", self.paths.rel(stale))
            self._load_state()
            self._events = JsonlWriter(self.paths.events, sync=True)
            self._metrics = JsonlWriter(self.paths.metrics)
            self.state["limits"] = self._limits()
            self.tracker = Tracker(self.cfg, self.state, self._metric, self.event)
            log.info("pipeline.py pid %s, run %s, config %s, phase %s, status %s",
                     os.getpid(), self.paths.run, self.config_path, self.state["phase"], self.state["status"])
            try:
                code = self._main()
            except KeyboardInterrupt:
                self.request_stop("KeyboardInterrupt")
                code = self._stop_everything_cleanly()
        except Exception:
            # A bug in this program, not in the trainer. Leave with code 1 (systemd starts it again
            # and the run resumes), but let the trainer save what it has first.
            error = ascii_safe(repr(sys.exc_info()[1]))
            log.exception("internal error in the supervisor")
            try:
                self.event("supervisor_error", {"error": error[:500]})
                if self.trainer is not None:
                    self._stop_trainer(True, float(self.cfg["trainer"]["failure_stop_timeout_seconds"]), "supervisor error")
                self._stop_servers()
            except Exception:
                log.exception("error while stopping after an internal error")
            code = EXIT_ERROR
            gave_up = False
            try:
                if self.state.get("status") not in ("failed", "done", "disk_low"):
                    gave_up = self._note_supervisor_error(error)
                    if not gave_up:
                        self.state["status"], self.state["status_reason"] = "error", "supervisor error: " + error[:300]
            except Exception:
                log.exception("error while recording an internal error")
            if gave_up:
                # This program itself keeps failing: starting it again does not help, and systemd
                # would stop trying a few starts later and leave the machine up with nothing on it.
                code = EXIT_FAILED
                try:
                    code = self._after_giving_up(EXIT_FAILED)
                except Exception:
                    log.exception("error while acting on on_failed")
        finally:
            self._kill_children()
            try:
                if self.state:
                    if self.state.get("trainer"):
                        self.state["trainer"]["running"] = False
                    self.save_state()
            except Exception:
                pass
            for w in (self._events, self._metrics):
                if w is not None:
                    w.close()
            self.lock.release()
            log.info("pipeline.py exits with code %s", code)
        return code

    def _main(self) -> int:
        st = self.state
        if st.get("status") == "failed":
            log.error("this run is marked failed: %s", ascii_safe(st.get("status_reason")))
            log.error("fix the cause, then run: pipeline.py --run-dir %s --clear-failed", self.paths.run)
            return EXIT_FAILED
        while True:
            phase = st["phase"]
            if phase == "done":
                return self._on_done()
            if self.cfg["phases"][phase]["skip"]:
                # the config changed since the state was written: this phase is now skipped
                st["phases"][phase]["status"] = "skipped"
                st["phase"] = next_phase(self.cfg, phase)
                self.event("phase_skipped", {"name": phase})
                continue
            result = self._run_phase(phase)
            if result == "phase_end":
                continue
            if result == "stop":
                return EXIT_OK
            if result == "failed":
                return self._after_giving_up(EXIT_FAILED)
            if result == "disk_low":
                return self._after_giving_up(EXIT_DISK)
            raise AssertionError(result)

    def _note_supervisor_error(self, error: str) -> bool:
        """
        Counts an internal error of this program. True when that is one too many (more than
        watchdog.max_restarts_per_hour within an hour): the run is then marked failed, like a
        trainer that keeps failing, instead of being started again until systemd gives up.
        """
        st = self.state
        now = time.time()
        recent = [ts for ts in (st.get("supervisor_errors") or []) if _is_num(ts) and now - float(ts) < 3600.0] + [round(now, 1)]
        st["supervisor_errors"] = recent
        limit = int(self.cfg["watchdog"]["max_restarts_per_hour"])
        if len(recent) <= limit:
            return False
        self._mark_failed(f"the supervisor itself failed {len(recent)} times within an hour (max_restarts_per_hour is {limit}); "
                          f"the last error: {error[:200]}")
        return True

    def _after_giving_up(self, code: int) -> int:
        """
        The run was marked failed, or stopped because the disk is full, by this process, just now.
        Nothing starts it again by itself (systemd is told not to, and at the next boot the unit's
        start is skipped), so with on_failed "idle" the machine stays up doing nothing. With
        "shutdown" this waits on_failed_grace_minutes, in which stopping this program cancels it,
        and then runs shutdown_command. Returns the exit code either way.

        A start on a run that is already marked failed does not come here: a person who boots
        the machine to look at it is not shut out again.
        """
        cfg, st = self.cfg, self.state
        if cfg["on_failed"] != "shutdown":
            return code
        grace = float(cfg["on_failed_grace_minutes"]) * 60.0
        why = str(st.get("status") or "failed")
        st["shutdown_planned"] = utc_iso(time.time() + grace)
        self.event("shutdown_planned", {"why": why, "in_seconds": grace, "command": cfg["shutdown_command"]})
        log.error("on_failed is \"shutdown\": running %s in %.0f minutes, so the machine does not stay up with nothing running. "
                  "Stop this program to cancel that.", quote_cmd(cfg["shutdown_command"]), grace / 60.0)
        self.save_state()
        if self.sleep(grace) or not self._shut_the_machine_down(why, {}):
            st["shutdown_planned"] = None
            self.event("shutdown_cancelled", {"reason": self._stop_reason, "why": why})
            self.save_state()
        return code

    def _shut_the_machine_down(self, why: str, detail: Dict[str, Any]) -> bool:
        """
        Runs shutdown_command until it reports success. True once it has: the machine is then on
        its way down. False when a stop request ended the retries.

        state["shutdown_issued"] is set by a command that returned 0 and by nothing else. A
        command that fails (no passwordless sudo, a timeout) is tried again every
        shutdown_retry_minutes for as long as this program runs, and each failure is in
        state["shutdown_failed"] and in events.jsonl, where status.py reads it: a machine that
        should be off and is not must not look finished.
        """
        cfg, st = self.cfg, self.state
        retry = float(cfg["shutdown_retry_minutes"]) * 60.0
        attempt = 0
        while True:
            attempt += 1
            error = None
            try:
                r = subprocess.run(cfg["shutdown_command"], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
                said = ascii_safe((r.stdout or "") + (r.stderr or ""))[:300].strip()
                log.info("shutdown command returned %s %s", r.returncode, said)
                if r.returncode != 0:
                    error = f"exit code {r.returncode}" + (f": {said}" if said else "")
            except Exception as e:
                error = ascii_safe(repr(e))[:300]
            if error is None:
                st["shutdown_issued"], st["shutdown_failed"], st["shutdown_planned"] = utc_iso(), None, None
                self.event("shutdown", dict(detail, command=cfg["shutdown_command"], why=why, attempt=attempt))
                self.save_state()
                return True
            st["shutdown_failed"] = {"time": utc_iso(), "attempts": attempt, "error": error, "why": why}
            log.error("the shutdown command failed (%s), attempt %d: the machine is still up. Trying again in %.0f minutes.",
                      error, attempt, retry / 60.0)
            self.event("shutdown_failed", {"error": error, "attempt": attempt, "why": why, "retry_in_seconds": retry})
            self.save_state()
            if self.sleep(retry):
                return False

    # -- one phase ----------------------------------------------------------------------------
    def _run_phase(self, phase: str) -> str:
        cfg, st, tr = self.cfg, self.state, self.tracker
        assert tr is not None
        first_start = st["phases"][phase]["status"] != "running"
        history = read_jsonl(self.paths.metrics) if cfg["phases"][phase].get("plateau") and not first_start else None
        tr.begin_phase(phase, time.time(), history)
        if first_start:
            self.event("phase_start", {"name": phase, "rule": describe_rule(cfg, phase)})
        else:
            self.event("phase_resume", {"name": phase, "decisions": st["phases"][phase]["decisions"],
                                        "updates": st["phases"][phase]["updates"]})
        while True:
            if self.stopping():
                return self._mark_stopped()
            low = self._disk_low()
            if low:
                return self._mark_disk_low(low)
            trainer_started = False
            try:
                self.set_status("starting", f"starting the servers and the trainer for {phase}")
                self._start_servers(phase)
                self._start_trainer(phase)
                trainer_started = True
                self.set_status("running", "")
                out = self._watch(phase)
            except StartError as e:
                out = Outcome("failure", "start_failed", str(e), False)
            except StopRequested:
                out = Outcome("stop", "stop", self._stop_reason or "stop requested", True)
            # Did this start produce a single decision? Starts that never do are counted below.
            progressed = trainer_started and bool((st.get("trainer") or {}).get("progress_seen"))
            if progressed:
                st["fruitless_starts"] = 0

            if out.kind == "phase_end":
                self.set_status("stopping", f"{phase} is over: {out.reason}")
                self.event("stop_requested", {"why": "phase_end", "reason": out.reason})
                final = self._stop_trainer(True, float(cfg["trainer"]["stop_timeout_seconds"]), out.reason)
                self._stop_servers()
                confirmed = final is not None
                if final is None:
                    final = self._phase_checkpoint(phase)
                if final is None:
                    out = Outcome("failure", "no_phase_checkpoint",
                                  f"{phase} reached its end ({out.reason}) but no checkpoint of this phase was saved", False)
                else:
                    self._finish_phase(phase, out.reason, final, confirmed)
                    return "phase_end"

            if out.kind == "stop":
                self.set_status("stopping", out.reason)
                self.event("stop_requested", {"why": "stop", "reason": out.reason})
                final = self._stop_trainer(True, float(cfg["trainer"]["stop_timeout_seconds"]), out.reason)
                self._stop_servers()
                return self._mark_stopped(final)

            if out.kind == "disk_low":
                self.set_status("stopping", out.reason)
                self.event("stop_requested", {"why": "disk_low", "reason": out.reason})
                self._stop_trainer(True, float(cfg["trainer"]["stop_timeout_seconds"]), out.reason)
                self._stop_servers()
                return self._mark_disk_low(out.reason)

            if out.kind == "recycle":
                self.set_status("stopping", out.reason)
                self.event("recycle", {"reason": out.reason})
                self._stop_trainer(True, float(cfg["trainer"]["stop_timeout_seconds"]), out.reason)
                self._stop_servers()
                st["recycles_total"] = int(st.get("recycles_total") or 0) + 1
                continue

            assert out.kind == "failure", out
            self.set_status("restarting", out.reason)
            self.event("failure", {"what": out.what, "reason": ascii_safe(out.reason),
                                   "trainer_log": (st.get("trainer") or {}).get("log")})
            self._stop_trainer(out.graceful, float(cfg["trainer"]["failure_stop_timeout_seconds"]), out.reason)
            self._stop_servers()
            if out.what == "resume_failed" and not self._quarantine((st.get("trainer") or {}).get("resume_failed")):
                return self._mark_failed(f"checkpoints keep failing to load ({out.reason}); "
                                         f"{len(st.get('quarantined') or [])} are already set aside in {self.paths.rel(self.paths.quarantine)}")
            if out.what == "nan_loss" and not self._after_nan_loss(out):
                return "failed"
            if out.what == "phase_backwards":
                # Not a thing a restart changes: every start would find the same checkpoint.
                return self._mark_failed(f"{out.reason}. Nothing was saved over it. If state.json was lost, mark the phases that "
                                         f"are finished \"skip\": true in the config (or restore state.json), then --clear-failed")
            if not progressed:
                st["fruitless_starts"] = int(st.get("fruitless_starts") or 0) + 1
                limit = int(cfg["watchdog"]["max_fruitless_starts"])
                if limit and st["fruitless_starts"] >= limit:
                    return self._mark_failed(f"{st['fruitless_starts']} starts in a row ended without a single decision "
                                             f"(max_fruitless_starts is {limit}); the last one: {out.reason}")
            delay = self._after_failure(out)
            if delay is None:
                return "failed"
            if self.sleep(delay):
                return self._mark_stopped()

    def _watch(self, phase: str) -> Outcome:
        cfg, st, tr = self.cfg, self.state, self.tracker
        assert tr is not None and self.trainer is not None
        poll = float(cfg["watchdog"]["poll_seconds"])
        recycle_hours = cfg["phases"][phase]["recycle_hours"]
        if recycle_hours is None:
            recycle_hours = cfg["watchdog"]["recycle_hours"]
        started = time.time()
        while True:
            now = time.time()
            self._drain(now)
            if self.stopping():
                return Outcome("stop", "stop", self._stop_reason or "stop requested", True)
            if self.trainer.output_done():
                self._drain(time.time())
                rc = self.trainer.returncode()
                t = tr.t or {}
                if rc == 0 and t.get("reached_minutes"):
                    return Outcome("recycle", "minutes", "the trainer reached its --minutes and left by itself", True)
                why = "; a rollout worker crashed" if t.get("worker_crashed") else ""
                return Outcome("failure", "trainer_exit", f"the trainer exited by itself with code {rc}{why}", False)
            if self.servers is not None and not self.servers.alive():
                return Outcome("failure", "servers_exit", f"the server launcher exited with code {self.servers.returncode()}", True)
            if self._server_restart_seen and cfg["servers"]["restart_trainer_on_server_restart"]:
                msg, self._server_restart_seen = self._server_restart_seen, None
                return Outcome("failure", "server_restart", f"a Showdown server went down and was restarted ({msg}): "
                                                            f"its players are gone, starting over", True)
            # A phase is not ended, and a planned restart not made, on weights that may be ruined
            # (a loss that is not a number, not yet often enough in a row to be called): both would
            # go on from the trainer's last checkpoint. Either the loss recovers and they follow an
            # update later, or it is a failure and handled as one.
            suspect = self._suspect_weights()
            reason = None if suspect else tr.phase_rule()
            if reason:
                return Outcome("phase_end", "rule", reason, True)
            bad = tr.health(now)
            if bad:
                return Outcome("failure", bad[0], bad[1], bad[2])
            if recycle_hours and not suspect and now - started > float(recycle_hours) * 3600.0:
                return Outcome("recycle", "recycle_hours", f"planned restart after {recycle_hours} hours", True)
            if now - self._last_disk_check > float(cfg["disk"]["check_seconds"]):
                low = self._disk_low()
                if low:
                    return Outcome("disk_low", "disk", low, True)
            self._retention(now)
            tr.tick(now)
            self._note_checkpoints(force=False)
            self._progress_log(now)
            self.save_state(force=False)
            time.sleep(poll)

    def _drain(self, now: float) -> None:
        if self.trainer is not None and self.tracker is not None:
            for line in self.trainer.lines():
                try:
                    self.tracker.feed(line, now)
                except Exception as e:
                    log.error("could not use a trainer line (%r): %s", e, ascii_safe(line)[:300])
        if self.servers is not None:
            for line in self.servers.lines():
                if "exited with" in line and "restarting" in line:
                    self._server_restart_seen = ascii_safe(line.strip())[:200]
                    log.warning("servers: %s", self._server_restart_seen)

    def _progress_log(self, now: float) -> None:
        if now - self._last_progress_log < float(self.cfg["reporting"]["progress_log_seconds"]):
            return
        self._last_progress_log = now
        st, t = self.state, (self.state.get("trainer") or {})
        ph = st["phases"][st["phase"]] if st["phase"] in PHASES else {}
        rate = rate_from_samples(st.get("samples") or [], 600.0, now)
        last_upd = f"{(now - t['last_update_ts']):.0f}s ago" if t.get("last_update_ts") and t.get("updates_seen") else "none since this trainer started"
        covered = "-" if rate is None else (f"{rate[1]:.0f}s" if rate[1] < 120 else f"{rate[1] / 60.0:.0f}m")
        log.info("phase=%s decisions=%s (overall %s, %s/s over the last %s) updates=%s last_update=%s active=%s failed_updates=%s",
                 st["phase"], f"{int(ph.get('decisions') or 0):,}", f"{int(st.get('decisions_total') or 0):,}",
                 "-" if rate is None else f"{rate[0]:.0f}", covered,
                 ph.get("updates"), last_upd, t.get("active"), t.get("update_failures"))

    def _note_checkpoints(self, force: bool = True) -> None:
        now = time.time()
        if not force and now - self._last_ckpt_scan < 10.0:
            return
        self._last_ckpt_scan = now
        files = list_checkpoints(self.paths.checkpoints)
        if files:
            c = files[-1]
            self.state["newest_checkpoint"] = {"name": c.name, "mtime": round(c.mtime, 1), "size": c.size}
        self.state["checkpoints"] = {"count": len(files), "bytes": sum(c.size for c in files)}

    # -- servers ------------------------------------------------------------------------------
    def _child_env(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        env = dict(os.environ)
        env.update({"PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"})
        env.update(extra or {})
        return env

    def _start_servers(self, phase: str) -> None:
        cfg = self.cfg
        ph = cfg["phases"][phase]
        s = phase_servers(cfg, ph)
        ports = phase_ports(cfg, ph)
        host = s["host"]
        self._stop_servers()
        self._server_restart_seen = None
        deadline = time.time() + float(s["port_free_timeout_seconds"])
        while True:
            busy = [p for p in ports if port_open(host, p, 0.3)]
            if not busy:
                break
            if time.time() > deadline:
                prev = (self.state.get("servers") or {}).get("pid")
                left = ((self.state.get("leftover") or {}).get("servers") or {}).get("pids") or []
                hint = f"; processes of our own servers that could not be stopped earlier: {left}" if left else ""
                raise StartError(f"port {busy[0]} on {host} is already in use by another process, so the servers cannot be "
                                 f"restarted there (ports wanted: {ports_spec(ports)}; the previous server launcher had pid {prev}{hint})")
            if self.stopping():
                raise StopRequested()
            time.sleep(0.5)
            self.save_state(force=False)      # still here: status.py calls a state that is not refreshed DEAD
        cmd = build_server_command(cfg, self.paths, phase)
        seg = int(self.state.get("segment") or 0) + 1
        log_path = unique_new_file(self.paths.logs, f"servers_{seg:04d}_{phase}_{utc_stamp()}", ".log")
        log.info("starting the servers: %s", ascii_safe(quote_cmd(cmd)))
        try:
            self.servers = Child("servers", cmd, cfg["repo_dir"], self._child_env(), log_path)
        except OSError as e:
            raise StartError(f"could not start the servers ({quote_cmd(cmd)}): {e}")
        self.state["servers"] = {"pid": self.servers.pid, "ports": ports, "host": host, "started": utc_iso(),
                                 "log": self.paths.rel(log_path), "running": True}
        self.save_state()
        deadline = time.time() + float(s["ready_timeout_seconds"])
        while True:
            if not self.servers.alive():
                raise StartError(f"the server launcher exited with code {self.servers.returncode()} before the servers were up "
                                 f"(see {self.paths.rel(log_path)})")
            waiting = [p for p in ports if not port_open(host, p, 0.5)]
            if not waiting:
                break
            if time.time() > deadline:
                raise StartError(f"ports {ports_spec(waiting)} did not accept connections within {s['ready_timeout_seconds']} seconds "
                                 f"(see {self.paths.rel(log_path)})")
            if self.stopping():
                raise StopRequested()
            time.sleep(0.5)
            self.save_state(force=False)
        log.info("all %d servers answer on %s:%s", len(ports), host, ports_spec(ports))
        # The ports were free before this start and answer now: nothing of an earlier stop is on them.
        self._note_leftover("servers", [], [])
        if self.sleep(float(s["settle_seconds"])):
            raise StopRequested()
        # What the launcher printed while the servers were coming up is not about a trainer: none
        # is connected yet. A server that fell over once during the start and was started again
        # before every port answered must not restart the trainer that is about to be started.
        for line in self.servers.lines():
            if "exited with" in line and "restarting" in line:
                log.warning("servers, while they were starting: %s", ascii_safe(line.strip())[:200])
        self._server_restart_seen = None

    def _stop_servers(self) -> None:
        srv, self.servers = self.servers, None
        if srv is None:
            return
        s = self.cfg["servers"]
        info = self.state.get("servers") or {}
        log.info("stopping the servers (pid %s)", srv.pid)
        left = srv.kill_tree(grace=float(s["stop_timeout_seconds"]))
        srv.join_output()
        ports, host = info.get("ports") or [], info.get("host") or s["host"]
        deadline = time.time() + 15.0
        while time.time() < deadline and any(port_open(host, p, 0.2) for p in ports):
            time.sleep(0.3)
        still = [p for p in ports if port_open(host, p, 0.2)]
        self._note_leftover("servers", left, still)
        if self.state.get("servers"):
            self.state["servers"]["running"] = False
        self.save_state()

    def _note_leftover(self, what: str, pids: Sequence[int], ports: Sequence[int] = ()) -> None:
        """
        Records what a stop could not stop, or that the last stop of `what` left nothing. Stopping
        must not be believed to have worked when it has not: a server that stays up keeps its port
        (the next start then fails on it) and a trainer that stays up keeps the GPU.
        """
        book = dict(self.state.get("leftover") or {})
        if not pids and not ports:
            if book.pop(what, None) is not None:
                self.state["leftover"] = book or None
            return
        book[what] = {"time": utc_iso(), "pids": [int(p) for p in list(pids)[:50]], "ports": [int(p) for p in ports]}
        self.state["leftover"] = book
        log.error("%s: still running after it was stopped: %s%s", what,
                  f"processes {book[what]['pids']}" if pids else "",
                  (" and " if pids else "") + f"ports {ports_spec(ports)} still answer" if ports else "")
        self.event("leftover", {"what": what, "pids": book[what]["pids"], "ports": book[what]["ports"]})

    # -- trainer ------------------------------------------------------------------------------
    def _start_trainer(self, phase: str) -> None:
        cfg, tr = self.cfg, self.tracker
        assert tr is not None
        stale = move_aside(self.paths.trainer_stop, self.paths.stop_files)
        if stale:
            log.warning("a stop file from an earlier run was in the way; moved to %s", self.paths.rel(stale))
        cmd = build_trainer_command(cfg, self.paths, phase)
        newest = newest_checkpoint(self.paths.checkpoints)
        seg = int(self.state.get("segment") or 0) + 1
        log_path = unique_new_file(self.paths.logs, f"trainer_{seg:04d}_{phase}_{utc_stamp()}", ".log")
        log.info("starting the trainer: %s", ascii_safe(quote_cmd(cmd)))
        try:
            self.trainer = Child("trainer", cmd, cfg["repo_dir"], self._child_env(cfg["trainer"]["env"]), log_path)
        except OSError as e:
            raise StartError(f"could not start the trainer ({quote_cmd(cmd)}): {e}")
        tr.begin_trainer(self.trainer.pid, time.time(), self.paths.rel(log_path), expect_resume=newest is not None)
        self.event("trainer_start", {"name": phase, "segment": self.state["segment"], "pid": self.trainer.pid,
                                     "log": self.paths.rel(log_path), "newest_checkpoint": newest.name if newest else None})
        self.save_state()

    def _stop_trainer(self, graceful: bool, timeout: float, why: str) -> Optional[str]:
        """
        Stops the trainer. Gracefully means: create the stop file, wait for train.py to leave its
        telemetry loop and save a checkpoint, and only when that takes longer than `timeout`
        terminate it. Returns the name of the checkpoint the trainer reported saving on its way
        out (and that exists), else None. The stop file is renamed into logs/stop_files afterwards.
        """
        child, tr = self.trainer, self.tracker
        if child is None:
            return None
        t = (tr.t if tr else None) or {}
        asked = time.time()
        # Whatever the reason for this stop (a phase that ended, a stop request, a full disk, a
        # server that went down): a trainer whose weights are known or feared to be bad is not
        # asked to save them. Its final checkpoint would be the newest file, the one every later
        # start resumes from.
        no_save = self._suspect_weights() if child.alive() else None
        if graceful and no_save:
            log.warning("the trainer (pid %s) is terminated without a final checkpoint: %s", child.pid, ascii_safe(no_save)[:300])
            graceful = False
        if graceful and child.alive():
            tmp = self.paths.trainer_stop + ".tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(json.dumps({"time": utc_iso(asked), "reason": ascii_safe(why)[:300], "by": os.getpid()}) + "\n")
                os.replace(tmp, self.paths.trainer_stop)
                log.info("asked the trainer (pid %s) to stop through %s; waiting up to %.0f s", child.pid,
                         self.paths.rel(self.paths.trainer_stop), timeout)
            except OSError as e:
                log.error("could not create the stop file (%r); the trainer will be terminated without a final save", e)
                timeout = 0.0
            # One stop request is what a clean stop looks like (it may be why we are here, or it
            # may arrive while a phase is being closed). One more than that means "now".
            stops_at_entry = self._stop_count
            patience = 0 if stops_at_entry else 1
            while child.alive() and time.time() - asked < timeout:
                self._drain(time.time())
                self._check_stop_file()
                if self._stop_count - stops_at_entry > patience:
                    log.warning("a second stop request arrived: not waiting for the trainer any longer")
                    break
                self.save_state(force=False)
                time.sleep(min(0.5, float(self.cfg["watchdog"]["poll_seconds"])))
        killed = False
        if child.alive():
            killed = True
            log.warning("terminating the trainer (pid %s)%s", child.pid,
                        " because it did not stop in time" if graceful else "")
        # Whether it left by itself or not: take down whatever is left under it (Ray's processes).
        # Not gracefully means no save: no SIGTERM, which train.py answers with a final checkpoint.
        left = child.kill_tree(grace=float(self.cfg["trainer"]["kill_timeout_seconds"]), hard=not graceful)
        child.join_output()
        self._note_leftover("trainer", left)
        self._drain(time.time())
        rc = child.returncode()
        final = t.get("final_checkpoint")
        if final and not os.path.isfile(os.path.join(self.paths.checkpoints, final)):
            log.warning("the trainer reported saving %s but it is not in %s", final, self.paths.checkpoints)
            final = None
        used = move_aside(self.paths.trainer_stop, self.paths.stop_files)
        if t:
            t["running"], t["exit_code"] = False, rc
        self.trainer = None
        self._note_checkpoints()
        self.event("trainer_exit", {"name": t.get("phase"), "segment": t.get("segment"), "code": rc, "graceful": bool(graceful),
                                    "killed": killed, "final_checkpoint": final, "seconds_to_stop": round(time.time() - asked, 1),
                                    "stop_file": self.paths.rel(used) if used else None,
                                    "total_steps": t.get("total_steps"), "update": t.get("update"),
                                    "no_save_because": ascii_safe(no_save)[:300] if no_save else None})
        self.save_state()
        return final

    def _suspect_weights(self) -> Optional[str]:
        """
        Why the running trainer must not write another checkpoint, or None. Its weights are known
        to be bad when the tracker has a verdict that asks for a stop without a save (the newest
        checkpoint would not load and it started from untrained weights, the loss is not a number,
        the checkpoint it loaded belongs to a later phase), and feared to be bad from the first
        update whose loss is not a number: the learner has no guard against that, the updates
        that follow it are made on the same weights, and watchdog.nan_updates_limit only decides
        when it is called a failure.
        """
        tr = self.tracker
        t = (tr.t if tr is not None else None) or {}
        if tr is not None and tr.fatal is not None and not tr.fatal[2]:
            return tr.fatal[1]
        if int(self.cfg["watchdog"]["nan_updates_limit"]) > 0 and int(t.get("nan_in_a_row") or 0) > 0:
            return (f"the loss was not a number in its last {t['nan_in_a_row']} update(s) "
                    f"(from update {t.get('first_nan_update')})")
        return None

    def _phase_checkpoint(self, phase: str) -> Optional[str]:
        """The newest checkpoint written since `phase` began, when the trainer did not confirm a
        final save. None when this phase has not produced a checkpoint at all."""
        started = self.state["phases"][phase].get("started_ts")
        newest = newest_checkpoint(self.paths.checkpoints)
        if newest is None or started is None or newest.mtime < float(started):
            return None
        return newest.name

    def _finish_phase(self, phase: str, reason: str, final: str, confirmed: bool) -> None:
        st = self.state
        ph = st["phases"][phase]
        now = time.time()
        copy_rel = None
        src = os.path.join(self.paths.checkpoints, final)
        dst = os.path.join(self.paths.phase_final, f"{phase}_{final}")
        try:
            os.makedirs(self.paths.phase_final, exist_ok=True)
            if not os.path.exists(dst):
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
            copy_rel = self.paths.rel(dst)
        except OSError as e:
            log.error("could not keep a copy of the last %s checkpoint: %r", phase, e)
        if not confirmed:
            log.warning("the trainer did not confirm a final save; %s continues from %s, the newest checkpoint of the phase", phase, final)
        ph.update({"status": "done", "ended": utc_iso(now), "end_reason": reason, "final_checkpoint": final,
                   "final_confirmed": bool(confirmed), "final_copy": copy_rel})
        nxt = next_phase(self.cfg, phase)
        self.event("phase_end", {"name": phase, "reason": reason, "decisions": ph["decisions"], "updates": ph["updates"],
                                 "final_checkpoint": final, "final_confirmed": bool(confirmed), "final_copy": copy_rel, "next": nxt})
        st["phase"] = nxt
        st["trainer"] = None
        if nxt == "done":
            # in the same write as the phase, so nobody reads "done" with a status from before it
            st["status"], st["status_reason"] = "done", "every phase is finished"
        self.save_state()

    # -- failures -----------------------------------------------------------------------------
    def _after_failure(self, out: Outcome) -> Optional[float]:
        """Records a failure. Returns the seconds to wait before the restart, or None when the
        restart limit is reached and the run is now marked failed."""
        st, w = self.state, self.cfg["watchdog"]
        now = time.time()
        recent = [ts for ts in (st.get("failures") or []) if now - float(ts) < 3600.0] + [round(now, 1)]
        st["failures"] = recent
        limit = int(w["max_restarts_per_hour"])
        if len(recent) > limit:
            self._mark_failed(f"{len(recent)} failures within an hour (max_restarts_per_hour is {limit}); the last one: {out.reason}")
            return None
        backoff = w["backoff_seconds"]
        delay = float(backoff[min(len(recent) - 1, len(backoff) - 1)])
        st["restarts_total"] = int(st.get("restarts_total") or 0) + 1
        self.event("restart", {"after": out.what, "wait_seconds": delay, "failures_last_hour": len(recent), "limit": limit})
        return delay

    def _quarantine(self, name: Optional[str]) -> bool:
        """Moves a checkpoint that would not load out of the trainer's sight (a rename into
        checkpoints/quarantine, nothing is deleted) so the restart resumes from the one before.
        False when the limit is reached: then something other than one torn file is wrong."""
        st = self.state
        done = st.setdefault("quarantined", [])
        now = time.time()
        # The limit is for checkpoints that would not load, lately: files set aside after a NaN
        # loss are another matter with a limit of its own (_after_nan_loss), and one torn file a
        # week is not "something other than one torn file". An entry without a time is counted.
        unreadable = [q for q in done if isinstance(q, dict) and q.get("why") != "nan_loss"
                      and now - float(q["ts"] if _is_num(q.get("ts")) else now) < 24 * 3600.0]
        if len(unreadable) >= int(self.cfg["watchdog"]["max_quarantined_checkpoints"]):
            return False
        if not name or not CKPT_RE.match(name):
            return True
        src = os.path.join(self.paths.checkpoints, name)
        moved = move_aside(src, self.paths.quarantine)
        if moved:
            done.append({"name": name, "moved_to": self.paths.rel(moved), "time": utc_iso(now), "ts": round(now, 1)})
            self.event("checkpoint_quarantined", {"name": name, "moved_to": self.paths.rel(moved)})
        return True

    def _after_nan_loss(self, out: Outcome) -> bool:
        """
        The loss became NaN: the trainer was terminated without a final save. Every checkpoint
        this trainer wrote from the first such update on holds the ruined weights, so those are
        moved to checkpoints/quarantine (renamed, not deleted) and the restart resumes from the
        one before. False, with the run marked failed, when this is the second time within six
        hours: then a restart does not help and it needs a person.
        """
        st = self.state
        now = time.time()
        t = st.get("trainer") or {}
        first, started = t.get("first_nan_update"), float(t.get("started_ts") or now)
        files = list_checkpoints(self.paths.checkpoints)
        # by this trainer (not a file of the same number from an earlier phase), at or after the first bad update
        bad = {c.name: "written after the loss became NaN" for c in files
               if first is not None and c.update >= int(first) and c.mtime >= started - 1.0}
        # When every update this trainer made was NaN, the weights were already ruined in the
        # checkpoint it loaded (written inside the grace of an earlier trainer): without setting
        # that one aside too, the restart loads it again and the second NaN ends the run. It
        # stays where it is when it is the only one left, so a restart never begins from nothing.
        resumed = t.get("resumed_from")
        if (resumed and resumed not in bad and int(t.get("updates_seen") or 0) <= int(t.get("nan_in_a_row") or 0)
                and any(c.name == resumed and c.mtime < started for c in files)
                and any(c.name != resumed and c.name not in bad for c in files)):
            bad[resumed] = "the trainer that loaded it had a NaN loss from its first update"
        for c in files:
            if c.name in bad:
                moved = move_aside(os.path.join(self.paths.checkpoints, c.name), self.paths.quarantine)
                if moved:
                    st.setdefault("quarantined", []).append({"name": c.name, "moved_to": self.paths.rel(moved), "time": utc_iso(now),
                                                             "ts": round(now, 1), "why": "nan_loss"})
                    self.event("checkpoint_quarantined", {"name": c.name, "moved_to": self.paths.rel(moved), "why": bad[c.name]})
        recent = [ts for ts in (st.get("nan_failures") or []) if now - float(ts) < 6 * 3600.0] + [round(now, 1)]
        st["nan_failures"] = recent
        if len(recent) >= 2:
            self._mark_failed(f"the loss became NaN again after a restart from an earlier checkpoint ({out.reason}). "
                              f"Checkpoints written since are in {self.paths.rel(self.paths.quarantine)}")
            return False
        return True

    def _mark_failed(self, reason: str) -> str:
        self.event("failed", {"reason": ascii_safe(reason)})
        self.set_status("failed", ascii_safe(reason))
        log.error("the run is marked failed and will not be restarted: %s", ascii_safe(reason))
        return "failed"

    def _mark_stopped(self, final: Optional[str] = None) -> str:
        self.event("stopped", {"reason": self._stop_reason, "final_checkpoint": final})
        self.set_status("stopped", self._stop_reason or "")
        return "stop"

    def _mark_disk_low(self, reason: str) -> str:
        self.event("disk_low", {"reason": reason})
        self.set_status("disk_low", reason)
        return "disk_low"

    def _stop_everything_cleanly(self) -> int:
        try:
            self._stop_trainer(True, float(self.cfg["trainer"]["stop_timeout_seconds"]), self._stop_reason or "stop")
            self._stop_servers()
            if self.state.get("status") not in ("failed", "done", "disk_low"):
                self._mark_stopped()
        except Exception:
            log.exception("error while stopping")
        return EXIT_OK

    def _kill_children(self) -> None:
        for name in ("trainer", "servers"):
            child = getattr(self, name)
            if child is None:
                continue
            try:
                log.warning("stopping a leftover %s process (pid %s)", name, child.pid)
                # a trainer that must not save (see _suspect_weights) gets no SIGTERM here either
                hard = name == "trainer" and child.alive() and self._suspect_weights() is not None
                left = child.kill_tree(grace=10.0, hard=hard)
                child.join_output(2.0)
                self._note_leftover(name, left)
            except Exception:
                log.exception("could not stop the %s process", name)
            setattr(self, name, None)
        try:
            leftover = move_aside(self.paths.trainer_stop, self.paths.stop_files)
            if leftover:
                log.info("moved a leftover stop file to %s", self.paths.rel(leftover))
        except Exception:
            pass

    # -- disk and retention -------------------------------------------------------------------
    def _disk_low(self) -> Optional[str]:
        self._last_disk_check = time.time()
        need = float(self.cfg["disk"]["min_free_gb"])
        try:
            free = free_gb(self.paths.run)
        except OSError as e:
            log.error("could not read the free disk space: %r", e)
            return None
        self.state["disk_free_gb"] = round(free, 2)
        if free >= need:
            return None
        if self.cfg["retention"]["enabled"]:
            self._retention(time.time(), force=True)
            free = free_gb(self.paths.run)
            self.state["disk_free_gb"] = round(free, 2)
            if free >= need:
                return None
        return f"{free:.1f} GB free under {self.paths.run}, below min_free_gb {need:g}"

    def protected_checkpoints(self) -> set:
        names = set()
        for name in PHASES:
            final = self.state["phases"][name].get("final_checkpoint")
            if final:
                names.add(final)
        names |= checkpoints_named_in_eval(self.paths.eval)
        t = self.state.get("trainer") or {}
        if t.get("resumed_from"):
            names.add(t["resumed_from"])
        return names

    def _retention(self, now: float, force: bool = False) -> None:
        r = self.cfg["retention"]
        if not r["enabled"]:
            return
        if not force and now - self._last_retention < float(r["check_minutes"]) * 60.0:
            return
        self._last_retention = now
        files = list_checkpoints(self.paths.checkpoints)
        _keep, remove = select_checkpoints_to_prune(files, r["keep_newest"], r["milestone_every"], self.protected_checkpoints(),
                                                    float(r["keep_hours"]) * 3600.0, now)
        if not remove:
            return
        sizes = {c.name: c.size for c in files}
        listed = {c.name: (c.mtime, c.size) for c in files}
        if r["dry_run"]:
            if tuple(remove) != self._last_prune_report:
                self._last_prune_report = tuple(remove)
                self.event("prune_dry_run", {"would_remove": remove, "mb": round(sum(sizes[n] for n in remove) / 1e6, 1)})
            return
        removed, errors = [], []
        for name in remove:
            try:
                # only the file that was chosen: the trainer may have written this name again meanwhile
                remove_checkpoint_file(os.path.join(self.paths.checkpoints, name), self.paths.checkpoints, listed[name])
                removed.append(name)
            except Exception as e:
                errors.append(f"{name}: {e!r}")
        self.event("prune", {"removed": removed, "mb": round(sum(sizes[n] for n in removed) / 1e6, 1),
                             "kept": len(files) - len(removed), "errors": errors[:5]})

    # -- done ---------------------------------------------------------------------------------
    def _on_done(self) -> int:
        cfg, st = self.cfg, self.state
        self._kill_children()
        if not st.get("done_ts"):
            st["done_ts"], st["done"] = time.time(), utc_iso()
            self.event("done", {"decisions_total": st.get("decisions_total"), "updates_total": st.get("updates_total")})
        st["trainer"] = None
        self.set_status("done", "every phase is finished")
        if cfg["on_done"] == "shutdown":
            if st.get("shutdown_issued"):
                log.info("the shutdown command was already run at %s; not running it again", st["shutdown_issued"])
            else:
                wait = float(cfg["on_done_wait_minutes"]) * 60.0
                log.info("waiting up to %.0f minutes for %s before running %s", wait / 60.0,
                         self.paths.rel(self.paths.final_done), quote_cmd(cfg["shutdown_command"]))
                deadline = time.time() + wait
                found = False
                while True:
                    if self.stopping():
                        self.event("stopped", {"reason": self._stop_reason, "while": "waiting for the final evaluation"})
                        self.save_state()
                        return EXIT_OK
                    try:
                        found = os.path.getmtime(self.paths.final_done) >= float(st["done_ts"]) - 60.0
                    except OSError:
                        found = False
                    if found or time.time() >= deadline:
                        break
                    time.sleep(min(1.0, float(cfg["watchdog"]["poll_seconds"])))
                    self.save_state(force=False)
                grace = float(cfg["on_done_grace_minutes"]) * 60.0
                log.info("%s; running the shutdown command in %.0f seconds (stop this program to cancel)",
                         "the final evaluation is in" if found else "the final evaluation did not arrive in time", grace)
                if self.sleep(grace):
                    self.event("stopped", {"reason": self._stop_reason, "while": "about to shut down"})
                    self.save_state()
                    return EXIT_OK
                if not self._shut_the_machine_down("done", {"final_evaluation": found}):
                    self.event("stopped", {"reason": self._stop_reason, "while": "the shutdown command kept failing"})
                    self.save_state()
                    return EXIT_OK
        log.info("the run is done. Idling until this program is stopped.")
        while not self.stopping():
            time.sleep(min(1.0, float(cfg["watchdog"]["poll_seconds"])))
            self.save_state(force=False)
        self.save_state()
        return EXIT_OK


def rate_from_samples(samples: Sequence[Sequence[float]], window: float, now: float) -> Optional[Tuple[float, float]]:
    """
    Decisions a second over the last `window` seconds from [[time, collected], ...] samples.
    Returns (rate, seconds actually covered), or None with fewer than two samples. Time without
    samples counts as time without decisions, so a dead trainer shows as a falling rate.
    """
    pts = [(float(s[0]), float(s[1])) for s in samples if isinstance(s, (list, tuple)) and len(s) >= 2]
    if len(pts) < 2:
        return None
    start = now - window
    if pts[0][0] >= start:
        t0, v0 = pts[0]                      # the samples do not reach back that far
    else:
        t0, v0 = start, pts[-1][1]           # nothing since the window began
        for a, b in zip(pts, pts[1:]):
            if a[0] <= start < b[0]:
                v0 = a[1] + (b[1] - a[1]) * (start - a[0]) / (b[0] - a[0])
                break
    span = now - t0
    if span <= 0:
        return None
    return max(0.0, pts[-1][1] - v0) / span, span


# --------------------------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------------------------

def dry_run(cfg: Dict[str, Any], run_dir: str) -> int:
    paths = Paths(run_dir)
    w, r = cfg["watchdog"], cfg["retention"]
    print(f"run directory   {paths.run}")
    print(f"repo            {cfg['repo_dir']}")
    print(f"checkpoints     {paths.checkpoints}")
    print(f"stop file       {paths.trainer_stop}")
    print()
    for name in PHASES:
        ph = cfg["phases"][name]
        print(f"== {name}" + ("  (skipped)" if ph["skip"] else ""))
        if ph["skip"]:
            print()
            continue
        ports = phase_ports(cfg, ph)
        print(f"  servers ({len(ports)}):  {quote_cmd(build_server_command(cfg, paths, name))}")
        print(f"  trainer:       {quote_cmd(build_trainer_command(cfg, paths, name))}")
        print(f"  rule:          {describe_rule(cfg, name)}")
        rh = ph["recycle_hours"] if ph["recycle_hours"] is not None else w["recycle_hours"]
        if rh:
            print(f"  planned restart every {rh} hours")
        print()
    print("== watchdog")
    print(f"  restarts the servers and the trainer when total_steps does not grow for {w['stall_minutes']} minutes ({w['startup_grace_minutes']} after a start),")
    print(f"  when no learner update arrives for {w['update_stall_minutes']} minutes while decisions do, when the trainer's telemetry fails for")
    print(f"  {w['telemetry_error_minutes']} minutes, when the loss is NaN in {w['nan_updates_limit']} updates in a row, when a Showdown server goes down, or when the trainer exits.")
    print(f"  waits {w['backoff_seconds']} seconds before restart 1, 2, ... The run is marked failed (and left alone) after more than")
    print(f"  {w['max_restarts_per_hour']} failures in an hour, {w['max_fruitless_starts']} starts in a row without a single decision, or a second NaN restart.")
    print(f"  stops cleanly when less than {cfg['disk']['min_free_gb']} GB is free under the run directory.")
    print("== checkpoints")
    if r["enabled"]:
        print(f"  pruning ON{' (dry run: only logged)' if r['dry_run'] else ''}: keeps the newest {r['keep_newest']}, every update divisible by "
              f"{r['milestone_every'] or '-'}, everything younger than {r['keep_hours']} hours, the last of each finished phase, "
              f"and everything named under eval/.")
    else:
        print("  pruning OFF: every checkpoint is kept.")
    print("== when ppo is done")
    if cfg["on_done"] == "shutdown":
        print(f"  waits up to {cfg['on_done_wait_minutes']} minutes for {paths.final_done}, then {cfg['on_done_grace_minutes']} more, then runs: {quote_cmd(cfg['shutdown_command'])}")
        print(f"  (again every {cfg['shutdown_retry_minutes']} minutes for as long as that command fails)")
    else:
        print("  stops the trainer and the servers and idles.")
    print("== when the run is marked failed, or stopped because the disk is full")
    if cfg["on_failed"] == "shutdown":
        print(f"  waits {cfg['on_failed_grace_minutes']} minutes (stopping this program cancels it), then runs: {quote_cmd(cfg['shutdown_command'])}")
    else:
        print("  exits and leaves the machine as it is: up, with nothing running on it.")
    return EXIT_OK


def can_start(cfg: Dict[str, Any], run_dir: str) -> int:
    """For systemd's ExecCondition: 0 when starting would do something, 1 when it would not."""
    paths = Paths(run_dir)
    if not os.path.exists(paths.state):
        print("startable: no state.json yet, a new run")
        return 0
    try:
        st = read_json(paths.state)
        status, phase = st.get("status"), st.get("phase")
    except Exception as e:
        print(f"startable: state.json is unreadable ({e!r}); the supervisor rebuilds it from events.jsonl")
        return 0
    if status == "failed":
        print(f"not startable: the run is marked failed ({ascii_safe(st.get('status_reason'))}). "
              f"Fix it, then: pipeline.py --run-dir {paths.run} --clear-failed")
        return 1
    if phase == "done":
        if cfg["on_done"] == "shutdown" and not st.get("shutdown_issued"):
            print("startable: the run is done and the shutdown has not been issued yet")
            return 0
        print("not startable: the run is done")
        return 1
    if status == "disk_low":
        try:
            free = free_gb(paths.run)
        except OSError:
            free = 0.0
        if free < float(cfg["disk"]["min_free_gb"]):
            print(f"not startable: {free:.1f} GB free, min_free_gb is {cfg['disk']['min_free_gb']}")
            return 1
    print(f"startable: phase {phase}, status {status}")
    return 0


def stop_running(run_dir: str, wait: float) -> int:
    paths = Paths(run_dir)
    lock = RunLock(paths.lock)
    if not os.path.isdir(paths.run):
        print(f"no run directory at {paths.run}")
        return EXIT_ERROR
    if lock.acquire():
        lock.release()
        print("no supervisor is running on this run directory")
        return EXIT_OK
    tmp = paths.pipeline_stop + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps({"time": utc_iso(), "by": os.getpid()}) + "\n")
    os.replace(tmp, paths.pipeline_stop)
    print(f"asked the supervisor to stop ({paths.pipeline_stop}); waiting up to {wait:.0f} s")
    deadline = time.time() + wait
    while time.time() < deadline:
        if lock.acquire():
            lock.release()
            print("the supervisor has stopped")
            try:
                left = (read_json(paths.state) or {}).get("leftover") or {}
            except (OSError, ValueError):
                left = {}
            for what, info in left.items():
                print(f"BUT {what} could not be stopped completely: processes {info.get('pids')}, ports still answering {info.get('ports')}. "
                      f"Stop them by their ids before the next start.")
            return EXIT_ERROR if left else EXIT_OK
        time.sleep(0.5)
    print("the supervisor is still running")
    return EXIT_ERROR


def clear_failed(run_dir: str) -> int:
    paths = Paths(run_dir)
    lock = RunLock(paths.lock)
    if not os.path.exists(paths.state):
        print(f"no state.json in {paths.run}")
        return EXIT_ERROR
    if not lock.acquire():
        print("a supervisor is running on this run directory; stop it first")
        return EXIT_LOCKED
    try:
        st = read_json(paths.state)
        if st.get("status") != "failed":
            print(f"the run is not marked failed (status {st.get('status')}); nothing to do")
            return EXIT_OK
        was = st.get("status_reason")
        st["status"], st["status_reason"], st["failures"] = "stopped", "", []
        st["fruitless_starts"], st["nan_failures"], st["supervisor_errors"] = 0, [], []
        st["shutdown_planned"] = None
        st["updated"], st["updated_ts"] = utc_iso(), time.time()
        w = JsonlWriter(paths.events, sync=True)
        w.write({"time": utc_iso(), "ts": round(time.time(), 3), "event": "failed_cleared", "phase": st.get("phase"), "was": was})
        w.close()
        atomic_write_json(paths.state, st)
        print(f"cleared. The next start resumes phase {st.get('phase')}.")
        return EXIT_OK
    finally:
        lock.release()


def _json_list(text: Optional[str]) -> Optional[List[str]]:
    if text is None:
        return None
    try:
        value = json.loads(text)
    except ValueError:
        return shlex.split(text, posix=not IS_WINDOWS)
    if not isinstance(value, list):
        raise ConfigError("a command given on the command line must be a JSON list of strings")
    return [str(x) for x in value]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="the run's JSON config (cloud/run_vm.json, cloud/run_laptop_test.json)")
    ap.add_argument("--run-dir", required=True, help="the run directory (checkpoints, logs, state.json, metrics.jsonl)")
    ap.add_argument("--dry-run", action="store_true", help="print the server and trainer command of every phase and the rules, then exit")
    ap.add_argument("--can-start", action="store_true", help="exit 0 if a start would do anything, 1 if the run is failed or done (for systemd's ExecCondition)")
    ap.add_argument("--stop", action="store_true", help="ask the supervisor running on this run directory to stop cleanly, and wait for it")
    ap.add_argument("--wait", type=float, default=600.0, help="with --stop: how long to wait, in seconds")
    ap.add_argument("--clear-failed", action="store_true", help="make a run that is marked failed startable again")
    ap.add_argument("--trainer-command", default=None, help="replace the trainer command (a JSON list, e.g. '[\"{python}\", \"tests/fake_trainer.py\"]'); same as trainer.command in the config")
    ap.add_argument("--servers-command", default=None, help="replace the server command (a JSON list); same as servers.command in the config")
    args = ap.parse_args(argv)

    if args.stop:
        return stop_running(args.run_dir, args.wait)
    if args.clear_failed:
        return clear_failed(args.run_dir)
    if not args.config:
        ap.error("--config is required")
    try:
        cfg = load_config(args.config, {"trainer": _json_list(args.trainer_command), "servers": _json_list(args.servers_command)})
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 255 if args.can_start else EXIT_CONFIG
    if args.dry_run:
        return dry_run(cfg, args.run_dir)
    if args.can_start:
        try:
            return can_start(cfg, args.run_dir)
        except Exception as e:
            print(f"error: {e!r}", file=sys.stderr)
            return 255
    return Supervisor(cfg, args.run_dir, args.config).run()


if __name__ == "__main__":
    sys.exit(main())
