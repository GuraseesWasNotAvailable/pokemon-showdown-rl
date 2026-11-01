#!/usr/bin/env python
"""
One screen that answers: is the run alive, and is it getting better?

    python status.py --run-dir /home/ubuntu/runs/main
    python status.py --run-dir RUN --json            the same facts as one JSON object
    python status.py --run-dir RUN --exit-code       exit 1 unless the health is OK, STARTING or DONE

The first line is "HEALTH:" followed by one or more of these words (another program reads them;
with --json they are the list under "health"):

    OK          the trainer is running and decisions and updates are arriving
    STARTING    the servers or the trainer are starting, or the trainer has not reported yet
    RESTARTING  the supervisor is between a failure and the restart that follows it
    STOPPING    the supervisor is stopping the trainer (end of a phase, or a stop request)
    STALLED     running, but total_steps has not grown for longer than stall_minutes
    NO_UPDATES  decisions arrive, but there was no learner update for update_stall_minutes
    NAN_LOSS    the latest learner update reported a loss that is not a number
    DEAD        state.json says the run is active, but the supervisor has stopped refreshing it.
                Also beside DONE: the machine was to be shut down after the last evaluation
                (on_done "shutdown"), it has not been, and no supervisor is there to do it
    STOPPED     stopped cleanly; starting pipeline.py again resumes it
    FAILED      the supervisor gave up (too many restarts); it needs a person
    DISK_LOW    free space under the run directory is below min_free_gb
    DONE        every phase is finished
    SHUTDOWN_FAILED  the supervisor ran the shutdown command (after DONE, or after FAILED or
                DISK_LOW with on_failed "shutdown") and it failed: the machine is still up and
                still costs money. The supervisor tries again every shutdown_retry_minutes
    NO_STATE    there is no usable state.json in the run directory
    EVAL_STALE  the evaluator (evaluate.py --watch) wrote eval/watch.json but has stopped
                refreshing it without saying it stopped: it hung or was killed. Always the last
                word, after the words about the training itself.

It reads what pipeline.py and evaluate.py write (state.json, metrics.jsonl, events.jsonl,
eval/results.jsonl, eval/watch.json, checkpoints/) and never writes anything. A missing, empty or
half-written file, or a torn last line, costs that part of the screen and nothing else.

The evaluation table shows results only: a line of results.jsonl with no finished game
(games == 0) or marked "valid": false is counted under the table, not shown in it, and is not
what "vs prev" and "vs first" are measured against. Its rows are in order of training (phase,
then decisions, then the checkpoint's own time), not in the order they were measured: an early
checkpoint that is evaluated late is not the latest policy.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

PHASES = ("imitation", "warmup", "ppo")
CKPT_RE = re.compile(r"^learner_update_(\d{6,})\.pt$")
HEALTHY = {"OK", "STARTING", "DONE"}
ACTIVE = ("starting", "running", "stopping", "restarting")


# --------------------------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------------------------

def read_json(path: str, tries: int = 5) -> Optional[dict]:
    """The object in `path`, or None. Retried briefly: on Windows a file that is being replaced
    cannot be opened for a moment."""
    for attempt in range(tries):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                value = json.load(f)
            return value if isinstance(value, dict) else None
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            time.sleep(0.05 * (attempt + 1))
    return None


def read_jsonl(path: str) -> List[dict]:
    """Every line that parses to an object; anything else (a torn last line) is skipped."""
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


def num(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def rate_from_samples(samples: Sequence[Any], window: float, now: float) -> Optional[Tuple[float, float]]:
    """Decisions a second over the last `window` seconds from pipeline.py's [[time, collected], ...]
    samples: (rate, seconds covered) or None. Time without samples counts as time without decisions."""
    pts = []
    for s in samples or []:
        if isinstance(s, (list, tuple)) and len(s) >= 2 and num(s[0]) is not None and num(s[1]) is not None:
            pts.append((float(s[0]), float(s[1])))
    if len(pts) < 2:
        return None
    start = now - window
    if pts[0][0] >= start:
        t0, v0 = pts[0]
    else:
        t0, v0 = start, pts[-1][1]
        for a, b in zip(pts, pts[1:]):
            if a[0] <= start < b[0]:
                v0 = a[1] + (b[1] - a[1]) * (start - a[0]) / (b[0] - a[0])
                break
    span = now - t0
    if span <= 0:
        return None
    return max(0.0, pts[-1][1] - v0) / span, span


def gpu_info() -> Optional[List[dict]]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15,
                           stdin=subprocess.DEVNULL)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    out = []
    for line in r.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue

        def f(text: str) -> Optional[float]:
            try:
                return float(text)
            except ValueError:
                return None
        out.append({"name": parts[0], "memory_used_mb": f(parts[1]), "memory_total_mb": f(parts[2]),
                    "utilization_pct": f(parts[3]), "temperature_c": f(parts[4])})
    return out or None


def checkpoint_info(ckpt_dir: str, now: float) -> Optional[dict]:
    newest = None
    count, total = 0, 0
    try:
        names = os.listdir(ckpt_dir)
    except OSError:
        return None
    for name in names:
        if not CKPT_RE.match(name):
            continue
        try:
            st = os.stat(os.path.join(ckpt_dir, name))
        except OSError:
            continue
        count += 1
        total += st.st_size
        if newest is None or st.st_mtime > newest[1]:
            newest = (name, st.st_mtime, st.st_size)
    if newest is None:
        return {"name": None, "count": 0, "total_gb": 0.0}
    return {"name": newest[0], "age_minutes": round((now - newest[1]) / 60.0, 1), "size_mb": round(newest[2] / 1e6, 1),
            "count": count, "total_gb": round(total / 1e9, 2)}


# --------------------------------------------------------------------------------------------
# the facts
# --------------------------------------------------------------------------------------------

def health_flags(state: Optional[dict], now: float, free_gb: Optional[float]) -> Tuple[List[str], List[str]]:
    """(flags, one sentence per flag that needs explaining)."""
    if not state or not isinstance(state.get("phases"), dict):
        return ["NO_STATE"], ["no usable state.json: the supervisor has not started here, or the file is damaged"]
    flags: List[str] = []
    notes: List[str] = []
    status, phase = state.get("status"), state.get("phase")
    limits = state.get("limits") if isinstance(state.get("limits"), dict) else {}
    t = state.get("trainer") if isinstance(state.get("trainer"), dict) else {}
    reason = str(state.get("status_reason") or "")

    if status == "failed":
        flags.append("FAILED")
        notes.append(f"failed: {reason}")
    done = phase == "done" or status == "done"
    if done:
        flags.append("DONE")
    # The shutdown command: failed when the supervisor says so (state["shutdown_failed"], cleared by
    # an attempt that works), or, in a state written before that field existed, when the last word
    # about it in the events is a failure.
    recent = [e.get("event") for e in (state.get("last_events") or []) if isinstance(e, dict)]
    words = [e for e in recent if e in ("shutdown", "shutdown_failed")]
    failed_shutdown = state.get("shutdown_failed") if isinstance(state.get("shutdown_failed"), dict) else None
    if failed_shutdown is not None or (words and words[-1] == "shutdown_failed"):
        flags.append("SHUTDOWN_FAILED")
        detail = ""
        if failed_shutdown:
            detail = f" ({failed_shutdown.get('attempts')} attempts, the last at {failed_shutdown.get('time')}: {failed_shutdown.get('error')})"
        notes.append(f"the shutdown command FAILED{detail}: the box is still running and still costs money")
    elif state.get("shutdown_issued"):
        notes.append(f"the shutdown command was run at {state.get('shutdown_issued')}")
    elif state.get("shutdown_planned"):
        notes.append(f"the box is shut down at {state.get('shutdown_planned')} (on_failed is \"shutdown\") unless the supervisor is stopped first")
    elif done and limits.get("on_done") == "shutdown":
        notes.append("waiting for eval/final.done (the last evaluation) before shutting the box down")
    min_free = num(limits.get("min_free_gb"))
    if status == "disk_low" or (free_gb is not None and min_free is not None and free_gb < min_free):
        flags.append("DISK_LOW")
        notes.append(f"{free_gb:.1f} GB free, the limit is {min_free:g} GB" if free_gb is not None and min_free is not None else reason)

    leftover = state.get("leftover") if isinstance(state.get("leftover"), dict) else {}
    for what, left in leftover.items():
        if isinstance(left, dict) and (left.get("pids") or left.get("ports")):
            notes.append(f"the {what} could not be stopped completely at {left.get('time')}: processes {left.get('pids')} were still alive"
                         + (f", ports {left.get('ports')} still answered" if left.get("ports") else "")
                         + ". Stop them by their ids, or the next start finds the ports or the GPU taken")

    age = now - (num(state.get("updated_ts")) or 0.0)
    stale_after = max(120.0, 10.0 * (num(limits.get("state_seconds")) or 5.0))
    dead = False
    if status in ACTIVE and age > stale_after:
        dead = True
        flags.append("DEAD")
        notes.append(f"state.json says \"{status}\" but was last refreshed {age / 60.0:.1f} minutes ago: the supervisor is not running")
    elif status == "error":
        dead = True
        flags.append("DEAD")
        notes.append(f"the supervisor stopped on an internal error ({reason}); systemd starts it again")
    elif done and limits.get("on_done") == "shutdown" and not state.get("shutdown_issued") and age > stale_after:
        # Finished, to be shut down, not shut down, and nobody is there to do it: not a healthy end.
        dead = True
        flags.append("DEAD")
        notes.append(f"the run is done and the box was to be shut down after the last evaluation, but the supervisor that does it "
                     f"last refreshed state.json {age / 60.0:.1f} minutes ago: nothing will shut the box down")

    if not dead:
        if status == "starting":
            flags.append("STARTING")
        elif status == "restarting":
            flags.append("RESTARTING")
            notes.append(f"restarting after: {reason}")
        elif status == "stopping":
            flags.append("STOPPING")
            notes.append(f"stopping: {reason}")
        elif status == "stopped":
            flags.append("STOPPED")
            notes.append(f"stopped ({reason or 'no reason recorded'}); starting pipeline.py again resumes phase {phase}")
        elif status == "running":
            stall = (num(limits.get("stall_minutes")) or 10.0) * 60.0
            grace = (num(limits.get("startup_grace_minutes")) or 15.0) * 60.0
            upd_stall = (num(limits.get("update_stall_minutes")) or 0.0) * 60.0
            last_progress = num(t.get("last_progress_ts"))
            idle = now - last_progress if last_progress is not None else 0.0
            allowed = stall if t.get("progress_seen") else max(stall, grace)
            steps, base = num(t.get("total_steps")), num(t.get("steps_at_last_update"))
            last_update = num(t.get("last_update_ts"))
            if idle > allowed:
                flags.append("STALLED")
                notes.append(f"total_steps has not grown for {idle / 60.0:.1f} minutes (limit {allowed / 60.0:.1f})")
            elif not t.get("telemetry_lines") or steps is None:
                flags.append("STARTING")
                notes.append("the trainer has not reported yet")
            elif upd_stall > 0 and last_update is not None and now - last_update > upd_stall and steps - (base or 0.0) > 0:
                flags.append("NO_UPDATES")
                err = f"; last error: {t.get('last_update_error')}" if t.get("last_update_error") else ""
                notes.append(f"no learner update for {(now - last_update) / 60.0:.1f} minutes while {int(steps - (base or 0.0)):,} "
                             f"decisions arrived ({t.get('update_failures') or 0} failed updates{err})")
            if t.get("last_loss_nonfinite"):
                flags.append("NAN_LOSS")
                notes.append("the latest learner update reported a loss that is not a number")
            if not flags:
                flags.append("OK")
    if not flags:
        flags.append(str(status or "unknown").upper())
    return flags, notes


def loss_summary(metrics: List[dict], phase: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    rows = [m for m in metrics if m.get("phase") == phase]
    for key in ("ploss", "vloss"):
        vals = [(num(m.get(key)), m.get("update")) for m in rows]
        vals = [(v, u) for v, u in vals if v is not None]
        if not vals:
            continue
        best = min(vals, key=lambda p: p[0])
        out[key] = {"first": vals[0][0], "best": best[0], "best_update": best[1], "latest": vals[-1][0],
                    "latest_update": vals[-1][1], "n": len(vals)}
    return out


def is_result(line: dict) -> bool:
    """Is this line of results.jsonl a measurement? Not when no game finished (evaluate.py writes
    such a line with winrate 0.0, which is not a win rate of zero) and not when evaluate.py
    marked it "valid": false (too many games lost to errors, or a side playing random moves)."""
    games = num(line.get("games"))
    if games is not None and games <= 0:
        return False
    return line.get("valid") is not False


def evaluation_skipped(results: List[dict]) -> Dict[str, int]:
    """How many lines the table leaves out, by reason."""
    out = {"no_finished_game": 0, "not_valid": 0}
    for r in results:
        if not isinstance(r.get("opponent"), str) or is_result(r):
            continue
        games = num(r.get("games"))
        out["no_finished_game" if games is not None and games <= 0 else "not_valid"] += 1
    return out


def evaluator_info(run_dir: str, now: float) -> Optional[dict]:
    """
    The evaluator's heartbeat, from eval/watch.json (evaluate.py --watch rewrites it every poll and
    every second while an evaluation runs). None when the file is not there: no evaluator has run
    on this directory. "stale" is True when it did not say it stopped and the heartbeat is older
    than five minutes (or three polls, when its poll is longer than that).
    """
    w = read_json(os.path.join(run_dir, "eval", "watch.json"))
    if w is None:
        return None
    beat = num(w.get("heartbeat_ts"))
    poll = num(w.get("poll_seconds")) or 60.0
    age = None if beat is None else round(now - beat, 1)
    stopped = w.get("status") == "stopped"
    limit = max(300.0, 3.0 * poll)
    return {"pid": w.get("pid"), "host": w.get("host"), "started": w.get("started"), "heartbeat": w.get("heartbeat"),
            "heartbeat_age_seconds": age, "status": w.get("status"), "evaluations": w.get("evaluations"),
            "stopped": stopped, "stale_after_seconds": limit,
            "stale": (not stopped) and (age is None or age > limit)}


def _training_order(r: dict) -> tuple:
    """Sorts result lines by how far the policy in them was trained: phase, then the decisions of
    that phase, then the checkpoint file's own time. When the line was written comes last: a
    checkpoint from early in the run that is measured late (the end-of-imitation copy, by hand)
    must not become "the latest". The same order evaluate.py's report uses."""
    phase = r.get("phase")
    rank = next((i for i, p in enumerate(PHASES) if isinstance(phase, str) and phase.startswith(p)), -1)
    steps, written = num(r.get("total_steps")), num(r.get("checkpoint_mtime"))
    return (rank, -1.0 if steps is None else steps, 0.0 if written is None else written, str(r.get("time") or ""), r.get("_order", 0))


def evaluation_table(results: List[dict], last: int = 6) -> Dict[str, List[dict]]:
    """Per opponent, the evaluations in order of training (see _training_order) with the change in
    win rate (percentage points) since the one before and since the first one; only the last
    `last` are returned. Lines that are not results (see is_result) are left out."""
    groups: Dict[str, List[dict]] = {}
    for i, r in enumerate(results):
        opp = r.get("opponent")
        wr = num(r.get("winrate"))
        if not isinstance(opp, str) or wr is None or not is_result(r):
            continue
        key = opp + (" (sampled)" if r.get("sample") is True else "")
        groups.setdefault(key, []).append(dict(r, _order=i))
    table: Dict[str, List[dict]] = {}
    for key, rows in groups.items():
        rows.sort(key=_training_order)
        first = num(rows[0].get("winrate"))
        out = []
        prev = None
        for r in rows:
            wr = num(r.get("winrate"))
            out.append({
                "time": r.get("time"), "checkpoint": r.get("checkpoint"), "phase": r.get("phase"), "update": r.get("update"),
                "total_steps": r.get("total_steps"), "games": r.get("games"), "wins": r.get("wins"), "losses": r.get("losses"),
                "ties": r.get("ties"), "errors": r.get("errors"),
                "winrate": wr, "ci_low": num(r.get("ci_low")), "ci_high": num(r.get("ci_high")),
                "delta_prev": None if prev is None else round((wr - prev) * 100.0, 2),
                "delta_first": None if prev is None else round((wr - first) * 100.0, 2),
            })
            prev = wr
        table[key] = out[-last:]
        table[key][0]["evaluations_total"] = len(out)
    return table


def gather(run_dir: str, now: Optional[float] = None, with_gpu: bool = True) -> Dict[str, Any]:
    now = time.time() if now is None else now
    run_dir = os.path.abspath(run_dir)
    state = read_json(os.path.join(run_dir, "state.json"))
    if state is not None and not isinstance(state.get("phases"), dict):
        state = None
    metrics = read_jsonl(os.path.join(run_dir, "metrics.jsonl"))
    events = read_jsonl(os.path.join(run_dir, "events.jsonl"))
    results = read_jsonl(os.path.join(run_dir, "eval", "results.jsonl"))

    disk = None
    try:
        du = shutil.disk_usage(run_dir if os.path.isdir(run_dir) else os.path.dirname(run_dir))
        disk = {"free_gb": round(du.free / 1024 ** 3, 1), "total_gb": round(du.total / 1024 ** 3, 1)}
    except OSError:
        pass
    flags, notes = health_flags(state, now, disk["free_gb"] if disk else None)
    evaluator = evaluator_info(run_dir, now)
    if evaluator and evaluator["stale"]:
        age = evaluator["heartbeat_age_seconds"]
        flags.append("EVAL_STALE")
        notes.append("the evaluator (evaluate.py --watch, pid %s) last wrote eval/watch.json %s: it is not running, so no "
                     "checkpoint is being evaluated and eval/final.done will not be written"
                     % (evaluator.get("pid"), "at an unknown time" if age is None else f"{age / 60.0:.1f} minutes ago"))

    st = state or {}
    limits = st.get("limits") if isinstance(st.get("limits"), dict) else {}
    if disk is not None:
        disk["min_free_gb"] = limits.get("min_free_gb")
    phase = st.get("phase")
    phases = st.get("phases") if isinstance(st.get("phases"), dict) else {}
    cur = phases.get(phase) if isinstance(phases.get(phase), dict) else {}
    loss_phase = phase if phase in PHASES else "ppo"
    if metrics and not any(m.get("phase") == loss_phase for m in metrics):
        loss_phase = str(metrics[-1].get("phase"))
    t = st.get("trainer") if isinstance(st.get("trainer"), dict) else {}

    target = None
    lim_ph = (limits.get("phases") or {}).get(phase) if isinstance(limits.get("phases"), dict) else None
    if isinstance(lim_ph, dict):
        target = lim_ph.get("max_decisions") or None

    samples = st.get("samples") or []
    speed = {}
    for label, window in (("last_10_min", 600.0), ("last_hour", 3600.0)):
        r = rate_from_samples(samples, window, now)
        speed[label] = None if r is None else {"per_second": round(r[0], 1), "covered_seconds": round(r[1])}

    # When the last learner update arrived. The trainer's last_update_ts is the watchdog's clock and
    # also holds the moment this trainer first reported, so it counts only once this trainer has
    # seen an update (updates_seen); before that the last update is the last record of metrics.jsonl.
    seen = t.get("updates_seen") if t else None
    last_update_ts = num(t.get("last_update_ts")) if (seen or (t and "updates_seen" not in t)) else None
    if last_update_ts is None and metrics:
        last_update_ts = num(metrics[-1].get("wall_time"))
    none_since_start = bool(t) and t.get("running") is not False and seen == 0
    failures =[f for f in (st.get("failures") or []) if num(f) is not None and now - float(f) < 3600.0]

    info: Dict[str, Any] = {
        "health": flags,
        "health_notes": notes,
        "time": datetime.datetime.fromtimestamp(now, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_dir": run_dir,
        "phase": phase,
        "status": st.get("status"),
        "status_reason": st.get("status_reason"),
        "state_age_seconds": None if num(st.get("updated_ts")) is None else round(now - float(st["updated_ts"]), 1),
        "supervisor_pid": st.get("pid"),
        "host": st.get("host"),
        "decisions": {"phase": cur.get("decisions"), "total": st.get("decisions_total"), "phase_target": target,
                      "lost": sum(int(num((phases.get(p) or {}).get("lost_decisions")) or 0) for p in PHASES)},
        "speed": speed,
        "updates": {"phase": cur.get("updates"), "total": st.get("updates_total"),
                    "minutes_since_last": None if last_update_ts is None else round((now - last_update_ts) / 60.0, 1),
                    "none_since_trainer_start": none_since_start,
                    "failed": t.get("update_failures"), "last_error": t.get("last_update_error")},
        "loss_phase": loss_phase,
        "loss": loss_summary(metrics, loss_phase),
        "phases": {p: {k: (phases.get(p) or {}).get(k) for k in
                       ("status", "decisions", "updates", "started", "ended", "end_reason", "final_checkpoint", "trainer_starts", "lost_decisions")}
                   for p in PHASES},
        "trainer": {k: t.get(k) for k in ("pid", "segment", "started", "log", "running", "active", "resumed_from", "phase_change", "total_steps", "update")} if t else None,
        "restarts": {"failures_last_hour": len(failures), "limit_per_hour": limits.get("max_restarts_per_hour"),
                     "total": st.get("restarts_total"), "planned": st.get("recycles_total"),
                     "trainer_starts": st.get("segment"), "quarantined_checkpoints": len(st.get("quarantined") or [])},
        "events": events[-8:],
        "disk": disk,
        "gpu": gpu_info() if with_gpu else None,
        "checkpoint": checkpoint_info(os.path.join(run_dir, "checkpoints"), now),
        "evaluation": evaluation_table(results),
        "evaluation_skipped": evaluation_skipped(results),
        "evaluator": evaluator,
        "counts": {"metrics_records": len(metrics), "events": len(events), "evaluations": len(results)},
    }
    return info


# --------------------------------------------------------------------------------------------
# the screen
# --------------------------------------------------------------------------------------------

def _n(value: Any) -> str:
    v = num(value)
    return "-" if v is None else f"{int(v):,}"


def _f(value: Any, digits: int = 4) -> str:
    v = num(value)
    return "-" if v is None else f"{v:.{digits}f}"


def _pct(value: Any) -> str:
    v = num(value)
    return "-" if v is None else f"{v * 100.0:.1f}%"


def _delta(value: Any) -> str:
    v = num(value)
    return "-" if v is None else f"{v:+.1f}"


def _ascii(text: Any) -> str:
    return str(text).encode("ascii", "replace").decode("ascii")


def render(info: Dict[str, Any]) -> str:
    L: List[str] = []
    L.append("HEALTH: " + " ".join(info["health"]))
    for note in info.get("health_notes") or []:
        L.append("  " + note)
    age = info.get("state_age_seconds")
    L.append(f"run        {info['run_dir']}   ({info['time']}" + (f", state written {age:.0f} s ago" if age is not None else "") + ")")
    if "NO_STATE" in info["health"]:
        L.append("phase      -")
    else:
        L.append(f"phase      {info.get('phase')}   status {info.get('status')}"
                 + (f"   ({info.get('status_reason')})" if info.get("status_reason") else ""))
        d = info["decisions"]
        target = num(d.get("phase_target"))
        share = ""
        if target and num(d.get("phase")) is not None:
            share = f" of {int(target):,} ({100.0 * float(d['phase']) / target:.1f}%)"
        L.append(f"decisions  this phase {_n(d.get('phase'))}{share}   overall {_n(d.get('total'))}"
                 + (f"   lost to restarts {_n(d.get('lost'))}" if d.get("lost") else ""))
        parts = []
        for label, text in (("last_10_min", "last 10 min"), ("last_hour", "last hour")):
            s = info["speed"].get(label)
            if s is None:
                parts.append(f"{text} -")
            else:
                short = "" if s["covered_seconds"] >= (590 if label == "last_10_min" else 3590) else f" (only {s['covered_seconds'] / 60.0:.0f} min of data)"
                parts.append(f"{text} {s['per_second']:,.0f}/s{short}")
        eta = ""
        s = info["speed"].get("last_hour") or info["speed"].get("last_10_min")
        if target and s and s["per_second"] > 0 and num(d.get("phase")) is not None and info.get("status") == "running":
            left = (target - float(d["phase"])) / s["per_second"]
            if left > 0:
                eta = f"   this phase ends in about {left / 3600.0:.1f} h" if left < 48 * 3600 else f"   this phase ends in about {left / 86400.0:.1f} days"
        L.append("speed      " + "   ".join(parts) + eta)
        u = info["updates"]
        since = u.get("minutes_since_last")
        L.append(f"updates    this phase {_n(u.get('phase'))}   overall {_n(u.get('total'))}   last one "
                 + ("-" if since is None else f"{since:.1f} min ago")
                 + (" (none yet from the trainer running now)" if u.get("none_since_trainer_start") else "")
                 + f"   failed since trainer start {_n(u.get('failed') or 0)}")
        if u.get("last_error"):
            L.append(f"           last update error: {_ascii(u['last_error'])[:160]}")
        tr = info.get("trainer")
        if tr:
            L.append(f"trainer    {'running' if tr.get('running') else 'not running'}, pid {tr.get('pid')}, start number {tr.get('segment')} "
                     f"at {tr.get('started')}, active battles {_n(tr.get('active'))}, log {tr.get('log')}"
                     + (f", resumed from {tr['resumed_from']}" if tr.get("resumed_from") else ""))
        loss = info.get("loss") or {}
        if loss:
            for i, key in enumerate(("ploss", "vloss")):
                if key not in loss:
                    continue
                v = loss[key]
                head = f"loss ({info.get('loss_phase')})" if i == 0 or "ploss" not in loss else ""
                L.append(f"{head:<10} {key}  first {_f(v['first'], 3)}   best {_f(v['best'], 3)} (update {v.get('best_update')})   "
                         f"latest {_f(v['latest'], 3)} (update {v.get('latest_update')})   {v['n']} updates")
        else:
            L.append("loss       no learner update recorded yet")
        cells = []
        for p in PHASES:
            ph = info["phases"][p]
            label = "in progress" if ph.get("status") == "running" else (ph.get("status") or "-")
            cell = f"{p} {label} {_n(ph.get('decisions'))}"
            if ph.get("final_checkpoint"):
                cell += f" -> {ph['final_checkpoint']}"
            cells.append(cell)
        L.append("phases     " + "  |  ".join(cells))
        for p in PHASES:
            if info["phases"][p].get("end_reason"):
                L.append(f"           {p} ended: {_ascii(info['phases'][p]['end_reason'])}")
        r = info["restarts"]
        L.append(f"restarts   failures in the last hour {r.get('failures_last_hour')} (limit {r.get('limit_per_hour')})   total {_n(r.get('total') or 0)}"
                 f"   planned {_n(r.get('planned') or 0)}   trainer starts {_n(r.get('trainer_starts') or 0)}"
                 + (f"   checkpoints set aside {r['quarantined_checkpoints']}" if r.get("quarantined_checkpoints") else ""))
    events = info.get("events") or []
    if events:
        L.append("events     (newest last)")
        for e in events:
            rest = {k: v for k, v in e.items() if k not in ("time", "ts", "event", "phase") and v is not None}
            text = " ".join(f"{k}={_ascii(v)}" for k, v in rest.items())
            L.append(f"  {e.get('time')}  {e.get('event')}  {text}"[:200])
    disk = info.get("disk")
    if disk:
        L.append(f"disk       {disk['free_gb']:.1f} GB free of {disk['total_gb']:.1f} GB" + (f"   (stops below {disk['min_free_gb']} GB)" if disk.get("min_free_gb") is not None else ""))
    gpu = info.get("gpu")
    if gpu:
        for g in gpu:
            L.append(f"gpu        {g['name']}   {_n(g['memory_used_mb'])} / {_n(g['memory_total_mb'])} MiB   utilisation {_n(g['utilization_pct'])}%   {_n(g['temperature_c'])} C")
    else:
        L.append("gpu        nvidia-smi not available")
    c = info.get("checkpoint")
    if c and c.get("name"):
        L.append(f"checkpoint {c['name']}   {c['size_mb']} MB   written {c['age_minutes']} min ago   ({c['count']} files, {c['total_gb']} GB)")
    else:
        L.append("checkpoint none yet")
    ev = info.get("evaluator")
    if ev is None:
        L.append("evaluator  no eval/watch.json: evaluate.py --watch has not run on this directory")
    else:
        age = ev.get("heartbeat_age_seconds")
        heard = "never" if age is None else (f"{age:.0f} s ago" if age < 120 else f"{age / 60.0:.1f} min ago")
        if ev.get("stopped"):
            head = f"STOPPED (it said so {heard}), pid {ev.get('pid')}"
        elif ev.get("stale"):
            head = f"STALE: last heartbeat {heard}, pid {ev.get('pid')} is not refreshing eval/watch.json"
        else:
            head = f"alive, pid {ev.get('pid')}, heartbeat {heard}"
        L.append(f"evaluator  {head}   evaluations finished {_n(ev.get('evaluations') or 0)}"
                 + ("" if ev.get("stopped") or not ev.get("status") else f"   now: {_ascii(ev['status'])[:110]}"))
    table = info.get("evaluation") or {}
    skipped = info.get("evaluation_skipped") or {}
    left_out = []
    if skipped.get("no_finished_game"):
        left_out.append(f"{skipped['no_finished_game']} with no finished game")
    if skipped.get("not_valid"):
        left_out.append(f"{skipped['not_valid']} marked not valid")
    if table:
        L.append("evaluation (win rate of the policy; changes in percentage points)")
        L.append(f"  {'opponent':<22}{'checkpoint':<16}{'win rate':>9}  {'95% interval':<16}{'games':>6}{'vs prev':>9}{'vs first':>9}")
        for key in sorted(table):
            for i, r in enumerate(table[key]):
                where = f"{str(r.get('phase') or '?')[:3]}:{r.get('update')}" if r.get("update") is not None else str(r.get("checkpoint") or "?")[-15:]
                ci = "-" if r.get("ci_low") is None or r.get("ci_high") is None else f"[{r['ci_low'] * 100:.1f}, {r['ci_high'] * 100:.1f}]"
                L.append(f"  {(key if i == 0 else ''):<22}{where:<16}{_pct(r.get('winrate')):>9}  {ci:<16}{_n(r.get('games')):>6}"
                         f"{_delta(r.get('delta_prev')):>9}{_delta(r.get('delta_first')):>9}")
        if left_out:
            L.append(f"  not shown: {', '.join(left_out)} (see eval/REPORT.md)")
    elif left_out:
        L.append(f"evaluation no usable result yet: {', '.join(left_out)} (see eval/REPORT.md)")
    else:
        L.append("evaluation no results yet (eval/results.jsonl)")
    return "\n".join(_ascii(line) for line in L)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--json", action="store_true", help="print one JSON object instead of the screen")
    ap.add_argument("--exit-code", action="store_true", help="exit 1 unless every health word is OK, STARTING or DONE")
    ap.add_argument("--no-gpu", action="store_true", help="do not call nvidia-smi")
    args = ap.parse_args(argv)
    try:
        info = gather(args.run_dir, with_gpu=not args.no_gpu)
    except Exception as e:  # this program must always print a health line
        info = {"health": ["NO_STATE"], "health_notes": [f"status.py failed: {e!r}"], "run_dir": os.path.abspath(args.run_dir),
                "time": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        if args.json:
            print(json.dumps(info))
        else:
            print("HEALTH: NO_STATE")
            print(f"  status.py failed: {_ascii(repr(e))}")
        return 1 if args.exit_code else 0
    if args.json:
        print(json.dumps(info, indent=1))
    else:
        print(render(info))
    if args.exit_code and not set(info["health"]) <= HEALTHY:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
