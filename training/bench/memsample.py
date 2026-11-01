#!/usr/bin/env python
"""
Samples the memory of a training run from outside: every process under the given root PIDs
(the trainer, and start_servers.py with its Showdown servers), every --seconds, one JSON line each
time. Read-only: it never signals anything.

    python bench/memsample.py --roots <trainer pid>,<start_servers pid> --out mem.jsonl --seconds 30

    python bench/memsample.py --report mem.jsonl [--memwatch <dir>] [--log <trainer log>]
        growth per role, and per 1,000 episodes when the trainer's log is given

Roles: "server" is the Showdown server itself (pokemon-showdown start), "sim" its simulator child
(room-battle.js), "sockets" its sockets.js, "node-other" the rest of its children; "py" is every
python process under the trainer (the rollout workers are told apart by the files memwatch.py
writes, see --memwatch).

"private" is the memory only that process holds: the commit charge on Windows, USS elsewhere
when it can be read, else RSS. A leak shows there. RSS on Windows is the working set, which the
system trims when it is short of memory, so it is not the figure to compare.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from typing import Any, Dict, List

import psutil


def role_of(cmdline: List[str], name: str) -> str:
    text = " ".join(cmdline).replace("\\", "/")
    low = name.lower()
    if "node" in low:
        if "room-battle" in text:
            return "sim"
        if "sockets.js" in text:
            return "sockets"
        if "pokemon-showdown" in text and " start" in text and "/dist/" not in text:
            return "server"
        if "engine.js" in text:
            return "psbot-engine"
        return "node-other"
    if "python" in low:
        if "default_worker" in text:
            return "py-worker"
        if "start_servers" in text:
            return "start_servers"
        if "train.py" in text:
            return "py-driver"
        return "py-other"
    if "raylet" in low or "gcs_server" in low:
        return "ray-core"
    return "other"


def sample(roots: List[int]) -> Dict[str, Any]:
    rows = []
    seen = set()
    for root in roots:
        try:
            top = psutil.Process(root)
            procs = [top] + top.children(recursive=True)
        except psutil.Error:
            continue
        for p in procs:
            if p.pid in seen:
                continue
            seen.add(p.pid)
            try:
                with p.oneshot():
                    info = p.memory_info()
                    cmd = p.cmdline()
                    private = getattr(info, "private", None)
                    if private is None:
                        try:
                            private = p.memory_full_info().uss
                        except psutil.Error:
                            private = info.rss
                    rows.append({"pid": p.pid, "ppid": p.ppid(), "root": root, "role": role_of(cmd, p.name()),
                                 "rss": int(info.rss), "private": int(private),
                                 "port": _port_of(cmd)})
            except psutil.Error:
                continue
    return {"t": round(time.time(), 1), "procs": rows}


def _port_of(cmd: List[str]) -> int:
    for i, a in enumerate(cmd):
        if a == "--port" and i + 1 < len(cmd) and cmd[i + 1].isdigit():
            return int(cmd[i + 1])
    return 0


def run(args: argparse.Namespace) -> int:
    roots = [int(x) for x in args.roots.split(",") if x.strip()]
    end = time.time() + args.minutes * 60 if args.minutes else None
    while end is None or time.time() < end:
        row = sample(roots)
        if not row["procs"]:
            break
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
        time.sleep(args.seconds)
    return 0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

_TELEMETRY = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ .*'episodes': (\d+).*'total_steps': (\d+)")


def episodes_from_log(path: str) -> List[tuple]:
    """(unix time, episodes, total_steps) from the trainer's [Telemetry] lines (local time stamps)."""
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _TELEMETRY.match(line)
            if m:
                out.append((time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")), int(m.group(2)), int(m.group(3))))
    return out


def _at(series: List[tuple], t: float, col: int) -> float:
    best = series[0]
    for row in series:
        if row[0] <= t:
            best = row
        else:
            break
    return best[col]


def _fit(xs: List[float], ys: List[float]) -> float:
    """Least-squares slope of ys over xs."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0


def report(args: argparse.Namespace) -> int:
    rows = [json.loads(l) for l in open(args.report, encoding="utf-8") if l.strip()]
    if len(rows) < 2:
        print("fewer than two samples")
        return 1
    t0 = rows[0]["t"]
    skip = args.skip_minutes * 60
    rows = [r for r in rows if r["t"] - t0 >= skip] or rows
    worker_pids: Dict[int, int] = {}
    if args.memwatch:
        for path in glob.glob(os.path.join(args.memwatch, "worker_*_*.jsonl")):
            m = re.search(r"worker_(\d+)_(\d+)\.jsonl$", path)
            if m:
                worker_pids[int(m.group(2))] = int(m.group(1))
    eps = episodes_from_log(args.log) if args.log else []
    first, last = rows[0], rows[-1]
    minutes = (last["t"] - first["t"]) / 60
    print(f"{len(rows)} samples over {minutes:.1f} minutes (the first {args.skip_minutes:g} minutes left out)")
    d_eps = 0.0
    if eps:
        d_eps = _at(eps, last["t"], 1) - _at(eps, first["t"], 1)
        d_steps = _at(eps, last["t"], 2) - _at(eps, first["t"], 2)
        print(f"episodes in that time: {d_eps:,.0f} (two a battle), decisions {d_steps:,.0f}, {d_steps / max(minutes * 60, 1):.0f} a second")

    def label(p: Dict[str, Any]) -> str:
        if p["pid"] in worker_pids:
            return f"rollout-worker:{worker_pids[p['pid']]}"
        if p["role"] in ("server", "sim", "sockets") :
            return p["role"]
        return p["role"]

    series: Dict[str, Dict[float, float]] = {}
    counts: Dict[str, int] = {}
    for r in rows:
        per: Dict[str, float] = {}
        n: Dict[str, int] = {}
        for p in r["procs"]:
            k = label(p)
            per[k] = per.get(k, 0.0) + p["private"] / 1e6
            n[k] = n.get(k, 0) + 1
        for k, v in per.items():
            series.setdefault(k, {})[r["t"]] = v
            counts[k] = max(counts.get(k, 0), n[k])
    print(f"{'role':24} {'n':>3} {'first MB':>10} {'last MB':>10} {'MB/hour (fit)':>14}" + (f" {'MB per 1,000 episodes':>22}" if d_eps else ""))
    for k in sorted(series):
        pts = sorted(series[k].items())
        if len(pts) < 2:
            continue
        slope = _fit([t for t, _v in pts], [v for _t, v in pts]) * 3600
        line = f"{k:24} {counts[k]:>3} {pts[0][1]:>10.1f} {pts[-1][1]:>10.1f} {slope:>14.1f}"
        if d_eps:
            total_eps_per_hour = d_eps / max(minutes / 60, 1e-9)
            line += f" {slope / (total_eps_per_hour / 1000):>22.2f}"
        print(line)
    if d_eps:
        print("(per 1,000 episodes: of the whole run, both workers together, so a per-worker row is that worker's growth per 1,000 run episodes)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", help="comma-separated PIDs whose process trees are sampled")
    ap.add_argument("--out", help="the file the samples are appended to")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--minutes", type=float, default=0.0, help="stop after this long (0: when the processes are gone)")
    ap.add_argument("--report", help="a file of samples to summarise")
    ap.add_argument("--memwatch", help="with --report: the PSPPO_MEMWATCH directory, to name the rollout workers")
    ap.add_argument("--log", help="with --report: the trainer's log, for the episode count")
    ap.add_argument("--skip-minutes", type=float, default=3.0, help="with --report: leave out the start-up")
    args = ap.parse_args()
    if args.report:
        return report(args)
    if not args.roots or not args.out:
        ap.error("--roots and --out, or --report")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
