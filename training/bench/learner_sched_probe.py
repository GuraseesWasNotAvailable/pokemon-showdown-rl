#!/usr/bin/env python3
"""
How the CPU serves the live learner's updates, read from outside the run (Linux /proc only).

    python3 bench/learner_sched_probe.py [--seconds 120] [--pid PID] [--log TRAINER_LOG] [--json OUT]

For every update that ends inside the window: its seconds and minibatches (from the trainer log's
"[learner] upd=" line) and, over the update's own span, the seconds the learner's work thread and its
autograd threads spent on a CPU and runnable in the run queue. The same figures bench/fast_update_digest.py
time records for the harness ("sched"), so box a and the box c emulation compare: the emulation is a
fair stand-in when its pristine figures per minibatch match these. Run it before and after a restart with
--fast-update to measure the change on the live box itself.

It reads /proc/<pid>/task/*/{schedstat,comm} and the log's size, four times a second, and nothing else:
it sends no signal, changes no priority and writes nothing but stdout (and --json, which belongs under
~/dev/fast*). The learner is found by its Ray process title (ray::LearnerActor), the log as the newest
trainer_*.log of ~/runs/v2/scratch/logs. The work thread is the learner's busiest thread that is not an
autograd thread (Ray names it after the task it runs, ray::IDLE or ray::LearnerActor).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time


def learner_pid() -> int:
    out = subprocess.run(["pgrep", "-f", "ray::LearnerActor"], capture_output=True, text=True).stdout.split()
    pids = [int(p) for p in out if int(p) != os.getpid()]
    if not pids:
        sys.exit("no ray::LearnerActor process")
    return pids[0]


def threads(pid: int) -> dict:
    out = {}
    for t in os.listdir(f"/proc/{pid}/task"):
        try:
            with open(f"/proc/{pid}/task/{t}/comm") as f:
                comm = f.read().strip()
            with open(f"/proc/{pid}/task/{t}/schedstat") as f:
                run, wait, _ = f.read().split()
            out[int(t)] = (comm, int(run), int(wait))
        except (OSError, ValueError):
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--pid", type=int, default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    pid = a.pid or learner_pid()
    log = a.log or max(glob.glob(os.path.expanduser("~/runs/v2/scratch/logs/trainer_*.log")), key=os.path.getmtime)
    th = threads(pid)
    work = max((t for t, v in th.items() if not v[0].startswith("pt_autograd")), key=lambda t: th[t][1])
    autograd = sorted(t for t, v in th.items() if v[0].startswith("pt_autograd"))
    print(f"learner pid {pid} (nice {os.getpriority(os.PRIO_PROCESS, pid)}), work thread {work} ({th[work][0]}), "
          f"autograd threads {autograd}; log {log}", flush=True)
    samples, names = [], {}
    t_end = time.time() + a.seconds
    while time.time() < t_end:
        th = threads(pid)
        if work not in th:
            sys.exit("the learner's work thread is gone (a restart?)")
        ag = [th[t] for t in autograd if t in th]
        samples.append((time.time(), os.path.getsize(log), th[work][1], th[work][2],
                        sum(v[1] for v in ag), sum(v[2] for v in ag),
                        {t: v[1] for t, v in th.items() if t != work and t not in autograd}))
        names.update({t: v[0] for t, v in th.items()})
        time.sleep(0.25)
    s0 = samples[0][1]
    with open(log, "rb") as f:
        f.seek(max(0, s0 - 4096))
        start = f.tell()
        data = f.read()
    ends = []
    pat = re.compile(rb"\[learner\] upd=(\d+) .*?n_mb=(\d+) sec=([0-9.]+)")
    for m in pat.finditer(data):
        off = start + data.rfind(b"\n", 0, m.start()) + 1
        ends.append((off, int(m.group(1)), int(m.group(2)), float(m.group(3))))

    def at(i, t):
        for (t0, *a0), (t1, *a1) in zip(samples, samples[1:]):
            if t0 <= t <= t1:
                w = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                return a0[i] + w * (a1[i] - a0[i])
        return None

    rows = []
    for off, upd, n_mb, sec in ends:
        after = [k for k, s in enumerate(samples) if s[1] > off]
        if not after or after[0] == 0:
            continue
        k = after[0]
        te = (samples[k][0] + samples[k - 1][0]) / 2          # the line appeared between these two samples
        ts = te - sec
        vals = [at(i, te) for i in (1, 2, 3, 4)], [at(i, ts) for i in (1, 2, 3, 4)]
        if any(v is None for v in vals[0] + vals[1]):
            continue
        d = [(e - s) / 1e9 for e, s in zip(*vals)]
        # the learner's other threads over the same span (nearest samples inside it)
        ks = [j for j, smp in enumerate(samples) if ts <= smp[0] <= te]
        oth = []
        if len(ks) >= 2:
            o0, o1 = samples[ks[0]][6], samples[ks[-1]][6]
            span = samples[ks[-1]][0] - samples[ks[0]][0]
            scale = sec / span if span > 0 else 1.0
            oth = sorted((((o1[t] - o0.get(t, o1[t])) / 1e9 * scale, f"{names.get(t, '?')}/{t}") for t in o1), reverse=True)
        other = sum(x[0] for x in oth)
        rows.append({"upd": upd, "n_mb": n_mb, "sec": sec, "work_run": d[0], "work_runq": d[1],
                     "autograd_run": d[2], "autograd_runq": d[3], "other_threads_run": other,
                     "other_top": [(round(x[0], 3), x[1]) for x in oth[:3]]})
        print(f"upd {upd} n_mb {n_mb} sec {sec:.2f} ({1000 * sec / n_mb:.1f} ms/mb) | work thread run {d[0]:.2f}s "
              f"runq {d[1]:.3f}s | autograd run {d[2]:.2f}s runq {d[3]:.3f}s | other threads {other:.2f}s "
              f"{[(round(x[0], 2), x[1]) for x in oth[:3]]}", flush=True)
    if rows:
        n = sum(r["n_mb"] for r in rows)
        tot = {k: sum(r[k] for r in rows) for k in ("sec", "work_run", "work_runq", "autograd_run", "autograd_runq",
                                                     "other_threads_run")}
        print(f"{len(rows)} updates, {n} minibatches, per minibatch: wall {1000 * tot['sec'] / n:.1f} ms | work thread "
              f"run {1000 * tot['work_run'] / n:.1f} ms, runq {1000 * tot['work_runq'] / n:.1f} ms | autograd run "
              f"{1000 * tot['autograd_run'] / n:.1f} ms, runq {1000 * tot['autograd_runq'] / n:.1f} ms | other threads "
              f"{1000 * tot['other_threads_run'] / n:.1f} ms")
    else:
        print("no update ended inside the window")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"pid": pid, "work": work, "autograd": autograd, "log": log, "updates": rows}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
