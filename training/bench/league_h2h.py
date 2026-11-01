#!/usr/bin/env python
"""
The league run's bars (bench/notes/LEAGUE.md section 7): a league checkpoint against frozen M24381, networks
alone, head to head, on one local server, with this line's own players (bench/gauntlet.py), which play
both networks on the observation each was trained on (hybrid.PolicyNet reads it from the checkpoint:
main_line for both here). The cells copy ladder-pick's B cells (its bench/ladderpick/ladderpick.py
cells_snew): the new network through ours (gauntlet.py ours --arms net, ladder.make_player, the app's
player) against the reference through the opponent (gauntlet.py opp --kind net, the batched network
player), 300 games, then the two implementations swapped, 100 games; each cell's sides balanced on its own.
The score is the new network's wins over both cells, out of 400, with a 95% Wilson interval.

    python bench/league_h2h.py run --new L100=CKPT --ref M24381=M24381w.pt --root DIR --port P [--main 300] [--swap 100]
    python bench/league_h2h.py report --root DIR

Why not ladder-pick itself: its players (branch decide) refuse a checkpoint with these settings in its
"obs_settings" (the league learner writes them so), and its game code is held to a manifest. A port of
obs_main.py to it is the follow-up; until then this harness plays the same cells with the same counts.
Local servers only; nothing but challenges, orders and the room leave is sent.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
SHOWDOWN = os.path.expanduser("~/pokemon-showdown")


def wilson(k: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def read_lines(path: str) -> List[Dict[str, Any]]:
    out = []
    if os.path.isfile(path):
        for line in open(path, encoding="utf-8"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def games(folder: str) -> Tuple[int, float]:
    """(games with a result, ours' score: a win 1, a tie 0.5)."""
    rows = [r for r in read_lines(os.path.join(folder, "results.jsonl")) if not r.get("stalled") and r.get("arm")]
    return len(rows), sum(1.0 if r.get("won") else (0.5 if r.get("tie") else 0.0) for r in rows)


def port_up(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def run_cell(root: str, name: str, ours: str, opp: str, n: int, port: int, idx: int, parallel: int) -> Dict[str, Any]:
    d = os.path.join(root, "cells", name)
    os.makedirs(d, exist_ok=True)
    done, wins = games(d)
    if done >= n:
        return {"name": name, "games": done, "wins": wins, "note": "already complete"}
    if port_up(port):
        raise SystemExit(f"port {port} is in use")
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")

    def spawn(argv, out):
        with open(os.path.join(d, out), "ab") as f:
            return subprocess.Popen(argv, cwd=REPO, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                                    start_new_session=True, env=env)

    procs = []
    try:
        procs.append(spawn([PY, "start_servers.py", "--dir", SHOWDOWN, "--ports", str(port), "--log-dir", os.path.join(d, "server")],
                           "server.out"))
        for _ in range(180):
            if port_up(port):
                break
            time.sleep(1)
        if not port_up(port):
            raise SystemExit(f"{name}: the server did not come up on {port}")
        acc, cha, us = f"lh{idx:02d}acc", f"lh{idx:02d}cha", f"lh{idx:02d}us"
        procs.append(spawn([PY, "bench/gauntlet.py", "opp", "--port", str(port), "--pairs", f"{acc}:{cha}:{us}", "--kind", "net",
                            "--checkpoint", opp, "--out", os.path.join(d, "opp.jsonl"), "--stop-file", os.path.join(d, "opp.stop")],
                           "opp.out"))
        for _ in range(300):
            try:
                if " ready: " in open(os.path.join(d, "opp.out"), encoding="utf-8", errors="replace").read():
                    break
            except FileNotFoundError:
                pass
            time.sleep(1)
        else:
            raise SystemExit(f"{name}: the opponent did not get ready (opp.out)")
        ours_p = spawn([PY, "bench/gauntlet.py", "ours", "--port", str(port), "--user", us, "--arms", "net", "--per-arm", str(n),
                        "--parallel", str(parallel), "--helpers", "1", "--checkpoint", ours, "--opp-accept", acc,
                        "--opp-challenge", cha, "--out", os.path.join(d, "results.jsonl"), "--stop-file", os.path.join(d, "stop"),
                        "--phase", name, "--seed", str(1000 + idx)], "ours.out")
        procs.append(ours_p)
        while ours_p.poll() is None:
            time.sleep(2)
        done, wins = games(d)
        return {"name": name, "games": done, "wins": wins, "ours_exit": ours_p.returncode}
    finally:
        open(os.path.join(d, "opp.stop"), "w").close()
        for p in reversed(procs):
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()


def report(root: str) -> Dict[str, Any]:
    meta = json.load(open(os.path.join(root, "h2h.json"), encoding="utf-8"))
    main_g, main_w = games(os.path.join(root, "cells", meta["main"]))
    swap_g, swap_w = games(os.path.join(root, "cells", meta["swap"]))
    k, n = main_w + (swap_g - swap_w), main_g + swap_g
    lo, hi = wilson(k, n)
    out = {"new": meta["new"], "ref": meta["ref"], "main": [main_w, main_g], "swap_new_wins": [swap_g - swap_w, swap_g],
           "score": [k, n], "rate": (k / n) if n else None, "ci95": [lo, hi]}
    print(f"{meta['new']} against {meta['ref']}: {k:g} of {n} ({100 * k / max(1, n):.1f}%, 95% {100 * lo:.1f} to {100 * hi:.1f}); "
          f"through ours {main_w:g}/{main_g}, swapped {swap_g - swap_w:g}/{swap_g} (a tie counts half)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--new", required=True, help="NAME=CHECKPOINT")
    r.add_argument("--ref", required=True, help="NAME=CHECKPOINT")
    r.add_argument("--root", required=True)
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--main", type=int, default=300)
    r.add_argument("--swap", type=int, default=100)
    r.add_argument("--parallel", type=int, default=6)
    p = sub.add_parser("report")
    p.add_argument("--root", required=True)
    a = ap.parse_args()
    root = os.path.abspath(os.path.expanduser(a.root))
    if a.cmd == "report":
        report(root)
        return 0
    (nn, npath), (rn, rpath) = (x.split("=", 1) for x in (a.new, a.ref))
    npath, rpath = (os.path.abspath(os.path.expanduser(x)) for x in (npath, rpath))
    os.makedirs(root, exist_ok=True)
    meta = {"new": nn, "ref": rn, "new_ckpt": npath, "ref_ckpt": rpath, "main": f"B-{nn}-{rn}", "swap": f"B-{rn}-{nn}",
            "counts": [a.main, a.swap], "port": a.port, "code": REPO}
    with open(os.path.join(root, "h2h.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(run_cell(root, meta["main"], npath, rpath, a.main, a.port, 1, a.parallel)), flush=True)
    print(json.dumps(run_cell(root, meta["swap"], rpath, npath, a.swap, a.port, 2, a.parallel)), flush=True)
    rep = report(root)
    with open(os.path.join(root, "h2h_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
