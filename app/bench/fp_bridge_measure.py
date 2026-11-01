"""
How hard does the search work on this machine, and how long does an answer take?

    python bench/fp_bridge_measure.py --helpers 2,4,6 --ms 100,250,500,1000 --points 12

For every helper count a bridge is started with that many helpers and as many streams as helpers
(--streams for another number: 3 for a ladder bot that plays 3 games at once). Each stream
replays recorded battles (the test fixture) and, at each of its first --points decision points,
asks the same position once at every search time. Per search time it prints

    decisions, iterations per decision (mean, median, lowest), the time inside the helper,
    the time from asked to answered (mean, median, 95th percentile, slowest), worlds per decision

Search strength is the ITERATIONS: the time only says what they cost here, now, under this load.
The same script runs on Windows (the helpers are then started inside WSL through wsl.exe, see
foulplay_bridge.py) and inside WSL or on Linux (the helpers are direct children), so the two can
be compared. --payoff adds a payoff table of that many ms to every decision, --threads N asks
poke-engine for N threads per world. --json FILE keeps the numbers.

Standard library only, plus foulplay_bridge.py. Nothing is installed, nothing is deleted.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from typing import Any, Dict, List

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import foulplay_bridge as fb  # noqa: E402


def percentile(values: List[float], share: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(share * (len(ordered) - 1))))]


def measure(helpers: int, streams: int, ms_list: List[int], points: int, more: Dict[str, Any]) -> Dict[str, Any]:
    recorded = fb.load_trace()
    bridge = fb.FoulPlayBridge(helpers=helpers, timeout=300.0)
    began = time.monotonic()
    bridge.start()
    start_s = time.monotonic() - began
    rows: Dict[int, Dict[str, List[float]]] = {ms: {"iterations": [], "helper_ms": [], "wall_ms": [], "worlds": [], "payoff_ms": [],
                                                    "payoff_known": [], "payoff_depth": []} for ms in ms_list}
    problems: List[str] = []
    lock = threading.Lock()

    def stream(n: int) -> None:
        done = 0
        for b in range(len(recorded)):
            events = recorded[(b + n) % len(recorded)]
            user = str(events[-1].get("username") or "measure")
            room = str(next(e["room"] for e in events if e["t"] == "msg"))
            key = "%s|%s#measure%d" % (user, room, n)
            try:
                for event in events:
                    if event["t"] == "msg":
                        bridge.feed(key, user, room, event["data"])
                    elif event["t"] == "order":
                        if done >= points:
                            return
                        asked = False
                        for ms in ms_list:
                            try:
                                advice = bridge.advice_future(key, search_ms=ms, rqid=event.get("rqid"), **more).result(timeout=300.0)
                            except Exception as exc:
                                if not getattr(exc, "revive", False):
                                    with lock:
                                        problems.append(str(exc) or type(exc).__name__)
                                break
                            asked = True
                            with lock:
                                row = rows[ms]
                                row["iterations"].append(advice.iterations)
                                row["helper_ms"].append(advice.ms)
                                row["wall_ms"].append(advice.wall_ms)
                                row["worlds"].append(advice.worlds)
                                if advice.payoff:
                                    row["payoff_ms"].append(float(advice.payoff.get("ms") or 0.0))
                                    for world in advice.payoff.get("worlds") or []:
                                        if world.get("cells"):
                                            row["payoff_known"].append(world["known"] / world["cells"])
                                            row["payoff_depth"].append(world.get("depth") or 0)
                        if asked:
                            done += 1
                        bridge.note(key, str(event.get("choice") or "default"))
            finally:
                bridge.forget(key)

    try:
        threads = [threading.Thread(target=stream, args=(n,), daemon=True) for n in range(streams)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        wsl = bool(bridge._wsl)
    finally:
        closing = time.monotonic()
        bridge.close()
        close_s = time.monotonic() - closing
    out_rows = []
    for ms in ms_list:
        row = rows[ms]
        n = len(row["iterations"])
        if not n:
            out_rows.append({"ms": ms, "decisions": 0})
            continue
        entry = {
            "ms": ms, "decisions": n,
            "iterations_mean": int(statistics.fmean(row["iterations"])), "iterations_median": int(statistics.median(row["iterations"])),
            "iterations_min": int(min(row["iterations"])),
            "helper_ms_mean": round(statistics.fmean(row["helper_ms"]), 1),
            "wall_ms_mean": round(statistics.fmean(row["wall_ms"]), 1), "wall_ms_median": round(statistics.median(row["wall_ms"]), 1),
            "wall_ms_p95": round(percentile(row["wall_ms"], 0.95), 1), "wall_ms_max": round(max(row["wall_ms"]), 1),
            "worlds_mean": round(statistics.fmean(row["worlds"]), 2),
        }
        if row["payoff_ms"]:
            entry["payoff_ms_mean"] = round(statistics.fmean(row["payoff_ms"]), 1)
            entry["payoff_known_share"] = round(statistics.fmean(row["payoff_known"]), 3) if row["payoff_known"] else None
            entry["payoff_depth_mean"] = round(statistics.fmean(row["payoff_depth"]), 2) if row["payoff_depth"] else None
            entry["payoff_depth_min"] = min(row["payoff_depth"]) if row["payoff_depth"] else None
            entry["payoff_depth_max"] = max(row["payoff_depth"]) if row["payoff_depth"] else None
        out_rows.append(entry)
    return {"helpers": helpers, "streams": streams, "points": points, "wsl": wsl, "start_s": round(start_s, 2),
            "close_s": round(close_s, 2), "rows": out_rows, "problems": problems[:10], "more": more}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--helpers", default="2,4,6")
    parser.add_argument("--streams", type=int, default=0, help="decisions asked at the same time (default: one per helper)")
    parser.add_argument("--ms", default="100,250,500,1000")
    parser.add_argument("--points", type=int, default=12, help="decision points per stream")
    parser.add_argument("--payoff", type=float, default=0.0, help="payoff table budget in ms, added to every decision")
    parser.add_argument("--payoff-mode", default="engine", choices=("engine", "full"))
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=0, help="exact iterations per world instead of a time")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    ok, text = fb.available()
    print(text, flush=True)
    if not ok:
        return 1
    more: Dict[str, Any] = {}
    if args.payoff:
        more.update(payoff_ms=args.payoff, payoff=args.payoff_mode)
    if args.threads:
        more["threads"] = args.threads
    if args.iterations:
        more["iterations"] = args.iterations
    ms_list = [int(x) for x in args.ms.split(",") if x.strip()]
    results = []
    for helpers in [int(x) for x in args.helpers.split(",") if x.strip()]:
        result = measure(helpers, args.streams or helpers, ms_list, args.points, more)
        results.append(result)
        print("%d helpers, %d at a time, %s%s (started in %.1f s, closed in %.1f s)" % (
            result["helpers"], result["streams"], "through wsl.exe" if result["wsl"] else "direct",
            (", " + json.dumps(more)) if more else "", result["start_s"], result["close_s"]))
        print("  search_ms  decisions  iterations/decision mean (median, min)   in helper ms   asked->answered ms mean (median, p95, max)  worlds")
        for row in result["rows"]:
            if not row["decisions"]:
                print("  %9d  none" % row["ms"])
                continue
            print("  %9d  %9d  %12d (%7d, %7d)            %10.1f   %10.1f (%7.1f, %7.1f, %7.1f)            %5.2f%s" % (
                row["ms"], row["decisions"], row["iterations_mean"], row["iterations_median"], row["iterations_min"],
                row["helper_ms_mean"], row["wall_ms_mean"], row["wall_ms_median"], row["wall_ms_p95"], row["wall_ms_max"],
                row["worlds_mean"],
                ("   payoff %.0f ms, depth %.2f (%s to %s), cells known %.0f%%" % (
                    row["payoff_ms_mean"], row.get("payoff_depth_mean") or 0, row.get("payoff_depth_min"), row.get("payoff_depth_max"),
                    100 * (row.get("payoff_known_share") or 0))) if "payoff_ms_mean" in row else ""))
        for problem in result["problems"]:
            print("  problem: %s" % problem)
        sys.stdout.flush()
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"platform": sys.platform, "results": results}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
