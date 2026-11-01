"""
Does mix keep its search when the deep arm searches beside it? Measured, not assumed.

    python bench/ab_deep_measure.py                      # the A/B's shape: 3 games at a time on 4 helpers
    python bench/ab_deep_measure.py --quick              # about 4 minutes (the launcher's check)
    python bench/ab_deep_measure.py --depth              # also the fixed-depth search's depth per arm's budget

The ladder A/B plays net, mix and mix-deep, one per game, 3 games at a time. A battle is pinned to
one helper (a process, one core while it searches), so the games never wait for each other's
helper; what they share is the CPU. Three phases, each with as many decisions asked at once as
games, each stream replaying recorded battles (the test fixture) and asking at their decision points:

    mix x3            today's worst case: three mix searches at once (500 ms per world, 1x the worlds)
    mix x1, deep x2   the A/B's worst case for mix: both other games are mix-deep
    mix x2, deep x1

and prints per phase and arm: decisions, iterations per decision (mean, median, lowest), worlds,
asked-to-answered ms (mean, 95th, slowest), and mix's iterations against the "mix x3" phase.
mix keeps its compute when it stays within 10% of "mix x3". The helpers' memory (their largest
resident size, from the helper's stats) says what WSL needs. --depth adds, per arm, the
depth poke-engine's iterative-deepening search reaches in the same time per world (the payoff
table of the engine, "engine" mode: its depth_searched per world), the "how many turns ahead"
proxy (the Monte Carlo search itself does not report a depth).

Standard library only, plus foulplay_bridge.py. Nothing is installed, nothing is deleted. Stops
its helpers at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from typing import Any, Dict, List, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import foulplay_bridge as fb  # noqa: E402

MIX = {"name": "mix", "ms": 500, "parallelism": 1, "workers": 1}
DEEP = {"name": "mix-deep", "ms": 1250, "parallelism": 2, "workers": 1}


def percentile(values: List[float], share: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(share * (len(ordered) - 1))))]


def phase(helpers: int, arms: List[Dict[str, Any]], points: int, depth: bool, threads: int = 0) -> Dict[str, Any]:
    """One bridge, one stream per entry of arms, all at once. Per arm: the numbers of its decisions."""
    recorded = fb.load_trace()
    bridge = fb.FoulPlayBridge(helpers=helpers, timeout=300.0)
    bridge.start()
    rows: Dict[str, Dict[str, List[float]]] = {}
    streams: Dict[int, Dict[str, Any]] = {}
    problems: List[str] = []
    lock = threading.Lock()

    def stream(n: int, arm: Dict[str, Any]) -> None:
        done = 0
        row = rows.setdefault(arm["name"], {"iterations": [], "wall_ms": [], "worlds": [], "depth": []})
        # This stream's decisions by position: stream n replays the same battles in every phase, so
        # its positions pair across phases (main: mix's change, paired).
        points_of = streams.setdefault(n, {"arm": arm["name"], "points": {}})["points"]
        for b in range(len(recorded)):
            events = recorded[(b + n) % len(recorded)]
            user = str(events[-1].get("username") or "measure")
            room = str(next(e["room"] for e in events if e["t"] == "msg"))
            key = "%s|%s#deep%d" % (user, room, n)
            try:
                for k, event in enumerate(events):
                    if event["t"] == "msg":
                        bridge.feed(key, user, room, event["data"])
                    elif event["t"] == "order":
                        if done >= points:
                            return
                        more: Dict[str, Any] = {}
                        if arm["parallelism"] > 1:
                            more["parallelism"] = arm["parallelism"]
                        if int(arm.get("workers") or 1) > 1:
                            more["workers"] = int(arm["workers"])
                        if threads > 1:
                            more["threads"] = threads
                        if depth:
                            # The fixed-depth search in the same time per world as the Monte Carlo one.
                            more.update(payoff_ms=float(arm["ms"]) * 2 * arm["parallelism"], payoff="engine")
                        try:
                            advice = bridge.advice_future(key, search_ms=arm["ms"], rqid=event.get("rqid"), **more).result(timeout=300.0)
                        except Exception as exc:
                            if not getattr(exc, "revive", False):
                                with lock:
                                    problems.append(str(exc) or type(exc).__name__)
                            advice = None
                        if advice is not None:
                            done += 1
                            with lock:
                                points_of["%d:%d" % ((b + n) % len(recorded), k)] = int(advice.iterations)
                                row["iterations"].append(advice.iterations)
                                row["wall_ms"].append(advice.wall_ms)
                                row["worlds"].append(advice.worlds)
                                for world in (advice.payoff or {}).get("worlds") or []:
                                    if world.get("depth") is not None:
                                        row["depth"].append(int(world["depth"]))
                        bridge.note(key, str(event.get("choice") or "default"))
            finally:
                bridge.forget(key)

    try:
        workers = [threading.Thread(target=stream, args=(n, arm), daemon=True) for n, arm in enumerate(arms)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        stats = [s for s in bridge.helper_stats(timeout=30.0) if s]
    finally:
        bridge.close()
    out: Dict[str, Any] = {"arms": [a["name"] for a in arms], "rows": {}, "problems": problems[:10],
                           "streams": {str(n): v for n, v in sorted(streams.items())},
                           "helper_rss_mb_max": max([float(s.get("rss_mb") or 0) for s in stats] or [0.0]),
                           "helper_peak_rss_mb_max": max([float(s.get("peak_rss_mb") or 0) for s in stats] or [0.0])}
    for name, row in rows.items():
        n = len(row["iterations"])
        if not n:
            out["rows"][name] = {"decisions": 0}
            continue
        out["rows"][name] = {
            "decisions": n, "iterations_mean": int(statistics.fmean(row["iterations"])),
            "iterations_median": int(statistics.median(row["iterations"])), "iterations_min": int(min(row["iterations"])),
            "worlds_mean": round(statistics.fmean(row["worlds"]), 2),
            "wall_ms_mean": round(statistics.fmean(row["wall_ms"]), 1), "wall_ms_p95": round(percentile(row["wall_ms"], 0.95), 1),
            "wall_ms_max": round(max(row["wall_ms"]), 1),
            "depth_mean": round(statistics.fmean(row["depth"]), 2) if row["depth"] else None,
        }
    return out


def paired_points(first: Dict[str, Any], later: Dict[str, Any], arm: str) -> List[Tuple[int, int]]:
    """(iterations in `first`, iterations in `later`) of every position a stream of `arm` searched in both phases."""
    pairs: List[Tuple[int, int]] = []
    for n, s_later in (later.get("streams") or {}).items():
        s_first = (first.get("streams") or {}).get(n)
        if not s_first or s_first.get("arm") != arm or s_later.get("arm") != arm:
            continue
        for point, after in s_later["points"].items():
            before = s_first["points"].get(point)
            if before:
                pairs.append((int(before), int(after)))
    return pairs


PHASES: List[Tuple[str, List[Dict[str, Any]]]] = [
    ("mix x3", [MIX, MIX, MIX]),
    ("mix x1, deep x2", [MIX, DEEP, DEEP]),
    ("mix x2, deep x1", [MIX, MIX, DEEP]),
]


def phases_for(streams: int) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """The three phases for `streams` games at a time: all mix, one mix among deep ones, half and half."""
    if streams == 3:
        return PHASES
    n = max(2, int(streams))
    half = n // 2
    return [("mix x%d" % n, [MIX] * n),
            ("mix x1, deep x%d" % (n - 1), [MIX] + [DEEP] * (n - 1)),
            ("mix x%d, deep x%d" % (n - half, half), [MIX] * (n - half) + [DEEP] * half)]


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description="mix's iterations with the deep arm searching beside it")
    ap.add_argument("--helpers", type=int, default=4)
    ap.add_argument("--points", type=int, default=12, help="decisions per stream and phase")
    ap.add_argument("--quick", action="store_true", help="30 decisions per stream, the first two phases (about 4 minutes)")
    ap.add_argument("--depth", action="store_true", help="also the fixed-depth search's depth in each arm's time per world")
    ap.add_argument("--deep-ms", type=int, default=DEEP["ms"])
    ap.add_argument("--deep-worlds", type=int, default=DEEP["parallelism"])
    ap.add_argument("--deep-workers", type=int, default=DEEP["workers"],
                    help="the deep arm's worlds searched this many at a time (ladder.py --deep-workers)")
    ap.add_argument("--mix-ms", type=int, default=MIX["ms"], help="mix's search time per world (ladder.py --search-ms)")
    ap.add_argument("--streams", type=int, default=3, help="games at a time (ladder.py --parallel): the phases are built for this many")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)
    DEEP.update(ms=args.deep_ms, parallelism=args.deep_worlds, workers=max(1, args.deep_workers))
    MIX.update(ms=args.mix_ms)
    ok, why = fb.available()
    print(why, flush=True)
    if not ok:
        return 1
    points = 30 if args.quick else args.points
    every = phases_for(args.streams)
    phases = every[:2] if args.quick else every
    print(f"mix {MIX['ms']} ms per world; mix-deep {DEEP['ms']} ms per world on {DEEP['parallelism']}x the worlds"
          + (f", {DEEP['workers']} at a time" if DEEP["workers"] > 1 else "") + f"; {args.streams} games at a time on {args.helpers} helpers",
          flush=True)
    results = []
    began = time.monotonic()
    for title, arms in phases:
        result = phase(args.helpers, arms, points, args.depth)
        result["phase"] = title
        results.append(result)
        print(f"{title}: {args.helpers} helpers, helpers' memory {result['helper_rss_mb_max']:.0f} MB now"
              + (f", {result['helper_peak_rss_mb_max']:.0f} MB at the most" if result["helper_peak_rss_mb_max"] else ""), flush=True)
        for name, row in result["rows"].items():
            if not row.get("decisions"):
                print(f"  {name:<9} no decisions")
                continue
            print(f"  {name:<9} {row['decisions']:3d} decisions, {row['iterations_mean']:>9,} iterations per decision "
                  f"(median {row['iterations_median']:,}, lowest {row['iterations_min']:,}), {row['worlds_mean']:.2f} worlds, "
                  f"asked to answered {row['wall_ms_mean']:.0f} ms (95th {row['wall_ms_p95']:.0f}, slowest {row['wall_ms_max']:.0f})"
                  + (f", fixed-depth search {row['depth_mean']:.2f} plies" if row.get("depth_mean") is not None else ""), flush=True)
        for problem in result["problems"]:
            print(f"  problem: {problem}")
    base = results[0]["rows"].get("mix", {}).get("iterations_mean")
    verdict = 0
    if base:
        for result in results[1:]:
            mix = result["rows"].get("mix", {}).get("iterations_mean")
            deep = result["rows"].get("mix-deep", {}).get("iterations_mean")
            if mix:
                # Paired: mix's streams on the same positions as in "mix x3" (stream n replays the same battles).
                pairs = paired_points(results[0], result, "mix")
                if len(pairs) >= 10:
                    change = sum(after for _b, after in pairs) / max(1, sum(before for before, _a in pairs)) - 1.0
                    how = f"paired, the same {len(pairs)} positions"
                else:
                    change = (mix - base) / base
                    how = f"unpaired, only {len(pairs)} positions in common"
                result["mix_change"] = {"change": round(change, 4), "how": how}
                ok_mix = abs(change) <= 0.10
                verdict |= 0 if ok_mix else 2
                print(f"{result['phase']}: mix {100 * change:+.1f}% against {results[0]['phase']} ({how}) "
                      + ("(within 10%: mix keeps its search)" if ok_mix else "(MORE THAN 10%: the deep arm takes mix's compute here)")
                      + (f"; mix-deep {deep / mix:.1f} times mix" if deep else ""))
    print(f"measured in {time.monotonic() - began:.0f} s")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"platform": sys.platform, "results": results}, fh, indent=1)
    return verdict


if __name__ == "__main__":
    sys.exit(main())
