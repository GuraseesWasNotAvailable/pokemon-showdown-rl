"""
The A/B's search budgets on another machine, set by the laptop's own games: the laptop's ladder
games are replayed through the bridge here, each decision searched with this machine's settings
for its arm, and the iterations here are put next to the iterations the laptop logged for the
same decision (its decision note, matched by request id). Measured in search iterations, not
milliseconds: the arms must get the compute per decision they had on the laptop.

    python bench/ab_box_calibrate.py BATTLES_DIR [BATTLES_DIR ...] --mix-ms 500 --deep-ms 1250 \
        --deep-worlds 2 --deep-workers 4 --helpers 7 --phase mix:6 --phase mix:3,mix-deep:3 \
        --phase mix:1,mix-deep:5 --seconds 240 [--json OUT]

BATTLES_DIR holds ladder.py battle files (runs/ladder/battles/*.jsonl). A game's arm is the "arm"
of its first note (mix or mix-deep; net games have no search and are skipped). Only decisions the
laptop searched with its arm's own budget are paired: mix at --laptop-mix-ms (500) on 1x the worlds,
mix-deep at --laptop-deep-ms (1250) on --deep-worlds the worlds (a deep game that fell back to
mix's budget is paired up to that point).

Every --phase is a set of games at once, e.g. "mix:3,mix-deep:3" (six streams, three of each arm):
each stream replays its arm's games one after the other (from its own place in the list, again
from the start when they run out) until --seconds have passed. Per phase and arm it prints the
decisions, this machine's iterations per decision against the laptop's on the same decisions
(ratio of the means, and the median of the ratios), the worlds on both, and asked to answered ms
(mean, 95th, slowest). A ratio of 1.00 is the laptop's compute; mix's ladder mean then stays the
laptop's (538,000 per decision on 024381, bench/ab_report.py --mix-iterations-ref).

Standard library only, plus foulplay_bridge.py. Nothing is installed, nothing is deleted. Stops
its helpers at the end.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import foulplay_bridge as fb  # noqa: E402

ARMS = ("mix", "mix-deep")


def to_id(s: Any) -> str:
    return "".join(c for c in str(s or "").lower() if c.isalnum())


def percentile(values: List[float], share: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(share * (len(ordered) - 1))))]


def _choice_of(text: str, request: Dict[str, Any]) -> str:
    """'/choose move flamethrower terastallize|7' as the request's slots: 'move 2 terastallize'."""
    text = text[len("/choose "):].split("|")[0].strip()
    tera = text.endswith(" terastallize")
    if tera:
        text = text[: -len(" terastallize")]
    choice = None
    if text.startswith("move "):
        name = to_id(text[5:])
        if name.isdigit():
            choice = "move " + name
        else:
            for i, m in enumerate(((request.get("active") or [{}])[0].get("moves") or []), 1):
                if name in (to_id(m.get("id")), to_id(m.get("move"))):
                    choice = "move %d" % i
                    break
    elif text.startswith("switch "):
        name = to_id(text[7:])
        if name.isdigit():
            choice = "switch " + name
        else:
            for j, p in enumerate((request.get("side") or {}).get("pokemon") or [], 1):
                species = to_id(str(p.get("details", "")).split(",")[0])
                nick = to_id(str(p.get("ident", "")).split(":", 1)[-1])
                if name in (species, nick):
                    choice = "switch %d" % j
                    break
    choice = choice or "default"
    return choice + (" terastallize" if tera else "")


def convert(path: str) -> Tuple[List[Dict[str, Any]], Dict[Any, Dict[str, Any]], Optional[str]]:
    """(trace events, {rqid: the laptop's decision note}, arm) of one ladder.py battle file."""
    events: List[Dict[str, Any]] = []
    laptop: Dict[Any, Dict[str, Any]] = {}
    room = user = arm = None
    request: Optional[Dict[str, Any]] = None
    last_order: Optional[Dict[str, Any]] = None       # the order before the next request: how long the opponent and server took
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            d, data = row.get("dir"), row.get("data")
            if (d == "in" and isinstance(data, str) and "|request|" in data and last_order is not None
                    and isinstance(row.get("t"), (int, float)) and last_order.get("sent") is not None):
                last_order["gap"] = max(0.0, float(row["t"]) - float(last_order.pop("sent")))
                last_order = None
            if d == "note" and isinstance(data, dict):
                if "battle" in data and "user" in data and room is None:
                    room, user, arm = data["battle"], data["user"], data.get("arm")
                dec = data.get("decision")
                if isinstance(dec, dict) and dec.get("rqid") is not None:
                    laptop[dec["rqid"]] = dec
            elif d == "in" and isinstance(data, str):
                events.append({"t": "msg", "room": room, "data": data})
                for part in data.split("\n"):
                    if part.startswith("|request|") and len(part) > 10:
                        try:
                            request = json.loads(part[len("|request|"):])
                        except ValueError:
                            pass
            elif d == "out" and isinstance(data, str) and data.startswith("/choose ") and request:
                last_order = {"t": "order", "rqid": request.get("rqid"), "choice": _choice_of(data, request), "sent": row.get("t")}
                events.append(last_order)
    for e in events:
        e.pop("sent", None)
    name = user
    for e in events:
        if e["t"] == "msg":
            for part in e["data"].split("\n"):
                bits = part.split("|")
                if len(bits) > 3 and bits[1] == "player" and to_id(bits[3]) == to_id(user):
                    name = bits[3]
    events.append({"t": "end", "username": name})
    return events, laptop, arm


def paired(dec: Dict[str, Any], arm: str, args: argparse.Namespace) -> bool:
    """A laptop decision searched with its arm's own budget (and so comparable)."""
    if not (dec.get("asked") and dec.get("iterations")):
        return False
    if dec.get("deep_fallback"):
        return False
    ms = dec.get("asked_ms")
    wx = int(dec.get("worlds_x") or 1)
    if arm == "mix":
        return (ms is None or int(ms) == args.laptop_mix_ms) and wx == 1
    return ms is not None and int(ms) == args.laptop_deep_ms and wx == args.deep_worlds


def parse_phase(text: str) -> List[str]:
    arms: List[str] = []
    for part in text.split(","):
        arm, _, n = part.strip().partition(":")
        if arm not in ARMS or not n.isdigit():
            raise SystemExit("a phase is like mix:3,mix-deep:3, not %r" % text)
        arms += [arm] * int(n)
    return arms


def run_phase(games: Dict[str, List[Tuple[List[Dict[str, Any]], Dict[Any, Dict[str, Any]]]]], arms: List[str],
              args: argparse.Namespace) -> Dict[str, Any]:
    bridge = fb.FoulPlayBridge(helpers=args.helpers, timeout=300.0)
    bridge.start()
    rows: Dict[str, List[Dict[str, Any]]] = {arm: [] for arm in ARMS}
    errors: List[str] = []
    lock = threading.Lock()
    began = time.monotonic()
    stop_at = began + args.seconds

    def stream(n: int, arm: str) -> None:
        own = games[arm]
        first = sum(1 for a in arms[:n] if a == arm)            # this stream's place among its arm's streams
        k, cycle = first * max(1, len(own) // max(1, arms.count(arm))), 0
        while time.monotonic() < stop_at:
            events, lap = own[k % len(own)]
            k += 1
            cycle += 1
            user = str(events[-1].get("username") or "replay")
            room = str(next(e["room"] for e in events if e["t"] == "msg"))
            key = "%s|%s#s%dc%d" % (user, room, n, cycle)
            try:
                for e in events:
                    if time.monotonic() >= stop_at:
                        return
                    if e["t"] == "msg":
                        bridge.feed(key, user, room, e["data"])
                    elif e["t"] == "order":
                        dec = lap.get(e.get("rqid")) or {}
                        if paired(dec, arm, args):
                            more: Dict[str, Any] = {}
                            if arm == "mix":
                                ms = args.mix_ms
                            else:
                                ms = args.deep_ms
                                if args.deep_worlds > 1:
                                    more["parallelism"] = args.deep_worlds
                                if args.deep_workers > 1:
                                    more["workers"] = args.deep_workers
                            try:
                                a = bridge.advice_future(key, search_ms=ms, rqid=e.get("rqid"), **more).result(timeout=300.0)
                            except Exception as exc:
                                a = None
                                with lock:
                                    errors.append(repr(exc)[:300])
                            if a is not None:
                                with lock:
                                    rows[arm].append({"box": int(a.iterations), "box_worlds": int(a.worlds), "box_ms": float(a.wall_ms),
                                                      "laptop": int(dec["iterations"]), "laptop_worlds": dec.get("worlds"),
                                                      "stream": n})
                        bridge.note(key, str(e.get("choice") or "default"))
                        if args.pace and e.get("gap"):
                            # The ladder's own rhythm: the opponent's time to answer, as it was on the laptop.
                            time.sleep(max(0.0, min(float(e["gap"]), args.pace_cap, stop_at - time.monotonic())))
            finally:
                bridge.forget(key)

    try:
        threads = [threading.Thread(target=stream, args=(n, arm), daemon=True) for n, arm in enumerate(arms)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stats = [s for s in bridge.helper_stats(timeout=30.0) if s]
    finally:
        bridge.close()
    out: Dict[str, Any] = {"arms": arms, "seconds": round(time.monotonic() - began, 1), "errors": errors[:10],
                           "n_errors": len(errors), "rows": {},
                           "helper_peak_rss_mb_max": max([float(s.get("peak_rss_mb") or 0) for s in stats] or [0.0])}
    for arm, got in rows.items():
        if not got:
            continue
        box = [r["box"] for r in got]
        lap = [r["laptop"] for r in got]
        ratios = [r["box"] / r["laptop"] for r in got if r["laptop"]]
        same = [r for r in got if r["laptop_worlds"] is not None and int(r["laptop_worlds"]) == r["box_worlds"]]
        out["rows"][arm] = {
            "decisions": len(got), "box_mean": int(statistics.fmean(box)), "laptop_mean": int(statistics.fmean(lap)),
            "box_over_laptop": round(statistics.fmean(box) / max(1.0, statistics.fmean(lap)), 4),
            "median_ratio": round(statistics.median(ratios), 4) if ratios else None,
            "same_worlds": len(same), "box_worlds": round(statistics.fmean(r["box_worlds"] for r in got), 2),
            "laptop_worlds": round(statistics.fmean(int(r["laptop_worlds"] or 0) for r in got), 2),
            "ms_mean": round(statistics.fmean(r["box_ms"] for r in got), 1),
            "ms_p95": round(percentile([r["box_ms"] for r in got], 0.95), 1),
            "ms_max": round(max(r["box_ms"] for r in got), 1),
        }
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="the A/B's search budgets here, paired with the laptop's own decisions")
    ap.add_argument("battles", nargs="+", help="folders of ladder.py battle files")
    ap.add_argument("--mix-ms", type=int, default=500)
    ap.add_argument("--deep-ms", type=int, default=1250)
    ap.add_argument("--deep-worlds", type=int, default=2)
    ap.add_argument("--deep-workers", type=int, default=1)
    ap.add_argument("--laptop-mix-ms", type=int, default=500)
    ap.add_argument("--laptop-deep-ms", type=int, default=1250)
    ap.add_argument("--helpers", type=int, default=7)
    ap.add_argument("--phase", action="append", default=[], help="e.g. mix:6 or mix:3,mix-deep:3 (repeat for more phases)")
    ap.add_argument("--seconds", type=float, default=240.0, help="each phase runs this long")
    ap.add_argument("--pace", action="store_true",
                    help="after each order wait as long as the laptop waited for the next request (the opponent's time; "
                         "up to --pace-cap s): the ladder's rhythm, instead of one search right after the other")
    ap.add_argument("--pace-cap", type=float, default=30.0)
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)
    phases = [parse_phase(p) for p in (args.phase or ["mix:3,mix-deep:3"])]

    games: Dict[str, List[Tuple[List[Dict[str, Any]], Dict[Any, Dict[str, Any]]]]] = {arm: [] for arm in ARMS}
    files = sorted(f for folder in args.battles for f in glob.glob(os.path.join(folder, "*.jsonl")))
    for f in files:
        events, lap, arm = convert(f)
        if arm in ARMS and any(e["t"] == "msg" and e["room"] for e in events) and any(paired(d, arm, args) for d in lap.values()):
            games[arm].append((events, lap))
    for arm in ARMS:
        n = sum(1 for _b, lap in games[arm] for d in lap.values() if paired(d, arm, args))
        print(f"{arm}: {len(games[arm])} laptop games, {n} decisions to pair", flush=True)
    ok, why = fb.available()
    print(why, flush=True)
    if not ok:
        return 1
    print(f"here: mix {args.mix_ms} ms per world; mix-deep {args.deep_ms} ms per world on {args.deep_worlds}x the worlds"
          + (f", {args.deep_workers} at a time" if args.deep_workers > 1 else "") + f"; {args.helpers} helpers; "
          f"the laptop: mix {args.laptop_mix_ms} ms, mix-deep {args.laptop_deep_ms} ms"
          + ("; paced as the laptop's games (the opponent's time, up to %.0f s)" % args.pace_cap if args.pace else
             "; every search right after the one before (the worst case)"), flush=True)
    results = []
    for arms in phases:
        if any(not games[a] for a in arms):
            print(f"phase {arms}: no laptop games of an arm in it, skipped")
            continue
        title = ", ".join(f"{a} x{arms.count(a)}" for a in ARMS if a in arms)
        result = run_phase(games, arms, args)
        result["phase"] = title
        results.append(result)
        print(f"{title}: {result['seconds']:.0f} s, helpers' memory {result['helper_peak_rss_mb_max']:.0f} MB at the most"
              + (f", {result['n_errors']} errors" if result["n_errors"] else ""), flush=True)
        for arm, row in result["rows"].items():
            print(f"  {arm:<9} {row['decisions']:4d} decisions: here {row['box_mean']:>10,} iterations per decision, the laptop "
                  f"{row['laptop_mean']:>10,} on the same decisions: {row['box_over_laptop']:.3f} (median of the ratios "
                  f"{row['median_ratio']:.3f}); worlds {row['box_worlds']:.2f} here, {row['laptop_worlds']:.2f} there; asked to "
                  f"answered {row['ms_mean']:.0f} ms (95th {row['ms_p95']:.0f}, slowest {row['ms_max']:.0f})", flush=True)
        for error in result["errors"][:3]:
            print("  error:", error)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"args": vars(args), "results": results}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
