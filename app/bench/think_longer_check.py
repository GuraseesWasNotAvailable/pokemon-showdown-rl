#!/usr/bin/env python
"""
What the longer search of a set-up turn (search_player.ThinkLonger) costs and buys on a real Foul
Play helper: every decision of the recorded battles (tests/fixtures/foulplay_bridge_trace.jsonl.gz)
is asked at the ladder's 500 ms and at the long search's time with 1 and with 2 engine threads,
in a quiet helper, and the wall time from the question to the answer, the iterations and whether
the best action changed are written per decision.

    nice -n 15 python bench/think_longer_check.py --ms 2000 --decisions 40 --out think_longer.json

It answers two questions before the flag is used on the ladder:
  * does the long answer come within ThinkLonger.wait_s() (the deadline the player gives it)?
  * do two threads buy about twice the iterations in the same wall time?
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import foulplay_bridge as fb  # noqa: E402
from search_player import ThinkLonger  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=int, default=2000)
    ap.add_argument("--base-ms", type=int, default=500)
    ap.add_argument("--decisions", type=int, default=40, help="decisions asked at most (3 questions each)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    wait = ThinkLonger(ms=a.ms).wait_s()
    recorded = fb.load_trace()
    bridge = fb.FoulPlayBridge(helpers=1, search_ms=a.base_ms, timeout=60.0, hang_timeout=120.0)
    bridge.start(timeout=180.0)
    rows = []
    try:
        for events in recorded:
            if len(rows) >= a.decisions:
                break
            user = str(events[-1].get("username") or "check")
            room = str(next(e["room"] for e in events if e["t"] == "msg"))
            key = "%s|%s#tl" % (user, room)
            try:
                for event in events:
                    if event["t"] == "msg":
                        bridge.feed(key, user, room, event["data"])
                    elif event["t"] == "order":
                        if len(rows) < a.decisions:
                            row = {"battle": room, "rqid": event.get("rqid")}
                            for name, ms, threads in (("base", a.base_ms, 0), ("long1", a.ms, 1), ("long2", a.ms, 2)):
                                began = time.monotonic()
                                try:
                                    more = {"threads": threads} if threads > 1 else {}
                                    advice = bridge.advice_future(key, search_ms=ms, rqid=event.get("rqid"), **more).result(timeout=120.0)
                                except Exception as exc:
                                    row[name] = {"error": str(exc)[:120]}
                                    continue
                                row[name] = {"wall_s": round(time.monotonic() - began, 3), "iterations": int(advice.iterations or 0),
                                             "best": advice.best}
                            if all("best" in row.get(n, {}) for n in ("base", "long1", "long2")):
                                rows.append(row)
                        bridge.note(key, str(event.get("choice") or "default"))
            finally:
                bridge.forget(key)
    finally:
        bridge.close()

    def col(name, field):
        return [r[name][field] for r in rows]

    out = {"decisions": len(rows), "long_ms": a.ms, "base_ms": a.base_ms, "wait_s": round(wait, 3)}
    for name in ("base", "long1", "long2"):
        walls = col(name, "wall_s")
        out[name] = {"wall_s_median": round(statistics.median(walls), 3) if walls else None,
                     "wall_s_max": round(max(walls), 3) if walls else None,
                     "iterations_median": int(statistics.median(col(name, "iterations"))) if rows else None,
                     "over_wait": sum(1 for w in walls if w > wait)}
    out["long1_best_differs_from_base"] = sum(1 for r in rows if r["long1"]["best"] != r["base"]["best"])
    out["long2_best_differs_from_base"] = sum(1 for r in rows if r["long2"]["best"] != r["base"]["best"])
    print(json.dumps(out, indent=1))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"summary": out, "rows": rows}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
