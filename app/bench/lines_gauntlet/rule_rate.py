"""On the box: recorded battles through the real helper at the play settings (500 ms search, then the
lines). How often the lines would change the search's favourite under the rule, how much the two
worlds' lines of one action differ, and the time. python rule_rate.py THREADS BATTLE [BATTLE ...]"""
import statistics
import sys
import time

sys.path.insert(0, ".")
import foulplay_bridge as fb  # noqa: E402

threads = int(sys.argv[1])
which = [int(x) for x in sys.argv[2:]] or [0]
spec = {"k": 3, "turns": 5, "ms": 1500, "worlds": 2, "threads": threads}
n = fire_old = fire_new = 0
spread, wall, line_ms, fav_minus_root = [], [], [], []
for index in which:
    events = fb.load_trace()[index]
    user, room = events[-1]["username"], events[0]["room"]
    key = user + "|" + room
    bridge = fb.FoulPlayBridge(helpers=1, search_ms=500, timeout=120.0).start()
    try:
        for event in events:
            if event["t"] == "msg":
                bridge.feed(key, user, room, event["data"])
            elif event["t"] == "order":
                began = time.monotonic()
                advice = bridge.advice_sync(key, rqid=event["rqid"], lines=spec)
                wall.append((time.monotonic() - began) * 1000.0)
                bridge.note(key, event["choice"])
                raw = advice.lines or {}
                line_ms.append(raw.get("ms") or 0.0)
                by = {}
                for world in raw.get("worlds") or []:
                    for one in world.get("lines") or []:
                        if one.get("value") is not None and one.get("turns", 0) >= 2:
                            by.setdefault(one["choice"], {})[world["world"]] = (world["chance"], one["value"])
                options = {o.choice: o for o in advice.options}
                if len(by) < 2:
                    continue
                mean = {c: sum(ch * v for ch, v in w.values()) / (sum(ch for ch, _ in w.values()) or 1.0) for c, w in by.items()}
                for w in by.values():
                    if len(w) == 2:
                        a, b = [v for _c, v in w.values()]
                        spread.append(abs(a - b))
                top = max(options.values(), key=lambda o: o.policy)
                if top.choice not in by:
                    continue
                n += 1
                if top.score is not None:
                    fav_minus_root.append(mean[top.choice] - top.score)
                rated = [c for c in by if c != top.choice and options.get(c) is not None and options[c].policy >= 0.25 * top.policy]
                if any(mean[c] - mean[top.choice] >= 0.08 for c in rated):
                    fire_old += 1
                ok = [c for c in rated if mean[c] - mean[top.choice] >= 0.10
                      and not any(wid in by[top.choice] and by[c][wid][1] < by[top.choice][wid][1] for wid in by[c])]
                fire_new += int(bool(ok))
    finally:
        bridge.close()
print("threads %d, battles %s: %d decisions with a line for the search's favourite and another" % (threads, which, n))
print("  the lines would change the favourite: margin 0.08 alone on %d (%.0f%%); margin 0.10 and no world worse on %d (%.0f%%)"
      % (fire_old, 100.0 * fire_old / max(1, n), fire_new, 100.0 * fire_new / max(1, n)))
print("  the same action's line in the two worlds differs by %.3f on average (median %.3f, largest %.2f)"
      % (statistics.mean(spread), statistics.median(spread), max(spread)))
print("  the favourite's line minus the search's own result for it: %.3f on average" % statistics.mean(fav_minus_root))
print("  the lines took %.0f ms on average (%.0f at the slowest); asked to answered %.0f ms on average, %.0f at the slowest"
      % (statistics.mean(line_ms), max(line_ms), statistics.mean(wall), max(wall)))
