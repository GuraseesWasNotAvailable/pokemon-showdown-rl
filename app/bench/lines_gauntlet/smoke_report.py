"""On the box: the model smoke's decisions, read from its battle files: per arm the decision times,
what the model did, and what mix played instead and why."""
import collections
import glob
import json
import statistics
import sys

folder = sys.argv[1] if len(sys.argv) > 1 else "model"
ms = collections.defaultdict(list)
why = collections.Counter()
asked = collections.Counter()
answered = collections.Counter()
lines_read = collections.Counter()
model_ms = collections.defaultdict(list)
cost = collections.Counter()
events = collections.Counter()
for path in glob.glob(folder + "/battles/*.jsonl"):
    for line in open(path, encoding="utf-8"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        note = d.get("note") if isinstance(d, dict) else None
        if not isinstance(note, dict):
            note = d.get("data") if isinstance(d.get("data"), dict) else None
        dec = (note or {}).get("decision") if isinstance(note, dict) else None
        if not isinstance(dec, dict):
            continue
        arm = dec.get("arm")
        ms[arm].append(float(dec.get("ms") or 0.0))
        lines_read[arm] += int(bool((dec.get("lines") or {}).get("values")))
        lu = dec.get("luna")
        if isinstance(lu, dict):
            asked[arm] += int(bool(lu.get("asked")))
            answered[arm] += int(lu.get("choice") is not None)
            cost[arm] += float(lu.get("cost_usd") or 0.0)
            if lu.get("latency_ms") is not None and lu.get("choice") is not None:
                model_ms[arm].append(float(lu["latency_ms"]))
            if lu.get("fell_back"):
                why[(arm, lu.get("why"))] += 1
for arm, v in sorted(ms.items()):
    v2 = sorted(v)
    print("%-11s %3d decisions: %5.0f ms on average, 95th %5.0f, slowest %5.0f; over 9.5 s: %d; lines read on %d; the model asked %d, answered %d (%.0f ms on average), $%.3f"
          % (arm, len(v), statistics.mean(v), v2[int(0.95 * (len(v2) - 1))], v2[-1], sum(1 for x in v if x > 9500), lines_read[arm],
             asked[arm], answered[arm], statistics.mean(model_ms[arm]) if model_ms[arm] else 0.0, cost[arm]))
print("mix played instead:", ", ".join("%s %s %d" % (a, w, n) for (a, w), n in sorted(why.items(), key=lambda kv: str(kv[0]))) or "never")
rows = [json.loads(line) for line in open(folder + "/results.jsonl", encoding="utf-8") if line.strip()]
print("games %d, wins %d, late decisions %d, lines_read %d, luna_cost_usd %.3f"
      % (len(rows), sum(1 for r in rows if r.get("won")), sum(int(r.get("late") or 0) for r in rows),
         sum(int(r.get("lines_read") or 0) for r in rows), sum(float(r.get("luna_cost_usd") or 0.0) for r in rows)))
