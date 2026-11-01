"""
What the "predict" mode did in a run of ladder.py: the predictions of the opponent's action, how
they turned out, and the turns on which the mix choice was left.

    python bench/predict_run_report.py runs/predict/main/results.jsonl [--json out.json] [--examples 5]

Reads the results file (one line per game, the pred_* counts of search_player.game_summary) and
the battle files next to it (battles/<tag>.jsonl: every decision's record with its "predict"
block, to which the player adds "outcome" after the turn: what the opponent did).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from collections import Counter
from typing import Any, Dict, List, Optional


def load(results: str) -> Dict[str, Any]:
    games = []
    with open(results, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                if d.get("arm") == "predict":
                    games.append(d)
    decisions: List[Dict[str, Any]] = []
    outcomes: Dict[Any, Dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(results)), "battles", "*.jsonl"))):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                data = row.get("data")
                if row.get("dir") != "note" or not isinstance(data, dict):
                    continue
                if "decision" in data and data["decision"].get("mode") == "predict":
                    decisions.append(data["decision"])
                elif "outcome" in data:
                    o = data["outcome"]
                    outcomes[(o.get("battle"), o.get("turn"))] = o.get("predict") or {}
    # The decision was written before the turn was played: its outcome is in the later note.
    for d in decisions:
        p = d.get("predict")
        if p and "top" in p and "outcome" not in p:
            later = outcomes.get((d.get("battle"), d.get("turn")))
            if later and "outcome" in later:
                p["outcome"] = later["outcome"]
    return {"games": games, "decisions": decisions}


def _rate(k: int, n: int) -> str:
    return "%d of %d (%.1f%%)" % (k, n, 100.0 * k / n) if n else "0 of 0"


def report(data: Dict[str, Any], examples: int = 5) -> Dict[str, Any]:
    games, decisions = data["games"], data["decisions"]
    wins = sum(1 for g in games if g.get("result") == "win" or g.get("won") is True)
    total = Counter()
    kept = Counter()
    for g in games:
        for key, value in g.items():
            if key.startswith("pred_") and isinstance(value, (int, float)):
                total[key] += value
        kept.update(g.get("pred_kept") or {})
    predicted = [d for d in decisions if d.get("predict") and "top" in d["predict"]]
    scored = [d for d in predicted if (d["predict"].get("outcome") or {}).get("name") is not None]
    confident = [d for d in scored if d["predict"].get("confident")]
    changed = [d for d in predicted if d["predict"].get("changed")]
    ms = sorted(float(d.get("ms") or 0.0) for d in decisions)
    asked = [d for d in decisions if d.get("asked")]
    its = sorted(int(d["iterations"]) for d in decisions if d.get("iterations"))
    payoff_ms = [float(d["predict"]["payoff_ms"]) for d in predicted if d["predict"].get("payoff_ms") is not None]
    known = [float(d["predict"]["payoff_known"]) for d in predicted if d["predict"].get("payoff_known") is not None]

    def reliability(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
        out = []
        for lo, hi in ((0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)):
            inside = [d for d in rows if lo <= float(d["predict"]["top"][0][1]) < hi]
            if inside:
                out.append({"from": lo, "to": min(1.0, hi), "n": len(inside),
                            "said": round(statistics.mean(float(d["predict"]["top"][0][1]) for d in inside), 4),
                            "happened": round(statistics.mean(1.0 if d["predict"]["outcome"].get(key) else 0.0 for d in inside), 4)})
        return out

    switch_reads = [d for d in confident if d["predict"].get("read") == "switch"]
    out: Dict[str, Any] = {
        "games": len(games), "wins": wins, "decisions": len(decisions), "asked": len(asked),
        "answered": sum(1 for d in decisions if d.get("search") is not None),
        "fallbacks": dict(Counter(d.get("reason") for d in decisions if d.get("asked") and d.get("search") is None)),
        "iterations_per_decision": {"mean": round(statistics.mean(its)) if its else None, "median": its[len(its) // 2] if its else None,
                                    "lowest_tenth": its[len(its) // 10] if its else None},
        "request_to_order_ms": {"median": ms[len(ms) // 2] if ms else None, "p95": ms[int(0.95 * (len(ms) - 1))] if ms else None,
                                "p99": ms[int(0.99 * (len(ms) - 1))] if ms else None, "max": ms[-1] if ms else None},
        "over_2500_ms": sum(1 for v in ms if v > 2500.0),
        "payoff_ms": {"mean": round(statistics.mean(payoff_ms), 1) if payoff_ms else None, "max": max(payoff_ms) if payoff_ms else None},
        "payoff_cells_known_mean": round(statistics.mean(known), 3) if known else None,
        "from_results": dict(total), "kept_why": dict(kept),
        "predictions": len(predicted), "scored": len(scored),
        "top1_right": sum(1 for d in scored if d["predict"]["outcome"].get("top1_right")),
        "p_true_mean": round(statistics.mean(float(d["predict"]["outcome"].get("p_true") or 0.0) for d in scored), 4) if scored else None,
        "confident_reads_scored": len(confident),
        "confident_reads_right": sum(1 for d in confident if d["predict"]["outcome"].get("read_right")),
        "switch_reads_scored": len(switch_reads),
        "switch_reads_right": sum(1 for d in switch_reads if d["predict"]["outcome"].get("read_right")),
        "opponent_switched_on_scored_turns": sum(1 for d in scored if str(d["predict"]["outcome"]["name"]).startswith("switch ")),
        "reliability_of_the_likeliest_action": reliability(scored, "top1_right"),
        "changed": len(changed),
        "changed_scored": sum(1 for d in changed if (d["predict"].get("outcome") or {}).get("name") is not None),
        "changed_read_right": sum(1 for d in changed if (d["predict"].get("outcome") or {}).get("read_right")),
        "changed_to_a_switch": sum(1 for d in changed if 4 <= int(d["played"]) <= 9),
        "changed_on_a_switch_read": sum(1 for d in changed if d["predict"].get("read") == "switch"),
        "double_switches": sum(1 for d in changed if 4 <= int(d["played"]) <= 9 and d["predict"].get("read") == "switch"),
        "games_with_a_change": len({d["battle"] for d in changed}),
    }
    out["examples"] = [{
        "battle": d["battle"], "turn": d["turn"], "position": d["predict"].get("position"),
        "expected": d["predict"].get("top"), "read": d["predict"].get("read"), "confidence": d["predict"].get("confidence"),
        "second": [d["predict"].get("second"), d["predict"].get("second_p")],
        "gain": d["predict"].get("gain"), "risk": d["predict"].get("risk"), "coverage": d["predict"].get("coverage"),
        "happened": (d["predict"].get("outcome") or {}).get("did") or (d["predict"].get("outcome") or {}).get("why_unknown"),
        "read_right": (d["predict"].get("outcome") or {}).get("read_right"),
    } for d in changed[: max(0, examples)]]
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="What the predict mode did in a ladder.py run.")
    parser.add_argument("results")
    parser.add_argument("--json", default="")
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args(argv)
    out = report(load(args.results), args.examples)
    print("games %d (won %d), decisions %d, search asked %d, answered %d, fallbacks %s" % (
        out["games"], out["wins"], out["decisions"], out["asked"], out["answered"], out["fallbacks"] or "none"))
    print("iterations per decision: %s   request to order ms: %s   over 2.5 s: %d" % (
        out["iterations_per_decision"], out["request_to_order_ms"], out["over_2500_ms"]))
    print("payoff table: %s ms, %s of the cells known on average" % (out["payoff_ms"], out["payoff_cells_known_mean"]))
    print("predictions made %d, scored %d; likeliest action right %s; mean probability given to what happened %s" % (
        out["predictions"], out["scored"], _rate(out["top1_right"], out["scored"]), out["p_true_mean"]))
    print("confident reads right %s; of them 'it switches' right %s; the opponent switched on %s of the scored turns" % (
        _rate(out["confident_reads_right"], out["confident_reads_scored"]), _rate(out["switch_reads_right"], out["switch_reads_scored"]),
        _rate(out["opponent_switched_on_scored_turns"], out["scored"])))
    print("when it gave its likeliest action this much, it was right:")
    for b in out["reliability_of_the_likeliest_action"]:
        print("   %.2f to %.2f: %4d turns, said %.1f%%, happened %.1f%%" % (b["from"], b["to"], b["n"], 100 * b["said"], 100 * b["happened"]))
    print("mix choice left on %d decisions in %d games (%d to a switch, %d on a switch read, %d both); read right on %s" % (
        out["changed"], out["games_with_a_change"], out["changed_to_a_switch"], out["changed_on_a_switch_read"], out["double_switches"],
        _rate(out["changed_read_right"], out["changed_scored"])))
    print("mix choice kept because: %s" % ", ".join("%s %d" % kv for kv in sorted(out["kept_why"].items())))
    for e in out["examples"]:
        print(json.dumps(e))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
