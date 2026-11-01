"""
What a run of ladder.py with a search player did, from its results file and its battle files.

    python bench/search_run_report.py runs/search_player/main/ours/results.jsonl [--deadline 2.5] [--json out.json]

Per arm: games and record (also by the side played), decisions, how many the search answered,
why not for the others, who was overruled, Terastallization, iterations and time per decision,
and the decisions that took longer than the deadline with what the time went on. The battle
files (battles/ next to the results file, written by ladder.py) hold one record per decision;
without them only the per-game sums of the results file are reported.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from typing import Any, Dict, List


def pct(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def load_results(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def load_decisions(folder: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for name in sorted(glob.glob(os.path.join(folder, "*.jsonl"))):
        with open(name, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("dir") == "note" and isinstance(row.get("data"), dict) and "decision" in row["data"]:
                    d = dict(row["data"]["decision"])
                    d["t"] = row.get("t")
                    out.append(d)
    return out


def report(results: List[Dict[str, Any]], decisions: List[Dict[str, Any]], deadline: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {"games": len(results), "arms": {}}
    limit = deadline * 1000.0
    for arm in sorted({str(r.get("arm")) for r in results if r.get("arm")}):
        games = [r for r in results if r.get("arm") == arm]
        own = [d for d in decisions if d.get("mode") == arm]
        wins = sum(1 for r in games if r.get("won"))
        a: Dict[str, Any] = {"games": len(games), "wins": wins, "losses": len(games) - wins}
        for role in ("p1", "p2"):
            side = [r for r in games if r.get("role") == role]
            a["as_" + role] = "%d-%d" % (sum(1 for r in side if r.get("won")), sum(1 for r in side if not r.get("won")))
        a["decisions"] = sum(int(r.get("decisions") or 0) for r in games)
        a["asked"] = sum(int(r.get("asked") or 0) for r in games)
        a["searched"] = sum(int(r.get("searched") or 0) for r in games)
        a["fallbacks"] = sum(int(r.get("fallbacks") or 0) for r in games)
        reasons: Dict[str, int] = {}
        for r in games:
            for why, n in (r.get("fallback_reasons") or {}).items():
                reasons[why] = reasons.get(why, 0) + int(n)
        a["fallback_reasons"] = reasons
        a["games_without_a_fallback"] = sum(1 for r in games if not r.get("fallbacks"))
        a["disagreed"] = sum(int(r.get("disagreed") or 0) for r in games)
        a["network_overruled"] = sum(int(r.get("overruled") or 0) for r in games)
        a["search_overruled"] = sum(int(r.get("search_overruled") or 0) for r in games)
        a["late_games"] = sum(1 for r in games if r.get("late"))
        a["late_decisions"] = sum(int(r.get("late") or 0) for r in games)
        tera = [r for r in games if r.get("tera")]
        turns = sorted(int(r["tera_turn"]) for r in tera if r.get("tera_turn") is not None)
        a["tera_games"] = len(tera)
        a["tera_median_turn"] = turns[len(turns) // 2] if turns else None
        a["tera_by"] = {str(k): sum(1 for r in tera if r.get("tera_by") == k) for k in {r.get("tera_by") for r in tera}}
        a["tera_wanted_by_network_too"] = sum(1 for r in tera if r.get("tera_net"))
        a["tera_added_decisions"] = sum(int(r.get("tera_added") or 0) for r in games)
        a["tera_dropped_decisions"] = sum(int(r.get("tera_dropped") or 0) for r in games)
        a["won_with_tera"] = "%d of %d" % (sum(1 for r in tera if r.get("won")), len(tera))
        a["turns_mean"] = round(statistics.mean(r.get("turns") or 0 for r in games), 1) if games else None
        if own:
            searched = [d for d in own if d.get("search") is not None]
            its = [float(d.get("iterations") or 0) for d in searched]
            wall = [float(d.get("search_ms") or 0) for d in searched]
            ms = [float(d.get("ms") or 0) for d in own]
            net = [float(d["net_ms"]) for d in own if d.get("net_ms") is not None]
            late = [d for d in own if float(d.get("ms") or 0) > limit]
            a["per_decision"] = {
                "records": len(own),
                "iterations_mean": int(statistics.mean(its)) if its else 0,
                "iterations_median": int(statistics.median(its)) if its else 0,
                "iterations_p10": int(pct(its, 0.10)),
                "under_60000_iterations": sum(1 for x in its if x < 60000),
                "search_ms_mean": round(statistics.mean(wall), 1) if wall else None,
                "search_ms_p95": round(pct(wall, 0.95), 1),
                "search_ms_max": round(max(wall), 1) if wall else None,
                "ms_mean": round(statistics.mean(ms), 1), "ms_median": round(statistics.median(ms), 1),
                "ms_p95": round(pct(ms, 0.95), 1), "ms_p99": round(pct(ms, 0.99), 1), "ms_max": round(max(ms), 1),
                "network_ms_median": round(statistics.median(net), 1) if net else None,
                "network_ms_p99": round(pct(net, 0.99), 1) if net else None,
                "network_ms_max": round(max(net), 1) if net else None,
                "over_deadline": len(late),
                "over_deadline_by_reason": {str(k): sum(1 for d in late if (d.get("reason") or "answered") == k)
                                            for k in {(d.get("reason") or "answered") for d in late}},
                "over_deadline_network_was_slow": sum(1 for d in late if float(d.get("net_ms") or 0) > 1000.0),
                "over_deadline_loop_was_late": sum(1 for d in late if float(d.get("wait_over_ms") or 0) > 100.0),
                "over_deadline_ms_median": round(statistics.median(float(d["ms"]) for d in late), 1) if late else None,
            }
            both = [d for d in searched if "disagree" in d]
            a["per_decision"]["disagree_share"] = round(sum(1 for d in both if d["disagree"]) / len(both), 3) if both else None
            a["per_decision"]["network_overruled_share"] = round(sum(1 for d in both if d["net_overruled"]) / len(both), 3) if both else None
            a["per_decision"]["search_overruled_share"] = round(sum(1 for d in both if d["search_overruled"]) / len(both), 3) if both else None
            a["per_decision"]["network_share_of_its_move_median"] = round(statistics.median(float(d.get("p_net") or 0) for d in both), 3) if both else None
            a["per_decision"]["search_share_of_networks_move_median"] = round(statistics.median(float(d.get("share_net") or 0) for d in both), 3) if both else None
            a["per_decision"]["search_share_of_its_favourite_median"] = round(statistics.median(float(d.get("share_search") or 0) for d in both), 3) if both else None
        out["arms"][arm] = a
    if decisions:
        # Where the slow decisions fall in time: slow decisions come in bursts when the machine is busy.
        t0 = min(float(d["t"]) for d in decisions if d.get("t"))
        late = sorted(float(d["t"]) - t0 for d in decisions if float(d.get("ms") or 0) > limit and d.get("t"))
        bursts: List[List[float]] = []
        for t in late:
            if bursts and t - bursts[-1][-1] < 20.0:
                bursts[-1].append(t)
            else:
                bursts.append([t])
        out["slow_bursts"] = [{"from_s": round(b[0]), "to_s": round(b[-1]), "decisions": len(b)} for b in bursts]
        out["run_seconds"] = round(max(float(d["t"]) for d in decisions if d.get("t")) - t0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("--battles", default=None, help="the folder of battle files (default: battles/ next to the results file)")
    ap.add_argument("--deadline", type=float, default=2.5)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    results = load_results(args.results)
    folder = args.battles or os.path.join(os.path.dirname(os.path.abspath(args.results)), "battles")
    decisions = load_decisions(folder) if os.path.isdir(folder) else []
    out = report(results, decisions, args.deadline)
    text = json.dumps(out, indent=1)
    print(text)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
