"""
Arms against each other, per opponent, with the prediction of the opponent's action.

    python bench/measure_report.py DIR [DIR ...] [--base mix] [--json FILE]
    python bench/measure_report.py fp+fp3 nt+nt3 su+su3                    a run and its top-up, pooled

DIR is a folder of bench/gauntlet.py (results.jsonl and decisions.jsonl of ours, opp.jsonl of the
opponent), or several joined by "+" (the same opponent and settings, a top-up started later in a
folder of its own): their games are reported together. For every DIR:

  * each arm's record with its Wilson 95% interval, as p1 and as p2;
  * every other arm minus --base, with the Newcombe 95% interval of the difference and its z;
  * our search's iterations per decision and, against Foul Play, Foul Play's, with the ratio
    (the compute is fair when the two are within 20%);
  * the guard: decisions it fired on and changed, and why;
  * the predictions of the opponent's action: made, scored, likeliest right, in the top 3, the mean
    probability given to what it did, and calibration by bin, twice: the likeliest action's
    probability against how often it was right, and every listed action's (the three likeliest)
    probability against how often it was what the opponent did.

Reads only. Pooled over the folders at the end (the predictions only: the records are per opponent).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from bench.gauntlet import difference, load_run, quantile, read_lines, wilson  # noqa: E402

EDGES = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def bin_index(p: float) -> int:
    for i in range(len(EDGES) - 1):
        if p < EDGES[i + 1]:
            return i
    return len(EDGES) - 2


def pct(k: float, n: int) -> str:
    if not n:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{100.0 * k / n:.1f}% ({100 * lo:.1f} to {100 * hi:.1f})"


def scored_predictions(decisions: List[Dict[str, Any]]) -> Tuple[int, List[Dict[str, Any]]]:
    """(predictions made, the scored ones: their "top" and "outcome"). A repeated request of a turn has no outcome."""
    made = 0
    scored = []
    for d in decisions:
        p = d.get("predict")
        if not isinstance(p, dict) or not p.get("top"):
            continue
        made += 1
        o = p.get("outcome")
        if isinstance(o, dict) and o.get("name") is not None and "top1_right" in o:
            scored.append({"top": p["top"], "outcome": o, "turn": d.get("turn")})
    return made, scored


def calibration(scored: List[Dict[str, Any]]) -> Dict[str, Any]:
    top1: List[List[float]] = [[0, 0.0, 0] for _ in range(len(EDGES) - 1)]     # n, sum p, hits
    listed: List[List[float]] = [[0, 0.0, 0] for _ in range(len(EDGES) - 1)]
    for s in scored:
        name = s["outcome"]["name"]
        p0 = float(s["top"][0][1])
        b = top1[bin_index(p0)]
        b[0] += 1
        b[1] += p0
        b[2] += int(bool(s["outcome"].get("top1_right")))
        for entry in s["top"][:3]:
            p = float(entry[1])
            c = listed[bin_index(p)]
            c[0] += 1
            c[1] += p
            c[2] += int(entry[0] == name)

    def rows(table: List[List[float]]) -> List[Dict[str, Any]]:
        out = []
        for i, (n, sp, k) in enumerate(table):
            if not n:
                continue
            lo, hi = wilson(k, int(n))
            out.append({"bin": f"{EDGES[i]:.1f}-{EDGES[i + 1]:.1f}", "n": int(n), "predicted": round(sp / n, 3),
                        "observed": round(k / n, 3), "low": round(lo, 3), "high": round(hi, 3)})
        return out

    # Expected calibration error of the likeliest action (weighted by the bins' counts)
    n_all = sum(int(b[0]) for b in top1)
    ece = sum(abs(b[1] / b[0] - b[2] / b[0]) * b[0] for b in top1 if b[0]) / n_all if n_all else None
    return {"top1": rows(top1), "listed": rows(listed), "ece_top1": round(ece, 4) if ece is not None else None}


def prediction_report(decisions: List[Dict[str, Any]], games: List[Dict[str, Any]]) -> Dict[str, Any]:
    made, scored = scored_predictions(decisions)
    n = len(scored)
    top1 = sum(1 for s in scored if s["outcome"].get("top1_right"))
    top3 = sum(1 for s in scored if s["outcome"].get("top3_right"))
    p_true = [float(s["outcome"].get("p_true") or 0.0) for s in scored]
    return {
        "made": made, "scored": n, "top1_right": top1, "top3_right": top3,
        "made_by_games": sum(g.get("pred_made") or 0 for g in games),
        "scored_by_games": sum(g.get("pred_scored") or 0 for g in games),
        "top1_by_games": sum(g.get("pred_top1_right") or 0 for g in games),
        "p_true_mean": round(statistics.fmean(p_true), 4) if p_true else None,
        "log_loss": round(-statistics.fmean([math.log(max(p, 1e-3)) for p in p_true]), 4) if p_true else None,
        "calibration": calibration(scored),
        "_scored": scored,
    }


def guard_report(decisions: List[Dict[str, Any]], games: List[Dict[str, Any]]) -> Dict[str, Any]:
    fired = [d for d in decisions if isinstance(d.get("guard"), dict) and d["guard"].get("guard")]
    changed = [d for d in fired if d["guard"].get("changed")]
    why: Dict[str, int] = {}
    for d in changed:
        key = str(d["guard"]["guard"]).split(" ")[0]
        why[key] = why.get(key, 0) + 1
    return {"fired": len(fired), "changed": len(changed), "changed_why": why,
            "fired_by_games": sum(g.get("guard_fired") or 0 for g in games),
            "changed_by_games": sum(g.get("guard_changed") or 0 for g in games),
            "games_with_a_change": sum(1 for g in games if g.get("guard_changed")),
            "won_with_a_change": sum(1 for g in games if g.get("guard_changed") and g.get("won"))}


def load_folders(spec: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    A folder, or several joined by "+" (a run and the top-up started later in a folder of its own):
    (the run with their games, decisions and summaries together, their opponent lines together).
    """
    parts = [p for p in spec.split("+") if p]
    runs = [load_run(p) for p in parts]
    summaries = [r["summary"] for r in runs if r["summary"]]
    summary: Optional[Dict[str, Any]] = None
    if summaries:
        audit: Dict[str, Any] = {}
        lag = {"checks": 0, "max_ms": 0.0, "over_500_ms": 0, "late_ms_total": 0.0}
        bridge: Dict[str, Any] = {}
        its_total = 0.0
        for s in summaries:
            for k, v in (s.get("audit") or {}).items():
                audit[k] = audit.get(k, 0) + (v or 0)
            ll = s.get("loop_lag") or {}
            lag["checks"] += ll.get("checks") or 0
            lag["max_ms"] = max(lag["max_ms"], ll.get("max_ms") or 0.0)
            lag["over_500_ms"] += ll.get("over_500_ms") or 0
            lag["late_ms_total"] += ll.get("late_ms_total") or 0.0
            b = s.get("bridge") or {}
            for k in ("decisions", "decision_errors", "timeouts", "expired", "helper_exits", "hangs"):
                bridge[k] = bridge.get(k, 0) + (b.get(k) or 0)
            its_total += (b.get("iterations_per_decision") or 0) * (b.get("decisions") or 0)
        bridge["iterations_per_decision"] = int(its_total / bridge["decisions"]) if bridge.get("decisions") else None
        summary = {"audit": audit, "loop_lag": lag, "bridge": bridge,
                   "minutes": round(sum(s.get("minutes") or 0 for s in summaries), 2)}
    run = {"folder": spec, "games": [g for r in runs for g in r["games"]], "stalled": [g for r in runs for g in r["stalled"]],
           "decisions": [d for r in runs for d in r["decisions"]], "summary": summary}
    opp = [x for p in parts for x in read_lines(os.path.join(p, "opp.jsonl"))]
    return run, opp


def folder_report(folder: str, base: str) -> Dict[str, Any]:
    run, opp = load_folders(folder)
    games, decisions = run["games"], run["decisions"]
    arms: Dict[str, Any] = {}
    for arm in sorted({g["arm"] for g in games}):
        g = [x for x in games if x["arm"] == arm]
        d = [x for x in decisions if x.get("arm") == arm]
        wins = sum(1 for x in g if x.get("won"))
        its = [x["iterations"] for x in d if x.get("iterations")]
        ms = [x["ms"] for x in d if x.get("ms") is not None]
        arms[arm] = {
            "games": len(g), "wins": wins, "ties": sum(1 for x in g if x.get("tie")), "wilson": wilson(wins, len(g)),
            "p1": [sum(1 for x in g if x.get("role") == "p1" and x.get("won")), sum(1 for x in g if x.get("role") == "p1")],
            "p2": [sum(1 for x in g if x.get("role") == "p2" and x.get("won")), sum(1 for x in g if x.get("role") == "p2")],
            "decisions": sum(x.get("decisions") or 0 for x in g), "searched": sum(x.get("searched") or 0 for x in g),
            "fallbacks": sum(x.get("fallbacks") or 0 for x in g), "late": sum(x.get("late") or 0 for x in g),
            "turns_mean": round(statistics.fmean([x.get("turns") or 0 for x in g]), 1) if g else None,
            "iterations_median": quantile(its, 0.5), "iterations_mean": int(statistics.fmean(its)) if its else None,
            "ms_mean": round(statistics.fmean(ms), 1) if ms else None, "ms_max": max(ms) if ms else None,
            "guard": guard_report(d, g),
            "predict": prediction_report(d, g),
        }
        if arm == "predict":
            arms[arm]["predict_changed"] = {"changed": sum(x.get("pred_changed") or 0 for x in g),
                                            "changed_scored": sum(x.get("pred_changed_scored") or 0 for x in g),
                                            "changed_right": sum(x.get("pred_changed_right") or 0 for x in g)}
    diffs = {}
    if base in arms:
        b = arms[base]
        for arm, a in arms.items():
            if arm != base:
                diffs[f"{arm} minus {base}"] = difference(a["wins"], a["games"], b["wins"], b["games"])
    fp_its =[x for r in opp for x in (r.get("its") or [])]
    fp = None
    if fp_its:
        fp = {"decisions": len(fp_its), "iterations_median": quantile(fp_its, 0.5), "iterations_mean": int(statistics.fmean(fp_its)),
              "under_60000": sum(1 for x in fp_its if x < 60000),
              "no_order": sum(r.get("no_order") or 0 for r in opp)}
    s = run["summary"] or {}
    return {"folder": folder, "arms": arms, "differences": diffs, "foulplay": fp, "stalled": len(run["stalled"]),
            "audit": s.get("audit"), "minutes": s.get("minutes"), "bridge": {k: (s.get("bridge") or {}).get(k) for k in
            ("decisions", "decision_errors", "timeouts", "expired", "helper_exits", "hangs", "iterations_per_decision")},
            "loop_lag": s.get("loop_lag"), "opp_games": len(opp), "first": games[0].get("time") if games else None,
            "last": games[-1].get("time") if games else None}


def print_calibration(cal: Dict[str, Any], indent: str = "      ") -> None:
    print(f"{indent}likeliest action, by its probability (ECE {cal['ece_top1']}):")
    for r in cal["top1"]:
        print(f"{indent}  {r['bin']}: n {r['n']:5d}  predicted {100 * r['predicted']:5.1f}%  right {100 * r['observed']:5.1f}% "
              f"({100 * r['low']:.1f} to {100 * r['high']:.1f})")
    print(f"{indent}every listed action (top 3), by its probability:")
    for r in cal["listed"]:
        print(f"{indent}  {r['bin']}: n {r['n']:5d}  predicted {100 * r['predicted']:5.1f}%  happened {100 * r['observed']:5.1f}% "
              f"({100 * r['low']:.1f} to {100 * r['high']:.1f})")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--base", default="mix", help="the control arm the others are compared with")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    out: Dict[str, Any] = {"folders": {}}
    pooled: List[Dict[str, Any]] = []
    for folder in args.folders:
        r = folder_report(folder, args.base)
        name = "+".join(os.path.basename(os.path.normpath(p)) for p in folder.split("+") if p)
        out["folders"][name] = r
        print(f"\n== {name} ({r['first']} to {r['last']}, {r['minutes']} min; opponent games {r['opp_games']}, stalled {r['stalled']})")
        print(f"   audit {r['audit']}; helpers {r['bridge']}; loop lag {r['loop_lag']}")
        if r["foulplay"]:
            fp = r["foulplay"]
            print(f"   Foul Play: {fp['decisions']} decisions, iterations median {fp['iterations_median']} mean {fp['iterations_mean']}, "
                  f"under 60k {fp['under_60000']}, no order {fp['no_order']}")
        for arm, a in r["arms"].items():
            print(f"   {arm:<10} {a['games']:4d} games {a['wins']}-{a['games'] - a['wins']}  {pct(a['wins'], a['games'])}  "
                  f"p1 {a['p1'][0]}/{a['p1'][1]} p2 {a['p2'][0]}/{a['p2'][1]}  turns {a['turns_mean']}")
            print(f"       decisions {a['decisions']}, searched {a['searched']}, fallbacks {a['fallbacks']}, late {a['late']}, "
                  f"ms mean {a['ms_mean']} max {a['ms_max']}, iterations median {a['iterations_median']} mean {a['iterations_mean']}")
            if r["foulplay"] and a["iterations_median"]:
                fp = r["foulplay"]
                print(f"       Foul Play / ours: median {fp['iterations_median'] / a['iterations_median']:.2f}, "
                      f"mean {fp['iterations_mean'] / max(1, a['iterations_mean']):.2f}")
            g = a["guard"]
            if g["fired"] or g["fired_by_games"]:
                print(f"       guard fired {g['fired']} ({g['fired_by_games']} by the game lines), changed {g['changed']} "
                      f"({g['changed_by_games']}), why {g['changed_why']}; games with a change {g['games_with_a_change']}, won {g['won_with_a_change']}")
            if a.get("predict_changed"):
                print(f"       predict left the mix choice {a['predict_changed']}")
            p = a["predict"]
            if p["scored"]:
                print(f"       predictions made {p['made']} ({p['made_by_games']} by the game lines), scored {p['scored']} ({p['scored_by_games']}): "
                      f"likeliest right {pct(p['top1_right'], p['scored'])}, top 3 {pct(p['top3_right'], p['scored'])}, "
                      f"mean p(what it did) {p['p_true_mean']}, log loss {p['log_loss']}")
                print_calibration(p["calibration"])
                pooled.extend(p["_scored"])
                p["_here"] = p["_scored"]
            p.pop("_scored", None)
        here = [s for a in r["arms"].values() for s in a["predict"].pop("_here", [])]
        if here:
            n = len(here)
            k1 = sum(1 for s in here if s["outcome"].get("top1_right"))
            k3 = sum(1 for s in here if s["outcome"].get("top3_right"))
            cal = calibration(here)
            p_true = [float(s["outcome"].get("p_true") or 0.0) for s in here]
            r["predictions"] = {"scored": n, "top1_right": k1, "top3_right": k3, "ece_top1": cal["ece_top1"],
                                "p_true_mean": round(statistics.fmean(p_true), 4), "calibration": cal}
            print(f"   predictions, the arms together: scored {n}, likeliest right {pct(k1, n)}, top 3 {pct(k3, n)}, "
                  f"mean p(what it did) {statistics.fmean(p_true):.3f}, ECE of the likeliest {cal['ece_top1']}")
            print_calibration(cal)
        for key, d in r["differences"].items():
            if d:
                print(f"   {key}: {100 * d['difference']:+.1f} points (95% {100 * d['low']:+.1f} to {100 * d['high']:+.1f}), z {d['z']:.2f}; "
                      f"games per arm for 80% power at this gap: {d['games_per_arm_for_80_power']}")
    # Every other arm minus the base, averaged over the opponents with equal weights (each DIR is one
    # opponent): the mean of the per-opponent differences, its standard error from theirs (Wald,
    # with half a game added on each side so that an arm near 100% still has a variance).
    across: Dict[str, Any] = {}
    for arm in sorted({a for r in out["folders"].values() for a in r["arms"] if a != args.base}):
        ds, vs, names = [], [], []
        for name, r in out["folders"].items():
            a, b = r["arms"].get(arm), r["arms"].get(args.base)
            if not a or not b or not a["games"] or not b["games"]:
                continue
            pa = (a["wins"] + 0.5) / (a["games"] + 1)
            pb = (b["wins"] + 0.5) / (b["games"] + 1)
            ds.append(a["wins"] / a["games"] - b["wins"] / b["games"])
            vs.append(pa * (1 - pa) / a["games"] + pb * (1 - pb) / b["games"])
            names.append(name)
        if len(ds) >= 2:
            mean = sum(ds) / len(ds)
            se = math.sqrt(sum(vs)) / len(ds)
            across[arm] = {"opponents": names, "difference": mean, "low": mean - 1.96 * se, "high": mean + 1.96 * se,
                           "z": mean / se if se > 0 else 0.0}
            print(f"\n== {arm} minus {args.base}, mean over {len(ds)} opponents ({', '.join(names)}), equal weights: "
                  f"{100 * mean:+.1f} points (95% {100 * (mean - 1.96 * se):+.1f} to {100 * (mean + 1.96 * se):+.1f}), z {mean / se:.2f}")
    out["across_opponents"] = across
    if pooled:
        n = len(pooled)
        k1 = sum(1 for s in pooled if s["outcome"].get("top1_right"))
        k3 = sum(1 for s in pooled if s["outcome"].get("top3_right"))
        cal = calibration(pooled)
        out["pooled_predictions"] = {"scored": n, "top1_right": k1, "top3_right": k3, "calibration": cal}
        print(f"\n== predictions, every folder and arm: scored {n}, likeliest right {pct(k1, n)}, top 3 {pct(k3, n)}")
        print_calibration(cal, "   ")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
