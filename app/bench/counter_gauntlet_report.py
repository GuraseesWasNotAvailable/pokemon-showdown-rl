#!/usr/bin/env python
"""
Round 3's no-cost check against bots: today's mix against mix-counter (search_player.ARM_FIXES: the
two counter rules, MixSettings.setup_counter and setup_other_side), per opponent and pooled, with
how often the rules fired, the errors, the fallbacks, the late decisions and the decision time of
each arm.

    python bench/counter_gauntlet_report.py ~/dev/counter-gauntlet/g/su ~/dev/counter-gauntlet/g/fp ~/dev/counter-gauntlet/g/nt \
        --json counter_gauntlet.json

Reads each phase folder of bench/gauntlet.py: results.jsonl (one line per game: arm, won, role, the
game's COUNTER_COUNTS for the arm with the rules, fallbacks, late, decision_ms_max), decisions.jsonl
(every decision: its ms; "counter" = the rules' record, search_player._counter_note), summary.json
(our run's totals: counter_errors, exceptions, audit, loop lag, helpers) and the opponent's opp.jsonl
(Foul Play's iterations per decision).

Wins with 95% Wilson intervals; mix-counter minus mix with a 95% Newcombe interval per opponent;
pooled over the opponents with inverse-variance weights (normal standard errors), as round 1's
bench/setup_gauntlet_report.py. Games in which the rules changed a move are counted, not compared:
a game with a set-up position is a different game.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from bench.gauntlet import difference, quantile, read_lines, wilson  # noqa: E402

BASE, ARM = "mix", "mix-counter"
COUNTS = ("counter_looked", "counter_played", "counter_switched", "counter_ko_kept", "counter_had", "other_side_looked", "other_side")


def share(k: int, n: int) -> Dict[str, Any]:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "pct": round(100.0 * k / n, 1) if n else None, "lo": round(100 * lo, 1), "hi": round(100 * hi, 1)}


def ms_stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": round(statistics.fmean(values), 1), "sd": round(statistics.pstdev(values), 1),
            "median": quantile(values, 0.5), "p95": quantile(values, 0.95), "p99": quantile(values, 0.99), "max": max(values)}


def mean_diff(a: List[float], b: List[float]) -> Dict[str, Any]:
    """Mean of a minus mean of b with a 95% normal (Welch) interval."""
    if len(a) < 2 or len(b) < 2:
        return {}
    d = statistics.fmean(a) - statistics.fmean(b)
    se = math.sqrt(statistics.variance(a) / len(a) + statistics.variance(b) / len(b))
    return {"ms": round(d, 1), "lo": round(d - 1.96 * se, 1), "hi": round(d + 1.96 * se, 1)}


def arm_block(games: List[Dict[str, Any]], decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(games)
    wins = sum(1 for g in games if g.get("won"))
    out: Dict[str, Any] = {"games": n, "wins": wins, "losses": n - wins, "won": share(wins, n)}
    for role in ("p1", "p2"):
        rows = [g for g in games if g.get("role") == role]
        out[role] = {"games": len(rows), "wins": sum(1 for g in rows if g.get("won"))}
    reasons: Counter = Counter()
    for g in games:
        reasons.update(g.get("fallback_reasons") or {})
    out["decisions"] = sum(g.get("decisions") or 0 for g in games)
    out["asked"] = sum(g.get("asked") or 0 for g in games)
    out["answered"] = sum(g.get("searched") or 0 for g in games)
    out["fallbacks"] = sum(g.get("fallbacks") or 0 for g in games)            # failures: "single" and "revive" are not counted
    out["fallback_reasons"] = dict(reasons)
    out["late"] = sum(g.get("late") or 0 for g in games)
    out["decision_ms_max"] = max((g.get("decision_ms_max") or 0.0 for g in games), default=None)
    out["turns_mean"] = round(statistics.fmean([g.get("turns") or 0 for g in games]), 1) if games else None
    out["report_errors"] = sum(1 for g in games if g.get("report_error"))
    ms = [float(d["ms"]) for d in decisions if d.get("ms") is not None]
    asked = [float(d["ms"]) for d in decisions if d.get("asked") and d.get("ms") is not None]
    out["ms_all"] = ms_stats(ms)
    out["ms_asked"] = ms_stats(asked)
    its = [d["iterations"] for d in decisions if d.get("iterations")]
    out["iterations"] = {"n": len(its), "median": quantile(its, 0.5), "mean": int(statistics.fmean(its)) if its else None}
    if any(name in g for g in games for name in COUNTS):
        totals = {name: sum(int(g.get(name) or 0) for g in games) for name in COUNTS}
        out["rules"] = {"totals": totals, "per_game": {k: round(v / n, 2) for k, v in totals.items()} if n else {}}
        changed = [g for g in games if (g.get("counter_played") or 0) + (g.get("counter_switched") or 0) + (g.get("other_side") or 0)]
        looked = [g for g in games if g.get("counter_looked") or g.get("other_side_looked")]
        out["rules"]["games_looked"] = len(looked)
        out["rules"]["games_changed"] = len(changed)
        out["rules"]["games_changed_won"] = sum(1 for g in changed if g.get("won"))
        out["rules"]["games_unchanged"] = n - len(changed)
        out["rules"]["games_unchanged_won"] = wins - out["rules"]["games_changed_won"]
        out["rules"]["changed_decisions"] = totals["counter_played"] + totals["counter_switched"] + totals["other_side"]
        out["rules"]["changed_share_of_decisions"] = round(100.0 * out["rules"]["changed_decisions"] / out["decisions"], 2) if out["decisions"] else None
        recs = [d for d in decisions if isinstance(d.get("counter"), dict)]
        decided: Counter = Counter()
        triggers: Counter = Counter()
        for d in recs:
            c = d["counter"].get("counter")
            if isinstance(c, dict):
                decided[str(c.get("decided"))] += 1
                triggers[re.sub(r"[+-]?\d+%?", "#", str(c.get("trigger")))] += 1     # "set-up read #", "set up last turn"...
        out["rules"]["decided"] = dict(decided.most_common())
        out["rules"]["triggers"] = dict(triggers.most_common())
        # The decision time where the rules looked at the position, and where they did not (both build the board).
        out["rules"]["ms_looked"] = ms_stats([float(d["ms"]) for d in recs if d.get("ms") is not None])
        out["rules"]["ms_not_looked"] = ms_stats([float(d["ms"]) for d in decisions
                                                  if not isinstance(d.get("counter"), dict) and d.get("ms") is not None])
        # Every lost game of the arm: what the rules changed in it (turn, why, mix's pick and the rule's).
        lost = {g["battle"]: g for g in games if not g.get("won")}
        out["rules"]["losses"] = [
            {"battle": tag, "turns": g.get("turns"), "role": g.get("role"),
             "changed": [{"turn": d.get("turn"), "why": d["counter"].get("why"),
                          "mix": (d["counter"].get("text") or {}).get("mix"), "pick": (d["counter"].get("text") or {}).get("pick")}
                         for d in recs if d.get("battle") == tag and d["counter"].get("changed")]}
            for tag, g in lost.items()]
        out["rules"]["examples"] = [
            {"battle": d.get("battle"), "turn": d.get("turn"), "why": d["counter"].get("why"), "text": d["counter"].get("text")}
            for d in recs if d["counter"].get("changed")][:12]
    return out


def phase(folder: str) -> Dict[str, Any]:
    games = [g for g in read_lines(os.path.join(folder, "results.jsonl")) if g.get("arm") and not g.get("stalled")]
    stalled = [g for g in read_lines(os.path.join(folder, "results.jsonl")) if g.get("stalled")]
    ties = [g for g in games if g.get("tie")]
    games = [g for g in games if not g.get("tie")]
    decisions = read_lines(os.path.join(folder, "decisions.jsonl"))
    summary: Dict[str, Any] = {}
    path = os.path.join(folder, "summary.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            summary = json.load(f)
    out: Dict[str, Any] = {"folder": folder, "arms": {}, "ties": len(ties), "stalled": len(stalled)}
    for arm in (BASE, ARM):
        out["arms"][arm] = arm_block([g for g in games if g["arm"] == arm], [d for d in decisions if d.get("arm") == arm])
    b, a = out["arms"][BASE], out["arms"][ARM]
    d = difference(a["wins"], a["games"], b["wins"], b["games"])
    if d:
        p1, n1, p0, n0 = a["wins"] / a["games"], a["games"], b["wins"] / b["games"], b["games"]
        se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
        # Agresti-Caffo (one win and one loss added to each arm): never 0, also when both arms win every game
        q1, q0 = (a["wins"] + 1) / (n1 + 2), (b["wins"] + 1) / (n0 + 2)
        se_ac = math.sqrt(q1 * (1 - q1) / (n1 + 2) + q0 * (1 - q0) / (n0 + 2))
        out["diff"] = {"points": round(100 * d["difference"], 1), "lo": round(100 * d["low"], 1), "hi": round(100 * d["high"], 1),
                       "se": se, "se_ac": se_ac, "z": round(d["difference"] / se, 2) if se > 0 else None}
    out["ms_diff_asked"] = mean_diff([float(x["ms"]) for x in decisions if x.get("arm") == ARM and x.get("asked") and x.get("ms") is not None],
                                     [float(x["ms"]) for x in decisions if x.get("arm") == BASE and x.get("asked") and x.get("ms") is not None])
    st = summary.get("search_totals") or {}
    out["errors"] = {"counter_errors": st.get("counter_errors"), "exceptions": st.get("exceptions"),
                     "predict_errors": st.get("predict_errors"), "trackers_broken": st.get("trackers_broken"),
                     "after_end": st.get("after_end"), "audit": summary.get("audit"), "inactivity": summary.get("inactivity"),
                     "forfeit": summary.get("forfeit"), "start_retries": summary.get("retries")}
    br = summary.get("bridge") or {}
    out["helpers"] = {k: br.get(k) for k in ("decisions", "decision_errors", "timeouts", "expired", "helper_exits", "hangs",
                                              "iterations_per_decision")}
    out["loop_lag"] = summary.get("loop_lag")
    out["minutes"] = summary.get("minutes")
    out["parallel"], out["helper_count"] = summary.get("parallel"), summary.get("helpers")
    out["first"] = games[0].get("time") if games else None
    out["last"] = games[-1].get("time") if games else None
    opp = read_lines(os.path.join(folder, "opp.jsonl"))
    its = [x for r in opp for x in (r.get("its") or [])]
    if its:
        ours = [x["iterations"] for x in decisions if x.get("iterations")]
        out["foulplay"] = {"decisions": len(its), "iterations_median": quantile(its, 0.5), "iterations_mean": int(statistics.fmean(its)),
                           "below_60000": sum(1 for x in its if x < 60_000),
                           "ours_median": quantile(ours, 0.5), "ours_mean": int(statistics.fmean(ours)) if ours else None,
                           "ratio_median": round(quantile(its, 0.5) / quantile(ours, 0.5), 2) if ours else None,
                           "ratio_mean": round(statistics.fmean(its) / statistics.fmean(ours), 2) if ours else None,
                           "no_order": sum(r.get("no_order") or 0 for r in opp)}
    return out


def fmt_share(s: Dict[str, Any]) -> str:
    return "%d-%d, %.1f%% (%.1f-%.1f)" % (s["k"], s["n"] - s["k"], s["pct"], s["lo"], s["hi"]) if s["n"] else "no games"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--json")
    args = ap.parse_args()
    result: Dict[str, Any] = {"phases": {}}
    for folder in args.folders:
        result["phases"][os.path.basename(os.path.normpath(folder))] = phase(folder)
    diffs = [(name, p["diff"]) for name, p in result["phases"].items() if p.get("diff")]
    if diffs:
        def pool(key: str) -> Optional[Dict[str, Any]]:
            if any(not d[key] for _n, d in diffs):
                return None
            w = [1 / d[key] ** 2 for _n, d in diffs]
            mean = sum(wi * d["points"] for wi, (_n, d) in zip(w, diffs)) / sum(w)
            se = 100 * math.sqrt(1 / sum(w))
            return {"points": round(mean, 1), "lo": round(mean - 1.96 * se, 1), "hi": round(mean + 1.96 * se, 1), "z": round(mean / se, 2),
                    "weights": {n: round(wi / sum(w), 3) for wi, (n, _d) in zip(w, diffs)}}

        plain = sum(d["points"] for _n, d in diffs) / len(diffs)
        se_plain = 100 * math.sqrt(sum(d["se_ac"] ** 2 for _n, d in diffs)) / len(diffs)
        # the weights from the Agresti-Caffo errors (the main figure); from the plain normal errors as round 1 (None if one is 0)
        result["pooled"] = {"inverse_variance": pool("se_ac"), "inverse_variance_normal_se": pool("se"),
                            "mean_of_opponents": {"points": round(plain, 1), "lo": round(plain - 1.96 * se_plain, 1),
                                                  "hi": round(plain + 1.96 * se_plain, 1)}}
    for name, p in result["phases"].items():
        b, a = p["arms"][BASE], p["arms"][ARM]
        print("== %s  (%s to %s, %s min, %s games at a time on %s helpers)" % (name, p["first"], p["last"], p["minutes"], p["parallel"], p["helper_count"]))
        for arm, o in ((BASE, b), (ARM, a)):
            print("  %-12s %s  p1 %d/%d  p2 %d/%d  turns %.1f" % (arm, fmt_share(o["won"]), o["p1"]["wins"], o["p1"]["games"],
                                                                 o["p2"]["wins"], o["p2"]["games"], o["turns_mean"] or 0))
        if p.get("diff"):
            print("  %s minus %s: %+.1f points (%+.1f, %+.1f), z %s" % (ARM, BASE, p["diff"]["points"], p["diff"]["lo"], p["diff"]["hi"], p["diff"]["z"]))
        for arm, o in ((BASE, b), (ARM, a)):
            print("  %-12s decisions %d, asked %d, answered %d, fallbacks %d %s, late %d, decision_ms_max %s, ms (asked) %s" % (
                arm, o["decisions"], o["asked"], o["answered"], o["fallbacks"], o["fallback_reasons"], o["late"], o["decision_ms_max"], o["ms_asked"]))
        print("  ms (asked) mix-counter minus mix: %s" % p["ms_diff_asked"])
        r = a.get("rules")
        if r:
            print("  rules: totals %s; per game %s" % (r["totals"], r["per_game"]))
            print("         games looked %d, changed %d (won %d), unchanged %d (won %d); changed decisions %d (%.2f%% of decisions)" % (
                r["games_looked"], r["games_changed"], r["games_changed_won"], r["games_unchanged"], r["games_unchanged_won"],
                r["changed_decisions"], r["changed_share_of_decisions"] or 0))
            print("         decided %s" % r["decided"])
            print("         triggers %s" % r["triggers"])
            print("         ms where the rules looked %s; elsewhere %s" % (r["ms_looked"], r["ms_not_looked"]))
            for e in r["losses"]:
                print("         lost %s (%s, %s turns): %s" % (e["battle"], e["role"], e["turns"], "; ".join(
                    "T%s %s -> %s (%s)" % (c["turn"], c["mix"], c["pick"], c["why"]) for c in e["changed"]) or "the rules changed nothing"))
            for e in r["examples"]:
                print("         e.g. %s T%s: %s  [%s]" % (e["battle"], e["turn"], e["why"], e["text"]))
        print("  errors %s; ties %d, stalled %d" % (p["errors"], p["ties"], p["stalled"]))
        print("  helpers %s; loop lag %s" % (p["helpers"], p["loop_lag"]))
        if p.get("foulplay"):
            print("  Foul Play %s" % p["foulplay"])
    if result.get("pooled"):
        print("pooled %s minus %s: %s (normal errors: %s); mean of the opponents %s" % (
            ARM, BASE, result["pooled"]["inverse_variance"], result["pooled"]["inverse_variance_normal_se"], result["pooled"]["mean_of_opponents"]))
    if args.json:
        for p in result["phases"].values():
            if p.get("diff"):
                p["diff"]["se"], p["diff"]["se_ac"] = round(p["diff"]["se"], 5), round(p["diff"]["se_ac"], 5)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
