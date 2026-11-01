#!/usr/bin/env python
"""
The set-up proof's gauntlets: today's mix against a set-up arm (search_player.ARM_FIXES), per
opponent and pooled, and the games in which the arm's fixes played another action than the plain
veto would have.

    python bench/setup_gauntlet_report.py ~/dev/setup-prove/g/nt ~/dev/setup-prove/g/su ~/dev/setup-prove/g/fp2 --json out.json

Reads each folder's results.jsonl (one line per game: arm, won, role) and decisions.jsonl (the
decision records; "fix" = what the fix changed and what the plain veto would have played).
Wins with 95% Wilson intervals, the arm's first/second player split, the difference arm minus
mix with a 95% normal interval, pooled over the opponents with inverse-variance weights. The
plain mix arm does not compute the fixes, so "games the fix changed" exist only for the fix arm;
they are not a randomised comparison (a game with a set-up is a different game), only a count.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List


def wilson(k: int, n: int) -> List[float]:
    if n == 0:
        return [float("nan")] * 3
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [100 * p, 100 * (c - h), 100 * (c + h)]


def lines(path: str) -> List[Dict[str, Any]]:
    out = []
    if os.path.exists(path):
        for l in open(path):
            l = l.strip()
            if l:
                try:
                    out.append(json.loads(l))
                except ValueError:
                    pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--base", default="mix")
    ap.add_argument("--json")
    a = ap.parse_args()
    result: Dict[str, Any] = {"opponents": {}}
    diffs = []
    for folder in a.folders:
        name = os.path.basename(os.path.normpath(folder))
        games = [g for g in lines(os.path.join(folder, "results.jsonl")) if g.get("arm") and not g.get("tie")]
        decs = lines(os.path.join(folder, "decisions.jsonl"))
        fixed_by_game: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        long_by_game: Dict[str, int] = Counter()
        for d in decs:
            if isinstance(d.get("fix"), dict):
                fixed_by_game[d.get("battle")].append(d["fix"])
            if isinstance(d.get("long"), dict) and d["long"].get("used") and d["long"].get("short_action") not in (None, d.get("played")):
                long_by_game[d.get("battle")] += 1
        arms: Dict[str, Any] = {}
        for arm in sorted({g["arm"] for g in games}):
            gs = [g for g in games if g["arm"] == arm]
            k = sum(1 for g in gs if g.get("won"))
            o: Dict[str, Any] = {"games": len(gs), "wins": k, "won": [round(x, 1) for x in wilson(k, len(gs))]}
            for role in ("p1", "p2"):
                r = [g for g in gs if g.get("role") == role]
                o[role] = "%d/%d" % (sum(1 for g in r if g.get("won")), len(r))
            ch = [g for g in gs if fixed_by_game.get(g["battle"])]
            if arm != a.base:
                kc = sum(1 for g in ch if g.get("won"))
                rest = [g for g in gs if not fixed_by_game.get(g["battle"])]
                kr = sum(1 for g in rest if g.get("won"))
                reasons = Counter(f.get("fix") for g in ch for f in fixed_by_game[g["battle"]])
                o["fix_changed_games"] = {"games": len(ch), "wins": kc, "won": [round(x, 1) for x in wilson(kc, len(ch))],
                                          "decisions": sum(len(fixed_by_game[g["battle"]]) for g in ch), "by_fix": dict(reasons)}
                o["fix_unchanged_games"] = {"games": len(rest), "wins": kr, "won": [round(x, 1) for x in wilson(kr, len(rest))]}
                lg = [g for g in gs if long_by_game.get(g["battle"])]
                if lg:
                    o["long_changed_games"] = {"games": len(lg), "wins": sum(1 for g in lg if g.get("won"))}
            arms[arm] = o
        opp: Dict[str, Any] = {"arms": arms}
        if a.base in arms:
            b = arms[a.base]
            for arm, o in arms.items():
                if arm == a.base or not o["games"] or not b["games"]:
                    continue
                p1, n1, p0, n0 = o["wins"] / o["games"], o["games"], b["wins"] / b["games"], b["games"]
                se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0) or 1e-9
                d = p1 - p0
                opp["diff_" + arm] = {"points": round(100 * d, 1), "lo": round(100 * (d - 1.96 * se), 1),
                                      "hi": round(100 * (d + 1.96 * se), 1), "z": round(d / se, 2)}
                diffs.append((arm, d, se))
        result["opponents"][name] = opp
    pooled: Dict[str, Any] = {}
    for arm in sorted({x[0] for x in diffs}):
        ds = [(d, se) for ar, d, se in diffs if ar == arm]
        w = [1 / se ** 2 for _d, se in ds]
        d = sum(wi * di for wi, (di, _s) in zip(w, ds)) / sum(w)
        se = math.sqrt(1 / sum(w))
        mean = sum(di for di, _s in ds) / len(ds)
        se_mean = math.sqrt(sum(s ** 2 for _d, s in ds)) / len(ds)
        pooled[arm] = {"inverse_variance": {"points": round(100 * d, 1), "lo": round(100 * (d - 1.96 * se), 1),
                                            "hi": round(100 * (d + 1.96 * se), 1), "z": round(d / se, 2)},
                       "mean_of_opponents": {"points": round(100 * mean, 1), "lo": round(100 * (mean - 1.96 * se_mean), 1),
                                             "hi": round(100 * (mean + 1.96 * se_mean), 1)}}
    result["pooled_minus_" + a.base] = pooled
    for name, opp in result["opponents"].items():
        print("== %s" % name)
        for arm, o in opp["arms"].items():
            print("  %-18s %4d games  %3d-%-3d  %.1f%% (%.1f to %.1f)  p1 %s  p2 %s" % (
                arm, o["games"], o["wins"], o["games"] - o["wins"], o["won"][0], o["won"][1], o["won"][2], o["p1"], o["p2"]))
            if "fix_changed_games" in o:
                c, r = o["fix_changed_games"], o["fix_unchanged_games"]
                print("      the fix changed a move in %d games (%d decisions; %s): won %d = %.1f%%; the other %d games: won %.1f%%" % (
                    c["games"], c["decisions"], c["by_fix"], c["wins"], c["won"][0], r["games"], r["won"][0]))
        for k, v in opp.items():
            if k.startswith("diff_"):
                print("  %s minus %s: %+.1f points (%+.1f to %+.1f), z %.2f" % (k[5:], a.base, v["points"], v["lo"], v["hi"], v["z"]))
    for arm, v in pooled.items():
        print("pooled %s minus %s: %s; mean of the opponents %s" % (arm, a.base, v["inverse_variance"], v["mean_of_opponents"]))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
