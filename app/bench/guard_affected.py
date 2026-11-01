"""
The guard's effect on the games it touches: mix against mix-guard, only over the games in which
the guard changes the move at least once (mix-guard: the live record; mix: bench/guard_replay.py
on the same numbers, what it would have changed).

    python bench/guard_affected.py DIR[+DIR] [DIR[+DIR] ...] [--json FILE]

DIR is a folder of bench/gauntlet.py with its battles/ (one file per game, the decision notes in
it). Why this is a fair comparison: up to the first decision on which the guard changes the move,
the two arms play the same policy, so "the game reaches such a decision" happens equally often in
both and in the same positions; from there one arm plays the plain veto and the other the guard.
The games that never reach one are the same game in both arms and only add noise to the full
comparison, so the difference over the touched games is the guard's whole effect, measured on
fewer games with less noise. It holds when the flag is computed the same way in both arms: the
replay of the mix-guard games against their live records says how close the replay is.

Reads only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import opponent_model as om  # noqa: E402
from bench.gauntlet import difference, wilson  # noqa: E402
from bench.guard_replay import replay  # noqa: E402


def live_changes(path: str) -> Dict[str, Any]:
    """The arm and result of a battle file, and the live guard's changes (turns) on mix-guard decisions."""
    arm = None
    won = None
    changed: List[int] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            data = d.get("data")
            if d.get("dir") != "note" or not isinstance(data, dict):
                continue
            if "result" in data:
                won = bool(data["result"].get("won"))
            rec = data.get("decision")
            if isinstance(rec, dict):
                arm = arm or rec.get("mode")
                g = rec.get("guard")
                if isinstance(g, dict) and g.get("changed"):
                    changed.append(int(rec.get("turn") or 0))
    return {"arm": arm, "won": won, "changed": changed}


def pct(k: int, n: int) -> str:
    if not n:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k}/{n} {100.0 * k / n:.1f}% ({100 * lo:.1f} to {100 * hi:.1f})"


def folder(spec: str, model: om.OpponentModel) -> Dict[str, Any]:
    files: List[str] = []
    for p in [x for x in spec.split("+") if x]:
        files.extend(sorted(glob.glob(os.path.join(p, "battles", "*.jsonl"))))
    games = []
    for path in files:
        live = live_changes(path)
        if live["arm"] not in ("mix", "mix-guard") or live["won"] is None:
            continue
        rep = replay(path, model)
        rep_turns = [d["turn"] for d in rep["decisions"] if d["changed"]]
        games.append({"file": path, "arm": live["arm"], "won": live["won"], "live": live["changed"], "replay": rep_turns})
    out: Dict[str, Any] = {"spec": spec, "games": len(games)}
    for arm in ("mix", "mix-guard"):
        rows = [g for g in games if g["arm"] == arm]
        flag = (lambda g: bool(g["live"])) if arm == "mix-guard" else (lambda g: bool(g["replay"]))
        touched = [g for g in rows if flag(g)]
        untouched = [g for g in rows if not flag(g)]
        out[arm] = {"games": len(rows), "wins": sum(g["won"] for g in rows),
                    "touched": len(touched), "touched_wins": sum(g["won"] for g in touched),
                    "untouched": len(untouched), "untouched_wins": sum(g["won"] for g in untouched)}
    # How close the replay is to the live guard, on the mix-guard games (game level and first change)
    mg = [g for g in games if g["arm"] == "mix-guard"]
    both = sum(1 for g in mg if g["live"] and g["replay"])
    live_only = sum(1 for g in mg if g["live"] and not g["replay"])
    replay_only = sum(1 for g in mg if g["replay"] and not g["live"])
    same_first = sum(1 for g in mg if g["live"] and g["replay"] and min(g["live"]) == min(g["replay"]))
    out["agreement"] = {"both": both, "live_only": live_only, "replay_only": replay_only, "neither": len(mg) - both - live_only - replay_only,
                        "same_first_turn": same_first}
    a, b = out["mix-guard"], out["mix"]
    out["touched_difference"] = difference(a["touched_wins"], a["touched"], b["touched_wins"], b["touched"])
    out["untouched_difference"] = difference(a["untouched_wins"], a["untouched"], b["untouched_wins"], b["untouched"])
    out["all_difference"] = difference(a["wins"], a["games"], b["wins"], b["games"])
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    model = om.OpponentModel.load()
    out = {}
    for spec in args.folders:
        r = folder(spec, model)
        name = "+".join(os.path.basename(os.path.normpath(p)) for p in spec.split("+") if p)
        out[name] = r
        print(f"\n== {name}: {r['games']} games")
        for arm in ("mix", "mix-guard"):
            x = r[arm]
            how = "the live guard changed a move" if arm == "mix-guard" else "the replayed guard would change a move"
            print(f"   {arm:<10} all {pct(x['wins'], x['games'])}; touched ({how}) {pct(x['touched_wins'], x['touched'])}; "
                  f"untouched {pct(x['untouched_wins'], x['untouched'])}")
        ag = r["agreement"]
        print(f"   replay vs live on the mix-guard games: both flag {ag['both']} (same first turn {ag['same_first_turn']}), "
              f"live only {ag['live_only']}, replay only {ag['replay_only']}, neither {ag['neither']}")
        for key in ("touched_difference", "untouched_difference", "all_difference"):
            d = r[key]
            if d:
                print(f"   mix-guard minus mix, {key.split('_')[0]} games: {100 * d['difference']:+.1f} points "
                      f"(95% {100 * d['low']:+.1f} to {100 * d['high']:+.1f}), z {d['z']:.2f}")
    # The touched games' difference averaged over the opponents with equal weights (Wald, half a game
    # added on each side), and what it is worth on the whole record: times the share of games touched.
    rows = []
    for name, r in out.items():
        a, b = r["mix-guard"], r["mix"]
        if a["touched"] and b["touched"]:
            pa = (a["touched_wins"] + 0.5) / (a["touched"] + 1)
            pb = (b["touched_wins"] + 0.5) / (b["touched"] + 1)
            d = a["touched_wins"] / a["touched"] - b["touched_wins"] / b["touched"]
            v = pa * (1 - pa) / a["touched"] + pb * (1 - pb) / b["touched"]
            share = (a["touched"] + b["touched"]) / max(1, a["games"] + b["games"])
            rows.append((name, d, v, share))
            print(f"   {name}: games touched {100 * share:.0f}%, so the guard's effect on the whole record is about "
                  f"{100 * d * share:+.1f} points (95% {100 * (d - 1.96 * v ** 0.5) * share:+.1f} to {100 * (d + 1.96 * v ** 0.5) * share:+.1f})")
    if len(rows) >= 2:
        mean = sum(r[1] for r in rows) / len(rows)
        se = sum(r[2] for r in rows) ** 0.5 / len(rows)
        print(f"\n== touched games, mix-guard minus mix, mean over {len(rows)} opponents, equal weights: {100 * mean:+.1f} points "
              f"(95% {100 * (mean - 1.96 * se):+.1f} to {100 * (mean + 1.96 * se):+.1f}), z {mean / se:.2f}")
        out["_across"] = {"opponents": [r[0] for r in rows], "difference": mean, "se": se}
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
