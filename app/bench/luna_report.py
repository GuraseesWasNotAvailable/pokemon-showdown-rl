"""
Luna's own summary over the games it has played, read from runs/ladder/results.jsonl.

    python bench/luna_report.py
    python bench/luna_report.py --results runs/ladder/results.jsonl --experiment luna-vs-mix-024381 --json

Every results line of the arm "luna" (ladder.py --player luna, or luna among --arms): the games
and their record, and Luna's counts added up (luna_player.result_fields): the decisions that had
a choice, the questions sent, the legal answers, how many were played, how many the blunder guard
overruled, how often Luna's pick was mix's own, the decisions mix played instead and why, the
latency, the input tokens and what they cost. Then the same per reason for mix playing, and the
games in which Luna never answered (those were today's bot under Luna's name).

The win rate here is Luna's games alone and says little by itself: bench/ab_report.py compares
it with mix's games of the same experiment. Pure JSON and the standard library plus luna_player:
nothing is started, nothing is written.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "runs", "ladder", "results.jsonl")
SUMS = ("luna_decisions", "luna_asked", "luna_answered", "luna_played", "luna_overruled", "luna_agreed_with_mix", "luna_changed",
        "luna_tokens", "luna_output_tokens")
ARMS = ("luna", "sol", "luna-lines", "sol-lines", "luna-solo", "sol-solo")      # the arms a model decides (search_player.DECIDERS)


def label_of(arm: str) -> str:
    return "Sol" if "sol" in str(arm) else "Luna"


def load(path: str, experiment: Optional[str] = None, arm: str = "luna") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.isfile(path):
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if not isinstance(d, dict) or d.get("arm") != arm:
                continue
            if experiment and d.get("experiment") != experiment:
                continue
            rows.append(d)
    return rows


def summary(rows: List[Dict[str, Any]], label: str = "Luna") -> Dict[str, Any]:
    import luna_player as lp

    decided = [d for d in rows if d.get("won") in (True, False)]
    total = lp.new_counts()
    for d in rows:
        for key in SUMS:
            total[key] += int(d.get(key) or 0)
        total["luna_cost"] += float(d.get("luna_cost_usd") or 0.0)
        for why, k in (d.get("luna_fallbacks") or {}).items():
            total["luna_fallbacks"][why] = total["luna_fallbacks"].get(why, 0) + int(k or 0)
        if d.get("luna_latency_ms_mean") is not None and d.get("luna_asked"):
            total["luna_latency_n"] += int(d["luna_asked"])
            total["luna_latency_ms"] += float(d["luna_latency_ms_mean"]) * int(d["luna_asked"])
            total["luna_latency_max"] = max(total["luna_latency_max"], float(d.get("luna_latency_ms_max") or 0.0))
    out = lp.result_fields(total)
    out.update(games=len(rows), decided=len(decided), wins=sum(1 for d in decided if d.get("won")),
               silent_games=sum(1 for d in rows if int(d.get("luna_decisions") or 0) > 0 and not int(d.get("luna_answered") or 0)),
               text=lp.counts_text(total, label))
    return out


def text(s: Dict[str, Any], experiment: Optional[str], arm: str = "luna") -> str:
    name = label_of(arm)
    if not s["games"]:
        return "no games of the arm %s" % arm + (" in experiment %s" % experiment if experiment else "") + " yet"
    lines = ["%s%s: %d games%s, %d wins and %d losses%s" % (
        name, "" if arm in ("luna", "sol") else " (arm %s)" % arm,
        s["games"], " of experiment %s" % experiment if experiment else "", s["wins"], s["decided"] - s["wins"],
        " (%.0f%%)" % (100.0 * s["wins"] / s["decided"]) if s["decided"] else "")]
    lines.append("  " + s["text"])
    if s["luna_decisions"]:
        share = s["luna_played"] / float(s["luna_decisions"])
        lines.append("  %s's choice was the order on %.0f%% of the decisions that had a choice; the rest were today's bot's (mix)."
                     % (name, 100.0 * share))
    if s["silent_games"]:
        lines.append("  In %d games %s never answered: those were today's bot under %s's name." % (s["silent_games"], name, name))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Luna's own summary over the games it has played.")
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--experiment", default=None, help="only the games of this experiment tag")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--arm", default="luna", choices=ARMS, help="the model's arm to report (default: luna)")
    args = ap.parse_args(argv)
    s = summary(load(args.results, args.experiment, args.arm), label_of(args.arm))
    print(json.dumps(s, indent=1) if args.json else text(s, args.experiment, args.arm))
    return 0


if __name__ == "__main__":
    sys.exit(main())
