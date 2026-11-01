"""
The tables of bench/notes/counter_replay.md from counter_replay.py's and counter_rerun.py's --json files.

    python bench/counter_replay_report.py --set "49 games of 2026-10-08" out/replay_1008.json out/rerun_1008.json out/rerun_live_1008.json \
        --set "round 1's 100 games" out/replay_r1.json out/rerun_r1.json out/rerun_live_r1.json

Per set-up sweep loss: who swept, whether a rule changed a decision, the first change (the only one
whose position is the game's own: after it the game would have gone otherwise), at whom it is aimed
and whether it answers the sweep, and the search's score gap (rule minus mix, points of win chance,
the two repeats) at 500 ms and at the live search's effort. Then the changes counted per kind of
game, and the verdicts per kind of change, with the mean gap and a 95% interval from a bootstrap
over games (the changes of one game are not independent).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from typing import Any, Dict, List, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "bench")):
    if p not in sys.path:
        sys.path.insert(0, p)

from counter_replay import same_mon, wilson  # noqa: E402


def short(tag: str) -> str:
    numbers = re.findall(r"-(\d+)", str(tag or ""))
    return numbers[0][-4:] if numbers else str(tag)[-4:]


def group(category: str) -> str:
    if category == "loss: set-up sweep":
        return "set-up sweep losses"
    if str(category).startswith("win"):
        return "wins"
    if category == "loss: other (no set-up)":
        return "losses without a set-up"
    return "other losses with a set-up"


def change_kind(c: Dict[str, Any]) -> str:
    if c["kind"] == "other side":
        return "other defence"
    decided = ((c.get("record") or {}).get("counter") or {}).get("decided")
    return "counter move" if decided == "played the counter" else "switch to a holder"


def answer_text(c: Dict[str, Any]) -> str:
    """Whether a change answers the sweep, in words: what the action does to the Pokemon that swept."""
    j = c["judge"]
    if not j.get("set_up_here"):
        return "no: that Pokemon never set up"
    if not j.get("sweeper"):
        return "no: another set-up Pokemon"
    kind = change_kind(c)
    if kind == "other defence":
        return "little: the new attack does under 20%" if j.get("answers") == "little" else "partly: hits the unboosted defence"
    what = j.get("counter_kind") or ""
    if kind == "counter move":
        return {"phaze": "yes: drags it out, boosts gone", "haze": "yes: its boosts gone", "encore": "yes: locks it into its set-up move",
                "sleep": "yes: puts it to sleep", "taunt": "yes: no more set-up", "perish": "partly: it must leave in 3 turns"}.get(
                    what, "yes")
    when = "before its set-up" if j.get("boosts") == "no boost" else "while it is boosted"
    return "partly: brings in a %s holder %s; the answer needs a second turn" % (what, when)


def gaps_text(row: Dict[str, Any]) -> str:
    if row is None:
        return "-"
    return "%s (%s)" % ("/".join("?" if g is None else "%+.0f" % g for g in row["gaps"]), row["verdict"])


def bootstrap_mean(values: Dict[str, List[float]], n: int = 4000, seed: int = 7) -> Tuple[float, float, float]:
    """The mean over all values, and a 95% interval from resampling the games (keys)."""
    flat = [v for vs in values.values() for v in vs]
    if not flat:
        return float("nan"), float("nan"), float("nan")
    keys = list(values)
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        pick = [v for k in (rng.choice(keys) for _ in keys) for v in values[k]]
        means.append(sum(pick) / len(pick))
    means.sort()
    return sum(flat) / len(flat), means[int(0.025 * n)], means[int(0.975 * n) - 1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", nargs=4, action="append", metavar=("NAME", "REPLAY", "RERUN", "RERUN_LIVE"), required=True)
    a = ap.parse_args(argv)
    out: List[str] = []
    every: List[Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []   # (set, game, change, r500, rlive)
    games_all: List[Tuple[str, Dict[str, Any]]] = []
    for name, rp, r5, rl in a.set:
        games = json.load(open(rp, encoding="utf-8"))["games"]
        r500 = {(r["battle"], r["rqid"]): r for r in json.load(open(r5, encoding="utf-8"))["rows"]}
        rlive = {(r["battle"], r["rqid"]): r for r in json.load(open(rl, encoding="utf-8"))["rows"]}
        for g in games:
            games_all.append((name, g))
            for c in g["changes"]:
                every.append((name, g, c, r500.get((g["battle"], c["rqid"])), rlive.get((g["battle"], c["rqid"]))))
        out.append("### %s: the set-up sweep losses" % name)
        out.append("")
        out.append("| game | swept by (set-up turn, KOs) | rule changed (of set-up positions looked at) | first change: turn, rule instead of mix | aimed at | answers the sweep? | search gap 500 ms | search gap, live effort | later changes (gap 500 ms) |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for g in games:
            if g["category"] != "loss: set-up sweep":
                continue
            sweep = [e for e in g.get("events") or [] if any(same_mon(e["mon"], s) for s in g.get("sweepers") or [])]
            who = "; ".join("%s (T%d, %d)" % (e["mon"], e["turn"], e.get("kos") or 0) for e in sweep)
            ch = g["changes"]
            looked = sum((g.get("decided") or {}).values())
            if not ch:
                why = ", ".join("%s %d" % (k.replace("kept: ", ""), v) for k, v in sorted((g.get("decided") or {}).items(), key=lambda kv: -kv[1]))
                out.append("| %s | %s | 0 of %d | none (%s) | | | | | |" % (short(g["battle"]), who, looked, why or "never a set-up position"))
                continue
            c = ch[0]
            j = c["judge"]
            aim = "%s (%s)" % (j["at"], j["boosts"])
            if j.get("sweeper"):
                aim += ", the sweeper"
            elif j.get("set_up_here"):
                aim += ", set up T%d, not the sweeper" % j["setup_turn"]
            else:
                aim += ", never set up"
            later = ", ".join("T%d %s %s" % (x["turn"], x["pick"], gaps_text(r500.get((g["battle"], x["rqid"]))).split(" (")[0])
                              for x in ch[1:])
            out.append("| %s | %s | %d of %d | T%d %s instead of %s | %s | %s | %s | %s | %s |" % (
                short(g["battle"]), who, len(ch), looked, c["turn"], c["pick"], c["mix"], aim, answer_text(c),
                gaps_text(r500.get((g["battle"], c["rqid"]))), gaps_text(rlive.get((g["battle"], c["rqid"]))), later or "-"))
        out.append("")
    # Changes per kind of game, both sets.
    out.append("### Changed decisions per kind of game (95% Wilson intervals)")
    out.append("")
    out.append("| set | kind of game | games | decisions | changed | per 100 decisions | games with a change |")
    out.append("|---|---|---|---|---|---|---|")
    names = [s[0] for s in a.set] + ["both"]
    for name in names:
        for grp in ("set-up sweep losses", "other losses with a set-up", "losses without a set-up", "wins"):
            rows = [g for s, g in games_all if (name == "both" or s == name) and group(g["category"]) == grp]
            n = sum(g["compared"] for g in rows)
            k = sum(len(g["changes"]) for g in rows)
            kg = sum(1 for g in rows if g["changes"])
            lo, hi = wilson(k, n)
            glo, ghi = wilson(kg, len(rows))
            out.append("| %s | %s | %d | %d | %d | %.1f (%.1f to %.1f) | %d (%d%%, %d to %d) |" % (
                name, grp, len(rows), n, k, 100.0 * k / max(1, n), 100 * lo, 100 * hi, kg, round(100.0 * kg / max(1, len(rows))),
                round(100 * glo), round(100 * ghi)))
    out.append("")
    # Verdicts per kind of change and kind of game.
    for label, idx in (("500 ms (round 1's rerun)", 3), ("the live search's effort", 4)):
        out.append("### The search asked again at %s: the rule's action against mix's (both sets)" % label)
        out.append("")
        out.append("| kind of game | kind of change | changes | clearly worse | worse | about even | better | mean gap, points (95% interval over games) | first changes only: mean gap |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for grp in ("set-up sweep losses", "wins", "other losses with a set-up"):
            for kind in ("counter move", "switch to a holder", "other defence", "all"):
                rows = [e for e in every if group(e[1]["category"]) == grp and (kind == "all" or change_kind(e[2]) == kind) and e[idx] is not None]
                if not rows:
                    continue
                verdicts = {}
                per_game: Dict[str, List[float]] = {}
                first: Dict[str, List[float]] = {}
                for e in rows:
                    r = e[idx]
                    verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
                    known = [x for x in r["gaps"] if x is not None]
                    if known:
                        per_game.setdefault(e[0] + e[1]["battle"], []).append(sum(known) / len(known))
                        if e[2].get("first"):
                            first.setdefault(e[0] + e[1]["battle"], []).append(sum(known) / len(known))
                m, lo, hi = bootstrap_mean(per_game)
                fm, flo, fhi = bootstrap_mean(first)
                out.append("| %s | %s | %d | %d | %d | %d | %d | %+.1f (%+.1f to %+.1f) | %s |" % (
                    grp, kind, len(rows), verdicts.get("clearly worse", 0), verdicts.get("worse", 0), verdicts.get("about even", 0),
                    verdicts.get("better", 0), m, lo, hi, ("%+.1f over %d (%+.1f to %+.1f)" % (fm, len(first), flo, fhi)) if first else "-"))
        out.append("")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
