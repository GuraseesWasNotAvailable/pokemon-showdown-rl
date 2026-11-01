"""
What the arm "luna-arb" would have played, replayed from logged games of the arm "luna".

    python bench/luna_arb_replay.py [--dir runs/ladder]

A "luna" game logs, for every decision Luna answered, its probability for each legal action
(the Decisions API's own "probabilities"), the network's pick ("net"), the search's favourite
("search") and mix's pick (luna "mix"). The arm "luna-arb" asks Luna only where the network's
pick and the search's favourite differ, and only between the advisers' picks (those two and
mix's). This script reads, from the probabilities Luna gave when it was offered every legal
action, what it would have preferred among those picks, and how sure: its probability for that
pick over its probability for all of the picks.

Per bar on that share: how many of mix's moves would have changed, in won and in lost games.
It is a count from games already played with another question, not a result: Luna asked about
two or three choices may answer differently, and a change in a lost game is not a bad change.
It sets where the bar starts (LunaSettings.min_probability), nothing more. Standard library only.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BARS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)


def decisions(folder: str, arm: str = "luna") -> List[Dict[str, Any]]:
    """Every answered decision of the arm's decided games: {"won", "net", "search", "mix", "choice", "p": {action: probability}}."""
    won: Dict[str, bool] = {}
    path = os.path.join(folder, "results.jsonl")
    if os.path.isfile(path):
        for line in open(path, encoding="utf-8"):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict) and d.get("arm") == arm and d.get("won") in (True, False):
                won[str(d.get("battle"))] = bool(d["won"])
    out: List[Dict[str, Any]] = []
    for name in sorted(glob.glob(os.path.join(folder, "battles", "*.jsonl"))):
        tag = os.path.basename(name)[:-6]
        if tag not in won:
            continue
        for line in open(name, encoding="utf-8"):
            if '"decision"' not in line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            note = d.get("note") if isinstance(d.get("note"), dict) else d.get("data") if isinstance(d.get("data"), dict) else {}
            dec = note.get("decision") if isinstance(note, dict) else None
            lu = dec.get("luna") if isinstance(dec, dict) else None
            if not isinstance(lu, dict) or lu.get("choice") is None or not isinstance(lu.get("probabilities"), dict):
                continue
            out.append({"won": won[tag], "net": dec.get("net"), "search": dec.get("search"), "mix": int(lu["mix"]), "choice": int(lu["choice"]),
                        "p": {int(a): float(p) for a, p in lu["probabilities"].items()}})
    return out


def arbiter(d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """What luna-arb reads out of one logged decision: None where the advisers agree, else {"pick", "share", "picks"}."""
    if d["search"] is None or d["net"] is None or int(d["net"]) == int(d["search"]):
        return None
    picks: List[int] = []
    for a in (d["mix"], int(d["net"]), int(d["search"])):
        if a not in picks:
            picks.append(a)
    total = sum(d["p"].get(a, 0.0) for a in picks)
    if total <= 0:
        return {"pick": d["mix"], "share": 0.0, "picks": picks}
    pick = max(picks, key=lambda a: (d["p"].get(a, 0.0), a == d["mix"]))
    return {"pick": pick, "share": d["p"].get(pick, 0.0) / total, "picks": picks}


def replay(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    answers = [sum(1 for d in rows if d["won"]), sum(1 for d in rows if not d["won"])]
    asked = [0, 0]
    outside = [0, 0]
    by_bar = {bar: [0, 0] for bar in BARS}
    plain = [0, 0]
    for d in rows:
        w = 0 if d["won"] else 1
        plain[w] += int(d["choice"] != d["mix"])
        got = arbiter(d)
        if got is None:
            continue
        asked[w] += 1
        outside[w] += int(d["choice"] not in got["picks"])
        for bar in BARS:
            by_bar[bar][w] += int(got["pick"] != d["mix"] and got["share"] >= bar)
    return {"answers": answers, "asked": asked, "plain_changed": plain, "chose_outside_the_picks": outside,
            "changed_by_bar": {str(bar): by_bar[bar] for bar in BARS}}


def text(r: Dict[str, Any]) -> str:
    a, b = r["answers"]
    if not a + b:
        return "no answered decisions of the arm in decided games"
    lines = ["%d answered decisions (%d in won games, %d in lost)" % (a + b, a, b),
             "the arm luna changed mix's move on %d (%d in won games, %d in lost): %.1f and %.1f per 100 answers"
             % (sum(r["plain_changed"]), r["plain_changed"][0], r["plain_changed"][1],
                100.0 * r["plain_changed"][0] / max(1, a), 100.0 * r["plain_changed"][1] / max(1, b)),
             "the network's pick and the search's favourite differ on %d (%.0f%%): there luna-arb asks, between their picks"
             % (sum(r["asked"]), 100.0 * sum(r["asked"]) / (a + b)),
             "  (on %d of those the arm luna chose an action that is neither adviser's pick nor mix's: not on offer in luna-arb)"
             % sum(r["chose_outside_the_picks"]),
             "luna-arb would have changed mix's move, by the bar on Luna's share for its pick among the picks:"]
    for bar, (x, y) in r["changed_by_bar"].items():
        lines.append("  bar %.2f: %3d moves (%3d in won games, %3d in lost): %.1f and %.1f per 100 answers"
                     % (float(bar), x + y, x, y, 100.0 * x / max(1, a), 100.0 * y / max(1, b)))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="What luna-arb would have played in logged games of the arm luna.")
    ap.add_argument("--dir", default=os.path.join(HERE, "runs", "ladder"), help="the folder of results.jsonl and battles/")
    ap.add_argument("--arm", default="luna")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    r = replay(decisions(args.dir, args.arm))
    print(json.dumps(r, indent=1) if args.json else text(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
