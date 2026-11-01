"""
Prints the brief and the question of real logged decisions, as Luna would get them.

    python bench/luna_sample.py                       one decision of the logged games (a move turn with a choice of 8 or more)
    python bench/luna_sample.py --game 3 --turn 12    the decision of that game (0-9) at that turn
    python bench/luna_sample.py --all                 one line per decision: tokens, detail, hidden-information problems

The games are the ten ladder games of tests/fixtures/luna_ladder_games.jsonl.gz, replayed through
poke-env as tests/test_luna.py does (the battle as our side saw it, the numbers of the decision's
note). No key and no request: it only writes the text.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (REPO, os.path.join(REPO, "tests"), os.path.join(REPO, "bench")):
    if path not in sys.path:
        sys.path.insert(0, path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Print the brief and question of a real logged decision.")
    ap.add_argument("--game", type=int, default=None)
    ap.add_argument("--turn", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--json", action="store_true", help="the request body as JSON")
    ap.add_argument("--dump", default=None, help="write every request's texts and token estimate to this JSON file")
    args = ap.parse_args()

    import hybrid
    import numpy as np

    import luna_audit
    import luna_player as lp
    import search_player as sp
    import test_luna as tl

    hybrid.PolicyNet = tl.tsp.StubPolicy

    class Stub(sp.search_player()):
        def _forward_batch(self, tags, obs):
            return np.stack([tl.tsp.stub_logits(row) for row in obs])

    built = tl.briefs_of_the_ladder_games({"search": Stub})
    games = []
    for d, b, seen, whole in built:
        if not games or games[-1][0] is not whole:
            games.append((whole, []))
        games[-1][1].append((d, b, seen))
    if args.dump:
        rows = [{"tokens": b.tokens, "turn": d["turn"], "legal": d["legal"],
                 "parts": [b.brief, b.question["instructions"]] + [c["value"] + " " + c["description"] for c in b.question["choices"]]}
                for d, b, _s, _w in built]
        with open(args.dump, "w", encoding="utf-8") as fh:
            json.dump(rows, fh)
        print("%d requests -> %s" % (len(rows), args.dump))
        return 0
    if args.all:
        for g, (whole, rows) in enumerate(games):
            for d, b, seen in rows:
                problems = luna_audit.hidden_problems(b.brief, b.question["choices"], seen, whole)
                print("game %d turn %3d rqid %3d legal %2d tokens %4d detail %d %s" % (
                    g, d["turn"], d["rqid"], d["legal"], b.tokens, b.detail, "; ".join(problems)))
        tokens = sorted(b.tokens for _d, b, _s, _w in built)
        print("%d decisions: tokens mean %.0f, median %d, largest %d" % (len(tokens), sum(tokens) / len(tokens), tokens[len(tokens) // 2], tokens[-1]))
        return 0
    if args.game is not None:
        rows = games[args.game][1]
        d, b, _seen = next((r for r in rows if args.turn is None or r[0]["turn"] == args.turn), rows[0])
    else:
        d, b, _seen, _whole = next(x for x in built if x[0].get("shares") and x[0]["legal"] >= 8 and not x[0].get("forced") and x[0]["turn"] >= 6)
    if args.json:
        print(json.dumps(b.body, indent=1))
        return 0
    print("=== the decision: turn %d, %d legal actions; mix played action %d; about %d tokens (detail %d) ===" % (
        d["turn"], d["legal"], d["played"], b.tokens, b.detail))
    print("--- input ---")
    print(b.brief)
    print("--- question %r (type choice) ---" % b.question["name"])
    print("instructions: " + b.question["instructions"])
    for c in b.question["choices"]:
        print("  [%s] %s" % (c["value"], c["description"]))
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)          # poke-env's loop thread would keep the process open
