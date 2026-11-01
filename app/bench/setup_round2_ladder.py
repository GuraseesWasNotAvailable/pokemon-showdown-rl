"""
A play rule replayed on recorded ladder battles: on which decisions it would have played something
else than what the plain veto played, and what (round 2 of the set-up work; bench/guard_replay.py's way).

    python bench/setup_round2_ladder.py --settings '{"setup_net_trust": 0.7}' --scan scan.json data/battles/*.jsonl

Every decision note of a battle file has the network's probabilities, the search's shares and the
legal actions; the battle's own lines give the opponent's boosts, its last action and the read
(opponent_model.py on the tracker's context and the six likeliest options the note kept), and our
moves (the last |request|). The search is NOT asked again: the rule is a rule over the numbers the
search already gave. Approximations as guard_replay.py: shares, not visits (a share > 0 counts as
seen); six options of the opponent, not all. --scan (bench/setup_ladder_scan.py's scan.json) marks
the decisions that answered an opponent's set-up in the games lost to a set-up sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "bench")):
    if p not in sys.path:
        sys.path.insert(0, p)

import opponent_model as om  # noqa: E402
import search_player as sp  # noqa: E402
import setup_moves as sm  # noqa: E402
from guard_replay import names_of, our_state, threat_of, to_id  # noqa: E402


def replay(path: str, model: om.OpponentModel, settings: sp.MixSettings) -> Dict[str, Any]:
    tracker = None
    me = None
    role = None
    req: Dict[str, Any] = {}
    team_order: List[str] = []
    out: List[Dict[str, Any]] = []
    won = None
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    for d in rows:
        data = d.get("data")
        if d.get("dir") == "note" and isinstance(data, dict):
            if "user" in data and me is None:
                me = data["user"]
            if "result" in data:
                won = bool(data["result"].get("won"))
            rec = data.get("decision")
            if not rec or rec.get("shares") is None or tracker is None or tracker.rows is None:
                continue
            names = names_of(req, team_order)
            sides = tracker.rows.sides
            opp = sides[tracker.rows.opp]
            boosts = dict(opp.boosts)
            their_types = om.dex().types(opp.active.species) if opp.active is not None else []
            our_types, our_status = our_state(req)
            view_dict = {"options": [(o[0], o[1], o[2]) for o in rec.get("opponent") or []], "sets": []}
            dist = None
            if not rec.get("forced"):
                ctx = tracker.context(rec["turn"])
                if ctx is not None:
                    dist = model.predict(ctx, view_dict).dist
                elif view_dict["options"]:
                    total = sum(o[1] for o in view_dict["options"]) or 1.0
                    dist = {o[0]: o[1] / total for o in view_dict["options"]}
            act = opp.action
            last = ("move " + str(act["name"])) if act and act.get("type") == "move" else (opp.last_action if act is None else "")
            last_setup = bool(last and last.startswith("move ") and sm.is_setup(last[5:], their_types))
            threat = threat_of(rec, names, boosts, last_setup, dist, their_types, our_types, our_status)
            shares = rec["shares"]
            view = sp.SearchView(share=list(shares), visits=[1 if s > 0 else 0 for s in shares], score=[None] * 14,
                                 best=int(rec["search"]), iterations=int(rec.get("iterations") or 0))
            explain: Dict[str, Any] = {}
            ruled = sp.mix(rec["probs"], rec["mask"], view, settings, threat=threat, explain=explain)
            plain = sp.mix(rec["probs"], rec["mask"], view, sp.MixSettings())
            out.append({"turn": rec["turn"], "forced": bool(rec.get("forced")), "source": rec.get("source"),
                        "net": names.get(rec["net"], rec["net"]), "search": names.get(rec["search"], rec["search"]),
                        "played": names.get(rec["played"], rec["played"]), "veto": names.get(plain, plain),
                        "rule": names.get(ruled, ruled), "changed": ruled != plain, "why": explain.get("setup_rule") or explain.get("guard") or "",
                        "setup": sp.setup_reason(threat, sp.MixSettings()), "p_net": round(max(rec["probs"]), 3),
                        "replayed_veto_is_played": plain == rec["played"]})
            continue
        if d.get("dir") != "in" or not isinstance(data, str):
            continue
        for line in data.split("\n"):
            parts = line.split("|")
            if len(parts) > 3 and parts[1] == "player" and to_id(parts[3]) == to_id(me):
                role = parts[2]
            if len(parts) > 2 and parts[1] == "request" and parts[2]:
                try:
                    req = json.loads(parts[2])
                except ValueError:
                    continue
                if not team_order and req.get("side"):
                    team_order = [p["ident"] for p in req["side"]["pokemon"]]
                continue
            if tracker is None:
                tracker = om.OpponentTracker(os.path.basename(path))
            tracker.feed(line, role)
    return {"file": os.path.basename(path), "won": won, "decisions": out}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--settings", required=True, help="search_player.MixSettings arguments as JSON")
    ap.add_argument("--scan", default="", help="scan.json: marks the answers to set-up moves in the games lost to a set-up sweep")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)
    settings = sp.MixSettings(**json.loads(args.settings))
    model = om.OpponentModel.load()
    games = [replay(path, model, settings) for path in sorted(args.files)]
    key: Dict[str, Dict[int, Dict[str, Any]]] = {}
    category: Dict[str, str] = {}
    if args.scan:
        for g in json.load(open(args.scan)):
            number = (re.findall(r"-(\d+)", g["battle"]) or ["?"])[0]
            category[number] = g["category"]
            if g["category"] == "loss: set-up sweep":
                for e in g["events"]:
                    for r in e["responses"]:
                        key.setdefault(number, {})[(int(r["turn"]), bool(r["forced"]))] = dict(r, sweeper=e["mon"], via=e["via"])
    totals = {"won": [0, 0, 0], "lost": [0, 0, 0], "sweep_losses": [0, 0, 0], "key_moves": [0, 0]}
    mismatch = 0
    report: List[Dict[str, Any]] = []
    if category:
        games = [g for g in games if (re.findall(r"-(\d+)", g["file"]) or ["?"])[0] in category]     # the scanned games only
    for game in games:
        number = (re.findall(r"-(\d+)", game["file"]) or ["?"])[0]
        ds = game["decisions"]
        mismatch += sum(1 for d in ds if not d["replayed_veto_is_played"] and d["source"] == "mix")
        ch = [d for d in ds if d["changed"]]
        bucket = "won" if game["won"] else "lost"
        totals[bucket][0] += len(ds)
        totals[bucket][1] += sum(1 for d in ds if d["setup"])
        totals[bucket][2] += len(ch)
        sweep = category.get(number) == "loss: set-up sweep"
        if sweep:
            totals["sweep_losses"][0] += len(ds)
            totals["sweep_losses"][1] += sum(1 for d in ds if d["setup"])
            totals["sweep_losses"][2] += len(ch)
        keys = key.get(number, {})
        for d in ds:
            k = keys.get((int(d["turn"]), d["forced"]))
            if k is not None:
                totals["key_moves"][0] += 1
                totals["key_moves"][1] += int(d["changed"])
            if d["changed"] and (sweep or k is not None):
                report.append(dict(d, game=number, key=k is not None, sweeper=(k or {}).get("sweeper")))
        print("%s %-5s %-34s decisions %3d, set-up positions %3d, changed %2d" % (number, "WON" if game["won"] else "LOST",
              category.get(number, ""), len(ds), sum(1 for d in ds if d["setup"]), len(ch)))
        for d in ch:
            k = keys.get((int(d["turn"]), d["forced"]))
            print("   T%-3d%s %-8s veto=%-24s RULE=%-24s net=%-24s (p %.2f) search=%-24s [%s]%s" % (
                d["turn"], "f" if d["forced"] else " ", "KEY" if k else "", d["veto"], d["rule"], d["net"], d["p_net"], d["search"],
                d["why"], (" answering " + k["sweeper"]) if k else ""))
    print("\nreplayed veto != played on mix decisions: %d" % mismatch)
    print("totals (decisions, set-up positions, changed):", json.dumps(totals))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"settings": json.loads(args.settings), "totals": totals, "mismatch": mismatch, "changed_in_sweep_losses": report,
                       "games": [{"file": g["file"], "won": g["won"], "changed": sum(1 for d in g["decisions"] if d["changed"])} for g in games]}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
