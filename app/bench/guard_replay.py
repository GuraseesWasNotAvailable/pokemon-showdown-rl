"""
The guard (search_player.MixSettings.guard) replayed on recorded ladder battles: on which
decisions it would have fired, and what it would have played instead of what was played.

    python bench/guard_replay.py runs/ladder/battles/*.jsonl [--all] [--turns 2693998716:3,2694007454:42]

Every decision note of a battle file (ladder.py writes them) has the network's probabilities, the
search's shares and the legal actions. The battle's own lines give the rest: the opponent's
boosts at the moment of the request (bench/opp_dataset.GameRows, fed line by line), our moves and
their order (the last |request|), our status, and the read (opponent_model.py on the tracker's
context and the search's view of the opponent as the note kept it: its six likeliest options).
The search is NOT asked again: the guard is a rule over the numbers the search already gave.

Approximations: the note keeps shares, not visits (an action with a share > 0 counts as seen), and
the six likeliest options of the opponent, not all. So a "would play" is the rule's answer on
those numbers, not a game played with it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import opponent_model as om  # noqa: E402
import search_player as sp  # noqa: E402
import setup_moves as sm  # noqa: E402


def to_id(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def names_of(req: Dict[str, Any], team_order: List[str]) -> Dict[int, str]:
    """Action index -> our move id or "switch <species>", from the request (moves) and the first request's team order (switches)."""
    out: Dict[int, str] = {}
    mine = next((p for p in (req.get("side") or {}).get("pokemon") or [] if p.get("active")), None)
    moves = list((mine or {}).get("moves") or [])
    if not moves:
        moves = [m.get("id") for m in ((req.get("active") or [{}])[0].get("moves") or [])]
    for i, move in enumerate(moves[:4]):
        out[i] = to_id(move)
        out[i + 10] = to_id(move) + " +tera"
    for i, ident in enumerate(team_order[:6]):
        out[4 + i] = "switch " + to_id(ident.split(": ", 1)[-1])
    return out


def our_state(req: Dict[str, Any]) -> Tuple[List[str], bool]:
    """Our active Pokemon's types and whether it has a status, from the request."""
    mine = next((p for p in (req.get("side") or {}).get("pokemon") or [] if p.get("active")), None) or {}
    species = str(mine.get("details") or "").split(",")[0]
    types = om.dex().types(species)
    condition = str(mine.get("condition") or "")
    return types, len(condition.split()) > 1 and condition.split()[1] not in ("fnt",)


def threat_of(rec: Dict[str, Any], names: Dict[int, str], boosts: Dict[str, int], last_setup: bool, dist: Optional[Dict[str, float]],
              their_types: List[str], our_types: List[str], our_status: bool) -> sp.Threat:
    boost = max([int(v) for k, v in boosts.items() if k in sm.STATS and int(v) > 0] or [0])
    setup = sleep = status = 0.0
    for name, p in (dist or {}).items():
        if not name.startswith("move "):
            continue
        move = name[5:].replace(" terastallize", "")
        setup += p if sm.is_setup(move, their_types) else 0.0
        sleep += p if sm.is_sleep(move) else 0.0
        status += p if sm.is_status_move(move) else 0.0
    if our_status:
        sleep = 0.0
    passive, fails = [], []
    for a in list(range(4)) + list(range(10, 14)):
        if not rec["mask"][a]:
            continue
        move = names.get(a, "").replace(" +tera", "")
        if not move:
            continue
        if sm.is_passive(move) or sm.is_setup(move, our_types):
            passive.append(a)
        if sm.fails_on_status(move):
            fails.append(a)
    return sp.Threat(boost=boost, last_setup=last_setup, setup_read=setup, sleep_read=sleep, status_read=status,
                     passive=tuple(passive), fails=tuple(fails))


def replay(path: str, model: om.OpponentModel) -> Dict[str, Any]:
    tracker = None
    me = None
    role = None
    req: Dict[str, Any] = {}
    team_order: List[str] = []
    out: List[Dict[str, Any]] = []
    won = None
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    tag = os.path.basename(path)
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
            settings = sp.MixSettings(guard=True)
            explain: Dict[str, Any] = {}
            guarded = sp.mix(rec["probs"], rec["mask"], view, settings, threat=threat, explain=explain)
            plain = sp.mix(rec["probs"], rec["mask"], view, sp.MixSettings())
            top = sorted((dist or {}).items(), key=lambda kv: -kv[1])[:3]
            out.append({"turn": rec["turn"], "forced": bool(rec.get("forced")), "mode": rec.get("mode"), "source": rec.get("source"),
                        "net": names.get(rec["net"], rec["net"]), "search": names.get(rec["search"], rec["search"]),
                        "played": names.get(rec["played"], rec["played"]), "veto": names.get(plain, plain),
                        "guard": explain.get("guard") or "", "guarded": names.get(guarded, guarded),
                        "changed": bool(explain.get("guard")) and guarded != plain, "dropped": [names.get(a, a) for a in explain.get("dropped") or []],
                        "by": explain.get("by"), "boost": threat.boost, "setup_read": round(threat.setup_read, 2),
                        "sleep_read": round(threat.sleep_read, 2), "read": [[n, round(p, 2)] for n, p in top],
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
                tracker = om.OpponentTracker(tag)
            tracker.feed(line, role)
    return {"file": tag, "won": won, "decisions": out}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--all", action="store_true", help="print every decision the guard fired on, not only the changed ones")
    ap.add_argument("--turns", default="", help="GAME:TURN[,GAME:TURN]: print these decisions whatever happened (GAME: digits of the battle id)")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)
    wanted = {(g.strip(), int(t)) for g, t in (x.split(":") for x in args.turns.split(",") if x.strip())}
    model = om.OpponentModel.load()
    games = [replay(path, model) for path in sorted(args.files)]
    fired = changed = decisions = mismatch = 0
    by_result = {True: [0, 0, 0], False: [0, 0, 0]}
    for game in games:
        number = (re.findall(r"-(\d+)", game["file"]) or ["?"])[0]
        searched = [d for d in game["decisions"]]
        decisions += len(searched)
        mismatch += sum(1 for d in searched if not d["replayed_veto_is_played"] and d["source"] == "mix")
        f = [d for d in searched if d["guard"]]
        c = [d for d in f if d["changed"]]
        fired += len(f)
        changed += len(c)
        if game["won"] is not None:
            row = by_result[game["won"]]
            row[0] += len(searched)
            row[1] += len(f)
            row[2] += len(c)
        shown = [d for d in searched if (d["guard"] and (args.all or d["changed"])) or (number, d["turn"]) in wanted]
        print("%s %s: %d decisions, guard fired %d, changed %d" % (number, {True: "WON ", False: "LOST", None: "?   "}[game["won"]],
                                                                   len(searched), len(f), len(c)))
        for d in shown:
            print("   T%-3d %-6s guard=%-18s played=%-22s veto=%-22s GUARD=%-22s net=%-20s search=%-20s dropped=%s read=%s"
                  % (d["turn"], "forced" if d["forced"] else "move", d["guard"] or "-", d["played"], d["veto"], d["guarded"],
                     d["net"], d["search"], ",".join(map(str, d["dropped"])) or "-", d["read"]))
    print("\nall: %d decisions, guard fired on %d, changed the move on %d; the replayed veto differs from what was played on %d mix decisions"
          % (decisions, fired, changed, mismatch))
    for result, (n, f, c) in by_result.items():
        print("  in games %s: %d decisions, fired %d, changed %d" % ("won" if result else "lost", n, f, c))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(games, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
