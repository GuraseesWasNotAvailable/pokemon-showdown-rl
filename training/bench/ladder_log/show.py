"""Compact turn-by-turn view of recorded battles: what each side did, HP, boosts, faints, and the bot's read.

    python bench/ladder_log/show.py <battle id prefix> [from turn]   (LADDER_ROOTS: ladder folders, default runs/ladder)
"""
import glob
import json
import os
import sys

ROOTS = os.environ.get("LADDER_ROOTS", "runs/ladder").split(os.pathsep)   # ladder folders
want = sys.argv[1]
start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
path = next(p for r in ROOTS for p in glob.glob(os.path.join(r, "battles", want + "*.jsonl")))
lines, preds, decs, res = [], {}, {}, {}
for raw in open(path, encoding="utf-8", errors="replace"):
    d = json.loads(raw)
    data = d.get("data")
    if d.get("dir") == "in" and isinstance(data, str):
        lines.extend(data.split("\n"))
    elif d.get("dir") == "note" and isinstance(data, dict):
        if "prediction" in data:
            preds[data["prediction"].get("turn")] = data["prediction"].get("line")
        if "decision" in data:
            x = data["decision"]
            decs.setdefault(x.get("turn"), []).append(x)
        if "result" in data:
            res = data["result"]
role = res.get("role")
print(os.path.basename(path)[:40], "won" if res.get("won") else "LOST", res.get("arm"), res.get("checkpoint"), "turns", res.get("turns"),
      "rating", res.get("rating_before"), "opp", res.get("opponent_rating"), "we are", role)
us = lambda who: who.startswith(role)
out = []
turn = 0
team = {}
for ln in lines:
    p = ln.split("|")
    if len(p) < 2:
        continue
    k = p[1]
    tag = lambda who: ("US " if us(who) else "foe ") + who.split(": ", 1)[-1]
    if k == "turn":
        if turn >= start and out:
            print("T%-2d %s" % (turn, " ; ".join(out)))
            for x in decs.get(turn, []) if turn else []:
                pass
        turn = int(p[2])
        out = []
        if turn >= start:
            ds = decs.get(turn, [])
            for x in ds:
                if not x.get("forced"):
                    print("     [we: est %.2f net %s search %s played %s src %s p_played %s]  %s" % (
                        x.get("win_est") or -1, x.get("net"), x.get("search"), x.get("played"), x.get("source"), x.get("p_played"),
                        (preds.get(turn) or "")[:150]))
    elif k in ("switch", "drag"):
        out.append("%s in (%s, %s)" % (tag(p[2]), p[3].split(",")[0], p[4] if len(p) > 4 else ""))
        team.setdefault(p[2][:2], set()).add(p[3].split(",")[0])
    elif k == "move":
        out.append("%s: %s%s" % (tag(p[2]), p[3], " (" + p[5] + ")" if len(p) > 5 and p[5] else ""))
    elif k in ("-damage", "-heal") and len(p) > 3:
        frm = p[4] if len(p) > 4 else ""
        out.append("%s %s%s" % (tag(p[2]), p[3].split(" ")[0], " " + frm if frm else ""))
    elif k in ("-boost", "-unboost") and len(p) > 4:
        out.append("%s %s%s%s" % (tag(p[2]), "+" if k == "-boost" else "-", p[4], p[3]))
    elif k in ("faint", "-terastallize", "-status", "-sidestart", "cant", "-crit", "-supereffective", "-resisted", "-immune", "-fail",
               "-miss", "-activate", "-weather", "-fieldstart", "-item", "-enditem", "-ability", "-transform", "detailschange"):
        out.append("%s %s" % (k.strip("-").upper(), " ".join(x for x in p[2:5] if x)))
    elif k == "win":
        out.append("WIN " + p[2])
if out:
    print("T%-2d %s" % (turn, " ; ".join(out)))
