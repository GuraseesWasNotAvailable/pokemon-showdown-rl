"""Reads every recorded ladder battle (app folders) into one row per game: result, teams, faints, who KOed whom,
boosts, Tera, hazards, status, and the bot's own decisions. Light log parsing only.

    python bench/ladder_log/games.py <out.json> [ladder folder ...]   (default runs/ladder)
"""
import collections
import glob
import json
import os
import re
import sys

ROOTS = sys.argv[2:] or ["runs/ladder"]     # ladder folders (each has battles/*.jsonl)
SETUP = {"swordsdance", "nastyplot", "dragondance", "calmmind", "bulkup", "quiverdance", "shellsmash", "irondefense", "agility",
         "rockpolish", "coil", "curse", "bellydrum", "tailglow", "shiftgear", "victorydance", "torchsong", "workup", "growth",
         "honeclaws", "amnesia", "cosmicpower", "stockpile", "filletaway", "clangoroussoul", "noretreat", "geomancy", "tidyup",
         "acidarmor", "autotomize", "cottonguard", "defendorder", "howl", "meditate", "sharpen", "withdraw", "takeheart"}
PROTECT = {"protect", "detect", "spikyshield", "kingsshield", "banefulbunker", "silktrap", "burningbulwark", "obstruct"}
HEAL = {"recover", "roost", "slackoff", "softboiled", "milkdrink", "synthesis", "moonlight", "morningsun", "shoreup", "strengthsap",
        "rest", "wish", "lunarblessing", "junglehealing", "lifedew"}


def ident(text):
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def species_of(details):
    return details.split(",")[0].strip()


def read_battle(path):
    notes, decisions, outcomes, lines = {}, [], [], []
    for raw in open(path, encoding="utf-8", errors="replace"):
        try:
            d = json.loads(raw)
        except ValueError:
            continue
        data = d.get("data")
        if d.get("dir") == "note" and isinstance(data, dict):
            for k, v in data.items():
                if k == "decision" and isinstance(v, dict):
                    decisions.append(v)
                elif k == "outcome" and isinstance(v, dict):
                    outcomes.append(v)
                elif k not in ("prediction", "order_wait_ms"):
                    notes[k] = v
        elif d.get("dir") == "in" and isinstance(data, str):
            lines.extend(data.split("\n"))
    res = notes.get("result")
    if not isinstance(res, dict):
        return None
    role = res.get("role")
    names = {}
    for ln in lines:
        p = ln.split("|")
        if len(p) > 3 and p[1] == "player" and p[3]:
            names[p[2]] = ident(p[3])
    if not role:
        user = ident(res.get("user") or notes.get("user"))
        role = next((s for s, n in names.items() if n == user), None)
    if role not in ("p1", "p2"):
        return None
    foe = "p2" if role == "p1" else "p1"
    side = lambda who: who[:2]
    g = {"battle": res.get("battle"), "time": res.get("time"), "arm": res.get("arm") or notes.get("arm"),
         "checkpoint": res.get("checkpoint") or notes.get("checkpoint"), "won": bool(res.get("won")), "turns": res.get("turns"),
         "rating_before": res.get("rating_before"), "rating_after": res.get("rating_after"), "opp_rating": res.get("opponent_rating"),
         "user": res.get("user"), "role": role, "folder": os.path.basename(os.path.dirname(os.path.dirname(path))),
         "result": {k: res.get(k) for k in ("tera_turn", "tera_by", "disagreed", "overruled", "search_overruled", "decisions",
                                             "pred_top1_right", "pred_scored", "luna_cost_usd", "luna_latency_ms_mean", "luna_fallbacks",
                                             "luna_decisions", "luna_played", "luna_model", "fallbacks", "late", "ending", "ending_by",
                                             "abandoned")}}
    team = {"us": [], "foe": []}
    active = {}                 # side -> nickname key 'p1a: X'
    nick_species = {}
    boosts = collections.defaultdict(lambda: collections.Counter())   # mon key -> stat stages
    faints = {"us": [], "foe": []}
    kos = []                    # (turn, killer side, killer species, move, victim species, killer boosted)
    tera = {"us": None, "foe": None}
    hazards = {"us": collections.Counter(), "foe": collections.Counter()}   # hazards ON that side
    status = {"us": collections.Counter(), "foe": collections.Counter()}
    moves = {"us": collections.Counter(), "foe": collections.Counter()}
    setups = {"us": 0, "foe": 0}
    max_boost = {"us": 0, "foe": 0}
    turn = 0
    last_move = {}              # side -> (turn, mon key, move)
    who = lambda s: "us" if s == role else "foe"
    winner = None
    for ln in lines:
        p = ln.split("|")
        if len(p) < 2:
            continue
        k = p[1]
        if k == "turn":
            turn = int(p[2])
        elif k in ("switch", "drag", "replace") and len(p) > 3:
            mon = p[2].split(":")[0][:2] + ": " + p[2].split(": ", 1)[-1]
            s = side(p[2])
            sp = species_of(p[3])
            nick_species[mon] = sp
            if sp not in team[who(s)]:
                team[who(s)].append(sp)
            old = active.get(s)
            if old:
                boosts[old].clear()
            active[s] = mon
        elif k == "move" and len(p) > 3:
            s = side(p[2])
            mon = p[2][:2] + ": " + p[2].split(": ", 1)[-1]
            m = ident(p[3])
            moves[who(s)][m] += 1
            if m in SETUP:
                setups[who(s)] += 1
            last_move[s] = (turn, mon, p[3])
        elif k in ("-boost", "-unboost") and len(p) > 4:
            mon = p[2][:2] + ": " + p[2].split(": ", 1)[-1]
            try:
                n = int(p[4])
            except ValueError:
                continue
            boosts[mon][p[3]] += n if k == "-boost" else -n
            s = who(side(p[2]))
            pos = sum(v for st, v in boosts[mon].items() if st in ("atk", "spa", "spe") and v > 0)
            max_boost[s] = max(max_boost[s], pos)
        elif k in ("-clearboost", "-clearallboost"):
            if k == "-clearallboost":
                boosts.clear()
            elif len(p) > 2:
                boosts[p[2][:2] + ": " + p[2].split(": ", 1)[-1]].clear()
        elif k == "faint" and len(p) > 2:
            s = side(p[2])
            mon = p[2][:2] + ": " + p[2].split(": ", 1)[-1]
            faints[who(s)].append((turn, nick_species.get(mon, mon)))
            other = "p2" if s == "p1" else "p1"
            lm = last_move.get(other)
            if lm and lm[0] == turn:
                kb = sum(v for st, v in boosts[lm[1]].items() if st in ("atk", "spa", "spe") and v > 0)
                kos.append((turn, who(other), nick_species.get(lm[1], lm[1]), lm[2], nick_species.get(mon, mon), kb))
        elif k == "-terastallize" and len(p) > 3:
            s = who(side(p[2]))
            if tera[s] is None:
                tera[s] = (turn, nick_species.get(p[2][:2] + ": " + p[2].split(": ", 1)[-1]), p[3])
        elif k == "-sidestart" and len(p) > 3:
            s = who(p[2][:2])
            hazards[s][ident(p[3].replace("move: ", ""))] += 1
        elif k == "-status" and len(p) > 3:
            status[who(side(p[2]))][p[3]] += 1
        elif k == "win" and len(p) > 2:
            winner = ident(p[2])
    g.update({"team": team, "faints": faints, "kos": kos, "tera": tera, "hazards": {s: dict(v) for s, v in hazards.items()},
              "status": {s: dict(v) for s, v in status.items()}, "setups": setups, "max_boost": max_boost,
              "moves": {s: dict(v.most_common(40)) for s, v in moves.items()}, "ended_turn": turn,
              "first_faint": "us" if faints["us"] and (not faints["foe"] or faints["us"][0][0] < faints["foe"][0][0]) else
                             "foe" if faints["foe"] else None,
              "left": {"us": 6 - len(faints["us"]), "foe": 6 - len(faints["foe"])}})
    # the bot's own decisions
    ds = []
    for d in decisions:
        lu = d.get("luna") if isinstance(d.get("luna"), dict) else {}
        ds.append({"turn": d.get("turn"), "forced": d.get("forced"), "legal": d.get("legal"), "net": d.get("net"), "search": d.get("search"),
                   "played": d.get("played"), "source": d.get("source"), "win_est": d.get("win_est"), "p_played": d.get("p_played"),
                   "p_net": d.get("p_net"), "disagree": d.get("disagree"), "tera_added": d.get("tera_added"), "tera_dropped": d.get("tera_dropped"),
                   "score_net": d.get("score_net"), "score_search": d.get("score_search"), "score_played": d.get("score_played"),
                   "opp": d.get("opponent_active"), "luna": bool(lu), "foe_next": lu.get("foe_next"), "fell_back": lu.get("fell_back")})
    g["decisions"] = ds
    pr = []
    for o in outcomes:
        pd = o.get("predict") or {}
        oc = o.get("outcome") or pd.get("outcome") or {}
        top = pd.get("top") or []
        did = oc.get("did") or oc.get("name")
        if not top or not did:
            continue
        pr.append({"p_switch": pd.get("p_switch"), "top1": top[0][0], "did": did,
                   "switched": str(did).startswith("switch"), "top1_right": top[0][0] == did or (top[0][0] == "switch ?" and str(did).startswith("switch"))})
    g["predictions"] = pr
    return g


games = []
for root in ROOTS:
    for path in glob.glob(os.path.join(root, "battles", "*.jsonl")):
        try:
            g = read_battle(path)
        except Exception as e:      # one bad file must not stop the read
            print("skip", path, e)
            continue
        if g:
            games.append(g)
games.sort(key=lambda g: g["time"] or "")
json.dump(games, open(sys.argv[1], "w", encoding="utf-8"))
print(len(games), "games read")
print(collections.Counter((g["folder"], g["arm"], g["checkpoint"]) for g in games))
