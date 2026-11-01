"""
The set-up problem on the real ladder: for every recorded ladder game (ladder.py's
runs/ladder/battles/*.jsonl and results.jsonl), every opponent set-up and what we did about it.

A set-up is, per stint of an opponent's Pokemon on the field (from its switch-in to its leaving):
  * a boosting move that raised its own stats (Swords Dance, Calm Mind, Dragon Dance, Curse...),
  * or reaching +2 in any of atk, def, spa, spd, spe by anything else (Speed Boost, Moxie,
    Weakness Policy, Flame Charge...),
whichever comes first in the stint. A Curse / Rest / Sleep Talk Pokemon is tagged "loop".

After it, the KOs it scores: a faint of ours whose last direct hit (a -damage line with no
[from], inside that Pokemon's move) came from it, while it is still in the same stint. A
Pokemon's KOs are summed over all its set-up stints: 2 or more is "took 2+".

For each set-up, the decisions of ours from its turn to the end of the stint (at most --window):
what was played (attack hitting the unboosted defence or into a boosted one, phaze / Haze /
Encore / Taunt, switch, our own set-up, status, recovery, a priority move, Sucker Punch into a
status move...), and who chose it (both, net, search, "neither" for the veto's third action,
fallback). Read only.

    python setup_ladder_scan.py --dir runs/ladder [--json out.json] [--window 4] [--games N]
    python bench/setup_ladder_scan.py --dir runs/ladder --quiet --by-arm --arms mix,mix-blend4

--by-arm (round 2): per arm of an A/B run (the "arm" of results.jsonl), the games, the record and the
games lost to a set-up sweep (an opponent's Pokemon that set up and then made 2 or more KOs), each with
a 95% Wilson interval, and each other arm minus the first with a 95% Newcombe interval. --arms keeps
only the games of those arms (the first is the control).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict

STATS = ("atk", "def", "spa", "spd", "spe")
ANTI = {"haze", "clearsmog", "roar", "whirlwind", "dragontail", "circlethrow", "encore", "taunt", "perishsong",
        "spectralthief", "topsyturvy", "yawn", "disable", "destinybond", "psychup", "memento", "partingshot",
        "charm", "featherdance", "screech", "faketears", "metalsound", "eerieimpulse", "nobleroar", "tickle",
        "strengthsap", "glare", "thunderwave", "willowisp", "toxic", "spore", "sleeppowder", "hypnosis",
        "lovelykiss", "sing", "darkvoid", "trick", "switcheroo", "whirlpool"}
PHAZE_HAZE = {"haze", "clearsmog", "roar", "whirlwind", "dragontail", "circlethrow", "encore", "taunt",
              "perishsong", "spectralthief", "topsyturvy", "yawn", "disable", "destinybond", "psychup"}
DEBUFF = {"memento", "partingshot", "charm", "featherdance", "screech", "faketears", "metalsound", "eerieimpulse",
          "nobleroar", "tickle", "strengthsap"}
HITS_DEF = {"psyshock", "psystrike", "secretsword"}            # special moves on the Defence
USES_TARGET_ATK = {"foulplay"}
FAILS_ON_STATUS = {"suckerpunch", "thunderclap"}
CODE_BOOSTS = {"curse": {"atk": 1, "def": 1}, "bellydrum": {"atk": 6}, "tidyup": {"atk": 1, "spe": 1},
               "acupressure": {"def": 2}}

_MOVES = None


def to_id(s):
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def moves():
    global _MOVES
    if _MOVES is None:
        try:
            from poke_env.data import GenData
            _MOVES = GenData.from_gen(9).moves
        except Exception:
            _MOVES = {}
    return _MOVES


def entry(mid):
    return moves().get(to_id(mid)) or {}


def raised(mid):
    m = to_id(mid)
    if m in CODE_BOOSTS:
        return CODE_BOOSTS[m]
    e = entry(m)
    if e.get("category") != "Status" or e.get("target") not in ("self", "adjacentAllyOrSelf", "allies"):
        return {}
    return {s: int(v) for s, v in (e.get("boosts") or {}).items() if s in STATS and int(v) > 0}


def classify(mid, opp_boosts, same_turn_setup):
    """What one move of ours was against a boosted / boosting Pokemon."""
    m = to_id(mid)
    e = entry(m)
    cat = e.get("category")
    if m in PHAZE_HAZE:
        return "phaze/haze/encore/taunt"
    if m in DEBUFF:
        return "debuff"
    if raised(m):
        return "own set-up"
    if cat == "Status":
        if e.get("heal") or (e.get("flags") or {}).get("heal") or m in ("rest", "wish", "painsplit", "roost", "recover",
                                                                        "slackoff", "softboiled", "moonlight", "synthesis",
                                                                        "morningsun", "shoreup", "milkdrink", "lifedew"):
            return "recovery"
        if e.get("status") or e.get("volatileStatus") in ("confusion", "leechseed", "attract") or m in ANTI:
            return "status"
        return "passive (protect/sub/hazard/screen)"
    if not cat:
        return "move?" + m
    pri = int(e.get("priority") or 0)
    if m in FAILS_ON_STATUS:
        return "sucker punch"
    if pri > 0:
        return "priority attack"
    if m in USES_TARGET_ATK:
        return "attack (Foul Play)"
    hits = "def" if (cat == "Physical" or m in HITS_DEF) else "spd"
    other = "spd" if hits == "def" else "def"
    b_hit, b_other = opp_boosts.get(hits, 0), opp_boosts.get(other, 0)
    if b_hit > 0:
        return "attack into boosted %s" % hits
    if b_other > 0:
        return "attack on the other defence"
    return "attack"


def load(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows


def base(a):
    return a - 10 if a is not None and a >= 10 else a


def chooser(rec):
    if rec.get("source") != "mix":
        return "fallback:%s" % (rec.get("reason") or rec.get("source"))
    net, srch, played = base(rec.get("net")), base(rec.get("search")), base(rec.get("played"))
    if srch is None:
        return "net"
    if played == net == srch:
        return "both"
    if played == net:
        return "net"
    if played == srch:
        return "search"
    return "neither"


def label(a, team, req):
    if a is None:
        return "-"
    b = base(a)
    if 0 <= b <= 3:
        act = ((req or {}).get("active") or [{}])[0] if (req or {}).get("active") else {}
        ms = [m["id"] for m in act.get("moves", [])]
        if not ms:
            mine = next((p for p in (req or {}).get("side", {}).get("pokemon", []) if p.get("active")), None)
            ms = mine["moves"] if mine else []
        name = ms[b] if b < len(ms) else "move%d" % b
        return name + ("+tera" if a >= 10 else "")
    i = b - 4
    return "sw " + (team[i] if team and i < len(team) else str(i))


class Game:
    def __init__(self, path, result):
        self.path = path
        self.result = result
        self.tag = os.path.basename(path)[:-6]
        self.rows = load(path)
        self.me = next((d["data"]["user"] for d in self.rows if d["dir"] == "note" and "user" in d["data"]), None)
        self.side = None
        self.won = bool(result.get("won")) if result else None
        self.events = []     # set-up events
        self.decisions = []  # per decision: dict
        self.our_faints = 0
        self.opp_faints = 0
        self.parse()

    def parse(self):
        turn = 0
        active = {}                 # side -> ident name (after "p1a: ")
        boosts = {"p1": Counter(), "p2": Counter()}
        stint = {"p1": 0, "p2": 0}
        last_move_user = None       # (side, name, move id)
        last_hit = {}               # our name -> (opp stint key, turn)
        cur_event = {}              # opp stint id -> event dict
        opp_last_move = {}
        team, req = None, None
        pending = None
        sucker_failed = False
        setup_moves_this_turn = set()
        types_tera = {}
        for d in self.rows:
            if d["dir"] == "note" and isinstance(d.get("data"), dict) and "decision" in d["data"]:
                rec = d["data"]["decision"]
                opp = "p1" if self.side == "p2" else "p2"
                pending = {"turn": int(rec.get("turn") or turn), "forced": bool(rec.get("forced")), "rec": rec,
                           "opp_active": active.get(opp), "opp_stint": (opp, stint[opp]),
                           "opp_boosts": {s: v for s, v in boosts[opp].items() if v},
                           "ours": active.get(self.side), "who": chooser(rec),
                           "net": label(rec.get("net"), team, req), "search": label(rec.get("search"), team, req),
                           "played": label(rec.get("played"), team, req),
                           "share_net": rec.get("share_net"), "share_search": rec.get("share_search"),
                           "share_played": rec.get("share_played"), "p_net": rec.get("p_net"),
                           "p_played": rec.get("p_played"),
                           "opp_read": [(o[0], o[1]) for o in (rec.get("opponent") or [])[:4]]}
                self.decisions.append(pending)
                continue
            if d["dir"] != "in" or not isinstance(d.get("data"), str):
                continue
            for x in d["data"].split("\n"):
                p = x.split("|")
                if len(p) < 2:
                    continue
                k = p[1]
                if k == "player" and len(p) > 3 and to_id(p[3]) == to_id(self.me):
                    self.side = p[2]
                elif k == "request" and len(p) > 2 and p[2]:
                    try:
                        r = json.loads(p[2])
                    except ValueError:
                        continue
                    if r.get("wait"):
                        continue
                    req = r
                    if team is None and r.get("side"):
                        team = [q["ident"][4:] for q in r["side"]["pokemon"]]
                elif k == "turn":
                    turn = int(p[2])
                    setup_moves_this_turn = set()
                elif k in ("switch", "drag", "replace") and len(p) > 3:
                    s = p[2][:2]
                    name = p[2][5:]
                    if k != "replace":
                        stint[s] += 1
                        boosts[s] = Counter()
                    active[s] = name
                elif k == "move" and len(p) > 3:
                    s = p[2][:2]
                    mid = to_id(p[3])
                    last_move_user = (s, p[2][5:], mid)
                    if s != self.side:
                        opp_last_move[s] = mid
                elif k in ("-boost", "-unboost") and len(p) > 4:
                    s = p[2][:2]
                    st = p[3]
                    v = int(p[4]) * (1 if k == "-boost" else -1)
                    boosts[s][st] = max(-6, min(6, boosts[s][st] + v))
                    if s != self.side and k == "-boost":
                        key = (s, stint[s])
                        via = None
                        if last_move_user and last_move_user[0] == s and raised(last_move_user[2]) \
                                and "[from]" not in x:
                            via = "move:" + last_move_user[2]
                        elif boosts[s][st] >= 2:
                            src = re.search(r"\[from\] (.*)", x)
                            via = "+2 %s (%s)" % (st, src.group(1) if src else
                                                  ("move:" + last_move_user[2] if last_move_user and last_move_user[0] == s else "?"))
                        if via and key not in cur_event:
                            cur_event[key] = {"turn": turn, "mon": p[2][5:], "via": via, "stint": key, "kos": 0,
                                              "ko_turns": [], "end": None, "end_turn": None, "moves": set(),
                                              "our_answers": answers(req), "material": (len(answers(req)), 6 - self.opp_faints)}
                            self.events.append(cur_event[key])
                        if key in cur_event and via and via.startswith("move:"):
                            cur_event[key]["moves"].add(via[5:])
                elif k == "-setboost" and len(p) > 4:
                    s = p[2][:2]
                    boosts[s][p[3]] = int(p[4])
                    if s != self.side and int(p[4]) >= 2:
                        key = (s, stint[s])
                        if key not in cur_event:
                            cur_event[key] = {"turn": turn, "mon": p[2][5:], "via": "move:bellydrum", "stint": key, "kos": 0,
                                              "ko_turns": [], "end": None, "end_turn": None, "moves": {"bellydrum"},
                                              "our_answers": answers(req), "material": (len(answers(req)), 6 - self.opp_faints)}
                            self.events.append(cur_event[key])
                elif k in ("-clearboost", "-clearpositiveboost", "-invertboost") and len(p) > 2:
                    s = p[2][:2]
                    if k == "-invertboost":
                        boosts[s] = Counter({a: -b for a, b in boosts[s].items()})
                    elif k == "-clearpositiveboost":
                        boosts[s] = Counter({a: b for a, b in boosts[s].items() if b < 0})
                    else:
                        boosts[s] = Counter()
                    key = (s, stint[s])
                    if key in cur_event and cur_event[key]["end"] is None:
                        cur_event[key]["end"] = "boosts cleared (%s)" % (p[4] if len(p) > 4 else k)
                        cur_event[key]["end_turn"] = turn
                        del cur_event[key]          # a new set-up in the same stint is a new event
                elif k == "-clearallboost":
                    for s in ("p1", "p2"):
                        boosts[s] = Counter()
                        key = (s, stint[s])
                        if key in cur_event and cur_event[key]["end"] is None:
                            cur_event[key]["end"] = "haze"
                            cur_event[key]["end_turn"] = turn
                            del cur_event[key]
                elif k == "-clearnegativeboost" and len(p) > 2:
                    s = p[2][:2]
                    boosts[s] = Counter({a: b for a, b in boosts[s].items() if b > 0})
                elif k == "-copyboost" and len(p) > 3:
                    src, dst = p[2][:2], p[3][:2]
                    boosts[dst] = Counter(boosts[src])
                elif k == "-damage" and len(p) > 2:
                    s = p[2][:2]
                    if s == self.side and "[from]" not in x and last_move_user and last_move_user[0] != s:
                        last_hit[p[2][5:]] = ((last_move_user[0], stint[last_move_user[0]]), turn)
                    elif s == self.side and "[from]" in x:
                        last_hit[p[2][5:]] = None       # finished by residual damage, recoil, a Rocky Helmet: not a KO
                elif k == "faint" and len(p) > 2:
                    s = p[2][:2]
                    if s == self.side:
                        self.our_faints += 1
                        hit = last_hit.get(p[2][5:])
                        if hit and hit[1] == turn and hit[0] in cur_event                                 and max(boosts[hit[0][0]].values() or [0]) > 0:
                            ev = cur_event[hit[0]]
                            ev["kos"] += 1
                            ev["ko_turns"].append(turn)
                    else:
                        self.opp_faints += 1
                        key = (s, stint[s])
                        if key in cur_event and cur_event[key]["end"] is None:
                            cur_event[key]["end"] = "fainted"
                            cur_event[key]["end_turn"] = turn
                elif k == "-fail" and len(p) > 2 and p[2][:2] == self.side and last_move_user \
                        and last_move_user[2] in FAILS_ON_STATUS:
                    sucker_failed = True
                    if self.decisions:
                        self.decisions[-1]["sucker_failed"] = True
            # a switch out ends the stint
            for key, ev in cur_event.items():
                if ev["end"] is None and stint[key[0]] != key[1]:
                    ev["end"] = "switched out"
                    ev["end_turn"] = turn
        for ev in self.events:
            ev["moves"] = sorted(ev["moves"])
            if ev["end"] is None:
                ev["end"] = "game over"
        # per Pokemon: KOs over all its set-up stints; loop tag
        opp_moves = defaultdict(set)
        for d in self.rows:
            if d["dir"] == "in" and isinstance(d.get("data"), str):
                for x in d["data"].split("\n"):
                    p = x.split("|")
                    if len(p) > 3 and p[1] == "move" and p[2][:2] != self.side:
                        opp_moves[p[2][5:]].add(to_id(p[3]))
        self.loop_mons = {m for m, s in opp_moves.items() if {"rest", "sleeptalk"} <= s or ({"curse", "rest"} <= s)}
        by_mon = Counter()
        for ev in self.events:
            by_mon[ev["mon"]] += ev["kos"]
        self.mon_kos = dict(by_mon)

    def responses(self, ev, window):
        """Our decisions from the set-up's turn to the end of its stint (at most `window` of them)."""
        out = []
        for dec in self.decisions:
            if dec["opp_stint"] != ev["stint"]:
                continue
            if dec["turn"] < ev["turn"]:
                continue
            if ev["end_turn"] is not None and dec["turn"] > ev["end_turn"]:
                continue
            out.append(dec)
        return out[:window]


def answers(req):
    """Our Pokemon still standing (from the last request) and their answers to a set-up: phaze / Haze /
    Encore / Taunt, debuffs, status, priority, Unaware."""
    out = []
    for q in ((req or {}).get("side") or {}).get("pokemon", []):
        if str(q.get("condition", "")).endswith(" fnt") or str(q.get("condition", "")).startswith("0"):
            continue
        tools = []
        for m in q.get("moves", []):
            e = entry(m)
            if m in PHAZE_HAZE or m in DEBUFF:
                tools.append(m)
            elif e.get("category") == "Status" and (e.get("status") in ("tox", "brn", "par", "slp")):
                tools.append(m)
            elif e.get("category") in ("Physical", "Special") and int(e.get("priority") or 0) > 0:
                tools.append(m)
        if q.get("ability") == "unaware":
            tools.append("Unaware")
        out.append("%s%s(%s)" % (q["ident"][4:], "*" if q.get("active") else "", q.get("condition", "").split(" ")[0])
                   + (":" + ",".join(tools) if tools else ""))
    return out


def response_kind(dec, ev):
    played = dec["played"]
    if played.startswith("sw "):
        return "switch"
    mid = played.replace("+tera", "")
    same = dec["turn"] == ev["turn"]
    k = classify(mid, dec["opp_boosts"], same)
    if k == "sucker punch":
        k = "sucker punch (failed)" if dec.get("sucker_failed") else "sucker punch (hit)"
    return k


def wilson(k, n, z=1.96):
    """95% Wilson interval of k out of n, as fractions (0, 1 for n = 0)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def newcombe(k1, n1, k0, n0):
    """95% interval of k1/n1 - k0/n0 (Newcombe's method 10, from the two Wilson intervals)."""
    p1, p0 = (k1 / n1 if n1 else 0.0), (k0 / n0 if n0 else 0.0)
    l1, u1 = wilson(k1, n1)
    l0, u0 = wilson(k0, n0)
    d = p1 - p0
    return d, d - math.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)


def by_arm_text(rows, arms):
    """rows: (arm, won, category) per game. One line per arm, then each arm minus the first."""
    order = list(arms) if arms else sorted({a for a, _, _ in rows})
    stats = {}
    lines = ["\nBy arm (95%% Wilson; differences against %s, 95%% Newcombe):" % (order[0] if order else "-")]
    for arm in order:
        g = [r for r in rows if r[0] == arm]
        n = len(g)
        w = sum(1 for r in g if r[1])
        sw = sum(1 for r in g if r[2] == "loss: set-up sweep")
        stats[arm] = (n, w, sw)
        lw, uw = wilson(w, n)
        ls, us = wilson(sw, n)
        lines.append("  %-14s %3d games, %d-%d, won %.0f%% (%.0f, %.0f), lost to a set-up sweep %d = %.0f%% of games (%.0f, %.0f)"
                     % (arm, n, w, n - w, 100.0 * w / n if n else 0.0, 100 * lw, 100 * uw, sw,
                        100.0 * sw / n if n else 0.0, 100 * ls, 100 * us))
    for arm in order[1:]:
        n1, w1, s1 = stats[arm]
        n0, w0, s0 = stats[order[0]]
        if not (n1 and n0):
            continue
        dw = newcombe(w1, n1, w0, n0)
        ds = newcombe(s1, n1, s0, n0)
        lines.append("  %s minus %s: won %+.1f points (%+.1f, %+.1f), set-up sweep losses %+.1f points of games (%+.1f, %+.1f)"
                     % (arm, order[0], 100 * dw[0], 100 * dw[1], 100 * dw[2], 100 * ds[0], 100 * ds[1], 100 * ds[2]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--json")
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--by-arm", action="store_true", help="per arm of an A/B run: record and set-up sweep losses, with intervals")
    ap.add_argument("--arms", default="", help="with --by-arm: only these arms, the first one the control (e.g. mix,mix-blend4)")
    ap.add_argument("--experiment", default="", help="only the games of this A/B (results.jsonl's \"experiment\": ladder.py's --arms, "
                                                     "e.g. mix,mix-counter), not the ordinary runs' games of the same arm names")
    a = ap.parse_args()
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    if not os.path.exists(os.path.join(a.dir, "results.jsonl")):
        print("no results.jsonl in %s: no games yet" % a.dir)
        return
    results = [json.loads(l) for l in open(os.path.join(a.dir, "results.jsonl"), encoding="utf-8") if l.strip()]
    if a.experiment:
        results = [r for r in results if r.get("experiment") == a.experiment]
    if a.by_arm:
        results = [r for r in results if r.get("arm") and (not arms or r.get("arm") in arms)]
    if a.games:
        results = results[:a.games]
    games = []
    for r in results:
        path = os.path.join(a.dir, "battles", r["battle"] + ".jsonl")
        if os.path.exists(path):
            games.append(Game(path, r))
    out = []
    arm_rows = []
    table = Counter()
    resp_counter = defaultdict(Counter)
    who_counter = defaultdict(Counter)
    for g in games:
        sweeps = [ev for ev in g.events if g.mon_kos.get(ev["mon"], 0) >= 2 and ev["kos"] > 0]
        swept = bool(sweeps)
        sweep_mons = sorted({ev["mon"] for ev in g.events if g.mon_kos.get(ev["mon"], 0) >= 2})
        if not g.won:
            cat = "loss: set-up sweep" if sweep_mons else ("loss: other (opp set up, took <2)" if g.events else "loss: other (no set-up)")
        else:
            cat = "win: opp set up and took 2+" if sweep_mons else ("win: opp set up, stopped" if g.events else "win: no set-up")
        table[cat] += 1
        arm_rows.append((g.result.get("arm") if g.result else None, bool(g.won), cat))
        evs = []
        for ev in g.events:
            resp = g.responses(ev, a.window)
            rows = []
            for dec in resp:
                kind = response_kind(dec, ev)
                rows.append({"turn": dec["turn"], "forced": dec["forced"], "ours": dec["ours"], "played": dec["played"],
                             "net": dec["net"], "search": dec["search"], "who": dec["who"], "kind": kind,
                             "opp_boosts": dec["opp_boosts"], "share_played": dec["share_played"],
                             "share_search": dec["share_search"], "p_net": dec["p_net"], "opp_read": dec["opp_read"]})
                if not dec["forced"]:
                    tag = "sweep-loss" if (not g.won and ev["mon"] in sweep_mons) else ("loss" if not g.won else "win")
                    resp_counter[tag][kind] += 1
                    who_counter[tag][dec["who"]] += 1
            evs.append(dict(ev, mon_kos=g.mon_kos.get(ev["mon"], 0), loop=ev["mon"] in g.loop_mons, responses=rows))
        out.append({"battle": g.tag, "won": g.won, "category": cat, "opponent": (g.result or {}).get("opponent"),
                    "opp_rating": (g.result or {}).get("opponent_rating"), "turns": (g.result or {}).get("turns"),
                    "our_faints": g.our_faints, "opp_faints": g.opp_faints, "sweepers": sweep_mons, "events": evs})
        if not a.quiet:
            print("=== %s %s (%s) vs %s %s, %s turns" % (g.tag[-10:], "WON " if g.won else "LOST", cat,
                                                         (g.result or {}).get("opponent"), (g.result or {}).get("opponent_rating"),
                                                         (g.result or {}).get("turns")))
            for ev in evs:
                print("   T%-3d %-22s %-34s KOs in stint %d (mon total %d)%s, ended: %s T%s, standing %s v %s" % (
                    ev["turn"], ev["mon"], ev["via"], ev["kos"], ev["mon_kos"], " [loop]" if ev["loop"] else "",
                    ev["end"], ev["end_turn"], ev["material"][0], ev["material"][1]))
                print("        ours standing: " + " ".join(ev.get("our_answers") or []))
                for r in ev["responses"]:
                    print("      T%-3d%s %-14s played %-24s %-30s who=%-14s net=%-22s search=%-22s opp%s" % (
                        r["turn"], "f" if r["forced"] else " ", r["ours"], r["played"], r["kind"], r["who"], r["net"],
                        r["search"], r["opp_boosts"]))
    if not (a.by_arm and a.quiet):          # --by-arm --quiet: the per-arm block only
        print("\nGames:", len(games), "won", sum(1 for g in games if g.won))
        for k, v in sorted(table.items()):
            print("  %-40s %d" % (k, v))
        for tag in ("sweep-loss", "loss", "win"):
            print("\nResponses (move decisions, not forced) in %s:" % tag)
            for k, v in resp_counter[tag].most_common():
                print("  %-36s %d" % (k, v))
            print("  who:", dict(who_counter[tag]))
    if a.by_arm:
        print(by_arm_text(arm_rows, arms))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=list)


if __name__ == "__main__":
    main()
