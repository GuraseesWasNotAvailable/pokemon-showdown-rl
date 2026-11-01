"""What the bot does after the foe sets up, and what it had to answer with.

    python bench/ladder_log/setups.py [ladder folder ...]   (default runs/ladder)
"""
import collections
import glob
import json
import os
import re
import sys

ROOTS = sys.argv[1:] or ["runs/ladder"]     # ladder folders (each has battles/*.jsonl)
SETUP = {"swordsdance", "nastyplot", "dragondance", "calmmind", "bulkup", "quiverdance", "shellsmash", "irondefense", "agility",
         "rockpolish", "coil", "curse", "bellydrum", "tailglow", "shiftgear", "victorydance", "torchsong", "workup", "growth",
         "honeclaws", "amnesia", "cosmicpower", "stockpile", "filletaway", "clangoroussoul", "noretreat", "geomancy", "tidyup",
         "acidarmor", "cottonguard", "takeheart"}
ANSWER = {"haze", "clearsmog", "roar", "whirlwind", "dragontail", "circlethrow", "encore", "taunt", "yawn", "perishsong",
          "spectralthief", "topsyturvy", "willowisp", "thunderwave", "glare", "nuzzle", "spore", "sleeppowder", "hypnosis",
          "lovelykiss", "darkvoid", "trick", "switcheroo", "knockoff", "partingshot", "memento", "destinybond", "counter",
          "mirrorcoat", "metalburst", "sing", "yawn", "freezedry", "icywind", "snarl", "screech", "charm", "featherdance",
          "nobleroar", "partingshot", "tickle", "growl", "faketears", "eerieimpulse", "confide", "spitup"}
STRICT = {"haze", "clearsmog", "roar", "whirlwind", "dragontail", "circlethrow", "encore", "perishsong", "spectralthief", "topsyturvy"}
ident = lambda t: re.sub(r"[^a-z0-9]", "", str(t or "").lower())

rows = []
for root in ROOTS:
    for path in glob.glob(os.path.join(root, "battles", "*.jsonl")):
        lines, res, our_moves = [], None, collections.defaultdict(set)
        for raw in open(path, encoding="utf-8", errors="replace"):
            try:
                d = json.loads(raw)
            except ValueError:
                continue
            data = d.get("data")
            if d.get("dir") == "in" and isinstance(data, str):
                for ln in data.split("\n"):
                    if ln.startswith("|request|") and len(ln) > 12:
                        try:
                            rq = json.loads(ln[9:])
                        except ValueError:
                            continue
                        for pk in (rq.get("side") or {}).get("pokemon") or []:
                            our_moves[pk.get("details", "").split(",")[0]].update(ident(m) for m in pk.get("moves") or [])
                    else:
                        lines.append(ln)
            elif d.get("dir") == "note" and isinstance(data, dict) and isinstance(data.get("result"), dict):
                res = data["result"]
        if not res or not res.get("role"):
            continue
        arm = res.get("arm")
        role = res["role"]
        tools = {m for ms in our_moves.values() for m in ms}
        # per turn: each side's actions in order
        turn = 0
        acts = collections.defaultdict(list)       # turn -> [(side, kind, name, mon)]
        faint = []                                  # (turn, side, mon)
        last = {}
        kos = []
        for ln in lines:
            p = ln.split("|")
            if len(p) < 3:
                continue
            if p[1] == "turn":
                turn = int(p[2])
            elif p[1] == "move" and len(p) > 3:
                s = "us" if p[2].startswith(role) else "foe"
                acts[turn].append((s, "move", ident(p[3]), p[2].split(": ", 1)[-1]))
                last[s] = (turn, p[2].split(": ", 1)[-1])
            elif p[1] in ("switch", "drag"):
                s = "us" if p[2].startswith(role) else "foe"
                acts[turn].append((s, "switch", p[3].split(",")[0], p[2].split(": ", 1)[-1]))
            elif p[1] == "faint":
                s = "us" if p[2].startswith(role) else "foe"
                mon = p[2].split(": ", 1)[-1]
                faint.append((turn, s, mon))
                o = "foe" if s == "us" else "us"
                if last.get(o) and last[o][0] == turn:
                    kos.append((turn, o, last[o][1], s, mon))
        seen = set()
        for t in sorted(acts):
            for s, kind, name, mon in acts[t]:
                if s != "foe" or kind != "move" or name not in SETUP or (mon, t) in seen:
                    continue
                seen.add((mon, t))
                # the setter's later fate: how many of ours it KOs afterwards, whether we KO it before it KOs anyone
                later = [k for k in kos if k[0] >= t and k[1] == "foe" and k[2] == mon]
                died = next((f[0] for f in faint if f[0] >= t and f[1] == "foe" and f[2] == mon), None)
                first_ko = later[0][0] if later else None
                # our reply on the next turn (t+1): the first action of ours
                nxt = [a for a in acts.get(t + 1, []) if a[0] == "us"]
                reply = nxt[0] if nxt else None
                rk = ("switch" if reply and reply[1] == "switch" else
                      "answer move (haze/phaze/encore/status...)" if reply and reply[2] in ANSWER else
                      "own set-up" if reply and reply[2] in SETUP else
                      "protect" if reply and reply[2] in ("protect", "detect", "kingsshield", "spikyshield", "banefulbunker") else
                      "attack/other" if reply else "none (we fainted or game over)")
                rows.append({"arm": arm, "won": bool(res.get("won")), "move": name, "kos_after": len(later),
                             "we_killed_first": died is not None and (first_ko is None or died <= first_ko),
                             "reply": rk, "reply_move": reply[2] if reply else None,
                             "had_strict": bool(tools & STRICT), "strict": sorted(tools & STRICT),
                             "used_strict_ever": any(a[0] == "us" and a[2] in STRICT for tt in acts if tt > t for a in acts[tt])})

MIX = ("mix", "mix-lines")


def summary(name, sub):
    if not sub:
        return
    n = len(sub)
    print("\n== %s: foe set-up moves %d (games with one %d)" % (name, n, 0))
    print("  the setter later KOed 2+ of ours: %d%%; we KOed it before it KOed anyone: %d%%" % (
        100 * sum(r["kos_after"] >= 2 for r in sub) / n, 100 * sum(r["we_killed_first"] for r in sub) / n))
    c = collections.Counter(r["reply"] for r in sub)
    for k, v in c.most_common():
        part = [r for r in sub if r["reply"] == k]
        print("  reply %-42s %4.0f%%  -> setter KOs 2+ after: %3.0f%%, game won %3.0f%% (n=%d)" % (
            k, 100 * v / n, 100 * sum(r["kos_after"] >= 2 for r in part) / len(part), 100 * sum(r["won"] for r in part) / len(part), len(part)))
    hs = [r for r in sub if r["had_strict"]]
    print("  we carried Haze/phazing/Encore/Perish Song on the team: %d of %d set-ups; used one after the set-up: %d%%" % (
        len(hs), n, 100 * sum(r["used_strict_ever"] for r in hs) / max(1, len(hs))))
    print("  most common set-up moves: %s" % collections.Counter(r["move"] for r in sub).most_common(8))


summary("network+search", [r for r in rows if r["arm"] in MIX])
summary("LLM alone", [r for r in rows if str(r["arm"]).endswith("-solo")])
summary("network+search + LLM (sol-lines, luna)", [r for r in rows if r["arm"] in ("sol-lines", "luna")])
