"""
Why the ladder bot loses: every recorded public ladder game of network + search (mix), wins against losses.
The numbers of reports/2026-10-15/LOSS_STUDY.md.

    python reports/2026-10-15/loss_study.py [BATTLES_DIR ...]

The folders are ladder.py's battle records (runs/ladder/battles); by default the owner's laptop's two.
"""
import glob, json, os, re, statistics as st, collections, sys

DIRS = sys.argv[1:] or [r"C:\Users\you\psbot-local\runs\ladder\battles", r"C:\Users\you\ps-ppo-search\runs\ladder\battles"]
ME = "yourshowdownname"
SETUP = {"swordsdance", "dragondance", "nastyplot", "calmmind", "bulkup", "quiverdance", "shellsmash", "agility",
         "bellydrum", "shiftgear", "coil", "irondefense", "victorydance", "tidyup", "filletaway", "tailglow",
         "geomancy", "clangoroussoul", "noretreat", "curse", "workup", "growth", "honeclaws", "rockpolish",
         "autotomize", "cottonguard", "acidarmor", "amnesia", "cosmicpower", "stockpile", "takeheart", "trailblaze",
         "flamecharge", "poweruppunch", "scaleshot", "rapidspin", "torchsong", "aquastep", "esperwing"}
ANSWERS = {"haze", "clearsmog", "roar", "whirlwind", "dragontail", "circlethrow", "encore", "yawn", "spore",
           "sleeppowder", "willowisp", "thunderwave", "glare", "nuzzle", "taunt", "perishsong", "destinybond", "counter",
           "mirrorcoat", "metalburst", "trick", "switcheroo", "knockoff", "partingshot", "memento", "topsyturvy", "spectralthief",
           "psychup", "unaware"}

def norm(s): return re.sub(r"[^a-z0-9]", "", (s or "").lower())

games = {}
for d in DIRS:
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        tag = os.path.basename(f)[:-6]
        if tag in games: continue
        head, result, lines, decs = None, None, [], []
        try:
            for l in open(f, encoding="utf-8"):
                r = json.loads(l)
                if r["dir"] == "in" and isinstance(r["data"], str):
                    lines.extend(r["data"].split("\n"))
                elif r["dir"] == "note" and isinstance(r["data"], dict):
                    if "server" in r["data"] and head is None: head = r["data"]
                    if "result" in r["data"]: result = r["data"]["result"]
                    if "decision" in r["data"]: decs.append(r["data"]["decision"])
        except Exception:
            continue
        if not head or not result or head.get("server") != "public": continue
        if result.get("arm") not in ("mix", None) and not str(result.get("arm", "")).startswith("mix"): continue
        if result.get("rating_before") is None: continue
        games[tag] = (head, result, lines, decs)

print("games:", len(games))
rows = []
for tag, (head, res, lines, decs) in games.items():
    me = None
    for x in lines:
        p = x.split("|")
        if len(p) > 3 and p[1] == "player" and norm(p[3]) == ME: me = p[2]
    if me not in ("p1", "p2"): continue
    opp = "p2" if me == "p1" else "p1"
    boosts = collections.defaultdict(int)        # opponent mon -> net positive boost stages now
    peak_boost = collections.defaultdict(int)
    setup_used = collections.Counter()            # opponent mon -> set-up moves used
    kos_by = collections.Counter()                 # opponent mon -> our faints right after its move
    kos_boosted = collections.Counter()
    our_answers = 0
    last_opp_mover = None
    our_faints = opp_faints = 0
    hazards_on_us = set(); cleared = 0
    our_status = 0
    turn = 0
    first_setup_turn = None
    for x in lines:
        p = x.split("|")
        if len(p) < 2: continue
        k = p[1]
        if k == "turn": turn = int(p[2]); last_opp_mover = None
        elif k == "move" and len(p) > 3:
            who, mv = p[2], norm(p[3])
            if who.startswith(opp):
                last_opp_mover = who.split(": ", 1)[-1]
                if mv in SETUP:
                    setup_used[last_opp_mover] += 1
                    if first_setup_turn is None: first_setup_turn = turn
            elif who.startswith(me) and mv in ANSWERS:
                our_answers += 1
        elif k in ("-boost", "-unboost") and len(p) > 4 and p[2].startswith(opp):
            mon = p[2].split(": ", 1)[-1]; n = int(p[4]) if p[4].lstrip("-").isdigit() else 0
            boosts[mon] += n if k == "-boost" else -n
            peak_boost[mon] = max(peak_boost[mon], boosts[mon])
        elif k in ("-clearallboost", "-clearboost", "-clearpositiveboost") :
            for m in list(boosts): boosts[m] = 0
        elif k == "switch" and len(p) > 2 and p[2].startswith(opp):
            boosts[p[2].split(": ", 1)[-1]] = 0
        elif k == "faint" and len(p) > 2:
            if p[2].startswith(me):
                our_faints += 1
                if last_opp_mover:
                    kos_by[last_opp_mover] += 1
                    if boosts[last_opp_mover] > 0: kos_boosted[last_opp_mover] += 1
            else:
                opp_faints += 1
        elif k == "-sidestart" and len(p) > 3 and p[2].startswith(me):
            hazards_on_us.add(norm(p[3]))
        elif k == "-sideend" and len(p) > 3 and p[2].startswith(me):
            cleared += 1
        elif k == "-status" and len(p) > 3 and p[2].startswith(me):
            our_status += 1
    top_mon, top_kos = (kos_by.most_common(1)[0] if kos_by else (None, 0))
    ests = [d.get("win_est") for d in decs if d.get("win_est") is not None]
    over = [d for d in decs if d.get("played") is not None and d.get("net") is not None and d["played"] != d["net"]]
    rows.append({
        "won": bool(res.get("won")), "turns": res.get("turns"), "opp_rating": res.get("opponent_rating"),
        "rating": res.get("rating_before"), "top_kos": top_kos, "top_boosted": kos_boosted[top_mon] if top_mon else 0,
        "sweeper_setup": setup_used[top_mon] if top_mon else 0, "any_setup": sum(setup_used.values()) > 0,
        "peak_boost": max(peak_boost.values()) if peak_boost else 0, "answers": our_answers,
        "hazards": len(hazards_on_us), "cleared": cleared, "status": our_status,
        "max_est": max(ests) if ests else None, "min_est": min(ests) if ests else None,
        "overruled": len(over), "decisions": len(decs), "first_setup": first_setup_turn,
        "tera_turn": res.get("tera_turn"), "tera_by": res.get("tera_by"),
        "est_series": [(d.get("turn"), d.get("win_est")) for d in decs if d.get("win_est") is not None],
        "over_list": [(d.get("turn"), d.get("win_est"), d.get("p_net"), d.get("p_played")) for d in over],
        "tag": tag,
    })

W = [r for r in rows if r["won"]]; L = [r for r in rows if not r["won"]]
def pct(xs, f): return 100.0 * sum(1 for r in xs if f(r)) / max(1, len(xs))
def mean(xs, k): v = [r[k] for r in xs if r[k] is not None]; return st.mean(v) if v else float("nan")
print(f"wins {len(W)}  losses {len(L)}")
print("\n== how the game was lost (losses) vs the same thing in wins (where it applies)")
print(f"one foe KO'd 3+ of ours:            losses {pct(L, lambda r: r['top_kos']>=3):5.1f}%   wins {pct(W, lambda r: r['top_kos']>=3):5.1f}%")
print(f"  ... and it was boosted for 2+:     losses {pct(L, lambda r: r['top_kos']>=3 and r['top_boosted']>=2):5.1f}%")
print(f"  ... and it had used a set-up move: losses {pct(L, lambda r: r['top_kos']>=3 and r['sweeper_setup']>0):5.1f}%")
print(f"foe reached +2 or more at some point: losses {pct(L, lambda r: r['peak_boost']>=2):5.1f}%   wins {pct(W, lambda r: r['peak_boost']>=2):5.1f}%")
print(f"foe reached +4 or more:               losses {pct(L, lambda r: r['peak_boost']>=4):5.1f}%   wins {pct(W, lambda r: r['peak_boost']>=4):5.1f}%")
print(f"we used a set-up answer (haze, phaze, status, encore, taunt...): mean per game  losses {mean(L,'answers'):.2f}  wins {mean(W,'answers'):.2f}")
print(f"hazards on our side (kinds): losses {mean(L,'hazards'):.2f}  wins {mean(W,'hazards'):.2f};  cleared: losses {mean(L,'cleared'):.2f} wins {mean(W,'cleared'):.2f}")
print(f"our Pokemon statused per game:  losses {mean(L,'status'):.2f}  wins {mean(W,'status'):.2f}")
print(f"turns: losses {mean(L,'turns'):.1f}  wins {mean(W,'turns'):.1f}")
print(f"opponent rating minus ours: losses {st.mean([r['opp_rating']-r['rating'] for r in L if r['opp_rating']]):+.0f}  wins {st.mean([r['opp_rating']-r['rating'] for r in W if r['opp_rating']]):+.0f}")
print("\n== throws: losses where the network once thought it was winning")
for th in (0.6, 0.7, 0.8):
    print(f"  win estimate reached {th:.0%} or more: {pct(L, lambda r: (r['max_est'] or 0) >= th):5.1f}% of losses  (wins that dipped under {1-th:.0%}: {pct(W, lambda r: r['min_est'] is not None and r['min_est'] <= 1-th):5.1f}%)")
print("\n== the search overruling the network")
print(f"overruled moves per game: losses {mean(L,'overruled'):.2f}  wins {mean(W,'overruled'):.2f}  (decisions {mean(L,'decisions'):.1f} / {mean(W,'decisions'):.1f})")
# after an overrule, does the network's own win estimate at the next decision go up or down, compared with after its own move?
def deltas(rs, overruled):
    out = []
    for r in rs:
        s = dict(r["est_series"]); turns = sorted(s)
        ov_turns = {t for t, *_ in r["over_list"]}
        for i, t in enumerate(turns[:-1]):
            if (t in ov_turns) == overruled and s[t] is not None and s[turns[i+1]] is not None:
                out.append(s[turns[i+1]] - s[t])
    return out
for name, rs in (("all", rows),):
    a, b = deltas(rs, True), deltas(rs, False)
    print(f"next-turn change in the network's win estimate: after an overrule {st.mean(a):+.4f} (n {len(a)}),  after the network's own move {st.mean(b):+.4f} (n {len(b)})")
lowp = [x for r in rows for x in r["over_list"] if x[3] is not None and x[3] < 0.02]
print(f"overrules to a move the network gave under 2%: {len(lowp)} of {sum(len(r['over_list']) for r in rows)}")
print("\n== Tera")
for name, xs in (("losses", L), ("wins", W)):
    tt = [r["tera_turn"] for r in xs if r["tera_turn"]]
    print(f"  {name}: Tera used in {100*len(tt)/max(1,len(xs)):.0f}%, median turn {st.median(tt) if tt else '-'}, by {collections.Counter(r['tera_by'] for r in xs if r['tera_turn']).most_common(3)}")
print("\n== when the losses turned: first turn the win estimate stayed under 30%")
turn_pts = []
for r in L:
    s = sorted((t, e) for t, e in r["est_series"] if e is not None)
    for i, (t, e) in enumerate(s):
        if e < 0.3 and all(e2 < 0.3 for _, e2 in s[i:]):
            turn_pts.append((t, r["turns"], r["first_setup"])); break
if turn_pts:
    print(f"  median turn {st.median([t for t,_,_ in turn_pts])} of median game length {st.median([n for _,n,_ in turn_pts])};  "
          f"a foe had used set-up before that turn in {100*sum(1 for t,_,fs in turn_pts if fs is not None and fs <= t)/len(turn_pts):.0f}% of them")

print("\n== overrules against the network's own move, positions matched by win estimate (next decision's estimate minus this one's)")
pairs = collections.defaultdict(lambda: {"over": [], "own": [], "over_low": []})
for tag, (head, res, lines, decs) in games.items():
    ds = [d for d in decs if d.get("win_est") is not None and d.get("played") is not None and d.get("net") is not None]
    for a, b in zip(ds, ds[1:]):
        bucket = min(9, int(a["win_est"] * 10))
        delta = b["win_est"] - a["win_est"]
        if a["played"] != a["net"]:
            pairs[bucket]["over"].append(delta)
            if (a.get("p_played") or 0) < 0.02: pairs[bucket]["over_low"].append(delta)
        else:
            pairs[bucket]["own"].append(delta)
tot = {"over": [], "own": [], "over_low": []}
for bkt in sorted(pairs):
    p = pairs[bkt]
    if len(p["over"]) < 30: continue
    print(f"  estimate {bkt*10:2d}-{bkt*10+10:3d}%: own move {st.mean(p['own'])*100:+.2f} pts (n {len(p['own'])}), overruled {st.mean(p['over'])*100:+.2f} (n {len(p['over'])}), overruled to a move the net gave <2% {st.mean(p['over_low'])*100 if p['over_low'] else float('nan'):+.2f} (n {len(p['over_low'])})")
    # weight the comparison by the overrules' own distribution
    for k in tot: tot[k].append((len(p["over"]), st.mean(p[k]) if p[k] else 0.0))
w = sum(n for n, _ in tot["over"])
print(f"  weighted by where overrules happen: own move {sum(n*m for (n,_),(_,m) in zip(tot['over'], tot['own']))/w*100:+.2f} pts, overruled {sum(n*m for n,m in tot['over'])/w*100:+.2f}, overruled to <2% {sum(n*m for (n,_),(_,m) in zip(tot['over'], tot['over_low']))/w*100:+.2f}")

print("\n== who sweeps us: the foe with the most KOs in each loss where it got 3+")
sweep = collections.Counter(); boosted = collections.Counter()
for tag, (head, res, lines, decs) in games.items():
    if res.get("won"): continue
    me = next((x.split("|")[2] for x in lines if x.startswith("|player|") and len(x.split("|")) > 3 and norm(x.split("|")[3]) == ME), None)
    if me not in ("p1", "p2"): continue
    opp = "p2" if me == "p1" else "p1"
    last, kos, species, bo = None, collections.Counter(), {}, collections.defaultdict(int)
    for x in lines:
        p = x.split("|")
        if len(p) < 3: continue
        if p[1] == "turn": last = None
        elif p[1] == "switch" and p[2].startswith(opp): species[p[2].split(": ",1)[-1]] = p[3].split(",")[0]; bo[p[2].split(": ",1)[-1]] = 0
        elif p[1] == "move" and p[2].startswith(opp): last = p[2].split(": ",1)[-1]
        elif p[1] == "-boost" and p[2].startswith(opp) and len(p) > 4 and p[4].isdigit(): bo[p[2].split(": ",1)[-1]] += int(p[4])
        elif p[1] == "faint" and p[2].startswith(me) and last: kos[last] += 1
    if kos:
        mon, n = kos.most_common(1)[0]
        if n >= 3:
            sp = species.get(mon, mon); sweep[sp] += 1
            if bo[mon] > 0: boosted[sp] += 1
print("  " + ", ".join(f"{sp} {n}" + (f" ({boosted[sp]} boosted)" if boosted[sp] else "") for sp, n in sweep.most_common(18)))
