"""Tables from games.json (games.py). Prints plain text; light work only.

    python bench/ladder_log/analyse.py games.json
"""
import collections
import json
import math
import re
import statistics
import sys

games = json.load(open(sys.argv[1], encoding="utf-8"))
MIX = ("mix", "mix-lines")


def wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def rec(sub):
    w = sum(g["won"] for g in sub)
    n = len(sub)
    lo, hi = wilson(w, n)
    return "%3d-%-3d %5.1f%% [%4.1f-%4.1f] n=%d" % (w, n - w, 100.0 * w / max(1, n), 100 * lo, 100 * hi, n)


def head(t):
    print("\n## " + t)


head("by mode and network")
order = []
for g in games:
    k = (g["arm"], g["checkpoint"])
    if k not in order:
        order.append(k)
for k in order:
    sub = [g for g in games if (g["arm"], g["checkpoint"]) == k]
    rb = [g["rating_before"] for g in sub if g["rating_before"]]
    ra = [g["rating_after"] for g in sub if g["rating_after"]]
    opp = [g["opp_rating"] for g in sub if g["opp_rating"]]
    exp = []
    for g in sub:
        if g["rating_before"] and g["opp_rating"]:
            exp.append(1 / (1 + 10 ** ((g["opp_rating"] - g["rating_before"]) / 400)))
    print("%-11s %-26s %s | %s .. %s | rating %s->%s peak %s | opp mean %d | Elo-expected %.1f%% | turns %.1f" % (
        k[0], k[1], rec(sub), sub[0]["time"][:16], sub[-1]["time"][11:16], rb[0] if rb else None, ra[-1] if ra else None,
        max(ra) if ra else None, statistics.mean(opp) if opp else 0, 100 * statistics.mean(exp) if exp else 0,
        statistics.mean(g["turns"] or 0 for g in sub)))

mix = [g for g in games if g["arm"] in MIX]
solo = [g for g in games if str(g["arm"]).endswith("-solo")]
withllm = [g for g in games if g["arm"] in ("sol-lines", "luna")]
print("\nnetwork+search (mix, mix-lines):", rec(mix))
print("network+search with an LLM on top (sol-lines, luna):", rec(withllm))
print("an LLM alone (sol-solo, fable-solo):", rec(solo))


def table(title, sub, keyf, order=None):
    head(title)
    c = collections.defaultdict(list)
    for g in sub:
        c[keyf(g)].append(g)
    for k in (order or sorted(c, key=str)):
        if k in c:
            print("  %-34s %s" % (k, rec(c[k])))


def tera_bucket(g):
    t = g["tera"]["us"]
    if not t:
        return "no Tera"
    n = t[0]
    return "Tera turn 1-3" if n <= 3 else "Tera turn 4-8" if n <= 8 else "Tera turn 9-15" if n <= 15 else "Tera turn 16+"


for name, sub in (("network+search", mix), ("LLM alone", solo)):
    table(name + ": our Tera timing", sub, tera_bucket, ["no Tera", "Tera turn 1-3", "Tera turn 4-8", "Tera turn 9-15", "Tera turn 16+"])
tt = [g["tera"]["us"][0] for g in mix if g["tera"]["us"]]
print("  our Tera turn: mean %.1f median %.1f; %d%% of games by turn 3; top-player reference: mean 19, 2%% by turn 2" % (
    statistics.mean(tt), statistics.median(tt), 100 * sum(t <= 3 for t in tt) / len(tt)))
table("network+search: foe Terastallized?", mix, lambda g: "foe Tera" if g["tera"]["foe"] else "foe no Tera")
table("network+search: who lost a Pokemon first", mix, lambda g: "we fainted first" if g["first_faint"] == "us" else "foe fainted first" if g["first_faint"] == "foe" else "none")
table("network+search: game length", mix, lambda g: "1-15 turns" if g["turns"] <= 15 else "16-25 turns" if g["turns"] <= 25 else "26-40 turns" if g["turns"] <= 40 else "41+ turns",
      ["1-15 turns", "16-25 turns", "26-40 turns", "41+ turns"])


def diff_bucket(g):
    if not (g["rating_before"] and g["opp_rating"]):
        return "unknown"
    d = g["opp_rating"] - g["rating_before"]
    return "foe 100+ lower" if d < -100 else "foe 0-100 lower" if d < 0 else "foe 0-100 higher" if d < 100 else "foe 100+ higher"


table("network+search: foe rating against ours", mix, diff_bucket, ["foe 100+ lower", "foe 0-100 lower", "foe 0-100 higher", "foe 100+ higher", "unknown"])
table("network+search: our rating band", mix, lambda g: "under 2000" if (g["rating_before"] or 0) < 2000 else "2000-2199" if g["rating_before"] < 2200 else "2200-2349" if g["rating_before"] < 2350 else "2350+",
      ["under 2000", "2000-2199", "2200-2349", "2350+"])

# sweeps: one foe Pokemon KOs 2+ / 3+ of ours
head("sweeps: one foe Pokemon knocks out several of ours")


def foe_multi(g, n, boosted=None):
    c = collections.Counter()
    b = collections.defaultdict(int)
    for t, side, killer, move, victim, kb in g["kos"]:
        if side == "foe":
            c[killer] += 1
            b[killer] = max(b[killer], kb)
    return any(v >= n and (boosted is None or (b[k] > 0) == boosted) for k, v in c.items())


for name, sub in (("network+search", mix), ("LLM alone", solo)):
    L = [g for g in sub if not g["won"]]
    W = [g for g in sub if g["won"]]
    for n in (2, 3):
        print("  %-15s a foe Pokemon KOs %d+ of ours: losses %d%% (%d of %d), wins %d%%" % (
            name, n, 100 * sum(foe_multi(g, n) for g in L) / max(1, len(L)), sum(foe_multi(g, n) for g in L), len(L),
            100 * sum(foe_multi(g, n) for g in W) / max(1, len(W))))
    print("  %-15s ... and it was boosted (atk/spa/spe above 0) when it did: losses %d%% (%d), wins %d%%" % (
        name, 100 * sum(foe_multi(g, 2, True) for g in L) / max(1, len(L)), sum(foe_multi(g, 2, True) for g in L),
        100 * sum(foe_multi(g, 2, True) for g in W) / max(1, len(W))))
    print("  %-15s our KOs by a boosted Pokemon of ours: %d of %d; foe KOs by a boosted foe: %d of %d" % (
        name, sum(1 for g in sub for k in g["kos"] if k[1] == "us" and k[5] > 0), sum(1 for g in sub for k in g["kos"] if k[1] == "us"),
        sum(1 for g in sub for k in g["kos"] if k[1] == "foe" and k[5] > 0), sum(1 for g in sub for k in g["kos"] if k[1] == "foe")))
    print("  %-15s set-up moves used per game: us %.2f, foe %.2f (in our losses: us %.2f, foe %.2f)" % (
        name, statistics.mean(g["setups"]["us"] for g in sub), statistics.mean(g["setups"]["foe"] for g in sub),
        statistics.mean(g["setups"]["us"] for g in L) if L else 0, statistics.mean(g["setups"]["foe"] for g in L) if L else 0))

head("network+search: foe species that beat us (on the foe team in 6+ games), lowest win rate first")
app = collections.defaultdict(list)
kos_by = collections.Counter()
for g in mix:
    for sp in set(g["team"]["foe"]):
        app[sp].append(g)
    for k in g["kos"]:
        if k[1] == "foe":
            kos_by[k[2]] += 1
rows = sorted(((sum(x["won"] for x in v) / len(v), sp, v) for sp, v in app.items() if len(v) >= 6))
base = sum(g["won"] for g in mix) / len(mix)
for wr, sp, v in rows[:22]:
    print("  %-22s %s  KOs of ours per game seen %.2f" % (sp, rec(v), kos_by[sp] / len(v)))
head("network+search: foe Pokemon with the most KOs of ours per game seen (6+ games)")
for sp, v in sorted(app.items(), key=lambda kv: -kos_by[kv[0]] / len(kv[1])):
    if len(v) >= 6 and kos_by[sp] / len(v) >= 1.0:
        print("  %-22s %.2f KOs per game, seen %d, our record %s" % (sp, kos_by[sp] / len(v), len(v), rec(v)))

head("network+search: our species with the lowest win rate (6+ games)")
ours = collections.defaultdict(list)
for g in mix:
    for sp in set(g["team"]["us"]):
        ours[sp].append(g)
for wr, sp, v in sorted(((sum(x["won"] for x in v) / len(v), sp, v) for sp, v in ours.items() if len(v) >= 6))[:15]:
    print("  %-22s %s" % (sp, rec(v)))

head("network+search: hazards and status")
table("Stealth Rock on our side", mix, lambda g: "rocks on us" if g["hazards"]["us"].get("stealthrock") else "no rocks on us")
table("Stealth Rock on their side", mix, lambda g: "rocks on them" if g["hazards"]["foe"].get("stealthrock") else "no rocks on them")
table("any spikes / toxic spikes / web on us", mix, lambda g: "yes" if any(k in g["hazards"]["us"] for k in ("spikes", "toxicspikes", "stickyweb")) else "no")
for st in ("slp", "par", "brn", "tox", "psn"):
    table("%s on one of ours" % st, mix, lambda g, st=st: "yes" if g["status"]["us"].get(st) else "no")

head("network+search: the value estimate (win_est) against the result")
buckets = collections.defaultdict(lambda: [0, 0])
for g in mix:
    for d in g["decisions"]:
        if d["win_est"] is None or d["forced"]:
            continue
        b = min(9, int(d["win_est"] * 10))
        buckets[b][0] += g["won"]
        buckets[b][1] += 1
for b in sorted(buckets):
    w, n = buckets[b]
    print("  win_est %.1f-%.1f: games won from here %5.1f%% of %d decisions" % (b / 10, (b + 1) / 10, 100 * w / n, n))
thrown = []
comeback = []
for g in mix:
    est = [(d["turn"], d["win_est"]) for d in g["decisions"] if d["win_est"] is not None and not d["forced"]]
    if not est:
        continue
    hi = max(est, key=lambda x: x[1])
    lo = min(est, key=lambda x: x[1])
    hi = max([e for e in est if (e[0] or 0) >= 6] or [(0, 0.0)], key=lambda x: x[1])
    if not g["won"] and hi[1] >= 0.75:
        thrown.append((g, hi))
    if g["won"] and lo[1] <= 0.25:
        comeback.append((g, lo))
L = [g for g in mix if not g["won"]]
print("  (thrown: win_est 0.75+ on turn 6 or later)")
print("  losses where we were at win_est 0.75+ at some point: %d of %d (%d%%)" % (len(thrown), len(L), 100 * len(thrown) / len(L)))
print("  wins from win_est 0.25 or lower: %d of %d" % (len(comeback), len(mix) - len(L)))
for g, hi in sorted(thrown, key=lambda x: -x[1][1])[:40]:
    after = [k for k in g["kos"] if k[0] >= hi[0] and k[1] == "foe"]
    print("   THROWN %s %s %s turn %s est %.2f -> lost turn %s; foe KOs after: %s" % (
        re.sub(r"^(battle-[a-z0-9]+-\d+)-.*$", r"\1", g["battle"]), g["arm"], g["checkpoint"][:22], hi[0], hi[1], g["turns"],
        "; ".join("%s KO %s (%s)%s" % (k[2], k[4], k[3], " +%d" % k[5] if k[5] else "") for k in after)[:300]))

head("network+search: what the decision rule played")
src = collections.Counter()
third = 0
nondec = 0
low = 0
sw = 0
for g in mix:
    for d in g["decisions"]:
        if d["forced"] or (d["legal"] or 0) <= 1:
            continue
        nondec += 1
        src[d["source"]] += 1
        if d["played"] not in (d["net"], d["search"]):
            third += 1
        if d["p_played"] is not None and d["p_played"] < 0.05:
            low += 1
        if d["played"] is not None and 4 <= d["played"] <= 9:
            sw += 1
print("  free decisions %d; source %s" % (nondec, dict(src)))
print("  network and search disagreed: %d%%" % (100 * sum(1 for g in mix for d in g["decisions"] if d["disagree"] and not d["forced"]) / nondec))
print("  played an action that was neither the network's nor the search's first choice: %d (%.1f%%)" % (third, 100 * third / nondec))
print("  played an action the network gave under 5%%: %d (%.1f%%)" % (low, 100 * low / nondec))
print("  switched on %.1f%% of free decisions (top-player reference: 23%% of turns)" % (100 * sw / nondec))
# do 'third option' games lose more?
tw = collections.defaultdict(lambda: [0, 0])
for g in mix:
    n3 = sum(1 for d in g["decisions"] if not d["forced"] and (d["legal"] or 0) > 1 and d["played"] not in (d["net"], d["search"]))
    k = "0" if n3 == 0 else "1-2" if n3 <= 2 else "3-5" if n3 <= 5 else "6+"
    tw[k][0] += g["won"]
    tw[k][1] += 1
for band in ((1, 20), (21, 30), (31, 999)):
    sub = [g for g in mix if band[0] <= (g["turns"] or 0) <= band[1]]
    hi_, lo_ = [], []
    for g in sub:
        fr = [d for d in g["decisions"] if not d["forced"] and (d["legal"] or 0) > 1]
        if not fr:
            continue
        share = sum(d["played"] not in (d["net"], d["search"]) for d in fr) / len(fr)
        (hi_ if share >= 0.08 else lo_).append(g)
    print("  games of %d-%d turns: neither-choice share 8%%+ %s | under 8%% %s" % (band[0], min(band[1], 99), rec(hi_), rec(lo_)))
for k in ("0", "1-2", "3-5", "6+"):
    if k in tw:
        print("  games with %s such moves: %d-%d (%.0f%%)" % (k, tw[k][0], tw[k][1] - tw[k][0], 100 * tw[k][0] / tw[k][1]))

head("network+search: the opponent model's predictions")
P = [p for g in mix for p in g["predictions"]]
print("  predictions scored %d; top-1 right %.1f%%; foe switched on %.1f%% of them" % (
    len(P), 100 * sum(p["top1_right"] for p in P) / len(P), 100 * sum(p["switched"] for p in P) / len(P)))
cb = collections.defaultdict(lambda: [0, 0, 0.0])
for p in P:
    if p["p_switch"] is None:
        continue
    b = min(4, int(p["p_switch"] * 5))
    cb[b][0] += p["switched"]
    cb[b][1] += 1
    cb[b][2] += p["p_switch"]
for b in sorted(cb):
    s, n, ps = cb[b]
    print("  predicted switch %.1f-%.1f (mean %.2f): foe switched %.1f%% of %d" % (b / 5, (b + 1) / 5, ps / n, 100 * s / n, n))
sw_pred = [p for p in P if p["switched"]]
print("  when the foe did switch, top-1 said switch %.1f%% of the time" % (100 * sum(str(p["top1"]).startswith("switch") for p in sw_pred) / max(1, len(sw_pred))))

head("LLM modes: cost, speed, fallbacks")
for arm in ("sol-lines", "luna", "sol-solo", "fable-solo"):
    sub = [g for g in games if g["arm"] == arm]
    if not sub:
        continue
    cost = sum((g["result"].get("luna_cost_usd") or 0) for g in sub)
    lat = [g["result"]["luna_latency_ms_mean"] for g in sub if g["result"].get("luna_latency_ms_mean")]
    fb = sum((sum(v.values()) if isinstance(v, dict) else (v or 0)) for v in (g["result"].get("luna_fallbacks") for g in sub))
    dec = sum((g["result"].get("luna_decisions") or 0) for g in sub)
    print("  %-11s %s cost $%.2f ($%.3f a game) latency mean %.1f s, fallbacks %d of %d decisions, model %s" % (
        arm, rec(sub), cost, cost / len(sub), statistics.mean(lat) / 1000 if lat else 0, fb, dec,
        collections.Counter(g["result"].get("luna_model") for g in sub).most_common(2)))
for name, sub in (("LLM alone", solo),):
    table(name + ": who lost a Pokemon first", sub, lambda g: "we fainted first" if g["first_faint"] == "us" else "foe fainted first")
    L = [g for g in sub if not g["won"]]
    print("  LLM alone, Pokemon left at the end of a loss (foe's): mean %.1f; network+search losses: %.1f" % (
        statistics.mean(g["left"]["foe"] for g in L), statistics.mean(g["left"]["foe"] for g in mix if not g["won"])))
    sw = sum(1 for g in sub for d in g["decisions"] if not d["forced"] and d["played"] is not None and 4 <= d["played"] <= 9)
    nd = sum(1 for g in sub for d in g["decisions"] if not d["forced"] and (d["legal"] or 0) > 1)
    print("  LLM alone switched on %.1f%% of free decisions" % (100 * sw / max(1, nd)))
    print("  LLM alone set-up moves per game %.2f (foe %.2f)" % (statistics.mean(g["setups"]["us"] for g in sub), statistics.mean(g["setups"]["foe"] for g in sub)))
