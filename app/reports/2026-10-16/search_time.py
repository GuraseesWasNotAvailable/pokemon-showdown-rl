"""
Does a longer search make the ladder bot worse? Every recorded public ladder game of the network
with the search, by the window app's thinking-time setting. The numbers of
reports/2026-10-16/SEARCH_TIME.md.

    python reports/2026-10-16/search_time.py

Read-only: it reads ladder.py's records (results.jsonl and runs/ladder/battles/*.jsonl) of the
owner's laptop's two folders and prints tables. One process, no network, nothing written.

The parsing of the battle records is loss_study.py's (reports/2026-10-15): the "note" lines carry
the game's head, its result and one "decision" per order (the network's choice and probability,
the search's choice, the move played, the network's own win estimate), and the "in" lines carry
the battle log, here read for the battle timer ("Time left: N sec this turn | M sec total") and
the end of the game ("forfeited", "lost due to inactivity").

The thinking-time setting is the game's search_ms (results.jsonl and every decision's asked_ms):
the app's 1 s = 500 ms, 2 s = 900 ms, 5 s = 2200 ms (app/PSBotLookahead.cs, ThinkSearchMs), with a
deadline of 2.5, 2.7 and 6.0 s from the request to the order (ThinkDeadline).
"""
import collections, glob, json, math, os, re, statistics as st, sys

import numpy as np

FOLDERS = [r"C:\Users\you\psbot-local\runs\ladder", r"C:\Users\you\ps-ppo-search\runs\ladder"]
ME = "yourshowdownname"
SETTING = {500: "1 s (500 ms)", 900: "2 s (900 ms)", 2200: "5 s (2200 ms)"}
DEADLINE = {500: 2.5, 900: 2.7, 2200: 6.0}
VETO_MODES = ("mix", "mix-trust")            # the network + search (veto rule), no lines, no model
ELO = 400.0 / math.log(10.0)                 # Elo points per unit of log-odds
RNG = np.random.default_rng(20261016)
BOOT = 2000


def norm(s): return re.sub(r"[^a-z0-9]", "", (s or "").lower())
def base(a): return a - 10 if a is not None and a >= 10 else a


def net_name(ck):
    ck = (ck or "?").replace(".pt", "")
    return {"learner_update_011850": "main 011850 (ladder best)", "learner_update_024381": "main 024381",
            "armC-full_update_011525_426M": "arm C 426M", "armC-full_update_012275_454M": "arm C 454M",
            "v2-setupw2_update_024150_56M": "v2 setupw2 56M", "v2-setupw2_update_024266_60M": "v2 setupw2 60M"}.get(ck, ck)


# ---------------------------------------------------------------- loading
def load():
    games, seen = [], set()
    for folder in FOLDERS:
        path = os.path.join(folder, "results.jsonl")
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            tag = r["battle"]
            if tag in seen:
                continue
            seen.add(tag)
            if r.get("server") != "public" or norm(r.get("user")) != ME or r.get("won") is None:
                continue
            g = dict(r)
            g["folder"] = os.path.basename(os.path.dirname(folder))
            g["net"] = net_name(r.get("checkpoint"))
            g.update(read_battle(os.path.join(folder, "battles", tag + ".jsonl")))
            games.append(g)
    games.sort(key=lambda g: g["time"])
    return games


def read_battle(path):
    out = {"decs": [], "outcomes": [], "timer_total": [], "timer_turn": [], "end_msgs": [], "have_file": False}
    if not os.path.exists(path):
        return out
    out["have_file"] = True
    lines = []
    for l in open(path, encoding="utf-8"):
        try:
            r = json.loads(l)
        except Exception:
            continue
        d = r.get("data")
        if r.get("dir") == "in" and isinstance(d, str):
            lines.extend(d.split("\n"))
        elif r.get("dir") == "note" and isinstance(d, dict):
            if "decision" in d:
                out["decs"].append(d["decision"])
            if "outcome" in d:
                o = (d["outcome"].get("predict") or {}).get("outcome")
                if o and o.get("p_true") is not None:
                    out["outcomes"].append(o)
    for x in lines:
        if x.startswith("|inactive|Time left:"):
            m = re.search(r"Time left: (\d+) sec this turn \| (\d+) sec total", x)
            if m:
                out["timer_turn"].append(int(m.group(1)))
                out["timer_total"].append(int(m.group(2)))
        elif x.startswith("|-message|") and ("forfeited." in x or "lost due to inactivity." in x):
            out["end_msgs"].append(x[len("|-message|"):])
    return out


def ending(g):
    """forfeit_them, forfeit_us, inactivity_them, inactivity_us, or played (from the log, since 'ending' is only in newer results)."""
    for m in g["end_msgs"]:
        who_me = norm(m.split(" forfeited")[0].split(" lost due")[0]) == ME
        kind = "forfeit" if "forfeited" in m else "inactivity"
        return kind + ("_us" if who_me else "_them")
    return "played"


# ---------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe(k1, n1, k2, n2):
    """95% interval of p1 - p2 (Newcombe's hybrid score method)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def t_ci(xs):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n < 2:
        return (st.mean(xs) if xs else float("nan")), float("nan"), float("nan")
    m, s = st.mean(xs), st.stdev(xs)
    h = 1.96 * s / math.sqrt(n)     # n >= 20 everywhere it is used
    return m, m - h, m + h


def perf_rating(opps, wins):
    """The Elo strength that makes the expected score against these opponents equal the score made,
    with its standard error from the Fisher information (only the opponents' ratings are used, so a
    climbing account's lagging own rating does not bias it)."""
    opps = np.asarray(opps, float)
    w = float(sum(wins))
    n = len(opps)
    if n == 0:
        return float("nan"), float("nan")
    lo, hi = 0.0, 4000.0
    if w <= 0 or w >= n:
        return (lo if w <= 0 else hi), float("inf")
    for _ in range(100):
        mid = (lo + hi) / 2
        e = (1 / (1 + 10 ** ((opps - mid) / 400))).sum()
        lo, hi = (mid, hi) if e < w else (lo, mid)
    th = (lo + hi) / 2
    p = 1 / (1 + 10 ** ((opps - th) / 400))
    return th, ELO / math.sqrt((p * (1 - p)).sum())


def logit_offset(X, y, off, iters=50):
    """Logistic regression with a fixed offset (IRLS). Returns beta and its standard errors."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ b + off
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        g = X.T @ (y - p)
        step = np.linalg.solve(H, g)
        b = b + step
        if np.abs(step).max() < 1e-9:
            break
    cov = np.linalg.inv(H)
    return b, np.sqrt(np.diag(cov))


def boot_ratio(num, den, reps=BOOT):
    """Cluster bootstrap (over games) of sum(num)/sum(den): point and 95% interval, plus the draws."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    n = len(num)
    if n == 0 or den.sum() == 0:
        return float("nan"), float("nan"), float("nan"), np.array([np.nan])
    idx = RNG.integers(0, n, size=(reps, n))
    draws = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return num.sum() / den.sum(), np.percentile(draws, 2.5), np.percentile(draws, 97.5), draws


def fmt_ci(m, lo, hi, scale=100.0, dec=1, unit=""):
    return f"{m*scale:.{dec}f}{unit} ({lo*scale:.{dec}f} to {hi*scale:.{dec}f})"


def diff_ci(d1, d2, scale=100.0, dec=1):
    d = d1 - d2
    return f"{np.nanmean(d)*scale:+.{dec}f} ({np.nanpercentile(d, 2.5)*scale:+.{dec}f} to {np.nanpercentile(d, 97.5)*scale:+.{dec}f})"


# ---------------------------------------------------------------- per-decision helpers
def searched(d):
    """A decision the search answered and mix decided (not forced single, not a fallback, not after a reconnect)."""
    return d.get("source") == "mix" and d.get("asked") and not d.get("reason")


def overruled(d):
    if d.get("net_overruled") is not None:
        return bool(d["net_overruled"])
    return base(d.get("played")) != base(d.get("net"))


def sound_count(d, keep=0.25):
    sh = d.get("shares")
    if not sh:
        return None
    mx = max(sh)
    return sum(1 for s in sh if s > 0 and s >= keep * mx)


def entropy(sh):
    s = [x for x in sh if x > 0]
    tot = sum(s)
    return -sum((x / tot) * math.log(x / tot) for x in s) if tot > 0 else None


# ---------------------------------------------------------------- tables
def outcome_row(name, gs):
    n = len(gs)
    w = sum(1 for g in gs if g["won"])
    lo, hi = wilson(w, n)
    dr = [g["rating_after"] - g["rating_before"] for g in gs if g.get("rating_after") is not None and g.get("rating_before") is not None]
    m, a, b = t_ci(dr)
    exp = [g["won"] - 1 / (1 + 10 ** ((g["opponent_rating"] - g["rating_before"]) / 400)) for g in gs
           if g.get("opponent_rating") and g.get("rating_before")]
    em, ea, eb = t_ci(exp)
    pr, se = perf_rating([g["opponent_rating"] for g in gs if g.get("opponent_rating")],
                         [g["won"] for g in gs if g.get("opponent_rating")])
    opp = st.mean([g["opponent_rating"] for g in gs if g.get("opponent_rating")])
    first = gs[0].get("rating_before"); last = gs[-1].get("rating_after")
    return (f"| {name} | {n} | {w}-{n-w} | {100*w/n:.1f}% ({100*lo:.1f} to {100*hi:.1f}) | {opp:.0f} | "
            f"{pr:.0f} +/- {1.96*se:.0f} | {100*em:+.1f} ({100*ea:+.1f} to {100*eb:+.1f}) | "
            f"{m:+.1f} ({a:+.1f} to {b:+.1f}) | {sum(dr):+d} |")


OUT_HEAD = ("| Group | Games | W-L | Win rate (95% Wilson) | Opp. rating (mean) | Performance rating (95%) | "
            "Score minus Elo expectation, pts (95%) | Rating change per game (95%) | Total rating change |\n"
            "|---|---|---|---|---|---|---|---|---|")


def main():
    games = load()
    print(f"loaded {len(games)} public ladder games of {ME} ({sum(g['have_file'] for g in games)} with a battle file)")
    for g in games:
        g["ending"] = ending(g)
        ds = g["decs"]
        g["n_dec"] = len(ds)
        g["n_searched"] = sum(1 for d in ds if searched(d))
        g["asked_ms"] = sorted({d.get("asked_ms") for d in ds if d.get("asked_ms")})
    veto = [g for g in games if g.get("arm") in VETO_MODES and g.get("search_ms") and g.get("rating_before")]
    # the setting recorded per decision agrees with the game's
    bad = [g["battle"] for g in veto if g["asked_ms"] and g["asked_ms"] != [g["search_ms"]]]
    print(f"veto-rule games (mix, mix-trust) with a rating: {len(veto)}; games whose decisions' asked_ms differs from the game's search_ms: {len(bad)}")
    by_set = collections.defaultdict(list)
    for g in veto:
        by_set[g["search_ms"]].append(g)
    print("settings seen (all search modes):", collections.Counter((g.get("arm"), g.get("search_ms")) for g in games))

    # ============================================================ 1. outcomes
    print("\n## 1. Outcomes\n")
    print("### By setting (veto rule: mix and mix-trust)\n")
    print(OUT_HEAD)
    for ms in sorted(by_set):
        print(outcome_row(SETTING[ms], by_set[ms]))
    full = {ms: [g for g in gs if g["n_dec"] and g["n_searched"] >= 0.5 * g["n_dec"]] for ms, gs in by_set.items()}
    for ms in sorted(full):
        print(outcome_row(SETTING[ms] + ", only games the search played (>= 50% of decisions)", full[ms]))
    k1, n1 = sum(g["won"] for g in by_set[2200]), len(by_set[2200])
    k2, n2 = sum(g["won"] for g in by_set[500]), len(by_set[500])
    d, lo, hi = newcombe(k1, n1, k2, n2)
    print(f"\nwin rate 5 s minus 1 s, all veto games: {100*d:+.1f} pts (95% {100*lo:+.1f} to {100*hi:+.1f})")

    print("\n### By setting and mode\n")
    print(OUT_HEAD)
    srch = [g for g in games if g.get("search_ms") and g.get("rating_before")]
    for key, gs in sorted(group(srch, lambda g: (g["search_ms"], g["arm"])).items()):
        print(outcome_row(f"{SETTING[key[0]]}, {key[1]}", gs))

    print("\n### By setting and network (veto rule)\n")
    print(OUT_HEAD)
    for key, gs in sorted(group(veto, lambda g: (g["net"], g["search_ms"], g["arm"])).items()):
        print(outcome_row(f"{key[0]}, {SETTING[key[1]]}, {key[2]}", gs))

    print("\n### Same network at both settings\n")
    both = [n for n in {g["net"] for g in veto} if {g["search_ms"] for g in veto if g["net"] == n} >= {500, 2200}]
    for n in sorted(both):
        a = [g for g in veto if g["net"] == n and g["search_ms"] == 2200]
        b = [g for g in veto if g["net"] == n and g["search_ms"] == 500]
        d, lo, hi = newcombe(sum(g["won"] for g in a), len(a), sum(g["won"] for g in b), len(b))
        ar, br = [g for g in a if g.get("opponent_rating")], [g for g in b if g.get("opponent_rating")]
        pa, sa = perf_rating([g["opponent_rating"] for g in ar], [g["won"] for g in ar])
        pb, sb = perf_rating([g["opponent_rating"] for g in br], [g["won"] for g in br])
        print(f"- {n}: 5 s {sum(g['won'] for g in a)}-{len(a)-sum(g['won'] for g in a)}, 1 s {sum(g['won'] for g in b)}-{len(b)-sum(g['won'] for g in b)};"
              f" win rate 5 s minus 1 s {100*d:+.1f} pts (95% {100*lo:+.1f} to {100*hi:+.1f});"
              f" performance rating {pa:.0f} vs {pb:.0f}, difference {pa-pb:+.0f} (95% {pa-pb-1.96*math.hypot(sa,sb):+.0f} to {pa-pb+1.96*math.hypot(sa,sb):+.0f})")
        a2 = [g for g in a if g["arm"] == "mix"]
        if len(a2) != len(a):
            d, lo, hi = newcombe(sum(g["won"] for g in a2), len(a2), sum(g["won"] for g in b), len(b))
            print(f"  (mix only at 5 s, {len(a2)} games: {100*d:+.1f} pts, 95% {100*lo:+.1f} to {100*hi:+.1f})")

    print("\n### Elo model: logit P(win) = (theta_network + delta x [5 s] + tau x [mix-trust] - opponent rating) / 173.7\n")
    rated = [g for g in veto if g.get("opponent_rating")]
    for label, pool in (("veto games, all", rated), ("veto games the search played", [g for ms in full for g in full[ms] if g.get("opponent_rating")])):
        nets = sorted({g["net"] for g in pool})
        X, y, off = [], [], []
        for g in pool:
            row = [1.0 if g["net"] == n else 0.0 for n in nets] + [1.0 if g["search_ms"] == 2200 else 0.0,
                                                                 1.0 if g["arm"] == "mix-trust" else 0.0]
            X.append(row); y.append(1.0 if g["won"] else 0.0); off.append(-(g["opponent_rating"] - 2250) / ELO)
        b, se = logit_offset(np.array(X), np.array(y), np.array(off))
        dl, ds_ = b[len(nets)] * ELO, se[len(nets)] * ELO
        tl, ts = b[len(nets) + 1] * ELO, se[len(nets) + 1] * ELO
        print(f"- {label} ({len(pool)} games, {len(nets)} networks): delta (5 s) {dl:+.0f} Elo (95% {dl-1.96*ds_:+.0f} to {dl+1.96*ds_:+.0f});"
              f" tau (mix-trust) {tl:+.0f} (95% {tl-1.96*ts:+.0f} to {tl+1.96*ts:+.0f})."
              f" At an even match, delta is {100*(1/(1+10**(-dl/400))-0.5):+.1f} win-rate pts"
              f" (95% {100*(1/(1+10**(-(dl-1.96*ds_)/400))-0.5):+.1f} to {100*(1/(1+10**(-(dl+1.96*ds_)/400))-0.5):+.1f}).")
        # without the networks seen at one setting only: the within-network comparison alone
    sub = [g for g in rated if g["net"] in both]
    nets = sorted({g["net"] for g in sub})
    X = np.array([[1.0 if g["net"] == n else 0.0 for n in nets] + [1.0 if g["search_ms"] == 2200 else 0.0, 1.0 if g["arm"] == "mix-trust" else 0.0] for g in sub])
    b, se = logit_offset(X, np.array([1.0 if g["won"] else 0.0 for g in sub]), np.array([-(g["opponent_rating"] - 2250) / ELO for g in sub]))
    dl, ds_ = b[len(nets)] * ELO, se[len(nets)] * ELO
    print(f"- only the networks played at both settings ({len(sub)} games: {', '.join(nets)}): delta {dl:+.0f} Elo (95% {dl-1.96*ds_:+.0f} to {dl+1.96*ds_:+.0f})")
    # checks of the Elo model: the opponent's rating with a free slope, and the opponent's rating minus our own (the 'expectation' view)
    nets = sorted({g["net"] for g in rated})
    for label, cov in (("opponent rating, free slope", lambda g: (g["opponent_rating"] - 2250) / ELO),
                       ("opponent minus own rating, free slope", lambda g: (g["opponent_rating"] - g["rating_before"]) / ELO)):
        X = np.array([[1.0 if g["net"] == n else 0.0 for n in nets] + [1.0 if g["search_ms"] == 2200 else 0.0,
                      1.0 if g["arm"] == "mix-trust" else 0.0, cov(g)] for g in rated])
        b, se = logit_offset(X, np.array([1.0 if g["won"] else 0.0 for g in rated]), np.zeros(len(rated)))
        k = len(nets)
        print(f"- {label}: slope {b[k+2]:+.2f} (95% {b[k+2]-1.96*se[k+2]:+.2f} to {b[k+2]+1.96*se[k+2]:+.2f}; the Elo model says -1);"
              f" delta (5 s) {b[k]*ELO:+.0f} Elo (95% {(b[k]-1.96*se[k])*ELO:+.0f} to {(b[k]+1.96*se[k])*ELO:+.0f})")

    # the same rating band at both settings (the 1 s games include the account's climb from 1969)
    print("\n### Games started in the same rating band, 2250 to 2400 (veto games)\n")
    print(OUT_HEAD)
    band = {ms: [g for g in gs if 2250 <= g["rating_before"] <= 2400 and g.get("opponent_rating")] for ms, gs in by_set.items()}
    for ms in sorted(band):
        print(outcome_row(SETTING[ms] + ", rating 2250 to 2400", band[ms]))
    for ms in sorted(band):
        mixonly = [g for g in band[ms] if g["arm"] == "mix"]
        print(outcome_row(SETTING[ms] + ", rating 2250 to 2400, mix only", mixonly))
    k1, n1 = sum(g["won"] for g in band[2200]), len(band[2200])
    k2, n2 = sum(g["won"] for g in band[500]), len(band[500])
    d, lo, hi = newcombe(k1, n1, k2, n2)
    print(f"\nin the band, win rate 5 s minus 1 s: {100*d:+.1f} pts (95% {100*lo:+.1f} to {100*hi:+.1f})")

    # how the games ended
    print("\n### How the games ended (veto games)\n")
    print("| Setting | Games | Played out | Opponent forfeited | Opponent timed out | We timed out | We forfeited | Win rate in played-out games |")
    print("|---|---|---|---|---|---|---|---|")
    for ms in sorted(by_set):
        c = collections.Counter(g["ending"] for g in by_set[ms])
        po = [g for g in by_set[ms] if g["ending"] == "played"]
        w = sum(g["won"] for g in po); lo, hi = wilson(w, len(po))
        print(f"| {SETTING[ms]} | {len(by_set[ms])} | {c['played']} | {c['forfeit_them']} | {c['inactivity_them']} | {c['inactivity_us']} | {c['forfeit_us']} | "
              f"{100*w/len(po):.1f}% ({100*lo:.1f} to {100*hi:.1f}) |")
    f1 = sum(1 for g in by_set[2200] if g["ending"] in ("forfeit_them", "inactivity_them"))
    f2 = sum(1 for g in by_set[500] if g["ending"] in ("forfeit_them", "inactivity_them"))
    d, lo, hi = newcombe(f1, len(by_set[2200]), f2, len(by_set[500]))
    print(f"\nopponent gave up (forfeit or timed out): 5 s {f1}/{len(by_set[2200])}, 1 s {f2}/{len(by_set[500])}; difference {100*d:+.1f} pts (95% {100*lo:+.1f} to {100*hi:+.1f})")
    for ms in sorted(by_set):
        ff = [g for g in by_set[ms] if g["ending"] in ("forfeit_them", "inactivity_them")]
        last = [next((d["win_est"] for d in reversed(g["decs"]) if d.get("win_est") is not None), None) for g in ff]
        last = [x for x in last if x is not None]
        print(f"- {SETTING[ms]}: opponents gave up at turn {st.median(g['turns'] for g in ff):.0f} (median);"
              f" the network's last win estimate before it: median {st.median(last) if last else float('nan'):.2f},"
              f" under 0.5 in {sum(1 for x in last if x < 0.5)} of {len(last)} with an estimate")
    allin = collections.Counter((g.get("search_ms"), g.get("arm"), g["ending"]) for g in games if g["ending"] == "inactivity_us")
    print(f"\nall games lost by our inactivity, any mode: {dict(allin)};"
          f" of them after a reconnect: {sum(1 for g in games if g['ending']=='inactivity_us' and any(d.get('reason')=='rejoined' for d in g['decs']) or (g['ending']=='inactivity_us' and g.get('rejoined')))}")

    # ============================================================ 2. mechanism
    print("\n## 2. Mechanism checks (veto games)\n")
    print("| Setting | Searched decisions | Network overruled (95%) | ... to a move it gave < 2% (95%, of overrules) | Search kept the network (disagreed, network played) | Iterations / decision (mean, median) | Iterations / world (median) | Worlds (mean) | Full-width depth, turns (mean) |")
    print("|---|---|---|---|---|---|---|---|---|")
    draws = {}
    rows_ = [(ms, SETTING[ms] + ", mix and mix-trust", by_set[ms]) for ms in sorted(by_set)]
    rows_ += [((k[0], k[1]), f"{SETTING[k[0]]}, {k[1]}", gs) for k, gs in sorted(group(veto, lambda g: (g["search_ms"], g["arm"])).items())]
    for ms, label, gs in rows_:
        sd = [[d for d in g["decs"] if searched(d)] for g in gs]
        ov = boot_ratio([sum(overruled(d) for d in x) for x in sd], [len(x) for x in sd])
        low = boot_ratio([sum(1 for d in x if overruled(d) and (d.get("p_played") or 0) < 0.02) for x in sd],
                         [sum(overruled(d) for d in x) for x in sd])
        kept = sum(1 for x in sd for d in x if d.get("search") is not None and base(d["search"]) != base(d.get("net")) and not overruled(d))
        n = sum(len(x) for x in sd)
        it = [d["iterations"] for x in sd for d in x if d.get("iterations")]
        ipw = [d["iterations_per_world"] for x in sd for d in x if d.get("iterations_per_world")]
        wo = [d["worlds"] for x in sd for d in x if d.get("worlds")]
        fw = [d["full_width_turns"] for x in sd for d in x if d.get("full_width_turns") is not None]
        draws[ms] = (ov[3], low[3])
        print(f"| {label} | {n} | {fmt_ci(*ov[:3], unit='%')} | {fmt_ci(*low[:3], unit='%')} | {100*kept/n:.1f}% | "
              f"{st.mean(it)/1e3:.0f}k, {st.median(it)/1e3:.0f}k | {st.median(ipw)/1e3:.0f}k | {st.mean(wo):.2f} | {st.mean(fw) if fw else float('nan'):.2f} |")
    print(f"\n5 s minus 1 s, mix and mix-trust: overrule share {diff_ci(draws[2200][0], draws[500][0])} pts; share of overrules under 2% {diff_ci(draws[2200][1], draws[500][1])} pts")
    print(f"5 s minus 1 s, mix only: overrule share {diff_ci(draws[(2200, 'mix')][0], draws[(500, 'mix')][0])} pts;"
          f" share of overrules under 2% {diff_ci(draws[(2200, 'mix')][1], draws[(500, 'mix')][1])} pts")
    for k, gs in sorted(group(veto, lambda g: (g["search_ms"], g["arm"])).items()):
        print(f"- {SETTING[k[0]]}, {k[1]}: searched decisions per game {st.mean(g['n_searched'] for g in gs):.1f}")

    print("\nBy network and mode:\n")
    print("| Network, setting, mode | Games | Searched decisions | Network overruled (95%) | ... to a move it gave < 2% (of overrules) | Iterations / decision (median) |")
    print("|---|---|---|---|---|---|")
    for key, gs in sorted(group(veto, lambda g: (g["net"], g["search_ms"], g["arm"])).items()):
        sd = [[d for d in g["decs"] if searched(d)] for g in gs]
        ov = boot_ratio([sum(overruled(d) for d in x) for x in sd], [len(x) for x in sd])
        nov = sum(overruled(d) for x in sd for d in x)
        low = sum(1 for x in sd for d in x if overruled(d) and (d.get("p_played") or 0) < 0.02)
        it = [d["iterations"] for x in sd for d in x if d.get("iterations")]
        print(f"| {key[0]}, {SETTING[key[1]]}, {key[2]} | {len(gs)} | {sum(len(x) for x in sd)} | {fmt_ci(*ov[:3], unit='%')} | "
              f"{100*low/max(1,nov):.1f}% | {st.median(it)/1e3:.0f}k |")

    print("\nSame network, mix at both settings (5 s minus 1 s, 95% by resampling games):\n")
    pooled = []
    for n in sorted(both):
        dd, cnt = {}, {}
        for ms in (500, 2200):
            gs = [g for g in veto if g["net"] == n and g["search_ms"] == ms and g["arm"] == "mix"]
            sd = [[d for d in g["decs"] if searched(d)] for g in gs]
            ov = boot_ratio([sum(overruled(d) for d in x) for x in sd], [len(x) for x in sd])
            lw = boot_ratio([sum(1 for d in x if overruled(d) and (d.get("p_played") or 0) < 0.02) for x in sd], [len(x) for x in sd])
            dd[ms] = (ov[3], lw[3]); cnt[ms] = sum(len(x) for x in sd)
        pooled.append((dd, cnt))
        print(f"- {n}: overruled {diff_ci(dd[2200][0], dd[500][0])} pts of decisions; overruled to a move it gave < 2%"
              f" {diff_ci(dd[2200][1], dd[500][1])} pts of decisions (decisions {cnt[500]} / {cnt[2200]})")
    w = np.array([1.0 / (1.0 / c[500] + 1.0 / c[2200]) for _, c in pooled]); w = w / w.sum()
    pd_ = sum(wi * (dd[2200][0] - dd[500][0]) for wi, (dd, _) in zip(w, pooled))
    pl_ = sum(wi * (dd[2200][1] - dd[500][1]) for wi, (dd, _) in zip(w, pooled))
    print(f"- both networks, weighted by decisions: overruled {100*np.mean(pd_):+.1f} pts (95% {100*np.percentile(pd_,2.5):+.1f} to {100*np.percentile(pd_,97.5):+.1f});"
          f" overruled to < 2% {100*np.mean(pl_):+.1f} pts (95% {100*np.percentile(pl_,2.5):+.1f} to {100*np.percentile(pl_,97.5):+.1f})")
    # all mix at 5 s against all mix at 1 s
    dd = {}
    for ms in (500, 2200):
        gs = [g for g in veto if g["search_ms"] == ms and g["arm"] == "mix"]
        sd = [[d for d in g["decs"] if searched(d)] for g in gs]
        dd[ms] = boot_ratio([sum(overruled(d) for d in x) for x in sd], [len(x) for x in sd])
    print(f"- every mix game (all networks): 1 s {fmt_ci(*dd[500][:3], unit='%')}, 5 s {fmt_ci(*dd[2200][:3], unit='%')}; difference {diff_ci(dd[2200][3], dd[500][3])} pts")

    print("\n### How sharp the search's answer is (veto games, searched decisions)\n")
    print("| Setting, mode | Search favourite's share (mean) | Moves the veto keeps (>= 1/4 of the favourite, mean) | Entropy of the shares (mean, nats) | Network's likeliest move vetoed |")
    print("|---|---|---|---|---|")
    for key, gs in sorted(group(veto, lambda g: (g["search_ms"], g["arm"])).items()):
        sd = [d for g in gs for d in g["decs"] if searched(d) and d.get("shares")]
        fav = st.mean(max(d["shares"]) for d in sd)
        keep = st.mean(sound_count(d) for d in sd)
        ent = st.mean(e for e in (entropy(d["shares"]) for d in sd) if e is not None)
        vet = sum(1 for d in sd if overruled(d)) / len(sd)
        print(f"| {SETTING[key[0]]}, {key[1]} | {fav:.3f} | {keep:.2f} | {ent:.2f} | {100*vet:.1f}% |")

    print("\n### Fallbacks, timing and the battle timer (veto games)\n")
    print("| Setting | Decisions | Search fell back (error/expired/timeout) | Played by the network after a reconnect | Games the search played < 50% of | Decision time median / p95 / max (s) | Deadline | Decisions within 0.5 s of the deadline / past it | Games with a late order (results 'late') | Lowest timer bank seen (s): median game / worst game | Games whose bank fell under 120 s / 60 s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for ms in sorted(by_set):
        gs = by_set[ms]
        ds = [d for g in gs for d in g["decs"]]
        fb = sum(1 for d in ds if d.get("asked") and d.get("source") == "net" and d.get("reason") in ("error", "expired", "timeout", "down"))
        rj = sum(1 for d in ds if d.get("reason") == "rejoined")
        half = sum(1 for g in gs if g["n_dec"] and g["n_searched"] < 0.5 * g["n_dec"])
        ms_ = sorted(d["ms"] / 1000 for d in ds if searched(d) and d.get("ms") is not None)
        near = sum(1 for x in ms_ if x >= DEADLINE[ms] - 0.5)
        past = sum(1 for x in ms_ if x > DEADLINE[ms])
        late = sum(1 for g in gs if (g.get("late") or 0) > 0)
        mins = [min(g["timer_total"]) for g in gs if g["timer_total"]]
        print(f"| {SETTING[ms]} | {len(ds)} | {fb} ({100*fb/len(ds):.2f}%) | {rj} ({100*rj/len(ds):.1f}%) | {half} of {len(gs)} | "
              f"{st.median(ms_):.2f} / {ms_[int(0.95*len(ms_))]:.2f} / {ms_[-1]:.2f} | {DEADLINE[ms]} s | {near} / {past} | {late} | "
              f"{st.median(mins):.0f} / {min(mins)} | {sum(1 for x in mins if x < 120)} / {sum(1 for x in mins if x < 60)} |")
    for ms in sorted(by_set):
        errs = collections.Counter((g["battle"][-10:], g["net"]) for g in by_set[ms] for d in g["decs"] if d.get("reason") == "error")
        rjs = collections.Counter((g["battle"][-10:], g["net"]) for g in by_set[ms] for d in g["decs"] if d.get("reason") == "rejoined")
        print(f"- {SETTING[ms]}: search errors by game {dict(errs)}; reconnect decisions by game {dict(rjs)}")
    for ms in sorted(by_set):
        gs = by_set[ms]
        lowtimer = [(g["battle"][-10:], min(g["timer_total"]), g["won"], g["ending"]) for g in gs if g["timer_total"] and min(g["timer_total"]) < 120]
        print(f"- {SETTING[ms]}: games whose bank fell under 120 s: {lowtimer}")
    print("\nGame length (veto games):")
    for ms in sorted(by_set):
        gs = by_set[ms]
        print(f"- {SETTING[ms]}: turns mean {st.mean(g['turns'] for g in gs):.1f}, game seconds median {st.median(g['seconds'] for g in gs if g.get('seconds')):.0f}, "
              f"seconds per turn median {st.median(g['seconds']/max(1,g['turns']) for g in gs if g.get('seconds')):.1f}")

    # ============================================================ 3. the network's win estimate after overrules
    print("\n## 3. The network's own win estimate at the next decision, after an overrule and after its own move\n")
    print("Positions matched by the network's estimate (10 bands), weighted as the overrules fall (loss_study.py's method),"
          " searched decisions only, consecutive decisions of a game. Points of win probability; 95% intervals by resampling games.\n")
    print("| Setting | Pairs (own / overrule / overrule < 2%) | After its own move | After an overrule | After an overrule to a move it gave < 2% | Overrule minus own |")
    print("|---|---|---|---|---|---|")
    wdraws = {}
    for ms in sorted(by_set):
        res = est_deltas(by_set[ms])
        wdraws[ms] = res
        print(f"| {SETTING[ms]} | {res['n']} | {res['own']} | {res['over']} | {res['low']} | {res['gap']} |")
    print(f"\n(gap) 5 s minus 1 s: {diff_ci(wdraws[2200]['gap_draws'], wdraws[500]['gap_draws'], dec=2)} pts")
    for key, gs in sorted(group(veto, lambda g: (g["search_ms"], g["arm"])).items()):
        res = est_deltas(gs)
        print(f"- {SETTING[key[0]]}, {key[1]}: own {res['own']}, overrule {res['over']}, < 2% {res['low']}, gap {res['gap']} (pairs {res['n']})")
    m1, m5 = est_deltas([g for g in by_set[500] if g["arm"] == "mix"]), est_deltas([g for g in by_set[2200] if g["arm"] == "mix"])
    print(f"(gap) 5 s mix minus 1 s mix: {diff_ci(m5['gap_draws'], m1['gap_draws'], dec=2)} pts")
    for key, gs in sorted(group(veto, lambda g: (g["net"], g["search_ms"], g["arm"])).items()):
        res = est_deltas(gs)
        if res["n"].startswith("0 /"):
            continue
        print(f"- {key[0]}, {SETTING[key[1]]}, {key[2]}: own {res['own']}, overrule {res['over']}, gap {res['gap']} (pairs {res['n']})")
    print("  (no win estimate in the older records: main 011850 and arm C 426M at 1 s, main 024381)")

    # ============================================================ 4. explanations
    print("\n## 4. What more search buys\n")
    print("### Reading the opponent: the search's own view of the opponent's next action, scored against what it did\n")
    print("| Setting, mode | Scored predictions | Top-1 right (95%) | Top-3 right | Mean probability on what it did (95%) | Mean log-loss (nats) |")
    print("|---|---|---|---|---|---|")
    pdraw = {}
    for key, gs in sorted(group([g for g in srch if g["outcomes"]], lambda g: (g["search_ms"], g["arm"])).items()):
        oc = [g["outcomes"] for g in gs]
        t1 = boot_ratio([sum(1 for o in x if o.get("top1_right")) for x in oc], [len(x) for x in oc])
        t3 = sum(1 for x in oc for o in x if o.get("top3_right")) / sum(len(x) for x in oc)
        pt = boot_ratio([sum(o["p_true"] for o in x) for x in oc], [len(x) for x in oc])
        ll = st.mean(-math.log(max(1e-4, o["p_true"])) for x in oc for o in x)
        pdraw[key] = (t1[3], pt[3])
        print(f"| {SETTING[key[0]]}, {key[1]} | {sum(len(x) for x in oc)} | {fmt_ci(*t1[:3], unit='%')} | {100*t3:.1f}% | {fmt_ci(*pt[:3], unit='%')} | {ll:.3f} |")
    a = [k for k in pdraw if k[0] == 2200]; b = [k for k in pdraw if k == (500, "mix")]
    if a and b:
        allt = np.concatenate([pdraw[k][0] for k in a])
        print(f"\ntop-1 accuracy, 5 s (mix) minus 1 s (mix): {diff_ci(pdraw[(2200,'mix')][0], pdraw[(500,'mix')][0])} pts;"
              f" probability on what it did: {diff_ci(pdraw[(2200,'mix')][1], pdraw[(500,'mix')][1])} pts")

    print("\n### Overrules in wins and in losses (searched decisions)\n")
    for ms in sorted(by_set):
        for won in (True, False):
            gs = [g for g in by_set[ms] if g["won"] == won]
            sd = [[d for d in g["decs"] if searched(d)] for g in gs]
            ov = boot_ratio([sum(overruled(d) for d in x) for x in sd], [len(x) for x in sd])
            low = boot_ratio([sum(1 for d in x if overruled(d) and (d.get("p_played") or 0) < 0.02) for x in sd], [len(x) for x in sd])
            print(f"- {SETTING[ms]}, {'wins' if won else 'losses'} ({len(gs)}): overruled {fmt_ci(*ov[:3], unit='%')} of decisions; "
                  f"overruled to < 2% {fmt_ci(*low[:3], unit='%')} of decisions")

    print("\n### Per-game overrule rate against the result (logistic, veto games: the network and the opponent's rating held as in section 1)\n")
    for label, pool in (("all veto games", rated),):
        nets = sorted({g["net"] for g in pool})
        X, y, off = [], [], []
        for g in pool:
            sd = [d for d in g["decs"] if searched(d)]
            if not sd:
                continue
            r_low = sum(1 for d in sd if overruled(d) and (d.get("p_played") or 0) < 0.02) / len(sd)
            r_hi = sum(1 for d in sd if overruled(d) and (d.get("p_played") or 0) >= 0.02) / len(sd)
            X.append([1.0 if g["net"] == n else 0.0 for n in nets] + [1.0 if g["search_ms"] == 2200 else 0.0, r_low * 10, r_hi * 10])
            y.append(1.0 if g["won"] else 0.0); off.append(-(g["opponent_rating"] - 2250) / ELO)
        b, se = logit_offset(np.array(X), np.array(y), np.array(off))
        k = len(nets)
        for i, nm in ((k, "5 s"), (k + 1, "+10 pts of decisions overruled to < 2%"), (k + 2, "+10 pts of decisions overruled to >= 2%")):
            print(f"- {nm}: {b[i]*ELO:+.0f} Elo (95% {(b[i]-1.96*se[i])*ELO:+.0f} to {(b[i]+1.96*se[i])*ELO:+.0f})")
        print("  (the overrule rate is not assigned at random: it is higher in harder positions, so this is association, not effect)")


def group(gs, key):
    out = collections.defaultdict(list)
    for g in gs:
        out[key(g)].append(g)
    return out


def est_deltas(gs):
    """loss_study.py's matched comparison, with cluster-bootstrap intervals."""
    per_game = []
    for g in gs:
        ds = [d for d in g["decs"] if searched(d) and d.get("win_est") is not None]
        rows = np.zeros((10, 3, 2))       # bucket x (own, over, over_low) x (sum, count)
        for a, b2 in zip(ds, ds[1:]):
            bk = min(9, int(a["win_est"] * 10))
            delta = b2["win_est"] - a["win_est"]
            if overruled(a):
                rows[bk, 1] += (delta, 1)
                if (a.get("p_played") or 0) < 0.02:
                    rows[bk, 2] += (delta, 1)
            else:
                rows[bk, 0] += (delta, 1)
        per_game.append(rows)
    arr = np.array(per_game)            # games x 10 x 3 x 2

    def stat(tot):
        n_over = tot[:, 1, 1]
        w = n_over / max(1e-12, n_over.sum())
        means = np.where(tot[:, :, 1] > 0, tot[:, :, 0] / np.maximum(tot[:, :, 1], 1e-12), 0.0)
        return [(w * means[:, j]).sum() for j in range(3)]

    point = stat(arr.sum(0))
    idx = RNG.integers(0, len(arr), size=(BOOT, len(arr)))
    dr = np.array([stat(arr[i].sum(0)) for i in idx])
    tot = arr.sum(0)

    def f(j):
        return f"{100*point[j]:+.2f} ({100*np.percentile(dr[:, j], 2.5):+.2f} to {100*np.percentile(dr[:, j], 97.5):+.2f})"
    gap = dr[:, 1] - dr[:, 0]
    return {"n": f"{int(tot[:,0,1].sum())} / {int(tot[:,1,1].sum())} / {int(tot[:,2,1].sum())}", "own": f(0), "over": f(1), "low": f(2),
            "gap": f"{100*(point[1]-point[0]):+.2f} ({100*np.percentile(gap, 2.5):+.2f} to {100*np.percentile(gap, 97.5):+.2f})",
            "gap_draws": gap}


if __name__ == "__main__":
    main()
