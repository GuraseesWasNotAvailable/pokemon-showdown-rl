#!/usr/bin/env python
"""
Fits the potential function on recorded games and reports how well it predicts the winner.

    .venv/Scripts/python.exe -m rl_reward.fit_potential [--top heuristic/cache/top] [--bot <battles folder>] [--write]

Every game is run through poke-env's protocol parser (the one the training workers use), the
feature vector is taken at the start of each turn, and a logistic model with no intercept is
fitted to "did this side go on to win". All numbers reported are out-of-fold: five folds split
by game, never by turn, so each game is scored by a model that did not see it.

Three potentials are compared:

  faints only      the count of Pokemon left, which is all the old +/-0.1-per-faint reward sees
  common shaping   Pokemon left, HP and status: what Metamon's and poke-env's shaping look at
  full             everything in features.py

After fitting, one number (a temperature) rescales the weights so that predicted chances match
actual win rates out-of-fold. A potential that claims 95% where the truth is 80% makes every
comeback look like a huge event, and that is noise in the reward.

Under potential-based shaping any of these leaves the best policy unchanged. What a better
predictor buys is a shaping signal that agrees with winning more often.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

from .features import FEATURES, replay_states

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MODELS: Dict[str, List[str]] = {
    "faints only": ["alive_diff"],
    "common shaping": ["alive_diff", "hp_diff", "slpfrz_diff", "par_diff", "brn_diff", "psn_diff"],
    "full": list(FEATURES),
}


def load_top(folder: str, min_rating: int = 0):
    """Public replays between high-rated players: {id, rating, players, log}."""
    out = []
    for f in sorted(glob.glob(os.path.join(folder, "*.json"))):
        try:
            r = json.load(io.open(f, encoding="utf-8"))
        except Exception:
            continue
        log = r.get("log") or ""
        players = r.get("players") or []
        if not log or len(players) < 2 or (r.get("rating") or 0) < min_rating:
            continue
        out.append((log.split("\n"), players[0], r.get("id", os.path.basename(f)), r.get("rating") or 0))
    return out


def load_bot(folder: str):
    """Games PSBot recorded on the ladder: {log: [...], me: {name}, result}."""
    out = []
    for f in sorted(glob.glob(os.path.join(folder, "**", "*.json"), recursive=True)):
        try:
            r = json.load(io.open(f, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(r, dict) or "log" not in r or "me" not in r:
            continue
        out.append((r["log"], r["me"].get("name", ""), r.get("id", os.path.basename(f)), 0))
    return out


def build(games) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    X, y, gid, frac = [], [], [], []
    info = []
    for gi, (lines, me, name, rating) in enumerate(games):
        feats, turns, winner = replay_states(lines, me, tag="battle-" + str(name).replace("battle-", ""))
        if winner is None or len(feats) < 3:
            continue
        won = 1.0 if winner == me else 0.0
        last = max(turns)
        for f, t in zip(feats, turns):
            X.append(f)
            y.append(won)
            gid.append(gi)
            frac.append(t / last)
        info.append({"game": name, "rating": rating, "won": won, "turns": last})
    return np.array(X), np.array(y), np.array(gid), np.array(frac), info


def fit_logistic(X: np.ndarray, y: np.ndarray, sw: np.ndarray, l2: float = 1.0, iters: int = 80) -> np.ndarray:
    """Newton's method for a weighted logistic regression with an L2 penalty and no intercept."""
    d = X.shape[1]
    w = np.zeros(d)
    for _ in range(iters):
        z = np.clip(X @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        g = X.T @ (sw * (p - y)) + l2 * w
        H = (X * (sw * p * (1 - p))[:, None]).T @ X + l2 * np.eye(d)
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return w


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def auc(p: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y > 0.5
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metrics(p: np.ndarray, y: np.ndarray, sw: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = -(sw * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / sw.sum()
    br = (sw * (p - y) ** 2).sum() / sw.sum()
    acc = (sw * ((p > 0.5) == (y > 0.5))).sum() / sw.sum()
    return {"logloss": float(ll), "brier": float(br), "acc": float(acc), "auc": auc(p, y)}


def squash(z: np.ndarray, cap: float, slope: float) -> np.ndarray:
    """Win chance from a score: 0.5 + (cap / 2) * tanh(slope * z). `cap` is the largest |potential|."""
    return 0.5 + 0.5 * cap * np.tanh(slope * z)


def best_link(z: np.ndarray, y: np.ndarray, sw: np.ndarray) -> Tuple[float, float]:
    """
    The cap and slope that minimise log loss. A plain logistic says 93% where these games show
    83%: even a large lead is lost about one time in six, because one boosted Pokemon can still
    take the game. A link that levels off below 1 says so.
    """
    best = (1e9, 1.0, 0.5)
    for cap in np.linspace(0.5, 1.0, 26):
        for slope in np.linspace(0.15, 1.2, 43):
            ll = metrics(squash(z, cap, slope), y, sw)["logloss"]
            if ll < best[0]:
                best = (ll, float(cap), float(slope))
    return best[1], best[2]


def out_of_fold(X, y, gid, sw, cols, l2, folds, seed):
    """Logits for every state from a model that never saw that state's game."""
    games = np.unique(gid)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(games)
    z = np.zeros(len(X))
    for k in range(folds):
        test_games = set(perm[k::folds].tolist())
        te = np.array([g in test_games for g in gid])
        tr = ~te
        w = fit_logistic(X[tr][:, cols], y[tr], sw[tr] * (len(games) - len(test_games)), l2=l2)
        z[te] = X[te][:, cols] @ w
    return z


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", default=os.path.join(ROOT, "heuristic", "cache", "top"))
    ap.add_argument("--bot", default="")
    ap.add_argument("--min-rating", type=int, default=0)
    ap.add_argument("--l2", type=float, default=3.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--write", action="store_true", help="write rl_reward/weights.json from a fit on all the expert games")
    args = ap.parse_args()

    top = load_top(args.top, args.min_rating) if args.top and os.path.isdir(args.top) else []
    bot = load_bot(args.bot) if args.bot and os.path.isdir(args.bot) else []
    print(f"expert replays: {len(top)}   bot ladder games: {len(bot)}")
    if not top:
        print("no expert replays found; fetch them with: node heuristic/toplay.js fetch")
        sys.exit(1)

    X, y, gid, frac, info = build(top)
    games = np.unique(gid)
    ratings = [i["rating"] for i in info if i["rating"]]
    print(f"expert games parsed: {len(games)}   states: {len(X)}   mean turns: {np.mean([i['turns'] for i in info]):.1f}"
          + (f"   ratings {min(ratings)} to {max(ratings)}, median {int(np.median(ratings))}" if ratings else ""))
    counts = {g: (gid == g).sum() for g in games}
    sw = np.array([1.0 / counts[g] for g in gid])  # every game counts the same, however long it ran
    print("the first-listed player won %.1f%% of these games (the public list is sorted by a rating that favours" % (100 * np.mean([i["won"] for i in info])))
    print("that player), so every state is scored from both sides below. The model has no intercept, so this")
    print("changes none of the weights, only the reported numbers.")

    def both(z_, y_, sw_, extra=None):
        zz, yy, ss = np.concatenate([z_, -z_]), np.concatenate([y_, 1 - y_]), np.concatenate([sw_, sw_])
        return (zz, yy, ss) if extra is None else (zz, yy, ss, np.concatenate([extra, extra]))

    print("\nout-of-fold on the expert games (%d folds by game). coin flip: logloss 0.693, brier 0.250" % args.folds)
    print("  %-16s %8s %8s %8s %8s %6s   logloss by third of the game: early / middle / late" % ("potential", "logloss", "brier", "acc", "auc", "cap"))
    oof = {}
    for name, feats in MODELS.items():
        cols = [FEATURES.index(n) for n in feats]
        z = out_of_fold(X, y, gid, sw, cols, args.l2, args.folds, args.seed)
        z2, y2, s2, f2 = both(z, y, sw, frac)
        cap, t = best_link(z2, y2, s2)
        p = squash(z, cap, t)
        p2 = squash(z2, cap, t)
        oof[name] = (cols, z, t, p, cap)
        m = metrics(p2, y2, s2)
        thirds = []
        for lo, hi in ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01)):
            sel = (f2 > lo) & (f2 <= hi)
            thirds.append(metrics(p2[sel], y2[sel], s2[sel])["logloss"])
        print("  %-16s %8.3f %8.3f %7.1f%% %8.3f %6.2f   %.3f / %.3f / %.3f" % (name, m["logloss"], m["brier"], 100 * m["acc"], m["auc"], cap, *thirds))

    cols, z, t, p, cap = oof["full"]
    z2, y2, s2 = both(z, y, sw)
    p2 = squash(z2, cap, t)
    print("\ncalibration of the full potential, out-of-fold (predicted -> actual, states). cap %.2f" % cap)
    for lo in np.arange(0.5, 1, 0.1):
        sel = (p2 >= lo) & (p2 < lo + 0.1)
        if sel.sum() >= 30:
            print("  %3.0f-%3.0f%%: predicted %5.1f%%  actual %5.1f%%  (%d)" % (100 * lo, 100 * lo + 10, 100 * p2[sel].mean(), 100 * y2[sel].mean(), sel.sum()))

    # what a per-faint reward is betting on
    first = []
    for gi in games:
        sel = np.where(gid == gi)[0]
        a = X[sel][:, FEATURES.index("alive_diff")]
        nz = np.nonzero(np.abs(a) > 1e-9)[0]
        if len(nz):
            first.append((a[nz[0]] > 0) == (y[sel][0] > 0.5))
    print("\nthe side that took the first knockout went on to win %.1f%% of %d expert games" % (100 * np.mean(first), len(first)))
    # how much the shaping signal moves: mean absolute change in potential per turn
    phi = 2 * p - 1
    moves = []
    for gi in games:
        sel = np.where(gid == gi)[0]
        if len(sel) > 1:
            moves.append(np.abs(np.diff(phi[sel])).mean())
    print("mean absolute change in the potential from one turn to the next: %.3f (potential runs from -1 to 1)" % np.mean(moves))

    # final fit on every expert game, temperature folded into the weights
    w_all = fit_logistic(X[:, cols], y, sw * len(games), l2=args.l2) * t
    print("\nweights of the full potential, fitted on all %d expert games: Phi = %.2f * tanh(w . features)" % (len(games), cap))
    for n, v in sorted(zip(FEATURES, w_all), key=lambda q: -abs(q[1])):
        print("  %-16s %+7.3f" % (n, v))

    if bot:
        Xb, yb, gb, fb, ib = build(bot)
        if len(Xb):
            cb = {g: (gb == g).sum() for g in np.unique(gb)}
            swb = np.array([1.0 / cb[g] for g in gb])
            print("\nthe same potentials on %d bot ladder games (%d states), which none of them was fitted on" % (len(cb), len(Xb)))
            for name, feats in MODELS.items():
                c2 = [FEATURES.index(n) for n in feats]
                w2 = fit_logistic(X[:, c2], y, sw * len(games), l2=args.l2)
                m = metrics(squash(Xb[:, c2] @ w2, oof[name][4], oof[name][2]), yb, swb)
                print("  %-16s logloss %.3f  brier %.3f  acc %.1f%%  auc %.3f" % (name, m["logloss"], m["brier"], 100 * m["acc"], m["auc"]))

    if args.write:
        m = metrics(p2, y2, s2)
        out = {
            "features": FEATURES,
            "weights": [float(v) for v in w_all],
            "cap": float(cap),
            "meta": {
                "fitted_on": "%d public gen9randombattle replays, ratings %s to %s (%d turn states)" % (
                    len(games), min(ratings) if ratings else "?", max(ratings) if ratings else "?", len(X)),
                "model": "Phi = cap * tanh(w . features); w from a logistic fit with no intercept and L2 %.1f, every game weighted equally, then cap and slope set out-of-fold" % args.l2,
                "out_of_fold": {k: round(v, 4) for k, v in m.items()},
            },
        }
        path = os.path.join(HERE, "weights.json")
        with io.open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=1)
            f.write("\n")
        print("\nwrote", path)


if __name__ == "__main__":
    main()
