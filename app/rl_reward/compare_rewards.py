#!/usr/bin/env python
"""
Compares reward designs on recorded expert games, without training anything.

    .venv/Scripts/python.exe -m rl_reward.compare_rewards [--top heuristic/cache/top]

For each design it replays every game from both seats, computes the rewards that design would
have paid turn by turn, and reports:

  agreement     of all the shaping reward paid before the final turn, the share that pointed the
                way the game ended: sum(r_t * outcome) / sum(|r_t|). 1.0 means every intermediate
                reward was on the winner's side, 0 means it carried no information.
  half-time     correlation between the shaping reward collected in the first half of a game and
                its result.
  target var    variance of the return-to-go the critic has to learn, over all states. Win/loss
                alone is 1.0 by construction; shaping that explains part of the result lowers it.
  margin        what a 6-0 win returns minus what a 6-5 win returns. Zero means the design pays
                for winning and nothing else.

The designs:

  sparse              win/loss only
  faint 0.1           upstream ps-ppo: +0.1 per opposing faint, -0.1 per own faint
  metamon             Metamon's default, rescaled so a win is 1: HP of the two active Pokemon,
                      half weight on status, one per faint, all divided by 100
  potential           this repository: scale * (gamma * Phi' - Phi), Phi fitted out-of-fold
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .features import FEATURES
from .fit_potential import ROOT, best_link, build, fit_logistic, load_top, squash
from .shaping import shaped_rewards

IDX = {n: i for i, n in enumerate(FEATURES)}


def per_game(X, y, gid, phi, gamma, scale):
    """Yields, for every game and both seats, the per-turn shaping rewards of each design and the outcome."""
    for g in np.unique(gid):
        sel = np.where(gid == g)[0]
        if len(sel) < 4:
            continue
        for sign in (1.0, -1.0):
            f = X[sel] * sign
            out = 1.0 if (y[sel][0] > 0.5) == (sign > 0) else -1.0
            alive = f[:, IDX["alive_diff"]] * 6.0
            act_hp = f[:, IDX["active_hp_diff"]]
            status = (f[:, IDX["slpfrz_diff"]] + f[:, IDX["par_diff"]] + f[:, IDX["brn_diff"]] + f[:, IDX["psn_diff"]]) * 6.0
            d_alive = np.diff(alive, append=alive[-1])
            faint = 0.1 * d_alive
            # Metamon: (damage done + hp gained) + 0.5 * (status given - taken) + (their faints - mine), win = 100
            metamon = (np.diff(act_hp, append=act_hp[-1]) - 0.5 * np.diff(status, append=status[-1]) + d_alive) / 100.0
            pot = shaped_rewards(phi[sel] * sign, 0.0, gamma, scale)
            # the last entry of the potential's shaping is the pay-back at the end of the game; keep it
            yield {"faint 0.1": faint, "metamon": metamon, "potential": pot, "sparse": np.zeros(len(sel))}, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", default=os.path.join(ROOT, "heuristic", "cache", "top"))
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--l2", type=float, default=3.0)
    args = ap.parse_args()

    X, y, gid, frac, info = build(load_top(args.top))
    games = np.unique(gid)
    counts = {g: (gid == g).sum() for g in games}
    sw = np.array([1.0 / counts[g] for g in gid])
    # potential out-of-fold, so no game is scored by weights fitted on it
    rng = np.random.default_rng(7)
    perm = rng.permutation(games)
    z = np.zeros(len(X))
    for k in range(5):
        test = set(perm[k::5].tolist())
        te = np.array([g in test for g in gid])
        w = fit_logistic(X[~te], y[~te], sw[~te] * (len(games) - len(test)), l2=args.l2)
        z[te] = X[te] @ w
    zz, yy, ss = np.concatenate([z, -z]), np.concatenate([y, 1 - y]), np.concatenate([sw, sw])
    cap, slope = best_link(zz, yy, ss)
    phi = 2 * squash(z, cap, slope) - 1

    names = ["sparse", "faint 0.1", "metamon", "potential"]
    agree_num = {n: 0.0 for n in names}
    agree_den = {n: 0.0 for n in names}
    half, outs = {n: [] for n in names}, []
    sq = {n: 0.0 for n in names}
    n_states = 0
    for rewards, out in per_game(X, y, gid, phi, args.gamma, args.scale):
        outs.append(out)
        T = len(rewards["sparse"])
        for n in names:
            r = rewards[n].astype(np.float64).copy()
            mid = r[:-1]  # everything paid before the game's last turn
            agree_num[n] += float((mid * out).sum())
            agree_den[n] += float(np.abs(mid).sum())
            half[n].append(float(r[: T // 2].sum()))
            # return-to-go at every state, with the result added on the last step
            r[-1] += out
            g = 0.0
            for x in r[::-1]:
                g = x + args.gamma * g
                sq[n] += g * g
        n_states += T
    outs = np.array(outs)
    print("%d expert games, both seats, %d states. Potential: cap %.2f, scale %.2f, gamma %.3f\n" % (len(games), n_states, cap, args.scale, args.gamma))
    print("  %-12s %10s %10s %11s %8s" % ("design", "agreement", "half-time", "target var", "margin"))
    margins = {"sparse": 0.0, "faint 0.1": 0.5, "metamon": 0.05, "potential": 0.0}
    for n in names:
        a = agree_num[n] / agree_den[n] if agree_den[n] > 0 else float("nan")
        h = np.array(half[n])
        c = float(np.corrcoef(h, outs)[0, 1]) if h.std() > 0 else float("nan")
        print("  %-12s %10s %10s %11.3f %8.2f" % (n, "-" if np.isnan(a) else "%.2f" % a, "-" if np.isnan(c) else "%.2f" % c, sq[n] / n_states, margins[n]))
    # what scale would make the critic's target smallest
    g_all = np.concatenate([np.where(y > 0.5, 1.0, -1.0), np.where(y > 0.5, -1.0, 1.0)])
    p_all = np.concatenate([phi, -phi])
    best = float((g_all * p_all).mean() / (p_all * p_all).mean())
    print("\nthe scale that makes the critic's target smallest on these games: %.2f" % best)
    for c in (0.0, 0.5, 1.0):
        print("  scale %.1f: mean squared target %.3f, returns within +/-%.2f" % (c, float(((g_all - c * p_all) ** 2).mean()), 1 + c * cap))


if __name__ == "__main__":
    main()
