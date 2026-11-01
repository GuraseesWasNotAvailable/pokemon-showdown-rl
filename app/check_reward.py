"""
Checks the reward on live games, without starting the Ray cluster.

Two of the rollout worker's own players (worker.SyncBridgePlayer) play each other on a local
Showdown server. The potential is read at every decision exactly as it is during training, and
the rewards are built by the same function the worker calls. Then the properties the reward is
supposed to have are checked on what actually came out:

  * every potential is inside its cap, and every return-to-go inside the value head's support
  * the discounted sum of a game's rewards is gamma^(T-1) * result - scale * (first potential)
  * the two seats of the same game see opposite potentials (up to what one side cannot see)
  * the potential near the end of a game agrees with who won

    .venv/Scripts/python.exe check_reward.py --games 20                  # SimpleHeuristics plays
    .venv/Scripts/python.exe check_reward.py --games 20 --teacher psbot  # PSBot plays
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import queue
import secrets
from collections import defaultdict

import numpy as np
from poke_env import AccountConfiguration

from config import RunConfig
from rl_reward import load_potential, shaped_rewards
from rl_reward.shaping import discounted_return
from worker import SyncBridgePlayer, make_server_conf


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--teacher", choices=("psbot", "simple"), default="simple")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    base = RunConfig.default()
    cfg = dataclasses.replace(
        base,
        learner=dataclasses.replace(base.learner, mode="imitation"),
        rollout=dataclasses.replace(base.rollout, imitation_teacher=args.teacher),
    )
    assert cfg.reward.shaping == "potential", "this check is for the potential-based reward"
    pot = load_potential(cfg.reward.potential_file or None)
    gamma, scale = cfg.learner.gamma, cfg.reward.shaping_scale
    events: "queue.Queue" = queue.Queue()
    sem = asyncio.Semaphore(10_000)
    tag = secrets.token_hex(3)

    def player(side: str) -> SyncBridgePlayer:
        return SyncBridgePlayer(
            cfg=cfg, event_queue=events, action_futures={}, ep_sem=sem,
            account_configuration=AccountConfiguration(f"rew{tag}{side}", None),
            server_configuration=make_server_conf("127.0.0.1", args.port),
            max_concurrent_battles=args.concurrency, battle_format=cfg.env.battle_format, log_level=40,
        )

    a, b = player("a"), player("b")
    print(f"{args.games} games on port {args.port}, {args.teacher} playing both seats. gamma {gamma}, scale {scale}, cap {pot.cap:.2f}")
    await a.battle_against(b, n_battles=args.games)

    phis = defaultdict(list)
    done = {}
    while True:
        try:
            kind, etag, _obs, extra = events.get_nowait()
        except queue.Empty:
            break
        if kind == "IMITATION":
            phis[etag].append(float(extra[1]))
        elif kind == "DONE":
            done[etag] = extra

    games = steps = 0
    worst_identity = 0.0
    out_of_cap = out_of_support = 0
    agree = total_end = 0
    rewards_abs = []
    by_battle = defaultdict(dict)
    for etag, battle in done.items():
        phi = np.array(phis.get(etag, []))
        if not len(phi):
            continue
        games += 1
        steps += len(phi)
        outcome = 1.0 if battle.won else (-1.0 if battle.lost else 0.0)
        r = shaped_rewards(phi, outcome, gamma, scale)
        worst_identity = max(worst_identity, abs(discounted_return(r, gamma) - (gamma ** (len(phi) - 1) * outcome - scale * phi[0])))
        out_of_cap += int((np.abs(phi) > pot.cap + 1e-9).sum())
        g = 0.0
        for x in r[::-1]:
            g = float(x) + gamma * g
            if not (cfg.learner.v_min - 1e-6 <= g <= cfg.learner.v_max + 1e-6):
                out_of_support += 1
        rewards_abs.append(np.abs(r[:-1]).mean() if len(r) > 1 else 0.0)
        if outcome != 0.0:
            total_end += 1
            agree += int(np.sign(phi[-1]) == np.sign(outcome))
        by_battle[etag.split("_", 1)[1]][etag.split("_", 1)[0]] = (phi, outcome)

    # the two seats of one game: potentials at the first decision of each side should be opposite,
    # and over the game they should move in opposite directions
    opp_corr = []
    for bt, sides in by_battle.items():
        if len(sides) != 2:
            continue
        (p1, _), (p2, _) = sides.values()
        n = min(len(p1), len(p2))
        if n >= 6 and np.std(p1[:n]) > 1e-6 and np.std(p2[:n]) > 1e-6:
            opp_corr.append(float(np.corrcoef(p1[:n], p2[:n])[0, 1]))

    print(f"player-games {games}, decisions {steps}")
    print(f"potentials outside the cap: {out_of_cap}")
    print(f"returns outside the value head's [{cfg.learner.v_min}, {cfg.learner.v_max}]: {out_of_support}")
    print(f"largest error in sum(gamma^t r_t) = gamma^(T-1) * result - scale * Phi(s0): {worst_identity:.2e}")
    print(f"mean |shaping reward| per decision: {np.mean(rewards_abs):.3f} (the result is worth 1)")
    print(f"potential at the last decision has the sign of the result in {agree} of {total_end} player-games")
    if opp_corr:
        print(f"correlation between the two seats' potentials over a game: mean {np.mean(opp_corr):.2f} over {len(opp_corr)} games (-1 is exactly opposite)")
    ok = games > 0 and out_of_cap == 0 and out_of_support == 0 and worst_identity < 1e-4 and (total_end == 0 or agree / total_end >= 0.7)
    print("OK" if ok else "PROBLEM")


if __name__ == "__main__":
    asyncio.run(main())
