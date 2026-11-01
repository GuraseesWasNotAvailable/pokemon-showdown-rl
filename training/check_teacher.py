"""
Checks the imitation teacher without starting the Ray cluster.

Two of the rollout worker's own players (worker.SyncBridgePlayer) play each other in
imitation mode on a local Showdown server. Every recorded step is then checked the way the
learner will use it: the observation has the right size, and the action the teacher chose is
one the action mask allows.

    python check_teacher.py --games 20                    # PSBot as the teacher
    python check_teacher.py --games 20 --teacher simple   # the original teacher
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import queue
import secrets
from collections import Counter

import numpy as np
from poke_env import AccountConfiguration

from config import RunConfig
from obs_assembler import ObservationAssembler
from worker import SyncBridgePlayer, make_server_conf


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--teacher", choices=("psbot", "simple"), default="psbot")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    base = RunConfig.default()
    cfg = dataclasses.replace(
        base,
        learner=dataclasses.replace(base.learner, mode="imitation"),
        rollout=dataclasses.replace(base.rollout, imitation_teacher=args.teacher),
    )
    events: "queue.Queue" = queue.Queue()
    sem = asyncio.Semaphore(10_000)
    tag = secrets.token_hex(3)

    def player(side: str) -> SyncBridgePlayer:
        return SyncBridgePlayer(
            cfg=cfg, event_queue=events, action_futures={}, ep_sem=sem,
            account_configuration=AccountConfiguration(f"teach{tag}{side}", None),
            server_configuration=make_server_conf("127.0.0.1", args.port),
            max_concurrent_battles=args.concurrency, battle_format=cfg.env.battle_format, log_level=40,
        )

    a, b = player("a"), player("b")
    print(f"teacher={args.teacher}: {args.games} imitation games on port {args.port}")
    await a.battle_against(b, n_battles=args.games)

    asm = ObservationAssembler()
    m0, m1 = asm.offsets["action_mask"]
    steps = illegal = bad_shape = finished = 0
    kinds: Counter = Counter()
    while True:
        try:
            kind, _tag, obs, action = events.get_nowait()
        except queue.Empty:
            break
        if kind == "DONE":
            finished += 1  # one per player per game (the worker drops finished battles itself)
        if kind != "IMITATION":
            continue
        steps += 1
        if obs.shape != (asm.total_dim,):
            bad_shape += 1
            continue
        if isinstance(action, tuple):  # (action, potential) since the shaped reward was added
            action = action[0]
        if obs[m0:m1][int(action)] < 0.5:
            illegal += 1
        kinds["move" if action < 4 else "switch" if action < 10 else "tera move"] += 1

    print(f"{finished // 2} games finished, {steps} teacher steps recorded")
    print(f"actions: {dict(kinds)}")
    print(f"observations with the wrong size: {bad_shape}")
    print(f"teacher actions outside the action mask: {illegal}")
    ok = steps > 0 and bad_shape == 0 and illegal == 0 and finished == 2 * args.games
    print("OK" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
