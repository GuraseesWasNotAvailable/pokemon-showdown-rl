"""
Recorded battles replayed through the worker's own path, for tests/test_obs_matchup.py and
tests/test_widen.py (not a test file itself).

The replay is tests/test_speed_obs.py's: bench/micro_obs.build_players (worker.SyncBridgePlayer
with worker.py's poke-env patches) and bench/micro_obs.replay over the players' message
streams that bench/micro_showdown_sim.py --record writes. At every decision the player's own
assembler (cfg.make_assembler(), the observation training uses) runs as always, and each extra
assembler given here is handed the same battle object and the same turn events.

    PSPPO_STREAMS   one or more streams.jsonl files, separated by the path separator
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np


def stream_paths() -> List[str]:
    return [p for p in os.environ.get("PSPPO_STREAMS", "").split(os.pathsep) if p.strip()]


def replay(extra: Dict[str, Any], max_battles: Optional[int] = None) -> Dict[str, Any]:
    """
    Replays the recorded streams. Returns {"records": [...], "time": {...}} where each record is
    {"key": (player, battle tag), "turn", "default": obs, <name>: obs or the exception it raised}.
    Records are in decision order; one key's records are one side of one battle, in order.
    """
    from bench import micro_obs
    from poke_env.concurrency import POKE_LOOP

    records: List[Dict[str, Any]] = []
    spent: Dict[str, float] = defaultdict(float)

    def watch(player: Any) -> None:
        inner = player.assembler.assemble

        def assemble(battle: Any, events: Any = None) -> np.ndarray:
            t0 = time.perf_counter()
            got = inner(battle, events=events)
            spent["default"] += time.perf_counter() - t0
            rec: Dict[str, Any] = {"key": (player.username, battle.battle_tag), "turn": battle.turn, "default": got}
            for name, asm in extra.items():
                t0 = time.perf_counter()
                try:
                    rec[name] = asm.assemble(battle, events=events)
                except Exception as e:  # noqa: BLE001  (the test decides what an exception means)
                    rec[name] = e
                spent[name] += time.perf_counter() - t0
            records.append(rec)
            return got

        player.assembler.assemble = assemble

    for path in stream_paths():
        _cfg, players, _queue = micro_obs.build_players("ppo", "simple", 0, 7)
        for p in players.values():
            watch(p)
        msgs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if max_battles is not None and d["b"] >= max_battles:
                    break
                msgs.append(d)
        asyncio.run_coroutine_threadsafe(micro_obs.replay(players, msgs, defaultdict(micro_obs.Acc)), POKE_LOOP).result()
    return {"records": records, "time": dict(spent)}


def sequences(records: List[Dict[str, Any]], name: str, min_decisions: int) -> List[np.ndarray]:
    """Whole sides of battles, in the order they first appear, as [T, dim] arrays of `name`'s observations, until min_decisions are covered."""
    by_key: Dict[Any, List[np.ndarray]] = {}
    order: List[Any] = []
    for r in records:
        if r["key"] not in by_key:
            by_key[r["key"]] = []
            order.append(r["key"])
        by_key[r["key"]].append(r[name])
    out, n = [], 0
    for k in order:
        if n >= min_decisions:
            break
        out.append(np.stack(by_key[k]))
        n += len(by_key[k])
    return out
