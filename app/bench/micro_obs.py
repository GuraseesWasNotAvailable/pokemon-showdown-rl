#!/usr/bin/env python
"""
Micro-benchmark of everything a rollout worker does in Python for one decision, without a
Showdown server, Ray or a GPU.

It replays recorded player message streams (bench/micro_showdown_sim.py --record) through the
REAL code path of worker.py: poke-env's PSClient._handle_message (with worker.py's patches) ->
SyncBridgePlayer._handle_battle_message -> poke-env's battle parser -> choose_move ->
ObservationAssembler.assemble -> the reward potential -> the order sent back. The only fakes
are the websocket (messages come from the file, orders are dropped) and the inference call
(a random legal action is returned at once).

It reports milliseconds per decision for the whole path and for each part, optionally a
cProfile listing, and saves real observations for bench/micro_gpu.py and
bench/micro_serialize.py.

    python bench/micro_obs.py --streams <dir>/streams.jsonl --battles 120 --out <dir>
    python bench/micro_obs.py --streams <dir>/streams.jsonl --mode imitation --teacher simple
    python bench/micro_obs.py --streams <dir>/streams.jsonl --profile
    python bench/micro_obs.py --streams <dir>/streams.jsonl --out <other dir> --no-save \
        --compare <dir>/obs_samples.npz        # after changing the assembler: still identical?

Run it from anywhere: it changes into the repository root itself (vocab.json is opened by a
relative path).
"""
from __future__ import annotations

import argparse
import asyncio
import cProfile
import dataclasses
import io
import json
import os
import pstats
import random
import sys
import time
import timeit
from collections import defaultdict
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Acc:
    __slots__ = ("ns", "n")

    def __init__(self) -> None:
        self.ns = 0
        self.n = 0


def timed(fn, acc: Acc):
    pc = time.perf_counter_ns

    def wrapper(*a, **k):
        t0 = pc()
        try:
            return fn(*a, **k)
        finally:
            acc.ns += pc() - t0
            acc.n += 1

    return wrapper


def load_streams(path: str, battles: int) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["b"] >= battles:
                break
            out.append(d)
    return out


class InstantQueue:
    """
    Stands in for the worker's event queue. A STEP is answered on the spot with a random legal
    action, which is what the RL thread would do after the inference call returns.
    """

    def __init__(self, action_futures: dict, mask_slice: slice, rng: random.Random, keep_obs: int):
        self.action_futures = action_futures
        self.mask_slice = mask_slice
        self.rng = rng
        self.steps = 0
        self.done = 0
        self.keep_obs = keep_obs
        self.obs_by_tag: Dict[str, list] = defaultdict(list)
        self.kept = 0
        self.legal_counts: Dict[int, int] = defaultdict(int)
        self.players: Dict[str, Any] = {}

    def put_nowait(self, item) -> None:
        kind, tag, payload, extra = item
        if kind in ("STEP", "IMITATION"):
            self.steps += 1
            mask = payload[self.mask_slice]
            legal = [i for i in range(mask.shape[0]) if mask[i] > 0.5]
            self.legal_counts[len(legal)] += 1
            if self.kept < self.keep_obs:
                self.obs_by_tag[tag].append(payload)
                self.kept += 1
            if kind == "STEP":
                fut, _loop = self.action_futures.pop(tag)
                fut.set_result(self.rng.choice(legal) if legal else 0)
        elif kind == "DONE":
            self.done += 1
            owner = self.players.get(tag.split("_", 1)[0])
            if owner is not None:
                owner._safe_release()


async def _noop_send(*_a, **_k) -> None:
    return None


def build_players(mode: str, teacher: str, queue_keep: int, seed: int):
    import worker  # applies worker.py's poke-env patches
    from config import RunConfig
    from poke_env import AccountConfiguration, ServerConfiguration

    cfg = RunConfig.default()
    cfg = dataclasses.replace(
        cfg,
        learner=dataclasses.replace(cfg.learner, mode=mode),
        rollout=dataclasses.replace(cfg.rollout, imitation_teacher=teacher),
    )
    action_futures: dict = {}
    sem = asyncio.Semaphore(100000)
    players = {}
    q = None
    for side, name in (("p1", "benchpa"), ("p2", "benchpb")):
        p = worker.SyncBridgePlayer(
            cfg=cfg, event_queue=None, action_futures=action_futures, ep_sem=sem,
            account_configuration=AccountConfiguration(name, None),
            server_configuration=ServerConfiguration("ws://127.0.0.1:1/showdown/websocket", "http://127.0.0.1:1/action.php?"),
            max_concurrent_battles=0, battle_format="gen9randombattle", start_listening=False, log_level=40,
        )
        if q is None:
            a = p.assembler
            m0, m1 = a.offsets["action_mask"]
            q = InstantQueue(action_futures, slice(m0, m1), random.Random(seed), queue_keep)
        p.event_queue = q
        q.players[p.username] = p
        p.ps_client.send_message = _noop_send
        players[side] = p
    return cfg, players, q


async def replay(players, msgs, per_kind: Dict[str, Acc]) -> None:
    pc = time.perf_counter_ns
    for d in msgs:
        client = players[d["p"]].ps_client
        m = d["m"]
        kind = "request" if "\n|request|" in m else "update"
        t0 = pc()
        await client._handle_message(m)
        acc = per_kind[kind]
        acc.ns += pc() - t0
        acc.n += 1


def run_pass(args, instrument: bool, profile: bool = False):
    from poke_env.concurrency import POKE_LOOP

    cfg, players, q = build_players(args.mode, args.teacher, args.keep_obs if not instrument else 0, args.seed)
    parts: Dict[str, Acc] = defaultdict(Acc)
    if instrument:
        import poke_env.player.player as pp
        from poke_env.battle.abstract_battle import AbstractBattle
        from poke_env.battle.battle import Battle

        class _Orjson:
            loads = staticmethod(timed(pp.orjson.loads, parts["request json (orjson.loads)"]))

        pp.orjson = _Orjson
        Battle.parse_request = timed(Battle.parse_request, parts["poke-env parse_request"])
        AbstractBattle.parse_message = timed(AbstractBattle.parse_message, parts["poke-env parse_message (per line)"])
        for p in players.values():
            p.assembler.assemble = timed(p.assembler.assemble, parts["obs assemble"])
            if p.potential is not None:
                p.potential = timed(p.potential, parts["reward potential"])
            p.assembler.map_index_to_order = timed(p.assembler.map_index_to_order, parts["map_index_to_order"])
            p.assembler.map_order_to_index = timed(p.assembler.map_order_to_index, parts["map_order_to_index"])
            p.create_order = timed(p.create_order, parts["create_order"])
        if args.mode != "ppo":
            from poke_env.player.baselines import SimpleHeuristicsPlayer
            SimpleHeuristicsPlayer.choose_move = timed(SimpleHeuristicsPlayer.choose_move, parts["SimpleHeuristics teacher"])

    msgs = load_streams(args.streams, args.battles)
    per_kind: Dict[str, Acc] = defaultdict(Acc)
    prof = cProfile.Profile() if profile else None

    async def go():
        if prof is not None:
            prof.enable()
        try:
            await replay(players, msgs, per_kind)
        finally:
            if prof is not None:
                prof.disable()

    t0 = time.perf_counter()
    c0 = time.process_time()
    asyncio.run_coroutine_threadsafe(go(), POKE_LOOP).result()
    wall = time.perf_counter() - t0
    cpu = time.process_time() - c0
    return {
        "cfg": cfg, "players": players, "queue": q, "parts": parts, "per_kind": per_kind,
        "wall": wall, "cpu": cpu, "messages": len(msgs), "prof": prof,
        "bytes": sum(len(d["m"]) for d in msgs),
    }


def primitives() -> Dict[str, float]:
    """Costs of the operations the assembler is made of, in nanoseconds each."""
    import numpy as np
    from obs_assembler import ObservationAssembler
    from obs_moves import get_move_metadata

    a = ObservationAssembler()
    b16 = np.zeros(a.total_dim, dtype=np.float16)
    b32 = np.zeros(a.total_dim, dtype=np.float32)
    lst = [0.0] * a.total_dim
    table = np.zeros((400, 85), dtype=np.float16)      # what a per-move table of static features would be
    n = 200000
    out = {
        "float16 buffer scalar write": timeit.timeit(lambda: b16.__setitem__(1234, 1.0), number=n) / n * 1e9,
        "float32 buffer scalar write": timeit.timeit(lambda: b32.__setitem__(1234, 1.0), number=n) / n * 1e9,
        "python list scalar write": timeit.timeit(lambda: lst.__setitem__(1234, 1.0), number=n) / n * 1e9,
        "np.clip on a scalar": timeit.timeit(lambda: np.clip(3.7, 0, 8), number=50000) / 50000 * 1e9,
        "min/max on a scalar": timeit.timeit(lambda: max(0, min(8, 3.7)), number=n) / n * 1e9,
        "get_move_metadata(vocab) (called 24x per decision)": timeit.timeit(lambda: get_move_metadata(a.vocab_lists), number=50000) / 50000 * 1e9,
        "calc_buf.fill(0) + copy()": timeit.timeit(lambda: (b16.fill(0), b16.copy()), number=20000) / 20000 * 1e9,
        "slice write of one 85-number move row from a table": timeit.timeit(lambda: b16.__setitem__(slice(4332, 4417), table[7]), number=n) / n * 1e9,
        "np.asarray(list, float16) of total_dim": timeit.timeit(lambda: np.asarray(lst, dtype=np.float16), number=2000) / 2000 * 1e9,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--streams", required=True, help="streams.jsonl written by micro_showdown_sim.py --record")
    ap.add_argument("--battles", type=int, default=120)
    ap.add_argument("--mode", default="ppo", choices=("ppo", "imitation", "warmup"))
    ap.add_argument("--teacher", default="simple", choices=("simple",), help="imitation teacher (psbot needs its node engine and is not benchmarked here)")
    ap.add_argument("--keep-obs", type=int, default=6000, help="observations to save for the GPU and serialisation benchmarks")
    ap.add_argument("--profile", action="store_true", help="also run once under cProfile and print the top functions")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", default="", help="an obs_samples.npz saved earlier from the same streams: report whether the observations are still identical")
    ap.add_argument("--no-save", action="store_true", help="do not write obs_samples.npz (use with --compare so the reference is not replaced)")
    args = ap.parse_args()

    args.streams = os.path.abspath(args.streams)
    if args.out:
        args.out = os.path.abspath(args.out)
        os.makedirs(args.out, exist_ok=True)
    os.chdir(REPO_ROOT)
    sys.path.insert(0, REPO_ROOT)

    import numpy as np

    # Pass 1: clean timing, nothing wrapped. Run twice and keep the second (imports, caches warm).
    run_pass(args, instrument=False)
    clean = run_pass(args, instrument=False)
    q = clean["queue"]
    steps = max(q.steps, 1)
    res: Dict[str, Any] = {
        "mode": args.mode, "teacher": args.teacher if args.mode != "ppo" else None,
        "battles": args.battles, "messages": clean["messages"], "decisions": q.steps, "episodes_finished": q.done,
        "decisions_per_battle": round(q.steps / max(args.battles, 1), 2),
        "bytes_received_per_decision": round(clean["bytes"] / steps, 1),
        "messages_per_decision": round(clean["messages"] / steps, 3),
        "ms_per_decision_total": round(clean["wall"] * 1000 / steps, 4),
        "cpu_ms_per_decision_total": round(clean["cpu"] * 1000 / steps, 4),
        "decisions_per_second_one_core": round(steps / clean["wall"], 1),
        "ms_per_request_message": round(clean["per_kind"]["request"].ns / 1e6 / max(clean["per_kind"]["request"].n, 1), 4),
        "ms_per_update_message": round(clean["per_kind"]["update"].ns / 1e6 / max(clean["per_kind"]["update"].n, 1), 4),
        "one_legal_action_share": round(q.legal_counts.get(1, 0) / steps, 4),
        "legal_action_histogram": {int(k): int(v) for k, v in sorted(q.legal_counts.items())},
    }

    # Observations in learner order: whole episodes back to back, with episode ids.
    obs_arr = None
    if q.kept:
        obs, ep = [], []
        for i, (tag, rows) in enumerate(q.obs_by_tag.items()):
            obs.extend(rows)
            ep.extend([i] * len(rows))
        obs_arr = np.stack(obs, axis=0)
    if args.out and obs_arr is not None and not args.no_save:
        np.savez(os.path.join(args.out, "obs_samples.npz"), obs=obs_arr, ep=np.asarray(ep, dtype=np.int64))
        res["obs_saved"] = {"file": os.path.join(args.out, "obs_samples.npz"), "rows": int(obs_arr.shape[0]),
                            "dim": int(obs_arr.shape[1]), "dtype": str(obs_arr.dtype), "episodes": len(q.obs_by_tag),
                            "nonzero_share": round(float((obs_arr != 0).mean()), 4)}

    # The observations depend only on the recorded messages, never on the action picked here, so
    # the same streams, --battles and --keep-obs give the same array every time. That makes this
    # the check for any change to the assembler: save once before the change, compare after it.
    if args.compare:
        ref = np.load(os.path.abspath(args.compare))["obs"]
        if obs_arr is None:
            res["compare"] = {"error": "nothing kept: raise --keep-obs"}
        elif ref.shape != obs_arr.shape or ref.dtype != obs_arr.dtype:
            res["compare"] = {"identical": False, "reference_shape": list(ref.shape), "shape": list(obs_arr.shape),
                              "reference_dtype": str(ref.dtype), "dtype": str(obs_arr.dtype)}
        else:
            diff = ref.astype(np.float32) - obs_arr.astype(np.float32)
            bad_rows = np.flatnonzero((diff != 0).any(axis=1))
            bad_cols = np.flatnonzero((diff != 0).any(axis=0))
            res["compare"] = {
                "reference": os.path.abspath(args.compare), "rows": int(ref.shape[0]),
                "identical": bool(np.array_equal(ref, obs_arr)),
                "rows_that_differ": int(bad_rows.size), "columns_that_differ": int(bad_cols.size),
                "max_abs_difference": float(np.abs(diff).max()) if diff.size else 0.0,
                "first_columns_that_differ": [int(c) for c in bad_cols[:20]],
            }

    # Pass 2: the same replay with the parts wrapped. In a fresh interpreter state the wrappers
    # stay installed, so this pass comes last (before the profile, which does not mind them).
    inst = run_pass(args, instrument=True)
    isteps = max(inst["queue"].steps, 1)
    total_ms = inst["wall"] * 1000 / isteps
    parts = {}
    known = 0.0
    for name, acc in sorted(inst["parts"].items(), key=lambda kv: -kv[1].ns):
        ms = acc.ns / 1e6 / isteps
        parts[name] = {"ms_per_decision": round(ms, 4), "calls_per_decision": round(acc.n / isteps, 2),
                       "us_per_call": round(acc.ns / 1e3 / max(acc.n, 1), 2)}
        # parse_request and the teacher are not nested inside each other; assemble, potential and
        # the order mapping are siblings inside choose_move. So the sum is a fair "accounted" figure.
        known += ms
    parts["everything else (split, asyncio, locks, event harvest, queue)"] = {
        "ms_per_decision": round(max(total_ms - known, 0.0), 4)}
    res["parts_instrumented_pass"] = parts
    res["ms_per_decision_instrumented_pass"] = round(total_ms, 4)

    res["primitive_costs_ns"] = {k: round(v, 1) for k, v in primitives().items()}

    if args.profile:
        pr = run_pass(args, instrument=False, profile=True)
        s = io.StringIO()
        st = pstats.Stats(pr["prof"], stream=s)
        st.sort_stats("tottime").print_stats(35)
        text = s.getvalue()
        psteps = max(pr["queue"].steps, 1)
        res["profile_decisions"] = psteps
        if args.out:
            with open(os.path.join(args.out, f"profile_{args.mode}.txt"), "w", encoding="utf-8") as f:
                f.write(f"decisions: {psteps}\n" + text)
        print(text[:9000])

    if args.out:
        with open(os.path.join(args.out, f"obs_bench_{args.mode}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
