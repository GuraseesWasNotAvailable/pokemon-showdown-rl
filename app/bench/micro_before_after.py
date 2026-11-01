#!/usr/bin/env python
"""
The code before the speed changes and the code after them, timed side by side in one process.

"Before" is the reference commit of bench/micro_ref.py, imported next to the working tree, so
both run on the same inputs, the same GPU and the same minute of machine load, taking turns.
Sections (pick with --sections, comma separated):

  update   ppo_core.ppo_update, the learner's whole update loop, in PPO mode: milliseconds per
           minibatch and peak GPU memory. Covers the block mask from episode ids and the
           update's data held on the GPU.
  infer    InferenceActor.infer_batch at several batch sizes: milliseconds per call. Covers
           the constant block mask, the pooled history and float16 observations.
  gae      ppo_core.gae_from_episode: microseconds per step.
  obs      ObservationAssembler.assemble on recorded battles (--streams, written by
           bench/micro_showdown_sim.py --record): milliseconds per decision, both assemblers
           called on the same battle at every decision.
  cast     what the worker does to a batch before sending it: stack, against stack and widen.

    python bench/micro_before_after.py --obs <dir>/obs_samples.npz --streams <dir>/streams.jsonl --out <file>.json

It uses the GPU in bursts for about a minute. No Ray, no servers, no training run. Whether
old and new compute the same thing is not this script's business: tests/test_speed_*.py
compare them bit for bit. All output is ASCII.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Remote:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *a, **k):
        return self._fn(*a, **k)


class _Store:
    """Stands in for the weight store actor: takes what the learner sends, has nothing new for inference."""
    update = _Remote(lambda *a: None)
    set_temp = _Remote(lambda *a: None)


def med(xs: List[float]) -> float:
    return float(statistics.median(xs)) if xs else float("nan")


def sync(dev: str) -> None:
    if dev.startswith("cuda"):
        import torch
        torch.cuda.synchronize()


def make_learner(learner_mod, core_mod, config_mod, dev: str):
    """A LearnerActor's network, optimizer and scheduler without Ray or its loops."""
    cfg = config_mod.RunConfig.default()
    cfg = dataclasses.replace(cfg, learner=dataclasses.replace(cfg.learner, mode="ppo", device=dev, resume=False))
    la = object.__new__(learner_mod.LearnerActor)
    la.run_cfg, la.cfg, la.weight_store = cfg, cfg.learner, _Store()
    la.net = la.opt = la.sched = None
    la.dataset = core_mod.AsyncEpisodeDataset(act_dim=cfg.env.act_dim, device="cpu")
    la.update_idx = la.total_episodes = la.total_steps = la.ep_counter = 0
    la._init_if_needed()
    return la


def update_columns(obs_np, ep_np, n: int, mask_slice: slice, seed: int = 0):
    import numpy as np
    import torch
    rng = np.random.default_rng(seed)
    mask = obs_np[:n, mask_slice].astype(np.float32) > 0.5
    act = np.array([rng.choice(np.flatnonzero(m)) if m.any() else 0 for m in mask], dtype=np.int64)
    logp = (-np.log(np.maximum(mask.sum(axis=1), 1.0))).astype(np.float32)
    z = torch.zeros(n)
    return (torch.from_numpy(np.ascontiguousarray(obs_np[:n])), torch.from_numpy(act), torch.from_numpy(logp), z,
            torch.from_numpy(rng.standard_normal(n).astype(np.float32)),
            torch.from_numpy(np.clip(rng.standard_normal(n) * 0.5, -1.4, 1.4).astype(np.float32)), z,
            torch.from_numpy(np.ascontiguousarray(ep_np[:n])).long())


def section_update(ref, old, obs_np, ep_np, dev: str, steps: int, mb: int, T: int, accum: int, rounds: int, mask_slice) -> Dict[str, Any]:
    import torch
    import config as new_config
    import learner as new_learner
    import ppo_core as new_core

    with ref.active():
        a = make_learner(old["learner"], old["core"], old["config"], dev)
    b = make_learner(new_learner, new_core, new_config, dev)
    b.net.load_state_dict(a.net.state_dict(), strict=True)
    b.opt.sync_masters_from_model()
    cols = update_columns(obs_np, ep_np, steps, mask_slice)

    def run(core_mod, la) -> Dict[str, float]:
        ds = core_mod.AsyncEpisodeDataset(14, "cpu")
        ds.add_steps(*cols)
        kw = dict(la.cfg.ppo_kwargs())
        kw.update(minibatch_size=mb, batch_seq_len=T, grad_accum_steps=accum, target_kl=None)   # no early stop: equal work
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        sync(dev)
        t0 = time.perf_counter()
        st = core_mod.ppo_update(net=la.net, opt=la.opt, dataset=ds, scheduler=la.sched, mode="ppo", target_net=None, **kw,
                                 v_min=la.cfg.v_min, v_max=la.cfg.v_max, v_bins=la.cfg.v_bins)
        sync(dev)
        sec = time.perf_counter() - t0
        out = {"seconds": sec, "minibatches": st.n_mb, "ms_per_minibatch": sec * 1000.0 / max(st.n_mb, 1)}
        if dev.startswith("cuda"):
            out["peak_memory_mib"] = torch.cuda.max_memory_allocated() / 2**20
        return out

    run(old["core"], a), run(new_core, b)                      # the first call of each compiles
    res: Dict[str, List[Dict[str, float]]] = {"old": [], "new": []}
    for _ in range(rounds):
        res["old"].append(run(old["core"], a))
        res["new"].append(run(new_core, b))
    out: Dict[str, Any] = {"steps": steps, "minibatch": mb, "seq_len": T, "grad_accum": accum, "epochs": a.cfg.update_epochs, "rounds": rounds,
                           "data_on_gpu_by_the_automatic_rule": bool(new_core.update_data_fits(_as_dataset(new_core, cols), next(b.net.parameters()).device))}
    for name in ("old", "new"):
        out[name] = {"ms_per_minibatch": round(med([r["ms_per_minibatch"] for r in res[name]]), 2),
                     "ms_per_minibatch_all": [round(r["ms_per_minibatch"], 2) for r in res[name]],
                     "minibatches": int(res[name][0]["minibatches"]),
                     "peak_memory_mib": round(max(r.get("peak_memory_mib", 0.0) for r in res[name]), 1)}
    out["speedup"] = round(out["old"]["ms_per_minibatch"] / out["new"]["ms_per_minibatch"], 3)
    del a, b
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def _as_dataset(core_mod, cols):
    ds = core_mod.AsyncEpisodeDataset(14, "cpu")
    ds.add_steps(*cols)
    return ds


def section_infer(ref, old, obs_np, dev: str, batch_sizes: List[int], iters: int) -> Dict[str, Any]:
    import numpy as np
    import config as new_config
    import inference as new_inf
    import ppo_core as new_core

    def cfg_for(config_mod):
        cfg = config_mod.RunConfig.default()
        return dataclasses.replace(cfg, learner=dataclasses.replace(cfg.learner, resume=False),
                                   infer=dataclasses.replace(cfg.infer, device=dev),
                                   rollout=dataclasses.replace(cfg.rollout, target_concurrent_battles=128, rooms_per_pair=16))

    # Each constructor wraps its module's flex_attention in torch.compile: put the shared one back.
    saved_old, saved_new = old["core"].flex_attention, new_core.flex_attention
    try:
        with ref.active():
            a = old["inference"].InferenceActor(cfg_for(old["config"]), None)
        b = new_inf.InferenceActor(cfg_for(new_config), None)
    finally:
        old["core"].flex_attention, new_core.flex_attention = saved_old, saved_new
    b.net.load_state_dict(a.net.state_dict(), strict=True)
    a._last_sync_time = b._last_sync_time = time.time() + 1e9       # there is no weight store here
    out: Dict[str, Any] = {}
    for B in batch_sizes:
        if B > obs_np.shape[0]:
            continue
        obs16 = np.ascontiguousarray(obs_np[:B])
        obs32 = np.ascontiguousarray(obs16, dtype=np.float32)        # what the old worker sent
        tags = [f"b{B}_{i}" for i in range(B)]
        for _ in range(4):
            a.infer_batch(tags, obs32), b.infer_batch(tags, obs16)
        ta, tb = [], []
        for _ in range(iters):
            sync(dev)
            t0 = time.perf_counter()
            a.infer_batch(tags, obs32)
            sync(dev)
            t1 = time.perf_counter()
            b.infer_batch(tags, obs16)
            sync(dev)
            t2 = time.perf_counter()
            ta.append((t1 - t0) * 1000.0)
            tb.append((t2 - t1) * 1000.0)
        out[str(B)] = {"old_ms_per_call": round(med(ta), 3), "new_ms_per_call": round(med(tb), 3),
                       "old_ms_per_decision": round(med(ta) / B, 4), "new_ms_per_decision": round(med(tb) / B, 4),
                       "speedup": round(med(ta) / med(tb), 3)}
        for t in tags:
            a.clear_cache(t), b.clear_cache(t)
    del a, b
    return out


def section_gae(old) -> Dict[str, Any]:
    import torch
    import ppo_core as new_core
    out: Dict[str, Any] = {}
    for T in (10, 40, 120):
        r, v, d, var = torch.randn(T) * 0.1, torch.randn(T), torch.zeros(T), torch.rand(T)
        d[-1] = 1.0
        row: Dict[str, Any] = {}
        for name, variances in (("dynamic_lambda", var), ("plain", None)):
            times = {}
            for label, core in (("old", old["core"]), ("new", new_core)):
                n = 300
                t0 = time.perf_counter()
                for _ in range(n):
                    core.gae_from_episode(r, v, d, gamma=0.999, lam=0.75, variances=variances)
                times[label] = (time.perf_counter() - t0) / n / T * 1e6
            row[name] = {"old_us_per_step": round(times["old"], 2), "new_us_per_step": round(times["new"], 2),
                         "speedup": round(times["old"] / times["new"], 1)}
        out[f"episode_of_{T}"] = row
    return out


def section_obs(ref, streams: str, battles: int) -> Dict[str, Any]:
    from poke_env.concurrency import POKE_LOOP
    import micro_obs

    with ref.active():
        old_asm = ref.load("obs_assembler").ObservationAssembler()
    msgs = micro_obs.load_streams(streams, battles)
    pc = time.perf_counter_ns
    acc = {"old": 0, "new": 0, "n": 0}

    def watch(player) -> None:
        inner = player.assembler.assemble

        def assemble(battle, events=None):
            # which of the two goes first alternates, so neither always pays for cold CPU caches
            if acc["n"] & 1:
                t0 = pc()
                old_asm.assemble(battle, events=events)
                t1 = pc()
                got = inner(battle, events=events)
                t2 = pc()
                acc["old"] += t1 - t0
                acc["new"] += t2 - t1
            else:
                t0 = pc()
                got = inner(battle, events=events)
                t1 = pc()
                old_asm.assemble(battle, events=events)
                t2 = pc()
                acc["new"] += t1 - t0
                acc["old"] += t2 - t1
            acc["n"] += 1
            return got

        player.assembler.assemble = assemble

    out: Dict[str, Any] = {}
    warm = None
    for label in ("first_pass_tables_empty", "second_pass_tables_filled"):
        _cfg, players, _q = micro_obs.build_players("ppo", "simple", 0, 7)
        if warm is None:
            for p in players.values():
                watch(p)
            warm = {side: p.assembler for side, p in players.items()}
        else:
            # The tables belong to the assembler. A long run has them filled: carry the first
            # pass's assemblers (and their timing wrappers) over to the new players.
            for side, p in players.items():
                p.assembler = warm[side]
        acc.update(old=0, new=0, n=0)
        asyncio.run_coroutine_threadsafe(micro_obs.replay(players, msgs, {"request": micro_obs.Acc(), "update": micro_obs.Acc()}), POKE_LOOP).result()
        n = max(acc["n"], 1)
        out[label] = {"decisions": acc["n"], "old_ms_per_decision": round(acc["old"] / 1e6 / n, 4),
                      "new_ms_per_decision": round(acc["new"] / 1e6 / n, 4), "speedup": round(acc["old"] / max(acc["new"], 1), 2)}
    return out


def section_cast(obs_np) -> Dict[str, Any]:
    import numpy as np
    out = {}
    for B in (8, 29, 64, 128):
        rows = [np.ascontiguousarray(obs_np[i % obs_np.shape[0]]) for i in range(B)]
        n = 400
        t0 = time.perf_counter()
        for _ in range(n):
            np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.float32)
        old = (time.perf_counter() - t0) / n * 1e3
        t0 = time.perf_counter()
        for _ in range(n):
            np.stack(rows, axis=0)
        new = (time.perf_counter() - t0) / n * 1e3
        out[str(B)] = {"old_ms_per_call": round(old, 4), "new_ms_per_call": round(new, 4),
                       "old_bytes": B * obs_np.shape[1] * 4, "new_bytes": B * obs_np.shape[1] * 2}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs", default=os.environ.get("PSPPO_OBS_SAMPLES", ""), help="obs_samples.npz from bench/micro_obs.py (made-up observations if absent)")
    ap.add_argument("--streams", default="", help="streams.jsonl from bench/micro_showdown_sim.py --record, for the obs section")
    ap.add_argument("--battles", type=int, default=60, help="recorded battles to replay in the obs section")
    ap.add_argument("--sections", default="update,infer,gae,obs,cast")
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=2048, help="steps in the timed update")
    ap.add_argument("--mb", type=int, default=256)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--accum", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3, help="old and new take turns this many times")
    ap.add_argument("--batch-sizes", default="1,8,32,64,128")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    obs_path = os.path.abspath(args.obs) if args.obs else ""
    streams = os.path.abspath(args.streams) if args.streams else ""
    out_path = os.path.abspath(args.out) if args.out else ""
    os.chdir(REPO_ROOT)
    for p in (REPO_ROOT, os.path.join(REPO_ROOT, "bench")):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch
    import micro_ref
    import ppo_core as new_core
    from obs_assembler import ObservationAssembler

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ref = micro_ref.Reference.export()
    old = {name: ref.load(name) for name in ("ppo_core", "learner", "config", "inference")}
    old["core"] = old.pop("ppo_core")
    if dev.startswith("cuda"):
        micro_ref.shared_compiled_flex_attention(old["core"], new_core)
    old["core"]._learner_flex_compiled = new_core._learner_flex_compiled = True
    real = micro_ref.real_observations(obs_path) if obs_path else None
    if real is not None:
        obs_np, ep_np, kind = real[0], real[1], "real"
    else:
        obs_np, ep_np = micro_ref.synthetic_observations(4096)
        kind = "synthetic"
    m0, m1 = ObservationAssembler().offsets["action_mask"]
    sections = [s.strip() for s in args.sections.split(",") if s.strip()]
    res: Dict[str, Any] = {"device": dev, "torch": torch.__version__, "reference_commit": micro_ref.REF_COMMIT,
                           "gpu": torch.cuda.get_device_name(0) if dev.startswith("cuda") else None,
                           "observations": kind, "modules_changed_since_reference": ref.differs_from_tree()}
    for s in sections:
        t0 = time.perf_counter()
        try:
            if s == "update":
                if not dev.startswith("cuda"):
                    res[s] = {"skipped": "the attention has no backward pass on a CPU"}
                else:
                    res[s] = section_update(ref, old, obs_np, ep_np, dev, min(args.steps, (obs_np.shape[0] // args.mb) * args.mb),
                                            args.mb, args.seq, args.accum, args.rounds, slice(m0, m1))
            elif s == "infer":
                res[s] = section_infer(ref, old, obs_np, dev, [int(x) for x in args.batch_sizes.split(",") if x.strip()], args.iters)
            elif s == "gae":
                res[s] = section_gae(old)
            elif s == "obs":
                res[s] = section_obs(ref, streams, args.battles) if streams else {"skipped": "no --streams given"}
            elif s == "cast":
                res[s] = section_cast(obs_np)
            else:
                res[s] = {"error": "unknown section"}
        except Exception as e:  # noqa: BLE001
            import traceback
            res[s] = {"error": repr(e)[:400], "trace": traceback.format_exc()[-1500:]}
        res.setdefault("section_seconds", {})[s] = round(time.perf_counter() - t0, 1)
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
