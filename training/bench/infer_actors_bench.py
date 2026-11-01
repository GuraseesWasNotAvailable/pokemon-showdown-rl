"""
How many decisions a second the inference service can give: N InferenceActors behind Ray, driven by
stand-in rollout workers that only call infer_batch (no Showdown, no observation building), the
way train.py wires them (infer_pool.plan_gpus, assign_workers, actor_options). It measures the
part of the run the speed profile of 2026-10-17 found to be the limit: the actor's Python thread
and Ray's transport, not the GPU.

    python bench/infer_actors_bench.py --actors 1 2 3 --workers 36 --rows 56 --seconds 40 [--fast-sample]

Each stand-in worker alternates two groups of `rows` battle sides (a real worker alternates its two
seats' decisions), each group its own tags, so the actor's history pool is used as in a run. The
model is the live run's (--obs-matchup --obs-movesets) with random weights; the observations are
zeros with every action legal, which costs the forward pass what real ones do. One JSON line per
setting: decisions a second, calls a second, mean ms a call as a worker sees it, and per actor.

GPU placement follows PSPPO_GPU_SPLIT and CUDA_VISIBLE_DEVICES, as in train.py. Starts its own
local Ray and stops it at the end of each setting.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np  # noqa: E402


class StandInWorker:
    def __init__(self, cfg, infer, rows: int, index: int):
        self.infer = infer
        asm = cfg.make_assembler()
        m0, m1 = asm.offsets["action_mask"]
        self.obs = np.zeros((rows, int(asm.total_dim)), dtype=np.float16)
        self.obs[:, m0:m1] = 1.0
        self.groups = [[f"w{index}_{g}_{i}" for i in range(rows)] for g in (0, 1)]

    def obs_dim(self) -> int:
        return int(self.obs.shape[1])

    def run(self, seconds: float) -> dict:
        import ray
        calls, rows, waited = 0, 0, 0.0
        end = time.time() + seconds
        g = 0
        while time.time() < end:
            t0 = time.perf_counter()
            ray.get(self.infer.infer_batch.remote(self.groups[g], self.obs))
            waited += time.perf_counter() - t0
            calls += 1
            rows += self.obs.shape[0]
            g ^= 1
        return {"calls": calls, "rows": rows, "waited": waited}


def one_setting(n_actors: int, workers: int, rows: int, seconds: float, fast_sample: bool, sync_s) -> dict:
    import ray
    import train
    import infer_pool
    from inference import InferenceActor, WeightStore

    args = train.parse_args(["--obs-matchup", "--obs-movesets", "--battles", str(rows * workers), "--rooms-per-pair", "16",
                             "--mode", "ppo"])
    cfg = train.build_config(args)
    cfg = dataclasses.replace(cfg, learner=dataclasses.replace(cfg.learner, resume=False))
    ray.init(address="local", include_dashboard=False, log_to_driver=False)
    try:
        cfg_ref = ray.put(cfg)
        store = WeightStore.remote()
        speed = infer_pool.SpeedOptions(infer_actors=n_actors, fast_sample=fast_sample, infer_sync_s=sync_s)
        shares, learner_share = infer_pool.plan_gpus(n_actors, float(ray.cluster_resources().get("GPU", 0)))
        Remote = ray.remote(InferenceActor)
        actors = [Remote.options(**infer_pool.actor_options(s, 0)).remote(cfg_ref, store, **speed.actor_kwargs()) for s in shares]
        actor_of = infer_pool.assign_workers([1] * workers, n_actors)
        W = ray.remote(num_cpus=0)(StandInWorker)
        ws = [W.remote(cfg_ref, actors[actor_of[i]], rows, i) for i in range(workers)]
        dim = ray.get(ws[0].obs_dim.remote())
        ray.get([w.run.remote(8.0) for w in ws])                     # warm up: compile, history pool, Ray's channels
        t0 = time.perf_counter()
        out = ray.get([w.run.remote(seconds) for w in ws])
        dt = time.perf_counter() - t0
        calls = sum(o["calls"] for o in out)
        per_actor = [sum(o["rows"] for o, a in zip(out, actor_of) if a == k) / dt for k in range(n_actors)]
        stats = ray.get([a.get_stats.remote() for a in actors])
        return {"actors": n_actors, "workers": workers, "rows_per_call": rows, "fast_sample": fast_sample, "obs_dim": dim,
                "gpu_shares": shares, "seconds": round(dt, 1),
                "decisions_per_s": round(sum(o["rows"] for o in out) / dt),
                "calls_per_s": round(calls / dt, 1),
                "ms_per_call_seen_by_worker": round(1000 * sum(o["waited"] for o in out) / max(1, calls), 2),
                "decisions_per_s_by_actor": [round(x) for x in per_actor],
                "avg_batch_by_actor": [s["avg_batch_size"] for s in stats]}
    finally:
        ray.shutdown()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actors", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--workers", type=int, default=36)
    ap.add_argument("--rows", type=int, default=56, help="rows per call (the live run: about 0.88 x battles per worker)")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--fast-sample", action="store_true")
    ap.add_argument("--both", action="store_true", help="each actor count with and without --fast-sample")
    ap.add_argument("--sync-s", type=float, default=None)
    a = ap.parse_args()
    for n in a.actors:
        for fast in ((False, True) if a.both else (a.fast_sample,)):
            print(json.dumps(one_setting(n, a.workers, a.rows, a.seconds, fast, a.sync_s)), flush=True)


if __name__ == "__main__":
    main()
