#!/usr/bin/env python
"""
Micro-benchmark of how observations travel: the worker's batch assembly, the serialisation Ray
does for one inference call and one learner submission, and (with --with-ray) the real round
trip through a small local Ray instance to a do-nothing actor.

    python bench/micro_serialize.py --obs <dir>/obs_samples.npz --out <dir>
    python bench/micro_serialize.py --obs <dir>/obs_samples.npz --out <dir> --with-ray

Without --with-ray nothing is started. With it, a private two-CPU Ray instance is started and
shut down again (ray.shutdown stops exactly the processes this script started). The actor
never touches the GPU.

The --with-ray pass also prints how Ray's actor processes look to psutil on this OS, and how
the actor class can be read from Ray's own worker log files. bench/throughput.py relies on it.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import statistics
import sys
import time
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def med_ms(fn, n: int) -> float:
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(ts))


def local_costs(obs16, batch_sizes: List[int]) -> Dict[str, Any]:
    import numpy as np
    out: Dict[str, Any] = {"obs_dim": int(obs16.shape[1]), "bytes_per_obs_float16": int(obs16.shape[1] * 2),
                           "bytes_per_obs_float32": int(obs16.shape[1] * 4), "inference_call": {}, "learner_submission": {}}
    for B in batch_sizes:
        rows = [obs16[i].copy() for i in range(B)]           # what choose_move puts on the queue
        tags = [f"p1a2b3c{i:03d}a_battle-gen9randombattle-{1000 + i}" for i in range(B)]
        as_sent = np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.float32)
        as_f16 = np.stack(rows, axis=0)

        def ser(x):
            bufs: List[Any] = []
            head = pickle.dumps((tags, x), protocol=5, buffer_callback=bufs.append)
            return head, bufs

        head, bufs = ser(as_sent)
        out["inference_call"][B] = {
            "stack_and_cast_to_float32_ms": round(med_ms(lambda: np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.float32), 200), 4),
            "stack_only_float16_ms": round(med_ms(lambda: np.stack(rows, axis=0), 200), 4),
            "pickle5_header_ms": round(med_ms(lambda: ser(as_sent), 200), 4),
            "payload_bytes_float32": int(as_sent.nbytes),
            "payload_bytes_float16": int(as_f16.nbytes),
            "copy_payload_once_float32_ms": round(med_ms(lambda: as_sent.copy(), 200), 4),
            "copy_payload_once_float16_ms": round(med_ms(lambda: as_f16.copy(), 200), 4),
            "above_ray_inline_limit_100KB": bool(as_sent.nbytes > 100 * 1024),
            "reply_bytes": int(B * (8 + 4 + 4 + 4)),
        }
    # One learner submission: up to learn_max_episodes (32) episodes of about 40 decisions.
    for eps, T in ((1, 40), (8, 40), (32, 40)):
        n = min(eps * T, obs16.shape[0])
        ep_obs = [obs16[i * T:(i + 1) * T].copy() for i in range(n // T)]
        out["learner_submission"][f"{eps} episodes x {T} steps"] = {
            "np_stack_one_episode_ms": round(med_ms(lambda: np.stack([r for r in ep_obs[0]], axis=0), 100), 4),
            "concatenate_ms": round(med_ms(lambda: np.concatenate(ep_obs, axis=0), 50), 4),
            "payload_megabytes_float16": round(sum(e.nbytes for e in ep_obs) / 2**20, 2),
        }
    return out


def ray_round_trips(obs16, batch_sizes: List[int], calls: int) -> Dict[str, Any]:
    import numpy as np
    import psutil
    import ray

    out: Dict[str, Any] = {"ray": ray.__version__}
    t0 = time.perf_counter()
    ctx = ray.init(num_cpus=2, include_dashboard=False, object_store_memory=300 * 1024 * 1024,
                   ignore_reinit_error=True, log_to_driver=False)
    out["ray_init_seconds"] = round(time.perf_counter() - t0, 2)
    try:
        @ray.remote(num_cpus=0.5)
        class EchoInfer:
            """Same call shapes as InferenceActor.infer_batch and LearnerActor.submit_packed_batch."""

            def pid(self):
                return os.getpid()

            def infer_batch(self, tags, obs_np):
                B = obs_np.shape[0]
                return (np.zeros(B, dtype=np.int64), np.zeros(B, dtype=np.float32),
                        np.zeros(B, dtype=np.float32), np.zeros(B, dtype=np.float32))

            def submit_packed_batch(self, obs_cat, act, logp, val, var, rew, done, lengths):
                return True

        actor = EchoInfer.remote()
        apid = ray.get(actor.pid.remote())
        infer: Dict[str, Any] = {}
        for B in batch_sizes:
            tags = [f"p1a2b3c{i:03d}a_battle-gen9randombattle-{1000 + i}" for i in range(B)]
            o32 = np.ascontiguousarray(obs16[:B], dtype=np.float32)
            o16 = np.ascontiguousarray(obs16[:B])
            for _ in range(20):
                ray.get(actor.infer_batch.remote(tags, o32))
            row = {}
            for name, o in (("float32_as_sent_today", o32), ("float16", o16)):
                ts = []
                for _ in range(calls):
                    t1 = time.perf_counter()
                    ray.get(actor.infer_batch.remote(tags, o))
                    ts.append((time.perf_counter() - t1) * 1000.0)
                ts.sort()
                row[name] = {"ms_per_call_median": round(statistics.median(ts), 3), "ms_per_call_p90": round(ts[int(0.9 * len(ts))], 3),
                             "ms_per_decision": round(statistics.median(ts) / B, 4), "payload_kilobytes": round(o.nbytes / 1024, 1)}
            infer[B] = row
        out["infer_batch_round_trip"] = infer

        sub: Dict[str, Any] = {}
        for eps, T in ((8, 40), (32, 40)):
            n = min(eps * T, obs16.shape[0])
            obs_cat = np.ascontiguousarray(obs16[:n])
            f = np.zeros(n, dtype=np.float32)
            args = (obs_cat, np.zeros(n, dtype=np.int64), f, f, f, f, f, np.full(n // T, T, dtype=np.int32))
            for _ in range(5):
                ray.get(actor.submit_packed_batch.remote(*args))
            ts = []
            for _ in range(max(calls // 5, 10)):
                t1 = time.perf_counter()
                ray.get(actor.submit_packed_batch.remote(*args))
                ts.append((time.perf_counter() - t1) * 1000.0)
            sub[f"{eps} episodes x {T} steps"] = {"ms_per_call_median": round(statistics.median(ts), 3),
                                                  "payload_megabytes": round(obs_cat.nbytes / 2**20, 2),
                                                  "ms_per_decision": round(statistics.median(ts) / n, 5)}
        out["submit_packed_batch_round_trip"] = sub

        # How the actor process looks from outside (throughput.py classifies by this).
        p = psutil.Process(apid)
        try:
            cmd = p.cmdline()
        except Exception as e:  # noqa: BLE001
            cmd = [repr(e)]
        session_dir = ctx.address_info.get("session_dir") if hasattr(ctx, "address_info") else None
        marker = None
        if session_dir:
            logs = os.path.join(session_dir, "logs")
            deadline = time.time() + 5.0
            while marker is None and time.time() < deadline:
                for fn in os.listdir(logs):
                    if fn.startswith("worker-") and fn.endswith(f"-{apid}.out"):
                        with open(os.path.join(logs, fn), "r", encoding="utf-8", errors="replace") as fh:
                            for line in fh:
                                if line.startswith(":actor_name:"):
                                    marker = {"file": fn, "line": line.strip()}
                                    break
                if marker is None:
                    time.sleep(0.25)
        out["actor_process_as_psutil_sees_it"] = {
            "platform": sys.platform, "pid": apid, "name": p.name(), "cmdline_first_3": cmd[:3],
            "cmdline_mentions_class": any("EchoInfer" in c for c in cmd),
            "parent_name": (p.parent().name() if p.parent() else None),
            "driver_pid": os.getpid(),
            "is_descendant_of_driver": os.getpid() in [a.pid for a in p.parents()],
            "ray_worker_log_marker": marker, "session_dir": session_dir,
        }
        kids = psutil.Process(os.getpid()).children(recursive=True)
        names: Dict[str, int] = {}
        for k in kids:
            try:
                names[k.name()] = names.get(k.name(), 0) + 1
            except Exception:  # noqa: BLE001
                pass
        out["driver_descendants_by_name"] = names
    finally:
        ray.shutdown()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs", required=True, help="obs_samples.npz from bench/micro_obs.py")
    ap.add_argument("--batch-sizes", default="8,29,64,128,256")
    ap.add_argument("--with-ray", action="store_true")
    ap.add_argument("--calls", type=int, default=200)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import numpy as np
    obs16 = np.load(os.path.abspath(args.obs))["obs"]
    bs = [int(x) for x in args.batch_sizes.split(",") if x.strip() and int(x) <= obs16.shape[0]]
    res: Dict[str, Any] = {"local": local_costs(obs16, bs)}
    if args.with_ray:
        try:
            res["ray"] = ray_round_trips(obs16, bs, args.calls)
        except Exception as e:  # noqa: BLE001
            import traceback
            res["ray"] = {"error": repr(e)[:400], "trace": traceback.format_exc()[-1500:]}
    if args.out:
        os.makedirs(os.path.abspath(args.out), exist_ok=True)
        with open(os.path.join(os.path.abspath(args.out), "serialize_bench.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
