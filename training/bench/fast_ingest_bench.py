#!/usr/bin/env python
"""
train.py --fast-ingest on recorded worker messages (box c). Every command that times something or
starts Ray runs under `flock ~/dev/fast/bench.lock`.

  record   A short train.py run WITHOUT the flag (4 Showdown servers, the live learner flags, a live
           checkpoint) whose learner also writes every packed message it is sent to --msgs, until
           --steps decisions are written. The recording lives only here: a subclass of LearnerActor
           defined in this script, put in train.py's place for this run.

               python bench/fast_ingest_bench.py record --msgs DIR --ckpt CKPT --ports 11448-11451

  check    The recorded messages through the learner's own _loop, flag off and then flag on, in one
           process, deterministic (CUBLAS workspace, torch deterministic algorithms), --updates
           updates each from the same checkpoint. For every update: every column handed to
           ppo_update torch.equal and bit-equal (obs, act, logp, val, adv, ret, next_hp, ep_ids,
           each extra), the stats floats equal, the digests of the weights, fp32 masters and Adam
           moments equal, the weight store's dicts equal and in update order. Then the checkpoint
           each learner writes after its last update: every entry equal.

               python bench/fast_ingest_bench.py check --det --msgs DIR --ckpt CKPT --json OUT

  digest   One path (--fast or not), any code tree (--code DIR: 4cfd496 for the pristine side), from
           any checkpoint, from message --start for --updates updates: the digests and stats as JSON,
           optionally the checkpoint after it (--save). For the cross-tree and cross-load checks.

  time     One path over the recorded messages: a warm-up update, then each n_mb regime (49: the KL
           stop at the first minibatch of epoch 2; 96: no stop) for --reps rounds of --updates
           updates, every round from the same checkpoint state. Per update: _perform_update's
           seconds and n_mb, its parts (the join or the buffer's view, the copy to the GPU in
           ppo_update, ppo_update, the push), and the learner thread's time outside it (taking the
           messages in: GAE, the arrival copy). The intercept of sec on n_mb is the fixed cost per
           update. --emu: the box-a CPU emulation (hogs on vCPUs 2-15, this process on 0-1 by the
           caller's taskset) plus --bw memory-bandwidth hogs.

               flock ~/dev/fast/bench.lock taskset -c 0,1 python bench/fast_ingest_bench.py time --path on --emu ...

           --store ray: the pushes go to a local Ray's real WeightStore actor (Ray's own pickling and
           object store, Ray's processes on the vCPUs the harness is not pinned to); --store fake: the
           stand-in below. --push-tensors: the flag on with the push as reviewed in b827896 (the
           tensors themselves to Ray, not out of band). Each JSON also has fixed_parts_s (the median
           of the measured fixed parts per update) and the time outside the update split by whether
           a push was in flight when the cycle began.

               flock ~/dev/fast/bench.lock taskset -c 0,1 python bench/fast_ingest_bench.py time --path on --emu --store ray ...

  compare  Flag off against flag on over any number of `time` JSONs per side (pooled): the measured
           fixed parts, the intercept with ONE slope for both paths (medians, Theil-Sen) and its 90%
           bootstrap interval, sec and the cycle per n_mb regime, the time outside the update, and the
           count of slow updates. The verdict on the fixed cost comes from here, not from the
           intercept each `time` run fits on its own (its slope differs by noise and the intercept
           swings with it).

               python bench/fast_ingest_bench.py compare --off A.json B.json --on C.json D.json --json OUT

  wire     The weight push through a local Ray (under the bench lock): the policy's state dict from
           --ckpt pushed as before (flag off), as the tensors from the push thread, and out of band
           (OutOfBandTensor); what the real WeightStore then holds against the learner's dict (keys,
           dtypes, shapes, strides, bits, not pinned), also for corner cases (bf16, bool, 0-dim, empty,
           non-contiguous); how many bytes Ray pickles in band; the time on the calling thread and on
           the push thread; and the Python lock a push takes from a pure-Python loop on another thread.

               flock ~/dev/fast/bench.lock python bench/fast_ingest_bench.py wire --json OUT

  hog bw SECONDS   a memory-bandwidth neighbour (streaming copies of 512 MB arrays)

The fake weight store keeps the dict a Ray actor would unpickle (wire(): protocol 5, out-of-band
buffers) and a digest of it; with serialize it also times the pickling on the calling thread.
"""
from __future__ import annotations

import sys
import time

if len(sys.argv) >= 3 and sys.argv[1] == "hog":
    # Before anything heavy is imported: a neighbour process only needs NumPy.
    import numpy as np
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else 600.0
    end = time.time() + dur
    a = np.ones(64 << 20, dtype=np.float64)          # 512 MB
    b = np.empty_like(a)
    while time.time() < end:
        np.copyto(b, a)
        np.copyto(a, b)
    sys.exit(0)

import argparse  # noqa: E402
import asyncio  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
import dataclasses  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import warnings  # noqa: E402
from typing import Tuple  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def _early(flag: str, default=None):
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return default


CODE = os.path.abspath(os.path.expanduser(_early("--code", REPO)))
sys.path.insert(0, CODE)
os.chdir(CODE)                  # vocab.json is opened by a relative path
DET = "--det" in sys.argv
# --fast-update (v2-fast): the learner also runs ppo_update's sync-free loop (train.py --fast-update). check: on the
# flag-on side only, so the comparison is no flag against both learner flags; digest and time: on the path given.
FAST_UPDATE = "--fast-update" in sys.argv
if DET:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

if DET:
    torch.use_deterministic_algorithms(True, warn_only=True)
    warnings.simplefilter("once")

import learner as learner_mod  # noqa: E402
import ppo_core  # noqa: E402
import train as T  # noqa: E402
from learner import LearnerActor  # noqa: E402

HAS_FAST = os.path.exists(os.path.join(CODE, "fast_ingest.py"))
if HAS_FAST:
    import fast_ingest  # noqa: E402

PROF = os.path.expanduser("~/dev/fast/prof")
SHOWDOWN = os.path.expanduser("~/pokemon-showdown")


def log(*a):
    print(*a, flush=True)


def live_argv(ckpt_dir: str, ports: str = "11448-11451", battles: int = 2304, infer_actors: int = 3,
              save_every: int = 50) -> list:
    """The live trainer's command line (run.json of the scratch run, human term off as now), with these ports and checkpoints."""
    return ["--mode", "ppo", "--ports", ports, "--battles", str(battles), "--rooms-per-pair", "16", "--teacher", "psbot",
            "--steps-per-update", "36864", "--minibatch", "768", "--seq-len", "256", "--grad-accum", "1",
            "--save-every", str(save_every), "--ckpt-dir", ckpt_dir, "--obs-matchup", "--obs-movesets",
            "--infer-actors", str(infer_actors), "--infer-sync-s", "1", "--set-head-coef", "0.1", "--opp-action-aux-coef", "0.1",
            "--read-bonus", "0.3", "--read-anneal", "20000000:45000000", "--opponent-seats", "setupnet:0.15",
            "--boosted-row-weight", "2", "--lr-hold-steps", "479713", "--lr-total-steps", "1113566"]


def ckpt_dir_with(ckpt: str, where: str) -> str:
    """A directory holding only this checkpoint (a hard link, a copy if that fails), for the learner to resume from."""
    os.makedirs(where, exist_ok=True)
    base = os.path.basename(ckpt)
    if not (base.startswith("learner_update_") and base.endswith(".pt")):
        base = f"learner_update_from_{base}"           # the learner resumes only from learner_update_*.pt
    dst = os.path.join(where, base)
    if not os.path.exists(dst):
        try:
            os.link(ckpt, dst)
        except OSError:
            import shutil
            shutil.copy2(ckpt, dst)
    others = [p for p in glob.glob(os.path.join(where, "learner_update_*.pt")) if os.path.abspath(p) != os.path.abspath(dst)]
    if others:
        raise SystemExit(f"{where} holds other checkpoints too ({others[:3]}); give an empty directory")
    return where


# --------------------------------------------------------------------------------------------------
# the messages
# --------------------------------------------------------------------------------------------------

def load_messages(d: str, limit_steps: int = 0) -> list:
    paths = sorted(glob.glob(os.path.join(d, "msgs", "msg_*.pkl")))
    out, steps = [], 0
    for p in paths:
        with open(p, "rb") as f:
            m = pickle.load(f)
        for a in m[1:9]:
            if isinstance(a, np.ndarray):
                a.flags.writeable = False                 # as Ray hands them over
        out.append(m)
        steps += len(m[2])
        if limit_steps and steps >= limit_steps:
            break
    if not out:
        raise SystemExit(f"no messages in {d}/msgs")
    return out


def rechunk(msgs: list, rows: int) -> list:
    """
    The recorded episodes, in their order, packed again into messages of whole episodes of about
    `rows` rows (box a's workers send about 184): consecutive messages are joined and cut, never
    across a change in which extra columns the messages carry.
    """
    out = []
    pend = []                                    # (message, first episode, last episode + 1, first row, last row)

    def flush():
        if not pend:
            return
        parts = [tuple(m[k][a:z] for k in range(1, 8)) for (m, e0, e1, a, z) in pend]
        cols = tuple(np.concatenate([p[k] for p in parts]) for k in range(7))
        L = np.concatenate([np.asarray(m[8][e0:e1]) for (m, e0, e1, a, z) in pend]).astype(np.int32)
        msg = ("packed",) + cols + (L,)
        if len(pend[0][0]) > 9:
            keys = list(pend[0][0][9].keys())
            msg = msg + ({k: np.concatenate([m[9][k][a:z] for (m, e0, e1, a, z) in pend]) for k in keys},)
        for x in msg[1:9]:
            x.flags.writeable = False
        out.append(msg)
        pend.clear()

    have = 0
    sig = None
    for m in msgs:
        s_ = tuple(sorted(m[9].keys())) if len(m) > 9 and m[9] is not None else None
        if s_ != sig:
            flush()
            have = 0
            sig = s_
        L = np.asarray(m[8]).astype(np.int64)
        e0, a = 0, 0
        for e in range(len(L)):
            have += int(L[e])
            if have >= rows:
                z = int(L[:e + 1].sum())
                pend.append((m, e0, e + 1, a, z))
                flush()
                have = 0
                e0, a = e + 1, z
        if e0 < len(L):
            pend.append((m, e0, len(L), a, int(L.sum())))
    flush()
    return out


# --------------------------------------------------------------------------------------------------
# the weight store stand-in, the digests
# --------------------------------------------------------------------------------------------------

def tensor_digest(named) -> str:
    h = hashlib.sha256()
    for k, t in named:
        h.update(str(k).encode())
        t = t.detach().contiguous().cpu()
        h.update(f"{t.dtype}{tuple(t.shape)}".encode())
        h.update(t.numpy().tobytes() if t.dtype == torch.bool else t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()[:16]


class _Remote:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *a, **k):
        return self._fn(*a, **k)


def wire(obj):
    """
    obj as a Ray actor receives it: pickled as Ray pickles task arguments (protocol 5, buffers that
    support it out of band) and unpickled from those bytes. Returns (the received object, in-band
    bytes, out-of-band bytes).
    """
    bufs = []
    blob = pickle.dumps(obj, protocol=5, buffer_callback=bufs.append)
    raws = [bytearray(b.raw()) for b in bufs]                  # the object store's copies
    got = pickle.loads(blob, buffers=raws)
    return got, len(blob), sum(len(r) for r in raws)


class Store:
    """
    The WeightStore as the learner uses it. The dict it keeps is the one a Ray actor would unpickle
    (wire()), so the out-of-band weights are compared as the real store gets them.
    serialize: also time Ray's pickling on the calling thread (pickle, then one more copy of the
    in-band bytes, as into the object store; out-of-band bytes are copied by Ray without the Python
    lock, so not here). For the real costs, `time --store ray` uses Ray's own WeightStore.
    """

    def __init__(self, serialize: bool = True, keep: bool = True):
        self.serialize = serialize
        self.keep = keep
        self.calls = []
        self.ser_s = 0.0
        self.update = _Remote(self._update)
        self.set_temp = _Remote(self._set_temp)

    def _update(self, weights, version):
        t0 = time.perf_counter()
        if self.serialize:
            bufs = []
            blob = bytearray(pickle.dumps((weights, version), protocol=5, buffer_callback=bufs.append))
            del blob, bufs
        self.ser_s += time.perf_counter() - t0
        if self.keep:
            weights, version = wire((weights, version))[0]
            meta = [(k, str(v.dtype), tuple(v.shape), tuple(v.stride()), v.device.type, bool(v.is_pinned())) for k, v in weights.items()]
            self.calls.append(("update", int(version), tensor_digest(weights.items()), meta, time.perf_counter()))
        else:
            self.calls.append(("update", int(version), "", None, time.perf_counter()))

    def _set_temp(self, temp):
        self.calls.append(("temp", float(temp), time.perf_counter()))


def update_digest(L) -> dict:
    sd = L.opt.state_dict()
    adam = []
    for pid, s in sorted(sd["inner"]["state"].items()):
        for k in ("exp_avg", "exp_avg_sq"):
            adam.append((f"{pid}.{k}", s[k]))
    return {"weights": tensor_digest(sorted(L.net.state_dict().items())),
            "masters": tensor_digest([(str(i), t) for i, t in enumerate(sd["masters"])]),
            "adam": tensor_digest(adam),
            "steps": sorted({float(s["step"]) for s in sd["inner"]["state"].values()})}


def stats_dict(st) -> dict:
    d = {k: getattr(st, k) for k in ("approx_kl", "clip_frac", "entropy", "v_loss", "pg_loss", "total_loss", "n_mb", "acc")}
    d["aux"] = dict(getattr(st, "aux", None) or {})
    return {k: (float(v).hex() if isinstance(v, float) else v) for k, v in d.items()} | {
        "aux": {k: float(v).hex() for k, v in d["aux"].items()}}


# --------------------------------------------------------------------------------------------------
# instrumentation (wrappers installed from here; the learner's code is not touched)
# --------------------------------------------------------------------------------------------------

class Probe:
    def __init__(self):
        self.perform_s = 0.0
        self.cur = None          # the update being timed
        self.updates = []        # one dict per update
        self.gae_s = 0.0
        self.capture = None      # check: fn(dataset) called with the update's data set before ppo_update
        self.after = None        # check: fn(learner) called after each update
        self.sync = torch.cuda.is_available()

    def install(self):
        P = self
        orig_perform = LearnerActor._perform_update

        async def perform(self_):
            if P.sync:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            P.cur = {"t0": t0, "join_s": 0.0, "copy_s": 0.0, "upd_s": 0.0, "upd_t0": None, "upd_t1": None}
            await orig_perform(self_)
            if P.sync:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            P.cur["sec"] = t1 - t0
            if P.cur["upd_t1"] is not None:
                P.cur["prep_s"] = P.cur["upd_t0"] - t0
                P.cur["tail_s"] = t1 - P.cur["upd_t1"]
            P.perform_s += t1 - t0
            P.updates.append(P.cur)
            P.cur = None
            if P.after is not None:
                P.after(self_)
        LearnerActor._perform_update = perform

        orig_update = learner_mod.ppo_update

        def ppo_update(*a, **k):
            ds = k.get("dataset")
            if P.capture is not None:
                P.capture(ds)
            if P.sync:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            st = orig_update(*a, **k)
            if P.sync:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            if P.cur is not None:
                P.cur.update(upd_t0=t0, upd_t1=t1, upd_s=t1 - t0, n_mb=int(st.n_mb), stats=stats_dict(st))
            return st
        learner_mod.ppo_update = ppo_update

        orig_swap = ppo_core.AsyncEpisodeDataset.swap_out_columns

        def swap(self_):
            t0 = time.perf_counter()
            out = orig_swap(self_)
            if P.cur is not None:
                P.cur["join_s"] += time.perf_counter() - t0
            return out
        ppo_core.AsyncEpisodeDataset.swap_out_columns = swap

        if hasattr(LearnerActor, "_take_update_data"):
            orig_take = LearnerActor._take_update_data

            def take(self_):
                buf = getattr(self_, "_obs_buf", None)
                w0 = buf.wait_s if buf is not None else 0.0
                t0 = time.perf_counter()
                out = orig_take(self_)
                if P.cur is not None:
                    P.cur["take_s"] = time.perf_counter() - t0
                    P.cur["obs_on_gpu"] = bool(out[2])
                    P.cur["copy_wait_s"] = (buf.wait_s - w0) if buf is not None else 0.0
                return out
            LearnerActor._take_update_data = take

        if HAS_FAST and hasattr(fast_ingest, "PendingAdvantages"):
            orig_flush = fast_ingest.PendingAdvantages.flush

            def flush(self_):
                t0 = time.perf_counter()
                out = orig_flush(self_)
                if P.cur is not None:
                    P.cur["gae_flush_s"] = P.cur.get("gae_flush_s", 0.0) + time.perf_counter() - t0
                return out
            fast_ingest.PendingAdvantages.flush = flush

        orig_on_device = ppo_core.AsyncEpisodeDataset.on_device

        def on_device(self_, device):
            hit = self_._device_cache is not None and self_._device_cache[0] == torch.device(device)
            if hit or not P.sync:
                return orig_on_device(self_, device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = orig_on_device(self_, device)
            torch.cuda.synchronize()
            if P.cur is not None:
                P.cur["copy_s"] += time.perf_counter() - t0
            return out
        ppo_core.AsyncEpisodeDataset.on_device = on_device

        orig_gae = learner_mod.gae_from_episode

        def gae(*a, **k):
            t0 = time.perf_counter()
            out = orig_gae(*a, **k)
            P.gae_s += time.perf_counter() - t0
            return out
        learner_mod.gae_from_episode = gae
        if HAS_FAST:
            orig_batch = fast_ingest.gae_batch

            def gae_b(*a, **k):
                t0 = time.perf_counter()
                out = orig_batch(*a, **k)
                P.gae_s += time.perf_counter() - t0
                return out
            fast_ingest.gae_batch = gae_b


async def build(ckpt_dir: str, fast: bool, store, extra=(), fast_update: bool = False):
    args = T.parse_args(live_argv(ckpt_dir) + list(extra))
    cfg = T.build_config(args)
    kw = {"fast_ingest": True} if fast else {}
    if fast_update:
        kw["fast_update"] = True            # only a tree that has it (v2-fast-syncfree, v2-fast) takes the keyword
    L = LearnerActor(cfg, store, **kw)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        await L._startup_task
    text = buf.getvalue()
    log("  " + " | ".join(x for x in text.splitlines() if "resumed" in x or "REFUSED" in x or "failed" in x)[:300])
    if "resumed from" not in text:
        raise SystemExit("the learner did not resume:\n" + text[-2000:])
    torch.cuda.manual_seed_all(1234)
    return L


async def feed(L, msgs, start: int, n_updates: int, probe: Probe, group: int = 1, quiet: bool = True):
    """msgs[start:] through L's own _loop (`group` queued at a time) until n_updates more updates have run. Returns (next index, per-update ingest records)."""
    target = L.update_idx + n_updates
    i = start
    cyc = {"ingest_s": 0.0, "msgs": 0, "rows": 0, "gae_s": 0.0, "arrival_copy_s": 0.0, "copy_thread_s": 0.0}

    def bufs():
        b = getattr(L, "_obs_buf", None)
        return (b.copy_s, getattr(b, "thread_s", 0.0)) if b is not None else (0.0, 0.0)
    gae0 = probe.gae_s
    copy0, thr0 = bufs()
    cycles = []
    while L.update_idx < target:
        if i >= len(msgs):
            raise SystemExit(f"ran out of messages at {i} with {target - L.update_idx} updates to go")
        chunk = msgs[i:i + group]
        i += len(chunk)
        want = L.total_steps + sum(len(m[2]) for m in chunk)
        u0, p0 = L.update_idx, probe.perform_s
        if cyc["msgs"] == 0:
            pu = getattr(L, "_pusher", None)
            cyc["push_in_flight"] = (not pu._idle.is_set()) if pu is not None else None
        t0 = time.perf_counter()
        for m in chunk:
            await L._q.put(m)
        while L.total_steps < want:
            await asyncio.sleep(0)
        dt = time.perf_counter() - t0 - (probe.perform_s - p0)
        cyc["ingest_s"] += dt
        cyc["msgs"] += len(chunk)
        cyc["rows"] += sum(len(m[2]) for m in chunk)
        if L.update_idx > u0:
            copy1, thr1 = bufs()
            cyc["gae_s"] = probe.gae_s - gae0           # every GAE call of the cycle, inside the update or not
            cyc["arrival_copy_s"] = copy1 - copy0       # the learner thread's time handing over (or making) the copies
            cyc["copy_thread_s"] = thr1 - thr0          # the copy thread's time
            cycles.append(cyc)
            gae0, copy0, thr0 = probe.gae_s, copy1, thr1
            cyc = {"ingest_s": 0.0, "msgs": 0, "rows": 0, "gae_s": 0.0, "arrival_copy_s": 0.0, "copy_thread_s": 0.0}
    return i, cycles


def stop_loop(L):
    L._task.cancel()
    if getattr(L, "_pusher", None) is not None:
        L._pusher.flush(60)


def snapshot(L):
    return {"net": {k: v.detach().clone() for k, v in L.net.state_dict().items()},
            "target": {k: v.detach().clone() for k, v in L.target_net.state_dict().items()},
            "opt": copy.deepcopy(L.opt.state_dict()), "sched": copy.deepcopy(L.sched.state_dict()),
            "upd": L.update_idx, "steps": L.total_steps, "eps": L.total_episodes, "epc": L.ep_counter,
            "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "np": np.random.get_state()}


def restore(L, s):
    with torch.no_grad():
        L.net.load_state_dict(s["net"])
        L.target_net.load_state_dict(s["target"])
    L.opt.load_state_dict(copy.deepcopy(s["opt"]))
    L.sched.load_state_dict(copy.deepcopy(s["sched"]))
    L.update_idx, L.total_steps, L.total_episodes, L.ep_counter = s["upd"], s["steps"], s["eps"], s["epc"]
    torch.set_rng_state(s["rng"])
    torch.cuda.set_rng_state_all(s["cuda_rng"])
    np.random.set_state(s["np"])
    L.dataset.clear()
    if getattr(L, "_obs_buf", None) is not None:
        L._obs_buf.reset()
    if getattr(L, "_gae_pending", None) is not None:
        L._gae_pending.clear()
    if getattr(L, "_pusher", None) is not None:
        L._pusher.flush(60)


def set_regime(L, mb: int):
    """96: no early stop (every minibatch of both epochs); 49: the KL stop after the first minibatch of epoch 2."""
    L.cfg = dataclasses.replace(L.cfg, target_kl=1e9 if mb == 96 else 1e-12)


# --------------------------------------------------------------------------------------------------
# record
# --------------------------------------------------------------------------------------------------

REC = {"dir": None, "steps": 0, "stop": None}


class RecordingLearner(LearnerActor):
    """The learner, which also writes every packed message it is sent (bench only; pickled by value into the actor)."""

    async def submit_packed_batch(self, *a, **k):
        st = getattr(self, "_rec", None)
        if st is None:
            st = self._rec = {"n": 0, "steps": 0, "dir": os.environ["FI_REC_DIR"], "want": int(os.environ["FI_REC_STEPS"]),
                              "stop": os.environ["FI_REC_STOP"]}
            os.makedirs(os.path.join(st["dir"], "msgs"), exist_ok=True)
        if st["steps"] < st["want"]:
            names = ("obs_cat", "act_cat", "logp_cat", "val_cat", "var_cat", "rew_cat", "done_cat", "lengths", "extras")
            vals = dict(zip(names, a))
            vals.update(k)
            msg = ("packed",) + tuple(vals[n] for n in names[:8])
            if vals.get("extras") is not None:
                msg = msg + (vals["extras"],)
            path = os.path.join(st["dir"], "msgs", f"msg_{st['n']:06d}.pkl")
            with open(path + ".tmp", "wb") as f:
                pickle.dump(msg, f, protocol=5)
            os.replace(path + ".tmp", path)
            st["n"] += 1
            st["steps"] += len(vals["act_cat"])
            if st["steps"] >= st["want"]:
                with open(st["stop"], "w") as f:
                    f.write(f"{st['n']} messages, {st['steps']} steps\n")
        return await super().submit_packed_batch(*a, **k)


def _listening(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def cmd_record(a):
    d = os.path.abspath(os.path.expanduser(a.msgs))
    os.makedirs(os.path.join(d, "servers"), exist_ok=True)
    if glob.glob(os.path.join(d, "msgs", "msg_*.pkl")):
        raise SystemExit(f"{d}/msgs already has messages; give a new directory")
    ck = ckpt_dir_with(os.path.expanduser(a.ckpt), os.path.join(d, "ckpt"))
    lo, hi = (int(x) for x in a.ports.split("-"))
    ports = list(range(lo, hi + 1))
    if any(_listening(p) for p in ports):
        raise SystemExit(f"a port of {a.ports} is taken")
    srv = subprocess.Popen([sys.executable, os.path.join(CODE, "start_servers.py"), "--dir", SHOWDOWN, "--ports", a.ports,
                            "--heap-mb", "2048", "--log-dir", os.path.join(d, "servers")],
                           stdout=open(os.path.join(d, "servers.log"), "w"), stderr=subprocess.STDOUT, cwd=CODE)
    log(f"servers pid {srv.pid}")
    try:
        for _ in range(240):
            if all(_listening(p) for p in ports):
                break
            time.sleep(1)
        else:
            raise SystemExit("the servers did not come up")
        stop = os.path.join(d, "rec.stop")
        os.environ.update(FI_REC_DIR=d, FI_REC_STEPS=str(a.steps), FI_REC_STOP=stop, RAY_ADDRESS="local",
                          PYTHONPATH=CODE + os.pathsep + os.environ.get("PYTHONPATH", ""))
        argv = live_argv(ck, a.ports, battles=a.battles, infer_actors=1, save_every=1000) + [
            "--minutes", str(a.minutes), "--stop-file", stop]
        log("train.py " + " ".join(argv))
        T.LearnerActor = RecordingLearner
        asyncio.run(T.main(T.parse_args(argv)))
    finally:
        srv.send_signal(2)
        try:
            srv.wait(60)
        except subprocess.TimeoutExpired:
            srv.terminate()
            srv.wait(30)
        log(f"servers stopped (exit {srv.returncode})")
    n = len(glob.glob(os.path.join(d, "msgs", "msg_*.pkl")))
    log(f"recorded {n} messages in {d}/msgs")


# --------------------------------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------------------------------

def same_bits(x: torch.Tensor, y: torch.Tensor) -> bool:
    if x.dtype != y.dtype or tuple(x.shape) != tuple(y.shape):
        return False
    x, y = x.detach().contiguous().cpu(), y.detach().contiguous().cpu()
    if x.dtype == torch.bool:
        return torch.equal(x, y)
    return x.reshape(-1).view(torch.uint8).numpy().tobytes() == y.reshape(-1).view(torch.uint8).numpy().tobytes()


def deep_equal(x, y, path="", ignore=()) -> list:
    """Where two checkpoint payloads differ ([] when nowhere). ignore: paths not compared (e.g. /run_cfg/learner/ckpt_dir)."""
    if path in ignore:
        return []
    if torch.is_tensor(x) or torch.is_tensor(y):
        if not (torch.is_tensor(x) and torch.is_tensor(y) and same_bits(x, y)):
            return [path]
        return []
    if isinstance(x, dict) and isinstance(y, dict):
        if list(x.keys()) != list(y.keys()):
            return [path + " (keys)"]
        out = []
        for k in x:
            out += deep_equal(x[k], y[k], f"{path}/{k}", ignore)
        return out
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        if type(x) is not type(y) or len(x) != len(y):
            return [path + " (length)"]
        out = []
        for i, (p, q) in enumerate(zip(x, y)):
            out += deep_equal(p, q, f"{path}[{i}]", ignore)
        return out
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        return [] if (isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and x.dtype == y.dtype and np.array_equal(x, y)) else [path]
    return [] if x == y else [path]


async def _check(a):
    d = os.path.abspath(os.path.expanduser(a.msgs))
    out = os.path.abspath(os.path.expanduser(a.out))
    os.makedirs(out, exist_ok=True)
    msgs = load_messages(d)
    if a.rechunk:
        msgs = rechunk(msgs, a.rechunk)
    log(f"{len(msgs)} messages, {sum(len(m[2]) for m in msgs)} steps")
    probe = Probe()
    probe.install()
    report = {"updates": a.updates, "paths": {}, "on_side": ["fast_ingest"] + (["fast_update"] if FAST_UPDATE else [])}
    want = []                    # per update: the flag-off columns
    for fast in (False, True):
        name = "on" if fast else "off"
        store = Store(serialize=False)
        # one directory for both learners: it is in run_cfg, which the checkpoints compared below hold
        ck = ckpt_dir_with(os.path.expanduser(a.ckpt), os.path.join(out, "ckpt"))
        L = await build(ck, fast, store, fast_update=FAST_UPDATE and fast)
        res = {"updates": []}
        k = {"i": 0}

        def capture(ds, fast=fast, res=res, k=k):
            data = ds.tensorize()
            names = tuple(ds.extra_names)
            if not fast:
                want.append((tuple(t.detach().cpu() if t.device.type != "cpu" else t for t in data), names))
                res["updates"].append({"obs_device": data[0].device.type, "type": type(ds).__name__, "rows": int(data[1].shape[0])})
            else:
                ref, ref_names = want[k["i"]]
                cols = []
                ok = names == ref_names and len(data) == len(ref)
                labels = ["obs", "act", "logp", "val", "adv", "ret", "next_hp", "ep_ids"] + list(names)
                for lab, x, y in zip(labels, data, ref):
                    xc = x.detach().cpu()
                    eq = bool(x.dtype == y.dtype and tuple(x.shape) == tuple(y.shape) and torch.equal(xc, y))
                    bits = same_bits(xc, y)
                    cols.append({"col": lab, "torch_equal": eq, "same_bits": bits, "dtype": str(x.dtype),
                                 "shape": list(x.shape), "device": x.device.type})
                    ok = ok and eq and bits
                res["updates"].append({"obs_device": data[0].device.type, "type": type(ds).__name__, "rows": int(data[1].shape[0]),
                                       "columns_equal": ok, "columns": cols})
            k["i"] += 1
        probe.capture = capture
        digests = []
        probe.after = lambda L_, digests=digests: digests.append(update_digest(L_))
        n0 = len(probe.updates)
        nxt, _ = await feed(L, msgs, 0, a.updates, probe, group=a.group)
        stop_loop(L)
        await asyncio.sleep(0.1)
        probe.after = None
        for j, u in enumerate(probe.updates[n0:]):
            res["updates"][j]["stats"] = u["stats"]
            res["updates"][j]["n_mb"] = u["n_mb"]
            res["updates"][j]["digest"] = digests[j]
        res["final"] = update_digest(L)
        res["pushes"] = [c[:4] for c in store.calls if c[0] == "update"]
        res["temps"] = [c[1] for c in store.calls if c[0] == "temp"]
        res["next_msg"] = nxt
        res["counters"] = [L.update_idx, L.total_steps, L.total_episodes, L.ep_counter, len(L.dataset)]
        if getattr(L, "_gae_pending", None) is not None:
            L._gae_pending.flush()                  # the leftover rows' advantages, as the next update fills them
        res["leftover_digest"] = tensor_digest(enumerate(L.dataset.tensorize())) if len(L.dataset) else ""
        if fast:
            res["buffer"] = {"bytes": L._obs_buf.nbytes() if L._obs_buf else 0, "grows": L._obs_buf.grows if L._obs_buf else 0,
                             "handed": L._obs_buf.handed if L._obs_buf else 0, "inline": L._obs_buf.inline if L._obs_buf else 0,
                             "fallbacks": L._fast_fallbacks, "pushed": L._pusher.pushed, "push_errors": L._pusher.errors,
                             "sync_pushes": L._pusher.sync_pushes, "gae_flushes": L._gae_pending.flushes,
                             "gae_fallbacks": L._gae_pending.fallbacks}
        path = os.path.join(out, f"after_{name}.pt")
        L._save_checkpoint(path)
        res["ckpt"] = path
        res["max_alloc_gb"] = torch.cuda.max_memory_allocated() / 2**30
        report["paths"][name] = res
        log(f"{name}: updates {[u['n_mb'] for u in res['updates']]} final {res['final']}")
        del L
        torch.cuda.empty_cache()
    off, on = report["paths"]["off"], report["paths"]["on"]
    checks = {
        "columns_equal_every_update": all(u.get("columns_equal") for u in on["updates"]) and len(on["updates"]) == a.updates,
        "obs_from_gpu_buffer_every_update": all(u["obs_device"] == "cuda" and u["type"] == "DeviceEpisodeDataset" for u in on["updates"]),
        "stats_equal": [u["stats"] for u in off["updates"]] == [u["stats"] for u in on["updates"]],
        "n_mb_equal": [u["n_mb"] for u in off["updates"]] == [u["n_mb"] for u in on["updates"]],
        "update_digests_equal": [u["digest"] for u in off["updates"]] == [u["digest"] for u in on["updates"]],
        "final_digests_equal": off["final"] == on["final"],
        "pushes_equal_in_order": off["pushes"] == on["pushes"],
        "push_versions": [p[1] for p in on["pushes"]],
        "temps_equal": off["temps"] == on["temps"],
        "counters_equal": off["counters"] == on["counters"],
        "leftover_equal": off["leftover_digest"] == on["leftover_digest"],
    }
    a_ck = torch.load(off["ckpt"], map_location="cpu", weights_only=False)
    b_ck = torch.load(on["ckpt"], map_location="cpu", weights_only=False)
    diffs = deep_equal(a_ck, b_ck)
    checks["checkpoints_equal"] = not diffs
    checks["checkpoint_diffs"] = diffs[:20]
    report["checks"] = checks
    report["ok"] = all(v for k, v in checks.items() if k not in ("push_versions", "checkpoint_diffs"))
    log(json.dumps(checks, indent=1, default=str))
    log("CHECK " + ("PASSED" if report["ok"] else "FAILED"))
    with open(a.json, "w") as f:
        json.dump(report, f, indent=1, default=str)


def cmd_check(a):
    asyncio.run(_check(a))
    os._exit(0)


# --------------------------------------------------------------------------------------------------
# digest (one path, any tree, any checkpoint)
# --------------------------------------------------------------------------------------------------

async def _digest(a):
    d = os.path.abspath(os.path.expanduser(a.msgs))
    msgs = load_messages(d)
    if a.rechunk:
        msgs = rechunk(msgs, a.rechunk)
    probe = Probe()
    probe.install()
    store = Store(serialize=False)
    work = os.path.abspath(os.path.expanduser(a.work))
    L = await build(ckpt_dir_with(os.path.expanduser(a.resume), work), a.fast, store, fast_update=FAST_UPDATE)
    res = {"code": CODE, "fast": a.fast, "fast_update": FAST_UPDATE, "resume": a.resume, "start": a.start, "updates": []}
    for _ in range(a.updates):
        n0 = len(probe.updates)
        a.start, _c = await feed(L, msgs, a.start, 1, probe, group=a.group)
        u = probe.updates[n0]
        res["updates"].append({"upd": L.update_idx, "n_mb": u["n_mb"], "stats": u["stats"], "digest": update_digest(L)})
        log(f"upd {L.update_idx}: n_mb={u['n_mb']} {res['updates'][-1]['digest']}")
    stop_loop(L)
    res["pushes"] = [c[:3] for c in store.calls if c[0] == "update"]
    res["next_msg"] = a.start
    if a.save:
        L._save_checkpoint(os.path.abspath(os.path.expanduser(a.save)))
        res["saved"] = a.save
    with open(a.json, "w") as f:
        json.dump(res, f, indent=1, default=str)


def cmd_digest(a):
    asyncio.run(_digest(a))
    os._exit(0)


# --------------------------------------------------------------------------------------------------
# time
# --------------------------------------------------------------------------------------------------

def start_hogs(a) -> list:
    hogs = []
    hog = os.path.join(PROF, "hog.py")
    dur = str(int(a.hog_seconds))
    for _ in range(a.nmem):
        hogs.append(subprocess.Popen(["taskset", "-c", a.hog_cpus, "python3", hog, "mem", dur], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for _ in range(a.npy):
        hogs.append(subprocess.Popen(["taskset", "-c", a.hog_cpus, "python3", hog, "py", dur], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for _ in range(a.bw):
        hogs.append(subprocess.Popen(["taskset", "-c", a.hog_cpus, sys.executable, os.path.abspath(__file__), "hog", "bw", dur],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    log(f"hogs: {a.nmem} mem + {a.npy} py + {a.bw} bandwidth on vCPUs {a.hog_cpus}, pids {[h.pid for h in hogs]}")
    time.sleep(3)
    return hogs


def stop_hogs(hogs: list) -> None:
    for h in hogs:
        if h.poll() is None:
            h.terminate()
    for h in hogs:
        try:
            h.wait(20)
        except subprocess.TimeoutExpired:
            h.kill()
            h.wait(10)
    log(f"hogs stopped: {sum(1 for h in hogs if h.poll() is None)} left")


def fit(xs, ys):
    """Least squares sec = c + m * n_mb for ONE path. Its c is noisy (see compare_paths): reported, not used for the verdict."""
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    if len(set(xs)) < 2:
        return float("nan"), float("nan")
    A = np.vstack([np.ones_like(x), x]).T
    (c, m), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(c), float(m)


def fixed_parts(r: dict) -> float:
    """One update's measured fixed cost: everything in _perform_update but ppo_update (prep: the join or the buffer's view and the advantages; the copy to the GPU; tail: the push)."""
    return float(r["prep_s"] or 0.0) + float(r["copy_s"] or 0.0) + float(r["tail_s"] or 0.0)


def shared_slope(off_rows: list, on_rows: list) -> Tuple[float, float, float]:
    """
    (c_off, c_on, m): sec = c_path + m * n_mb with one slope m for both paths, fitted by medians
    (Theil-Sen): m is the median of (sec_i - sec_j) / (n_i - n_j) over pairs of updates of the same
    path with different n_mb, c_path the median of sec - m * n_mb over the path's updates. The flag
    does not touch the per-minibatch work (the minibatches index the same GPU rows either way), so
    one slope is the right model; a slope per path only adds its noise to the intercepts, and a
    median ignores the odd slow update that a least-squares line follows.
    """
    slopes = []
    for rows in (off_rows, on_rows):
        x = np.asarray([r["n_mb"] for r in rows], float)
        y = np.asarray([r["sec"] for r in rows], float)
        i, j = np.triu_indices(len(x), 1)
        k = x[i] != x[j]
        slopes.append((y[i][k] - y[j][k]) / (x[i][k] - x[j][k]))
    s = np.concatenate(slopes) if slopes else np.asarray([])
    if s.size == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(np.median(s))
    c = [float(np.median([r["sec"] - m * r["n_mb"] for r in rows])) for rows in (off_rows, on_rows)]
    return c[0], c[1], m


def compare_paths(off_rows: list, on_rows: list, boot: int = 2000, seed: int = 0) -> dict:
    """
    Flag off against flag on from `time` rows (one dict per update, any number of runs pooled).

      fixed_parts_s   the measured fixed cost per update (fixed_parts), median per path: the direct
                      measure, little noise
      intercept_s     the fixed cost from sec on n_mb with one slope for both paths (shared_slope),
                      with a 90% bootstrap interval of the change (updates resampled within each path)
      per_regime      the median sec per n_mb regime and its change (the intercept's change twice
                      over, when the slope is shared)
      outside_s       the learner thread's time outside _perform_update per update (taking the
                      messages in), median per path and its change in percent
      cycle_s         sec + outside per update, median per regime
      slow_updates    updates whose fixed parts took over twice their path's median (as the 2 s joins
                      of the first report's round 1): counted, not left out of anything
    The least-squares intercept of each path on its own (the `time` JSON's intercept_s) is not used:
    its slope differs between paths by noise alone and the intercept swings with it.
    """
    med = lambda v: float(np.median(v)) if len(v) else float("nan")      # noqa: E731
    out = {"updates": {"off": len(off_rows), "on": len(on_rows)}}
    fp = {k: [fixed_parts(r) for r in rows] for k, rows in (("off", off_rows), ("on", on_rows))}
    out["fixed_parts_s"] = {"off": med(fp["off"]), "on": med(fp["on"]), "change": med(fp["on"]) - med(fp["off"])}
    c_off, c_on, m = shared_slope(off_rows, on_rows)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(boot):
        a = [off_rows[i] for i in rng.integers(0, len(off_rows), len(off_rows))]
        b = [on_rows[i] for i in rng.integers(0, len(on_rows), len(on_rows))]
        x0, x1, _m = shared_slope(a, b)
        if np.isfinite(x0) and np.isfinite(x1):
            bs.append(x1 - x0)
    lo, hi = (float(np.percentile(bs, 5)), float(np.percentile(bs, 95))) if bs else (float("nan"), float("nan"))
    out["intercept_s"] = {"off": c_off, "on": c_on, "per_mb_s": m, "change": c_on - c_off, "change_90": [lo, hi]}
    out["per_regime"] = {}
    for mb in sorted({r["n_mb"] for r in off_rows} & {r["n_mb"] for r in on_rows}):
        x = med([r["sec"] for r in off_rows if r["n_mb"] == mb])
        y = med([r["sec"] for r in on_rows if r["n_mb"] == mb])
        cx = med([r["sec"] + r["ingest_s"] for r in off_rows if r["n_mb"] == mb])
        cy = med([r["sec"] + r["ingest_s"] for r in on_rows if r["n_mb"] == mb])
        out["per_regime"][str(mb)] = {"sec_off": x, "sec_on": y, "change": y - x, "cycle_off": cx, "cycle_on": cy,
                                      "cycle_change": cy - cx}
    o0, o1 = med([r["ingest_s"] for r in off_rows]), med([r["ingest_s"] for r in on_rows])
    out["outside_s"] = {"off": o0, "on": o1, "change_pct": 100.0 * (o1 - o0) / o0 if o0 else float("nan")}
    out["slow_updates"] = {k: sum(1 for v in fp[k] if v > 2 * med(fp[k])) for k in fp}
    return out


def start_ray_store():
    """
    A local Ray of this process's own (address "local": never another session's) and the real
    WeightStore actor, so the pushes cost what they cost the learner on box a: Ray's own pickling
    and object store. Ray's processes go on the vCPUs this process is not pinned to. Returns
    (ray, the store's handle); ray.shutdown() stops what it started.
    """
    import ray
    from inference import WeightStore
    mine = os.sched_getaffinity(0)
    others = (set(range(os.cpu_count() or 1)) - mine) or mine
    os.sched_setaffinity(0, others)
    try:
        ray.init(address="local", include_dashboard=False, num_cpus=2, num_gpus=0, object_store_memory=2 << 30,
                 log_to_driver=False)
        store = WeightStore.remote()
        ray.get(store.get_version.remote())
    finally:
        os.sched_setaffinity(0, mine)
    log(f"ray: local session {ray.get_runtime_context().get_node_id()[:8]}, the WeightStore actor up, on vCPUs {sorted(others)}")
    return ray, store


async def _time(a):
    d = os.path.abspath(os.path.expanduser(a.msgs))
    fast = a.path == "on"
    msgs = load_messages(d)
    if a.rechunk:
        msgs = rechunk(msgs, a.rechunk)
    log(f"path {a.path}{' + fast_update' if FAST_UPDATE else ''}, code {CODE}: {len(msgs)} messages ({np.mean([len(m[2]) for m in msgs]):.0f} rows each), "
        f"affinity {sorted(os.sched_getaffinity(0))}, torch threads {torch.get_num_threads()}")
    probe = Probe()
    probe.install()
    ray_ = None
    if a.store == "ray":
        ray_, store = start_ray_store()
    else:
        store = Store(serialize=True, keep=False)
    ser = lambda: getattr(store, "ser_s", 0.0) if a.store == "fake" else 0.0      # noqa: E731
    work = os.path.abspath(os.path.expanduser(a.work))
    L = await build(ckpt_dir_with(os.path.expanduser(a.ckpt), work), fast, store, fast_update=FAST_UPDATE)
    if getattr(L, "_pusher", None) is not None and a.push_tensors:
        L._pusher.out_of_band = False               # the push as reviewed in b827896: the tensors themselves
    log(f"store {a.store}; push {'none (flag off)' if getattr(L, '_pusher', None) is None else ('out of band' if L._pusher.out_of_band else 'tensors')}")
    snap = snapshot(L)
    torch.cuda.reset_peak_memory_stats()
    hogs = start_hogs(a) if a.emu else []
    rows = []
    try:
        # warm-up: compile and caches, both regimes' shapes
        for mb in (49, 96):
            restore(L, snap)
            set_regime(L, mb)
            await feed(L, msgs, 0, 1, probe, group=a.group)
        for rep in range(a.reps):
            for mb in (49, 96):
                restore(L, snap)
                set_regime(L, mb)
                pushed0 = (L._pusher.snap_s, L._pusher.wait_s, L._pusher.thread_s) if getattr(L, "_pusher", None) else (0, 0, 0)
                n0 = len(probe.updates)
                ser0 = ser()
                _, cycles = await feed(L, msgs, 0, a.updates, probe, group=a.group)
                ups = probe.updates[n0:]
                for j, (u, c) in enumerate(zip(ups, cycles)):
                    row = {"rep": rep, "regime": mb, "k": j, "sec": u["sec"], "n_mb": u["n_mb"], "prep_s": u.get("prep_s"),
                           "join_s": u.get("take_s", u["join_s"]), "copy_s": u["copy_s"], "upd_s": u["upd_s"],
                           "tail_s": u.get("tail_s"), "ingest_s": c["ingest_s"], "gae_s": c["gae_s"],
                           "gae_flush_s": u.get("gae_flush_s", 0.0), "copy_wait_s": u.get("copy_wait_s", 0.0),
                           "arrival_copy_s": c["arrival_copy_s"], "copy_thread_s": c["copy_thread_s"], "msgs": c["msgs"],
                           "rows": c["rows"], "obs_on_gpu": u.get("obs_on_gpu", False), "push_in_flight": c.get("push_in_flight")}
                    rows.append(row)
                    log(f"rep {rep} mb{mb} #{j}: sec={u['sec']:.3f} n_mb={u['n_mb']} prep={row['prep_s']:.3f} (join/take {row['join_s']:.3f}: "
                        f"gae {row['gae_flush_s']:.3f}, copy wait {row['copy_wait_s']:.3f}) "
                        f"copy={row['copy_s']:.3f} upd={row['upd_s']:.3f} tail={row['tail_s']:.3f} | ingest={c['ingest_s']:.3f} "
                        f"(gae {c['gae_s'] - row['gae_flush_s']:.3f}, arrival copy {c['arrival_copy_s']:.3f}, copy thread "
                        f"{c['copy_thread_s']:.3f}) {c['msgs']} msgs {c['rows']} rows, push in flight at the start: {c.get('push_in_flight')}")
                if getattr(L, "_pusher", None):
                    L._pusher.flush(60)
                    p1 = (L._pusher.snap_s, L._pusher.wait_s, L._pusher.thread_s)
                    log(f"   push: snapshot {p1[0] - pushed0[0]:.3f}s, waited {p1[1] - pushed0[1]:.3f}s on the learner thread; "
                        f"push thread {p1[2] - pushed0[2]:.3f}s; store serialisation {ser() - ser0:.3f}s for {len(ups)} pushes")
    finally:
        stop_hogs(hogs)
        stop_loop(L)
        if ray_ is not None:
            ray_.shutdown()
            log("ray: shut down")
    c, m = fit([r["n_mb"] for r in rows], [r["sec"] for r in rows])
    med = lambda k: float(np.median([r[k] for r in rows if r[k] is not None]))  # noqa: E731
    busy = [r["ingest_s"] for r in rows if r.get("push_in_flight")]
    idle = [r["ingest_s"] for r in rows if r.get("push_in_flight") is False]
    out = {"path": a.path, "fast_update": FAST_UPDATE, "code": CODE, "ckpt": a.ckpt, "emu": a.emu, "bw": a.bw, "rechunk": a.rechunk,
           "store": a.store,
           "push": None if getattr(L, "_pusher", None) is None else ("out_of_band" if L._pusher.out_of_band else "tensors"),
           "rows": rows,
           "intercept_s": c, "per_mb_s": m,
           "fixed_parts_s": float(np.median([fixed_parts(r) for r in rows])),
           "outside_push_in_flight_s": float(np.median(busy)) if busy else None,
           "outside_push_idle_s": float(np.median(idle)) if idle else None,
           "median": {k: med(k) for k in ("sec", "prep_s", "join_s", "copy_s", "upd_s", "tail_s", "ingest_s", "gae_s", "gae_flush_s",
                                          "copy_wait_s", "arrival_copy_s", "copy_thread_s", "msgs")},
           "median_cycle_s": float(np.median([r["sec"] + r["ingest_s"] for r in rows])),
           "cycle_outside_update_s": med("ingest_s"),
           "max_alloc_gb": torch.cuda.max_memory_allocated() / 2**30, "max_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
           "buffer_gb": (L._obs_buf.nbytes() / 2**30) if getattr(L, "_obs_buf", None) else 0.0,
           "fallbacks": getattr(L, "_fast_fallbacks", 0)}
    log(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    with open(a.json, "w") as f:
        json.dump(out, f, indent=1)


def cmd_time(a):
    asyncio.run(_time(a))
    os._exit(0)


# --------------------------------------------------------------------------------------------------
# compare (flag off against flag on, robust)
# --------------------------------------------------------------------------------------------------

def cmd_compare(a):
    """compare_paths over the rows of every --off and every --on JSON of `time` (pooled), printed and written to --json."""
    rows = {}
    for k, paths in (("off", a.off), ("on", a.on)):
        rows[k] = []
        for p in paths:
            with open(os.path.expanduser(p)) as f:
                rows[k] += json.load(f)["rows"]
    out = compare_paths(rows["off"], rows["on"])
    out["files"] = {"off": a.off, "on": a.on}
    log(json.dumps(out, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


# --------------------------------------------------------------------------------------------------
# wire (the weight push through a real Ray)
# --------------------------------------------------------------------------------------------------

def lock_rate(seconds: float, work=None) -> Tuple[float, int, float]:
    """
    (iterations per second of a pure-Python loop on this thread, how many times work() ran on another
    thread meanwhile, the seconds measured). The loop wants the Python lock all the time, so the share
    of the lock work() takes comes out as the loop's slowdown against work=None.
    """
    stop = threading.Event()
    runs = [0]

    def bg():
        while not stop.is_set():
            work()
            runs[0] += 1
    th = threading.Thread(target=bg, name="wire-bg", daemon=True) if work is not None else None
    if th is not None:
        th.start()
    n = 0
    t0 = time.perf_counter()
    end = t0 + seconds
    while True:
        for _ in range(2000):
            n += 1
        if time.perf_counter() >= end:
            break
    dt = time.perf_counter() - t0
    stop.set()
    if th is not None:
        th.join(120)
    return n / dt, runs[0], dt


def same_as_today(got: dict, want: dict) -> list:
    """Where the dict the store holds differs from the one the learner built ([] when nowhere)."""
    if list(got.keys()) != list(want.keys()):
        return ["keys"]
    bad = []
    for k, w in want.items():
        g = got[k]
        if not torch.is_tensor(g):
            bad.append(f"{k}: a {type(g).__name__}")
            continue
        meta_g = (g.dtype, tuple(g.shape), tuple(g.stride()), g.device.type, g.is_pinned(), g.requires_grad)
        meta_w = (w.dtype, tuple(w.shape), tuple(w.stride()), "cpu", False, False)
        if meta_g != meta_w or not same_bits(g, w):
            bad.append(k)
    return bad


def corner_cases() -> dict:
    """Tensors a state dict can hold, for the round trip: bf16, fp16, bool, ints, 0-dim, empty, non-contiguous, NaN and -0.0."""
    g = torch.Generator().manual_seed(7)
    return {"bf16": torch.randn(33, 17, generator=g).to(torch.bfloat16), "fp32": torch.randn(5, 3, 2, generator=g),
            "fp16": torch.randn(9, generator=g).half(), "bool": torch.rand(4, 6, generator=g) > 0.5,
            "int64": torch.randint(-5, 5, (7,), generator=g), "int8": torch.randint(-5, 5, (3, 3), generator=g).to(torch.int8),
            "scalar": torch.tensor(3.25), "scalar_long": torch.tensor(11), "empty": torch.empty(0, 4),
            "noncontig": torch.randn(5, 7, generator=g).t(), "nan": torch.tensor([float("nan"), -0.0, float("inf")])}


def cmd_wire(a):
    import fast_ingest as F
    ck = torch.load(os.path.expanduser(a.ckpt), map_location="cpu", weights_only=False)
    gpu_sd = {k: v.clone().to("cuda") for k, v in ck["model"].items()}
    mb = sum(v.numel() * v.element_size() for v in gpu_sd.values()) / 2**20
    log(f"the policy: {len(gpu_sd)} tensors, {mb:.1f} MB, dtypes {sorted({str(v.dtype) for v in gpu_sd.values()})}")

    def today(sd):
        return {k: v.cpu().detach() for k, v in sd.items()}
    out = {"tensors": len(gpu_sd), "mb": mb}
    # what Ray's pickling (protocol 5) keeps in band, where the Python lock is held, and hands over out of band
    for name, obj in (("tensors", today(gpu_sd)), ("out_of_band", F.for_the_wire(today(gpu_sd)))):
        _got, inband, oob = wire(obj)
        out[f"pickle_{name}"] = {"inband_bytes": inband, "out_of_band_bytes": oob}
        log(f"pickle protocol 5, {name}: {inband} bytes in band, {oob} out of band")
    ray, store = start_ray_store()
    hogs = start_hogs(a) if a.emu else []
    try:
        ver = [0]

        def settle():
            return ray.get(store.get_version.remote())

        # 0. how long the store (a nice-15 Ray worker) takes to run an update, the first time and after:
        # the out-of-band one imports fast_ingest there the first time
        lat = {}
        for name in ("tensors", "out_of_band"):
            got = []
            for _ in range(a.store_pushes):
                ver[0] += 1
                w = today(gpu_sd)
                t0 = time.perf_counter()
                ray.get(store.update.remote(F.for_the_wire(w) if name == "out_of_band" else w, ver[0]))
                got.append(time.perf_counter() - t0)
            lat[name] = {"first_s": got[0], "after_median_s": float(np.median(got[1:])) if len(got) > 1 else None, "all_s": got}
            log(f"store update round trip, {name}: first {got[0]:.3f} s, then median {lat[name]['after_median_s']:.3f} s "
                f"({' '.join(f'{x:.3f}' for x in got)})")
        out["store_round_trip"] = lat

        # 1. what the store holds after each way of pushing
        checks = {}
        corner_gpu = {k: v.to("cuda") for k, v in corner_cases().items()}
        for name in ("flag_off", "tensors", "out_of_band"):
            for label, sd in (("policy", gpu_sd), ("corner_cases", corner_gpu)):
                ver[0] += 1
                want = today(sd)
                if name == "flag_off":
                    store.update.remote(today(sd), ver[0])
                    store.set_temp.remote(0.5)
                else:
                    p = F.WeightPusher(store, out_of_band=(name == "out_of_band"))
                    p.push(sd, ver[0], 0.5)
                    assert p.flush(60)
                    p.close()
                v = settle()
                got = ray.get(store.get_weights.remote())
                bad = same_as_today(got, want)
                checks[f"{name}/{label}"] = {"version": v == ver[0], "equal": not bad, "diffs": bad[:10]}
                log(f"store holds, {name} {label}: version {v == ver[0]}, equal to the learner's dict: {not bad} {bad[:5]}")
        out["store_equal"] = checks
        # 2. the time on the calling thread, and the Python lock the push takes from another thread
        times = {}
        for name in ("flag_off", "tensors", "out_of_band"):
            calls, thread = [], []
            p = F.WeightPusher(store, out_of_band=(name == "out_of_band")) if name != "flag_off" else None
            for _ in range(a.pushes):
                ver[0] += 1
                torch.cuda.synchronize()
                th0 = p.thread_s if p is not None else 0.0
                t0 = time.perf_counter()
                if p is None:
                    store.update.remote(today(gpu_sd), ver[0])
                    store.set_temp.remote(0.5)
                else:
                    p.push(gpu_sd, ver[0], 0.5)
                calls.append(time.perf_counter() - t0)
                if p is not None:
                    p.flush(60)
                    thread.append(p.thread_s - th0)
                settle()
            if p is not None:
                p.close()
            times[name] = {"calling_thread_ms": 1000 * float(np.median(calls)),
                           "push_thread_ms": 1000 * float(np.median(thread)) if thread else None}
            log(f"{name}: {times[name]}")
        base = float(np.median([lock_rate(a.seconds)[0] for _ in range(2)]))
        lock = {"loop_alone_per_s": base}
        for name in ("tensors", "out_of_band"):
            p = F.WeightPusher(store, out_of_band=(name == "out_of_band"))

            def work(p=p):
                ver[0] += 1
                p.push(gpu_sd, ver[0], 0.5)
                p.flush(60)
                settle()
            rate, runs, dt = lock_rate(a.seconds, work)
            p.close()
            held = (1.0 - rate / base) * dt / max(runs, 1)
            lock[name] = {"loop_per_s": rate, "pushes": runs, "lock_s_per_push": held, "slowdown_pct": 100 * (1 - rate / base)}
            log(f"python lock, {name}: the loop ran at {100 * rate / base:.1f}% of its speed alone during {runs} pushes: "
                f"{1000 * held:.1f} ms of the lock per push")
        out["times"] = times
        out["lock"] = lock
    finally:
        stop_hogs(hogs)
        ray.shutdown()
        log("ray: shut down")
    out["emu"] = bool(a.emu)
    out["ok"] = all(c["version"] and c["equal"] for c in out["store_equal"].values())
    log("WIRE " + ("PASSED" if out["ok"] else "FAILED"))
    with open(a.json, "w") as f:
        json.dump(out, f, indent=1)
    os._exit(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--code", default=REPO, help="the code tree to import (read before anything is imported)")
    common.add_argument("--det", action="store_true", help="deterministic CUDA (CUBLAS workspace, torch deterministic algorithms)")
    common.add_argument("--fast-update", action="store_true",
                        help="the learner also gets fast_update=True (train.py --fast-update; check: the flag-on side only)")
    common.add_argument("--msgs", required=True, help="the recording directory (msgs/msg_*.pkl)")
    common.add_argument("--group", type=int, default=1, help="messages queued at once (the loop drains them together); at most 5")
    common.add_argument("--rechunk", type=int, default=0, help="cut the messages into ones of about this many rows (0: as recorded)")

    r = sub.add_parser("record", parents=[common])
    r.add_argument("--ckpt", default=os.path.join(PROF, "ckpt", "learner_update_009750.pt"))
    r.add_argument("--ports", default="11448-11451")
    r.add_argument("--battles", type=int, default=256)
    r.add_argument("--steps", type=int, default=160000, help="decisions to write (3 updates need 3 x 36864 and a margin)")
    r.add_argument("--minutes", type=float, default=30.0)

    c = sub.add_parser("check", parents=[common])
    c.add_argument("--ckpt", default=os.path.join(PROF, "ckpt", "learner_update_009750.pt"))
    c.add_argument("--updates", type=int, default=3)
    c.add_argument("--out", required=True, help="working directory (checkpoint links, the checkpoints written after the updates)")
    c.add_argument("--json", required=True)

    g = sub.add_parser("digest", parents=[common])
    g.add_argument("--resume", required=True)
    g.add_argument("--work", required=True, help="an empty directory for the checkpoint link")
    g.add_argument("--fast", action="store_true")
    g.add_argument("--start", type=int, default=0)
    g.add_argument("--updates", type=int, default=1)
    g.add_argument("--save", default=None)
    g.add_argument("--json", required=True)

    t = sub.add_parser("time", parents=[common])
    t.add_argument("--path", choices=("off", "on"), required=True)
    t.add_argument("--ckpt", default=os.path.join(PROF, "ckpt", "learner_update_009750.pt"))
    t.add_argument("--work", required=True)
    t.add_argument("--updates", type=int, default=3)
    t.add_argument("--reps", type=int, default=2)
    t.add_argument("--emu", action="store_true")
    t.add_argument("--nmem", type=int, default=11)
    t.add_argument("--npy", type=int, default=11)
    t.add_argument("--bw", type=int, default=2)
    t.add_argument("--hog-cpus", default="2-15")
    t.add_argument("--hog-seconds", type=float, default=1500)
    t.add_argument("--store", choices=("fake", "ray"), default="fake",
                   help="fake: Store, which times a pickle of each dict; ray: a local Ray's real WeightStore actor")
    t.add_argument("--push-tensors", action="store_true",
                   help="flag on: push the tensors themselves (as reviewed in b827896), not out of band")
    t.add_argument("--json", required=True)

    w = sub.add_parser("wire")
    w.add_argument("--ckpt", default=os.path.join(PROF, "ckpt", "learner_update_009750.pt"))
    w.add_argument("--pushes", type=int, default=20)
    w.add_argument("--seconds", type=float, default=8.0)
    w.add_argument("--store-pushes", type=int, default=6, help="updates timed to the store's reply, per way of pushing")
    w.add_argument("--emu", action="store_true", help="the box-a emulation's hogs on --hog-cpus (Ray's processes go there too)")
    w.add_argument("--nmem", type=int, default=11)
    w.add_argument("--npy", type=int, default=11)
    w.add_argument("--bw", type=int, default=2)
    w.add_argument("--hog-cpus", default="2-15")
    w.add_argument("--hog-seconds", type=float, default=900)
    w.add_argument("--json", required=True)

    k = sub.add_parser("compare")
    k.add_argument("--off", nargs="+", required=True, help="time JSONs of the flag off")
    k.add_argument("--on", nargs="+", required=True, help="time JSONs of the flag on")
    k.add_argument("--json", default=None)
    a = ap.parse_args()
    {"record": cmd_record, "check": cmd_check, "digest": cmd_digest, "time": cmd_time, "wire": cmd_wire,
     "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
