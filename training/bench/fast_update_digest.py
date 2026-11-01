#!/usr/bin/env python
"""
The live learner update replayed outside the run, to hold --fast-update to the code without it.

The LearnerActor of a code tree (--code; default: this repository) is built exactly as train.py
builds it from the live run's flags (no Ray: the weight store is a stub), resumes from a learner
checkpoint (--ckpt-dir; default <prof>/ckpt, which holds learner_update_009750.pt of the live run)
and runs LearnerActor._perform_update on update batches recorded from the same checkpoint
(<prof>/batches/batch_{0,1,2}.pt). Before every repetition the network, the optimizer, the
schedule and the random number generators are put back to the checkpoint's, so each repetition
is the same update. Ported from the box c profiler (~/dev/fast/prof/harness.py), whose digests of
v2-humanaux 4cfd496 (out/digest_det/pristine.json) these are comparable with: the same hashes of
the same tensors.

    python bench/fast_update_digest.py [--code DIR] [--fast-update] [--deterministic] CMD ...

    digest  --regime stop49|t96kl|t96|live [--target-kl X] [--reps 2] [--batch 0] [--save CKPT] [--json OUT]
            hashes of the weights, the fp32 masters and Adam's moments after one update, the
            update's figures as float.hex (kl, clip, ent, v, pg, total and the aux sums) and n_mb.
            --save writes the learner checkpoint after the first repetition (the cross-load test).
    scan    --target-kl X [X ...]   n_mb for each early-stop target (to find a stop mid epoch 2)
    trace   [--batch 0]   the update's own KL at every minibatch, the epoch means the early stop
            tests, and every minibatch where some target can stop it, with that target
    time    --regime t96kl [--reps 4] [--warmup 1] [--chunks 200] [--smi] [--json OUT]
            seconds per update and per minibatch, peak GPU memory, GPU busy (nvidia-smi, 200 ms),
            and how the CPU served each update ("sched": seconds on a CPU and waiting in the run
            queue for the main and autograd threads, and the busy seconds of the harness's CPUs
            that went to other processes, "foreign": a timing with foreign seconds was shared)
    syncs   --regime stop49     every host-device synchronisation of one update, by source line

Regimes: stop49 = target_kl 1e-12, so the early stop fires at the first minibatch of epoch 2
(minibatch 49); t96kl = target_kl 1e9, so the test runs at every minibatch of epoch 2 and never
fires (96 minibatches, as in the live run, whose KL stays under 1.5 x 0.02); t96 = no target;
live = the flags' own target (0.02).

--deterministic (the bit-identity runs): CUBLAS_WORKSPACE_CONFIG and
torch.use_deterministic_algorithms(True, warn_only=True), set before torch is imported.
--fast-update needs a tree that has the flag; the pristine tree is run without it.

Reads only the checkpoint, the batches and the code. Writes only --json, --save and nothing else.
Starts no Ray, no server and no trainer. The nvidia-smi it starts for --smi is its own child,
stopped by its PID.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CODE = os.path.dirname(HERE)
DEFAULT_PROF = os.path.expanduser("~/dev/fast/prof")


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", default=DEFAULT_CODE, help="the code tree whose learner runs (default: this repository)")
    ap.add_argument("--prof", default=DEFAULT_PROF, help="where ckpt/ and batches/ are (default ~/dev/fast/prof)")
    ap.add_argument("--ckpt-dir", default=None, help="resume from the newest checkpoint here (default <prof>/ckpt)")
    ap.add_argument("--fast-update", action="store_true", help="build the learner with train.py --fast-update")
    ap.add_argument("--deterministic", action="store_true", help="deterministic kernels (the bit-identity runs)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def regime(p, default):
        p.add_argument("--regime", default=default, choices=("stop49", "t96kl", "t96", "live"))
        p.add_argument("--target-kl", type=float, default=None, help="this early-stop target instead of the regime's")

    d = sub.add_parser("digest")
    regime(d, "stop49")
    d.add_argument("--reps", type=int, default=2)
    d.add_argument("--batch", type=int, default=0)
    d.add_argument("--chunks", type=int, default=1)
    d.add_argument("--save", default=None, help="write the learner checkpoint here after the first repetition")
    d.add_argument("--json", default=None)
    s = sub.add_parser("scan")
    s.add_argument("--target-kl", type=float, nargs="+", required=True)
    s.add_argument("--batch", type=int, default=0)
    s.add_argument("--json", default=None)
    t = sub.add_parser("time")
    regime(t, "t96kl")
    t.add_argument("--reps", type=int, default=4)
    t.add_argument("--warmup", type=int, default=1)
    t.add_argument("--batches", type=int, nargs="+", default=[0, 1, 2])
    t.add_argument("--chunks", type=int, default=200, help="the update's data fed in this many pieces (the live learner gets ~200)")
    t.add_argument("--smi", action="store_true", help="sample nvidia-smi every 200 ms during the timed repetitions")
    t.add_argument("--json", default=None)
    y = sub.add_parser("syncs")
    regime(y, "stop49")
    y.add_argument("--json", default=None)
    k = sub.add_parser("trace")
    k.add_argument("--batch", type=int, default=0)
    k.add_argument("--chunks", type=int, default=1)
    k.add_argument("--json", default=None)
    return ap.parse_args()


A = parse()
CODE = os.path.abspath(os.path.expanduser(A.code))
PROF = os.path.abspath(os.path.expanduser(A.prof))
CKPT_DIR = os.path.abspath(os.path.expanduser(A.ckpt_dir)) if A.ckpt_dir else os.path.join(PROF, "ckpt")
sys.path.insert(0, CODE)
os.chdir(CODE)          # vocab.json is opened by a relative path

if A.deterministic:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import asyncio  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

if A.deterministic:
    torch.use_deterministic_algorithms(True, warn_only=True)
    warnings.simplefilter("always")
    _nd_seen = {}

    def _show(message, category, filename, lineno, file=None, line=None):
        key = str(message)[:160]
        if key not in _nd_seen:
            _nd_seen[key] = 1
            print(f"[nondeterministic] {key}", flush=True)
    warnings.showwarning = _show

import train as T  # noqa: E402
import learner as learner_mod  # noqa: E402
from learner import LearnerActor  # noqa: E402

# The live run's trainer flags (run.json of 2026-10-17 10:38: the top-player term is off), with
# this harness's checkpoint directory. The ports are never opened.
LIVE = ["--mode", "ppo", "--ports", "11440-11443", "--battles", "2304", "--rooms-per-pair", "16", "--teacher", "psbot",
        "--steps-per-update", "36864", "--minibatch", "768", "--seq-len", "256", "--grad-accum", "1", "--save-every", "50",
        "--ckpt-dir", CKPT_DIR, "--obs-matchup", "--obs-movesets", "--infer-actors", "3", "--infer-sync-s", "1",
        "--set-head-coef", "0.1", "--opp-action-aux-coef", "0.1", "--read-bonus", "0.3", "--read-anneal", "20000000:45000000",
        "--opponent-seats", "setupnet:0.15", "--boosted-row-weight", "2", "--lr-hold-steps", "479713", "--lr-total-steps", "1113566"]


def log(*a):
    print(*a, flush=True)


class _R:
    def remote(self, *a, **k):
        return None


class StubStore:
    update = _R()
    set_temp = _R()


# Every PPOUpdateStats the learner gets back, captured where learner.py calls ppo_update.
CAPTURED = []
_orig_ppo_update = learner_mod.ppo_update


def _capturing_ppo_update(*a, **k):
    st = _orig_ppo_update(*a, **k)
    CAPTURED.append((st, bool(k.get("fast_update", False))))
    return st


learner_mod.ppo_update = _capturing_ppo_update


async def build():
    argv = LIVE + (["--fast-update"] if A.fast_update else [])
    args = T.parse_args(argv)
    cfg = T.build_config(args)
    kw = T.learner_speed_kwargs(args) if hasattr(T, "learner_speed_kwargs") else {}
    if A.fast_update and not kw.get("fast_update"):
        raise SystemExit(f"{CODE} has no --fast-update")
    L = LearnerActor(cfg, StubStore(), **kw)
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        await L._startup_task
    log(buf.getvalue().strip()[-400:])
    log(f"code {CODE} fast_update={getattr(L, 'fast_update', None)} resumed update {L.update_idx} from {CKPT_DIR}")
    return L


def snapshot(L):
    return {
        "net": {k: v.detach().clone() for k, v in L.net.state_dict().items()},
        "opt": copy.deepcopy(L.opt.state_dict()),
        "sched": copy.deepcopy(L.sched.state_dict()),
        "upd": L.update_idx, "steps": L.total_steps, "eps": L.total_episodes,
        "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "np": np.random.get_state(),
    }


def restore(L, s):
    with torch.no_grad():
        L.net.load_state_dict(s["net"])
    L.opt.load_state_dict(copy.deepcopy(s["opt"]))
    L.sched.load_state_dict(copy.deepcopy(s["sched"]))
    L.update_idx, L.total_steps, L.total_episodes = s["upd"], s["steps"], s["eps"]
    torch.set_rng_state(s["rng"])
    torch.cuda.set_rng_state_all(s["cuda_rng"])
    np.random.set_state(s["np"])


def load_batch(k: int):
    return torch.load(os.path.join(PROF, "batches", f"batch_{k}.pt"), weights_only=False)


def feed(L, b, chunks: int):
    """The batch into the learner's dataset in `chunks` pieces (the live learner gets one per worker submit)."""
    data, names = b["data"], b["names"]
    n = int(data[1].shape[0])
    k = max(1, int(chunks))
    edges = [round(i * n / k) for i in range(k + 1)]
    for a, z in zip(edges[:-1], edges[1:]):
        part = [d[a:z] for d in data]
        extras = dict(zip(names, part[8:]))
        L.dataset.add_steps(*part[:8], extras=extras if extras else None)


async def one_update(L, b, snap, chunks: int, probe=None):
    """One update from snap. With probe (a function returning a snapshot): also the probe's
    snapshots just before and just after the update, as cpu_delta's figures (a fifth value)."""
    restore(L, snap)
    feed(L, b, chunks)
    torch.cuda.synchronize()
    buf = io.StringIO()
    n0 = len(CAPTURED)
    p0 = probe() if probe else None
    w0 = time.time()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        await L._perform_update()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    p1 = probe() if probe else None
    WINDOWS.append((w0, time.time()))
    line = [x for x in buf.getvalue().splitlines() if "[learner] upd=" in x]
    if len(CAPTURED) != n0 + 1 or not line:
        raise RuntimeError(f"the update did not complete: {buf.getvalue()[-1500:]}")
    st, fast = CAPTURED[-1]
    if probe:
        return dt, st, fast, line[-1], cpu_delta(p0, p1)
    return dt, st, fast, line[-1]


# The wall-clock span of every _perform_update (one_update appends; the GPU-busy figure of `time`
# is the nvidia-smi samples inside these spans, so the restores and feeds between them do not count).
WINDOWS = []


def set_regime(L, regime: str, target_kl=None):
    if target_kl is not None:
        L.cfg = dataclasses.replace(L.cfg, target_kl=float(target_kl))
    elif regime == "stop49":
        L.cfg = dataclasses.replace(L.cfg, target_kl=1e-12)
    elif regime == "t96kl":
        L.cfg = dataclasses.replace(L.cfg, target_kl=1e9)
    elif regime == "t96":
        L.cfg = dataclasses.replace(L.cfg, target_kl=None)
    elif regime != "live":
        raise ValueError(regime)
    log(f"regime {regime} target_kl={L.cfg.target_kl} update_epochs={L.cfg.update_epochs}")


def _digest_tensors(named):
    h = hashlib.sha256()
    for k, t in named:
        h.update(k.encode())
        h.update(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes() if t.dtype != torch.bool else t.cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def digests(L):
    w = _digest_tensors(sorted(L.net.state_dict().items()))
    sd = L.opt.state_dict()
    m = _digest_tensors([(str(i), t) for i, t in enumerate(sd["masters"])])
    st = []
    for pid, s in sorted(sd["inner"]["state"].items()):
        for k in ("exp_avg", "exp_avg_sq"):
            st.append((f"{pid}.{k}", s[k]))
    adam = _digest_tensors(st)
    steps = sorted({float(s["step"]) for s in sd["inner"]["state"].values()})
    return {"weights": w, "masters": m, "adam": adam, "steps": steps}


def stats_bits(st) -> dict:
    """The update's figures as exact hex floats."""
    out = {}
    for f in ("approx_kl", "clip_frac", "entropy", "v_loss", "pg_loss", "total_loss"):
        out[f] = float(getattr(st, f)).hex()
    for k, v in sorted((getattr(st, "aux", None) or {}).items()):
        out["aux." + k] = float(v).hex()
    return out


async def cmd_digest():
    L = await build()
    snap = snapshot(L)
    b = load_batch(A.batch)
    set_regime(L, A.regime, A.target_kl)
    out = {"code": CODE, "fast_update": A.fast_update, "regime": A.regime, "target_kl": L.cfg.target_kl,
           "ckpt_dir": CKPT_DIR, "batch": A.batch, "reps": {}}
    for r in range(A.reps):
        dt, st, fast, line = await one_update(L, b, snap, A.chunks)
        dg = digests(L)
        rec = dict(dg, n_mb=int(st.n_mb), stats=stats_bits(st), passed_fast_update=fast, sec=round(dt, 3), update=L.update_idx)
        out["reps"][str(r)] = rec
        log(f"rep {r}: n_mb={st.n_mb} sec={dt:.2f} weights={dg['weights']} masters={dg['masters']} adam={dg['adam']} "
            f"steps={dg['steps']} fast={fast}")
        log(f"   stats: {' '.join(f'{k}={v}' for k, v in rec['stats'].items())}")
        if r == 0 and A.save:
            os.makedirs(os.path.dirname(os.path.abspath(A.save)), exist_ok=True)
            L._save_checkpoint(os.path.abspath(A.save))
            log(f"   checkpoint written: {A.save} (update {L.update_idx})")
    same = len({json.dumps({k: v for k, v in r.items() if k != "sec"}, sort_keys=True) for r in out["reps"].values()}) == 1
    out["reps_agree"] = same
    log(f"repetitions agree: {same}")
    if A.json:
        with open(A.json, "w") as f:
            json.dump(out, f, indent=1)


async def cmd_scan():
    L = await build()
    snap = snapshot(L)
    b = load_batch(A.batch)
    out = {}
    for tk in A.target_kl:
        L.cfg = dataclasses.replace(L.cfg, target_kl=float(tk))
        dt, st, fast, line = await one_update(L, b, snap, 1)
        out[repr(tk)] = int(st.n_mb)
        log(f"target_kl {tk:g}: n_mb={st.n_mb}")
    if A.json:
        with open(A.json, "w") as f:
            json.dump(out, f, indent=1)


def stop_at(kls, per_epoch: int, target: float) -> int:
    """The minibatch at which ppo_update's KL test stops for this target (len(kls): never), in its own Python floats."""
    n = 0
    for epoch in range(len(kls) // per_epoch):
        epoch_kl, epoch_mb = 0.0, 0
        for _ in range(per_epoch):
            epoch_kl += kls[n]
            n += 1
            epoch_mb += 1
            if epoch > 0 and target and (epoch_kl / epoch_mb) > target * 1.5:
                return n
    return n


def stop_targets(kls, per_epoch: int) -> dict:
    """{minibatch: a target that stops the update there} for every minibatch where some target can (as tests/test_fast_update.py)."""
    out, best, n = {}, None, 0
    for epoch in range(len(kls) // per_epoch):
        epoch_kl, epoch_mb = 0.0, 0
        for _ in range(per_epoch):
            epoch_kl += kls[n]
            n += 1
            epoch_mb += 1
            if epoch == 0:
                continue
            mean = epoch_kl / epoch_mb
            if best is None or mean > best:
                lo = best if best is not None else mean - max(abs(mean), 1e-6)
                target = (lo + mean) / 3.0
                if target and stop_at(kls, per_epoch, target) == n:
                    out[n] = target
                best = mean
    return out


async def cmd_trace():
    """
    The approx_kl of every minibatch of one update that never stops (target_kl 1e9), read back as
    the update computed it: spies keep copies of the log-probabilities masked_logprob_entropy
    returned and of each minibatch's old log-probabilities, and (old - new).mean() of those is the
    update's own expression on the same tensors. From it: the epoch means the early stop tests,
    and every minibatch where some target can stop this update, with one target each (half way
    between the epoch means, so rounding cannot move it). `digest --target-kl` then checks a stop.
    """
    import ppo_core as core
    L = await build()
    snap = snapshot(L)
    b = load_batch(A.batch)
    L.cfg = dataclasses.replace(L.cfg, target_kl=1e9)
    news, olds = [], []
    real = core.masked_logprob_entropy
    real_iter = core.AsyncEpisodeDataset.iter_minibatches

    def spy(logits, mask, actions, *a, **k):
        lp, ent = real(logits, mask, actions, *a, **k)
        news.append(lp.detach().clone())
        return lp, ent

    def iter_spy(self, *a, **k):
        for x in real_iter(self, *a, **k):
            olds.append(x[2].detach().clone())
            yield x

    core.masked_logprob_entropy = spy
    core.AsyncEpisodeDataset.iter_minibatches = iter_spy
    try:
        dt, st, fast, line = await one_update(L, b, snap, A.chunks)
    finally:
        core.masked_logprob_entropy = real
        core.AsyncEpisodeDataset.iter_minibatches = real_iter
    n_mb = int(st.n_mb)
    if not (len(news) == len(olds) == n_mb):
        raise SystemExit(f"the spies saw {len(news)} and {len(olds)} minibatches of {n_mb}")
    kls = [(o.to(n.device) - n).mean().item() for o, n in zip(olds, news)]
    per_epoch = n_mb // int(L.cfg.update_epochs)
    means = []
    for e in range(1, n_mb // per_epoch):
        s = 0.0
        for j in range(per_epoch):
            s += kls[e * per_epoch + j]
            means.append(s / (j + 1))
    targets = stop_targets(kls, per_epoch)
    mid = sorted(k for k in targets if k % per_epoch not in (0, 1))
    log(f"batch {A.batch}: n_mb={n_mb} ({per_epoch} an epoch); the KL test's epoch means, minibatches {per_epoch + 1}-{n_mb}:")
    log("   " + " ".join(f"{m:.5f}" for m in means))
    log(f"minibatches where a target can stop the update: {sorted(targets)}; inside an epoch: {mid}")
    for k, v in sorted(targets.items()):
        log(f"   stop at {k}: target_kl {v!r}")
    if A.json:
        with open(A.json, "w") as f:
            json.dump({"batch": A.batch, "n_mb": n_mb, "per_epoch": per_epoch, "kls": kls, "means": means,
                       "targets": {str(k): v for k, v in targets.items()}, "mid_epoch": mid}, f, indent=1)


def thread_cpu(native_id: int) -> float:
    try:
        with open(f"/proc/self/task/{native_id}/stat") as f:
            parts = f.read().rsplit(")", 1)[1].split()
        return (int(parts[11]) + int(parts[12])) / os.sysconf("SC_CLK_TCK")
    except Exception:
        return float("nan")


# How the CPU served the update, per repetition (Linux /proc; what box a's live learner can be read
# for too, so the two compare). For the main thread and PyTorch's autograd threads (which run the
# backward while the main thread waits): seconds on a CPU, seconds runnable but waiting for one
# (the run queue), and the switches. For the CPUs this process may use: their busy seconds, and of
# those the "foreign" ones, spent on anything but this process and its nvidia-smi child (another
# process sharing the harness's CPUs). And the CPU each of the two threads was last on.


def _sched(tid: int) -> dict:
    out = {}
    try:
        with open(f"/proc/self/task/{tid}/schedstat") as f:
            run, wait, _slices = f.read().split()
        out["run"], out["wait"] = int(run) / 1e9, int(wait) / 1e9
        with open(f"/proc/self/task/{tid}/status") as f:
            for ln in f:
                if ln.startswith("voluntary_ctxt_switches"):
                    out["vol"] = int(ln.split()[1])
                elif ln.startswith("nonvoluntary_ctxt_switches"):
                    out["invol"] = int(ln.split()[1])
        with open(f"/proc/self/task/{tid}/stat") as f:
            out["cpu"] = int(f.read().rsplit(")", 1)[1].split()[36])
    except (OSError, ValueError):
        pass
    return out


def _autograd_tids() -> list:
    out = []
    for t in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{t}/comm") as f:
                if f.read().startswith("pt_autograd"):
                    out.append(int(t))
        except OSError:
            pass
    return sorted(out)


def _pid_cpu(pid) -> float:
    try:
        with open(f"/proc/{pid}/stat") as f:
            p = f.read().rsplit(")", 1)[1].split()
        return (int(p[11]) + int(p[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError):
        return 0.0


def _cpus_busy(cpus) -> float:
    busy = 0
    with open("/proc/stat") as f:
        for ln in f:
            if ln.startswith("cpu") and ln[3:4].isdigit():
                p = ln.split()
                if int(p[0][3:]) in cpus:
                    v = [int(x) for x in p[1:9]]          # user nice system idle iowait irq softirq steal
                    busy += sum(v) - v[3] - v[4]
    return busy / os.sysconf("SC_CLK_TCK")


def _all_threads() -> dict:
    """{tid: (comm, seconds on a CPU)} of every thread of this process."""
    out = {}
    for t in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{t}/comm") as f:
                comm = f.read().strip()
            with open(f"/proc/self/task/{t}/schedstat") as f:
                out[int(t)] = (comm, int(f.read().split()[0]) / 1e9)
        except (OSError, ValueError):
            pass
    return out


def cpu_snapshot(main_tid: int, child_pid=None) -> dict:
    cpus = sorted(os.sched_getaffinity(0))
    return {"main": _sched(main_tid), "autograd": {t: _sched(t) for t in _autograd_tids()},
            "own": _pid_cpu("self") + (_pid_cpu(child_pid) if child_pid else 0.0),
            "child": _pid_cpu(child_pid) if child_pid else 0.0, "threads": _all_threads(), "main_tid": main_tid,
            "cpus": cpus, "busy": _cpus_busy(set(cpus))}


def cpu_delta(a: dict, b: dict) -> dict:
    def d(x, y):
        return {k: round(y[k] - x[k], 4) for k in ("run", "wait", "vol", "invol") if k in x and k in y}
    ag = {}
    for t, y in b["autograd"].items():
        x = a["autograd"].get(t, {k: 0 for k in ("run", "wait", "vol", "invol")})
        for k, v in d(x, y).items():
            ag[k] = round(ag.get(k, 0) + v, 4)
    busy, own = b["busy"] - a["busy"], b["own"] - a["own"]
    # the other threads of this process (not main, not autograd) that ran most, and the nvidia-smi child
    skip = {b["main_tid"], *b["autograd"]}
    others = sorted(((round(v[1] - a["threads"].get(t, (v[0], 0.0))[1], 3), f"{v[0]}/{t}") for t, v in b["threads"].items()
                     if t not in skip), reverse=True)
    return {"main": d(a["main"], b["main"]), "autograd": ag, "main_cpu_at_end": b["main"].get("cpu"),
            "other_threads": round(sum(x[0] for x in others), 3), "other_top": others[:3],
            "smi_child": round(b["child"] - a["child"], 3),
            "autograd_cpu_at_end": [y.get("cpu") for y in b["autograd"].values()],
            "cpus": a["cpus"] if len(a["cpus"]) <= 8 else f"{len(a['cpus'])} CPUs",
            "cpus_busy": round(busy, 3), "own": round(own, 3),
            # meaningful when the harness is pinned to a few CPUs (unpinned it is the rest of the box)
            "foreign": round(busy - own, 3)}


async def cmd_time():
    L = await build()
    snap = snapshot(L)
    batches = [load_batch(k) for k in A.batches]
    set_regime(L, A.regime, A.target_kl)
    for w in range(A.warmup):
        dt, st, fast, _ = await one_update(L, batches[w % len(batches)], snap, A.chunks)
        log(f"warmup {w}: {dt:.2f}s n_mb={st.n_mb}")
    main_tid = threading.get_native_id()
    py_threads = {t.native_id: t.name for t in threading.enumerate()}
    log("threads: " + ", ".join(f"{tid} {v[0]}{' (Python ' + py_threads[tid] + ')' if tid in py_threads else ''}"
                                 for tid, v in sorted(_all_threads().items())))
    smi = smi_path = None
    if A.smi:
        smi_path = (A.json or os.path.join(os.getcwd(), "time")) + ".smi.csv"
        smi_f = open(smi_path, "w")
        smi = subprocess.Popen(["nvidia-smi", "-i", os.environ.get("SMI_GPU", "0"),
                                "--query-gpu=timestamp,utilization.gpu,memory.used", "--format=csv,noheader,nounits",
                                "-lms", "200"], stdout=smi_f, stderr=subprocess.DEVNULL)
        await asyncio.sleep(1.0)
    torch.cuda.reset_peak_memory_stats()
    res = []
    t_start = time.time()
    w_first = len(WINDOWS)
    for r in range(A.reps):
        b = batches[r % len(batches)]
        m0 = thread_cpu(main_tid)
        w0 = time.time()
        dt, st, fast, line, cpu = await one_update(L, b, snap, A.chunks,
                                                   probe=lambda: cpu_snapshot(main_tid, smi.pid if smi else None))
        w1 = time.time()
        m1 = thread_cpu(main_tid)
        n = int(st.n_mb)
        res.append({"rep": r, "sec": dt, "n_mb": n, "per_mb": dt / max(n, 1), "main_cpu": m1 - m0, "t0": w0, "t1": w1,
                    "sched": cpu})
        mn, ag = cpu["main"], cpu["autograd"]
        log(f"rep {r}: {dt:.3f}s n_mb={n} per_mb={1000 * dt / max(n, 1):.1f}ms main_cpu={m1 - m0:.2f}s fast={fast} | "
            f"{line[line.find('kl='):line.find('kl=') + 60]}")
        log(f"   sched: main run {mn.get('run', 0):.2f}s runq {mn.get('wait', 0):.3f}s invol {mn.get('invol')} on cpu "
            f"{cpu['main_cpu_at_end']} | autograd run {ag.get('run', 0):.2f}s runq {ag.get('wait', 0):.3f}s on cpu "
            f"{cpu['autograd_cpu_at_end']} | cpus {cpu['cpus']} busy {cpu['cpus_busy']:.2f}s own {cpu['own']:.2f}s "
            f"foreign {cpu['foreign']:.2f}s | other threads {cpu['other_threads']:.2f}s {cpu['other_top']} smi {cpu['smi_child']:.2f}s")
    t_end = time.time()
    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_res = torch.cuda.max_memory_reserved() / 2**30
    util = None
    if smi is not None:
        smi.terminate()
        try:
            smi.wait(timeout=5)
        except subprocess.TimeoutExpired:
            smi.kill()
            smi.wait()
        smi_f.close()
        util = {"all": smi_window(smi_path, [(t_start, t_end)]), "updates": smi_window(smi_path, WINDOWS[w_first:])}
    out = {"cmd": "time", "code": CODE, "fast_update": A.fast_update, "regime": A.regime, "chunks": A.chunks, "reps": res,
           "median_sec": float(np.median([x["sec"] for x in res])),
           "median_per_mb_ms": float(np.median([x["per_mb"] for x in res]) * 1000),
           "peak_alloc_gb": round(peak_alloc, 3), "peak_reserved_gb": round(peak_res, 3), "gpu_util": util,
           "sched_median": {
               "main_run": float(np.median([x["sched"]["main"].get("run", np.nan) for x in res])),
               "main_runq": float(np.median([x["sched"]["main"].get("wait", np.nan) for x in res])),
               "autograd_run": float(np.median([x["sched"]["autograd"].get("run", np.nan) for x in res])),
               "autograd_runq": float(np.median([x["sched"]["autograd"].get("wait", np.nan) for x in res])),
               "foreign": float(np.median([x["sched"]["foreign"] for x in res])),
               "foreign_max": float(max(x["sched"]["foreign"] for x in res))}}
    log(json.dumps({k: v for k, v in out.items() if k != "reps"}))
    if A.json:
        with open(A.json, "w") as f:
            json.dump(out, f, indent=1)


def smi_window(path: str, spans):
    """
    Mean GPU utilisation (and samples, and the peak memory used) of the nvidia-smi lines inside any
    of the (t0, t1) spans (local wall clock). nvidia-smi's figure is the share of its last sample
    period with a kernel running, so the first sample of a span still looks back a little.
    """
    import datetime
    vals, mem = [], []
    with open(path) as f:
        for ln in f:
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) < 3:
                continue
            try:
                ts = datetime.datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
                u = float(parts[1])
                m = float(parts[2])
            except ValueError:
                continue
            if any(t0 <= ts <= t1 for t0, t1 in spans):
                vals.append(u)
                mem.append(m)
    return {"mean": round(float(np.mean(vals)), 1) if vals else None, "n": len(vals),
            "max_mem_mib": max(mem) if mem else None}


async def cmd_syncs():
    L = await build()
    snap = snapshot(L)
    b = load_batch(0)
    set_regime(L, A.regime, A.target_kl)
    await one_update(L, b, snap, 1)                 # warm (compile)
    restore(L, snap)
    feed(L, b, 1)
    counts = {}
    n0 = len(CAPTURED)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                await L._perform_update()
        finally:
            torch.cuda.set_sync_debug_mode(0)
    st, _fast = CAPTURED[n0]
    n_mb = int(st.n_mb)
    for w in ws:
        if "called a synchronizing CUDA operation" not in str(w.message):    # not the debug mode's own note
            continue
        key = f"{os.path.relpath(w.filename, CODE) if w.filename.startswith(CODE) else w.filename}:{w.lineno}"
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    log(f"n_mb={n_mb} syncs={total} per_mb={total / max(n_mb, 1):.2f}")
    src = {}
    for k, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        fn, ln = k.rsplit(":", 1)
        path = os.path.join(CODE, fn) if not os.path.isabs(fn) else fn
        text = ""
        try:
            with open(path) as f:
                text = f.readlines()[int(ln) - 1].strip()[:110]
        except Exception:
            pass
        src[k] = text
        log(f"{c:6d}  {c / max(n_mb, 1):5.2f}/mb  {k}  | {text}")
    if A.json:
        with open(A.json, "w") as f:
            json.dump({"n_mb": n_mb, "total": total, "counts": counts, "src": src}, f, indent=1)


def main():
    fn = {"digest": cmd_digest, "scan": cmd_scan, "time": cmd_time, "syncs": cmd_syncs, "trace": cmd_trace}[A.cmd]
    asyncio.run(fn())
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
