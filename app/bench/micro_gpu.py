#!/usr/bin/env python
"""
Micro-benchmark of the model on the GPU: the inference actor's call and the learner's
minibatch, each in short bursts, without Ray, servers or a training run.

Sections (pick with --sections, comma separated):

  infer      InferenceActor.infer_batch as inference.py runs it, at several batch sizes, with a
             breakdown: key-value cache gather and store, block-mask creation, encoder, layers,
             readout.
  learn      one learner minibatch as ppo_core.ppo_update runs it (bfloat16 weights, fp32
             master optimizer): forward, block mask, loss, backward, optimizer step.
  precision  the same minibatch, reduced in size, with (a) bfloat16 weights as coded,
             (b) float32 weights, (c) float32 weights under bfloat16 autocast. And the inference
             forward in the same three.
  proto      a prototype of the same attention computed per turn (64 current tokens against
             64 current + up to 64 earlier field tokens) with plain matmuls and no block mask,
             no flex_attention and no torch.compile. First checked against the model's own
             output and gradients, then timed. It uses the model's weights unchanged.
  kvpool     the per-battle cache kept as one tensor and indexed (what inference.py does now),
             against the per-battle clone/cat lists it replaced (kept here as ListCacheActor).
  fastmask   the block mask built from the episode ids at block level (what ppo_core.py does
             now) against torch's create_block_mask evaluated on every token pair (what it
             did, still reachable by switching ppo_core._FAST_BLOCK_MASKS off). First checked:
             the BlockMask tensors and the model outputs are bit-for-bit the same. Then the
             learner minibatch and the inference call are timed both ways, the latter also
             with the old per-battle lists.

Both changes were prototypes in this file when the audit was written (fast_create_block_mask
and the pooled actor). They are in the training code now. bench/micro_before_after.py times
the code before them against the code after them, and tests/test_speed_*.py compare the two
bit for bit: prefer those. The sections here stay as a self-contained second opinion.
  share      (not in the default list) two processes on the one GPU, as in training: a child
             process loops learner minibatches while this one times the inference call. About
             ten seconds with both busy. Run it only when nothing else needs the GPU.

Real observations come from bench/micro_obs.py (--obs obs_samples.npz). Random numbers would
index outside the embedding tables.

    python bench/micro_gpu.py --obs <dir>/obs_samples.npz --out <dir>
    python bench/micro_gpu.py --obs <dir>/obs_samples.npz --device cpu --sections proto --check-only

Every timed burst is a handful of calls. The first call of each kind compiles the attention
kernel (mostly CPU). All output is ASCII.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- helpers

def smi() -> Dict[str, Any]:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        name, util, used, total = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        r2 = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=10)
        apps = [ln.strip() for ln in r2.stdout.strip().splitlines() if ln.strip()]
        return {"gpu": name, "util_percent": int(util), "memory_used_mib": int(used), "memory_total_mib": int(total), "compute_apps": apps}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


class Clock:
    """Accumulates wall time per label, synchronising the GPU at both ends of each span."""

    def __init__(self, dev: str):
        self.cuda = dev.startswith("cuda")
        self.t: Dict[str, float] = {}
        self.n: Dict[str, int] = {}

    def sync(self) -> None:
        if self.cuda:
            import torch
            torch.cuda.synchronize()

    @contextlib.contextmanager
    def span(self, label: str):
        self.sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.t[label] = self.t.get(label, 0.0) + (time.perf_counter() - t0)
            self.n[label] = self.n.get(label, 0) + 1

    def wrap(self, fn: Callable, label: str) -> Callable:
        def w(*a, **k):
            with self.span(label):
                return fn(*a, **k)
        return w

    def ms(self, per: int) -> Dict[str, float]:
        return {k: round(v * 1000.0 / max(per, 1), 3) for k, v in self.t.items()}


def med(xs: List[float]) -> float:
    return float(statistics.median(xs)) if xs else float("nan")


def load_obs(path: str):
    import numpy as np
    d = np.load(path)
    return d["obs"], d["ep"]


def ensure_compiled() -> None:
    """Compile flex_attention once, the way both actors do before their first forward."""
    import torch
    import ppo_core
    if not getattr(ppo_core, "_bench_flex_compiled", False):
        ppo_core.flex_attention = torch.compile(ppo_core.flex_attention)
        ppo_core._bench_flex_compiled = True


def build_actor(cfg_i, cls=None):
    """
    InferenceActor wraps ppo_core.flex_attention in torch.compile in its constructor. In one
    process with several sections that would wrap the wrapper again, so the single compiled
    function is put back afterwards.
    """
    import ppo_core
    from inference import InferenceActor
    ensure_compiled()
    saved = ppo_core.flex_attention
    try:
        actor = (cls or InferenceActor)(cfg_i, None)
    finally:
        ppo_core.flex_attention = saved
    actor._last_sync_time = time.time() + 1e9      # no weight store here: never try to sync
    return actor


def make_net(cfg, dev: str, precision: str):
    net = cfg.make_model().to(dev)
    if precision == "bf16":
        net.enable_bf16_recurrent_path()
    return net


def make_optimizer(cfg, net):
    """The learner's own optimizer and scheduler, built by LearnerActor's code."""
    from learner import LearnerActor
    la = object.__new__(LearnerActor)
    la.run_cfg = cfg
    la.cfg = cfg.learner
    la.net = net
    la.opt = None
    la.sched = None
    la._init_optimizer()
    return la.opt, la.sched


def synth_batch(obs_np, ep_np, n: int, mask_slice: slice, seed: int = 0):
    """n steps in learner order with plausible PPO targets."""
    import numpy as np
    import torch
    rng = np.random.default_rng(seed)
    obs = torch.from_numpy(np.ascontiguousarray(obs_np[:n]))            # float16, as the learner holds it
    ep = torch.from_numpy(np.ascontiguousarray(ep_np[:n])).long()
    mask = obs_np[:n, mask_slice].astype(np.float32)
    legal_n = np.maximum(mask.sum(axis=1), 1.0)
    act = np.array([rng.choice(np.flatnonzero(m > 0.5)) if (m > 0.5).any() else 0 for m in mask], dtype=np.int64)
    logp = (-np.log(legal_n)).astype(np.float32)
    adv = rng.standard_normal(n).astype(np.float32)
    ret = np.clip(rng.standard_normal(n) * 0.5, -1.4, 1.4).astype(np.float32)
    return obs, torch.from_numpy(act), torch.from_numpy(logp), torch.from_numpy(adv), torch.from_numpy(ret), ep


# --------------------------------------------------------------------------- the prototype

def attend_and_ff(layer, x, q, k, v, kh_t, vh, hist_valid, attn_fp32: bool = True):
    """
    One encoder layer for N independent turns.
      x           [N, S, D]   the turn's own tokens (layer input)
      q, k, v     [N, H, S, dh]
      kh_t        [N, H, dh, K]   keys of the earlier field tokens, already transposed
      vh          [N, H, K, dh]   their values
      hist_valid  [N, K] bool
    Same maths as FlexEncoderLayer.forward with pokemon_*_mask: softmax over the turn's own S
    tokens plus the valid earlier field tokens, scores soft-capped at 50.
    """
    import torch
    N, S, D = x.shape
    dh = q.shape[-1]
    scale = dh ** -0.5
    if attn_fp32:
        q, k, v, kh_t, vh = q.float(), k.float(), v.float(), kh_t.float(), vh.float()
    scores = torch.cat([q @ k.transpose(-1, -2), q @ kh_t], dim=-1) * scale
    scores = torch.tanh(scores / 50.0) * 50.0
    keep = torch.cat([hist_valid.new_ones(N, S), hist_valid], dim=1)[:, None, None, :]
    scores = scores.masked_fill(~keep, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = attn[..., :S] @ v + attn[..., S:] @ vh
    out = out.transpose(1, 2).reshape(N, S, D).to(x.dtype)
    x = x + layer.dropout(layer.out_proj(out))
    return x + layer.ff(layer.norm2(x))


def _current_turn(net, obs):
    import torch
    field_tok, p_toks, m_toks = net.encode_features(obs)
    B = obs.shape[0]
    return torch.cat([
        field_tok, p_toks, m_toks,
        net.register_toks.expand(B, 1, -1), net.actor_tok.expand(B, 1, -1), net.critic_tok.expand(B, 1, -1),
    ], dim=1)


def dense_train_forward(net, obs, ep_ids, seq_len: int, attn_fp32: bool = True, _no_history: bool = False):
    """
    Drop-in for net(obs, ep_ids, batch_seq_len=seq_len): returns (logits, v_logits, v_exp).
    _no_history is only for the check's negative control: it must change the output.
    """
    import torch
    import torch.nn.functional as F
    x = _current_turn(net, obs)
    B, S, D = x.shape
    K = net.kv_cache_len
    T = seq_len
    n_seq = B // T
    ep = ep_ids.view(n_seq, T)
    # Window j of turn t is turn t-K+j of the same packed sequence; it counts when it is the same episode.
    win = F.pad(ep, (K, 0), value=-1).unfold(1, K, 1)[:, :T]
    hist_valid = (win == ep.unsqueeze(-1)).reshape(B, K)
    if _no_history:
        hist_valid = torch.zeros_like(hist_valid)
    for layer in net.transformer:
        H = layer.n_heads
        dh = D // H
        qkv = layer.qkv_proj(layer.norm1(x)).reshape(B, S, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        kf = F.pad(k[:, :, 0, :].reshape(n_seq, T, H, dh), (0, 0, 0, 0, K, 0))
        vf = F.pad(v[:, :, 0, :].reshape(n_seq, T, H, dh), (0, 0, 0, 0, K, 0))
        kh_t = kf.unfold(1, K, 1)[:, :T].reshape(B, H, dh, K)
        vh = vf.unfold(1, K, 1)[:, :T].permute(0, 1, 2, 4, 3).reshape(B, H, K, dh)
        x = attend_and_ff(layer, x, q, k, v, kh_t, vh, hist_valid, attn_fp32)
    return net.readout_policy(x)


def dense_infer_forward(net, obs, hist, hist_len, attn_fp32: bool = True):
    """
    Drop-in for net(obs, kv_cache=..., kv_cache_len=...).
      hist      [B, L, K, D]  the same content as the model's kv cache (layer-input field tokens)
      hist_len  [B] long
    Returns (logits, v_logits, v_exp, new_hist, new_len).
    """
    import torch
    import torch.nn.functional as F
    x = _current_turn(net, obs)
    B, S, D = x.shape
    K = net.kv_cache_len
    pos = torch.arange(K, device=x.device).unsqueeze(0)
    hist_valid = pos >= (K - hist_len.unsqueeze(1))
    new_hist = []
    for i, layer in enumerate(net.transformer):
        H = layer.n_heads
        dh = D // H
        cur_field_in = x[:, 0:1, :]
        h_in = hist[:, i].to(x.dtype) * hist_valid.unsqueeze(-1).to(x.dtype)
        W, b = layer.qkv_proj.weight, layer.qkv_proj.bias
        kv = F.linear(layer.norm1(h_in), W[D:], b[D:]).reshape(B, K, 2, H, dh).permute(2, 0, 3, 1, 4)
        qkv = layer.qkv_proj(layer.norm1(x)).reshape(B, S, 3, H, dh).permute(2, 0, 3, 1, 4)
        x = attend_and_ff(layer, x, qkv[0], qkv[1], qkv[2], kv[0].transpose(-1, -2), kv[1], hist_valid, attn_fp32)
        new_hist.append(torch.cat([h_in, cur_field_in], dim=1)[:, -K:, :])
    pi, v, v_exp = net.readout_policy(x)
    return pi, v, v_exp, torch.stack(new_hist, dim=1), torch.clamp(hist_len + 1, max=K)


# --------------------------------------------------------------------------- cheap block mask

TOKENS = 64


def fast_create_block_mask(mask_mod, B, H, Q_LEN, KV_LEN, device=None, BLOCK_SIZE=128):
    """
    Same BlockMask as torch's create_block_mask for the two mask functions of ppo_core, built
    from the episode ids at block level instead of from a dense Q_LEN x KV_LEN boolean tensor
    (16384 x 16384 per packed sequence in training).

    It has create_block_mask's signature so the benchmark can swap it in without touching
    ppo_core: the episode ids are read out of the mask function's closure. A real change would
    pass them in directly.
    """
    import torch
    from torch.nn.attention.flex_attention import BlockMask

    fv = dict(zip(mask_mod.__code__.co_freevars, (c.cell_contents for c in (mask_mod.__closure__ or ()))))
    if "expanded_ids" in fv:
        ids, K = fv["expanded_ids"], int(fv["K"])
        ep = ids[:, ::TOKENS]                                   # [n_seq, T] one id per turn
        n_seq, T = ep.shape
        bt = BLOCK_SIZE // TOKENS                               # turns per block
        nb = -(-T // bt)
        if nb * bt != T:
            ep = torch.nn.functional.pad(ep, (0, nb * bt - T), value=-1)
        Tp = nb * bt
        t = torch.arange(Tp, device=ids.device)
        d = t[:, None] - t[None, :]                             # query turn minus key turn
        real = t < T
        A = (ep[:, :, None] == ep[:, None, :]) & ((d >= 0) & (d <= K) & real[:, None] & real[None, :])[None]
        partial = A.view(n_seq, nb, bt, nb, bt).permute(0, 1, 3, 2, 4).reshape(n_seq, nb, nb, bt * bt).any(-1)
        partial = partial[:, None]                              # [n_seq, 1, nb, nb]
    elif "kv_cache_len" in fv:
        if Q_LEN > BLOCK_SIZE or KV_LEN > BLOCK_SIZE:
            raise ValueError("the inference shortcut assumes one block")
        partial = torch.ones(B, 1, 1, 1, dtype=torch.bool, device=device)
    else:
        raise ValueError("unknown mask function")
    p32 = partial.to(torch.int32)
    kv_num = p32.sum(dim=-1).to(torch.int32).contiguous()
    kv_idx = torch.argsort(p32, dim=-1, descending=True, stable=True).to(torch.int32).contiguous()
    f32 = torch.zeros_like(p32)
    f_num = f32.sum(dim=-1).to(torch.int32).contiguous()
    f_idx = torch.argsort(f32, dim=-1, descending=True, stable=True).to(torch.int32).contiguous()
    return BlockMask.from_kv_blocks(kv_num, kv_idx, f_num, f_idx, BLOCK_SIZE=BLOCK_SIZE, mask_mod=mask_mod, seq_lengths=(Q_LEN, KV_LEN))


def block_masks_equal(a, b) -> Dict[str, bool]:
    import torch
    out = {}
    for name in ("kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices",
                 "q_num_blocks", "q_indices", "full_q_num_blocks", "full_q_indices"):
        x, y = getattr(a, name), getattr(b, name)
        out[name] = bool(x is None and y is None) or bool(x is not None and y is not None and x.shape == y.shape and torch.equal(x, y))
    out["seq_lengths"] = tuple(a.seq_lengths) == tuple(b.seq_lengths)
    out["BLOCK_SIZE"] = tuple(a.BLOCK_SIZE) == tuple(b.BLOCK_SIZE)
    return out


def make_list_cache_actor(cfg_i):
    """
    InferenceActor with the per-battle cache the way inference.py kept it before the pool:
    per tag a list of one small tensor per layer, cloned after every call and concatenated
    before the next. The four methods are the old code, moved here to be timed against.
    """
    import torch
    from inference import InferenceActor

    class ListCacheActor(InferenceActor):
        def _lists_init(self):
            self.kv_cache_by_tag: Dict[str, List[Any]] = {}
            self.kv_cache_len_by_tag: Dict[str, int] = {}

        def clear_cache(self, tag: str):
            self.kv_cache_by_tag.pop(tag, None)
            self.kv_cache_len_by_tag.pop(tag, None)

        def clear_all_caches(self):
            self.kv_cache_by_tag.clear()
            self.kv_cache_len_by_tag.clear()

        def _build_batched_kv_cache(self, tags):
            K = self.net.kv_cache_len
            rows_per_layer = [[] for _ in range(self.net.n_layers)]
            lens = []
            for tag in tags:
                cached = self.kv_cache_by_tag.get(tag)
                cached_len = int(self.kv_cache_len_by_tag.get(tag, 0))
                if cached is None or len(cached) != self.net.n_layers:
                    cached = [torch.zeros(1, K, self.net.d_model, device=self.device) for _ in range(self.net.n_layers)]
                    cached_len = 0
                lens.append(max(0, min(cached_len, K)))
                for i in range(self.net.n_layers):
                    rows_per_layer[i].append(cached[i])
            batched = [torch.cat(rows, dim=0) for rows in rows_per_layer]
            return batched, torch.tensor(lens, dtype=torch.long, device=self.device)

        def _store_new_kv_cache(self, tags, new_kv_cache, new_kv_cache_len):
            new_kv_cache_len = new_kv_cache_len.detach().to("cpu")
            for b, tag in enumerate(tags):
                self.kv_cache_len_by_tag[tag] = int(new_kv_cache_len[b].item())
                self.kv_cache_by_tag[tag] = [layer_cache[b:b + 1].detach().clone() for layer_cache in new_kv_cache]

    a = build_actor(cfg_i, ListCacheActor)
    a._lists_init()
    return a


@contextlib.contextmanager
def dense_block_masks(on: bool = True):
    """Inside the block ppo_core builds its masks with torch's create_block_mask, as it did before."""
    import ppo_core
    saved = ppo_core._FAST_BLOCK_MASKS
    ppo_core._FAST_BLOCK_MASKS = (not on) and saved
    try:
        yield
    finally:
        ppo_core._FAST_BLOCK_MASKS = saved


# --------------------------------------------------------------------------- sections

def section_infer(cfg, obs_np, dev: str, batch_sizes: List[int], iters: int) -> Dict[str, Any]:
    import numpy as np
    import torch
    import ppo_core
    from inference import InferenceActor

    cfg_i = dataclasses.replace(cfg, learner=dataclasses.replace(cfg.learner, resume=False),
                                infer=dataclasses.replace(cfg.infer, device=dev))
    t0 = time.perf_counter()
    actor = build_actor(cfg_i)
    build_s = time.perf_counter() - t0
    out: Dict[str, Any] = {"actor_build_seconds": round(build_s, 2), "by_batch": {}}
    for B in batch_sizes:
        if B > obs_np.shape[0]:
            continue
        obs = np.ascontiguousarray(obs_np[:B], dtype=np.float32)       # what worker.py sends
        tags = [f"b{B}_{i}" for i in range(B)]
        tc = time.perf_counter()
        for _ in range(3):                                            # compile / warm
            actor.infer_batch(tags, obs)
        warm_s = time.perf_counter() - tc
        ts = []
        for _ in range(iters):
            t1 = time.perf_counter()
            actor.infer_batch(tags, obs)
            ts.append((time.perf_counter() - t1) * 1000.0)
        # Breakdown: the same calls with the GPU synchronised around each part.
        ck = Clock(dev)
        saved = (actor._build_batched_kv_cache, actor._store_new_kv_cache, actor.net.forward,
                 actor.net.encode_features, actor.net.readout_policy, ppo_core.inference_block_mask)
        actor._build_batched_kv_cache = ck.wrap(saved[0], "kv gather (one index into the pool)")
        actor._store_new_kv_cache = ck.wrap(saved[1], "kv store (one indexed write)")
        actor.net.forward = ck.wrap(saved[2], "net forward")
        actor.net.encode_features = ck.wrap(saved[3], "  of which encoder")
        actor.net.readout_policy = ck.wrap(saved[4], "  of which readout")
        ppo_core.inference_block_mask = ck.wrap(saved[5], "  of which create_block_mask")
        tb = []
        for _ in range(iters):
            ck.sync()
            t1 = time.perf_counter()
            actor.infer_batch(tags, obs)
            ck.sync()
            tb.append((time.perf_counter() - t1) * 1000.0)
        (actor._build_batched_kv_cache, actor._store_new_kv_cache) = saved[0], saved[1]
        del actor.net.forward, actor.net.encode_features, actor.net.readout_policy
        ppo_core.inference_block_mask = saved[5]
        parts = ck.ms(iters)
        layers = parts.get("net forward", 0) - parts.get("  of which encoder", 0) - parts.get("  of which readout", 0) - parts.get("  of which create_block_mask", 0)
        parts["  of which 3 layers (128 tokens each, compiled flex_attention)"] = round(layers, 3)
        parts["everything else in infer_batch (to GPU, softmax, sample, to CPU)"] = round(
            med(tb) - parts.get("net forward", 0) - parts.get("kv gather (one index into the pool)", 0) - parts.get("kv store (one indexed write)", 0), 3)
        out["by_batch"][B] = {
            "ms_per_call": round(med(ts), 3), "ms_per_call_min": round(min(ts), 3),
            "ms_per_decision": round(med(ts) / B, 4), "decisions_per_second_if_back_to_back": round(B * 1000.0 / med(ts), 1),
            "first_three_calls_seconds": round(warm_s, 2), "ms_per_call_instrumented": round(med(tb), 3), "parts_ms": parts,
        }
        for t in tags:
            actor.clear_cache(t)
    if dev.startswith("cuda"):
        out["max_memory_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    return out


def minibatch_step(net, opt, sched, batch, kw: Dict[str, Any], dev: str, ck: Optional[Clock], T: int,
                   autocast_bf16: bool = False, forward_fn: Optional[Callable] = None) -> float:
    """One optimizer step on one minibatch: the 'ppo' branch of ppo_core.ppo_update, line for line."""
    import torch
    import ppo_core
    from ppo_core import MasterWeightOptimizer, masked_logprob_entropy, twohot_targets, dist_value_loss, _clip_and_step

    span = ck.span if ck is not None else (lambda _l: contextlib.nullcontext())
    obs, act, logp_old, adv, ret, ep = batch
    with span("to GPU"):
        act, logp_old, adv, ret, ep = (t.to(dev) for t in (act, logp_old, adv, ret, ep))
        obs = obs.to(dev, non_blocking=True)
        m0, m1 = net.unpacker.offsets["action_mask"]
        mask = obs[:, m0:m1]
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if (autocast_bf16 and dev.startswith("cuda")) else contextlib.nullcontext()
    with span("forward (incl. block mask)"), ctx:
        if forward_fn is None:
            logits, v_logits, _, _, _ = net(obs, ep, batch_seq_len=T)
        else:
            logits, v_logits, _ = forward_fn(net, obs, ep, T)
    with span("loss"):
        logp, ent = masked_logprob_entropy(logits.float(), mask, act)
        ratio = (logp - logp_old).exp()
        pg = torch.max(-adv * ratio, -adv * ratio.clamp(1 - kw["clip_coef"], 1 + kw["clip_coef"])).mean()
        v_loss = dist_value_loss(v_logits.float(), twohot_targets(ret, v_min=kw["v_min"], v_max=kw["v_max"], v_bins=kw["v_bins"]))
        loss = pg + kw["vf_coef"] * v_loss - kw["ent_coef"] * ent.mean()
    with span("backward"):
        loss.backward()
    with span("optimizer (fp32 masters, clip, step, copy back)"):
        if isinstance(opt, MasterWeightOptimizer):
            opt.accumulate_grads()
        _clip_and_step(net, opt, kw["max_grad_norm"])
        opt.zero_grad(set_to_none=True)
        if sched is not None:
            sched.step()
    with span("stats (.item x6)"):
        val = loss.item() + pg.item() + v_loss.item() + ent.mean().item() + ratio.mean().item() + logp.mean().item()
    return float(val)


def time_minibatches(net, opt, sched, obs_np, ep_np, mb: int, T: int, iters: int, dev: str, kw, mask_slice,
                     autocast_bf16: bool = False, forward_fn: Optional[Callable] = None, wrap_mask: bool = True) -> Dict[str, Any]:
    import torch
    import ppo_core
    batch = synth_batch(obs_np, ep_np, mb, mask_slice)
    if dev.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    minibatch_step(net, opt, sched, batch, kw, dev, None, T, autocast_bf16, forward_fn)       # compiles
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    first = time.perf_counter() - t0
    plain = []
    for _ in range(iters):
        t1 = time.perf_counter()
        minibatch_step(net, opt, sched, batch, kw, dev, None, T, autocast_bf16, forward_fn)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        plain.append((time.perf_counter() - t1) * 1000.0)
    ck = Clock(dev)
    # Whichever builder is in use: the one from episode ids, or torch's when that is switched off.
    orig = (ppo_core.episodic_block_mask, ppo_core.create_block_mask)
    if wrap_mask and forward_fn is None:
        ppo_core.episodic_block_mask = ck.wrap(orig[0], "  of which create_block_mask")
        ppo_core.create_block_mask = ck.wrap(orig[1], "  of which create_block_mask")
    try:
        for _ in range(iters):
            minibatch_step(net, opt, sched, batch, kw, dev, ck, T, autocast_bf16, forward_fn)
    finally:
        ppo_core.episodic_block_mask, ppo_core.create_block_mask = orig
    res = {
        "minibatch_steps": mb, "seq_len": T, "tokens": mb * 64,
        "ms_per_minibatch": round(med(plain), 2), "ms_per_minibatch_min": round(min(plain), 2),
        "ms_per_step_one_epoch": round(med(plain) / mb, 4),
        "first_call_seconds_incl_compile": round(first, 2),
        "parts_ms": ck.ms(iters),
    }
    if dev.startswith("cuda"):
        res["max_memory_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    return res


def section_learn(cfg, obs_np, ep_np, dev: str, mb: int, T: int, iters: int, kw, mask_slice) -> Dict[str, Any]:
    import torch
    import ppo_core
    ensure_compiled()
    net = make_net(cfg, dev, "bf16").train()
    opt, sched = make_optimizer(cfg, net)
    res = time_minibatches(net, opt, sched, obs_np, ep_np, mb, T, iters, dev, kw, mask_slice)

    # Cross-check with the real ppo_update on the same data (one epoch, `iters` minibatches).
    from ppo_core import AsyncEpisodeDataset, ppo_update
    n = mb * iters
    if n <= obs_np.shape[0]:
        obs, act, logp, adv, ret, ep = synth_batch(obs_np, ep_np, n, mask_slice)
        ds = AsyncEpisodeDataset(cfg.env.act_dim, "cpu")
        ds.add_steps(obs, act, logp, torch.zeros(n), adv, ret, torch.zeros(n), ep)
        pk = dict(cfg.learner.ppo_kwargs())
        pk.update(update_epochs=1, minibatch_size=mb, batch_seq_len=T, grad_accum_steps=1)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        st = ppo_update(net=net, opt=opt, dataset=ds, scheduler=sched, mode="ppo", target_net=None, **pk,
                        v_min=cfg.learner.v_min, v_max=cfg.learner.v_max, v_bins=cfg.learner.v_bins)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        res["ppo_update_ms_per_minibatch"] = round((time.perf_counter() - t0) * 1000.0 / max(st.n_mb, 1), 2)
        res["ppo_update_minibatches"] = st.n_mb
    del net, opt
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    return res


def section_precision(cfg, obs_np, ep_np, dev: str, mb: int, T: int, iters: int, kw, mask_slice, infer_B: int) -> Dict[str, Any]:
    import numpy as np
    import torch
    import ppo_core
    ensure_compiled()
    out: Dict[str, Any] = {"learner_minibatch": {}, "inference_forward": {}}
    variants = (("bf16 weights (as coded)", "bf16", False), ("fp32 weights", "fp32", False), ("fp32 weights + bf16 autocast", "fp32", True))
    for name, prec, ac in variants:
        try:
            net = make_net(cfg, dev, prec).train()
            opt, sched = make_optimizer(cfg, net)
            out["learner_minibatch"][name] = time_minibatches(net, opt, sched, obs_np, ep_np, mb, T, iters, dev, kw, mask_slice, autocast_bf16=ac)
            del opt
            net.eval()
            B = min(infer_B, obs_np.shape[0])
            obs = torch.from_numpy(np.ascontiguousarray(obs_np[:B], dtype=np.float32)).to(dev)
            ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if (ac and dev.startswith("cuda")) else contextlib.nullcontext
            ts = []
            with torch.no_grad():
                for i in range(iters + 2):
                    if dev.startswith("cuda"):
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    with ctx():
                        net(obs)
                    if dev.startswith("cuda"):
                        torch.cuda.synchronize()
                    if i >= 2:
                        ts.append((time.perf_counter() - t1) * 1000.0)
            out["inference_forward"][name] = {"batch": B, "ms_per_forward": round(med(ts), 3)}
            del net
        except Exception as e:  # noqa: BLE001  (out of memory on a small GPU is a result too)
            out["learner_minibatch"].setdefault(name, {"error": repr(e)[:300]})
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
    return out


def proto_check(cfg, obs_np, ep_np, dev: str, mask_slice) -> Dict[str, Any]:
    """The prototype against the model's own forward, in float32, values and gradients."""
    import numpy as np
    import torch
    res: Dict[str, Any] = {}
    torch.manual_seed(0)
    net = make_net(cfg, dev, "fp32").eval()
    # Training path. Short sequences so that several episodes and both window edges are covered.
    # K_over shrinks the history window (both implementations read net.kv_cache_len), so the
    # "at most K earlier turns" edge is exercised without a sequence longer than 64 turns: on a
    # CPU the model's own eager attention materialises every token pair, which needs gigabytes
    # beyond a few thousand tokens.
    k_default = net.kv_cache_len
    cases = [(32, 4, None), (32, 4, 8)]
    if dev.startswith("cuda"):
        cases.append((128, 2, None))
    for T, n_seq, K_over in cases:
        n = T * n_seq
        if n > obs_np.shape[0]:
            continue
        net.kv_cache_len = K_over or k_default
        obs = torch.from_numpy(np.ascontiguousarray(obs_np[:n])).to(dev)
        ep = torch.from_numpy(np.ascontiguousarray(ep_np[:n])).long().to(dev)
        ga: Dict[str, Any] = {}
        gmax, gref = None, None
        if dev.startswith("cuda"):
            net.zero_grad(set_to_none=True)
            a_logits, a_v, _, _, _ = net(obs, ep, batch_seq_len=T)
            (a_logits.square().mean() + a_v.square().mean()).backward()
            ga = {n_: p.grad.detach().clone() for n_, p in net.named_parameters() if p.grad is not None}
            net.zero_grad(set_to_none=True)
            b_logits, b_v, _ = dense_train_forward(net, obs, ep, T)
            (b_logits.square().mean() + b_v.square().mean()).backward()
            gmax, gref = 0.0, 0.0
            for n_, p in net.named_parameters():
                if p.grad is None or n_ not in ga:
                    continue
                gmax = max(gmax, float((p.grad - ga[n_]).abs().max()))
                gref = max(gref, float(ga[n_].abs().max()))
            a_logits, a_v, b_logits, b_v = a_logits.detach(), a_v.detach(), b_logits.detach(), b_v.detach()
        else:
            # flex_attention has no backward on a CPU, so only the values can be compared there.
            with torch.no_grad():
                a_logits, a_v, _, _, _ = net(obs, ep, batch_seq_len=T)
                b_logits, b_v, _ = dense_train_forward(net, obs, ep, T)
        with torch.no_grad():
            c_logits, _, _ = dense_train_forward(net, obs, ep, T, _no_history=True)
        res[f"train T={T} n_seq={n_seq} K={net.kv_cache_len}"] = {
            "episodes_in_batch": int(torch.unique(ep).numel()),
            "negative_control_logits_diff_without_history": float((a_logits - c_logits).abs().max()),
            "logits_max_abs_diff": float((a_logits - b_logits).abs().max()),
            "logits_max_abs": float(a_logits.abs().max()),
            "value_logits_max_abs_diff": float((a_v - b_v).abs().max()),
            "grad_max_abs_diff": gmax, "grad_max_abs": gref,
            "params_with_grad_compared": len(ga),
        }
    # Inference path: several battles stepped through consecutive turns with both implementations.
    # With K=4 the cache fills and rolls over inside the seven turns.
    B, steps = 6, 7
    starts = [int(np.flatnonzero(ep_np == e)[0]) for e in np.unique(ep_np)[:B]]
    ok = all(s + steps <= obs_np.shape[0] for s in starts)
    for K_over in (None, 4):
        if not ok:
            break
        net.kv_cache_len = K_over or k_default
        K, L, D = net.kv_cache_len, net.n_layers, net.d_model
        kv, kv_len = None, None
        hist = torch.zeros(B, L, K, D, device=dev)
        hlen = torch.zeros(B, dtype=torch.long, device=dev)
        worst, worst_v, biggest = 0.0, 0.0, 0.0
        with torch.no_grad():
            for s in range(steps):
                rows = [st + s for st in starts]
                obs = torch.from_numpy(np.ascontiguousarray(obs_np[rows], dtype=np.float32)).to(dev)
                a_logits, a_v, _, kv, kv_len = net(obs, kv_cache=kv, kv_cache_len=kv_len)
                b_logits, b_v, _, hist, hlen = dense_infer_forward(net, obs, hist, hlen)
                worst = max(worst, float((a_logits - b_logits).abs().max()))
                worst_v = max(worst_v, float((a_v - b_v).abs().max()))
                biggest = max(biggest, float(a_logits.abs().max()))
            cache_diff = max(float((kv[i] - hist[:, i]).abs().max()) for i in range(L))
        res[f"inference {steps} consecutive turns x {B} battles K={K}"] = {
            "logits_max_abs_diff": worst, "logits_max_abs": biggest, "value_logits_max_abs_diff": worst_v,
            "cache_max_abs_diff_after_last_turn": cache_diff, "cache_len_equal": bool((kv_len == hlen).all()),
        }
    net.kv_cache_len = k_default
    del net
    return res


def section_proto(cfg, obs_np, ep_np, dev: str, mb: int, T: int, iters: int, kw, mask_slice,
                  batch_sizes: List[int], check_only: bool, big_mb: int) -> Dict[str, Any]:
    import numpy as np
    import torch
    import ppo_core
    # On the GPU the model's own path must run compiled, as the actors run it: the eager
    # fallback of flex_attention materialises every token pair in its backward pass and
    # runs an 8 GB card out of memory at a few thousand tokens.
    if dev.startswith("cuda"):
        ensure_compiled()
    out: Dict[str, Any] = {"check_fp32": proto_check(cfg, obs_np, ep_np, dev, mask_slice)}
    if check_only:
        return out
    # bf16 agreement with the compiled path on a full-size sequence (rounding level, not exact).
    net = make_net(cfg, dev, "bf16").train()
    n = min(mb, obs_np.shape[0])
    obs = torch.from_numpy(np.ascontiguousarray(obs_np[:n])).to(dev)
    ep = torch.from_numpy(np.ascontiguousarray(ep_np[:n])).long().to(dev)
    with torch.no_grad():
        a, av, _, _, _ = net(obs, ep, batch_seq_len=T)
        b, bv, _ = dense_train_forward(net, obs, ep, T)
        m0, m1 = mask_slice.start, mask_slice.stop
        legal = obs[:, m0:m1] > 0.5
        pa = torch.softmax(a.float().masked_fill(~legal, -1e4), -1)
        pb = torch.softmax(b.float().masked_fill(~legal, -1e4), -1)
    out["bf16_agreement_full_sequence"] = {
        "steps": n, "logits_max_abs_diff": float((a - b).abs().max()), "logits_mean_abs_diff": float((a - b).abs().mean()),
        "action_prob_max_abs_diff": float((pa - pb).abs().max()), "same_argmax_share": float((pa.argmax(-1) == pb.argmax(-1)).float().mean()),
    }
    with torch.no_grad():
        c, _, _ = dense_train_forward(net, obs, ep, T, attn_fp32=False)
        pc = torch.softmax(c.float().masked_fill(~legal, -1e4), -1)
        a2, _, _, _, _ = net(obs, ep, batch_seq_len=T)
    out["bf16_agreement_full_sequence_bf16_scores"] = {
        "logits_max_abs_diff": float((a - c).abs().max()), "logits_mean_abs_diff": float((a - c).abs().mean()),
        "action_prob_max_abs_diff": float((pa - pc).abs().max()), "same_argmax_share": float((pa.argmax(-1) == pc.argmax(-1)).float().mean()),
    }
    out["compiled_path_run_to_run_logits_max_abs_diff"] = float((a - a2).abs().max())
    del c, pc, a2
    del obs, ep, a, b, av, bv
    opt, sched = make_optimizer(cfg, net)
    out["learner_minibatch_dense_attention"] = time_minibatches(net, opt, sched, obs_np, ep_np, mb, T, iters, dev, kw, mask_slice, forward_fn=dense_train_forward)
    out["learner_minibatch_dense_attention_bf16_scores"] = time_minibatches(
        net, opt, sched, obs_np, ep_np, mb, T, iters, dev, kw, mask_slice,
        forward_fn=lambda n_, o, e, t: dense_train_forward(n_, o, e, t, attn_fp32=False))
    if big_mb and big_mb <= obs_np.shape[0]:
        try:
            out[f"learner_minibatch_dense_attention_mb{big_mb}"] = time_minibatches(
                net, opt, sched, obs_np, ep_np, big_mb, T, iters, dev, kw, mask_slice, forward_fn=dense_train_forward)
        except Exception as e:  # noqa: BLE001
            out[f"learner_minibatch_dense_attention_mb{big_mb}"] = {"error": repr(e)[:300]}
    del opt
    net.eval()
    K, L, D = net.kv_cache_len, net.n_layers, net.d_model
    inf: Dict[str, Any] = {}
    with torch.no_grad():
        for B in batch_sizes:
            if B > obs_np.shape[0]:
                continue
            obs_cpu = np.ascontiguousarray(obs_np[:B], dtype=np.float32)
            pool = torch.zeros(B, L, K, D, device=dev, dtype=net.recurrent_dtype())
            lens = torch.zeros(B, dtype=torch.long, device=dev)
            idx = torch.arange(B, device=dev)
            ts = []
            for i in range(iters + 3):
                if dev.startswith("cuda"):
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                o = torch.from_numpy(obs_cpu).to(dev, non_blocking=True)
                logits, v_logits, v_exp, new_hist, new_len = dense_infer_forward(net, o, pool[idx], lens[idx])
                pool[idx] = new_hist
                lens[idx] = new_len
                v_probs = torch.softmax(v_logits, dim=-1)
                v_var = (v_probs * (net.v_support - v_exp.unsqueeze(-1)) ** 2).sum(dim=-1)
                m = o[:, mask_slice].float()
                act, logp, _ = ppo_core.masked_sample(logits, m, temp=1.0)
                res = (act.cpu().numpy(), logp.cpu().numpy(), v_exp.cpu().numpy(), v_var.cpu().numpy())
                if i >= 3:
                    ts.append((time.perf_counter() - t1) * 1000.0)
            inf[B] = {"ms_per_call": round(med(ts), 3), "ms_per_decision": round(med(ts) / B, 4),
                      "decisions_per_second_if_back_to_back": round(B * 1000.0 / med(ts), 1)}
            del pool
    out["inference_call_dense_attention_pooled_cache"] = inf
    del net
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def section_kvpool(cfg, dev: str, batch_sizes: List[int], iters: int) -> Dict[str, Any]:
    import torch
    from inference import InferenceActor
    cfg_i = dataclasses.replace(cfg, learner=dataclasses.replace(cfg.learner, resume=False),
                                infer=dataclasses.replace(cfg.infer, device=dev))
    actor = build_actor(cfg_i)                # compiles nothing until a forward is run
    lists = make_list_cache_actor(cfg_i)      # the same actor with the old per-battle lists
    net = actor.net
    K, L, D = net.kv_cache_len, net.n_layers, net.d_model
    dt = net.recurrent_dtype()
    out: Dict[str, Any] = {}
    ck = Clock(dev)
    for B in batch_sizes:
        tags = [f"k{B}_{i}" for i in range(B)]
        new = [torch.randn(B, K, D, device=dev, dtype=dt) for _ in range(L)]
        new_len = torch.full((B,), 5, dtype=torch.long, device=dev)
        row = {}
        for name, a in (("lists", lists), ("pool", actor)):
            a._build_batched_kv_cache(tags)
            a._store_new_kv_cache(tags, new, new_len)
            gather, store = [], []
            for _ in range(iters):
                ck.sync()
                t = time.perf_counter()
                a._build_batched_kv_cache(tags)
                ck.sync()
                gather.append((time.perf_counter() - t) * 1000.0)
                ck.sync()
                t = time.perf_counter()
                a._store_new_kv_cache(tags, new, new_len)
                ck.sync()
                store.append((time.perf_counter() - t) * 1000.0)
            row[f"{name}_gather_ms"] = round(med(gather), 3)
            row[f"{name}_store_ms"] = round(med(store), 3)
            for t in tags:
                a.clear_cache(t)
        out[B] = row
    return out


def section_fastmask(cfg, obs_np, ep_np, dev: str, mb: int, T: int, iters: int, kw, mask_slice, batch_sizes: List[int]) -> Dict[str, Any]:
    """The block mask from episode ids against torch's own, and then the old per-battle lists too."""
    import numpy as np
    import torch
    import ppo_core
    from torch.nn.attention.flex_attention import create_block_mask as torch_cbm
    ensure_compiled()
    out: Dict[str, Any] = {}

    # --- 1. the block masks are the same tensors, and so are the outputs
    captured: List[Any] = []
    real_train, real_infer = ppo_core.episodic_block_mask, ppo_core.inference_block_mask

    def spy_train(turn_ids, K, rule):
        got = real_train(turn_ids, K, rule)
        n_seq, t = turn_ids.shape
        captured.append(block_masks_equal(torch_cbm(rule, n_seq, 1, t * TOKENS, t * TOKENS, device=turn_ids.device), got))
        return got

    def spy_infer(batch, seq_len, rule, device):
        got = real_infer(batch, seq_len, rule, device)
        captured.append(block_masks_equal(torch_cbm(rule, batch, 1, seq_len, seq_len, device=device), got))
        return got

    net = make_net(cfg, dev, "bf16").eval()
    checks = []
    sizes = [(T, max(mb // T, 1)), (64, 4), (30, 3)]
    try:
        ppo_core.episodic_block_mask, ppo_core.inference_block_mask = spy_train, spy_infer
        with torch.no_grad():
            for off, (TT, ns) in zip((0, 700, 1900), sizes):
                n = TT * ns
                if off + n > obs_np.shape[0]:
                    continue
                obs = torch.from_numpy(np.ascontiguousarray(obs_np[off:off + n])).to(dev)
                ep = torch.from_numpy(np.ascontiguousarray(ep_np[off:off + n])).long().to(dev)
                with dense_block_masks():
                    a_logits, a_v, _, _, _ = net(obs, ep, batch_seq_len=TT)
                captured.clear()
                b_logits, b_v, _, _, _ = net(obs, ep, batch_seq_len=TT)
                eq = dict(captured[0])
                checks.append({"seq_len": TT, "sequences": ns, "episodes": int(torch.unique(ep).numel()),
                               "block_mask_fields_equal": all(eq.values()),
                               "fields_not_equal": [k for k, v in eq.items() if not v],
                               "logits_bitwise_equal": bool(torch.equal(a_logits, b_logits)),
                               "value_logits_bitwise_equal": bool(torch.equal(a_v, b_v)),
                               "logits_max_abs_diff": float((a_logits - b_logits).abs().max())})
            # inference
            B = 16
            obs = torch.from_numpy(np.ascontiguousarray(obs_np[:B], dtype=np.float32)).to(dev)
            kv, kvl = None, None
            eq_all, same = True, True
            for _step in range(4):
                with dense_block_masks():
                    a = net(obs, kv_cache=kv, kv_cache_len=kvl)
                captured.clear()
                b = net(obs, kv_cache=kv, kv_cache_len=kvl)
                eq_all = eq_all and all(captured[0].values())
                same = same and bool(torch.equal(a[0], b[0])) and bool(torch.equal(a[1], b[1]))
                kv, kvl = a[3], a[4]
            checks.append({"inference_batch": B, "turns": 4, "block_mask_fields_equal": eq_all, "logits_and_value_bitwise_equal": same})
    finally:
        ppo_core.episodic_block_mask, ppo_core.inference_block_mask = real_train, real_infer
    out["checks"] = checks
    del net
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()

    # --- 2. the learner minibatch, mask from episode ids (as coded now) and torch's dense mask (as before)
    for name, dense in (("learner_minibatch_cheap_mask", False), ("learner_minibatch_dense_mask", True)):
        try:
            with dense_block_masks(dense):
                net = make_net(cfg, dev, "bf16").train()
                opt, sched = make_optimizer(cfg, net)
                out[name] = time_minibatches(net, opt, sched, obs_np, ep_np, mb, T, iters, dev, kw, mask_slice, wrap_mask=False)
                del net, opt
        except Exception as e:  # noqa: BLE001  (the dense mask may not fit a small GPU: that is a result too)
            out[name] = {"error": repr(e)[:300]}
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()

    # --- 3. the inference call: as it was (dense mask, per-battle lists), with the cheap mask
    #        alone, and as it is now (cheap mask, pooled cache)
    cfg_i = dataclasses.replace(cfg, learner=dataclasses.replace(cfg.learner, resume=False),
                                infer=dataclasses.replace(cfg.infer, device=dev))
    res_inf: Dict[str, Any] = {}
    actor = build_actor(cfg_i)
    lists = make_list_cache_actor(cfg_i)
    for B in batch_sizes:
        if B > obs_np.shape[0]:
            continue
        obs = np.ascontiguousarray(obs_np[:B], dtype=np.float32)
        row = {}
        for name, a, dense in (("dense_mask_and_lists", lists, True), ("cheap_mask", lists, False), ("cheap_mask_and_pooled_cache", actor, False)):
            tags = [f"f{B}_{i}" for i in range(B)]
            with dense_block_masks(dense):
                for _ in range(3):
                    a.infer_batch(tags, obs)
                ts = []
                for _ in range(iters):
                    t1 = time.perf_counter()
                    a.infer_batch(tags, obs)
                    ts.append((time.perf_counter() - t1) * 1000.0)
            row[name] = {"ms_per_call": round(med(ts), 3), "ms_per_decision": round(med(ts) / B, 4)}
            for t in tags:
                a.clear_cache(t)
        res_inf[B] = row
    out["inference_call"] = res_inf
    return out


def learner_loop_child(obs_path: str, mb: int, T: int, alone_s: float, shared_s: float) -> int:
    """
    The other half of the `share` section, run as a second process the way the learner actor is
    a second process in training: minibatch after minibatch for alone_s + shared_s seconds.
    Prints READY when warm, GO when the shared part begins, then one line of JSON.
    """
    os.chdir(REPO_ROOT)
    sys.path.insert(0, REPO_ROOT)
    import torch
    from config import RunConfig
    from obs_assembler import ObservationAssembler

    dev = "cuda"
    # The coded mask builder needs several GB for moments (see the `learn` section), so the cap
    # is generous: the laptop runs showed 6.4 GB used with both actors up.
    torch.cuda.set_per_process_memory_fraction(0.78)
    cfg = RunConfig.default()
    cfg = dataclasses.replace(cfg, learner=dataclasses.replace(cfg.learner, mode="ppo", device=dev))
    obs_np, ep_np = load_obs(obs_path)
    m0, m1 = ObservationAssembler().offsets["action_mask"]
    kw = dict(cfg.learner.ppo_kwargs())
    kw.update(v_min=cfg.learner.v_min, v_max=cfg.learner.v_max, v_bins=cfg.learner.v_bins)
    ensure_compiled()
    net = make_net(cfg, dev, "bf16").train()
    opt, sched = make_optimizer(cfg, net)
    batch = synth_batch(obs_np, ep_np, mb, slice(m0, m1))
    for _ in range(2):
        minibatch_step(net, opt, sched, batch, kw, dev, None, T)
    torch.cuda.synchronize()
    print("READY", flush=True)

    def run_for(seconds: float) -> Tuple[float, int]:
        n = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            minibatch_step(net, opt, sched, batch, kw, dev, None, T)
            torch.cuda.synchronize()
            n += 1
        return (time.perf_counter() - t0) * 1000.0 / max(n, 1), n

    alone, n1 = run_for(alone_s)
    print("GO", flush=True)
    shared, n2 = run_for(shared_s)
    print(json.dumps({"ms_per_minibatch_alone": round(alone, 2), "minibatches_alone": n1,
                      "ms_per_minibatch_shared": round(shared, 2), "minibatches_shared": n2}), flush=True)
    return 0


def section_share(cfg, obs_path: str, obs_np, dev: str, mb: int, T: int, B: int = 32,
                  alone_s: float = 3.0, paced_s: float = 4.0, flat_s: float = 3.0, pace_ms: float = 75.0) -> Dict[str, Any]:
    """
    What a running learner does to the inference call, measured with two processes on the one
    GPU as in training: this process is the inference actor, a child process loops learner
    minibatches. About ten seconds of GPU time with both busy.

    The inference calls are first paced at one per 75 ms (the laptop run made 13.3 a second),
    then issued back to back.
    """
    import numpy as np
    if not dev.startswith("cuda"):
        return {"error": "needs a GPU"}
    cfg_i = dataclasses.replace(cfg, learner=dataclasses.replace(cfg.learner, resume=False),
                                infer=dataclasses.replace(cfg.infer, device=dev))
    actor = build_actor(cfg_i)
    obs = np.ascontiguousarray(obs_np[:B], dtype=np.float32)
    tags = [f"s{i}" for i in range(B)]
    for _ in range(5):
        actor.infer_batch(tags, obs)

    def calls(seconds: float, pace: float) -> List[float]:
        out: List[float] = []
        t_end = time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            t1 = time.perf_counter()
            actor.infer_batch(tags, obs)
            dt = (time.perf_counter() - t1) * 1000.0
            out.append(dt)
            if pace > dt:
                time.sleep((pace - dt) / 1000.0)
        return out

    def summ(xs: List[float]) -> Dict[str, Any]:
        xs = sorted(xs)
        if not xs:
            return {"calls": 0}
        return {"calls": len(xs), "ms_median": round(xs[len(xs) // 2], 2), "ms_p90": round(xs[int(0.9 * (len(xs) - 1))], 2),
                "ms_max": round(xs[-1], 2)}

    before = calls(2.0, pace_ms)
    child = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--_learner-loop", obs_path, str(mb), str(T),
                              str(alone_s), str(paced_s + flat_s + 1.0)],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    res: Dict[str, Any] = {"inference_batch": B, "learner_minibatch": mb, "pace_ms": pace_ms}
    try:
        got = {"READY": False, "GO": False}
        deadline = time.time() + 300.0
        while time.time() < deadline and not got["GO"]:
            line = child.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line in got:
                got[line] = True
        if not got["GO"]:
            res["error"] = "the learner process did not get going"
            return res
        paced = calls(paced_s, pace_ms)
        flat = calls(flat_s, 0.0)
        tail = child.stdout.read()
        child.wait(timeout=120)
        for ln in reversed(tail.strip().splitlines()):
            if ln.startswith("{"):
                res["learner"] = json.loads(ln)
                break
    finally:
        if child.poll() is None:
            child.kill()          # by the pid this function started
            child.wait(timeout=30)
    after = calls(2.0, pace_ms)
    res["inference_alone_before"] = summ(before)
    res["inference_while_learner_runs_paced"] = summ(paced)
    res["inference_while_learner_runs_back_to_back"] = summ(flat)
    res["inference_alone_after"] = summ(after)
    if "learner" in res and res["learner"].get("ms_per_minibatch_alone"):
        res["learner_slowdown"] = round(res["learner"]["ms_per_minibatch_shared"] / res["learner"]["ms_per_minibatch_alone"], 2)
    a = res["inference_alone_before"].get("ms_median")
    p = res["inference_while_learner_runs_paced"].get("ms_median")
    if a and p:
        res["inference_slowdown_paced"] = round(p / a, 2)
    return res


# --------------------------------------------------------------------------- main

def main() -> int:
    if len(sys.argv) >= 7 and sys.argv[1] == "--_learner-loop":
        return learner_loop_child(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs", required=True, help="obs_samples.npz from bench/micro_obs.py")
    ap.add_argument("--device", default=None)
    ap.add_argument("--sections", default="infer,learn,fastmask,precision,proto,kvpool")
    ap.add_argument("--batch-sizes", default="1,8,16,32,64,128,256")
    ap.add_argument("--mb", type=int, default=256, help="learner minibatch (steps)")
    ap.add_argument("--seq", type=int, default=256, help="learner sequence length")
    ap.add_argument("--precision-mb", type=int, default=128, help="reduced minibatch and sequence for the precision comparison")
    ap.add_argument("--big-mb", type=int, default=768, help="also try the prototype at this minibatch (0 = skip)")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--check-only", action="store_true", help="proto: only the equivalence check")
    ap.add_argument("--mem-fraction", type=float, default=0.85,
                    help="cap on this process's share of GPU memory, so a mistake fails fast instead of spilling into system RAM")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    args.obs = os.path.abspath(args.obs)
    if args.out:
        args.out = os.path.abspath(args.out)
        os.makedirs(args.out, exist_ok=True)
    os.chdir(REPO_ROOT)
    sys.path.insert(0, REPO_ROOT)

    import torch
    from config import RunConfig
    from obs_assembler import ObservationAssembler

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dev.startswith("cuda") and 0.0 < args.mem_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    # This script pushes one compiled function through many dtypes, batch sizes and masks in a
    # single process. Past torch's recompile limit (8) torch.compile gives up on the function
    # and runs it EAGERLY from then on, and eager flex_attention materialises every token pair:
    # the learner path then needs tens of gigabytes. The training actors compile far fewer
    # variants, but the same silent fallback would hit them if they ever passed the limit.
    try:
        import torch._dynamo
        torch._dynamo.config.recompile_limit = 64
    except Exception:  # noqa: BLE001
        pass
    cfg = RunConfig.default()
    cfg = dataclasses.replace(cfg, learner=dataclasses.replace(cfg.learner, mode="ppo", device=dev))
    obs_np, ep_np = load_obs(args.obs)
    m0, m1 = ObservationAssembler().offsets["action_mask"]
    mask_slice = slice(m0, m1)
    kw = dict(cfg.learner.ppo_kwargs())
    kw.update(v_min=cfg.learner.v_min, v_max=cfg.learner.v_max, v_bins=cfg.learner.v_bins)
    bs = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    sections = [s.strip() for s in args.sections.split(",") if s.strip()]

    res: Dict[str, Any] = {
        "device": dev, "torch": torch.__version__, "self_pid": os.getpid(),
        "gpu_before": smi() if dev.startswith("cuda") else None,
        "obs_rows": int(obs_np.shape[0]), "obs_dim": int(obs_np.shape[1]),
        "model": {"parameters": None, "d_model": None},
    }
    probe = cfg.make_model()
    res["model"] = {"parameters": sum(p.numel() for p in probe.parameters()), "d_model": probe.d_model,
                    "layers": probe.n_layers, "heads": probe.n_heads, "history_fields": probe.kv_cache_len,
                    "tokens_per_turn": 64}
    del probe

    def save() -> None:
        if args.out:
            with open(os.path.join(args.out, "gpu_bench.json"), "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)

    t_all = time.perf_counter()
    for s in sections:
        t0 = time.perf_counter()
        try:
            if s == "infer":
                res["infer"] = section_infer(cfg, obs_np, dev, bs, args.iters)
            elif s == "learn":
                res["learn"] = section_learn(cfg, obs_np, ep_np, dev, args.mb, args.seq, args.iters, kw, mask_slice)
            elif s == "precision":
                res["precision"] = section_precision(cfg, obs_np, ep_np, dev, args.precision_mb, args.precision_mb, args.iters, kw, mask_slice, 64)
            elif s == "proto":
                res["proto"] = section_proto(cfg, obs_np, ep_np, dev, args.mb, args.seq, args.iters, kw, mask_slice, bs, args.check_only, args.big_mb)
            elif s == "fastmask":
                res["fastmask"] = section_fastmask(cfg, obs_np, ep_np, dev, args.mb, args.seq, args.iters, kw, mask_slice, [b for b in bs if b >= 8])
            elif s == "kvpool":
                res["kvpool"] = section_kvpool(cfg, dev, [b for b in bs if b >= 8], args.iters)
            elif s == "share":
                res["share"] = section_share(cfg, args.obs, obs_np, dev, args.mb, args.seq)
            else:
                res[s] = {"error": "unknown section"}
        except Exception as e:  # noqa: BLE001
            import traceback
            res[s] = {"error": repr(e)[:500], "trace": traceback.format_exc()[-1500:]}
        res.setdefault("section_seconds", {})[s] = round(time.perf_counter() - t0, 1)
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
        save()
    res["total_seconds"] = round(time.perf_counter() - t_all, 1)
    res["gpu_after"] = smi() if dev.startswith("cuda") else None
    save()
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
