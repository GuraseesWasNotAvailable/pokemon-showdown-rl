#!/usr/bin/env python
"""
Is the league run's start exactly M24381? Recorded battles replayed through this line's training
worker (worker.SyncBridgePlayer, the assembler RunConfig.make_assembler builds from the checkpoint's
run_cfg["obs"], main_line included) and this line's network, against the window app's player on the
original M24381 (branch local-app's bench/v2_exactness.py app mode, whose output this reads).

    python bench/league_exactness.py ref --checkpoint WIDENED.pt --streams streams.jsonl --out league.npz [--bf16]
    python bench/league_exactness.py compare league.npz app.npz --prefix 8567 [--out report.json]

ref       this code: the players and their observations as the training workers build them; the network
          as the learner builds it (RunConfig.make_model with the checkpoint's blocks, its SetHead and
          opponent-action head attached when the checkpoint has them, every tensor loaded strictly), run
          one decision at a time with its history as players.BatchedPolicyPlayer._forward_batch calls it.
          fp32 (as the app plays) by default; --bf16 runs it on the inference actor's bf16 recurrent path
          (InferenceActor: enable_bf16_recurrent_path, float32 observations).
compare   decision by decision: the first --prefix numbers of the observations byte for byte (the whole
          observation without --prefix), the logits, value logits and expected value bit for bit.

The streams are the 300 recorded battles box c has in ~/dev/streams/streams.jsonl (the app support's test
set, 12,048 decisions). The fakes are the app harness's: messages come from the file, orders are dropped.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


class Recorder:
    """Per (player, battle tag): the observations and the network's rows, in decision order (v2_exactness's format)."""

    def __init__(self) -> None:
        self.obs: Dict[Tuple[str, str], List[Any]] = defaultdict(list)
        self.rows: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        self.order: List[Tuple[str, str]] = []
        self.notes: Dict[str, Any] = defaultdict(int)

    def add_obs(self, key: Tuple[str, str], obs: Any) -> None:
        if key not in self.obs:
            self.order.append(key)
        self.obs[key].append(obs.copy())

    def save(self, path: str, meta: Dict[str, Any]) -> None:
        import numpy as np

        keys = [k for k in self.order if self.obs[k]]
        obs = np.concatenate([np.stack(self.obs[k]) for k in keys]) if keys else np.zeros((0, 0), np.float16)
        out: Dict[str, Any] = {"obs": obs, "counts": np.array([len(self.obs[k]) for k in keys], dtype=np.int64)}
        if keys and all(len(self.rows[k]) == len(self.obs[k]) for k in keys) and self.rows[keys[0]]:
            for name in ("logits", "v_logits", "v_exp"):
                out[name] = np.concatenate([np.stack([r[name] for r in self.rows[k]]) for k in keys])
        meta = dict(meta, keys=[list(k) for k in keys], notes=dict(self.notes))
        out["meta"] = np.array(json.dumps(meta))
        np.savez_compressed(path, **out)


def obs_config_of(ckpt: Dict[str, Any], base: Any) -> Any:
    """ObsConfig with every field the checkpoint's run_cfg["obs"] names (the others at their defaults; own_tera_type absent is False)."""
    saved = dict((ckpt.get("run_cfg") or {}).get("obs") or {})
    known = {f.name for f in dataclasses.fields(base)}
    kw = {k: v for k, v in saved.items() if k in known}
    kw.setdefault("own_tera_type", False)
    return dataclasses.replace(base, **kw)


def run_ref(args: argparse.Namespace) -> int:
    os.chdir(REPO)
    import numpy as np
    import torch

    import config as config_mod
    from bench import micro_obs
    from poke_env.concurrency import POKE_LOOP

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base = config_mod.RunConfig.default()
    cfg = dataclasses.replace(base, obs=obs_config_of(ck, base.obs))

    real_default = config_mod.RunConfig.default
    config_mod.RunConfig.default = classmethod(lambda cls: cfg)
    try:
        _cfg, players, _q = micro_obs.build_players("ppo", "simple", 0, 7)
    finally:
        config_mod.RunConfig.default = real_default
    rec = Recorder()
    for p in players.values():
        inner = p.assembler.assemble

        def assemble(battle: Any, events: Any = None, _inner=inner, _p=p):
            got = _inner(battle, events=events)
            rec.add_obs((_p.username, battle.battle_tag), got)
            return got

        p.assembler.assemble = assemble
    assembler_name = type(players["p1"].assembler).__name__
    msgs = micro_obs.load_streams(args.streams, args.battles)
    t0 = time.time()
    asyncio.run_coroutine_threadsafe(micro_obs.replay(players, msgs, defaultdict(micro_obs.Acc)), POKE_LOOP).result()
    replay_s = time.time() - t0
    rec.notes["refused_choices"] = int(sum(getattr(p, "choices_refused", 0) for p in players.values()))

    net = cfg.make_model()
    model = ck["model"]
    if any(str(k).startswith("set_head.") for k in model):
        net.attach_set_head()
    if any(str(k).startswith("opp_action_head.") for k in model):
        net.attach_opp_action_head()
    state = {k: v for k, v in model.items() if "attn_mask" not in k}
    missing, unexpected = net.load_state_dict(state, strict=False)
    learned = {n for n, _p in net.named_parameters()}
    if [k for k in missing if k in learned] or unexpected:
        raise SystemExit(f"the checkpoint does not fit this network: missing {missing}, unexpected {unexpected}")
    rec.notes["buffers_not_in_checkpoint"] = len(missing)
    dev = torch.device(args.device)
    net = net.float().to(dev).eval()
    if args.bf16:
        net.enable_bf16_recurrent_path()
    t0 = time.time()
    with torch.no_grad():
        for key in rec.order:
            kv = [torch.zeros(1, net.kv_cache_len, net.d_model, device=dev, dtype=net.recurrent_dtype() if args.bf16 else torch.float32)
                  for _ in range(net.n_layers)]
            kv_len = torch.zeros(1, dtype=torch.long, device=dev)
            for obs in rec.obs[key]:
                x = torch.from_numpy(np.ascontiguousarray(obs[None], dtype=np.float32)).to(dev)
                kv_in = [torch.cat([layer], dim=0) for layer in kv]
                len_in = torch.cat([kv_len.reshape(1).to(dev)], dim=0)
                out, v_logits, v_exp, new_kv, new_len = net(x, kv_cache=kv_in, kv_cache_len=len_in)
                kv = [layer[0:1].clone() for layer in new_kv]
                kv_len = new_len[0:1].clone()
                rec.rows[key].append({"logits": out[0].float().cpu().numpy(), "v_logits": v_logits[0].float().cpu().numpy(),
                                      "v_exp": v_exp.reshape(-1)[0].float().cpu().numpy()})
    rec.save(args.out, {"mode": "league-ref" + ("-bf16" if args.bf16 else ""), "code": REPO,
                        "checkpoint": os.path.abspath(args.checkpoint), "assembler": assembler_name,
                        "obs_flags": dataclasses.asdict(cfg.obs),
                        "total_dim": int(rec.obs[rec.order[0]][0].shape[0]) if rec.order else 0,
                        "replay_s": round(replay_s, 1), "net_s": round(time.time() - t0, 1), "messages": len(msgs)})
    print(f"ref: {sum(len(v) for v in rec.obs.values())} decisions in {len(rec.order)} battle sides "
          f"({assembler_name}) -> {args.out}", flush=True)
    return 0


def run_compare(args: argparse.Namespace) -> int:
    import numpy as np

    a, b = np.load(args.a), np.load(args.b)
    ma, mb = json.loads(str(a["meta"])), json.loads(str(b["meta"]))
    ka, kb = [tuple(k) for k in ma["keys"]], [tuple(k) for k in mb["keys"]]
    ca, cb = dict(zip(ka, a["counts"].tolist())), dict(zip(kb, b["counts"].tolist()))
    sa, sb, pos = {}, {}, 0
    for k in ka:
        sa[k] = (pos, pos + ca[k]); pos += ca[k]
    pos = 0
    for k in kb:
        sb[k] = (pos, pos + cb[k]); pos += cb[k]
    common = [k for k in ka if k in sb and ca[k] == cb[k]]
    report: Dict[str, Any] = {
        "a": {k: ma.get(k) for k in ("mode", "code", "checkpoint", "assembler", "total_dim", "notes", "obs_line")},
        "b": {k: mb.get(k) for k in ("mode", "code", "checkpoint", "assembler", "total_dim", "notes", "obs_line")},
        "battle_sides": [len(ka), len(kb)], "decisions": [int(a["counts"].sum()), int(b["counts"].sum())],
        "sides_in_one_only": len(set(ka) ^ set(kb)), "sides_with_other_counts": sum(1 for k in ka if k in cb and ca[k] != cb[k]),
        "compared_decisions": int(sum(ca[k] for k in common)), "prefix": args.prefix,
    }
    ia = np.concatenate([np.arange(*sa[k]) for k in common]) if common else np.zeros(0, dtype=np.int64)
    ib = np.concatenate([np.arange(*sb[k]) for k in common]) if common else np.zeros(0, dtype=np.int64)
    oa, ob = a["obs"][ia], b["obs"][ib]
    if args.prefix:
        oa, ob = oa[:, :args.prefix], ob[:, :args.prefix]
    if oa.shape[1:] != ob.shape[1:]:
        report["obs"] = f"different sizes: {oa.shape[1:]} and {ob.shape[1:]}"
        report["obs_differing_decisions"] = -1
    else:
        same_rows = np.all(oa.view(np.uint16) == ob.view(np.uint16), axis=1)
        report["obs_identical_decisions"] = int(same_rows.sum())
        report["obs_differing_decisions"] = int((~same_rows).sum())
        if (~same_rows).any():
            cols = np.nonzero(np.any(oa.view(np.uint16) != ob.view(np.uint16), axis=0))[0]
            report["obs_differing_columns"] = int(cols.size)
            report["obs_first_differing_columns"] = cols[:20].tolist()
    for name in ("logits", "v_logits", "v_exp"):
        if name in a.files and name in b.files:
            xa, xb = a[name][ia].astype(np.float32), b[name][ib].astype(np.float32)
            same = np.all(xa.reshape(len(xa), -1).view(np.uint32) == xb.reshape(len(xb), -1).view(np.uint32), axis=1)
            report[name] = {"bit_identical_decisions": int(same.sum()), "differing_decisions": int((~same).sum()),
                            "max_abs_diff": float(np.max(np.abs(xa - xb))) if xa.size else 0.0}
            if name == "logits" and xa.size:
                # The same choice is what matters for play: the greedy action under the observation's own mask.
                m0 = None
                report[name]["argmax_equal_decisions"] = int((xa.argmax(axis=1) == xb.argmax(axis=1)).sum())
    text = json.dumps(report, indent=1, default=str)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    ok = (report.get("sides_in_one_only") == 0 and report.get("sides_with_other_counts") == 0
          and report.get("obs_differing_decisions") == 0
          and all(report.get(n, {}).get("differing_decisions", 0) == 0 for n in ("logits", "v_logits", "v_exp")))
    print("IDENTICAL" if ok else "DIFFERENT", flush=True)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ref")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--streams", required=True)
    p.add_argument("--battles", type=int, default=1000000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bf16", action="store_true", help="the inference actor's bf16 recurrent path instead of fp32")
    p.add_argument("--out", required=True)
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--prefix", type=int, default=0, help="compare only the first N numbers of each observation (0: all)")
    p.add_argument("--out", default="")
    args = ap.parse_args()
    for name in ("checkpoint", "streams", "out"):
        if getattr(args, name, None):
            setattr(args, name, os.path.abspath(os.path.expanduser(getattr(args, name))))
    if args.cmd == "ref":
        return run_ref(args)
    return run_compare(args)


if __name__ == "__main__":
    sys.exit(main())
