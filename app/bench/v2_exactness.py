#!/usr/bin/env python
"""
Is the window app's player exactly branch train-v2's worker and network? Recorded battles
replayed through both, and every decision compared: the observation byte for byte, the logits, the
value logits and the expected value bit for bit.

    ref      train-v2's own code (--code: a train-v2 release, e.g. ~/rel/v2-movesets). The streams
             go through its worker path (bench/micro_obs.py: worker.SyncBridgePlayer with
             worker.py's poke-env patches), each player with the assembler worker.py builds from
             the checkpoint's run_cfg["obs"] (RunConfig.make_assembler). Then its network
             (RunConfig.make_model with those flags, the training-only heads attached as the
             learner does, the FULL checkpoint loaded strictly) runs every side of every battle one
             decision at a time with its history, as players.BatchedPolicyPlayer._forward_batch
             calls it.
    app      this branch's code (or --code: another copy of it, e.g. the commit before a change):
             the window app's network-only player (players.audited(batched_policy_player()) with
             ladder.make_player's HybridConfig), the network loaded from --checkpoint (an export, or
             the full checkpoint) by hybrid.PolicyNet. The same streams through poke-env's client;
             every observation the player assembles and every row its forward passes return.
    compare  two of the files above, decision by decision.

The fakes are the same in both: the websocket (messages come from the file, orders are dropped).

    python bench/v2_exactness.py ref --code ~/rel/v2-movesets --checkpoint FULL.pt --streams streams.jsonl --out ref.npz
    python bench/v2_exactness.py app --checkpoint EXPORT.pt --streams streams.jsonl --out app.npz
    python bench/v2_exactness.py compare ref.npz app.npz

The same app mode run with the code before a change and after it (--code) shows that a network
of another line is played byte for byte as before.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _use_code(code: str) -> None:
    """Imports come from `code` only (not from this file's repository), and vocab.json is opened there."""
    code = os.path.abspath(code)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") not in (here, REPO, os.getcwd())]
    sys.path.insert(0, code)
    os.chdir(code)


class Recorder:
    """Per (player, battle tag): the observations, and the network's rows, in decision order."""

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
        if all(len(self.rows[k]) == len(self.obs[k]) for k in keys) and keys and self.rows[keys[0]]:
            for name in ("logits", "v_logits", "v_exp"):
                out[name] = np.concatenate([np.stack([r[name] for r in self.rows[k]]) for k in keys])
        meta = dict(meta, keys=[list(k) for k in keys], notes=dict(self.notes))
        out["meta"] = np.array(json.dumps(meta))
        np.savez_compressed(path, **out)


def _load_streams(path: str, battles: int) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["b"] >= battles:
                break
            out.append(d)
    return out


async def _replay(players: Dict[str, Any], msgs: List[Dict[str, Any]]) -> None:
    for d in msgs:
        await players[d["p"]].ps_client._handle_message(d["m"])


def _watch_assembler(player: Any, asm: Any, rec: Recorder) -> None:
    inner = asm.assemble

    def assemble(battle: Any, events: Any = None):
        got = inner(battle, events=events)
        rec.add_obs((player.username, battle.battle_tag), got)
        return got

    asm.assemble = assemble


# --------------------------------------------------------------------------------------- ref


def run_ref(args: argparse.Namespace) -> int:
    _use_code(args.code)
    import dataclasses

    import numpy as np
    import torch

    import config as config_mod
    from bench import micro_obs
    from poke_env.concurrency import POKE_LOOP

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    flags = (ck.get("run_cfg") or {}).get("obs") or {}
    base = config_mod.RunConfig.default()
    cfg = dataclasses.replace(base, obs=dataclasses.replace(
        base.obs, own_tera_type=bool(flags.get("own_tera_type", False)), matchup=bool(flags.get("matchup", False)),
        movesets=bool(flags.get("movesets", False))))

    # The worker's players: micro_obs.build_players with RunConfig.default() giving this run's
    # observation flags, so each SyncBridgePlayer builds its own assembler as worker.py does.
    real_default = config_mod.RunConfig.default
    config_mod.RunConfig.default = classmethod(lambda cls: cfg)
    try:
        _cfg, players, _q = micro_obs.build_players("ppo", "simple", 0, 7)
    finally:
        config_mod.RunConfig.default = real_default
    rec = Recorder()
    for p in players.values():
        _watch_assembler(p, p.assembler, rec)
    msgs = _load_streams(args.streams, args.battles)
    t0 = time.time()
    asyncio.run_coroutine_threadsafe(micro_obs.replay(players, msgs, defaultdict(micro_obs.Acc)), POKE_LOOP).result()
    replay_s = time.time() - t0
    rec.notes["refused_choices"] = int(sum(getattr(p, "choices_refused", 0) for p in players.values()))

    # The network: train-v2's, with the heads its learner attaches, the whole checkpoint loaded.
    net = cfg.make_model()
    model = ck["model"]
    if any(str(k).startswith("set_head.") for k in model):
        net.attach_set_head()
    if any(str(k).startswith("opp_action_head.") for k in model):
        net.attach_opp_action_head()
    missing, unexpected = net.load_state_dict(model, strict=False)
    learned = {n for n, _p in net.named_parameters()}
    if [k for k in missing if k in learned] or unexpected:
        raise SystemExit(f"the checkpoint does not fit train-v2's network: missing {missing}, unexpected {unexpected}")
    rec.notes["buffers_not_in_checkpoint"] = len(missing)
    dev = torch.device(args.device)
    net = net.float().to(dev).eval()
    t0 = time.time()
    with torch.no_grad():
        for key in rec.order:
            kv = [torch.zeros(1, net.kv_cache_len, net.d_model, device=dev) for _ in range(net.n_layers)]
            kv_len = torch.zeros(1, dtype=torch.long, device=dev)
            for obs in rec.obs[key]:
                # As players.BatchedPolicyPlayer._forward_batch calls it, for a batch of one battle.
                x = torch.from_numpy(np.ascontiguousarray(obs[None], dtype=np.float32)).to(dev)
                kv_in = [torch.cat([layer], dim=0) for layer in kv]
                len_in = torch.cat([kv_len.reshape(1).to(dev)], dim=0)
                out, v_logits, v_exp, new_kv, new_len = net(x, kv_cache=kv_in, kv_cache_len=len_in)
                kv = [layer[0:1].clone() for layer in new_kv]
                kv_len = new_len[0:1].clone()
                rec.rows[key].append({"logits": out[0].float().cpu().numpy(), "v_logits": v_logits[0].float().cpu().numpy(),
                                      "v_exp": v_exp.reshape(-1)[0].float().cpu().numpy()})
    rec.save(args.out, {"mode": "ref", "code": os.path.abspath(args.code), "checkpoint": os.path.abspath(args.checkpoint),
                        "obs_flags": {k: flags.get(k) for k in ("own_tera_type", "matchup", "movesets")},
                        "total_dim": int(rec.obs[rec.order[0]][0].shape[0]) if rec.order else 0,
                        "replay_s": round(replay_s, 1), "net_s": round(time.time() - t0, 1), "messages": len(msgs)})
    print(f"ref: {sum(len(v) for v in rec.obs.values())} decisions in {len(rec.order)} battle sides -> {args.out}", flush=True)
    return 0


# --------------------------------------------------------------------------------------- app


async def _noop_send(*_a: Any, **_k: Any) -> None:
    return None


def run_app(args: argparse.Namespace) -> int:
    _use_code(args.code or REPO)
    import numpy as np
    import torch

    from hybrid import HybridConfig
    from players import audited, batched_policy_player
    from poke_env import AccountConfiguration, ServerConfiguration
    from poke_env.concurrency import POKE_LOOP

    cls = audited(batched_policy_player())
    rec = Recorder()
    players: Dict[str, Any] = {}
    for side, name in (("p1", "benchpa"), ("p2", "benchpb")):
        # ladder.make_player's HybridConfig for the network alone.
        hcfg = HybridConfig(checkpoint=args.checkpoint, beta=0.0, veto_gap=None, policy_weight=1.0,
                            greedy=True, device=args.device, random_init=False)
        p = cls(hybrid=hcfg, account_configuration=AccountConfiguration(name, None),
                server_configuration=ServerConfiguration("ws://127.0.0.1:1/showdown/websocket", "http://127.0.0.1:1/action.php?"),
                battle_format="gen9randombattle", max_concurrent_battles=0, start_listening=False, log_level=40)
        p.ps_client.send_message = _noop_send
        _watch_assembler(p, p.assembler, rec)
        net = p.policy.net
        last: Dict[str, Any] = {}
        real_forward = net.forward

        def forward(*a: Any, _real=real_forward, _last=last, **k: Any):
            out = _real(*a, **k)
            _last["v_logits"] = out[1].detach().float().cpu().numpy()
            _last["v_exp"] = out[2].detach().float().reshape(-1).cpu().numpy()
            return out

        net.forward = forward
        real_batch = p._forward_batch

        def forward_batch(tags: List[str], obs: Any, _real=real_batch, _last=last, _p=p):
            rows = _real(tags, obs)
            for b, tag in enumerate(tags):
                rec.rows[(_p.username, tag)].append({"logits": np.asarray(rows[b], dtype=np.float32),
                                                     "v_logits": _last["v_logits"][b], "v_exp": _last["v_exp"][b]})
            return rows

        p._forward_batch = forward_batch
        real_choose = p.choose_move

        async def choose_move(battle: Any, _real=real_choose, _p=p):
            order = await _real(battle)
            # The decisions where the choice mask is not the observation's (a switch the server refuses).
            try:
                m0, m1 = _p.assembler.offsets["action_mask"]
                obs_mask = rec.obs[(_p.username, battle.battle_tag)][-1][m0:m1].astype(np.float32)
                pick = getattr(_p.assembler, "choice_mask", None)
                if pick is not None and not np.array_equal(pick(battle, rec.obs[(_p.username, battle.battle_tag)][-1]), obs_mask):
                    rec.notes["choice_mask_differs"] += 1
            except Exception:
                rec.notes["choice_mask_check_failed"] += 1
            return order

        p.choose_move = choose_move
        players[side] = p
    pol = players["p1"].policy
    msgs = _load_streams(args.streams, args.battles)
    t0 = time.time()
    asyncio.run_coroutine_threadsafe(_replay(players, msgs), POKE_LOOP).result()
    for p in players.values():
        for k in ("decisions", "fallbacks"):
            rec.notes[k] += int(p.counts.get(k, 0))
        for k, v in getattr(p, "audit", {}).items():
            rec.notes["audit_" + k] += int(v)
    rec.save(args.out, {"mode": "app", "code": os.path.abspath(args.code or REPO), "checkpoint": os.path.abspath(args.checkpoint),
                        "obs_line": getattr(pol, "obs_line", "own"), "v2_blocks": getattr(pol, "v2_blocks", None),
                        "league_blocks": getattr(pol, "league_blocks", None),
                        "dropped_heads": getattr(pol, "dropped_heads", None), "assembler": type(pol.assembler).__name__,
                        "total_dim": int(pol.assembler.total_dim), "replay_s": round(time.time() - t0, 1), "messages": len(msgs),
                        "torch": torch.__version__})
    print(f"app: {sum(len(v) for v in rec.obs.values())} decisions in {len(rec.order)} battle sides "
          f"(obs line {getattr(pol, 'obs_line', 'own')}, {type(pol.assembler).__name__}) -> {args.out}", flush=True)
    return 0


# --------------------------------------------------------------------------------------- compare


def run_compare(args: argparse.Namespace) -> int:
    import numpy as np

    a, b = np.load(args.a), np.load(args.b)
    ma, mb = json.loads(str(a["meta"])), json.loads(str(b["meta"]))
    ka, kb = [tuple(k) for k in ma["keys"]], [tuple(k) for k in mb["keys"]]
    ca, cb = dict(zip(ka, a["counts"].tolist())), dict(zip(kb, b["counts"].tolist()))
    sa, sb = {}, {}
    pos = 0
    for k in ka:
        sa[k] = (pos, pos + ca[k]); pos += ca[k]
    pos = 0
    for k in kb:
        sb[k] = (pos, pos + cb[k]); pos += cb[k]
    common = [k for k in ka if k in sb and ca[k] == cb[k]]
    report: Dict[str, Any] = {
        "a": {k: ma.get(k) for k in ("mode", "code", "checkpoint", "obs_line", "assembler", "total_dim", "notes", "obs_flags", "dropped_heads")},
        "b": {k: mb.get(k) for k in ("mode", "code", "checkpoint", "obs_line", "assembler", "total_dim", "notes", "obs_flags", "dropped_heads")},
        "battle_sides": [len(ka), len(kb)], "decisions": [int(a["counts"].sum()), int(b["counts"].sum())],
        "sides_in_one_only": len(set(ka) ^ set(kb)), "sides_with_other_counts": sum(1 for k in ka if k in cb and ca[k] != cb[k]),
        "compared_decisions": int(sum(ca[k] for k in common)),
    }
    ia = np.concatenate([np.arange(*sa[k]) for k in common]) if common else np.zeros(0, dtype=np.int64)
    ib = np.concatenate([np.arange(*sb[k]) for k in common]) if common else np.zeros(0, dtype=np.int64)
    oa, ob = a["obs"][ia], b["obs"][ib]
    if oa.shape[1:] != ob.shape[1:]:
        report["obs"] = f"different sizes: {oa.shape[1:]} and {ob.shape[1:]}"
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
    for name in ("ref", "app"):
        p = sub.add_parser(name)
        p.add_argument("--code", default="" if name == "app" else None, required=(name == "ref"),
                       help="the code to run: a train-v2 release (ref), a copy of this branch (app; default this one)")
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--streams", required=True)
        p.add_argument("--battles", type=int, default=1000000)
        p.add_argument("--device", default="cuda")
        p.add_argument("--out", required=True)
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--out", default="")
    args = ap.parse_args()
    for name in ("checkpoint", "streams", "out", "code"):
        if getattr(args, name, None):
            setattr(args, name, os.path.abspath(os.path.expanduser(getattr(args, name))))
    if args.cmd == "ref":
        return run_ref(args)
    if args.cmd == "app":
        return run_app(args)
    return run_compare(args)


if __name__ == "__main__":
    sys.exit(main())
