#!/usr/bin/env python
"""
Widens a learner checkpoint trained without the matchup block (obs_matchup.py) into one whose
network takes it, without changing what the network does.

    python tools/widen_checkpoint.py --in ckpt.pt --out ckpt_matchup.pt [--set-head]

With --movesets it widens a checkpoint that HAS the matchup block into one that also takes the
moveset block (obs_movesets.py), the same way:

    python tools/widen_checkpoint.py --movesets --in ckpt_matchup.pt --out ckpt_movesets.pt [--set-head]

  * pokemon_net.0.weight gets 103 zero columns after the matchup block's (move embedding 64, then
    the block's 39 dense numbers), in "model", "target_model", the fp32 master and both Adam moments;
    move_net is not touched;
  * an existing SetHead keeps its weights and its Adam state; its new move and Tera layers start
    fresh (small random, zero bias, seeded so the tool is repeatable, no Adam state). An
    OppActionHead is kept as it is. --set-head adds a fresh SetHead to a checkpoint without one;
  * records run_cfg["obs"]["movesets"] = True and a "widened_movesets" entry.

What it does:
  * builds the network with the block (ObservationAssembler(matchup=True).meta): the first Linear
    of move_net and of pokemon_net gets extra input columns at the end (18 and 99);
  * copies every weight of the checkpoint into it, and sets the NEW columns of move_net.0.weight
    and pokemon_net.0.weight to zero, in "model" and in "target_model";
  * widens the optimizer the same way: the fp32 master copies and AdamW's exp_avg and exp_avg_sq
    get zero columns, "step" and every hyperparameter stay, the param groups are rebuilt by the
    learner's own LearnerActor._init_optimizer so their order is the one a learner will expect;
  * records the setting: run_cfg["obs"]["matchup"] = True, and a "widened" entry saying from what;
  * --set-head also attaches a fresh SetHead (ppo_core.attach_set_head): its weights are new (small
    random, zero bias) and have no Adam state yet; it changes no logit or value.

Everything else (scheduler, counters, RNG states, the rest of run_cfg) is copied as it is. On the
first day the widened network gives the same logits and values as the original on any
observation: the new columns multiply the block by zero, and ppo_core._wide_subnet computes the
old product with the old shapes (tests/test_widen.py checks it on 2,000 recorded decisions).

Loading the result needs a network built with the block (obs_matchup.build_model, or make_model
once config passes ObservationAssembler(matchup=cfg.obs.matchup)) and an observation with it.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import os
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

WIDENED = ("move_net.0.weight", "pokemon_net.0.weight")
WIDENED_MOVESETS = ("pokemon_net.0.weight",)
AUX_PREFIXES = ("set_head.", "opp_action_head.")


@contextlib.contextmanager
def in_repo_root() -> Iterator[None]:
    """ObservationAssembler opens vocab.json by a relative path."""
    here = os.getcwd()
    os.chdir(REPO)
    try:
        yield
    finally:
        os.chdir(here)


def run_config_for(ckpt: Dict[str, Any]) -> Any:
    """RunConfig.default() with the checkpoint's model and learner fields where they exist (the network's shape comes from them)."""
    from config import RunConfig
    cfg = RunConfig.default()
    rc = ckpt.get("run_cfg") or {}
    parts = {}
    for name in ("model", "learner"):
        sub = getattr(cfg, name)
        saved = rc.get(name) if isinstance(rc.get(name), dict) else {}
        known = {f.name for f in dataclasses.fields(sub) if f.init}
        parts[name] = dataclasses.replace(sub, **{k: v for k, v in saved.items() if k in known})
    return dataclasses.replace(cfg, **parts)


def widen_tensor(old: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    """old with zero columns appended along the last dimension, up to `shape`."""
    if old.dim() != 2 or tuple(old.shape[:-1]) != tuple(shape[:-1]) or old.shape[-1] > shape[-1]:
        raise ValueError(f"cannot widen {tuple(old.shape)} to {tuple(shape)}")
    out = torch.zeros(shape, dtype=old.dtype)
    out[:, : old.shape[-1]] = old
    return out


def widen_state_dict(old: Dict[str, torch.Tensor], template: Dict[str, torch.Tensor],
                     widened: Tuple[str, ...] = WIDENED) -> Dict[str, torch.Tensor]:
    """
    The new network's state dict (template: its keys, shapes and fresh values) filled from the old
    one: equal shapes copied, `widened` keys given zero columns, a SetHead's new keys kept fresh.
    """
    out: Dict[str, torch.Tensor] = {}
    for k, fresh in template.items():
        if k in old:
            v = old[k]
            if tuple(v.shape) == tuple(fresh.shape):
                out[k] = v.detach().clone()
            elif k in widened:
                out[k] = widen_tensor(v.detach(), tuple(fresh.shape))
            else:
                raise ValueError(f"{k}: the checkpoint has {tuple(v.shape)}, the widened network {tuple(fresh.shape)}")
        elif k.startswith("set_head."):
            out[k] = fresh.detach().clone()
        else:
            raise ValueError(f"{k} is not in the checkpoint")
    extra = [k for k in old if k not in template and "attn_mask" not in k]
    if extra:
        raise ValueError(f"the checkpoint has tensors the network does not: {extra[:5]}")
    return out


def learner_optimizer(net: torch.nn.Module, learner_cfg: Any) -> Any:
    """The optimizer a LearnerActor would build for `net`: its own _init_optimizer, run on a stand-in actor."""
    from learner import LearnerActor
    fake = SimpleNamespace(net=net, cfg=learner_cfg, opt=None, sched=None)
    fake._get_zone_config = lambda: LearnerActor._get_zone_config(fake)
    fake._init_scheduler = lambda: None
    LearnerActor._init_optimizer(fake)
    return fake.opt


def _names(net: torch.nn.Module) -> Dict[int, str]:
    return {id(p): n for n, p in net.named_parameters()}


def widen_optimizer(saved: Dict[str, Any], old_net: torch.nn.Module, new_net: torch.nn.Module,
                    learner_cfg: Any, old_model_sd: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    The checkpoint's optimizer state for new_net. The old state is first loaded into the optimizer
    the learner builds for old_net, and every master is checked against the checkpoint's model
    weight of the same name, so the name of each saved tensor is known, not guessed.
    """
    if not isinstance(saved, dict) or "inner" not in saved or "masters" not in saved:
        raise ValueError("the optimizer state is not the learner's MasterWeightOptimizer format")
    opt_old = learner_optimizer(old_net, learner_cfg)
    if len(saved["masters"]) != len(opt_old.pairs):
        raise ValueError(f"the checkpoint has {len(saved['masters'])} master weights, the learner's optimizer {len(opt_old.pairs)}")
    for (p, _m), s in zip(opt_old.pairs, saved["masters"]):
        if tuple(s.shape) != tuple(p.shape):
            raise ValueError(f"master shapes differ: {tuple(s.shape)} and {tuple(p.shape)}")
    opt_old.load_state_dict(saved)
    old_names = _names(old_net)
    by_name: Dict[str, Tuple[torch.Tensor, Dict[str, Any]]] = {}
    for p, m in opt_old.pairs:
        name = old_names[id(p)]
        want = old_model_sd[name]
        if not torch.equal(m.detach().to(want.dtype), want):
            raise ValueError(f"master of {name} is not the checkpoint's weight: the learner's parameter order has changed")
        by_name[name] = (m, opt_old.inner.state.get(m, {}))

    opt_new = learner_optimizer(new_net, learner_cfg)
    new_names = _names(new_net)
    seen = set()
    with torch.no_grad():
        for p, m in opt_new.pairs:
            name = new_names[id(p)]
            if name not in by_name:
                if not name.startswith("set_head."):
                    raise ValueError(f"{name} has no optimizer state in the checkpoint")
                continue                          # a new SetHead: master = its fresh weight, no Adam state yet
            seen.add(name)
            m_old, st_old = by_name[name]
            m.copy_(m_old if m_old.shape == m.shape else widen_tensor(m_old, tuple(m.shape)))
            st_new: Dict[str, Any] = {}
            for k, v in st_old.items():
                if torch.is_tensor(v) and v.dim() == 2 and tuple(v.shape) != tuple(m.shape):
                    st_new[k] = widen_tensor(v, tuple(m.shape)).to(v.device)
                else:
                    st_new[k] = v.clone() if torch.is_tensor(v) else copy.deepcopy(v)
            if st_new:
                opt_new.inner.state[m] = st_new
    missing = set(by_name) - seen
    if missing:
        raise ValueError(f"optimizer state of {sorted(missing)[:5]} has no place in the widened network")

    out = opt_new.state_dict()
    # Hyperparameters (lr, initial_lr, betas, weight_decay...) of each group as they were saved.
    old_groups = {g.get("name"): g for g in saved["inner"]["param_groups"]}
    new_groups = out["inner"]["param_groups"]
    new_names = [g.get("name") for g in new_groups]
    added = [n for n in new_names if n not in old_groups]
    if sorted(old_groups) != sorted(n for n in new_names if n in old_groups) or any(not str(n).startswith("aux_") for n in added):
        raise ValueError("the optimizer's parameter groups differ")
    for g in new_groups:
        if g.get("name") not in old_groups:
            continue                                  # the auxiliary heads' groups (a new SetHead): their own settings
        for k, v in old_groups[g.get("name")].items():
            if k != "params":
                g[k] = copy.deepcopy(v)
    return out


def check_optimizer(cfg: Any, out: Dict[str, Any], set_head: bool, old_in: Dict[str, int], movesets: bool = False,
                    opp_action_head: bool = False) -> None:
    """Loads the widened optimizer state the way the learner does, into a fresh network and optimizer, and checks every master against its weight."""
    import obs_matchup
    with in_repo_root():
        net = obs_matchup.build_model(cfg, matchup=True, set_head=set_head, movesets=movesets, opp_action_head=opp_action_head)
    net.load_state_dict(out["model"], strict=True)
    opt = learner_optimizer(net, cfg.learner)
    opt.load_state_dict(out["optimizer"])
    names = _names(net)
    for p, m in opt.pairs:
        name = names[id(p)]
        want = out["model"][name]
        if not torch.equal(m.detach().to(want.dtype), want):
            raise ValueError(f"widened master of {name} is not the widened weight")
        if name in old_in:
            st = opt.inner.state.get(m, {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in st and bool(st[key][:, old_in[name]:].abs().sum() != 0):
                    raise ValueError(f"{name}: {key} is not zero on the new columns")


def _target_template(new_sd: Dict[str, torch.Tensor], old_target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """The widened network's keys that the EMA target holds: everything but the auxiliary heads the old target did not hold."""
    have = {p for p in AUX_PREFIXES if any(str(k).startswith(p) for k in old_target)}
    return {k: v for k, v in new_sd.items() if not k.startswith(AUX_PREFIXES) or k.startswith(tuple(have))}


def widen_movesets(ckpt: Dict[str, Any], set_head: bool = False, source: str = "") -> Dict[str, Any]:
    """A checkpoint with the matchup block, widened to take the moveset block too (a new dict; the given one is not changed)."""
    import obs_matchup
    import obs_movesets
    if not obs_matchup.checkpoint_obs_matchup(ckpt):
        raise ValueError("the checkpoint has no matchup block: widen it to the matchup block first (without --movesets)")
    if obs_movesets.checkpoint_obs_movesets(ckpt):
        raise ValueError("the checkpoint already has the moveset block")
    cfg = run_config_for(ckpt)
    had_sh = obs_matchup.checkpoint_set_head(ckpt)
    had_opp = any(str(k).startswith("opp_action_head.") for k in ckpt["model"])
    sh = bool(had_sh or set_head)
    with in_repo_root(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)              # the new SetHead layers: the same values at every run of the tool
        old_net = obs_matchup.build_model(cfg, matchup=True, set_head=had_sh, opp_action_head=had_opp)
        new_net = obs_matchup.build_model(cfg, matchup=True, movesets=True, set_head=sh, opp_action_head=had_opp)
    old_sd = {k: v for k, v in ckpt["model"].items() if "attn_mask" not in k}
    old_net.load_state_dict(old_sd, strict=True)     # the checkpoint is this network, every tensor of it
    new_sd = new_net.state_dict()

    out = {k: v for k, v in ckpt.items()}
    out["model"] = widen_state_dict(old_sd, new_sd, WIDENED_MOVESETS)
    new_net.load_state_dict(out["model"], strict=True)
    if ckpt.get("target_model") is not None:
        tgt = {k: v for k, v in ckpt["target_model"].items() if "attn_mask" not in k}
        out["target_model"] = widen_state_dict(tgt, _target_template(new_sd, tgt), WIDENED_MOVESETS)
    old_in = {k: int(old_sd[k].shape[1]) for k in WIDENED_MOVESETS}
    if ckpt.get("optimizer") is not None:
        out["optimizer"] = widen_optimizer(ckpt["optimizer"], old_net, new_net, cfg.learner, old_sd)
        check_optimizer(cfg, out, sh, old_in, movesets=True, opp_action_head=had_opp)

    run_cfg = copy.deepcopy(ckpt.get("run_cfg") or {})
    run_cfg.setdefault("obs", {})[obs_movesets.SETTING] = True
    out["run_cfg"] = run_cfg
    out["widened_movesets"] = {
        "from": source, "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "movesets_dim": obs_movesets.BLOCK_DIM, "sets_md5": obs_matchup.SetTable.default().md5,
        "pokemon_net_in": [old_in["pokemon_net.0.weight"], int(out["model"]["pokemon_net.0.weight"].shape[1])],
        "set_head": "kept, move and Tera layers added" if had_sh else ("added" if sh else "none"),
        "opp_action_head": bool(had_opp),
    }
    return out


def ff_expansion_of(model: Dict[str, torch.Tensor]) -> Optional[float]:
    """The feed-forward width the weights have (transformer.0.ff.0.weight rows over d_model), None without that tensor."""
    w = model.get("transformer.0.ff.0.weight")
    if not torch.is_tensor(w) or w.dim() != 2 or not w.shape[1]:
        return None
    return float(w.shape[0]) / float(w.shape[1])


def widen_main_line(ckpt: Dict[str, Any], source: str = "") -> Dict[str, Any]:
    """
    A network of the main line (M24381: the own Tera type and obs_abil's six fixes in its "obs_settings", no
    block) made into one of this line with the main line's observation (obs_main.py) and both blocks:
    the matchup block (widen, with a fresh SetHead), then the moveset block (widen_movesets), each by zero
    columns, so it gives the main line's logits and values on day one. Then run_cfg["obs"] says
    main_line True and own_tera_type False (the main line's own Tera type is part of main_line), and the
    "obs_settings" entry is the one a main-line learner of this line writes (obs_main.checkpoint_obs_settings).

    Two repairs, both recorded in "widened_main_line":
      * run_cfg["model"]["ff_expansion"]: the main line's config said 4.0 but every layer was built at 2.0
        (config.ModelConfig's note); this line builds the layers from the config, so the value the weights
        have is written in;
      * the optimizer: widened when it is this line's MasterWeightOptimizer format, else left out (the
        start of a run from it is a phase change, warmup or PPO from mode ppo, which builds a fresh one
        and never reads the saved one).
    """
    import obs_main
    import obs_matchup
    if obs_matchup.checkpoint_obs_matchup(ckpt):
        raise ValueError("the checkpoint already has the matchup block: --main-line widens a main-line network without blocks")
    if not obs_main.checkpoint_has_main_settings(ckpt):
        raise ValueError(f"the checkpoint is not the main line's (its obs_settings {ckpt.get('obs_settings')!r} do not have "
                         f"all of {sorted(obs_main.MAIN_LINE_SETTINGS)})")
    src = {k: v for k, v in ckpt.items()}
    run_cfg = copy.deepcopy(ckpt.get("run_cfg") or {})
    ff_weights = ff_expansion_of(ckpt["model"])
    ff_saved = (run_cfg.get("model") or {}).get("ff_expansion")
    if ff_weights is not None and ff_saved != ff_weights:
        run_cfg.setdefault("model", {})["ff_expansion"] = ff_weights
    src["run_cfg"] = run_cfg
    optimizer_note = "widened"
    try:
        step1 = widen(src, set_head=True, source=source)
    except (ValueError, KeyError, RuntimeError) as e:
        if src.get("optimizer") is None:
            raise
        optimizer_note = f"left out: {type(e).__name__}: {str(e)[:200]}"
        src = dict(src, optimizer=None)
        step1 = widen(src, set_head=True, source=source)
    out = widen_movesets(step1, set_head=True, source=source)
    rc = out["run_cfg"]
    rc.setdefault("obs", {})["own_tera_type"] = False
    rc["obs"][obs_main.SETTING] = True
    out["obs_settings"] = obs_main.checkpoint_obs_settings(True, True)
    out["widened_main_line"] = {
        "from": source, "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "obs_settings_before": copy.deepcopy(ckpt.get("obs_settings")),
        "ff_expansion": [ff_saved, ff_weights], "optimizer": optimizer_note,
        "update_idx": int(ckpt.get("update_idx", -1)), "total_steps": int(ckpt.get("total_steps", -1)),
        "first_part_dim": obs_main.MAIN_LINE_DIM,
    }
    return out


def widen(ckpt: Dict[str, Any], set_head: bool = False, source: str = "", movesets: bool = False) -> Dict[str, Any]:
    """The widened checkpoint (a new dict; the given one is not changed). movesets=True: widen_movesets."""
    if movesets:
        return widen_movesets(ckpt, set_head=set_head, source=source)
    import obs_matchup
    if obs_matchup.checkpoint_obs_matchup(ckpt):
        raise ValueError("the checkpoint already has the matchup block")
    cfg = run_config_for(ckpt)
    with in_repo_root():
        old_net = obs_matchup.build_model(cfg, matchup=False)
        new_net = obs_matchup.build_model(cfg, matchup=True, set_head=set_head)
    old_sd = {k: v for k, v in ckpt["model"].items() if "attn_mask" not in k}
    old_net.load_state_dict(old_sd, strict=True)     # the checkpoint is this network, every tensor of it

    out = {k: v for k, v in ckpt.items()}
    out["model"] = widen_state_dict(old_sd, new_net.state_dict())
    new_net.load_state_dict(out["model"], strict=True)
    if ckpt.get("target_model") is not None:
        # A SetHead's keys get the same fresh values as in "model" (both come from new_net).
        tgt = {k: v for k, v in ckpt["target_model"].items() if "attn_mask" not in k}
        out["target_model"] = widen_state_dict(tgt, new_net.state_dict())
    if ckpt.get("optimizer") is not None:
        out["optimizer"] = widen_optimizer(ckpt["optimizer"], old_net, new_net, cfg.learner, old_sd)
        check_optimizer(cfg, out, set_head, {k: int(old_sd[k].shape[1]) for k in WIDENED})

    run_cfg = copy.deepcopy(ckpt.get("run_cfg") or {})
    run_cfg.setdefault("obs", {})[obs_matchup.SETTING] = True
    out["run_cfg"] = run_cfg
    out["widened"] = {
        "from": source, "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "matchup_dim": obs_matchup.BLOCK_DIM, "sets_md5": obs_matchup.SetTable.default().md5,
        "move_net_in": [int(ckpt["model"]["move_net.0.weight"].shape[1]), int(out["model"]["move_net.0.weight"].shape[1])],
        "pokemon_net_in": [int(ckpt["model"]["pokemon_net.0.weight"].shape[1]), int(out["model"]["pokemon_net.0.weight"].shape[1])],
        "set_head": bool(set_head),
    }
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True, help="a learner checkpoint without the block")
    ap.add_argument("--out", required=True, help="where to write the widened checkpoint (not overwritten)")
    ap.add_argument("--set-head", action="store_true", help="also attach a fresh SetHead")
    ap.add_argument("--movesets", action="store_true",
                    help="widen a checkpoint WITH the matchup block to take the moveset block too (obs_movesets.py)")
    ap.add_argument("--main-line", action="store_true",
                    help="a main-line network (M24381) into one of this line with the main line's observation "
                         "(obs_main.py, --obs-main-line) and both blocks, with a fresh SetHead (widen_main_line)")
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists; not overwritten")
    if args.main_line and (args.movesets or args.set_head):
        raise SystemExit("--main-line does both blocks and the SetHead itself: give it alone")
    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    if args.main_line:
        out = widen_main_line(ckpt, source=os.path.abspath(args.src))
    else:
        out = widen(ckpt, set_head=args.set_head, source=os.path.abspath(args.src), movesets=args.movesets)
    tmp = args.out + ".tmp"
    torch.save(out, tmp)
    os.replace(tmp, args.out)
    if args.main_line:
        w, m = out["widened_main_line"], out["widened_movesets"]
        print(f"{args.out}: main line -> main_line + matchup + movesets; move_net in {out['widened']['move_net_in']}, "
              f"pokemon_net in {out['widened']['pokemon_net_in'][0]} -> {m['pokemon_net_in'][1]}, set head {m['set_head']}, "
              f"ff_expansion {w['ff_expansion'][0]} -> {w['ff_expansion'][1]}, optimizer {w['optimizer']}")
        return 0
    if args.movesets:
        w = out["widened_movesets"]
        print(f"{args.out}: pokemon_net in {w['pokemon_net_in'][0]} -> {w['pokemon_net_in'][1]}, block {w['movesets_dim']}, "
              f"set head {w['set_head']}, opp action head {w['opp_action_head']}, "
              f"optimizer {'widened' if out.get('optimizer') is not None else 'absent'}")
        return 0
    w = out["widened"]
    print(f"{args.out}: move_net in {w['move_net_in'][0]} -> {w['move_net_in'][1]}, pokemon_net in "
          f"{w['pokemon_net_in'][0]} -> {w['pokemon_net_in'][1]}, block {w['matchup_dim']}, set head {w['set_head']}, "
          f"optimizer {'widened' if out.get('optimizer') is not None else 'absent'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
