#!/usr/bin/env python
"""
Arm B of the imitation test (REPORT_LADDER_AND_BASELINE_RUN.md, 2026-10-14): the network learns
the top players' own decisions FROM RANDOM WEIGHTS, the way arm A (train.py --mode imitation
--teacher psbot) learns PSBot's. The output is a learner checkpoint that pipeline.py can continue
with warmup and PPO, and that evaluate.py plays like any other.

    python human/imitate.py train --data DIR --out-dir RUN/checkpoints [--passes 16] [--lr 3e-4]

The data are human/rebuild.py's shards built with THIS branch's observation (one episode = one side
of one game at the 2100 bar, every decision in order). Splits come from the collector's files
(~/dev/human-data/splits) as in human/finetune.py: train (+ train_loose), test_player (held-out
players' own sides, used to pick the checkpoint), test_game (reported).

The loss, per minibatch of whole episodes packed into sequences of --seq decisions (3 x 256 = the
learner's 768-row minibatch), is the learner's imitation loss with the human's choice as the
teacher's action:

    smoothed CE(pi, a_player; 0.1 on the legal actions) over the labelled rows, each player capped
    at --cap of the labelled weight  +  vf_coef * distributional value loss on every real row

The value target is the return a rollout worker would see on that side with the run's reward
(RewardConfig: win +1 / loss -1 at the end, potential shaping scale * (gamma Phi(next) - Phi(now))),
taken whole (Monte Carlo): G_t = gamma^(T-1-t) z - scale * Phi_t (the shaping telescopes; Phi at the
end is 0). Learning rates follow LearnerConfig's imitation multipliers (backbone 1, actor 1,
critic 0.6), AdamW, warmup then cosine. Rows the replay could not label, or with one legal action,
are history (and value) only.

After every half pass the held-out players' cross-entropy (unsmoothed) and top-1 agreement are
measured; the checkpoint with the lowest held-out cross-entropy is the arm's result. Training stops
early after --patience measurements without a better one.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

SPLITS = os.path.expanduser("~/dev/human-data/splits")
SALT = os.path.expanduser("~/dev/human-rebuild/salt")
ACT = 14


def log(*a: Any) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def salted_game(salt: str, rid: str) -> str:
    """rebuild.salted("g", rid), with the salt read once."""
    return "g" + hashlib.sha256((salt + "|" + rid.lower()).encode()).hexdigest()[:12]


def read_splits(splits_dir: str, salt_path: str) -> Dict[str, Tuple[str, str]]:
    """salted game hash -> (split, sides to use)."""
    with open(salt_path) as f:
        salt = f.read().strip()
    out: Dict[str, Tuple[str, str]] = {}
    for name in ("train", "test_game", "test_player"):
        with open(os.path.join(splits_dir, name + ".tsv")) as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    out[salted_game(salt, parts[0])] = (name, parts[1])
    return out


class Data:
    """The shards' episodes in memory (human/finetune.py's Data, plus each episode's result and each
    step's Phi for the value target)."""

    def __init__(self, data_dir: str, splits_dir: str, salt_path: str, threads: int = 16) -> None:
        files = sorted(f for f in glob.glob(os.path.join(data_dir, "shard-*.npz")) if not f.endswith(".tmp.npz"))
        if not files:
            raise SystemExit(f"no shards in {data_dir}")
        with np.load(files[0]) as z:
            self.own_tera_type = bool(z["own_tera_type"]) if "own_tera_type" in z.files else None
            self.obs_abilities = str(z["obs_abilities"]) if "obs_abilities" in z.files else ""
            dim = int(z["obs"].shape[1])
        split_of = read_splits(splits_dir, salt_path)
        self.ep: List[Dict[str, Any]] = []
        seen = set()
        base = 0
        counts: Dict[str, int] = defaultdict(int)
        plan: List[Tuple[str, List[Tuple[int, int, int]]]] = []
        player_id: Dict[str, int] = {}
        for f in files:
            with np.load(f) as z:
                st, games, sides, players, won = z["ep_start"], z["ep_game"], z["ep_side"], z["ep_player"], z["ep_won"]
            keep: List[Tuple[int, int, int]] = []
            for e in range(len(st) - 1):
                g, side = str(games[e]), "p1" if int(sides[e]) == 1 else "p2"
                if (g, side) in seen:
                    counts["duplicate"] += 1
                    continue
                seen.add((g, side))
                sp = split_of.get(g)
                if sp is None:
                    counts["no_split"] += 1
                    continue
                name, use = sp
                if name == "test_player":
                    role = "test_player" if side in use else "train_loose"
                elif side in use:
                    role = name
                else:
                    role = "unused"
                counts[role] += 1
                if role == "unused":
                    continue
                a, b = int(st[e]), int(st[e + 1])
                keep.append((a, b, base))
                p = str(players[e])
                self.ep.append({"start": base, "len": b - a, "role": role, "player": p, "won": int(won[e]),
                                "pid": player_id.setdefault(p, len(player_id)), "game": g})
                base += b - a
            if keep:
                plan.append((f, keep))
        self.obs = np.empty((base, dim), dtype=np.float16)
        self.act = np.empty(base, dtype=np.int64)
        self.mask = np.empty((base, ACT), dtype=np.float32)
        self.train = np.empty(base, dtype=bool)
        self.phi = np.zeros(base, dtype=np.float32)
        missing_phi = [0]

        def fill(item: Tuple[str, List[Tuple[int, int, int]]]) -> None:
            f, keep = item
            with np.load(f) as z:
                obs, act, mask, tr = z["obs"], z["act"], z["mask"], z["train"]
                phi = z["phi"] if "phi" in z.files else None
            for a, b, d in keep:
                self.obs[d:d + b - a] = obs[a:b]
                self.act[d:d + b - a] = act[a:b]
                self.mask[d:d + b - a] = mask[a:b]
                self.train[d:d + b - a] = tr[a:b].astype(bool)
                if phi is not None:
                    self.phi[d:d + b - a] = np.nan_to_num(phi[a:b], nan=0.0)
                else:
                    missing_phi[0] += b - a

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max(1, threads)) as ex:
            list(ex.map(fill, plan))
        self.train &= self.act >= 0
        self.roles = dict(counts)
        self.missing_phi = missing_phi[0]
        self.players = [p for p, _ in sorted(player_id.items(), key=lambda kv: kv[1])]
        self.row_player = np.empty(base, dtype=np.int64)
        self.row_ep = np.empty(base, dtype=np.int64)
        for i, e in enumerate(self.ep):
            self.row_player[e["start"]:e["start"] + e["len"]] = e["pid"]
            self.row_ep[e["start"]:e["start"] + e["len"]] = i

    def episodes(self, roles: Tuple[str, ...]) -> List[int]:
        return [i for i, e in enumerate(self.ep) if e["role"] in roles]

    def rows(self, eps: List[int]) -> np.ndarray:
        return np.concatenate([np.arange(self.ep[i]["start"], self.ep[i]["start"] + self.ep[i]["len"]) for i in eps]) if eps else np.zeros(0, int)

    def returns(self, gamma: float, scale: float, shaping: bool) -> np.ndarray:
        """The value target of every row (see the module's docstring)."""
        g = np.empty(len(self.act), dtype=np.float32)
        for e in self.ep:
            n = e["len"]
            z = 1.0 if e["won"] > 0 else -1.0 if e["won"] < 0 else 0.0
            k = np.arange(n - 1, -1, -1, dtype=np.float64)      # steps to the end, minus one
            ret = (gamma ** k) * z
            if shaping:
                ret = ret - scale * self.phi[e["start"]:e["start"] + n]
            g[e["start"]:e["start"] + n] = ret
        return g

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"episodes_by_role": self.roles, "obs_dim": int(self.obs.shape[1]),
                               "own_tera_type": self.own_tera_type, "obs_abilities": self.obs_abilities,
                               "rows_without_phi": self.missing_phi}
        for role in ("train", "train_loose", "test_player", "test_game"):
            eps = self.episodes((role,))
            r = self.rows(eps)
            out[role] = {"sides": len(eps), "steps": int(len(r)), "labelled": int(self.train[r].sum()) if len(r) else 0,
                         "players": len({self.ep[i]["player"] for i in eps}), "games": len({self.ep[i]["game"] for i in eps})}
        return out


def pack(data: Data, eps: List[int], L: int, rng: Optional[np.random.Generator]) -> np.ndarray:
    """Sequences of L rows (-1 = padding) holding whole episodes (human/finetune.py's packing)."""
    order = list(eps)
    if rng is not None:
        rng.shuffle(order)
    pieces: List[np.ndarray] = []
    for i in order:
        e = data.ep[i]
        rows = np.arange(e["start"], e["start"] + e["len"])
        for k in range(0, len(rows), L):
            pieces.append(rows[k:k + L])
    seqs: List[List[np.ndarray]] = []
    fill: List[int] = []
    open_: List[int] = []
    for p in pieces:
        for j in open_:
            if fill[j] + len(p) <= L:
                seqs[j].append(p)
                fill[j] += len(p)
                if L - fill[j] < 8:
                    open_.remove(j)
                break
        else:
            seqs.append([p])
            fill.append(len(p))
            open_.append(len(seqs) - 1)
            if len(open_) > 32:
                open_.pop(0)
    out = np.full((len(seqs), L), -1, dtype=np.int64)
    for j, s in enumerate(seqs):
        r = np.concatenate(s)
        out[j, :len(r)] = r
    return out


def batch(data: Data, ret: np.ndarray, seqs: np.ndarray, dev: torch.device) -> Dict[str, torch.Tensor]:
    flat = seqs.reshape(-1)
    real = flat >= 0
    src = np.where(real, flat, flat[real][0])
    ep = np.where(real, data.row_ep[src], 10 ** 9 + np.arange(len(flat)))
    return {
        "obs": torch.from_numpy(data.obs[src]).to(dev, non_blocking=True),
        "ep": torch.from_numpy(ep.astype(np.int64)).to(dev),
        "act": torch.from_numpy(np.where(real, data.act[src], -1)).to(dev),
        "mask": torch.from_numpy(data.mask[src]).to(dev),
        "lab": torch.from_numpy(real & data.train[src]).to(dev),
        "real": torch.from_numpy(real).to(dev),
        "rows": torch.from_numpy(flat).to(dev),
        "ret": torch.from_numpy(ret[src]).to(dev),
    }


def player_weights(data: Data, rows: np.ndarray, cap: float) -> np.ndarray:
    """Row weights: a player holding more than `cap` of the labelled rows is scaled to `cap`."""
    lab = rows[data.train[rows]]
    count = np.bincount(data.row_player[lab], minlength=len(data.players)).astype(np.float64)
    w_p = np.ones(len(data.players), dtype=np.float64)
    if cap > 0:
        big = count > 0
        w_p[big] = np.minimum(1.0, cap * len(lab) / count[big])
    return w_p[data.row_player].astype(np.float32)


def fresh_net(dev: torch.device, seed: int) -> Tuple[torch.nn.Module, Any]:
    """A new network of this branch's RunConfig, random weights (seeded), on the learner's bf16 path."""
    from config import RunConfig
    import ppo_core
    if not getattr(ppo_core, "_learner_flex_compiled", False):
        ppo_core.flex_attention = torch.compile(ppo_core.flex_attention)
        ppo_core._learner_flex_compiled = True
    torch.manual_seed(seed)
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        cfg = RunConfig.default()
        cfg = dataclasses.replace(cfg, learner=dataclasses.replace(cfg.learner, mode="imitation"))
        net = cfg.make_model()
    finally:
        os.chdir(cwd)
    net = net.to(dev)
    net.enable_bf16_recurrent_path()
    return net, cfg


def smoothed_ce_rows(logits: torch.Tensor, act: torch.Tensor, mask: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row (smoothed CE, plain CE) on the legal actions: ppo_core.masked_smoothed_cross_entropy row by row
    (the target action is always counted legal, as there)."""
    m = mask.clone()
    m.scatter_(1, act[:, None], 1.0)
    legal = m > 0.5
    logp = torch.log_softmax(logits.masked_fill(~legal, -1e4), dim=-1)
    lp_a = logp.gather(1, act[:, None]).squeeze(1)
    n_legal = legal.float().sum(-1).clamp_min(1.0)
    mean_legal = logp.masked_fill(~legal, 0.0).sum(-1) / n_legal
    return -((1.0 - eps) * lp_a + eps * mean_legal), -lp_a


@torch.no_grad()
def measure(net: torch.nn.Module, data: Data, ret: np.ndarray, seqs: np.ndarray, L: int, S: int, dev: torch.device,
            v_min: float, v_max: float, v_bins: int) -> Dict[str, float]:
    import ppo_core
    net.eval()
    ce_sum = n = right = 0.0
    v_sum = v_n = 0.0
    for j in range(0, len(seqs), S):
        b = batch(data, ret, seqs[j:j + S], dev)
        logits, v_logits, _v, _a, _b = net(b["obs"], b["ep"], batch_seq_len=L)
        logits, v_logits = logits.float(), v_logits.float()
        lab = b["lab"]
        act = b["act"].clamp_min(0)
        _, ce = smoothed_ce_rows(logits, act, b["mask"], 0.0)
        best = logits.masked_fill(b["mask"] <= 0.5, -1e4).argmax(-1)
        ce_sum += float(ce[lab].sum())
        right += float(((best == b["act"]) & lab).sum())
        n += float(lab.sum())
        real = b["real"]
        t = ppo_core.twohot_targets(b["ret"][real], v_min=v_min, v_max=v_max, v_bins=v_bins)
        v_sum += float(-(t * torch.log_softmax(v_logits[real], dim=-1)).sum(-1).sum())
        v_n += float(real.sum())
    net.train()
    return {"ce": ce_sum / max(1.0, n), "top1": right / max(1.0, n), "vloss": v_sum / max(1.0, v_n), "labelled": int(n)}


def save(path: str, net: torch.nn.Module, cfg: Any, info: Dict[str, Any], step: int, rows_seen: int) -> None:
    payload = {"model": net.state_dict(), "target_model": None, "update_idx": int(step), "total_episodes": 0,
               "total_steps": int(rows_seen), "run_cfg": cfg.as_dict(), "human_imitation": info}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path + ".tmp")
    os.replace(path + ".tmp", path)


def cmd_train(a: argparse.Namespace) -> int:
    import ppo_core
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    torch.set_num_threads(a.threads)
    dev = torch.device("cuda")
    t0 = time.time()
    data = Data(a.data, a.splits, a.salt, threads=a.threads)
    summ = data.summary()
    log("data", json.dumps(summ), f"loaded in {time.time() - t0:.0f} s, obs {data.obs.nbytes / 2 ** 30:.1f} GiB")
    net, cfg = fresh_net(dev, a.seed)
    if a.init:
        # A diagnostic / arm C: start from a checkpoint's weights instead of random ones.
        ck0 = torch.load(a.init, map_location="cpu", weights_only=False)
        state = {k: v for k, v in ck0["model"].items() if "attn_mask" not in k}
        missing, unexpected = net.load_state_dict(state, strict=False)
        learned = {n for n, _ in net.named_parameters()}
        if [k for k in missing if k in learned] or unexpected:
            raise SystemExit(f"{a.init} does not fit the network")
        log("weights from", a.init)
    want_dim = int(sum(e - s for s, e in net.unpacker.offsets.values())) if hasattr(net.unpacker, "offsets") else None
    if data.obs_abilities:
        raise SystemExit(f"the shards were built with obs_abilities={data.obs_abilities!r}: not this branch's observation")
    if data.own_tera_type is not None and bool(data.own_tera_type) != bool(cfg.obs.own_tera_type):
        raise SystemExit(f"the shards have own_tera_type={data.own_tera_type}, the network {cfg.obs.own_tera_type}")
    if want_dim is not None and want_dim != data.obs.shape[1]:
        raise SystemExit(f"the shards' observation has {data.obs.shape[1]} values, the network takes {want_dim}")
    lc, rc = cfg.learner, cfg.reward
    shaping = str(getattr(rc, "shaping", "none")) == "potential"
    if shaping and data.missing_phi:
        raise SystemExit(f"{data.missing_phi} rows have no Phi: the value target needs it (rebuild with this branch's human/rebuild.py)")
    ret = data.returns(float(lc.gamma), float(getattr(rc, "shaping_scale", 0.0)), shaping)

    tr_eps = data.episodes(("train", "train_loose"))
    tr_rows = data.rows(tr_eps)
    w_row = player_weights(data, tr_rows, a.cap)
    w_dev = torch.from_numpy(w_row).to(dev)
    L, S = a.seq, a.seqs_per_batch
    held = pack(data, data.episodes(("test_player",)), L, None)
    held_game = pack(data, data.episodes(("test_game",)), L, None)
    v_min, v_max, v_bins = float(lc.v_min), float(lc.v_max), int(lc.v_bins)

    # All zones but the JEPA predictor train, at the imitation multipliers (backbone 1, actor 1, critic 0.6).
    mults = {"pi": 1.0, "v": 0.6}
    zones = net.get_parameter_zones()
    pairs, groups = [], defaultdict(list)
    for zone, params in zones.items():
        train_zone = zone != "jepa"
        for name, p in params:
            p.requires_grad = train_zone
            if not train_zone:
                continue
            m = p.detach().to(torch.float32).clone()
            pairs.append((p, m))
            nodecay = any(x in name for x in ["_emb", "norm"]) or name.endswith(".bias") or p.ndim == 1
            groups[(mults.get(zone, 1.0), nodecay)].append(m)
    inner = torch.optim.AdamW([{"params": ps, "weight_decay": 0.0 if nd else a.wd, "mult": mult}
                               for (mult, nd), ps in groups.items()], lr=a.lr, eps=1e-5)
    opt = ppo_core.MasterWeightOptimizer(inner, pairs)

    steps_per_pass = len(pack(data, tr_eps, L, np.random.default_rng(0))) // S
    total = int(math.ceil(a.passes * steps_per_pass))
    every = max(1, steps_per_pass // 2)
    log(f"{steps_per_pass} steps a pass, at most {total} steps ({a.passes} passes), held-out every {every} steps; "
        f"value target: gamma {lc.gamma} shaping {'potential x %.2f' % rc.shaping_scale if shaping else 'none'}")
    info = {"code": "human/imitate.py", "from": a.init or "random weights", "seed": a.seed, "lr": a.lr, "wd": a.wd, "warmup": a.warmup,
            "passes": a.passes, "cap": a.cap, "seq": L, "seqs_per_batch": S, "smoothing": a.smoothing, "vf_coef": float(lc.vf_coef),
            "data_dir": a.data, "data": summ, "history": []}
    best = (float("inf"), None)
    bad = 0
    step = rows_seen = 0
    hist: List[Tuple[float, float, float]] = []
    done = False
    epoch = 0
    while not done:
        seqs = pack(data, tr_eps, L, rng)
        rng.shuffle(seqs)
        for j in range(0, len(seqs) - S + 1, S):
            lr = a.lr * min(1.0, (step + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, total))))
            for g in inner.param_groups:
                g["lr"] = lr * g["mult"]
            b = batch(data, ret, seqs[j:j + S], dev)
            logits, v_logits, _v, _a, _b = net(b["obs"], b["ep"], batch_seq_len=L)
            logits, v_logits = logits.float(), v_logits.float()
            lab = b["lab"]
            wl = w_dev[b["rows"].clamp_min(0)] * lab.float()
            act = b["act"].clamp_min(0)
            sce, _ = smoothed_ce_rows(logits, act, b["mask"], a.smoothing)
            pi_loss = (wl * sce).sum() / wl.sum().clamp_min(1e-6)
            real = b["real"]
            t = ppo_core.twohot_targets(b["ret"][real], v_min=v_min, v_max=v_max, v_bins=v_bins)
            v_loss = ppo_core.dist_value_loss(v_logits[real], t)
            loss = pi_loss + float(lc.vf_coef) * v_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.accumulate_grads()
            opt.clip_and_step(a.max_grad_norm)
            with torch.no_grad():
                bst = logits.masked_fill(b["mask"] <= 0.5, -1e4).argmax(-1)
                acc = float(((bst == b["act"]) & lab).sum()) / max(1, int(lab.sum()))
            hist.append((pi_loss.item(), v_loss.item(), acc))
            step += 1
            rows_seen += int(real.sum())
            if step % a.log_every == 0:
                h = np.array(hist[-a.log_every:])
                log(f"step {step}/{total} pass {epoch} lr {lr:.2e} pi {h[:, 0].mean():.4f} v {h[:, 1].mean():.4f} train top1 {h[:, 2].mean():.4f}")
            if step % every == 0 or step >= total:
                m = measure(net, data, ret, held, L, S, dev, v_min, v_max, v_bins)
                rec = {"step": step, "pass": round(step / steps_per_pass, 2), "seconds": round(time.time() - t0),
                       "test_player": m, "train_pi": float(np.mean([x[0] for x in hist[-every:]]))}
                better = m["ce"] < best[0] - 1e-4
                if better:
                    best = (m["ce"], step)
                    bad = 0
                    info["best"] = rec
                    save(os.path.join(a.work, "best.pt"), net, cfg, info, step, rows_seen)
                else:
                    bad += 1
                info["history"].append(rec)
                log(f"held-out players: ce {m['ce']:.4f} top1 {m['top1']:.4f} vloss {m['vloss']:.4f}"
                    + ("  <- best" if better else f"  (no better for {bad})"))
                with open(os.path.join(a.work, "history.json"), "w") as f:
                    json.dump(info["history"], f, indent=1)
                if bad >= a.patience:
                    log(f"stopping: {bad} measurements without a better held-out cross-entropy")
                    done = True
            if step >= total or done:
                done = True
                break
        epoch += 1
    # The arm's result: the best held-out checkpoint, with the test_game measurement added, into the run's checkpoints.
    ck = torch.load(os.path.join(a.work, "best.pt"), map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model"])
    info = ck["human_imitation"]
    info["test_game"] = measure(net, data, ret, held_game, L, S, dev, v_min, v_max, v_bins)
    info["history"] = json.load(open(os.path.join(a.work, "history.json")))
    info["seconds"] = round(time.time() - t0, 1)
    out = os.path.join(a.out_dir, "learner_update_%06d.pt" % int(ck["update_idx"]))
    save(out, net, cfg, info, int(ck["update_idx"]), int(ck["total_steps"]))
    log(f"wrote {out}: best held-out ce {info['best']['test_player']['ce']:.4f} top1 {info['best']['test_player']['top1']:.4f} "
        f"at step {ck['update_idx']}; test_game top1 {info['test_game']['top1']:.4f}; {time.time() - t0:.0f} s")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--data", required=True, help="shards of this branch's human/rebuild.py")
    p.add_argument("--splits", default=SPLITS)
    p.add_argument("--salt", default=SALT)
    p.add_argument("--out-dir", required=True, help="where the result goes as learner_update_N.pt (a run's checkpoints/)")
    p.add_argument("--work", required=True, help="the best checkpoint so far and the held-out history")
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--seqs-per-batch", type=int, default=3)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--passes", type=float, default=16.0)
    p.add_argument("--patience", type=int, default=4, help="held-out measurements (every half pass) without a better one")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--smoothing", type=float, default=0.1)
    p.add_argument("--cap", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--init", default="", help="start from this checkpoint's weights (default: random weights)")
    p.add_argument("--log-every", type=int, default=100)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    return cmd_train(a)


if __name__ == "__main__":
    raise SystemExit(main())
