#!/usr/bin/env python
"""
With every league flag off, is this code what branch topstart was? Run from a code directory, it writes
what that code makes of a fixed set of inputs; run it from topstart's and from league's and the two files
must be equal (bench/notes/LEAGUE.md, section "off by default").

    cd <code> && python bench/league_offcheck.py --streams ~/dev/streams/streams.jsonl --out OUT.json [--battles 60] [--no-gpu]
    python bench/league_offcheck.py --compare A.json B.json

What it writes, for each of run T's, run S's and other argument lists (train.py's parse_args and
build_config): the RunConfig as saved in a checkpoint (as_dict), the lines train.py logs (features,
ppo settings), the opponent seats' plan and its log line, the speed options, the learner keywords; the
observations the training worker's players build on recorded battles (sha256 of every byte) for the
three observations of this line; the worker queue's size; a learner checkpoint's keys and run_cfg; the
seats' log line; check_restart's observation settings; and two PPO updates on recorded observations
(fast_update off and on), their figures and the network's weights afterwards (sha256), on the GPU.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

T_DEFAULT = ["--battles", "2304", "--rooms-per-pair", "16", "--minibatch", "768", "--seq-len", "256", "--grad-accum", "1",
             "--steps-per-update", "36864", "--teacher", "psbot", "--save-every", "200",
             "--obs-matchup", "--obs-movesets", "--infer-actors", "3", "--infer-sync-s", "1", "--rollout-nice", "5"]
ARGVS = {
    "empty": [],
    "T_warmup": ["--mode", "warmup"] + T_DEFAULT + ["--set-head-coef", "0.1"],
    "T_ppo": ["--mode", "ppo"] + T_DEFAULT + ["--set-head-coef", "0.1", "--opp-action-aux-coef", "0.1", "--read-bonus", "0.3",
                                            "--read-anneal", "20000000:45000000", "--opponent-seats", "setupnet:0.15",
                                            "--boosted-row-weight", "2", "--lr-hold-decisions", "357000000",
                                            "--lr-total-decisions", "600000000", "--fast-update", "--fast-ingest"],
    "S_ppo": ["--mode", "ppo", "--obs-matchup", "--obs-movesets", "--set-head-coef", "0.1", "--lr-hold-steps", "479713",
              "--lr-total-steps", "1113566", "--opponent-seats", "setupnet:0.15"],
    "seats_all": ["--mode", "ppo", "--opponent-seats", "setupnet:0.25,psbot:0.25,foulplay:0.25", "--seat-foulplay-iterations", "500"],
    "fp10": ["--mode", "ppo", "--opponent-seats", "foulplay:0.1", "--seat-foulplay-iterations", "88000"],
    "lr_restart": ["--mode", "ppo", "--lr", "3e-05", "--lr-hold-steps", "100", "--lr-total-steps", "200000", "--no-own-tera-type"],
}
PAIRS = {"box_a": [4] * 36, "small": [2, 2, 1], "rehearsal": [4] * 6}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def plan_view(plan):
    return {"workers": [{"seats": w["seats"], "helpers": w["helpers"]} for w in plan["workers"]],
            **{k: plan[k] for k in ("pairs_total", "rooms_per_pair", "wanted", "placed", "net_p1", "net_p2", "helpers",
                                    "helpers_capped", "fp_hosts")}}


def probe_config(out):
    import config
    import infer_pool
    import opponent_seats
    import train
    out["default_as_dict"] = config.RunConfig.default().as_dict()
    res = {}
    for name, argv in ARGVS.items():
        args = train.parse_args(argv)
        cfg = train.build_config(args)
        r = {"as_dict": cfg.as_dict(), "features": train.features_line(cfg), "ppo": train.ppo_settings_line(cfg),
             "speed": dataclasses.asdict(infer_pool.from_args(args)), "learner_speed": train.learner_speed_kwargs(args),
             "ingest": train.fast_ingest_kwargs(args), "nice": train.rollout_nice_of(args)}
        seats = opponent_seats.from_args(args)
        r["seats"] = None if seats is None else repr(seats)
        r["plans"] = {}
        for pname, pairs in PAIRS.items():
            plan = opponent_seats.plan_seats(pairs, 16, seats) if seats else None
            r["plans"][pname] = None if plan is None else {"view": plan_view(plan), "line": opponent_seats.describe_plan(plan)}
        res[name] = r
    out["argvs"] = res


def probe_seats_line(out):
    import opponent_seats as os_
    stats = {k: os_.SeatStats(k).as_dict() for k in os_.SEAT_KINDS}
    stats["setupnet"].update(games_p1=10, wins_p1=6, decisions=100, bot_decisions=7, override_setup=7, setup_played=9)
    stats["foulplay"].update(games_p2=4, wins_p2=1, decisions=50, bot_decisions=40, net_decisions=10, mixed_games=1, handed_timeout=1)
    beat = {"port": 1, "pairs": {"setupnet": 1, "foulplay": 1}, "planned": {"setupnet": 1, "foulplay": 1}, "unavailable": {},
            "failed_start": "", "stats": stats}
    out["seats_line"] = os_.report_line([beat, {"port": 2, "report_error": "x"}])


def probe_worker(out):
    import config
    import worker
    learner = SimpleNamespace(submit_packed_batch=SimpleNamespace(remote=lambda *a, **k: None))
    cl = worker.SyncLearnerClient(learner, config.RunConfig.default())
    out["worker_queue_maxsize"] = cl.q.maxsize
    out["worker_globals"] = {"ZOMBIE_LIMIT_S": worker.ZOMBIE_LIMIT_S, "REPAIR_INTERVAL_S": worker.REPAIR_INTERVAL_S}


def probe_check_restart(out):
    sys.path.insert(0, os.path.join(REPO, "cloud"))
    import check_restart as cr
    import train
    cks = {"none": {}, "tera": {"run_cfg": {"obs": {"own_tera_type": True}}},
           "v2": {"run_cfg": {"obs": {"own_tera_type": True, "matchup": True, "movesets": True}}},
           "main_line_ckpt": {"obs_settings": {"own_tera_type": True, "abil_sets": True}}}
    out["check_restart_ckpt"] = {k: cr.checkpoint_obs_settings(v) for k, v in cks.items()}
    out["check_restart_cfg"] = {k: cr.obs_settings_of_config(train.build_config(train.parse_args(v))) for k, v in ARGVS.items()}


def probe_checkpoint(out, tmp):
    """A learner checkpoint's keys and run_cfg (LearnerActor._save_checkpoint on stand-ins)."""
    import torch
    import config
    import train
    from learner import LearnerActor
    res = {}
    for name in ("T_ppo", "S_ppo", "empty"):
        cfg = train.build_config(train.parse_args(ARGVS[name]))
        fake = SimpleNamespace(net=torch.nn.Linear(2, 2), target_net=torch.nn.Linear(2, 2),
                               opt=SimpleNamespace(state_dict=lambda: {}), sched=None, update_idx=3, total_steps=7,
                               run_cfg=cfg, total_episodes=1, cfg=cfg.learner, read_origin=0, _sync_directory=lambda d: None)
        path = os.path.join(tmp, f"ck_{name}.pt")
        LearnerActor._save_checkpoint(fake, path)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        res[name] = {"keys": sorted(ck), "run_cfg": ck["run_cfg"]}
    out["checkpoint"] = res


def probe_observations(out, streams, battles):
    import config
    from bench import micro_obs
    from poke_env.concurrency import POKE_LOOP
    base = config.RunConfig.default()
    variants = {"own_tera": base.obs, "matchup": dataclasses.replace(base.obs, matchup=True),
                "movesets": dataclasses.replace(base.obs, matchup=True, movesets=True),
                "no_tera": dataclasses.replace(base.obs, own_tera_type=False)}
    msgs = micro_obs.load_streams(streams, battles)
    res, kept = {}, {}
    for name, obs_cfg in variants.items():
        cfg = dataclasses.replace(base, obs=obs_cfg)
        real = config.RunConfig.default
        config.RunConfig.default = classmethod(lambda cls, _c=cfg: _c)
        try:
            _cfg, players, _q = micro_obs.build_players("ppo", "simple", 0, 7)
        finally:
            config.RunConfig.default = real
        rows = []
        for p in players.values():
            inner = p.assembler.assemble

            def assemble(battle, events=None, _inner=inner):
                got = _inner(battle, events=events)
                rows.append(got.copy())
                return got

            p.assembler.assemble = assemble
        asyncio.run_coroutine_threadsafe(micro_obs.replay(players, msgs, defaultdict(micro_obs.Acc)), POKE_LOOP).result()
        h = hashlib.sha256()
        for r in rows:
            h.update(r.tobytes())
        res[name] = {"decisions": len(rows), "sha256": h.hexdigest(), "dim": int(rows[0].shape[0]) if rows else 0}
        kept[name] = rows
    out["observations"] = res
    return kept


def probe_update(out, rows):
    """Two PPO updates (fast_update off, on) on recorded observations, from the same seed: figures and weights."""
    import numpy as np
    import torch
    import config
    import ppo_core
    if not torch.cuda.is_available():
        out["update"] = "no GPU"
        return
    base = config.RunConfig.default()
    cfg = dataclasses.replace(base, obs=dataclasses.replace(base.obs, matchup=True, movesets=True),
                              learner=dataclasses.replace(base.learner, mode="ppo"))
    obs = np.stack(rows[:2048]).astype(np.float16)
    n = obs.shape[0]
    from obs_assembler import ObservationAssembler
    m0, m1 = ObservationAssembler(matchup=True, movesets=True).offsets["action_mask"]
    res = {}
    for fast in (False, True):
        torch.manual_seed(0)
        np.random.seed(0)
        here = os.getcwd()
        os.chdir(REPO)
        try:
            net = cfg.make_model().cuda().train()
        finally:
            os.chdir(here)
        net.enable_bf16_recurrent_path()
        from learner import LearnerActor
        fake = SimpleNamespace(net=net, cfg=cfg.learner, opt=None, sched=None)
        fake._get_zone_config = lambda: LearnerActor._get_zone_config(fake)
        fake._init_scheduler = lambda: None
        LearnerActor._init_optimizer(fake)
        rng = np.random.default_rng(1)
        mask = obs[:, m0:m1].astype(np.float32) > 0.5
        act = np.array([int(rng.choice(np.flatnonzero(m))) if m.any() else 0 for m in mask], dtype=np.int64)
        ds = ppo_core.AsyncEpisodeDataset(act_dim=14, device="cpu")
        ep = np.repeat(np.arange(n // 64), 64)[:n]
        ds.add_steps(torch.from_numpy(obs[:len(ep)]), torch.from_numpy(act[:len(ep)]), torch.from_numpy(rng.normal(-1.5, 0.3, len(ep)).astype(np.float32)),
                     torch.from_numpy(rng.normal(0, 0.5, len(ep)).astype(np.float32)), torch.from_numpy(rng.normal(0, 1, len(ep)).astype(np.float32)),
                     torch.from_numpy(rng.normal(0, 0.5, len(ep)).astype(np.float32)), torch.zeros(len(ep)), torch.from_numpy(ep.astype(np.int64)))
        kw = {"fast_update": True} if fast else {}
        lc = cfg.learner
        st = ppo_core.ppo_update(net=net, opt=fake.opt, dataset=ds, scheduler=None, mode="ppo", target_net=None,
                                 **dict(lc.ppo_kwargs(), minibatch_size=256, batch_seq_len=64), v_min=lc.v_min, v_max=lc.v_max,
                                 v_bins=lc.v_bins, **kw)
        h = hashlib.sha256()
        for k, v in sorted(net.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().float().cpu().numpy().tobytes())
        res["fast" if fast else "plain"] = {"figures": {k: getattr(st, k) for k in ("approx_kl", "clip_frac", "entropy", "v_loss",
                                                                                      "pg_loss", "total_loss", "n_mb", "acc", "aux")},
                                            "weights_sha256": h.hexdigest()}
    out["update"] = res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--streams", default=os.path.expanduser("~/dev/streams/streams.jsonl"))
    ap.add_argument("--battles", type=int, default=60)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--compare", nargs=2, default=None)
    a = ap.parse_args()
    if a.compare:
        x, y = (json.load(open(p, encoding="utf-8")) for p in a.compare)
        bad = []

        def walk(p, u, v):
            if isinstance(u, dict) and isinstance(v, dict):
                for k in sorted(set(u) | set(v)):
                    walk(f"{p}.{k}", u.get(k, "<absent>"), v.get(k, "<absent>"))
            elif u != v:
                bad.append(f"{p}: {str(u)[:160]} | {str(v)[:160]}")

        walk("", {k: x[k] for k in x if k != "code"}, {k: y[k] for k in y if k != "code"})
        print("\n".join(bad) if bad else f"EQUAL ({len(x) - 1} probes)")
        return 1 if bad else 0
    os.chdir(REPO)
    import tempfile
    out = {"code": REPO}
    probe_config(out)
    probe_seats_line(out)
    probe_worker(out)
    probe_check_restart(out)
    with tempfile.TemporaryDirectory() as tmp:
        probe_checkpoint(out, tmp)
    rows = probe_observations(out, a.streams, a.battles)
    if not a.no_gpu:
        probe_update(out, rows["movesets"])
    text = json.dumps(out, indent=1, sort_keys=True, default=str)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(f"wrote {a.out or 'stdout'}: {len(out)} probes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
