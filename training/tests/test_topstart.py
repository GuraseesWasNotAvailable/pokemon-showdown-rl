"""
human/topstart.py (run T's start: offline imitation of the top players from random weights) and
cloud/v2/topstart_setup.py (the run that continues it), plus check_restart's --phase.

    nice -n 10 python -m pytest tests/test_topstart.py -q -p no:cacheprovider -rs -s
    PSPPO_TOPSTART_DATA=<dir with 40k/, bulk/ and imitation_data.json>   (real shards; GPU tests need them)
    PSPPO_TOPSTART_HEAD=<a copy of the commit before (4cfd496)>           (check_restart without --phase is that commit's)
    PSPPO_TEST_DIR=<where the tests write>                                (default: a new temp directory; nothing is removed)

CPU (synthetic shards in human/rebuild.py's layout, a stand-in network):
  * (e) the schedule as pure functions: warmup, hold, the plateau trigger (min hold, window, the
    best so far), the latest start, the cosine to 0, the stop step, the abort rule, C0 from step 0
  * the value target: ep_won 1 / 0 / -1 is +1 / -1 / 0 (a loss is a loss), Phi's NaN is 0, one value a row
  * (a) the link recipe never links a skip shard; one player in both sets is capped over both:
    weight min(1, 0.03 N / count over both)
  * (b) an episode with a NaN is dropped at load and counted, the value targets stay aligned
  * (c) a shard of another observation (or one without the fields the start needs) is refused, exit 2
  * the measure (evaluate) gives HumanImitation.holdout_agreement's top-1, per-player top-1 and CE
    on the same sequences; its per-set and per-period parts add up to the whole
  * the optimizer is the learner's for mode imitation (groups, rates, decay, the JEPA zone frozen)
  * (h) topstart_setup: run.json from S's, the supervisor takes it, check_restart --phase warmup says
    the checkpoint is of phase 'imitation' and the start 'warmup' (exit 0); every refusal fires;
    the copy of the start, the units, the rehearsal
  * check_restart without --phase: the commit before's output, line for line (PSPPO_TOPSTART_HEAD)
Real shards (PSPPO_TOPSTART_DATA):
  * (a) one full pass of the small set: every gathered row comes from an ep_split == train side; the
    skip shard is never linked or opened
  * (c) the network takes 10,295 values and has the SetHead
GPU + real shards:
  * (d) 300 steps on 2 shards: train top-1 10 points above the 0.23 prior, SetHead and value losses fall;
    no player above 3% of the capped held-out set's target
  * (f) 2 groups, stop, resume: the same batch ids at every step and the same step count as an
    uninterrupted run; nothing written before is overwritten
  * (g) the start through the learner (no Ray): mode imitation, the settings, set_head.*, no optimizer;
    warmup loads it as a phase change with every weight equal; a warmup update leaves set_head.* as
    it was; PPO loads the warmup save with set_head equal; without --set-head-coef it is dropped
  * (b) a real shard with a NaN in one episode: dropped, counted, no step skipped
  * (i) measure writes every field; no player or game hash in any output, log or checkpoint
  * (j) steps a second and the loader's keep-up on 12 shards, under ~/dev/fast/bench.lock (reported)
"""
from __future__ import annotations

import asyncio
import dataclasses
import glob
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "tests"), os.path.join(REPO, "cloud"), os.path.join(REPO, "cloud", "v2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(REPO)

import human_imitation as hi  # noqa: E402
from human import topstart as ts  # noqa: E402
import topstart_setup as tss  # noqa: E402

GPU = torch.cuda.is_available()
needs_gpu = pytest.mark.skipif(not GPU, reason="flex_attention has no backward on a CPU")
DATA = os.path.expanduser(os.environ.get("PSPPO_TOPSTART_DATA", ""))
HAVE_DATA = bool(DATA) and os.path.isfile(os.path.join(DATA, "imitation_data.json"))
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="PSPPO_TOPSTART_DATA names no dir with 40k/, bulk/ and imitation_data.json")
HEAD = os.path.expanduser(os.environ.get("PSPPO_TOPSTART_HEAD", ""))
ROOT = os.environ.get("PSPPO_TEST_DIR") or tempfile.mkdtemp(prefix="topstart_test_")
os.makedirs(ROOT, exist_ok=True)
PY = sys.executable

D = 140
TRANS0 = 20
A0, A1 = D - 14, D
OFFSETS = {"pokemon_body": (0, 20), "transition_scalars": (TRANS0, TRANS0 + 2 * hi.SIDE_SCALAR_DIM),
           "global_scalars": (TRANS0 + 2 * hi.SIDE_SCALAR_DIM, A0), "action_mask": (A0, A1)}
T2025, T2026A, T2026B = ts._utc(2025, 8), ts._utc(2026, 3), ts._utc(2026, 9)


def _dir(name):
    d = os.path.join(ROOT, name)
    k = 1
    while os.path.exists(d):
        d = os.path.join(ROOT, f"{name}.{k}")
        k += 1
    os.makedirs(d)
    return d


# =============================================================================================
# synthetic shards
# =============================================================================================

def _ep(n, rng, game, player, side, split, won=1, t=T2026B, nan_row=None, phi=None):
    obs = np.zeros((n, D), np.float16)
    mask = np.zeros((n, 14), np.uint8)
    mask[:, 0:4] = 1
    mask[:, 4:6] = 1
    obs[:, A0:A1] = mask
    if nan_row is not None:
        obs[nan_row, 3] = np.nan
    act = rng.integers(0, 6, n).astype(np.int16)
    ph = (np.linspace(-0.3, 0.3, n).astype(np.float32) if phi is None else np.asarray(phi, np.float32))
    return dict(obs=obs, act=act, mask=mask, train=np.ones(n, np.uint8), game=game, player=player, side=side, split=split,
                won=won, time=t, phi=ph)


def _write(path, eps, own_tera=True, matchup=True, movesets=True, abil="", obs_dim=D, drop=()):
    starts = np.cumsum([0] + [len(e["act"]) for e in eps]).astype(np.int64)
    n = int(starts[-1])
    lab = np.zeros((n, 48), np.int32)
    lab[:, 0] = 1 + (np.arange(n) % 50)
    kw = dict(obs=np.concatenate([e["obs"] for e in eps]), act=np.concatenate([e["act"] for e in eps]),
              mask=np.concatenate([e["mask"] for e in eps]), train=np.concatenate([e["train"] for e in eps]),
              phi=np.concatenate([e["phi"] for e in eps]), set_labels=lab, ep_start=starts,
              ep_game=np.array([e["game"] for e in eps]), ep_player=np.array([e["player"] for e in eps]),
              ep_side=np.array([e["side"] for e in eps], np.int8), ep_split=np.array([e["split"] for e in eps]),
              ep_won=np.array([e["won"] for e in eps], np.int8), ep_time=np.array([e["time"] for e in eps], np.int64),
              own_tera_type=np.array(own_tera), obs_matchup=np.array(matchup), obs_movesets=np.array(movesets),
              obs_abilities=np.array(abil), obs_dim=np.array(obs_dim))
    for k in drop:
        kw.pop(k)
    np.savez_compressed(path, **kw)


def _synthetic(d, nan=False, **flags):
    """Two sets in one directory (as linked): player "both" plays in each; held-out players, loose and test_game sides."""
    rng = np.random.default_rng(3)
    a, b = [], []
    for g in range(6):
        a.append(_ep(30, rng, f"g{g:012d}", "both" if g < 3 else f"pa{g}", 1, "train", won=g % 2, t=T2026B,
                     nan_row=(5 if (nan and g == 1) else None)))
        a.append(_ep(20, rng, f"g{g:012d}", f"qa{g}", 2, "loose" if g == 5 else "train", won=1 - g % 2, t=T2026B))
    a.append(_ep(25, rng, "gh00000000001", "held1", 1, "test_player", won=1, t=T2026B))
    a.append(_ep(25, rng, "gt00000000001", "tg1", 1, "test_game", won=0, t=T2026B))
    for g in range(4):
        b.append(_ep(40, rng, f"gb{g:011d}", "both", 2, "train", won=-1 if g == 0 else 1, t=T2025))
        b.append(_ep(10, rng, f"gb{g:011d}", f"pb{g}", 1, "train", won=0, t=T2026A))
    b.append(_ep(30, rng, "gh00000000002", "held2", 2, "test_player", won=0, t=T2025))
    b.append(_ep(15, rng, "gt00000000002", "tg2", 2, "test_game", won=1, t=T2026A))
    _write(os.path.join(d, "shard-40k-00000-aaaaaaaaaa.npz"), a, **flags)
    _write(os.path.join(d, "shard-bulk-00000-bbbbbbbbbb.npz"), b, **flags)
    return d


def _topdata(d, **kw):
    return ts.TopData(d, OFFSETS, seed=1, holdout_decisions=kw.pop("holdout_decisions", 64), threads=2, **kw)


# =============================================================================================
# (e) the schedule
# =============================================================================================

def test_the_schedule_as_pure_functions():
    s = ts.Schedule()
    assert (s.warmup, s.min_hold, s.plateau_window, s.plateau_gain, s.anneal_steps, s.max_steps) == (1000, 60000, 20000, 0.005, 25000, 150000)
    assert s.anneal_len() == 25_000 and s.latest_start() == 125_000
    # warmup and hold
    assert ts.lr_share(0, s, None) == pytest.approx(1 / 1000)
    assert ts.lr_share(499, s, None) == pytest.approx(0.5)
    assert ts.lr_share(999, s, None) == 1.0 and ts.lr_share(100_000, s, None) == 1.0
    # the cosine from its start to 0 at its end
    st0 = 80_000
    assert ts.lr_share(st0, s, st0) == 1.0
    assert ts.lr_share(st0 + 12_500, s, st0) == pytest.approx(0.5)
    assert ts.lr_share(st0 + 25_000, s, st0) == pytest.approx(0.0, abs=1e-12)
    assert all(ts.lr_share(k, s, st0) >= ts.lr_share(k + 1, s, st0) for k in range(st0, st0 + 25_000, 97))
    assert ts.stop_step(st0, s) == 105_000 and ts.stop_step(None, s) is None
    # the trigger: not before min_hold; the gain of the best so far over the window
    rising = [(k, 0.40 + 0.01 * k / 2000) for k in range(2000, 100_001, 2000)]
    assert not any(ts.plateau_reached(rising[:i + 1], r[0], s) for i, r in enumerate(rising))
    flat = [(k, min(0.50, 0.30 + 0.01 * k / 2000)) for k in range(2000, 100_001, 2000)]       # 0.50 from step 40,000 on
    hits = [r[0] for i, r in enumerate(flat) if ts.plateau_reached(flat[:i + 1], r[0], s)]
    assert hits[0] == 60_000                    # min_hold: flat from 40k, but not before 60k
    early = [(k, 0.5) for k in range(2000, 60_001, 2000)]
    assert ts.plateau_reached(early, 60_000, s) and not ts.plateau_reached(early, 58_000, s)
    # one noisy dip does not start it, one noisy peak does not delay it (the best so far)
    noisy = [(k, 0.4 + 0.004 * k / 2000) for k in range(2000, 80_001, 2000)]
    noisy[-1] = (80_000, 0.30)
    assert not ts.plateau_reached(noisy, 80_000, s)
    # a gain just under 0.5 points over 20,000 steps is a plateau, just over is not
    assert ts.plateau_reached([(40_000, 0.500), (60_000, 0.5049)], 60_000, s)
    assert not ts.plateau_reached([(40_000, 0.500), (60_000, 0.5051)], 60_000, s)
    assert ts.anneal_start_after_measure(60_000, None, [(40_000, 0.5), (60_000, 0.5)], s) == 60_000
    assert ts.anneal_start_after_measure(62_000, 60_000, [(40_000, 0.5), (62_000, 0.5)], s) == 60_000
    # the latest start, before the step of that index
    assert ts.anneal_start_before_step(124_999, None, s) is None
    assert ts.anneal_start_before_step(125_000, None, s) == 125_000
    assert ts.anneal_start_before_step(125_000, 90_000, s) == 90_000
    assert ts.stop_step(125_000, s) == 150_000
    # the abort rule: the first measurement at or after the step, below the bar
    ab = ts.parse_abort("40000:0.30")
    assert ab == (40_000, 0.30) and ts.parse_abort("off") is None
    assert not ts.abort_due(38_000, 0.10, ab, False)
    assert ts.abort_due(40_000, 0.29, ab, False) and not ts.abort_due(40_000, 0.31, ab, False)
    assert not ts.abort_due(42_000, 0.10, ab, True)
    # the best: top-1 decides, within 0.2 points the lower CE
    assert ts.better({"top1": 0.55, "ce": 1.2}, {"top1": 0.54, "ce": 1.0})
    assert ts.better({"top1": 0.541, "ce": 1.0}, {"top1": 0.542, "ce": 1.1})
    assert not ts.better({"top1": 0.543, "ce": 1.2}, {"top1": 0.542, "ce": 1.1})
    assert not ts.better({"top1": 0.535, "ce": 0.1}, {"top1": 0.542, "ce": 1.1})
    # C0: warmup 100, min hold 0, max steps of 3 passes (here 21,000): the cosine runs from step 0 to the end
    c0 = ts.Schedule(warmup=100, min_hold=0, max_steps=21_000)
    assert c0.anneal_len() == 21_000 and c0.latest_start() == 0
    a0 = ts.anneal_start_before_step(0, None, c0)
    assert a0 == 0 and ts.stop_step(a0, c0) == 21_000
    assert ts.lr_share(0, c0, a0) == pytest.approx(0.01)
    assert ts.lr_share(10_500, c0, a0) == pytest.approx(0.5)
    assert ts.lr_share(20_999, c0, a0) < 1e-7


def test_the_value_target_reads_ep_won_as_rebuild_writes_it():
    assert [ts.outcome_z(w) for w in (1, 0, -1)] == [1.0, -1.0, 0.0]
    phi = np.array([0.2, np.nan, -0.4], np.float32)
    g = ts.value_targets(phi, -1.0, 0.999, 0.5, True)
    assert g.dtype == np.float32 and len(g) == 3
    assert g[0] == pytest.approx(-(0.999 ** 2) - 0.1) and g[1] == pytest.approx(-0.999) and g[2] == pytest.approx(-1.0 + 0.2)
    assert np.allclose(ts.value_targets(phi, 1.0, 0.999, 0.5, False), [0.999 ** 2, 0.999, 1.0])


# =============================================================================================
# (a) (b) (c) with synthetic shards
# =============================================================================================

def test_link_recipe_links_read_shards_only(tmp_path):
    a, b, dst = tmp_path / "A", tmp_path / "B", tmp_path / "ALL"
    a.mkdir(), b.mkdir()
    for n in ("shard-00000-c1.npz", "shard-00001-c1.npz"):
        (a / n).write_bytes(b"x")
    for n in ("shard-00000-c2.npz", "shard-00007-c3.npz"):
        (b / n).write_bytes(b"y")
    man = {"sets": [{"name": "40k", "read_shards": ["shard-00000-c1.npz", "shard-00001-c1.npz"], "skip_shards": []},
                    {"name": "bulk", "read_shards": ["shard-00000-c2.npz"], "skip_shards": ["shard-00007-c3.npz"]}]}
    (tmp_path / "m.json").write_text(json.dumps(man))
    assert ts.main(["link", "--manifest", str(tmp_path / "m.json"), "--set", f"40k={a}", "--set", f"bulk={b}", "--dst", str(dst)]) == 0
    names = sorted(os.listdir(dst))
    assert names == ["shard-40k-00000-c1.npz", "shard-40k-00001-c1.npz", "shard-bulk-00000-c2.npz"]
    assert all(os.path.realpath(dst / n).startswith(str(tmp_path)) for n in names)
    assert not any("00007" in n for n in names)
    # again: nothing new, nothing replaced
    assert ts.main(["link", "--manifest", str(tmp_path / "m.json"), "--set", f"40k={a}", "--set", f"bulk={b}", "--dst", str(dst)]) == 0
    # a name in both lists, or a missing read shard, is refused
    man["sets"][1]["skip_shards"].append("shard-00000-c2.npz")
    (tmp_path / "m2.json").write_text(json.dumps(man))
    with pytest.raises(SystemExit) as e:
        ts.main(["link", "--manifest", str(tmp_path / "m2.json"), "--set", f"40k={a}", "--set", f"bulk={b}", "--dst", str(dst)])
    assert e.value.code == 2
    (a / "shard-00001-c1.npz").rename(tmp_path / "aside.npz")
    with pytest.raises(SystemExit) as e:
        ts.main(["link", "--manifest", str(tmp_path / "m.json"), "--set", f"40k={a}", "--set", f"bulk={b}", "--dst", str(tmp_path / "ALL2")])
    assert e.value.code == 2


def test_roles_and_the_player_cap_over_both_sets(tmp_path):
    d = _synthetic(str(tmp_path))
    td = _topdata(d)
    idx = td.h.index
    # train sides only: "loose", "test_game" and "test_player" are never train
    train_eps = {(os.path.basename(f), a) for f, e in idx["episodes"].items() for a, _b, _p in e["train"]}
    assert len(train_eps) == 6 + 5 + 8 and idx["roles"]["train"] == 19
    n = idx["train_labelled"]
    assert n == 280 + 200
    both = 3 * 30 + 4 * 40
    assert idx["player_lab"]["both"] == both                       # counted over both sets
    assert idx["player_w"]["both"] == pytest.approx(min(1.0, 0.03 * n / both))
    for p, c in idx["player_lab"].items():
        assert idx["player_w"][p] == pytest.approx(min(1.0, 0.03 * n / c))
    # the loaded rows carry exactly that weight, on labelled rows only
    shards = td.load_group(td.files)
    for sh in shards:
        for k, (a, b) in enumerate(sh.eps):
            w = idx["player_w"].get(sh.players[k], 1.0)
            assert np.allclose(sh.w[a:b][sh.lab[a:b]], w)
    # test_game: its own selection; test_player: the held-out set
    assert sum(len(v) for v in td.test_game.values()) == 2
    assert td.h.ho["n"] > 0 and td.counts()["test_player_sides"] == 2
    assert td.counts()["train_ep_won"] == {"-1": 1, "0": 9, "1": 9}


def test_an_episode_with_a_nan_is_dropped_and_counted_and_the_targets_stay_aligned(tmp_path):
    d = _synthetic(str(tmp_path), nan=True)
    td = _topdata(d)
    shards = td.load_group(td.files)
    assert td.dropped == 1
    rows = sum(sh.rows() for sh in shards)
    assert rows == (6 * 30 + 5 * 20 - 30) + (4 * 40 + 4 * 10)
    for sh in shards:
        assert len(sh.ret) == sh.rows() and np.isfinite(sh.ret).all() and np.isfinite(sh.obs.astype(np.float32)).all()
        with np.load(sh.path) as z:
            st = z["ep_start"]
            for k, (a, b) in enumerate(sh.ep_src):
                e = int(np.searchsorted(st, a, side="right") - 1)
                la, lb = sh.eps[k]
                want = ts.value_targets(z["phi"][a:b], ts.outcome_z(int(z["ep_won"][e])), 0.999, 0.5, True)
                assert np.array_equal(sh.ret[la:lb], want)
                assert np.array_equal(sh.obs[la:lb], z["obs"][a:b])
    # loading the same shards again does not count the episode again
    td.load_group(td.files)
    assert td.dropped == 1


@pytest.mark.parametrize("flags", [dict(matchup=False, movesets=False), dict(movesets=False), dict(own_tera=False),
                                   dict(abil="v1"), dict(obs_dim=D + 1)])
def test_a_shard_of_another_observation_is_refused(tmp_path, flags):
    d = _synthetic(str(tmp_path))
    rng = np.random.default_rng(0)
    _write(os.path.join(d, "shard-bulk-00001-cccccccccc.npz"), [_ep(10, rng, "gzz000000000", "x", 1, "train")], **flags)
    with pytest.raises(SystemExit) as e:
        _topdata(d)
    assert e.value.code == 2


@pytest.mark.parametrize("field", ["phi", "ep_won", "ep_time", "set_labels"])
def test_a_shard_without_a_needed_field_is_refused(tmp_path, field):
    d = _synthetic(str(tmp_path))
    rng = np.random.default_rng(0)
    _write(os.path.join(d, "shard-bulk-00001-cccccccccc.npz"), [_ep(10, rng, "gzz000000000", "x", 1, "train")], drop=(field,))
    with pytest.raises(SystemExit) as e:
        _topdata(d)
    assert e.value.code == 2


def test_an_empty_or_missing_directory_is_refused(tmp_path):
    with pytest.raises(SystemExit) as e:
        _topdata(str(tmp_path / "nothing"))
    assert e.value.code == 2


# =============================================================================================
# the measure against holdout_agreement
# =============================================================================================

class _Unpacker:
    offsets = OFFSETS


class TinyNet(torch.nn.Module):
    """net(obs, ep, batch_seq_len[, return_z]) -> (logits, v_logits over 51 bins, ...), as the learner's network."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lin = torch.nn.Linear(D, 14)
        self.v = torch.nn.Linear(D, 51)
        self.unpacker = _Unpacker()

    def forward(self, obs, ep, batch_seq_len=256, return_z=False):
        x = obs.float() + torch.arange(D, dtype=torch.float32) * 0.01 * (ep % 7)[:, None].float()
        return self.lin(x), self.v(x), None, None, None


def test_the_measure_is_holdout_agreements_and_its_parts_add_up(tmp_path):
    d = _synthetic(str(tmp_path))
    td = _topdata(d, holdout_decisions=10_000)
    net = TinyNet()
    tally = ts.evaluate(net, td.holdout_seqs, "cpu")
    r = tally.results()
    td.h.holdout_agreement(net, "cpu", torch.float16, 7)
    assert r["all"]["top1"] == pytest.approx(td.h.ho["agree"], abs=1e-12)
    assert r["all"]["top1_per_player"] == pytest.approx(td.h.ho["agree_pp"], abs=1e-12)
    assert r["all"]["ce"] == pytest.approx(td.h.ho["ce"], rel=1e-5)
    assert r["all"]["labelled"] == td.h.ho["n"] == 55 and r["all"]["players"] == 2
    assert math.isfinite(r["all"]["vloss"])
    for prefix in ("set:", "period:"):
        parts = [v for k, v in r.items() if k.startswith(prefix)]
        assert sum(p["labelled"] for p in parts) == r["all"]["labelled"]
    assert r["set:40k"]["labelled"] == 25 and r["set:bulk"]["labelled"] == 30
    assert r["period:2026-07+"]["labelled"] == 25 and r["period:2025"]["labelled"] == 30
    assert ts.period_of(T2025) == "2025" and ts.period_of(T2026A) == "2026-01..06" and ts.period_of(T2026B) == "2026-07+"
    assert ts.set_of("/x/shard-40k-00012-2f5b273925.npz") == "40k" and ts.set_of("shard-bulk-00368-2990bc79e2.npz") == "bulk"
    assert ts.set_of("shard-00012-2f5b273925.npz") == "other"


# =============================================================================================
# the capped held-out set samples both sets (review finding: in name order it was 76% 40k games)
# =============================================================================================

def _many_holdout_shards(d, n40k=4, nbulk=8, eps_per_shard=4, rows=25):
    """Shards of two sets in one directory, each with held-out episodes of players seen nowhere else."""
    rng = np.random.default_rng(11)
    for name, n, t in (("40k", n40k, T2026B), ("bulk", nbulk, T2025)):
        for k in range(n):
            eps = [_ep(30, rng, f"t{name}{k:03d}000000"[:13], f"tr_{name}_{k}", 1, "train", t=t)]
            for e in range(eps_per_shard):
                eps.append(_ep(rows, rng, f"h{name}{k:03d}{e:02d}0000"[:13], f"ho_{name}_{k}_{e}", 2, "test_player", t=t))
            _write(os.path.join(d, f"shard-{name}-{k:05d}-{k:010d}.npz"), eps)
    return d


def test_the_held_out_order_is_a_fixed_permutation_of_the_names():
    names = [f"/d/shard-40k-{k:05d}-aaaaaaaaaa.npz" for k in range(60)] + \
            [f"/d/shard-bulk-{k:05d}-bbbbbbbbbb.npz" for k in range(140)]
    a = ts.order_holdout_shards(names)
    assert sorted(a) == sorted(names) and a != sorted(names)
    rev = list(reversed(names))
    assert ts.order_holdout_shards(rev) == a                    # the listing order does not matter, the names do
    assert ts.order_holdout_shards(names, "spread") == names    # the old order: as listed (40k first)
    head = a[:40]
    share = sum("shard-40k-" in p for p in head) / len(head)
    assert 0.15 <= share <= 0.45, share                         # 30% of the names are 40k
    with pytest.raises(SystemExit):
        ts.order_holdout_shards(names, "sorted")


def test_the_capped_held_out_set_samples_both_sets_not_the_first_names(tmp_path):
    d = _many_holdout_shards(str(tmp_path), n40k=10, nbulk=20, eps_per_shard=2)
    # 30 shards of 50 held-out decisions, a third of them 40k; a 600-decision set takes about 12 shards
    old = _topdata(d, holdout_decisions=600, holdout_order="spread")
    new = _topdata(d, holdout_decisions=600)
    ho_old, ho_new = old.counts()["holdout"], new.counts()["holdout"]
    share = lambda ho: ho["by_set"].get("40k", 0) / ho["decisions"]
    # the old order (what the review found): the first names, so the 40k set first (500 of 600 here)
    assert ho_old["order"] == "spread" and share(ho_old) >= 0.8, ho_old
    # the fixed permutation: both sets near their third and two thirds, both periods
    assert ho_new["order"] == "shuffled"
    assert 0 < share(ho_new) <= 0.6 and ho_new["by_set"].get("bulk", 0) > 0, ho_new
    assert set(ho_new["by_period"]) == {"2025", "2026-07+"}, ho_new
    for ho in (ho_old, ho_new):
        assert sum(ho["by_set"].values()) == sum(ho["by_period"].values()) == ho["decisions"]
        assert ho["digest"].startswith("ho") and ho["digest"][2:].isdigit()     # digits only: no hex run in a log
    assert ho_old["digest"] != ho_new["digest"]
    # the same data gives the same set (every arm and every measure), and the index is left in its own order
    again = _topdata(d, holdout_decisions=600)
    assert again.counts()["holdout"]["digest"] == ho_new["digest"]
    assert new.h.index["holdout_shards"] == old.h.index["holdout_shards"] == sorted(new.h.index["holdout_shards"])


def test_a_resume_file_from_another_held_out_set_is_refused():
    now = {"seed": 1, "holdout_order": "shuffled", "holdout_digest": "ho-aaaa"}
    before_option = {"seed": 1}                                  # a resume file written before --holdout-order (spread)
    assert ts._same_run(before_option, now) == ["holdout_order"]
    assert ts._same_run(before_option, dict(now, holdout_order="spread", holdout_digest="ho-bbbb")) == []
    assert ts._same_run(dict(now), dict(now)) == []
    assert ts._same_run(dict(now, holdout_digest="ho-bbbb"), now) == ["holdout_digest"]


# =============================================================================================
# the optimizer is the learner's
# =============================================================================================

def test_the_optimizer_is_the_learners_for_mode_imitation():
    import test_speed_update as tsu
    import learner as new_learner
    import ppo_core as new_core
    cfg = ts.run_config()
    cfg = dataclasses.replace(cfg, learner=dataclasses.replace(cfg.learner, device="cpu", resume=False,
                                                               ckpt_dir=os.path.join(ROOT, "ckpt_unused")))
    la = tsu._make_learner(new_learner, new_core, cfg)
    net = ts.build_net(cfg, 1, "cpu", set_head=True)
    opt = ts.zone_groups(net, 3e-4, 0.01)
    mine = [(g["name"], len(g["params"]), g["lr"], g["weight_decay"]) for g in opt.param_groups]
    theirs = [(g["name"], len(g["params"]), g.get("initial_lr", g["lr"]), g["weight_decay"]) for g in la.opt.param_groups]
    assert mine == theirs, (mine, theirs)
    assert any(n.startswith("aux_") for n, *_ in mine) and not any(n.startswith("jepa") for n, *_ in mine)
    assert {n for n, p in net.named_parameters() if not p.requires_grad} == {n for n, p in la.net.named_parameters() if not p.requires_grad}
    assert sum(len(g["params"]) for g in opt.param_groups) == len(opt.pairs)


# =============================================================================================
# (h) topstart_setup and check_restart --phase
# =============================================================================================

def _fake_start(path, **over):
    """A start checkpoint as human/topstart.py train writes it (a small model: what the setup and the check read)."""
    cfg = ts.run_config()
    model = {"set_head.ability.weight": torch.zeros(2, 2), "pi_head.weight": torch.zeros(2, 2)}
    ck = {"model": model, "target_model": None, "update_idx": 0, "total_episodes": 0, "total_steps": 123,
          "run_cfg": cfg.as_dict(), "topstart": {"x": 1}}
    for k, v in over.items():
        if v is None:
            ck.pop(k, None)
        else:
            ck[k] = v
    torch.save(ck, path)
    return path


def _fill_cache(rel, species=150, battle=True):
    """heuristic/cache as a release copied from ~/rel/v2-humanaux has it (a set table and PSBot's battle.js)."""
    cache = os.path.join(str(rel), "heuristic", "cache")
    os.makedirs(cache, exist_ok=True)
    with open(os.path.join(cache, "gen9randombattle.json"), "w", encoding="utf-8") as f:
        json.dump({f"mon{k}": {"level": 80, "abilities": ["Levitate"]} for k in range(species)}, f)
    if battle:
        with open(os.path.join(cache, "battle.js"), "w", encoding="utf-8") as f:
            f.write("// client\n")


def _has_flags(*flags):
    import train
    try:
        train.parse_args(list(flags))
        return True
    except SystemExit:
        return False


def _has_decision_flags():
    import train
    try:
        train.parse_args(["--lr-hold-decisions", "1", "--lr-total-decisions", "2"])
        return True
    except SystemExit:
        return False


def test_the_run_json_is_s_with_the_start_changed_and_the_supervisor_takes_it(tmp_path):
    import pipeline
    cfg = tss.build_run_json(tss.S_LIVE, "/home/ubuntu/rel/topstart-x")
    assert cfg["repo_dir"] == "/home/ubuntu/rel/topstart-x"
    assert cfg["servers"] == dict(tss.S_LIVE["servers"], base_port=10700)
    assert cfg["eval"]["port"] == 10790 and cfg["eval"]["foulplay"] == tss.S_LIVE["eval"]["foulplay"]
    assert cfg["train_defaults"]["save_every"] == 200 and cfg["train_defaults"]["extra_train_args"] == tss.DEFAULT_ARGS
    assert cfg["phases"]["imitation"] == {"skip": True, "min_decisions": 0, "max_decisions": 1}
    assert cfg["phases"]["warmup"]["extra_train_args"] == ["--set-head-coef", "0.1"]
    assert {k: v for k, v in cfg["phases"]["warmup"].items() if k != "extra_train_args"} == tss.S_LIVE["phases"]["warmup"]
    assert cfg["phases"]["ppo"]["extra_train_args"] == [
        "--set-head-coef", "0.1", "--opp-action-aux-coef", "0.1", "--read-bonus", "0.3", "--read-anneal", "20000000:45000000",
        "--opponent-seats", "setupnet:0.15", "--boosted-row-weight", "2", "--lr-hold-decisions", "357000000",
        "--lr-total-decisions", "600000000"]
    assert cfg["phases"]["ppo"]["target_decisions"] == 600_000_000
    for k in ("watchdog", "disk", "trainer", "python", "showdown_dir", "on_done", "on_failed"):
        assert cfg[k] == tss.S_LIVE[k]
    assert cfg["retention"]["enabled"] is False
    p = tmp_path / "run.json"
    p.write_text(json.dumps(cfg))
    got = pipeline.load_config(str(p))
    assert pipeline.first_phase(got) == "warmup" and got["phases"]["warmup"]["mode"] == "warmup"
    assert pipeline.phase_ports(got, got["phases"]["ppo"])[0] == 10700
    # the rehearsal
    r = tss.build_run_json(tss.S_LIVE, "/rel", rehearsal=True)
    assert r["servers"]["count"] == 4 and r["servers"]["base_port"] == 10800 and r["train_defaults"]["battles"] == 256
    assert (r["phases"]["warmup"]["min_decisions"], r["phases"]["warmup"]["max_decisions"]) == (100_000, 200_000)
    assert r["phases"]["ppo"]["target_decisions"] == 2_000_000 and "eval" not in r
    assert r["phases"]["ppo"]["extra_train_args"][-4:] == ["--lr-hold-decisions", "1000000", "--lr-total-decisions", "2000000"]
    p.write_text(json.dumps(r))
    pipeline.load_config(str(p))
    # the rehearsal at the real run's size, on other ports: only the lengths, the schedule's points and the ports differ
    rf = tss.build_run_json(tss.S_LIVE, "/rel", rehearsal=True, base_port=11700, rehearsal_full=True)
    assert rf["servers"] == dict(tss.S_LIVE["servers"], base_port=11700) and "eval" not in rf
    assert rf["train_defaults"] == cfg["train_defaults"]
    assert (rf["phases"]["warmup"]["min_decisions"], rf["phases"]["warmup"]["max_decisions"]) == (100_000, 200_000)
    assert rf["phases"]["ppo"]["target_decisions"] == 745_000 and rf["phases"]["ppo"]["train"] == {"save_every": 10}
    assert rf["phases"]["ppo"]["extra_train_args"] == cfg["phases"]["ppo"]["extra_train_args"][:-4] + [
        "--lr-hold-decisions", "300000", "--lr-total-decisions", "600000"]
    p.write_text(json.dumps(rf))
    got = pipeline.load_config(str(p))
    assert pipeline.phase_ports(got, got["phases"]["ppo"])[0] == 11700
    with pytest.raises(tss.Refusal):
        tss.build_run_json(tss.S_LIVE, "/rel", rehearsal_full=True)
    # --base-port moves the evaluator with the servers; without it the ports are the plan's
    moved = tss.build_run_json(tss.S_LIVE, "/rel", base_port=11000)
    assert moved["servers"]["base_port"] == 11000 and moved["eval"]["port"] == 11090
    assert tss.build_run_json(tss.S_LIVE, "/home/ubuntu/rel/topstart-x", base_port=10700) == cfg


def test_the_setup_refuses_what_would_go_wrong(tmp_path):
    with pytest.raises(tss.Refusal):            # S's arguments changed
        b = json.loads(json.dumps(tss.S_LIVE))
        b["phases"]["ppo"]["extra_train_args"].append("--human-imitation-coef")
        tss.build_run_json(b, "/rel")
    with pytest.raises(tss.Refusal):
        b = json.loads(json.dumps(tss.S_LIVE))
        b["train_defaults"]["extra_train_args"] = ["--obs-matchup"]
        tss.build_run_json(b, "/rel")
    cfg = tss.build_run_json(tss.S_LIVE, "/rel")
    bad = json.loads(json.dumps(cfg))
    bad["phases"]["warmup"]["extra_train_args"] = []
    assert "--set-head-coef" in tss.config_problem(bad)
    bad = json.loads(json.dumps(cfg))
    bad["phases"]["ppo"]["extra_train_args"] += ["--lr-hold-steps", "5"]
    assert "decisions schedule" in tss.config_problem(bad)
    bad = json.loads(json.dumps(cfg))
    bad["retention"]["enabled"] = True
    assert "retention" in tss.config_problem(bad)
    # the start
    good = torch.load(_fake_start(str(tmp_path / "s.pt")), weights_only=False)
    assert tss.start_problem(good) is None
    nohead = dict(good, model={"pi_head.weight": torch.zeros(1)})
    assert "set_head" in tss.start_problem(nohead)
    o = json.loads(json.dumps(good["run_cfg"]))
    o["obs"]["movesets"] = False
    assert "observation" in tss.start_problem(dict(good, run_cfg=o))
    m = json.loads(json.dumps(good["run_cfg"]))
    m["learner"]["mode"] = "ppo"
    assert "mode" in tss.start_problem(dict(good, run_cfg=m))
    assert "optimizer" in tss.start_problem(dict(good, optimizer={}))
    # the release
    rel = tmp_path / "rel"
    rel.mkdir()
    (rel / "train.py").write_text("# old\n")
    (rel / "infer_pool.py").write_text("")
    assert "--lr-hold-decisions" in tss.release_problem(str(rel))
    (rel / "train.py").write_text("ap.add_argument('--lr-hold-decisions')\n")
    assert "heuristic/cache" in tss.release_problem(str(rel))       # a bare git archive (test_the_release_needs_...)
    _fill_cache(rel)
    assert tss.release_problem(str(rel)) is None
    # the run directory: started, another run.json, checkpoints there
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    assert tss.run_dir_problem(str(run), cfg) is None
    (run / "run.json").write_text(json.dumps(cfg))
    assert tss.run_dir_problem(str(run), cfg) is None
    (run / "run.json").write_text("{}")
    assert "differs" in tss.run_dir_problem(str(run), cfg)
    (run / "run.json").write_text(json.dumps(cfg))
    (run / "checkpoints" / "learner_update_000005.pt").write_bytes(b"x")
    assert "checkpoints" in tss.run_dir_problem(str(run), cfg)
    (run / "state.json").write_text("{}")
    assert "state.json" in tss.run_dir_problem(str(run), cfg)
    # main() refuses each with exit 2 and writes nothing
    start = _fake_start(str(tmp_path / "start.pt"))
    common = ["--rel", str(rel), "--base", "builtin", "--start", start, "--write"]
    for argv, why in ((["--run", str(run)], "state.json"),
                      (["--run", str(tmp_path / "r2"), "--start", _fake_start(str(tmp_path / "nh.pt"), model={"x": torch.zeros(1)})], "set_head"),
                      (["--run", str(tmp_path / "r3"), "--start", _fake_start(str(tmp_path / "opt.pt"), optimizer={"a": 1})], "optimizer")):
        before = sorted(os.listdir(tmp_path))
        assert tss.main(common + argv) == 2, why
        assert sorted(os.listdir(tmp_path)) == before
    (rel / "train.py").write_text("# old\n")
    assert tss.main(common + ["--run", str(tmp_path / "r4")]) == 2
    assert not os.path.exists(tmp_path / "r4")
    # a release without heuristic/cache (a bare git archive) is refused by main() too, and nothing is written
    (rel / "train.py").write_text("ap.add_argument('--lr-hold-decisions')\n")
    os.rename(rel / "heuristic", tmp_path / "heuristic.aside")
    assert tss.main(common + ["--run", str(tmp_path / "r5")]) == 2
    assert not os.path.exists(tmp_path / "r5")


def test_the_setup_writes_the_run_and_check_restart_says_warmup_starts_afresh(tmp_path, monkeypatch, capsys):
    import check_restart
    import pipeline
    start = _fake_start(str(tmp_path / "topstart_T0.pt"))
    digest = ts.sha256_of(start)
    (tmp_path / "topstart_T0.pt.sha256").write_text(f"{digest}  topstart_T0.pt\n")
    (tmp_path / "topstart_T0.pt.json").write_text("{}")
    run = tmp_path / "run"
    time.sleep(0.01)
    assert tss.main(["--rel", REPO, "--base", "builtin", "--start", start, "--run", str(run), "--write", "--no-release-check"]) == 0
    out = capsys.readouterr().out
    ck = run / "checkpoints" / "learner_update_000000.pt"
    assert ts.sha256_of(str(ck)) == digest and ts.sha256_of(str(run / "topstart" / "topstart_T0.pt")) == digest
    assert (run / "topstart" / "topstart_T0.pt.sha256").read_text().split()[0] == digest
    assert (run / "topstart" / "topstart_T0.pt.json").exists()
    assert os.path.getmtime(ck) >= os.path.getmtime(start)
    cfg = json.loads((run / "run.json").read_text())
    assert cfg == tss.build_run_json(tss.S_LIVE, REPO, base_name="builtin copy", base_sha="builtin")
    assert pipeline.newest_checkpoint(str(run / "checkpoints")).name == "learner_update_000000.pt"
    cmds = tss.check_commands(REPO, str(run))
    assert all(c in out for c in cmds)
    # the check the setup prints: the warmup start, a phase change from the start's imitation mode
    lines = []
    code = check_restart.check(str(run / "run.json"), str(run), out=lines.append, phase_override="warmup")
    assert code == 0, lines
    assert any("the checkpoint is of phase 'imitation' and the start is 'warmup': the optimizer and the schedule start afresh" in ln
               for ln in lines), lines
    cmd = next(ln for ln in lines if ln.startswith("trainer command"))
    assert "--mode warmup" in cmd and "--set-head-coef 0.1" in cmd and "--ports 10700-10735" in cmd
    assert any(ln.startswith("obs_settings     checkpoint ['matchup', 'movesets', 'own_tera_type']; this start ['matchup', 'movesets', 'own_tera_type']")
               for ln in lines), lines
    # and the PPO phase's command line through train.py: it needs the decisions schedule (item lr-decisions)
    lines = []
    code = check_restart.check(str(run / "run.json"), str(run), out=lines.append, phase_override="ppo", checkpoint=str(ck))
    if _has_decision_flags():
        assert code == 0 and any("the start is 'ppo'" in ln for ln in lines), lines
    else:
        assert code == 2 and "train.py refuses this command line" in lines[-1], lines
    # "first" is the supervisor's first phase: warmup here
    lines = []
    assert check_restart.check(str(run / "run.json"), str(run), out=lines.append, phase_override="first") == 0
    assert any("--phase warmup" in ln for ln in lines)
    # a second setup into the same directory is refused (the checkpoint is there), the first's files are as they were
    assert tss.main(["--rel", REPO, "--base", "builtin", "--start", start, "--run", str(run), "--write", "--no-release-check"]) == 2
    assert ts.sha256_of(str(ck)) == digest


def test_the_units_are_s_template_with_ts_names(tmp_path):
    u = tss.unit_texts("/home/ubuntu/rel/topstart-x", "/home/ubuntu/runs/v2/topstart", "0,1", py="/v/py", home="/home/ubuntu")
    assert sorted(u) == ["psppo-v2-topstart-eval", "psppo-v2-topstart-train"]
    t, e = u["psppo-v2-topstart-train"], u["psppo-v2-topstart-eval"]
    assert "WorkingDirectory=/home/ubuntu/rel/topstart-x" in t and "PSPPO_GPU_SPLIT=1 CUDA_VISIBLE_DEVICES=0,1" in t
    assert "ExecStart=/v/py /home/ubuntu/rel/topstart-x/pipeline.py --config /home/ubuntu/runs/v2/topstart/run.json --run-dir /home/ubuntu/runs/v2/topstart" in t
    assert "ExecCondition=" in t and "--can-start" in t and "Nice=" not in t
    assert "Nice=5" in e and "evaluate.py --run-dir /home/ubuntu/runs/v2/topstart --watch --config /home/ubuntu/runs/v2/topstart/run.json" in e
    assert "RestartPreventExitStatus=2 3 4 5" in t
    r = tss.unit_texts("/rel", "/runs/v2/topstart-rehearsal", "0,1", rehearsal=True)
    assert sorted(r) == ["psppo-v2-topstart-rehearsal-train"]
    # S's template, line for line, apart from the names
    import scratch_setup
    assert [ln for ln in tss.UNIT.splitlines() if not ln.startswith("Description=")] == \
           [ln for ln in scratch_setup.UNIT.splitlines() if not ln.startswith("Description=")]


# =============================================================================================
# review findings: the release's heuristic cache, v2-fast's settings, the comparison with S
# =============================================================================================

def test_the_release_needs_heuristic_cache_which_a_git_archive_never_has(tmp_path):
    # .gitignore keeps heuristic/cache out of git, so `git archive` alone never carries it
    with open(os.path.join(REPO, ".gitignore"), encoding="utf-8") as f:
        assert "heuristic/cache/" in [ln.strip() for ln in f]
    import matchup
    assert matchup.load_randbats_sets(str(tmp_path / "no_such.json")) == {}      # what the read tracker gets, without a word
    rel = tmp_path / "rel"
    rel.mkdir()
    (rel / "train.py").write_text("ap.add_argument('--lr-hold-decisions')\n")   # a release of the right code
    (rel / "infer_pool.py").write_text("")
    p = tss.release_problem(str(rel))                            # the plan's release: the archive alone
    assert "heuristic/cache/gen9randombattle.json" in p and "cp -a ~/rel/v2-humanaux" in p, p
    _fill_cache(rel, species=150, battle=False)
    assert "battle.js" in tss.release_problem(str(rel))
    _fill_cache(rel, species=0)
    assert "holds 0 species" in tss.release_problem(str(rel))
    _fill_cache(rel)
    assert tss.release_problem(str(rel)) is None
    (rel / "heuristic" / "cache" / "gen9randombattle.json").write_text("{not json")
    assert "cannot be read" in tss.release_problem(str(rel))


def _fast_release(tmp_path, fast=True):
    rel = tmp_path / ("rel_fast" if fast else "rel_plain")
    rel.mkdir()
    flags = ["--lr-hold-decisions"] + (list(tss.FAST_TRAIN_FLAGS) if fast else [])
    (rel / "train.py").write_text("".join(f"ap.add_argument('{f}')\n" for f in flags))
    (rel / "infer_pool.py").write_text("")
    if fast:
        (rel / "fast_ingest.py").write_text("")
    _fill_cache(rel)
    return rel


def test_fast_adds_v2_fasts_three_settings_and_nothing_else(tmp_path):
    import pipeline
    plain = tss.build_run_json(tss.S_LIVE, "/rel")
    fast = tss.build_run_json(tss.S_LIVE, "/rel", fast=True)
    assert "command" not in plain["servers"] and "--rollout-nice" not in json.dumps(plain)
    assert fast["servers"] == dict(plain["servers"], command=tss.FAST_SERVER_COMMAND)
    assert fast["train_defaults"] == dict(plain["train_defaults"],
                                          extra_train_args=tss.DEFAULT_ARGS + ["--rollout-nice", "5"])
    assert fast["phases"]["ppo"] == dict(plain["phases"]["ppo"],
                                         extra_train_args=plain["phases"]["ppo"]["extra_train_args"] + ["--fast-update", "--fast-ingest"])
    assert fast["phases"]["warmup"] == plain["phases"]["warmup"] and fast["phases"]["imitation"] == plain["phases"]["imitation"]
    assert {k: v for k, v in fast.items() if k not in ("servers", "train_defaults", "phases", "_notes")} == \
           {k: v for k, v in plain.items() if k not in ("servers", "train_defaults", "phases", "_notes")}
    # the supervisor: the servers' command is its default with `nice -n 5` before it; both phases get --rollout-nice 5
    pf, pp = tmp_path / "fast.json", tmp_path / "plain.json"
    pf.write_text(json.dumps(fast))
    pp.write_text(json.dumps(plain))
    cf, cp = pipeline.load_config(str(pf)), pipeline.load_config(str(pp))
    paths = pipeline.Paths(str(tmp_path / "run"))
    for phase in ("warmup", "ppo"):
        assert pipeline.build_server_command(cf, paths, phase) == ["nice", "-n", "5"] + pipeline.build_server_command(cp, paths, phase)
        tf, tp = pipeline.build_trainer_command(cf, paths, phase), pipeline.build_trainer_command(cp, paths, phase)
        extra = ["--rollout-nice", "5"] + (["--fast-update", "--fast-ingest"] if phase == "ppo" else [])
        assert sorted(tf) == sorted(tp + extra), phase
        if _has_flags("--fast-update", "--fast-ingest", "--rollout-nice", "5", "--lr-hold-decisions", "1", "--lr-total-decisions", "2"):
            import train
            train.parse_args(tf[tf.index("--mode"):])             # train.py takes the whole command line (merged tree)
    # the rehearsal with --fast
    r = tss.build_run_json(tss.S_LIVE, "/rel", rehearsal=True, fast=True)
    assert r["servers"]["command"] == tss.FAST_SERVER_COMMAND and r["phases"]["ppo"]["extra_train_args"][-2:] == ["--fast-update", "--fast-ingest"]
    # a base with v2-fast already deployed (SPEED.md section 5 on S's run.json) gives the same two run.json
    deployed = json.loads(json.dumps(tss.S_LIVE))
    deployed["servers"]["command"] = list(tss.FAST_SERVER_COMMAND)
    deployed["phases"]["ppo"]["extra_train_args"] += ["--fast-update", "--fast-ingest", "--rollout-nice", "5"]
    assert tss.build_run_json(deployed, "/rel") == plain and tss.build_run_json(deployed, "/rel", fast=True) == fast
    other = json.loads(json.dumps(tss.S_LIVE))
    other["servers"]["command"] = ["nice", "-n", "9", "x"]
    with pytest.raises(tss.Refusal):
        tss.build_run_json(other, "/rel")
    # half the settings would starve or mislead: refused
    bad = json.loads(json.dumps(fast))
    bad["train_defaults"]["extra_train_args"] = list(tss.DEFAULT_ARGS)
    assert "--rollout-nice" in tss.config_problem(bad)
    bad = json.loads(json.dumps(plain))
    bad["phases"]["ppo"]["extra_train_args"] += ["--rollout-nice", "5"]
    assert "servers.command" in tss.config_problem(bad)


def test_fast_needs_a_v2_fast_release_and_a_v2_fast_release_needs_a_choice(tmp_path, capsys):
    plain_rel, fast_rel = _fast_release(tmp_path, fast=False), _fast_release(tmp_path, fast=True)
    assert tss.release_problem(str(plain_rel)) is None and "not a v2-fast release" in tss.release_problem(str(plain_rel), fast=True)
    assert tss.release_problem(str(fast_rel), fast=True) is None and tss.release_problem(str(fast_rel)) is None
    assert tss.release_has_fast(str(fast_rel)) and not tss.release_has_fast(str(plain_rel))
    start = _fake_start(str(tmp_path / "start.pt"))
    common = ["--base", "builtin", "--start", start]
    assert tss.main(common + ["--rel", str(fast_rel), "--run", str(tmp_path / "r1")]) == 2      # neither --fast nor --no-fast
    assert "say --fast" in capsys.readouterr().out
    assert tss.main(common + ["--rel", str(plain_rel), "--run", str(tmp_path / "r2"), "--fast"]) == 2
    assert tss.main(common + ["--rel", str(fast_rel), "--run", str(tmp_path / "r3"), "--fast"]) == 0
    out = capsys.readouterr().out
    assert '"command"' in out and "--fast-update" in out
    assert tss.main(common + ["--rel", str(fast_rel), "--run", str(tmp_path / "r4"), "--no-fast"]) == 0
    assert "--fast-update" not in capsys.readouterr().out
    assert tss.main(common + ["--rel", str(plain_rel), "--run", str(tmp_path / "r5")]) == 0
    with pytest.raises(SystemExit):
        tss.main(common + ["--rel", str(fast_rel), "--fast", "--no-fast"])
    assert not any(os.path.exists(tmp_path / f"r{k}") for k in range(1, 6))       # no --write: nothing written


def test_the_setup_with_fast_writes_a_run_check_restart_passes_on_a_v2_fast_tree(tmp_path):
    import check_restart
    start = _fake_start(str(tmp_path / "topstart_T0.pt"))
    run = tmp_path / "run"
    assert tss.main(["--rel", REPO, "--base", "builtin", "--start", start, "--run", str(run), "--write", "--fast",
                     "--no-release-check"]) == 0
    cfg = json.loads((run / "run.json").read_text())
    assert cfg == tss.build_run_json(tss.S_LIVE, REPO, base_name="builtin copy", base_sha="builtin", fast=True)
    ok = _has_flags("--fast-update", "--fast-ingest", "--rollout-nice", "5", "--lr-hold-decisions", "1", "--lr-total-decisions", "2")
    for phase, kw in (("warmup", {}), ("ppo", {"checkpoint": str(run / "checkpoints" / "learner_update_000000.pt")})):
        lines = []
        code = check_restart.check(str(run / "run.json"), str(run), out=lines.append, phase_override=phase, **kw)
        if ok:                                            # the merged release tree (topstart: lr-decisions + v2-fast)
            assert code == 0, lines
            cmd = next(ln for ln in lines if ln.startswith("trainer command"))
            assert "--rollout-nice 5" in cmd and (("--fast-update --fast-ingest" in cmd) == (phase == "ppo")), cmd
        else:                                             # this branch alone: train.py refuses the flags
            assert code == 2 and "train.py refuses this command line" in lines[-1], lines


def test_the_comparison_with_s_says_which_milestones_read_the_start_alone(capsys):
    t = {m["decisions"]: m for m in tss.milestone_table()}
    assert sorted(t) == list(tss.MILESTONES)
    # S had no moveset block before 60.9M PPO decisions: B3 and B4 measure the start and the block together
    for d in (0, 25_000_000, 100_000_000):
        assert t[d]["reads"] == "start + moveset block" and any("moveset block" in x for x in t[d]["differs"]), t[d]
    for d in (150_000_000, 300_000_000, 600_000_000):
        assert t[d]["reads"] == "start", t[d]
    assert "S 39.1M" in t[100_000_000]["differs"][0] and "S 89.1M" in t[150_000_000]["differs"][0]
    # the boundary: S with the block for half of its decisions
    edge = 2 * tss.S_MOVESETS_FROM
    assert tss.comparison_with_s(edge)["reads"] == "start" and tss.comparison_with_s(edge - 1)["reads"] == "start + moveset block"
    # S's PPO-time imitation term: not by 300M, 18.0M of it by 600M
    assert not any("imitation term" in x for x in t[300_000_000]["differs"])
    assert any("imitation term for 18.0M" in x for x in t[600_000_000]["differs"])
    assert t[25_000_000]["bar"] == "B3" and t[150_000_000]["bar"].startswith("B5")
    # the run.json carries the reading next to the run
    notes = tss.build_run_json(tss.S_LIVE, "/rel")["_notes"]["comparison with S"]
    assert len(notes) == len(tss.MILESTONES) and notes[1].startswith("25M B3: reads start + moveset block")
    assert tss.main(["--milestones"]) == 0
    out = capsys.readouterr().out
    assert "B3" in out and "start + moveset block" in out and "reads: start\n" in out


def test_set_head_drift_compares_the_heads_only():
    a = {"set_head.w": torch.zeros(2, 2), "pi.w": torch.zeros(1)}
    b = {"set_head.w": torch.full((2, 2), 0.25), "pi.w": torch.ones(1)}
    assert tss.set_head_drift(a, a) == 0.0 and tss.set_head_drift(a, b) == 0.25
    assert tss.set_head_drift({"pi.w": torch.zeros(1)}, b) is None and tss.set_head_drift(a, {"pi.w": torch.zeros(1)}) is None


def test_check_restart_without_phase_is_the_commit_befores(tmp_path, monkeypatch):
    if not HEAD or not os.path.isfile(os.path.join(HEAD, "cloud", "check_restart.py")):
        pytest.skip("PSPPO_TOPSTART_HEAD names no copy of the commit before (4cfd496)")
    import check_restart as new
    import pipeline  # noqa: F401  (imported from this checkout before the old file puts its own directory first)
    import train  # noqa: F401
    import test_check_restart as tcr
    path_before = list(sys.path)
    spec = importlib.util.spec_from_file_location("check_restart_head", os.path.join(HEAD, "cloud", "check_restart.py"))
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    sys.path[:] = path_before
    old.REPO = REPO                 # the same train.py for both (only check_restart.py differs between the two commits)
    assert "phase_override" not in old.check.__code__.co_varnames
    for extra, ck in (([], tcr._ckpt(409_000)), (["--lr-totl-steps", "5"], tcr._ckpt(409_000)),
                      (tcr._restart_args(), tcr._ckpt(tcr.STEP, total=1_000_000)), ([], None)):
        outs = []
        for mod in (old, new):
            lines = []
            if ck is not None:
                monkeypatch.setattr(mod, "read_checkpoint", lambda path, ck=ck: dict(ck))
            code = mod.check(tcr._config(tmp_path, extra), str(tmp_path / "run"), checkpoint="mem" if ck else None, out=lines.append)
            outs.append((code, lines))
        assert outs[0] == outs[1]
    # and with a state.json naming warmup, as before
    (tmp_path / "run").mkdir(exist_ok=True)
    (tmp_path / "run" / "state.json").write_text(json.dumps({"phase": "warmup"}))
    outs = []
    for mod in (old, new):
        lines = []
        outs.append((mod.check(tcr._config(tmp_path, []), str(tmp_path / "run"), out=lines.append), lines))
    assert outs[0] == outs[1]


# =============================================================================================
# real shards
# =============================================================================================

@pytest.fixture(scope="module")
def real():
    if not HAVE_DATA:
        pytest.skip("no real shards")
    man = os.path.join(DATA, "imitation_data.json")
    with open(man) as f:
        m = json.load(f)
    have = {s["name"]: sorted(os.listdir(os.path.join(DATA, s["name"]))) for s in m["sets"]}
    skip = [os.path.join(DATA, s["name"], n) for s in m["sets"] for n in s["skip_shards"] if n in have[s["name"]]]
    assert skip, "the test data holds no skip shard"
    links = {}
    for name, pick in (("ALL", None), ("TWO", 1)):
        dst = _dir(f"data_{name}")
        assert ts.main(["link", "--manifest", man, "--set", f"40k={os.path.join(DATA, '40k')}", "--set",
                        f"bulk={os.path.join(DATA, 'bulk')}", "--dst", dst, "--allow-missing"]) == 0
        if pick:
            keep = [sorted(n for n in os.listdir(dst) if n.startswith("shard-40k-"))[0],
                    sorted(n for n in os.listdir(dst) if n.startswith("shard-bulk-"))[0]]
            two = _dir("data_TWO_only")
            for n in keep:
                os.symlink(os.path.realpath(os.path.join(dst, n)), os.path.join(two, n))
            dst = two
        links[name] = dst
    players, games = set(), set()
    for f in glob.glob(os.path.join(DATA, "*", "shard-*.npz")):
        with np.load(f) as z:
            players.update(map(str, z["ep_player"]))
            games.update(map(str, z["ep_game"]))
    return {"manifest": man, "all": links["ALL"], "two": links["TWO"], "skip": skip, "players": players, "games": games}


def _real_offsets():
    return ts.build_net(ts.run_config(), 1, "cpu", set_head=False).unpacker.offsets


@needs_data
def test_one_pass_of_the_real_set_reads_train_sides_only_and_never_a_skip_shard(real, monkeypatch):
    linked = sorted(os.listdir(real["all"]))
    assert len(linked) == 12 and all(n.startswith(("shard-40k-", "shard-bulk-")) for n in linked)
    assert not any(os.path.realpath(os.path.join(real["all"], n)) in map(os.path.realpath, real["skip"]) for n in linked)
    opened = []
    orig = np.load

    def spy(f, *a, **k):
        opened.append(os.path.realpath(str(f)))
        return orig(f, *a, **k)
    monkeypatch.setattr(np, "load", spy)
    td = ts.TopData(real["all"], _real_offsets(), seed=1, threads=4)
    split_of = {}
    for f in hi.shard_files(real["all"]):
        with orig(f) as z:
            for e in range(len(z["ep_start"]) - 1):
                split_of[(os.path.realpath(f), int(z["ep_start"][e]))] = str(z["ep_split"][e])
    rows = labelled = 0
    seen_eps = set()
    for g, files in enumerate(ts.pass_groups(td.files, 1, 0, 4)):
        shards = td.load_group(files)
        seqs = hi.pack(hi.pieces_of(shards, ts.L_SEQ), ts.L_SEQ, random.Random(ts.group_seed(1, 0, g)))
        for j in range(0, len(seqs) - ts.SEQS + 1, ts.SEQS):
            chunk = seqs[j:j + ts.SEQS]
            b = hi.gather(chunk, ts.L_SEQ, with_set_labels=True)
            assert b["obs"].shape == (768, 10_295) and b["set_labels"].shape == (768, 48)
            for seq in chunk:
                for sh, a, c, _u in seq:
                    for k in set(sh.row_ep[a:c].tolist()):
                        src = sh.ep_src[k]
                        assert split_of[(os.path.realpath(sh.path), src[0])] == "train"
                        seen_eps.add((os.path.realpath(sh.path), src[0]))
            rows += int(ts.gather_extra(chunk, ts.L_SEQ)["real"].sum())
            labelled += int((b["w"] > 0).sum())
    want = sum(1 for v in split_of.values() if v == "train")
    assert len(seen_eps) >= 0.95 * want            # the leftover sequences of a group (fewer than 3) are not stepped on
    assert all(os.path.realpath(s) not in opened for s in real["skip"])
    c = td.counts()
    assert c["train_sides"] == want and c["files"] == 12
    print(f"one pass: {rows:,} rows, {labelled:,} labelled, {len(seen_eps):,} of {want:,} train sides; counts {json.dumps(c)}")


@needs_data
def test_the_network_takes_10295_values_and_has_the_sethead():
    cfg = ts.run_config()
    net = ts.build_net(cfg, 1, "cpu", set_head=True)
    assert int(max(e for _s, e in net.unpacker.offsets.values())) == 10_295
    with ts._in_repo():
        assert int(cfg.make_assembler().total_dim) == 10_295
    assert getattr(net, "set_head", None) is not None and net.set_head.has_sets and net.matchup and net.movesets
    assert getattr(net, "opp_action_head", None) is None
    n = sum(p.numel() for p in net.parameters())
    print(f"parameters {n:,}")


# =============================================================================================
# GPU + real shards: training runs (subprocesses: the command line as the box runs it)
# =============================================================================================

def _train(data, work, out, *extra, log_name="train.log", check=True):
    env = dict(os.environ, PYTHONUNBUFFERED="1", TORCHINDUCTOR_CACHE_DIR=os.path.join(ROOT, ".inductor"))
    cmd = [PY, os.path.join(REPO, "human", "topstart.py"), "train", "--data", data, "--work", work, "--out", out,
           "--threads", "4", *map(str, extra)]
    with open(os.path.join(work, log_name), "a") as f:
        p = subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    if check:
        assert p.returncode == 0, open(f.name).read()[-4000:]
    return p.returncode


SMALL = ["--warmup", "5", "--anneal-steps", "10", "--min-hold", "1000000", "--measure-every", "20", "--best-after", "0",
         "--abort-at", "off", "--resume-every", "1000000", "--holdout-decisions", "1024", "--log-every", "10"]


@pytest.fixture(scope="module")
def overfit(real):
    if not GPU:
        pytest.skip("no GPU")
    work = _dir("overfit")
    out = os.path.join(work, "start.pt")
    t = time.time()
    _train(real["two"], work, out, "--max-steps", "300", "--warmup", "30", "--anneal-steps", "60", "--min-hold", "1000000",
           "--measure-every", "50", "--best-after", "0", "--abort-at", "off", "--group-shards", "2", "--log-every", "25")
    return {"work": work, "out": out, "seconds": time.time() - t}


@needs_gpu
@needs_data
def test_300_steps_on_two_shards_learn_and_the_held_out_set_is_capped(overfit, real):
    info = json.load(open(os.path.join(overfit["work"], "info.json")))
    h = info["history"]
    assert info["steps"] == 300 and info["stop_reason"].startswith("the anneal ended")
    first = next(r["train"] for r in h if "train" in r)
    last = next(r["train"] for r in reversed(h) if "train" in r)
    # From random weights the policy sits near the prior (the most common action, about 0.23) for a while
    # (arm B: about 17k steps on the full set); 300 steps on 2 shards (13 passes) must clearly leave it.
    assert last["top1"] > 0.33 and last["top1"] > first["top1"] + 0.10, h
    assert last["set"] < first["set"] and last["v"] < first["v"], (first, last)
    assert info["skipped_nonfinite"] == 0
    ck = torch.load(overfit["out"], map_location="cpu", weights_only=False)
    assert ck["topstart"]["best"]["step"] == info["best"]["step"]
    # no player above 3% of the capped set's target (32,768: at most 983 decisions each)
    td = ts.TopData(real["two"], _real_offsets(), seed=1, threads=4)
    per = {}
    for seq in td.holdout_seqs:
        for sh, a, c, _u in seq:
            for k in set(sh.row_ep[a:c].tolist()):
                ea, eb = sh.eps[k]
                lo, hi_ = max(a, ea), min(c, eb)
                per[sh.players[k]] = per.get(sh.players[k], 0) + int(sh.lab[lo:hi_].sum())
    assert sum(per.values()) == td.h.ho["n"] and max(per.values()) <= td.h.holdout["limit"] == 983
    print(f"overfit: {overfit['seconds']:.0f} s; train {last}; held-out {h[-1]['top1']:.4f}; capped set {td.h.ho['n']:,} decisions "
          f"of {len(per)} players, the largest {max(per.values())}")


@needs_gpu
@needs_data
def test_two_groups_stop_resume_gives_the_same_batches(real):
    common = ["--max-steps", "70", "--group-shards", "2", *SMALL]
    wa, wb = _dir("resume_A"), _dir("resume_B")
    _train(real["all"], wa, os.path.join(wa, "start.pt"), *common, "--trace", os.path.join(wa, "trace.txt"))
    _train(real["all"], wb, os.path.join(wb, "start.pt"), *common, "--trace", os.path.join(wb, "trace.txt"), "--stop-after-groups", "2")
    assert not os.path.exists(os.path.join(wb, "start.pt"))
    res = glob.glob(os.path.join(wb, "resume_*.pt"))
    assert len(res) == 1
    snap = {n: ts.sha256_of(os.path.join(wb, n)) for n in os.listdir(wb) if os.path.isfile(os.path.join(wb, n))}
    appended = {n: open(os.path.join(wb, n), "rb").read() for n in ("log.jsonl", "trace.txt", "train.log")}
    stopped_at = int(torch.load(res[0], map_location="cpu", weights_only=False)["state"]["step"])
    assert 0 < stopped_at < 70
    _train(real["all"], wb, os.path.join(wb, "start.pt"), *common, "--trace", os.path.join(wb, "trace.txt"))
    ta = open(os.path.join(wa, "trace.txt")).read().splitlines()
    tb = open(os.path.join(wb, "trace.txt")).read().splitlines()
    assert len(ta) == 70 and ta == tb
    ia, ib = json.load(open(os.path.join(wa, "info.json"))), json.load(open(os.path.join(wb, "info.json")))
    assert ia["steps"] == ib["steps"] == 70 and ia["rows_seen"] == ib["rows_seen"] and ia["steps_by_pass"] == ib["steps_by_pass"]
    for n, digest in snap.items():
        if n in appended:
            assert open(os.path.join(wb, n), "rb").read().startswith(appended[n]), n
        else:
            assert ts.sha256_of(os.path.join(wb, n)) == digest, f"{n} was overwritten"
    assert [r["step"] for r in ia["history"]] == [r["step"] for r in ib["history"]]
    print(f"resume: stopped at step {stopped_at}, both runs 70 steps with the same batch ids; held-out top1 "
          f"{ia['history'][-1]['top1']:.4f} / {ib['history'][-1]['top1']:.4f}")


def _learner(mode, **over):
    import test_speed_update as tsu
    import learner as new_learner
    import ppo_core as new_core
    cfg = ts.run_config(set_head_coef=over.pop("set_head_coef", 0.1), mode=mode)
    lc = dict(device="cuda", minibatch_size=768, batch_seq_len=256, grad_accum_steps=1, steps_per_update=1536, resume=False,
              save_every_updates=10 ** 9, ckpt_dir=os.path.join(ROOT, "ckpt_unused"))
    lc.update(over)
    cfg = dataclasses.replace(cfg, learner=dataclasses.replace(cfg.learner, **lc))
    return tsu._make_learner(new_learner, new_core, cfg)


@needs_gpu
@needs_data
def test_the_start_through_the_learner_keeps_the_sethead_into_ppo(overfit, real, capsys):
    out = overfit["out"]
    ck = torch.load(out, map_location="cpu", weights_only=False)
    rc = ck["run_cfg"]
    assert rc["learner"]["mode"] == "imitation" and rc["learner"]["set_head_coef"] == 0.1 and rc["learner"]["opp_action_aux_coef"] == 0.0
    assert all(rc["obs"][k] is True for k in ("own_tera_type", "matchup", "movesets"))
    assert "optimizer" not in ck and "scheduler" not in ck and ck["target_model"] is None and ck["update_idx"] == 0
    sh_keys = [k for k in ck["model"] if k.startswith("set_head.")]
    assert sh_keys and ck["total_steps"] > 0 and tss.start_problem(ck) is None
    # warmup with --set-head-coef 0.1: a phase change, every weight as the start has it
    wu = _learner("warmup")
    wu._load_checkpoint(out)
    said = capsys.readouterr().out
    assert "PHASE CHANGE DETECTED (imitation -> warmup)" in said
    state = wu.net.state_dict()
    assert sorted(k for k in state if "attn_mask" not in k) == sorted(k for k in ck["model"] if "attn_mask" not in k)
    assert all(torch.equal(state[k].cpu(), ck["model"][k]) for k in ck["model"] if "attn_mask" not in k)
    assert wu.opt.inner.state_dict()["state"] == {} and wu.sched.last_epoch == 0
    assert (wu.update_idx, wu.total_steps, wu.total_episodes) == (0, 0, 0)
    base = {k: v for k, v in state.items() if not k.startswith(("set_head.", "opp_action_head."))}
    assert all(torch.equal(wu.target_net.state_dict()[k], v) for k, v in base.items())
    # one warmup update (rollout-like rows from a real shard, with set labels): the head is not touched, the critic is
    td = ts.TopData(real["two"], _real_offsets(), seed=1, threads=2, build_holdout=False)
    sh = td.load(td.files[0], "train", 1)
    n = 1536
    rows = np.arange(n) % sh.rows()
    obs = torch.from_numpy(np.ascontiguousarray(sh.obs[rows]))
    rng = np.random.default_rng(4)
    act = torch.from_numpy(np.array([rng.choice(np.flatnonzero(m > 0.5)) for m in sh.mask[rows]], dtype=np.int64))
    logp = torch.from_numpy((-np.log(np.maximum((sh.mask[rows] > 0.5).sum(1), 1.0))).astype(np.float32))
    ep = torch.from_numpy(np.repeat(np.arange(n // 64), 64)).long()
    z = torch.zeros(n)
    extras = {"set_labels": torch.from_numpy(np.ascontiguousarray(sh.set_labels[rows]))}
    wu.dataset.add_steps(obs, act, logp, z, torch.from_numpy(rng.standard_normal(n).astype(np.float32)),
                         torch.from_numpy(np.clip(rng.standard_normal(n) * 0.5, -1.4, 1.4).astype(np.float32)), z, ep, extras=extras)
    wu.total_steps += n
    asyncio.run(wu._perform_update())
    assert wu.update_idx == 1
    after = wu.net.state_dict()
    assert all(torch.equal(after[k].cpu(), ck["model"][k]) for k in sh_keys)
    assert any(not torch.equal(after[k].cpu(), ck["model"][k]) for k in ck["model"] if k.startswith("v_head."))
    path = os.path.join(_dir("learner_g"), "learner_update_000001.pt")
    wu._save_checkpoint(path)
    # PPO (the run's heads) loads the warmup save: set_head as the start has it
    ppo = _learner("ppo", opp_action_aux_coef=0.1)
    ppo._load_checkpoint(path)
    assert "PHASE CHANGE DETECTED (warmup -> ppo)" in capsys.readouterr().out
    ps = ppo.net.state_dict()
    assert all(torch.equal(ps[k].cpu(), ck["model"][k]) for k in sh_keys)
    # the gotcha: a warmup without --set-head-coef drops set_head.* (strict=False), and PPO trains a fresh head
    bare = _learner("warmup", set_head_coef=0.0)
    bare._load_checkpoint(out)
    assert getattr(bare.net, "set_head", None) is None
    p2 = os.path.join(_dir("learner_g_bare"), "learner_update_000000.pt")
    bare._save_checkpoint(p2)
    assert not any(k.startswith("set_head.") for k in torch.load(p2, map_location="cpu", weights_only=False)["model"])
    ppo2 = _learner("ppo", opp_action_aux_coef=0.1)
    ppo2._load_checkpoint(p2)
    ps2 = ppo2.net.state_dict()
    assert not all(torch.equal(ps2[k].cpu(), ck["model"][k]) for k in sh_keys)
    del wu, ppo, bare, ppo2
    torch.cuda.empty_cache()


@needs_gpu
@needs_data
def test_a_real_shard_with_a_nan_in_one_episode(real):
    d = _dir("data_nan")
    names = sorted(os.listdir(real["two"]))
    src = os.path.realpath(os.path.join(real["two"], names[0]))
    with np.load(src) as z:
        arrays = {k: z[k] for k in z.files}
    st, split = arrays["ep_start"], arrays["ep_split"]
    e = next(i for i in range(len(st) - 1) if str(split[i]) == "train")
    arrays["obs"] = arrays["obs"].copy()
    arrays["obs"][int(st[e]) + 2, 17] = np.nan
    np.savez_compressed(os.path.join(d, "shard-40k-99999-nantestaaa.npz"), **arrays)
    os.symlink(os.path.realpath(os.path.join(real["two"], names[1])), os.path.join(d, names[1]))
    work = _dir("nan_run")
    _train(d, work, os.path.join(work, "start.pt"), "--max-steps", "30", "--group-shards", "2", *SMALL)
    info = json.load(open(os.path.join(work, "info.json")))
    assert info["data"]["dropped_episodes"] == 1 and info["skipped_nonfinite"] == 0
    assert all(math.isfinite(r["vloss"]) and math.isfinite(r["ce"]) for r in info["history"])


def _hash_hits(root, needles):
    """
    Files under root holding a salted hash of the data (human/rebuild.py: "u" or "g" and 12 hex digits):
    one regex pass a file, the matches against the data's own hashes. Text outputs (json, logs, traces)
    may hold no such pattern at all, whoever's.
    """
    import re
    pat = re.compile(rb"[ug][0-9a-f]{12}")
    want = {n.encode() for n in needles}
    hits = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(("data_", ".inductor"))]      # the inputs (shard copies), the compiler's cache
        for n in files:
            p = os.path.join(dirpath, n)
            if os.path.islink(p):
                continue
            with open(p, "rb") as f:
                blob = f.read()
            found = set(pat.findall(blob))
            if found & want:
                hits.append((p, "a hash of the data"))
            elif n.endswith((".json", ".jsonl", ".log", ".txt", ".sha256")) and any(
                    re.search(rb"(?<![0-9a-zA-Z])[ug][0-9a-f]{12}(?![0-9a-zA-Z])", blob) for _ in (0,)):
                hits.append((p, "a hash-shaped string"))
    return hits


@needs_gpu
@needs_data
def test_measure_writes_every_field_and_nothing_holds_a_hash(overfit, real):
    work = _dir("measure")
    env = dict(os.environ, PYTHONUNBUFFERED="1", TORCHINDUCTOR_CACHE_DIR=os.path.join(ROOT, ".inductor"))
    out = os.path.join(work, "T.json")
    with open(os.path.join(work, "measure.log"), "w") as f:
        p = subprocess.run([PY, "human/topstart.py", "measure", "--data", real["all"], "--checkpoint", overfit["out"], "--out", out,
                            "--group-shards", "4", "--sha"], cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    assert p.returncode == 0, open(f.name).read()[-3000:]
    r = json.load(open(out))
    for part in ("capped", "test_player", "test_game"):
        assert {"all", "set:40k", "set:bulk"} <= set(r[part]), (part, sorted(r[part]))
        assert any(k.startswith("period:") for k in r[part])
        a = r[part]["all"]
        for k in ("top1", "top1_per_player", "ce", "vloss", "labelled", "players", "rows"):
            assert a[k] is not None, (part, k)
        assert 0 < a["top1"] < 1 and a["labelled"] > 0
        s = r[part]["sethead"]
        for k in ("loss", "ability_top1", "item_top1", "move_hits", "tera_top1"):
            assert s[k] is not None and math.isfinite(s[k]), (part, k)
        assert sum(v["labelled"] for k, v in r[part].items() if k.startswith("set:")) == a["labelled"]
    assert r["test_player"]["all"]["labelled"] >= r["capped"]["all"]["labelled"]
    assert r["has_sethead"] and len(r["checkpoint_sha256"]) == 64
    # a checkpoint without the head (as P0, the PSBot imitation's end): measured, no SetHead part
    ck = torch.load(overfit["out"], map_location="cpu", weights_only=False)
    ck["model"] = {k: v for k, v in ck["model"].items() if not k.startswith("set_head.")}
    ck["run_cfg"]["learner"]["set_head_coef"] = 0.0
    p0 = os.path.join(work, "nohead.pt")
    torch.save(ck, p0)
    out2 = os.path.join(work, "P0like.json")
    with open(os.path.join(work, "measure2.log"), "w") as f:
        p = subprocess.run([PY, "human/topstart.py", "measure", "--data", real["two"], "--checkpoint", p0, "--out", out2, "--capped-only"],
                           cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    assert p.returncode == 0, open(f.name).read()[-3000:]
    r2 = json.load(open(out2))
    assert not r2["has_sethead"] and "sethead" not in r2["capped"] and r2["capped"]["all"]["labelled"] > 0
    # no player hash and no game hash in any output, log, info file or checkpoint the tests wrote
    hits = _hash_hits(ROOT, sorted(real["players"]) + sorted(real["games"]))
    assert hits == [], hits[:5]
    print("measure:", json.dumps({k: r[k]["all"] for k in ("capped", "test_player", "test_game")}))


@needs_gpu
@needs_data
def test_speed_and_loader_keep_up_on_12_shards_under_the_bench_lock(real):
    import fcntl
    lock_path = os.path.expanduser("~/dev/fast/bench.lock")
    if not os.path.exists(lock_path):
        pytest.skip("no ~/dev/fast/bench.lock on this machine")
    work = _dir("speed")
    with open(lock_path, "a") as lk:
        t = time.time()
        while True:
            try:
                fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() - t > 1800:
                    pytest.skip("the bench lock was not free for 30 minutes")
                time.sleep(5)
        waited = time.time() - t
        try:
            _train(real["all"], work, os.path.join(work, "start.pt"), "--max-steps", "400", "--group-shards", "4", "--threads", "6",
                   "--warmup", "5", "--anneal-steps", "10", "--min-hold", "1000000", "--measure-every", "1000000", "--best-after", "0",
                   "--abort-at", "off", "--resume-every", "1000000", "--holdout-decisions", "1024", "--log-every", "50")
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)
    lines = [ln for ln in open(os.path.join(work, "train.log")) if " steps/s " in ln]
    rates = [float(ln.split(" steps/s")[0].rsplit(" ", 1)[1]) for ln in lines]
    waits = [float(ln.split("(waited ")[1].split(" s")[0]) for ln in lines]
    rep = {"steps_per_second_by_50": rates, "median_after_the_first": sorted(rates[1:])[len(rates[1:]) // 2] if len(rates) > 1 else None,
           "data_wait_s_by_50": waits, "lock_waited_s": round(waited, 1), "gpu": torch.cuda.get_device_name(0),
           "note": "box c's GPU is shared with the speed workflow; reported, not asserted"}
    with open(os.path.join(work, "speed.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print("speed:", json.dumps(rep))


@needs_gpu
@needs_data
def test_the_control_c0_starts_from_its_init_and_anneals_from_step_0(overfit, real):
    work = _dir("c0")
    _train(real["two"], work, os.path.join(work, "C0.pt"), "--init", overfit["out"], "--warmup", "100", "--min-hold", "0",
           "--max-passes", "3", "--abort-at", "off", "--group-shards", "2", "--measure-every", "20", "--best-after", "0",
           "--log-every", "10")
    info = json.load(open(os.path.join(work, "info.json")))
    sched = info["settings"]["schedule"]
    assert sched["max_steps"] == math.ceil(3 * info["steps_per_pass_estimate"]) and sched["warmup"] == 100
    assert info["anneal_start"] == 0 and info["steps"] == sched["max_steps"]
    assert info["init"]["init"] == os.path.abspath(overfit["out"]) and info["init"]["set_head_from_init"] is True
    assert info["init"]["mode"] == "imitation"
    log = open(os.path.join(work, "train.log")).read()
    assert "the anneal starts at step 0 (the latest start)" in log
    # the weights came from the init: 20 steps on (in the warmup ramp) they are still next to the init's,
    # far from the random network the same seed builds
    init = torch.load(overfit["out"], map_location="cpu", weights_only=False)["model"]
    at20 = torch.load(os.path.join(work, "best_000020.pt"), map_location="cpu", weights_only=False)["model"]
    fresh = ts.build_net(ts.run_config(), 1, "cpu", set_head=True).state_dict()
    mats = [k for k in init if init[k].is_floating_point() and init[k].dim() == 2 and k in fresh]
    assert any(k.startswith("set_head.") for k in mats) and len(mats) > 20
    for group in (mats, [k for k in mats if k.startswith("set_head.")]):
        near = sum(float((at20[k].float() - init[k].float()).abs().sum()) for k in group)
        far = sum(float((fresh[k].float() - init[k].float()).abs().sum()) for k in group)
        assert near < far / 10, (near, far)
    # a checkpoint without the moveset block (as S's P0) is refused, with the widening command
    ck = torch.load(overfit["out"], map_location="cpu", weights_only=False)
    ck["run_cfg"]["obs"]["movesets"] = False
    p = os.path.join(work, "no_movesets.pt")
    torch.save(ck, p)
    w2 = _dir("c0_refused")
    code = _train(real["two"], w2, os.path.join(w2, "x.pt"), "--init", p, "--max-steps", "5", check=False)
    assert code == 2
    assert "widen_checkpoint.py --movesets" in open(os.path.join(w2, "train.log")).read()


@needs_gpu
@needs_data
def test_steps_that_are_not_finite_are_skipped_and_three_in_a_row_stop_with_exit_3(real):
    """A rate no network survives: the weights blow up within a few steps, every step after that is not
    finite, is skipped and counted, and the third in a row ends the run with exit code 3 and no start."""
    work = _dir("nonfinite")
    code = _train(real["two"], work, os.path.join(work, "start.pt"), "--lr", "1e12", "--warmup", "1", "--max-steps", "200",
                  "--group-shards", "2", "--measure-every", "1000", "--best-after", "0", "--abort-at", "off",
                  "--holdout-decisions", "512", "--log-every", "1", check=False)
    log = open(os.path.join(work, "train.log")).read()
    assert code == 3, log[-3000:]
    assert "three steps in a row were not finite: stopped, no start written" in log
    assert not os.path.exists(os.path.join(work, "start.pt"))
    info = json.load(open(os.path.join(work, "info.json")))
    assert info["stop_reason"] == "nonfinite" and info["skipped_nonfinite"] >= 3
    events = [json.loads(ln) for ln in open(os.path.join(work, "log.jsonl"))]
    runs = [e["in_a_row"] for e in events if e.get("event") == "nonfinite"]
    assert runs[-3:] == [1, 2, 3]
