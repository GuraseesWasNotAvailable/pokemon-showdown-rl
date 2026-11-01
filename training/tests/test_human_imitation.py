"""
human_imitation.py: the top players' decisions as an auxiliary loss of the live PPO learner
(ported from branch stage3 to this code line).

    python -m pytest tests/test_human_imitation.py -q -p no:cacheprovider -rs
    PSPPO_HEAD_DIR=<a copy of the commit before the port> ...       (the bit-identity tests, GPU)
    PSPPO_HUMAN_SHARDS=<dir of human/rebuild.py shards, this run's observation> ...   (real shards, GPU)

Held here (CPU unless said; synthetic shards in human/rebuild.py's layouts, a stand-in network):
  * off is nothing: without --human-imitation-coef train.py gives the learner no setting, the
    learner's and ppo_update's new parameters default to None, ppo_update with human=None is the
    update of before bit for bit, and so is an update whose term is not ready (still loading, off);
  * values that cannot be are refused by train.py's parser; data that cannot be had never are
    (the term is off at the start, with the reason; check_restart refuses them);
  * the roles: a shard's own ep_split (train sides; the held-out players' own sides; loose and
    test_game never), the collector's split files for shards without it (stage3's layout); the
    player cap, the switch and double-switch weights, the double-switch label read off the next
    observation, the shuffled control (labels permuted within a legal mask);
  * a shard built otherwise than the run's observation (own Tera type, matchup, movesets,
    obs_abilities, obs_dim) is never loaded, and with none left the term is off, with the reason;
  * the term: coef x the weighted CE, gradients reach the policy and never the value head, the
    update line's figures, the held-out agreement and CE, the SetHead part (labels gathered,
    padding unlabelled, the head's loss added);
  * values that are not finite: an episode with one in its observation or mask is dropped at load
    (counted once), so every term and gradient stays finite; a term that is still not finite (a
    network already NaN) is counted and left in the loss, never hidden by a zero (whose gradient
    would be NaN all the same, also held here); settings that are not finite are refused;
  * the held-out set: no player above cap x holdout_decisions (a player's first episode always in);
  * a checkpoint written without the term resumes with it: the optimizer's layout, every learning
    rate, the schedule's step and the counters are what they were;
  * check_restart: the live run's arguments with the term say OK; data without a shard of the
    run's observation, a missing directory, a coef without data are refused.
  GPU: with PSPPO_HEAD_DIR, a whole update with the flags off (every feature off, and the live run's
  features on) is bit-identical to the commit before; with PSPPO_HUMAN_SHARDS, the real network on
  real shards (term, SetHead part, timing, held-out agreement) and a whole learner update with the
  term on whose line the supervisor parses.
"""
from __future__ import annotations

import asyncio
import dataclasses
import inspect
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "tests"), os.path.join(REPO, "cloud")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(REPO)

import human_imitation as hi  # noqa: E402

GPU = torch.cuda.is_available()
needs_gpu = pytest.mark.skipif(not GPU, reason="flex_attention has no backward on a CPU")
HEAD_DIR = os.path.expanduser(os.environ.get("PSPPO_HEAD_DIR", ""))
REAL = os.path.expanduser(os.environ.get("PSPPO_HUMAN_SHARDS", ""))

D = 140
TRANS0 = 20                          # transition scalars [20, 110), action mask [126, 140)
A0, A1 = D - 14, D
OFFSETS = {"pokemon_body": (0, 20), "transition_scalars": (TRANS0, TRANS0 + 2 * hi.SIDE_SCALAR_DIM),
           "global_scalars": (TRANS0 + 2 * hi.SIDE_SCALAR_DIM, A0), "action_mask": (A0, A1)}
SALT = "testsalt"
OPP = TRANS0 + hi.SIDE_SCALAR_DIM
RUN_OBS = {"own_tera_type": True, "matchup": True, "movesets": True}       # the live run's observation
LAB_W = 48                           # set_labels per row with the moveset block: 6 Pokemon x 8


class _Unpacker:
    offsets = OFFSETS


class TinyNet(torch.nn.Module):
    """net(obs, ep_ids, batch_seq_len[, return_z]) -> (logits, v_logits, ...[, z]), as the learner's network."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lin = torch.nn.Linear(D, 14)
        self.v = torch.nn.Linear(D, 5)
        self.z = torch.nn.Linear(D, 13 * 4)
        self.unpacker = _Unpacker()

    def forward(self, obs, ep, batch_seq_len=256, return_z=False):
        x = obs.float()
        out = (self.lin(x), self.v(x), None, None, None)
        if return_z:
            return out + (self.z(x).reshape(-1, 13, 4),)
        return out


def _episode(n, rng, switch_rows=(), opp_switch_next=(), forced=()):
    obs = rng.random((n, D)).astype(np.float16) * 0
    act = rng.integers(0, 4, n).astype(np.int16)
    mask = np.zeros((n, 14), np.uint8)
    mask[:, 0:4] = 1
    mask[:, 4:7] = 1
    for r in switch_rows:
        act[r] = 5
    for r in forced:
        mask[r, 0:4] = 0
        mask[r, 10:14] = 0
        act[r] = 4
    for r in opp_switch_next:
        obs[r, OPP + hi.IDX_DID_SWITCH] = 1
    obs[:, A0:A1] = mask
    train = np.ones(n, np.uint8)
    return obs, act, mask, train


def _write_shard(path, episodes, own_tera=True, matchup=True, movesets=True, abil="", obs_dim=D, ep_split=True,
                 set_labels=True):
    """episodes: (obs, act, mask, train, game, player, side, split). human/rebuild.py's layout (branch v2-human)."""
    obs = np.concatenate([e[0] for e in episodes])
    act = np.concatenate([e[1] for e in episodes])
    mask = np.concatenate([e[2] for e in episodes])
    train = np.concatenate([e[3] for e in episodes])
    starts = np.cumsum([0] + [len(e[1]) for e in episodes]).astype(np.int64)
    kw = dict(obs=obs, act=act, mask=mask, train=train, ep_start=starts,
              ep_game=np.array([e[4] for e in episodes]), ep_player=np.array([e[5] for e in episodes]),
              ep_side=np.array([e[6] for e in episodes], dtype=np.int8),
              ep_elo=np.full(len(episodes), 2200, np.int16), own_tera_type=np.array(own_tera))
    if matchup is not None:
        kw["obs_matchup"] = np.array(matchup)
    if movesets is not None:
        kw["obs_movesets"] = np.array(movesets)
    if abil is not None:
        kw["obs_abilities"] = np.array(abil)
    if obs_dim is not None:
        kw["obs_dim"] = np.array(obs_dim)
    if ep_split:
        kw["ep_split"] = np.array([e[7] for e in episodes])
    if set_labels:
        lab = np.zeros((len(act), LAB_W), np.int32)
        lab[:, 0] = 1 + (np.arange(len(act)) % 50)          # a species id per row, recognisable
        lab[:, 1] = 7
        kw["set_labels"] = lab
    np.savez_compressed(path, **kw)


def _data(tmp_path, n_train_games=6, **shard_kw):
    """6 train games (both sides), 3 games of a held-out player (p1 held out, p2 loose), 1 test_game side."""
    rng = np.random.default_rng(1)
    data = tmp_path / "data"
    splits = tmp_path / "splits"
    data.mkdir()
    splits.mkdir()
    (tmp_path / "salt").write_text(SALT)
    rows = {"train": [], "test_player": [], "test_game": []}
    eps = []
    for g in range(n_train_games):
        rid = f"gen9randombattle-{1000 + g}"
        rows["train"].append((rid, "p1p2"))       # the collector writes the sides as "p1p2"
        for side in (1, 2):
            o, a, m, t = _episode(30, rng, switch_rows=(3, 10), opp_switch_next=(4,), forced=(20,))
            # player "big" holds most of the data, so the cap weighs it down
            eps.append((o, a, m, t, hi.salted_game(SALT, rid), "big" if g < 4 else f"u{g}{side}", side, "train"))
    for g in range(3):
        rid = f"gen9randombattle-{2000 + g}"
        rows["test_player"].append((rid, "p1"))           # only p1 is the held-out player
        for side in (1, 2):
            o, a, m, t = _episode(25, rng)
            eps.append((o, a, m, t, hi.salted_game(SALT, rid), f"h{g}{side}", side, "test_player" if side == 1 else "loose"))
    rid = "gen9randombattle-3000"
    rows["test_game"].append((rid, "p1p2"))
    o, a, m, t = _episode(20, rng)
    eps.append((o, a, m, t, hi.salted_game(SALT, rid), "tg", 1, "test_game"))
    for name, rr in rows.items():
        (splits / f"{name}.tsv").write_text("id\tsides\n" + "".join(f"{r}\t{s}\n" for r, s in rr))
    _write_shard(str(data / "shard-00000-aaaa.npz"), eps[:8], **shard_kw)
    _write_shard(str(data / "shard-00001-bbbb.npz"), eps[8:], **shard_kw)
    return str(data), str(splits), str(tmp_path / "salt")


def _provider(tmp_path, obs=RUN_OBS, data_kw=None, **kw):
    data, splits, salt = _data(tmp_path, **(data_kw or {}))
    s = dict(data=data, coef=0.5, splits=splits, salt=salt, cap=0.5, pool_decisions=1000, holdout_decisions=75, holdout_every=1)
    s.update(kw)
    return hi.HumanImitation(s, obs)


def _wait(h, timeout=30.0):
    t = time.time()
    while h.status == "starting" and time.time() - t < timeout:
        time.sleep(0.05)
    return h.status


# ---------------------------------------------------------------------------------------------
# off is nothing
# ---------------------------------------------------------------------------------------------

LIVE_EXTRA = ["--obs-matchup", "--obs-movesets", "--infer-actors", "3", "--infer-sync-s", "1",
              "--set-head-coef", "0.1", "--opp-action-aux-coef", "0.1", "--read-bonus", "0.3", "--read-anneal", "20000000:45000000",
              "--opponent-seats", "setupnet:0.15", "--boosted-row-weight", "2", "--lr-hold-steps", "800000",
              "--lr-total-steps", "1600000"]


def test_off_is_nothing_in_train_learner_and_update():
    import train
    for argv in (["--mode", "ppo"], ["--mode", "ppo"] + LIVE_EXTRA, ["--mode", "ppo", "--human-imitation-coef", "0"]):
        args = train.parse_args(argv)
        assert train.human_imitation_settings(args) is None
    assert train.parse_args(["--mode", "ppo"]).human_imitation_coef is None
    # the flags change no part of the config: the run's config is what it was
    assert train.build_config(train.parse_args(LIVE_EXTRA + ["--human-imitation-coef", "0.01", "--human-imitation-data", "/x"])) \
        == train.build_config(train.parse_args(LIVE_EXTRA))
    import learner
    params = inspect.signature(learner.LearnerActor.__init__).parameters
    # human_imitation keeps its place after cfg and weight_store; the keywords the speed flags added after it
    # (v2-fast: fast_update, fast_ingest) are off by default, so a learner made without them is the one of before
    names = list(params)
    assert names[:4] == ["self", "cfg", "weight_store", "human_imitation"] and params["human_imitation"].default is None
    assert all(params[n].default is False for n in names[4:]), names
    import ppo_core
    assert inspect.signature(ppo_core.ppo_update).parameters["human"].default is None
    # main() hands the learner the argument only when the term is on
    src = inspect.getsource(train.main)
    assert "learner_kwargs = {}" in src and "if hum is not None:" in src


def test_values_that_cannot_be_are_refused_and_data_are_not_read_at_parse(tmp_path):
    import train
    for bad in (["--human-imitation-coef", "0.01"],                                   # no data directory
                ["--human-imitation-coef", "-0.01", "--human-imitation-data", str(tmp_path)],
                ["--human-imitation-coef", "0.01", "--human-imitation-data", str(tmp_path), "--human-imitation-switch-weight", "0"],
                ["--human-imitation-coef", "0.01", "--human-imitation-data", str(tmp_path), "--human-imitation-cap", "2"],
                ["--human-imitation-coef", "0.01", "--human-imitation-data", str(tmp_path), "--human-imitation-every", "0"],
                ["--human-imitation-coef", "0.01", "--human-imitation-data", str(tmp_path), "--human-imitation-pool", "10"],
                # the SetHead part without the SetHead: it would need a head of its own (a new optimizer layout)
                ["--obs-matchup", "--human-imitation-coef", "0.01", "--human-imitation-data", str(tmp_path),
                 "--human-imitation-set-coef", "0.1"],
                # values that are not finite: inf passes "above zero", and its gradient is inf x 0 = NaN
                ["--human-imitation-coef", "inf", "--human-imitation-data", str(tmp_path)],
                ["--human-imitation-coef", "nan", "--human-imitation-data", str(tmp_path)],
                ["--human-imitation-coef", "0.01", "--human-imitation-data", str(tmp_path), "--human-imitation-switch-weight", "inf"],
                ["--human-imitation-coef", "0.01", "--human-imitation-data", str(tmp_path),
                 "--human-imitation-double-switch-weight", "inf"],
                ["--human-imitation-coef", "0.01", "--human-imitation-data", str(tmp_path), "--human-imitation-cap", "nan"],
                ["--obs-matchup", "--set-head-coef", "0.1", "--human-imitation-coef", "0.01", "--human-imitation-data",
                 str(tmp_path), "--human-imitation-set-coef", "inf"]):
        with pytest.raises(SystemExit):
            train.parse_args(bad)
    for name in ("coef", "switch_weight", "double_switch_weight", "set_coef"):
        with pytest.raises(ValueError, match="finite"):
            hi.HumanImitationSettings(data=str(tmp_path), **dict({"coef": 0.01}, **{name: float("inf")})).check()
    # a directory that is not there is not refused at parse: the learner says so and runs without the term
    args = train.parse_args(["--human-imitation-coef", "0.02", "--human-imitation-data", str(tmp_path / "nope")])
    assert train.human_imitation_settings(args)["data"] == str(tmp_path / "nope")
    args = train.parse_args(LIVE_EXTRA + ["--human-imitation-coef", "0.02", "--human-imitation-data", str(tmp_path),
                                          "--human-imitation-switch-weight", "2", "--human-imitation-double-switch-weight", "5",
                                          "--human-imitation-shuffled", "--human-imitation-set-coef", "0.05",
                                          "--human-imitation-every", "2"])
    s = train.human_imitation_settings(args)
    assert s["coef"] == 0.02 and s["switch_weight"] == 2 and s["double_switch_weight"] == 5 and s["shuffled"] is True
    assert s["seqs"] == 1 and s["every"] == 2 and s["cap"] == 0.03 and s["set_coef"] == 0.05 and s["pool_decisions"] == 120_000
    assert "coef 0.02 SHUFFLED" in train.human_imitation_line(s) and "SetHead x0.05" in train.human_imitation_line(s)


# ---------------------------------------------------------------------------------------------
# labels and weights
# ---------------------------------------------------------------------------------------------

def test_double_switch_rows_need_a_choice_switch_and_the_opponents_quiet_switch_next():
    rng = np.random.default_rng(0)
    o, a, m, _t = _episode(12, rng, switch_rows=(2, 5, 8, 11), opp_switch_next=(3, 6, 9), forced=())
    o[6, OPP + hi.IDX_DID_MOVE] = 1            # row 5: the opponent moved too (a U-turn-like turn): not both switching
    m[8, 0:4] = 0
    m[8, 10:14] = 0                            # row 8: a forced switch
    starts = np.array([0, 12])
    dsw = hi.double_switch_rows(o.astype(np.float32), a.astype(np.int64), m.astype(np.float32), starts, TRANS0)
    assert np.flatnonzero(dsw).tolist() == [2]
    # the last row of an episode has no next row: never a double switch, whatever comes after it
    starts = np.array([0, 3, 12])
    dsw = hi.double_switch_rows(o.astype(np.float32), a.astype(np.int64), m.astype(np.float32), starts, TRANS0)
    assert 2 not in np.flatnonzero(dsw).tolist()


def test_shuffle_permutes_labels_only_among_rows_with_the_same_legal_mask():
    rng = np.random.default_rng(3)
    n = 400
    mask = np.zeros((n, 14), np.float32)
    mask[:, :4] = 1
    mask[n // 2:, 4:9] = 1
    act = np.where(np.arange(n) < n // 2, rng.integers(0, 4, n), rng.integers(0, 9, n)).astype(np.int64)
    lab = np.ones(n, bool)
    lab[::7] = False
    out = hi.shuffle_labels(act, mask, lab, np.random.default_rng(0))
    assert (out[~lab] == act[~lab]).all()
    for half in (slice(0, n // 2), slice(n // 2, n)):
        sel = np.zeros(n, bool)
        sel[half] = True
        assert sorted(out[sel & lab].tolist()) == sorted(act[sel & lab].tolist())
    assert (out != act).mean() > 0.3
    assert (mask[np.arange(n), out] > 0.5).all()


def _check_roles_and_weights(h):
    idx = h.index
    assert idx["roles"]["train"] == 12 and idx["roles"]["test_player"] == 3 and idx["roles"]["unused"] == 4, dict(idx["roles"])
    assert idx["mismatch"] == 0
    # "big" holds 8 of the 12 train sides: weighted down to the cap of 0.5
    lab_total = idx["train_labelled"]
    big = idx["player_lab"]["big"]
    assert idx["player_w"]["big"] == pytest.approx(0.5 * lab_total / big)
    assert idx["player_w"]["u41"] == 1.0
    for sh in h._pool:
        for k, (a, b) in enumerate(sh.eps):
            pw = idx["player_w"][sh.players[k]]
            w = sh.w[a:b]
            assert w[3] == pytest.approx(pw * 2.0 * 5.0)        # a switch with the opponent's quiet switch next
            assert w[10] == pytest.approx(pw * 2.0)             # a switch
            assert w[0] == pytest.approx(pw)                    # a move
            assert sh.dsw[a + 3] and not sh.dsw[a + 10]
            assert w[20] == pytest.approx(pw * 2.0)             # a forced switch: a switch, never a double switch
    # nothing of the held-out players, their opponents (loose) or test_game is in the pool
    assert {p for sh in h._pool for p in sh.players} <= {"big", "u41", "u42", "u51", "u52"}
    # the held-out set: the held-out players' own sides only (p1 of each test_player game)
    assert h.holdout is not None and h.holdout["players"] == 3 and h.ho["n"] == 3 * 25


def test_index_roles_from_the_shards_own_split_cap_and_weights(tmp_path):
    # the split files are wrong on purpose: a shard with ep_split never reads them
    h = _provider(tmp_path, switch_weight=2.0, double_switch_weight=5.0, splits=str(tmp_path / "no_such_dir"))
    h.start(TinyNet(), 64)
    assert _wait(h) == "on", h.off_reason
    _check_roles_and_weights(h)
    h.stop()


def test_stage3_shards_without_ep_split_read_the_split_files(tmp_path):
    h = _provider(tmp_path, data_kw={"ep_split": False}, switch_weight=2.0, double_switch_weight=5.0)
    h.start(TinyNet(), 64)
    assert _wait(h) == "on", h.off_reason
    _check_roles_and_weights(h)
    h.stop()
    # and without readable split files such shards have no roles: off, with the reason
    (tmp_path / "b").mkdir()
    h2 = _provider(tmp_path / "b", data_kw={"ep_split": False}, splits=str(tmp_path / "nope"))
    h2.start(TinyNet(), 64)
    assert _wait(h2) == "off" and "no train sides" in h2.off_reason and "split files" in h2.off_reason, h2.off_reason
    h2.stop()


@pytest.mark.parametrize("built", [
    {"matchup": False, "movesets": False},       # stage3's shards: no matchup block
    {"movesets": False},                         # matchup only
    {"own_tera": False},
    {"abil": "abil_sets"},                       # stage3's abilities fixes (this line has none)
    {"obs_dim": D + 1},                          # the same flags, another width (another code's blocks)
    {"matchup": None, "movesets": None},         # no fields at all: off
])
def test_a_shard_built_otherwise_is_never_loaded(tmp_path, built):
    h = _provider(tmp_path, data_kw=built)
    h.start(TinyNet(), 64)
    assert _wait(h) == "off"
    assert "rebuild" in h.off_reason and h.index["mismatch"] == 2, h.off_reason
    assert h.report() == " hum=off" and h.next_batch() is None and not h.wants(0)


def test_shards_that_do_not_say_their_width_are_checked_row_by_row(tmp_path):
    h = _provider(tmp_path, data_kw={"obs_dim": None})
    h.start(TinyNet(), 64)
    assert _wait(h) == "on", h.off_reason
    h.stop()
    net = TinyNet()
    net.unpacker = SimpleNamespace(offsets=dict(OFFSETS, action_mask=(A0, A1 + 1)))     # one column wider
    (tmp_path / "b").mkdir()
    h2 = _provider(tmp_path / "b", data_kw={"obs_dim": None})
    h2.start(net, 64)
    assert _wait(h2) == "off" and "no train shard could be loaded" in h2.off_reason, h2.off_reason


def test_a_missing_directory_is_off_with_the_reason(tmp_path):
    h = hi.HumanImitation(dict(data=str(tmp_path / "nope"), coef=0.1), RUN_OBS)
    h.start(TinyNet(), 64)
    assert _wait(h) == "off" and "no shards" in h.off_reason
    assert h.report() == " hum=off"


# ---------------------------------------------------------------------------------------------
# the term
# ---------------------------------------------------------------------------------------------

def test_batches_are_whole_sequences_with_weightless_padding(tmp_path):
    h = _provider(tmp_path, seqs=2, set_coef=0.1)
    net = TinyNet()
    net.set_head = object()                      # the SetHead part is kept only when the network has a head
    h.start(net, 64)
    assert _wait(h) == "on"
    b = h.next_batch()
    assert b["obs"].shape == (128, D) and b["ep"].shape == (128,)
    pad = b["ep"] >= hi.PAD_EPISODE
    assert pad.any()
    assert (b["w"][pad] == 0).all() and len(set(b["ep"][pad].tolist())) == int(pad.sum())
    assert np.array_equal(b["obs"][:, A0:A1].astype(np.float32) > 0.5, b["mask"] > 0.5)    # padding copies a row whole
    real_eps = set(b["ep"][~pad].tolist())
    assert all(e < hi.PAD_EPISODE for e in real_eps)
    # the SetHead's labels: the shards' rows, and nothing on padding
    assert b["set_labels"].shape == (128, LAB_W)
    assert (b["set_labels"][pad] == 0).all() and (b["set_labels"][~pad, 0] > 0).all()
    h.stop()


def _same_batch(h):
    """The batch the term will take next (no begin_update: its repack would race the snapshot)."""
    with h._lock:
        at, seqs = h._seq_at, list(h._seqs)
    return hi.gather(seqs[at:at + h.s.seqs], h.L, with_set_labels=h.set_coef > 0)


def test_the_term_is_coef_times_the_weighted_ce_and_trains(tmp_path):
    h = _provider(tmp_path, seqs=1, coef=0.5)
    net = TinyNet()
    h.start(net, 64)
    assert _wait(h) == "on"
    b = _same_batch(h)
    logits = net.lin(torch.from_numpy(b["obs"]).float())
    logp = torch.log_softmax(logits.masked_fill(torch.from_numpy(b["mask"]) <= 0.5, -1e4), -1)
    w = torch.from_numpy(b["w"])
    ce = float((w * -logp.detach().gather(1, torch.from_numpy(b["act"])[:, None]).squeeze(1)).sum() / w.sum())
    term = h.term(net, torch.device("cpu"), torch.float32, 64, True)
    assert float(term.detach()) == pytest.approx(0.5 * ce, rel=1e-5)
    term.backward()
    assert net.lin.weight.grad is not None and float(net.lin.weight.grad.abs().sum()) > 0
    assert net.v.weight.grad is None                     # nothing trains the value on the human games
    assert h.track["n"] == int((w > 0).sum())
    net.zero_grad()
    h.holdout_agreement(net, torch.device("cpu"), torch.float32, 7)
    assert 0.0 <= h.ho["agree"] <= 1.0 and h.ho["upd"] == 7 and h.ho["agree_pp"] is not None and h.ho["ce"] > 0
    line = h.report()
    for key in ("hum_n=", "hum_ce=", "hum_agree=", "hum_sw=", "hum_dsw=", "hum_ho=", "hum_ho_pp=", "hum_ho_ce=", "hum_ho_upd=7",
                "hum_total=", "hum_pool=", "hum_wait=0", "hum_err=0", "hum_nonfinite=0", "hum_drop=0"):
        assert key in line, (key, line)
    assert f"hum_ce={ce:.4f}" in line and "hum_set" not in line
    assert int(line.split("hum_sw=")[1].split()[0]) > 0
    # the supervisor reads every figure as a finite number (no nan, which its NaN guard would count)
    import pipeline
    ev = pipeline.parse_trainer_line("[learner] upd=7 kl=0.0010 loss=1.0" + line)
    assert ev["kind"] == "learner_update" and ev["nonfinite"] == [] and ev["values"]["hum_ho_upd"] == 7
    h.stop()


def test_a_term_that_is_not_finite_is_counted_and_not_hidden(tmp_path):
    # Only a network that is already not finite can give one (the data are checked at load). Its
    # gradient is NaN whatever its value, so it goes into the loss as it is: the update's loss
    # is NaN at the update it happened, and the supervisor's NaN guard acts on that update.
    h = _provider(tmp_path, coef=0.5)
    net = TinyNet()
    h.start(net, 64)
    assert _wait(h) == "on"
    with torch.no_grad():
        net.lin.bias[:] = float("nan")
    term = h.term(net, torch.device("cpu"), torch.float32, 64, True)
    assert not torch.isfinite(term)
    line = h.report()
    assert "hum_nonfinite=1" in line and "hum_err=0" in line, line
    import pipeline
    ev = pipeline.parse_trainer_line("[learner] upd=3 kl=0.0010 loss=nan" + line)
    assert ev["nonfinite"] == ["loss"]                    # the human figures themselves stay numbers
    h.stop()


def _bad_value_data(tmp_path, value):
    """One shard: 4 train games (8 sides), 2 held-out sides; a weightless history row of one train side and
    of one held-out side holds `value` in an observation column."""
    rng = np.random.default_rng(5)
    data = tmp_path / "data"
    data.mkdir()
    eps = []
    for g in range(4):
        for side in (1, 2):
            o, a, m, t = _episode(30, rng, switch_rows=(3,))
            eps.append((o, a, m, t, f"g{g}", f"t{g}{side}", side, "train"))
    for g in range(2):
        o, a, m, t = _episode(20, rng)
        eps.append((o, a, m, t, f"h{g}", f"ho{g}", 1, "test_player"))
    for k in (2, 8):                       # train side t11, held-out side ho0: row 7, a history row (no label)
        eps[k][0][7, 5] = value
        eps[k][3][7] = 0
    _write_shard(str(data / "shard-00000-cccc.npz"), eps)
    return str(data)


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_an_episode_with_a_value_that_is_not_finite_is_dropped_at_load_and_gradients_stay_finite(tmp_path, value):
    data = _bad_value_data(tmp_path, value)
    h = hi.HumanImitation(dict(data=data, coef=0.5, cap=1.0, pool_decisions=1000, holdout_decisions=1000), RUN_OBS)
    net = TinyNet()
    h.start(net, 64)
    assert _wait(h) == "on", h.off_reason
    # the two sides with the bad row are not loaded, the other 7 train sides and 1 held-out side are
    assert h.dropped == 2
    assert sorted(p for sh in h._pool for p in sh.players) == sorted(f"t{g}{s}" for g in range(4) for s in (1, 2) if (g, s) != (1, 1))
    assert all(np.isfinite(sh.obs).all() for sh in h._pool)
    assert h.holdout["players"] == 1 and h.ho["n"] == 20
    # every batch the term can draw is finite, and so is every gradient
    for _ in range(len(h._seqs) + 1):
        net.zero_grad(set_to_none=True)
        term = h.term(net, torch.device("cpu"), torch.float32, 64, True)
        assert term is not None and torch.isfinite(term)
        term.backward()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    line = h.report()
    assert "hum_drop=2" in line and "hum_nonfinite=0" in line, line
    h.stop()


def test_why_the_rows_are_dropped_a_zeroed_term_still_has_a_nan_gradient():
    # The mechanism the load check exists for (review of 75e55f2): a weightless row with an inf
    # observation, its term zeroed with torch.where, still leaves every weight's gradient NaN.
    net = TinyNet()
    obs = torch.zeros(4, D)
    obs[1, 5] = float("inf")
    w = torch.tensor([1.0, 0.0, 1.0, 1.0])
    logp = torch.log_softmax(net.lin(obs), -1)
    term = (w * -logp[:, 0]).sum() / w.sum()
    term = torch.where(torch.isfinite(term), term, torch.zeros_like(term))
    assert float(term.detach()) == 0.0
    term.backward()
    assert torch.isnan(net.lin.weight.grad).all()


def test_the_held_out_set_caps_each_player(tmp_path):
    # one prolific held-out account first in the shard, then 20 others: without the cap it would be the whole set
    rng = np.random.default_rng(9)
    data = tmp_path / "data"
    data.mkdir()
    eps = []
    for g in range(2):
        o, a, m, t = _episode(30, rng)
        eps.append((o, a, m, t, f"t{g}", f"t{g}", 1, "train"))
    for g in range(30):
        o, a, m, t = _episode(10, rng)
        eps.append((o, a, m, t, f"w{g}", "whale", 1, "test_player"))
    for p in range(20):
        for g in range(2):
            o, a, m, t = _episode(10, rng)
            eps.append((o, a, m, t, f"o{p}-{g}", f"o{p}", 1, "test_player"))
    _write_shard(str(data / "shard-00000-dddd.npz"), eps)
    h = hi.HumanImitation(dict(data=str(data), coef=0.5, cap=0.2, pool_decisions=1000, holdout_decisions=200), RUN_OBS)
    h.start(TinyNet(), 64)
    assert _wait(h) == "on", h.off_reason
    # at most 0.2 x 200 = 40 decisions each: the whale's first 4 episodes, then 2 of each other player's
    assert h.holdout["limit"] == 40 and h.ho["n"] == 200
    assert h.holdout["players"] == 9 and h.holdout["top_share"] == pytest.approx(0.2)
    cnt = {}
    for seq in h.holdout["seqs"]:
        for sh, a, b, _u in seq:
            k = next(i for i, (ea, eb) in enumerate(sh.eps) if ea <= a < eb)
            cnt[sh.players[k]] = cnt.get(sh.players[k], 0) + int(sh.lab[a:b].sum())
    assert cnt["whale"] == 40 and max(cnt.values()) == 40 and sum(cnt.values()) == 200
    # a player's first episode is taken even when it alone is past the limit
    h.stop()
    h2 = hi.HumanImitation(dict(data=str(data), coef=0.5, cap=0.01, pool_decisions=1000, holdout_decisions=200), RUN_OBS)
    h2.start(TinyNet(), 64)
    assert _wait(h2) == "on", h2.off_reason
    assert h2.holdout["limit"] == 2 and h2.holdout["players"] == 20 and h2.ho["n"] == 200
    h2.stop()


def test_the_sethead_part_adds_the_heads_loss_on_the_shards_labels(tmp_path, monkeypatch):
    import ppo_core
    seen = {}

    def fake_set_loss(net, z_tokens, labels, coef=None):
        seen["labels"], seen["coef"], seen["z"] = labels.detach().clone(), coef, z_tokens.shape
        loss = z_tokens.float().mean() * 0 + 2.0
        return coef * loss, {"set_loss": loss.detach(), "set_ability_n": torch.tensor(3.0), "set_item_n": torch.tensor(4.0)}

    monkeypatch.setattr(ppo_core, "compute_set_head_loss", fake_set_loss)
    h = _provider(tmp_path, seqs=1, coef=0.5, set_coef=0.25)
    net = TinyNet()
    net.set_head = object()
    h.start(net, 64)
    assert _wait(h) == "on"
    b = _same_batch(h)
    term = h.term(net, torch.device("cpu"), torch.float32, 64, True)
    plain = hi.HumanImitation(dict(h.s.as_dict(), set_coef=0.0), RUN_OBS)       # the CE alone, on the same batch
    logits = net.lin(torch.from_numpy(b["obs"]).float())
    logp = torch.log_softmax(logits.masked_fill(torch.from_numpy(b["mask"]) <= 0.5, -1e4), -1)
    w = torch.from_numpy(b["w"])
    ce = float((w * -logp.detach().gather(1, torch.from_numpy(b["act"])[:, None]).squeeze(1)).sum() / w.sum())
    assert float(term.detach()) == pytest.approx(0.5 * ce + 0.25 * 2.0, rel=1e-5) and plain.set_coef == 0.0
    assert seen["coef"] == 0.25 and tuple(seen["labels"].shape) == (64, 6, 8) and seen["z"] == (64, 13, 4)
    assert torch.equal(seen["labels"].reshape(64, -1), torch.from_numpy(b["set_labels"]))
    line = h.report()
    assert "hum_set_loss=2.0000" in line and "hum_set_n=7" in line, line
    h.stop()
    # without a SetHead on the network the part is off (never a head of its own)
    (tmp_path / "b").mkdir()
    h2 = _provider(tmp_path / "b", set_coef=0.25)
    h2.start(TinyNet(), 64)
    assert h2.set_coef == 0.0
    h2.stop()


def _toy_update(human=None, seed=0):
    """One ppo_update on a stand-in network with ppo_update's own code."""
    import ppo_core
    torch.manual_seed(seed)
    net = TinyNet()
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    ds = ppo_core.AsyncEpisodeDataset(14, "cpu")
    n = 128
    g = torch.Generator().manual_seed(seed)
    obs = torch.zeros(n, D)
    obs[:, A0:A0 + 7] = 1
    act = torch.randint(0, 7, (n,), generator=g)
    ds.add_steps(obs, act, torch.full((n,), -1.9), torch.zeros(n), torch.randn(n, generator=g), torch.randn(n, generator=g),
                 torch.zeros(n), torch.repeat_interleave(torch.arange(4), 32))
    stats = ppo_core.ppo_update(net=net, opt=opt, dataset=ds, mode="ppo", update_epochs=1, minibatch_size=64,
                                batch_seq_len=64, clip_coef=0.2, vf_coef=0.5, ent_coef=0.0, max_grad_norm=1.0,
                                v_min=-1.0, v_max=1.0, v_bins=5, data_on_device=False, **({"human": human} if human else {}))
    return net, stats


def _fixed_minibatches(ds, mb):
    """The dataset's minibatches in order (no shuffling), so two updates see the same data."""
    obs, act, logp, val, adv, ret, nhp, ep = ds.tensorize()[:8]
    for i in range(0, len(act) - mb + 1, mb):
        yield (obs[i:i + mb], act[i:i + mb], logp[i:i + mb], val[i:i + mb], adv[i:i + mb], ret[i:i + mb], nhp[i:i + mb], ep[i:i + mb])


def test_ppo_update_with_human_none_or_not_ready_is_bit_for_bit_and_with_it_changes(tmp_path, monkeypatch):
    import ppo_core
    monkeypatch.setattr(ppo_core.AsyncEpisodeDataset, "iter_minibatches",
                        lambda self, mb, shuffle_episodes=True, device=None: _fixed_minibatches(self, mb))
    net0, s0 = _toy_update()
    net1, s1 = _toy_update(human=None)
    for a, b in zip(net0.parameters(), net1.parameters()):
        assert torch.equal(a, b)
    assert s0 == s1
    # a term still loading (or off) is never asked: the update is the one without it
    loading = hi.HumanImitation(dict(data=str(tmp_path / "x"), coef=1.0), RUN_OBS)
    assert loading.status == "starting" and not loading.wants(0)
    net_l, s_l = _toy_update(human=loading)
    for a, b in zip(net0.parameters(), net_l.parameters()):
        assert torch.equal(a, b)
    assert s_l == s0
    h = _provider(tmp_path, coef=1.0)
    h.start(net0, 64)
    assert _wait(h) == "on"
    h.begin_update()
    net2, s2 = _toy_update(human=h)
    assert any(not torch.equal(a, b) for a, b in zip(net0.parameters(), net2.parameters()))
    assert h.track["n"] > 0 and h.track["err"] == 0

    # a failing term is counted and left out: the update is the one without it
    class Broken:
        track = {"err": 0}

        def wants(self, i):
            return True

        def term(self, *a, **k):
            raise RuntimeError("boom")
    br = Broken()
    net3, s3 = _toy_update(human=br)
    assert br.track["err"] == 2
    for a, b in zip(net0.parameters(), net3.parameters()):
        assert torch.equal(a, b)
    # every k-th minibatch only
    h.begin_update()
    h.every = 2
    asked = []
    orig = h.term

    def counting(*a, **k):
        asked.append(1)
        return orig(*a, **k)
    h.term = counting
    _toy_update(human=h)
    assert len(asked) == 1                       # minibatches 0 and 1: only 0
    h.stop()


# ---------------------------------------------------------------------------------------------
# a resume with the term on
# ---------------------------------------------------------------------------------------------

def _learner(**over):
    import test_speed_update as tsu
    return tsu._new_learner("ppo", save_every_updates=10**9, **over)


@pytest.mark.parametrize("heads", [{}, {"opp_action_aux_coef": 0.1}])
def test_a_checkpoint_without_the_term_resumes_with_it_and_nothing_moves(tmp_path, heads, capsys):
    import test_speed_update as tsu
    a = _learner(**heads)
    # some Adam state and a schedule step, as a run has
    for _p, m in a.opt.pairs:
        m.grad = torch.full_like(m, 1e-3)
    a.opt.clip_and_step(None)            # Adam's step on the masters, rounded back into the model, as an update does
    a.sched.step()
    a.update_idx, a.total_steps, a.total_episodes = 7748, 286_661_547, 9_064_976
    path = str(tmp_path / "learner_update_007748.pt")
    a._save_checkpoint(path)
    assert "human_imitation" not in torch.load(path, map_location="cpu", weights_only=False)
    data, _splits, _salt = _data(tmp_path)
    b = _learner(**heads)
    b.human = hi.HumanImitation(dict(data=data, coef=0.01), b.run_cfg.obs)
    b._load_checkpoint(path)
    out = capsys.readouterr().out
    assert "Optimizer state loaded successfully." in out and "topology mismatch" not in out, out
    sa, sb = tsu._state(a), tsu._state(b)
    assert sa.keys() == sb.keys()
    assert [k for k in sa if not torch.equal(sa[k], sb[k])] == []           # weights, masters, Adam, every rate, the step
    assert (b.update_idx, b.total_steps, b.total_episodes) == (7748, 286_661_547, 9_064_976)
    assert b.sched.last_epoch == a.sched.last_epoch
    # its checkpoints then say which term trained them
    b._save_checkpoint(str(tmp_path / "learner_update_007749.pt"))
    ck = torch.load(str(tmp_path / "learner_update_007749.pt"), map_location="cpu", weights_only=False)
    assert ck["human_imitation"]["coef"] == 0.01
    # and back: a checkpoint written with the term resumes without it, the same way
    c = _learner(**heads)
    c._load_checkpoint(str(tmp_path / "learner_update_007749.pt"))
    assert "Optimizer state loaded successfully." in capsys.readouterr().out
    assert [k for k in sb if not torch.equal(sb[k], tsu._state(c)[k])] == []


# ---------------------------------------------------------------------------------------------
# check_restart
# ---------------------------------------------------------------------------------------------

def _live_config(extra_ppo):
    def edit(raw):
        raw["train_defaults"]["extra_train_args"] = LIVE_EXTRA[:6]
        raw["phases"]["ppo"]["extra_train_args"] = LIVE_EXTRA[6:] + list(extra_ppo)
    return edit


def _live_ckpt():
    import test_check_restart as tcr
    ck = tcr._ckpt(390_000, total=1_600_000, hold=800_000, lr=3e-4)
    ck["obs_settings"] = {"own_tera_type": True, "matchup": True, "movesets": True}
    return ck


def test_check_restart_says_ok_for_the_live_arguments_with_the_term_and_refuses_bad_data(tmp_path, monkeypatch):
    import test_check_restart as tcr
    import check_restart
    from config import RunConfig
    live = dataclasses.replace(RunConfig.default(), obs=dataclasses.replace(RunConfig.default().obs, matchup=True, movesets=True))
    dim = check_restart.run_obs_dim(live)
    assert dim == 10_295                          # the width the live run's shards record (obs_dim)
    data, _s, _salt = _data(tmp_path, obs_dim=dim)
    on = ["--human-imitation-coef", "0.01", "--human-imitation-data", data]
    code, lines = tcr._run(tmp_path, monkeypatch, None, _live_ckpt(), edit=_live_config(on))
    assert code == 0 and lines[-1].startswith("OK to start"), lines
    assert any(ln.startswith("human imitation: coef 0.01") for ln in lines), lines
    assert any(ln.startswith("human data:      2 of 2 shards fit") and "12 train sides" in ln and "3 held-out players' sides" in ln
               for ln in lines), lines
    # without the flags it says off
    code, lines = tcr._run(tmp_path, monkeypatch, None, _live_ckpt(), edit=_live_config([]))
    assert code == 0 and "human imitation: off" in lines, lines
    # shards of another observation (stage3's: no matchup block)
    (tmp_path / "other").mkdir()
    other, _s, _salt = _data(tmp_path / "other", matchup=False, movesets=False, obs_dim=dim)
    code, lines = tcr._run(tmp_path, monkeypatch, None, _live_ckpt(),
                           edit=_live_config(["--human-imitation-coef", "0.01", "--human-imitation-data", other]))
    assert code == 2 and "none of the 2 shards" in lines[-1] and "--obs-matchup --obs-movesets" in lines[-1], lines
    # the right flags, the wrong width
    (tmp_path / "wide").mkdir()
    wide, _s, _salt = _data(tmp_path / "wide", obs_dim=dim + 1)
    code, lines = tcr._run(tmp_path, monkeypatch, None, _live_ckpt(),
                           edit=_live_config(["--human-imitation-coef", "0.01", "--human-imitation-data", wide]))
    assert code == 2 and "none of the 2 shards" in lines[-1], lines
    # a directory that is not there
    code, lines = tcr._run(tmp_path, monkeypatch, None, _live_ckpt(),
                           edit=_live_config(["--human-imitation-coef", "0.01", "--human-imitation-data", str(tmp_path / "nope")]))
    assert code == 2 and "is not a directory" in lines[-1], lines
    # a coef without data: train.py refuses the command line
    code, lines = tcr._run(tmp_path, monkeypatch, None, _live_ckpt(), edit=_live_config(["--human-imitation-coef", "0.01"]))
    assert code == 2 and "train.py refuses this command line" in lines[-1], lines


# ---------------------------------------------------------------------------------------------
# GPU: flags off is the commit before, bit for bit
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def head():
    """The commit before the port (PSPPO_HEAD_DIR: its files), beside the working tree's."""
    if not HEAD_DIR or not os.path.isdir(HEAD_DIR):
        pytest.skip("PSPPO_HEAD_DIR names no copy of the commit before the port")
    from bench import micro_ref
    import ppo_core as new_core
    ref = micro_ref.Reference(HEAD_DIR, source="PSPPO_HEAD_DIR")
    core, lrn, cfg = ref.load("ppo_core"), ref.load("learner"), ref.load("config")
    assert lrn.ppo_core is core and core is not new_core
    assert "human" not in inspect.signature(core.ppo_update).parameters, "PSPPO_HEAD_DIR holds this branch's code"
    if GPU:
        micro_ref.shared_compiled_flex_attention(core, new_core)
    core._learner_flex_compiled = True
    new_core._learner_flex_compiled = True
    return SimpleNamespace(core=core, learner=lrn, config=cfg, ref=ref, micro_ref=micro_ref)


@pytest.fixture(scope="module")
def obs_data():
    from bench import micro_ref
    return micro_ref.observations(n_min=4096)


@needs_gpu
@pytest.mark.parametrize("features", ["off", "live"])
def test_with_the_flags_off_a_whole_update_is_bit_identical_to_the_commit_before(head, obs_data, features):
    import test_speed_update as tsu
    obs_np, ep_np, _kind = obs_data
    n = 1536
    cols = tsu._update_data(obs_np, ep_np, n, seed=46)
    over = {"save_every_updates": 10**9}
    extras = None
    if features == "live":
        # the live run's learner features that this observation allows (the SetHead needs the matchup block)
        over.update(opp_action_aux_coef=0.1, boosted_row_weight=2.0, read_bonus=0.3, read_anneal_start=20_000_000,
                    read_anneal_end=45_000_000)
        rng = np.random.default_rng(8)
        extras = {"read": torch.from_numpy(rng.choice([0.0, 0.6, -0.6], n).astype(np.float32)),
                  "opp_action": torch.from_numpy(rng.integers(-1, 8, n).astype(np.int8)),
                  "boosted": (torch.arange(n) % 4 == 0).to(torch.int8)}
    with head.ref.active():
        old = tsu._make_learner(head.learner, head.core, tsu._learner_cfg(head.config, "ppo", **over))
    new = tsu._new_learner("ppo", **over)
    tsu._copy_state(old, new)
    for la in (old, new):
        if extras is None:
            la.dataset.add_steps(*cols)
        else:
            la.dataset.add_steps(*cols, extras=extras)
        la.total_steps += n
        torch.manual_seed(79)
        asyncio.run(la._perform_update())
        assert la.update_idx == 1 and len(la.dataset) == 0, "the update did not run"
    want, got = tsu._state(old), tsu._state(new)
    assert want.keys() == got.keys()
    diff = [k for k in want if not head.micro_ref.same_bits(want[k], got[k])]
    assert diff == [], diff[:5]
    del old, new
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------
# GPU, real shards: the real network, the learner's own update
# ---------------------------------------------------------------------------------------------

def _real_cfg(**learner_over):
    from config import RunConfig
    cfg = RunConfig.default()
    files = hi.shard_files(REAL)
    with np.load(files[0]) as z:
        tera, matchup, movesets, _abil = hi.shard_obs_settings(z)
    cfg = dataclasses.replace(cfg, obs=dataclasses.replace(cfg.obs, own_tera_type=tera, matchup=matchup, movesets=movesets))
    return dataclasses.replace(cfg, learner=dataclasses.replace(cfg.learner, **learner_over)) if learner_over else cfg


def _compiled(net):
    import ppo_core
    if not getattr(ppo_core, "_learner_flex_compiled", False):
        ppo_core.flex_attention = torch.compile(ppo_core.flex_attention)
        ppo_core._learner_flex_compiled = True
    net.enable_bf16_recurrent_path()
    return net


@needs_gpu
@pytest.mark.skipif(not os.path.isdir(REAL), reason="PSPPO_HUMAN_SHARDS not set (a directory of human/rebuild.py shards)")
def test_the_real_network_on_real_shards():
    dev = torch.device("cuda")
    cfg = _real_cfg()
    net = _compiled(cfg.make_model().to(dev))
    if cfg.obs.matchup:
        net.attach_set_head(0.1)
    h = hi.HumanImitation(dict(data=REAL, coef=0.01, pool_decisions=20_000, holdout_decisions=2048, set_coef=0.1), cfg.obs)
    L = int(cfg.learner.batch_seq_len)
    h.start(net, L)
    assert _wait(h, 900) == "on", h.off_reason
    assert h.dim == int(cfg.make_assembler().total_dim)
    h.begin_update()
    b = h.next_batch()
    a0, a1 = net.unpacker.offsets["action_mask"]
    assert b["obs"].shape[1] == h.dim
    # the shards' masks are the observation's own mask block, the labels are legal
    assert np.array_equal(b["obs"][:, a0:a1].astype(np.float32) > 0.5, b["mask"] > 0.5)
    lab = b["w"] > 0
    assert (b["mask"][np.arange(len(b["act"])), b["act"]][lab] > 0.5).all()
    dt = torch.float16
    term = h.term(net, dev, dt, L, True)                 # the first call compiles
    term.backward()
    pi_grad = sum(float(p.grad.abs().sum()) for n, p in net.named_parameters() if p.grad is not None and "set_head" not in n)
    set_grad = sum(float(p.grad.abs().sum()) for p in net.set_head.parameters() if p.grad is not None) if cfg.obs.matchup else 1.0
    assert pi_grad > 0 and set_grad > 0
    net.zero_grad(set_to_none=True)
    times = {}
    for name, coef in (("ce", 0.0), ("ce+set", 0.1)):
        h.set_coef = coef
        ts = []
        for _ in range(7):
            torch.cuda.synchronize()
            t0 = time.time()
            term = h.term(net, dev, dt, L, True)
            term.backward()
            torch.cuda.synchronize()
            ts.append(time.time() - t0)
            net.zero_grad(set_to_none=True)
        times[name] = sorted(ts)[len(ts) // 2]
    t0 = time.time()
    for _ in range(10):
        h.next_batch()
    gather_s = (time.time() - t0) / 10
    h.holdout_agreement(net, dev, dt, 1)
    print("real shards:", h.report(), f"term+backward median {times['ce']:.3f} s (with the SetHead part {times['ce+set']:.3f} s)"
          f" per minibatch, gather {gather_s * 1000:.1f} ms; roles {dict(h.index['roles'])}; train decisions "
          f"{h.index['train_labelled']:,} of {h.index['train_players']} players")
    assert h.track["n"] > 0 and 0 < h.ho["agree"] < 1
    roles = h.index["roles"]
    assert roles["train"] > 0 and roles["test_player"] > 0
    assert sum(sh.rows() for sh in h._pool) >= 20_000 or len(h._pool) == len(h.index["train_shards"])
    # no held-out player's side and no loose side in the pool
    pool_players = {p for sh in h._pool for p in sh.players}
    ho_players = {p for path in h.index["holdout_shards"] for _a, _b, p in h.index["episodes"][path]["test_player"]}
    assert not (pool_players & ho_players)
    h.stop()


@needs_gpu
@pytest.mark.skipif(not os.path.isdir(REAL), reason="PSPPO_HUMAN_SHARDS not set (a directory of human/rebuild.py shards)")
def test_a_whole_learner_update_with_the_term_on_real_shards(capsys):
    """The learner's own _perform_update, the live run's heads, rollouts made of the shards' observations."""
    import test_speed_update as tsu
    import learner as new_learner
    import ppo_core as new_core
    import pipeline
    cfg = _real_cfg(mode="ppo", device="cuda", minibatch_size=768, batch_seq_len=256, grad_accum_steps=1, steps_per_update=3072,
                    resume=False, save_every_updates=10**9, opp_action_aux_coef=0.1, boosted_row_weight=2.0,
                    set_head_coef=0.1 if _real_cfg().obs.matchup else 0.0)
    la = tsu._make_learner(new_learner, new_core, cfg)
    la.human = hi.HumanImitation(dict(data=REAL, coef=0.01, pool_decisions=20_000, holdout_decisions=2048), cfg.obs)
    la.human.start(la.net, 256)
    assert _wait(la.human, 900) == "on", la.human.off_reason
    # rollouts: the shards' own rows as the observations, a legal action, plausible targets
    sh = la.human._pool[0]
    n = 3072
    rows = np.arange(n) % sh.rows()
    obs = torch.from_numpy(np.ascontiguousarray(sh.obs[rows]))
    rng = np.random.default_rng(4)
    act = torch.from_numpy(np.array([rng.choice(np.flatnonzero(m > 0.5)) for m in sh.mask[rows]], dtype=np.int64))
    logp = torch.from_numpy((-np.log(np.maximum((sh.mask[rows] > 0.5).sum(1), 1.0))).astype(np.float32))
    ep = torch.from_numpy(np.repeat(np.arange(n // 64), 64)).long()
    z = torch.zeros(n)
    extras = {"opp_action": torch.from_numpy(rng.integers(-1, 8, n).astype(np.int8)),
              "boosted": (torch.arange(n) % 4 == 0).to(torch.int8)}
    if cfg.obs.matchup:
        lab = sh.set_labels[rows] if sh.set_labels is not None else np.zeros((n, LAB_W), np.int32)
        extras["set_labels"] = torch.from_numpy(np.ascontiguousarray(lab))
    la.dataset.add_steps(obs, act, logp, z, torch.from_numpy(rng.standard_normal(n).astype(np.float32)),
                         torch.from_numpy(np.clip(rng.standard_normal(n) * 0.5, -1.4, 1.4).astype(np.float32)), z, ep, extras=extras)
    la.total_steps += n
    asyncio.run(la._perform_update())
    assert la.update_idx == 1
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[learner] upd="))
    print(line)
    ev = pipeline.parse_trainer_line(line)
    assert ev["kind"] == "learner_update" and ev["nonfinite"] == []
    v = ev["values"]
    assert v["hum_n"] > 0 and v["hum_err"] == 0 and v["hum_wait"] == 0 and 0 < v["hum_ho"] < 1 and v["hum_ho_upd"] == 1
    assert v["hum_nonfinite"] == 0 and v["hum_drop"] == 0
    assert "opp_n" in v and "boosted_share" in v
    la.human.stop()
    del la
    torch.cuda.empty_cache()
