"""
The stage 2-4 training features wired into the training loop (plan "teach the bot reads, set-up play,
abilities and items": Stages 2, 3, 4, "Wiring"). Every feature is behind a flag; with every flag off the
worker, the learner and the update are what they were.

    python -m pytest tests/test_train_v2.py -q -p no:cacheprovider -rs
    PSPPO_HEAD_DIR=~/dev/v2-head ...   (a copy of the commit before the wiring: the bit-identity tests)

Held here:
  * reads, worker side: one ReadTracker per battle and seat, fed the lines the worker harvests, decide()
    once per STEP, finalize() at the end; the read column (our tier minus the tier of being read) sums to
    zero over both seats of every self-play game; opp_action labels; the DONE event carries the columns
    into the trajectory and on to the learner;
  * reads, learner side: the bonus schedule (full, linear fade, 0), the per-game cap (scaled together,
    so the seats stay opposite), the bonus lands in the advantages before they are normalised and never
    in the returns, the log keys;
  * the opponent-action head and the SetHead: their losses, their zone and optimizer groups, the
    optimizer state of a checkpoint without them (or with them) carried over for every other parameter;
  * the boosted-row weight: the weights' math and W = 1 leaving the update bit-identical;
  * flags off: a whole _perform_update bit-identical to the commit before the wiring, with and without
    unused columns;
  * train.py's flags, the opponent seats' wiring, check_restart refusing a matchup change.

Whole updates need a GPU (the attention has no backward pass on a CPU) and are skipped without one.
Nothing here starts Ray, a server or a training run.
"""
from __future__ import annotations

import asyncio
import dataclasses
import gzip
import json
import os
import queue
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "tests"), os.path.join(ROOT, "cloud")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as new_config  # noqa: E402
import learner as new_learner  # noqa: E402
import ppo_core as new_core  # noqa: E402
import step_columns  # noqa: E402
import worker as new_worker  # noqa: E402
import test_speed_update as tsu  # noqa: E402
from bench import micro_ref  # noqa: E402
from bench.read_stats import selfplay_steps  # noqa: E402
from read_reward import OPP_NA  # noqa: E402

DEV = tsu.DEV
GPU = tsu.GPU
needs_gpu = pytest.mark.skipif(not GPU, reason="flex_attention has no backward on a CPU")
HEAD_DIR = os.path.expanduser(os.environ.get("PSPPO_HEAD_DIR", ""))
FIX = os.path.join(ROOT, "tests", "fixtures", "reads")


@pytest.fixture(scope="module")
def data():
    obs, ep, kind = micro_ref.observations(n_min=4096)
    return obs, ep, kind


# =========================================================================== reads: the worker


def _player(**learner_over):
    """A SyncBridgePlayer without poke-env's client: only the state the per-step columns use."""
    cfg = new_config.RunConfig.default()
    cfg = dataclasses.replace(cfg, learner=dataclasses.replace(cfg.learner, **learner_over))
    p = object.__new__(new_worker.SyncBridgePlayer)
    p.cfg = cfg
    lc = cfg.learner
    p.reads_on = lc.read_bonus > 0 or lc.opp_action_aux_coef > 0
    p.boosted_on = lc.boosted_row_weight != 1.0
    p.set_labels_on = lc.set_head_coef > 0 and cfg.obs.matchup
    p.step_features = p.reads_on or p.boosted_on or p.set_labels_on
    p._read_trackers, p._step_cols, p._battle_events = {}, {}, {}
    p.partner = None
    p._account_configuration = SimpleNamespace(username="tester")
    return p


def test_the_player_turns_the_columns_on_only_with_their_flags(monkeypatch):
    from poke_env.player import Player
    monkeypatch.setattr(Player, "__init__", lambda self, *a, **k: None)        # no client, no login
    cfg = new_config.RunConfig.default()
    mu = dataclasses.replace(cfg, obs=dataclasses.replace(cfg.obs, matchup=True))
    for base, over, want in ((cfg, {}, (False, False, False)), (cfg, {"read_bonus": 0.3}, (True, False, False)),
                             (cfg, {"opp_action_aux_coef": 0.1}, (True, False, False)),
                             (cfg, {"boosted_row_weight": 2.0}, (False, True, False)),
                             (cfg, {"set_head_coef": 0.1}, (False, False, False)),      # the SetHead needs the matchup block too
                             (mu, {"set_head_coef": 0.1}, (False, False, True)),
                             (cfg, {"read_bonus": 0.3, "mode": "imitation"}, (False, False, False))):
        c = dataclasses.replace(base, learner=dataclasses.replace(base.learner, **over))
        p = new_worker.SyncBridgePlayer(cfg=c, event_queue=queue.Queue(), action_futures={}, ep_sem=None)
        assert (p.reads_on, p.boosted_on, p.set_labels_on) == want, over
        assert p.step_features == any(want) and p.partner is None


def _parts(lines):
    return [ln.split("|") for ln in lines if ln.startswith("|")]


def _play_seat(game_entries, seat, **over):
    """Drive one seat of a recorded self-play game through the player's column code, as choose_move does."""
    p = _player(read_bonus=0.3, **over)
    tag = f"tester_b-{seat}"
    battle = SimpleNamespace(player_role=seat, force_switch=False, opponent_team={}, battle_tag="b1",
                             opponent_active_pokemon=None)
    pending, n = [], 0
    for st in selfplay_steps(game_entries, drop_turn_lines=True):
        if st[0] == "lines":
            pending.extend(_parts(st[1]))
        else:
            battle.force_switch = bool(st[2])
            idx = p._note_step(tag, battle, pending)
            p._note_action(tag, idx, st[1])
            pending, n = [], n + 1
    p._battle_events["b1"] = pending          # the last turn's lines, not yet taken by a decision
    cols = p._finish_step_columns(tag, battle)
    return cols, n, p


def _games():
    with gzip.open(os.path.join(FIX, "selfplay3.jsonl.gz"), "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_the_read_column_is_zero_sum_over_both_seats_of_every_game():
    total_reads = 0
    for g in _games():
        c1, n1, p1 = _play_seat(g["p1"], "p1")
        c2, n2, p2 = _play_seat(g["p2"], "p2")
        assert len(c1["read"]) == n1 and len(c2["read"]) == n2              # one value per STEP
        assert len(c1["opp_action"]) == n1 and c1["opp_action"].dtype == np.int8
        assert float(c1["read"].sum()) + float(c2["read"].sum()) == pytest.approx(0.0, abs=1e-6)
        assert {round(float(x), 3) for x in np.abs(c1["read"])} <= {0.0, 0.6, 1.0}
        assert ((c1["opp_action"] >= OPP_NA) & (c1["opp_action"] <= 7)).all()
        total_reads += int((c1["read"] > 0).sum() + (c2["read"] > 0).sum())
        assert not p1._read_trackers and not p1._step_cols                   # nothing left behind
    assert total_reads >= 6


def test_a_dropped_battle_leaves_nothing_and_a_broken_one_sends_no_columns():
    g = _games()[0]
    p = _player(read_bonus=0.3)
    battle = SimpleNamespace(player_role="p1", force_switch=False, opponent_team={}, battle_tag="b1", opponent_active_pokemon=None)
    p._note_step("t", battle, [])
    assert "t" in p._read_trackers
    p._drop_step_columns("t")
    assert not p._read_trackers and not p._step_cols
    bad = SimpleNamespace(player_role="p1", force_switch=False, battle_tag="b2")      # no opponent_team: _note_step fails
    assert p._note_step("u", bad, []) is None
    assert p._finish_step_columns("u", bad) is None
    assert p._finish_step_columns("never", bad) is None
    del g


def test_boosted_and_set_label_columns():
    from obs_matchup import build_set_labels  # noqa: F401
    p = _player(boosted_row_weight=2.0)
    mon = SimpleNamespace(boosts={"atk": 1, "def": 0, "spa": 0, "spd": 0, "spe": 0, "accuracy": 0, "evasion": 0})
    calm = SimpleNamespace(boosts={"atk": 0, "accuracy": 2})
    b = SimpleNamespace(player_role="p1", force_switch=False, opponent_team={}, battle_tag="b1", opponent_active_pokemon=mon)
    p._note_step("t", b, [])
    b.opponent_active_pokemon = calm
    p._note_step("t", b, [])
    b.opponent_active_pokemon = None
    p._note_step("t", b, [])
    cols = p._finish_step_columns("t", b)
    assert cols["boosted"].tolist() == [1, 0, 0] and cols["boosted"].dtype == np.int8 and "read" not in cols

    # set labels: the partner seat's battle of the same room holds the truth; no partner, no labels
    p = _player()
    p.set_labels_on = p.step_features = True
    p.assembler = SimpleNamespace(vocab_map=None)
    b = SimpleNamespace(player_role="p1", force_switch=False, opponent_team={}, battle_tag="b1", opponent_active_pokemon=None)
    assert p._set_labels(b).tolist() == [0] * 18
    seen = {}

    def fake_labels(battle, team, vocab_map=None):
        seen["team"] = team
        return np.arange(18, dtype=np.int32).reshape(6, 3)

    import obs_matchup
    orig = obs_matchup.build_set_labels
    obs_matchup.build_set_labels = fake_labels
    try:
        p.partner = SimpleNamespace(_battles={"b1": SimpleNamespace(team={"p2: X": "truth"})})
        p._note_step("t", b, [])
        cols = p._finish_step_columns("t", b)
    finally:
        obs_matchup.build_set_labels = orig
    assert seen["team"] == {"p2: X": "truth"}
    assert cols["set_labels"].shape == (1, 18) and cols["set_labels"].dtype == np.int32 and cols["set_labels"][0].tolist() == list(range(18))


def test_the_done_event_puts_the_columns_into_the_trajectory_and_on_to_the_learner():
    w = object.__new__(new_worker.RolloutWorker)
    w.cfg = new_config.RunConfig.default()
    w.calls = []
    w.learner_client = SimpleNamespace(submit_episode=lambda *a, **k: w.calls.append((a, k)))
    w.inference_actor = SimpleNamespace(clear_cache=SimpleNamespace(remote=lambda tag: None))
    w._traj = {"t": {"obs": [np.zeros(4, np.float16)] * 3, "act": [1, 2, 3], "logp": [0.0] * 3, "val": [0.0] * 3,
                     "var": [0.0] * 3, "phi": [0.0] * 3}}
    cols = {"read": np.asarray([0.0, 1.0, -0.6], np.float32), "opp_action": np.asarray([0, 3, -1], np.int8)}
    events = [("DONE", "t", cols, SimpleNamespace(won=True, lost=False)), ("DONE", "gone", cols, SimpleNamespace(won=True, lost=False))]

    class _Stop(Exception):
        pass

    def step(actions):
        if w.calls:
            raise _Stop()
        return list(events)

    w.vec_env = SimpleNamespace(release_slot=lambda: None, step=step)
    try:
        w._run_rl_loop()
    except _Stop:
        pass
    (args, kwargs), = w.calls
    assert kwargs["extras"]["read"].tolist() == pytest.approx([0.0, 1.0, -0.6]) and kwargs["extras"]["opp_action"].tolist() == [0, 3, -1]
    # without columns the DONE goes out exactly as before
    w.calls.clear()
    w._traj = {"t": {"obs": [np.zeros(4, np.float16)] * 2, "act": [1, 2], "logp": [0.0] * 2, "val": [0.0] * 2,
                     "var": [0.0] * 2, "phi": [0.0] * 2}}
    events[:] = [("DONE", "t", None, SimpleNamespace(won=False, lost=True))]
    try:
        w._run_rl_loop()
    except _Stop:
        pass
    (args, kwargs), = w.calls
    assert len(args) == 7 and kwargs == {}


def test_the_fill_value_of_opp_action_is_no_label():
    assert step_columns.fill_value("opp_action") == -1 and step_columns.fill_value("read") == 0
    out = step_columns.pack_columns([{"opp_action": np.asarray([2, 3], np.int8)}, None], [2, 3])
    assert out["opp_action"].tolist() == [2, 3, -1, -1, -1]


# =========================================================================== reads: the learner


def test_the_bonus_schedule():
    f = new_learner.read_bonus_at
    assert f(0, 0.3, 20_000_000, 45_000_000) == 0.3
    assert f(19_999_999, 0.3, 20_000_000, 45_000_000) == 0.3
    assert f(20_000_000, 0.3, 20_000_000, 45_000_000) == pytest.approx(0.3)
    assert f(32_500_000, 0.3, 20_000_000, 45_000_000) == pytest.approx(0.15)
    assert f(45_000_000, 0.3, 20_000_000, 45_000_000) == 0.0
    assert f(10**10, 0.3, 20_000_000, 45_000_000) == 0.0
    assert f(0, 0.0, 20_000_000, 45_000_000) == 0.0
    xs = [f(d, 0.3, 20, 45) for d in range(0, 50)]
    assert all(a >= b for a, b in zip(xs, xs[1:]))                       # never rises


def test_the_bonus_math_and_the_per_game_cap():
    ep = torch.tensor([7, 7, 7, 8, 8, 9, 9, 9])
    read = torch.tensor([1.0, 0.0, 1.0, -1.0, -0.6, 0.6, 0.0, -0.6])
    bonus, info = new_learner.read_bonus_terms(read, ep, 0.3, 0.5)
    # game 7: 0.3 + 0.3 = 0.6 > 0.5: both scaled by 0.5/0.6; game 8: -0.48 stays; game 9: 0 stays
    assert bonus.tolist() == pytest.approx([0.25, 0.0, 0.25, -0.3, -0.18, 0.18, 0.0, -0.18])
    assert info["read_n"] == 6 and info["read_capped"] == 1
    assert info["read_sum"] == pytest.approx(float(bonus.sum())) and info["read_abs"] == pytest.approx(float(bonus.abs().sum()))
    # the two seats of a game, opposite before, stay opposite after the cap
    a, _ = new_learner.read_bonus_terms(torch.tensor([1.0, 1.0, 0.6]), torch.tensor([1, 1, 1]), 0.3, 0.5)
    b, _ = new_learner.read_bonus_terms(torch.tensor([-1.0, -0.6, -1.0]), torch.tensor([2, 2, 2]), 0.3, 0.5)
    assert float(a.sum()) == pytest.approx(0.5) and float(a.sum() + b.sum()) == pytest.approx(0.0, abs=1e-7)
    # no cap
    c, info = new_learner.read_bonus_terms(read, ep, 0.3, 0.0)
    assert c.tolist() == pytest.approx((read * 0.3).tolist()) and info["read_capped"] == 0


def _captured_update(monkeypatch, la):
    """Runs la._perform_update with ppo_update replaced by a recorder: returns the data set it was given."""
    got = {}

    def fake_update(net, opt, dataset, **kw):
        got["data"] = dataset.tensorize()
        got["names"] = dataset.extra_names
        got["kw"] = kw
        return new_core.PPOUpdateStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, None, None)

    monkeypatch.setattr(new_learner, "ppo_update", fake_update)
    asyncio.run(la._perform_update())
    return got


def _small_learner(**over):
    la = tsu._new_learner("ppo", save_every_updates=10**9, **over)
    return la


@needs_gpu
def test_the_bonus_goes_into_the_advantages_before_normalising_and_never_into_the_returns(data, monkeypatch, capsys):
    obs_np, ep_np, _kind = data
    n = 1024
    cols = tsu._update_data(obs_np, ep_np, n, seed=41)
    rng = np.random.default_rng(3)
    read = torch.from_numpy(rng.choice([0.0, 0.0, 0.0, 1.0, -1.0, 0.6, -0.6], n).astype(np.float32))
    adv, ret, ep = cols[4].clone(), cols[5].clone(), cols[7]
    la = _small_learner(read_bonus=0.3, read_anneal_start=20_000_000, read_anneal_end=45_000_000)
    la.read_origin = 1_000_000
    la.total_steps = 1_000_000 + 32_500_000            # halfway down the fade: b = 0.15
    la.dataset.add_steps(*cols, extras={"read": read})
    got = _captured_update(monkeypatch, la)
    bonus, info = new_learner.read_bonus_terms(read, ep, 0.15, 0.5)
    want = adv + bonus
    want = torch.clamp((want - want.mean()) / (want.std() + 1e-8), -10.0, 10.0)
    assert torch.equal(got["data"][4], want)
    assert torch.equal(got["data"][5], ret)                                     # the critic's targets as they were
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[learner] upd="))
    keys = [kv.split("=")[0] for kv in line.split()[1:]]
    assert keys[-7:] == ["adv_pre_std", "lr", "read_n", "read_sum", "read_abs", "read_capped", "b"], line
    assert f"read_n={int(info['read_n'])}" in line and "b=0.15" in line
    # past the end of the fade: b = 0, the advantages are the plain ones
    la.total_steps = la.read_origin + 50_000_000
    la.dataset.add_steps(*cols, extras={"read": read})
    got = _captured_update(monkeypatch, la)
    assert torch.equal(got["data"][4], torch.clamp((adv - adv.mean()) / (adv.std() + 1e-8), -10.0, 10.0))
    del la
    torch.cuda.empty_cache()


@needs_gpu
def test_a_read_column_without_the_bonus_changes_nothing(data, monkeypatch, capsys):
    obs_np, ep_np, _kind = data
    n = 1024
    cols = tsu._update_data(obs_np, ep_np, n, seed=42)
    la = _small_learner()
    la.dataset.add_steps(*cols)
    plain = _captured_update(monkeypatch, la)
    la.dataset.add_steps(*cols, extras={"read": torch.ones(n), "opp_action": torch.zeros(n, dtype=torch.int8)})
    with_col = _captured_update(monkeypatch, la)
    assert all(torch.equal(a, b) for a, b in zip(plain["data"][:8], with_col["data"][:8]))
    assert plain["kw"] == with_col["kw"] and "boosted_row_weight" not in plain["kw"]
    out = capsys.readouterr().out
    assert "read_n=" not in out and "opp_acc" not in out
    del la
    torch.cuda.empty_cache()


@needs_gpu
def test_the_read_origin_is_saved_and_read_back(tmp_path, capsys):
    # the checkpoint of another run (no key): this arm counts its decisions from the checkpoint's total_steps
    plain = _small_learner()
    plain.total_steps = 7_000
    p_plain = str(tmp_path / "plain.pt")
    plain._save_checkpoint(p_plain)
    assert "read_anneal_origin" not in torch.load(p_plain, map_location="cpu", weights_only=False)
    arm = _small_learner(read_bonus=0.3)
    arm._load_checkpoint(p_plain)
    assert arm.total_steps == 7_000 and arm.read_origin == 7_000
    assert "this arm's decisions count from total_steps 7,000" in capsys.readouterr().out
    # the arm's own checkpoint: the origin it saved
    arm.total_steps = 9_000
    p_arm = str(tmp_path / "arm.pt")
    arm._save_checkpoint(p_arm)
    assert torch.load(p_arm, map_location="cpu", weights_only=False)["read_anneal_origin"] == 7_000
    again = _small_learner(read_bonus=0.3)
    again._load_checkpoint(p_arm)
    assert again.total_steps == 9_000 and again.read_origin == 7_000
    del plain, arm, again
    torch.cuda.empty_cache()


# =========================================================================== the auxiliary heads


def test_the_opp_action_loss_ignores_rows_without_a_label():
    net = SimpleNamespace(opp_action_head=new_core.OppActionHead(16), opp_action_coef=0.1)
    z = torch.randn(5, 64, 16)
    labels = torch.tensor([0, -1, 3, 7, -1])
    term, st = new_core.compute_opp_action_loss(net, z, labels)
    logits = net.opp_action_head(z)
    assert logits.shape == (5, 8)
    keep = torch.tensor([0, 2, 3])
    want = torch.nn.functional.cross_entropy(logits[keep], labels[keep])
    assert float(st["opp_loss"]) == pytest.approx(float(want), rel=1e-5) and float(st["opp_n"]) == 3
    assert float(st["opp_moved"]) == 1                                         # the always-"moved" baseline's hits
    assert float(term) == pytest.approx(0.1 * float(want), rel=1e-5)
    # the slot logits come from that opponent token alone
    z2 = z.clone()
    z2[:, 9, :] += 1.0
    d = net.opp_action_head(z2) - logits
    assert torch.allclose(d[:, [1, 2, 4, 5, 6]], torch.zeros(5, 5), atol=1e-6) and not torch.allclose(d[:, 3], torch.zeros(5))
    _t, st = new_core.compute_opp_action_loss(net, z, torch.full((5,), -1))
    assert float(st["opp_n"]) == 0 and float(st["opp_loss"]) == 0.0


def test_the_heads_go_to_the_aux_zone_and_the_last_optimizer_groups():
    cfg = new_config.RunConfig.default()
    net = cfg.make_model()
    zones = net.get_parameter_zones()
    assert zones["aux"] == []
    net.attach_opp_action_head(0.1)
    zones = net.get_parameter_zones()
    assert [n for n, _ in zones["aux"]] == ["opp_action_head.slot.weight", "opp_action_head.slot.bias",
                                             "opp_action_head.other.weight", "opp_action_head.other.bias"]
    assert all(not n.startswith("opp_action_head") for z, ps in zones.items() if z != "aux" for n, _ in ps)
    assert list(zones)[-1] == "aux"


@needs_gpu
def test_the_optimizer_state_of_a_checkpoint_without_heads_carries_over(tmp_path):
    plain = _small_learner()
    with_head = _small_learner(opp_action_aux_coef=0.1)
    # some Adam state in the plain one
    for _p, m in plain.opt.pairs:
        m.grad = torch.ones_like(m) * 1e-3
    plain.opt.inner.step()
    saved = plain.opt.state_dict()
    names = [g["name"] for g in with_head.opt.inner.state_dict()["param_groups"]]
    assert names[-2:] == ["aux_wd", "aux_stable"] and names[:-2] == [g["name"] for g in saved["inner"]["param_groups"]]
    laid = new_learner.optimizer_state_for_layout(saved, with_head.opt.state_dict())
    assert laid is not None
    with_head.opt.load_state_dict(laid)
    n_base = len(saved["masters"])
    for (p_a, m_a), (p_b, m_b) in zip(plain.opt.pairs, with_head.opt.pairs[:n_base]):
        assert torch.equal(m_a, m_b)
        sa, sb = plain.opt.inner.state[m_a], with_head.opt.inner.state[m_b]
        assert torch.equal(sa["exp_avg"], sb["exp_avg"]) and torch.equal(sa["exp_avg_sq"], sb["exp_avg_sq"])
    for p, m in with_head.opt.pairs[n_base:]:
        assert not with_head.opt.inner.state.get(m) and torch.equal(m, p.detach().float())
    # and back: a checkpoint with the head, resumed without it
    back = new_learner.optimizer_state_for_layout(with_head.opt.state_dict(), plain.opt.state_dict())
    assert back is not None and len(back["masters"]) == n_base
    plain.opt.load_state_dict(back)
    # a layout that differs elsewhere is not forced
    broken = with_head.opt.state_dict()
    broken["inner"]["param_groups"][0] = dict(broken["inner"]["param_groups"][0], name="other_wd")
    assert new_learner.optimizer_state_for_layout(broken, plain.opt.state_dict()) is None
    del plain, with_head
    torch.cuda.empty_cache()


@needs_gpu
def test_an_update_with_the_opp_action_head_trains_it_and_reports_its_accuracy(data, capsys):
    obs_np, ep_np, _kind = data
    n = 1024
    cols = tsu._update_data(obs_np, ep_np, n, seed=43)
    la = _small_learner(opp_action_aux_coef=0.1)
    assert la.net.opp_action_head is not None and not hasattr(la.target_net, "opp_action_head")
    before = la.net.opp_action_head.other.weight.detach().clone()
    opp = torch.from_numpy(np.random.default_rng(5).integers(-1, 8, n).astype(np.int8))
    la.dataset.add_steps(*cols, extras={"opp_action": opp})
    la.total_steps += n
    asyncio.run(la._perform_update())
    assert not torch.equal(before, la.net.opp_action_head.other.weight.detach())
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[learner] upd="))
    assert " opp_n=" in line and " opp_acc=" in line and " opp_loss=" in line, line
    # the weights the players get have no head (they load with strict=False; the inference resume filters it)
    assert all(not k.startswith("opp_action_head") for k in la._base_state_dict())
    del la
    torch.cuda.empty_cache()


# =========================================================================== the boosted-row weight


def test_the_boosted_row_weights():
    w = new_core.boosted_row_weights(torch.tensor([1, 0, 0, 1, 0, 0], dtype=torch.int8), 2.0)
    assert float(w.mean()) == pytest.approx(1.0)
    assert float(w[0] / w[1]) == pytest.approx(2.0) and w.dtype == torch.float32
    assert torch.equal(new_core.boosted_row_weights(torch.zeros(4, dtype=torch.int8), 2.0), torch.ones(4))
    assert torch.equal(new_core.boosted_row_weights(torch.tensor([1, 0, 1, 0]), 1.0), torch.ones(4))
    # the weighted surrogate is the per-row loss times the weights, averaged
    per_row = torch.tensor([0.5, -1.0, 2.0, 0.0])
    b = torch.tensor([1, 0, 0, 1])
    ww = new_core.boosted_row_weights(b, 3.0)
    assert float((per_row * ww).mean()) == pytest.approx(float((per_row * torch.tensor([3.0, 1, 1, 3])).sum() / 8.0))


@needs_gpu
@pytest.mark.parametrize("mode", ["ppo"])
def test_w_1_leaves_the_update_bit_identical_and_w_2_changes_it(data, mode):
    obs_np, ep_np, _kind = data
    n = 1536
    cols = tsu._update_data(obs_np, ep_np, n, seed=44)
    boosted = (torch.arange(n) % 5 == 0).to(torch.int8)
    a, b, c = (tsu._new_learner(mode) for _ in range(3))
    tsu._copy_state(a, b)
    tsu._copy_state(a, c)

    def run(la, extras, **kw):
        ds = new_core.AsyncEpisodeDataset(14, "cpu")
        ds.add_steps(*cols, extras=extras)
        torch.manual_seed(9)
        return new_core.ppo_update(net=la.net, opt=la.opt, dataset=ds, scheduler=la.sched, mode=la.cfg.mode, target_net=None,
                                   **la.cfg.ppo_kwargs(), v_min=la.cfg.v_min, v_max=la.cfg.v_max, v_bins=la.cfg.v_bins, **kw)

    sa = run(a, None)
    sb = run(b, {"boosted": boosted}, boosted_row_weight=1.0)
    want, got = tsu._state(a), tsu._state(b)
    assert [k for k in want if not micro_ref.same_bits(want[k], got[k])] == []
    assert sa.pg_loss == sb.pg_loss and sb.aux is None
    sc = run(c, {"boosted": boosted}, boosted_row_weight=2.0)
    assert sc.aux and sc.aux["boosted_share"] == pytest.approx(0.2, abs=0.02)
    assert sc.pg_loss != sa.pg_loss
    del a, b, c
    torch.cuda.empty_cache()


# =========================================================================== flags off: today's update


@pytest.fixture(scope="module")
def head():
    """The commit before the wiring (PSPPO_HEAD_DIR: its files), beside the working tree's."""
    if not HEAD_DIR or not os.path.isdir(HEAD_DIR):
        pytest.skip("PSPPO_HEAD_DIR names no copy of the commit before the wiring")
    ref = micro_ref.Reference(HEAD_DIR, source="PSPPO_HEAD_DIR")
    core, lrn, cfg = ref.load("ppo_core"), ref.load("learner"), ref.load("config")
    assert lrn.ppo_core is core and core is not new_core
    assert not hasattr(lrn, "read_bonus_terms"), "PSPPO_HEAD_DIR holds this branch's code, not the commit before it"
    if GPU:
        micro_ref.shared_compiled_flex_attention(core, new_core)
    core._learner_flex_compiled = True
    new_core._learner_flex_compiled = True
    return SimpleNamespace(core=core, learner=lrn, config=cfg, ref=ref)


@needs_gpu
@pytest.mark.parametrize("mode", ["ppo", "imitation"])
def test_with_every_flag_off_a_whole_update_is_bit_identical_to_the_commit_before(head, data, mode):
    obs_np, ep_np, _kind = data
    n = 1536
    cols = tsu._update_data(obs_np, ep_np, n, seed=45)
    # the new columns as the workers would send them, every feature off
    extras = {"read": torch.full((n,), 0.6), "opp_action": torch.zeros(n, dtype=torch.int8),
              "boosted": torch.ones(n, dtype=torch.int8), "set_labels": torch.zeros((n, 18), dtype=torch.int32)}
    with head.ref.active():
        old = tsu._make_learner(head.learner, head.core, tsu._learner_cfg(head.config, mode, save_every_updates=10**9))
    new, new_x = tsu._new_learner(mode, save_every_updates=10**9), tsu._new_learner(mode, save_every_updates=10**9)
    for other in (new, new_x):
        tsu._copy_state(old, other)
    for la, x in ((old, None), (new, None), (new_x, extras)):
        if x is None:
            la.dataset.add_steps(*cols)
        else:
            la.dataset.add_steps(*cols, extras=x)
        la.total_steps += n
        torch.manual_seed(78)
        asyncio.run(la._perform_update())
        assert la.update_idx == 1 and len(la.dataset) == 0, "the update did not run"
    want = tsu._state(old)
    for la in (new, new_x):
        got = tsu._state(la)
        assert want.keys() == got.keys()
        diff = [k for k in want if not micro_ref.same_bits(want[k], got[k])]
        assert diff == [], (mode, diff[:5])
    del old, new, new_x
    torch.cuda.empty_cache()


# =========================================================================== train.py, check_restart


def test_train_flags():
    import train
    a = train.parse_args(["--read-bonus", "0.3", "--read-anneal", "20e6:45e6", "--opp-action-aux-coef", "0.1",
                          "--boosted-row-weight", "2", "--obs-matchup", "--set-head-coef", "0.1",
                          "--opponent-seats", "setupnet:0.15", "--setupnet-p", "0.5", "--setupnet-max-stage", "4"])
    cfg = train.build_config(a)
    lc = cfg.learner
    assert (lc.read_bonus, lc.read_anneal_start, lc.read_anneal_end, lc.opp_action_aux_coef, lc.boosted_row_weight, lc.set_head_coef) == \
        (0.3, 20_000_000, 45_000_000, 0.1, 2.0, 0.1)
    assert cfg.obs.matchup is True
    assert "read_bonus=0.3" in train.features_line(cfg)
    import opponent_seats
    seats = opponent_seats.from_args(a)
    assert seats is not None and dict(seats.shares)["setupnet"] == 0.15
    # none given: the config is config.py's
    assert train.build_config(train.parse_args([])) == new_config.RunConfig.default()
    assert opponent_seats.from_args(train.parse_args([])) is None
    for bad in (["--set-head-coef", "0.1"], ["--read-anneal", "45e6:20e6"], ["--read-anneal", "x"], ["--boosted-row-weight", "0"],
                ["--read-bonus", "-1"], ["--opponent-seats", "nosuchkind:0.1"]):
        with pytest.raises(SystemExit):
            train.parse_args(bad)


def test_the_config_builds_the_matchup_network_and_assembler():
    cfg = new_config.RunConfig.default()
    assert cfg.obs.matchup is False and not getattr(cfg.make_model(), "matchup", False)
    mu = dataclasses.replace(cfg, obs=dataclasses.replace(cfg.obs, matchup=True))
    assert mu.make_model().matchup is True
    a0, a1 = cfg.make_assembler(), mu.make_assembler()
    assert "matchup" in a1.offsets and "matchup" not in a0.offsets and a1.total_dim == a0.total_dim + 1068


def test_check_restart_refuses_a_matchup_change(tmp_path, monkeypatch):
    import check_restart
    import test_check_restart as tcr
    ck = tcr._ckpt(409_000)
    code, lines = tcr._run(tmp_path, monkeypatch, ["--obs-matchup"], ck)
    assert code == 2 and any("switches ON ['matchup']" in ln for ln in lines), lines
    code, lines = tcr._run(tmp_path, monkeypatch, ["--obs-matchup"], ck, allow_obs_change=True)
    assert code == 2 and any("widen" in ln.lower() for ln in lines), lines
    widened = dict(ck, obs_settings={"own_tera_type": True, "matchup": True})
    code, lines = tcr._run(tmp_path, monkeypatch, ["--obs-matchup"], widened)
    assert code == 0, lines
    assert check_restart.checkpoint_obs_settings({"run_cfg": {"obs": {"own_tera_type": True, "matchup": True}}}) == \
        {"own_tera_type": True, "matchup": True}
