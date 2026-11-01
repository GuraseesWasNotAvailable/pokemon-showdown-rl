"""
obs_v2: a network of branch train-v2 with its matchup block (and moveset block) is played on the
observation it was trained with, by the network it was trained as; every other network as before.

    python -m pytest -q tests/test_obs_v2.py

What is held:

  * which checkpoints: run_cfg["obs"]["matchup"] on and no "obs_settings" (checkpoint_v2_blocks);
    hybrid.PolicyNet then builds V2ObservationAssembler and the wider network, leaves the
    training-only heads out, and refuses an observation override; the other lines get the
    classes they got before
  * the layout: this line's offsets, then the matchup block (1,068) and the moveset block (660)
  * the bytes: before the action mask the search line's observation (own Tera type, STAB after a
    Tera); the mask train-v2's, which offers a switch the server refuses, while what the player
    chooses from (hybrid.choice_mask) and what the search is told (create_action_mask,
    foulplay_bridge.legal_mask) leave it out; and against train-v2's own obs_assembler.py
    (PSPPO_V2_REF: a directory holding that branch's files, else git show train-v2:...) the
    observation byte for byte in every fixture state
  * the copied files are train-v2's, but for the one import line each
  * the network: make_model() without the blocks is the network of before; with them it loads a
    train-v2 state dict and gives finite logits
  * the window names the run
  * with PSPPO_V2_REF, PSPPO_STREAMS and PSPPO_V2_CKPT (a train-v2 checkpoint with the block),
    bench/v2_exactness.py on a few recorded battles: IDENTICAL

No Showdown server is started. Nothing is removed.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(REPO)          # vocab.json is opened by a relative path

pytest.importorskip("torch")
pytest.importorskip("poke_env")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import foulplay_bridge  # noqa: E402
import test_own_tera as tot  # noqa: E402
import test_switch_names as tsn  # noqa: E402
from hybrid import choice_mask  # noqa: E402
from obs_assembler import ObservationAssembler  # noqa: E402
from obs_search import SearchOwnTeraObservationAssembler  # noqa: E402
from obs_v2 import AUX_HEAD_PREFIXES, LINE, V2ObservationAssembler, checkpoint_v2_blocks  # noqa: E402

V2_REF = "train-v2"
BASE_OBS = {"n_slots": 6, "n_move_slots": 4, "hp_bins": 11, "vocab_path": "vocab.json"}
SCRATCH_OBS = dict(BASE_OBS, own_tera_type=True, matchup=True, movesets=True)


def _opponent_terastallized():
    battle = tot.with_an_opponent(tot.make_battle())
    own = tot.name_of(tot.REQUEST["side"]["pokemon"][0])
    battle.parse_message(["", "move", "p2a: Arceus", "Judgment", f"p1a: {own}"])
    battle.parse_message(["", "-terastallize", "p2a: Arceus", "Fire"])
    return battle


STATES = dict(tot.STATES)
STATES["an Illusion's Tera"] = tot.illusion_battle
STATES["the opponent Terastallized"] = _opponent_terastallized
STATES["Ditto Transformed into our Garchomp"] = tsn.battle_at
STATES["before the Transform"] = lambda: tsn.battle_at(upto="|switch|p2a: Garchomp|Garchomp, L77, F|100/100")
EVENTS = ([], [["", "move", "p2a: Arceus", "Judgment", "p1a: X"], ["", "-damage", "p1a: X", "80/100"]])


# --------------------------------------------------------------------------- which checkpoints


def test_a_v2_checkpoint_is_one_whose_run_cfg_has_the_matchup_block_and_that_has_no_obs_settings():
    assert checkpoint_v2_blocks({"run_cfg": {"obs": SCRATCH_OBS}}) == {"own_tera_type": True, "matchup": True, "movesets": True}
    assert checkpoint_v2_blocks({"run_cfg": {"obs": dict(BASE_OBS, own_tera_type=True, matchup=True)}}) == \
        {"own_tera_type": True, "matchup": True, "movesets": False}
    assert checkpoint_v2_blocks({"run_cfg": {"obs": dict(BASE_OBS, matchup=True)}})["own_tera_type"] is False
    for other in ({"run_cfg": {"obs": dict(BASE_OBS, own_tera_type=True)}},                  # the search line
                  {"run_cfg": {"obs": dict(BASE_OBS, own_tera_type=True, matchup=False)}},
                  {"run_cfg": {"obs": BASE_OBS}}, {"run_cfg": {}}, {"model": {}}, None, "x",
                  {"run_cfg": {"obs": SCRATCH_OBS}, "obs_settings": {"own_tera_type": True}}):   # this line's own record decides
        assert checkpoint_v2_blocks(other) is None, other


# --------------------------------------------------------------------------- the layout and the bytes


@pytest.mark.parametrize("movesets", [True, False])
def test_the_layout_is_this_lines_then_the_blocks(movesets):
    plain = ObservationAssembler()
    asm = V2ObservationAssembler(movesets=movesets)
    for name, (s, e) in plain.offsets.items():
        assert asm.offsets[name] == (s, e), name
    end = plain.total_dim
    assert asm.offsets["matchup"] == (end, end + 1068) and asm.meta["matchup"]["dim"] == 1068
    if movesets:
        assert asm.offsets["movesets"] == (end + 1068, end + 1068 + 660) and asm.total_dim == end + 1728
    else:
        assert "movesets" not in asm.offsets and "movesets" not in asm.meta and asm.total_dim == end + 1068
    assert asm.meta["offsets"] is asm.offsets and asm.calc_buf.shape == (asm.total_dim,)


@pytest.mark.parametrize("fast", [True, False])
def test_before_the_mask_it_is_the_search_lines_observation_and_the_blocks_are_written(fast):
    search = SearchOwnTeraObservationAssembler(fast=fast)
    asm = V2ObservationAssembler(fast=fast)
    m0, m1 = asm.offsets["action_mask"]
    written = 0
    for name, make in STATES.items():
        for events in EVENTS:
            want = search.assemble(make(), events=events)
            got = asm.assemble(make(), events=events)
            assert got.dtype == want.dtype == np.float16 and got.shape == (asm.total_dim,)
            assert np.array_equal(got[:m0], want[:m0]), (name, np.nonzero(got[:m0] != want[:m0])[0][:10])
            written += int(np.any(got[m1:] != 0))
    assert written >= len(STATES), "the blocks hold numbers in every state"
    off = V2ObservationAssembler(own_tera_type=False, fast=fast)
    plain = ObservationAssembler(fast=fast)
    for name, make in STATES.items():
        a, b = off.assemble(make(), events=[]), plain.assemble(make(), events=[])
        assert np.array_equal(a[:m0], b[:m0]), name


def test_the_observation_offers_the_refused_switch_and_the_player_and_the_search_do_not():
    battle = tsn.battle_at()
    garchomp = 4 + tsn.slot(battle, "Garchomp")
    asm = V2ObservationAssembler()
    obs = asm.assemble(battle, events=[])
    m0, m1 = asm.offsets["action_mask"]
    assert obs[m0 + garchomp] == 1.0, "train-v2's mask, which the network was trained on, offers it"
    mine = choice_mask(asm, battle, obs)
    assert mine.dtype == np.float32 and mine[garchomp] == 0.0 and mine[4 + tsn.slot(battle, "Pikachu")] == 1.0
    assert np.array_equal(mine, asm.create_action_mask(battle, list(battle.team.values())))
    assert foulplay_bridge.legal_mask(battle) == [int(x) for x in mine], "the search is told the same"
    # everywhere else the two masks are one
    differ = np.flatnonzero(obs[m0:m1].astype(np.float32) != mine)
    assert differ.tolist() == [garchomp]
    for name, make in STATES.items():
        if name == "Ditto Transformed into our Garchomp":
            continue
        b = make()
        o = asm.assemble(b, events=[])
        assert np.array_equal(o[m0:m1].astype(np.float32), choice_mask(asm, b, o)), name


def test_every_other_assembler_chooses_from_its_observations_mask_as_before():
    for asm in (ObservationAssembler(), SearchOwnTeraObservationAssembler()):
        battle = tsn.battle_at()
        obs = asm.assemble(battle, events=[])
        m0, m1 = asm.offsets["action_mask"]
        got = choice_mask(asm, battle, obs)
        assert got.dtype == np.float32 and np.array_equal(got, obs[m0:m1].astype(np.float32))
        assert not hasattr(asm, "choice_mask")


def _v2_source(name: str) -> bytes:
    """A file of branch train-v2: from PSPPO_V2_REF, else from this clone's train-v2."""
    folder = os.environ.get("PSPPO_V2_REF")
    if folder:
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            pytest.skip(f"no {name} in PSPPO_V2_REF={folder}")
        with open(path, "rb") as f:
            return f.read().replace(b"\r\n", b"\n")
    got = subprocess.run(["git", "-C", REPO, "show", f"{V2_REF}:{name}"], capture_output=True)
    if got.returncode != 0:
        pytest.skip(f"{V2_REF} is not in this clone and PSPPO_V2_REF is not set")
    return got.stdout.replace(b"\r\n", b"\n")


@pytest.mark.parametrize("fast", [True, False])
@pytest.mark.parametrize("movesets", [True, False])
def test_the_bytes_are_those_of_train_v2s_own_assembler(fast, movesets):
    source = _v2_source("obs_assembler.py").decode("utf-8")
    if "movesets" not in source:
        pytest.skip("that obs_assembler.py is from before the moveset block")
    spec = importlib.util.spec_from_loader("v2_ref_obs_assembler", loader=None)
    module = importlib.util.module_from_spec(spec)
    # Its blocks come from the copies here (test_the_copied_files_are_train_v2s says they are its
    # files); its own_tera_type_name is its own.
    exec(compile(source, "v2_ref_obs_assembler.py", "exec"), module.__dict__)
    reference = module.ObservationAssembler(fast=fast, own_tera_type=True, matchup=True, movesets=movesets)
    here = V2ObservationAssembler(fast=fast, movesets=movesets)
    assert reference.offsets == here.offsets and reference.total_dim == here.total_dim
    assert reference.meta.keys() == here.meta.keys() and reference.meta["matchup"] == here.meta["matchup"]
    for name, make in STATES.items():
        for events in EVENTS:
            want = reference.assemble(make(), events=events)
            got = here.assemble(make(), events=events)
            assert want.dtype == got.dtype and want.tobytes() == got.tobytes(), (name, np.nonzero(want != got)[0][:10])


def test_the_copied_files_are_train_v2s_but_for_where_they_import_own_tera_type_name():
    edit = (b"    from obs_assembler import own_tera_type_name\n",
            b"    from obs_tera import own_tera_type_name   # branch decide keeps it there (train-v2: obs_assembler; the same function)\n")
    for name in ("obs_matchup.py", "obs_movesets.py", "matchup.py", "randbats_sets_gen9.json"):
        theirs = _v2_source(name)
        with open(os.path.join(REPO, name), "rb") as f:
            ours = f.read().replace(b"\r\n", b"\n")
        if name in ("obs_matchup.py", "obs_movesets.py"):
            assert theirs.count(edit[0]) == 1, name
            theirs = theirs.replace(*edit)
        assert ours == theirs, name


# --------------------------------------------------------------------------- the network and the players


def _v2_state_dict(movesets: bool = True):
    import config
    net = config.RunConfig.default().make_model(matchup=True, movesets=movesets)
    state = net.state_dict()
    # the training-only heads train-v2's learner attaches (shapes as there; their values do not matter here)
    d = net.d_model
    state.update({"set_head.ability.weight": torch.zeros(5, d), "set_head.ability.bias": torch.zeros(5),
                  "set_head.item.weight": torch.zeros(4, d), "set_head.item.bias": torch.zeros(4),
                  "opp_action_head.slot.weight": torch.zeros(1, d), "opp_action_head.slot.bias": torch.zeros(1),
                  "opp_action_head.other.weight": torch.zeros(2, 2 * d), "opp_action_head.other.bias": torch.zeros(2)})
    return state


def test_without_the_blocks_make_model_is_the_network_of_before():
    import config
    net = config.RunConfig.default().make_model()
    assert net.matchup is False and net.movesets is False and net.mu_move_in == net.mu_pok_in == net.ms_pok_in == 0
    assert "matchup" not in net.meta and net.unpacker.mu_meta is None and net.unpacker.ms_meta is None
    # the first Linear of the subnets takes what it always took
    asm = ObservationAssembler()
    assert net.pokemon_net[0].in_features == net._calc_pok_in(config.ModelConfig().emb_dims, config.ModelConfig().bank_dims,
                                                             config.ModelConfig().out_dims)
    assert net.move_net[0].in_features == net._calc_move_in(config.ModelConfig().emb_dims, config.ModelConfig().bank_dims)
    assert net.meta["offsets"]["action_mask"][1] == asm.total_dim
    with pytest.raises(ValueError):
        config.RunConfig.default().make_model(movesets=True)


def test_with_the_blocks_the_network_takes_their_columns_and_runs():
    import config
    net = config.RunConfig.default().make_model(matchup=True, movesets=True).eval()
    plain = config.RunConfig.default().make_model()
    assert net.move_net[0].in_features == plain.move_net[0].in_features + net.mu_move_in
    assert net.pokemon_net[0].in_features == plain.pokemon_net[0].in_features + net.mu_pok_in + net.ms_pok_in
    asm = V2ObservationAssembler()
    obs = asm.assemble(tot.with_an_opponent(tot.make_battle()), events=[])
    with torch.no_grad():
        out = net(torch.from_numpy(obs.astype(np.float32))[None])
    assert out[0].shape == (1, 14) and torch.isfinite(out[0]).all()


def _save(tmp_path, name, ck):
    path = tmp_path / name
    torch.save(ck, path)
    return str(path)


@pytest.mark.parametrize("movesets", [True, False])
def test_the_policy_plays_a_v2_checkpoint_as_trained_and_leaves_the_heads_out(tmp_path, movesets):
    from hybrid import HybridConfig, PolicyNet

    obs_cfg = dict(SCRATCH_OBS, movesets=movesets)
    state = _v2_state_dict(movesets)
    path = _save(tmp_path, "v2-scratch_update_000001_1M.pt", {"model": state, "run_cfg": {"obs": obs_cfg}})
    pol = PolicyNet(HybridConfig(checkpoint=path, device="cpu"))
    assert pol.obs_line == LINE == "v2" and type(pol.assembler) is V2ObservationAssembler
    assert pol.assembler.movesets is movesets and pol.own_tera_type is True and pol.obs_abilities == {}
    assert pol.dropped_heads == 8 == sum(1 for k in state if k.startswith(AUX_HEAD_PREFIXES))
    assert not any(k.startswith(AUX_HEAD_PREFIXES) for k in pol.net.state_dict())
    for k, v in pol.net.state_dict().items():
        assert torch.equal(v.cpu(), state[k].float()), k
    obs = pol.assembler.assemble(tsn.battle_at(), events=[])
    assert np.isfinite(pol.logits("t", obs)).all()

    # an override of the observation is refused: it is played only as it was trained
    with pytest.raises(RuntimeError, match="matchup block"):
        PolicyNet(HybridConfig(checkpoint=path, device="cpu", own_tera_type=False))
    # a tensor that is neither the network's nor a training head is still refused
    bad = dict(state, **{"something_else.weight": torch.zeros(3)})
    with pytest.raises(RuntimeError, match="does not fit"):
        PolicyNet(HybridConfig(checkpoint=_save(tmp_path, "bad.pt", {"model": bad, "run_cfg": {"obs": obs_cfg}}), device="cpu"))
    # so is one without the block's columns
    import config
    narrow = config.RunConfig.default().make_model().state_dict()
    with pytest.raises(RuntimeError):
        PolicyNet(HybridConfig(checkpoint=_save(tmp_path, "narrow.pt", {"model": narrow, "run_cfg": {"obs": obs_cfg}}), device="cpu"))


SCRATCH_MODEL = {"dropout": 0.0, "n_layers": 3, "n_heads": 8, "ff_expansion": 2.0, "kv_cache_len": 64}   # the run's run_cfg["model"]


def test_a_v2_network_whose_run_built_it_otherwise_is_refused(tmp_path):
    from hybrid import HybridConfig, PolicyNet
    from obs_v2 import architecture_mismatch

    state = _v2_state_dict(True)
    path = _save(tmp_path, "ok.pt", {"model": state, "run_cfg": {"obs": SCRATCH_OBS, "model": SCRATCH_MODEL}})
    pol = PolicyNet(HybridConfig(checkpoint=path, device="cpu"))
    assert pol.obs_line == "v2" and architecture_mismatch({"run_cfg": {"model": SCRATCH_MODEL}}, pol.net) == []
    assert architecture_mismatch({"run_cfg": {}}, pol.net) == [] and architecture_mismatch(None, pol.net) == []
    # same tensor sizes, another network: refused, not played
    for key, value in (("n_heads", 4), ("kv_cache_len", 32)):
        bad = {"model": state, "run_cfg": {"obs": SCRATCH_OBS, "model": dict(SCRATCH_MODEL, **{key: value})}}
        with pytest.raises(RuntimeError, match=key):
            PolicyNet(HybridConfig(checkpoint=_save(tmp_path, f"{key}.pt", bad), device="cpu"))
    assert architecture_mismatch({"run_cfg": {"model": dict(SCRATCH_MODEL, ff_expansion=4.0, n_layers=4)}}, pol.net) == [
        "n_layers 4 (this code builds 3)", "ff_expansion 4.0 (this code builds 2.0)"]


def test_the_other_lines_get_the_classes_they_got_before(tmp_path):
    import config
    from hybrid import HybridConfig, PolicyNet
    from obs_tera import OwnTeraObservationAssembler

    plain = config.RunConfig.default().make_model().state_dict()
    search = PolicyNet(HybridConfig(checkpoint=_save(tmp_path, "a.pt", {"model": plain, "run_cfg": {"obs": dict(BASE_OBS, own_tera_type=True)}}), device="cpu"))
    assert search.obs_line == "search" and type(search.assembler) is SearchOwnTeraObservationAssembler and search.dropped_heads == 0
    # a train-v2 arm without the block, its heads still in the file: the search line, the heads left out
    with_heads = dict(plain, **{k: v for k, v in _v2_state_dict().items() if k.startswith(AUX_HEAD_PREFIXES)})
    arm = PolicyNet(HybridConfig(checkpoint=_save(tmp_path, "b.pt", {"model": with_heads, "run_cfg": {"obs": dict(BASE_OBS, own_tera_type=True, matchup=False)}}), device="cpu"))
    assert arm.obs_line == "search" and arm.dropped_heads == 8
    old = PolicyNet(HybridConfig(checkpoint=_save(tmp_path, "c.pt", {"model": plain, "run_cfg": {"obs": BASE_OBS}}), device="cpu"))
    assert old.obs_line == "own" and type(old.assembler) is ObservationAssembler
    r5 = PolicyNet(HybridConfig(checkpoint=_save(tmp_path, "d.pt", {"model": plain, "run_cfg": {"obs": BASE_OBS}, "obs_settings": {"own_tera_type": True}}), device="cpu"))
    assert r5.obs_line == "own" and type(r5.assembler) is OwnTeraObservationAssembler
    assert all(p.v2_blocks is None for p in (search, arm, old, r5))


def test_the_window_names_the_from_scratch_run():
    source = open(os.path.join(REPO, "app", "PSBotLookahead.cs"), encoding="utf-8").read()
    assert '{ "scratch", "from scratch, every technique" },' in source
    assert '"NEW TRAINING: " + (V2Names.TryGetValue(c.V2Arm, out v2Name) ? v2Name : c.V2Arm)' in source


# --------------------------------------------------------------------------- against train-v2's own code


def test_recorded_battles_through_the_app_and_through_train_v2_are_identical(tmp_path):
    ref, streams, ckpt = (os.environ.get(k) for k in ("PSPPO_V2_REF", "PSPPO_STREAMS", "PSPPO_V2_CKPT"))
    if not (ref and streams and ckpt):
        pytest.skip("needs PSPPO_V2_REF (train-v2's code), PSPPO_STREAMS (streams.jsonl) and PSPPO_V2_CKPT (a train-v2 checkpoint with the block)")
    device = os.environ.get("PSPPO_V2_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    tool = os.path.join(REPO, "bench", "v2_exactness.py")
    common = ["--checkpoint", ckpt, "--streams", streams, "--battles", os.environ.get("PSPPO_V2_BATTLES", "3"), "--device", device]
    a, b = str(tmp_path / "ref.npz"), str(tmp_path / "app.npz")
    for cmd in (["ref", "--code", ref, "--out", a] + common, ["app", "--out", b] + common):
        got = subprocess.run([sys.executable, tool] + cmd, capture_output=True, text=True)
        assert got.returncode == 0, got.stdout[-2000:] + got.stderr[-4000:]
    got = subprocess.run([sys.executable, tool, "compare", a, b], capture_output=True, text=True)
    assert got.returncode == 0 and got.stdout.strip().endswith("IDENTICAL"), got.stdout[-3000:] + got.stderr[-2000:]
