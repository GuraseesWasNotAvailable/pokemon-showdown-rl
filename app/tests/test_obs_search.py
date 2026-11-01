"""
obs_search: a checkpoint of the search branch (the baseline run) is played on the observation it
was trained with.

    python -m pytest -q tests/test_obs_search.py

The comparison with the search branch's own obs_assembler.py needs that file: from
PSPPO_SEARCH_REF (a directory holding it, e.g. the baseline release on the box) or from this clone
(git show origin/search:obs_assembler.py). Without either that one test is skipped.
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

pytest.importorskip("torch")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import test_own_tera as tot  # noqa: E402
from obs_assembler import ObservationAssembler  # noqa: E402
from obs_search import LINE, SearchOwnTeraObservationAssembler, checkpoint_search_own_tera  # noqa: E402
from obs_tera import OwnTeraObservationAssembler  # noqa: E402

SEARCH_REF = "origin/search"


# --------------------------------------------------------------------------- battles


def opponent_terastallized():
    """The opponent's Arceus (Normal) has used Judgment (Normal) and Terastallized into Fire."""
    battle = tot.with_an_opponent(tot.make_battle())
    own = tot.name_of(tot.REQUEST["side"]["pokemon"][0])
    battle.parse_message(["", "move", "p2a: Arceus", "Judgment", f"p1a: {own}"])
    battle.parse_message(["", "-terastallize", "p2a: Arceus", "Fire"])
    return battle


STATES = dict(tot.STATES)
STATES["an Illusion's Tera"] = tot.illusion_battle
STATES["the opponent Terastallized"] = opponent_terastallized
STATES["both Terastallized"] = lambda: _both()


def _both():
    battle = tot.terastallized_battle()
    tot.with_an_opponent(battle)
    own = tot.name_of(tot.REQUEST["side"]["pokemon"][0])
    battle.parse_message(["", "move", "p2a: Arceus", "Judgment", f"p1a: {own}"])
    battle.parse_message(["", "-terastallize", "p2a: Arceus", "Fire"])
    return battle


def stab_positions(asm, slot: int) -> list:
    single = asm.move_scalar_dim // 4
    start = asm.offsets["move_scalars"][0] + slot * asm.move_scalar_dim
    return [start + i * single + asm.meta["feature_map"]["move"]["stab_flag"] for i in range(4)]


# --------------------------------------------------------------------------- which checkpoints


def test_a_search_checkpoint_is_one_whose_run_cfg_says_so_and_that_has_no_obs_settings():
    on = {"model": {}, "run_cfg": {"obs": {"own_tera_type": True}}}
    assert checkpoint_search_own_tera(on) is True
    assert checkpoint_search_own_tera({"model": {}, "run_cfg": {"obs": {"own_tera_type": False}}}) is False
    assert checkpoint_search_own_tera({"model": {}, "run_cfg": {"obs": {"n_slots": 6}}}) is False      # from before the switch
    assert checkpoint_search_own_tera({"model": {}, "run_cfg": {}}) is False
    assert checkpoint_search_own_tera({"model": {}}) is False and checkpoint_search_own_tera(None) is False
    # this line's own record decides, whatever the run_cfg holds (r5 and later write both a run_cfg and obs_settings)
    assert checkpoint_search_own_tera(dict(on, obs_settings={"own_tera_type": True})) is False
    assert checkpoint_search_own_tera(dict(on, obs_settings={})) is False
    assert checkpoint_search_own_tera(dict(on, obs_settings=None)) is True


# --------------------------------------------------------------------------- the bytes


@pytest.mark.parametrize("state", ["first request", "after a switch", "with an opponent"])
def test_before_anyone_terastallizes_it_is_this_lines_own_tera_observation(state):
    ours = OwnTeraObservationAssembler().assemble(STATES[state](), events=[])
    theirs = SearchOwnTeraObservationAssembler().assemble(STATES[state](), events=[])
    assert np.array_equal(ours, theirs)
    assert not np.array_equal(theirs, ObservationAssembler().assemble(STATES[state](), events=[]))    # the types are shown


def test_after_a_tera_the_original_types_keep_stab_and_nothing_else_differs_from_the_plain_observation():
    battle = tot.terastallized_battle()
    mon = list(battle.team.values())[0]
    assert mon.is_terastallized and mon.active
    asm = SearchOwnTeraObservationAssembler()
    plain = ObservationAssembler().assemble(tot.terastallized_battle(), events=[])
    obs = asm.assemble(battle, events=[])
    moves = list(mon.moves.values())[:4]
    original = set(mon.base_types)
    restored = [i for i, mv in enumerate(moves) if mv.type in original]
    assert restored, "the fixture's active Pokemon has a move of one of its original types"
    flags = stab_positions(asm, 0)
    assert all(obs[flags[i]] == 1.0 for i in restored)
    # what differs from the plain observation: those flags (where the encoder had dropped them) and
    # the Tera type blocks of the own Pokemon that have not Terastallized
    types_start, types_end = asm.meta["feature_map"]["body"]["types_raw"]
    half = (types_end - types_start) // 2
    tera_cells = {asm.offsets["pokemon_body"][0] + slot * asm.pokemon_scalar_dim + types_start + half + k
                  for slot in range(1, 6) for k in range(half)}
    allowed = set(flags[i] for i in restored) | tera_cells
    differ = set(int(x) for x in np.nonzero(obs != plain)[0])
    assert differ and differ <= allowed
    assert len(differ & tera_cells) == 5, "each of the five that have not Terastallized shows its type"
    assert any(plain[flags[i]] == 0.0 for i in restored), "poke-env reports only the Tera type: the encoder drops the flag"
    # this line's own-Tera observation never restores the flag: the two lines differ exactly there
    ours = OwnTeraObservationAssembler().assemble(tot.terastallized_battle(), events=[])
    assert set(int(x) for x in np.nonzero(obs != ours)[0]) == {flags[i] for i in restored if plain[flags[i]] == 0.0}


def test_the_opponents_terastallized_pokemon_keeps_stab_on_its_original_types_too():
    asm = SearchOwnTeraObservationAssembler()
    battle = opponent_terastallized()
    foe = list(battle.opponent_team.values())[0]
    assert foe.is_terastallized and [m.id for m in foe.moves.values()] == ["judgment"]
    obs = asm.assemble(battle, events=[])
    ours = OwnTeraObservationAssembler().assemble(opponent_terastallized(), events=[])
    flag = stab_positions(asm, 6)[0]
    assert obs[flag] == 1.0 and ours[flag] == 0.0
    assert [int(x) for x in np.nonzero(obs != ours)[0]] == [flag]


def _reference_class():
    """The search branch's ObservationAssembler, from PSPPO_SEARCH_REF or from this clone."""
    folder = os.environ.get("PSPPO_SEARCH_REF")
    if folder:
        path = os.path.join(folder, "obs_assembler.py")
        if not os.path.exists(path):
            pytest.skip(f"no obs_assembler.py in PSPPO_SEARCH_REF={folder}")
        source = open(path, encoding="utf-8").read()
    else:
        got = subprocess.run(["git", "-C", REPO, "show", f"{SEARCH_REF}:obs_assembler.py"], capture_output=True)
        if got.returncode != 0:
            pytest.skip(f"{SEARCH_REF} is not in this clone and PSPPO_SEARCH_REF is not set")
        source = got.stdout.decode("utf-8")
    if "_restore_original_stab" not in source:
        pytest.skip("that obs_assembler.py is from before the search branch's STAB change")
    spec = importlib.util.spec_from_loader("search_ref_obs_assembler", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, "search_ref_obs_assembler.py", "exec"), module.__dict__)
    return module.ObservationAssembler


@pytest.mark.parametrize("fast", [True, False])
def test_the_bytes_are_those_of_the_search_branchs_own_assembler(fast):
    reference = _reference_class()(fast=fast, own_tera_type=True)
    here = SearchOwnTeraObservationAssembler(fast=fast)
    seen_restore = 0
    for name, make in STATES.items():
        for events in ([], [["", "move", "p2a: Arceus", "Judgment", "p1a: X"], ["", "-damage", "p1a: X", "80/100"]]):
            want = reference.assemble(make(), events=events)
            got = here.assemble(make(), events=events)
            assert want.shape == got.shape and np.array_equal(want, got), (name, np.nonzero(want != got)[0][:10])
        seen_restore += int(not np.array_equal(got, OwnTeraObservationAssembler(fast=fast).assemble(make(), events=[])))
    assert seen_restore >= 3, "the states include Terastallized Pokemon on both sides"


# --------------------------------------------------------------------------- the players


def _checkpoint(tmp_path, name, run_obs, obs_settings="absent"):
    import config
    ck = {"model": config.RunConfig.default().make_model().state_dict(), "run_cfg": {"obs": dict(run_obs)}}
    if obs_settings != "absent":
        ck["obs_settings"] = obs_settings
    path = tmp_path / name
    torch.save(ck, path)
    return str(path)


def test_the_policy_builds_the_search_observation_for_a_search_checkpoint_and_only_for_one(tmp_path):
    from hybrid import HybridConfig, PolicyNet

    base = {"n_slots": 6, "n_move_slots": 4, "hp_bins": 11, "vocab_path": "vocab.json"}
    search = PolicyNet(HybridConfig(checkpoint=_checkpoint(tmp_path, "baseline_update_000001.pt", dict(base, own_tera_type=True)), device="cpu"))
    assert search.obs_line == LINE == "search" and type(search.assembler) is SearchOwnTeraObservationAssembler
    assert search.own_tera_type is True and search.obs_abilities == {}
    obs = search.assembler.assemble(tot.terastallized_battle(), events=[])
    assert np.isfinite(search.logits("t", obs.astype(np.float32))).all()

    off = PolicyNet(HybridConfig(checkpoint=_checkpoint(tmp_path, "learner_update_000002.pt", dict(base, own_tera_type=False)), device="cpu"))
    old = PolicyNet(HybridConfig(checkpoint=_checkpoint(tmp_path, "learner_update_000003.pt", base), device="cpu"))
    for pol in (off, old):
        assert pol.obs_line == "own" and type(pol.assembler) is ObservationAssembler and pol.own_tera_type is False

    # release r5 and later: "obs_settings" decides, and builds this line's classes as before
    r5 = PolicyNet(HybridConfig(checkpoint=_checkpoint(tmp_path, "learner_update_000004.pt", base, {"own_tera_type": True}), device="cpu"))
    assert r5.obs_line == "own" and type(r5.assembler) is OwnTeraObservationAssembler
    both = PolicyNet(HybridConfig(checkpoint=_checkpoint(tmp_path, "learner_update_000005.pt", dict(base, own_tera_type=True), {"own_tera_type": True}), device="cpu"))
    assert both.obs_line == "own" and type(both.assembler) is OwnTeraObservationAssembler

    # a caller that names the observation itself gets what it names
    asked = PolicyNet(HybridConfig(checkpoint=search.path, device="cpu", own_tera_type=False))
    assert asked.obs_line == "own" and type(asked.assembler) is ObservationAssembler


def test_the_window_lists_baseline_networks_under_their_own_name_after_the_main_line():
    source = open(os.path.join(REPO, "app", "PSBotLookahead.cs"), encoding="utf-8").read()
    for text in ('Directory.GetFiles(checkpointsDir, "*_update_*.pt")',
                 '@"^(learner|baseline)_update_(\\d+)\\.pt$"',
                 'c.Baseline = m.Groups[1].Value == "baseline";',
                 'a.Baseline != b.Baseline ? (a.Baseline ? 1 : -1)',          # the main line first: the default pick stays its newest
                 '(c.Baseline ? "baseline run, update " : "update ")',
                 '(i == 0 && !c.Baseline ? "   newest" : "")'):
        assert text in source, text
