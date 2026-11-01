"""
obs_league: a network of the league run is played on the observation it was trained with (this
line's own, the main line's, then train-v2's blocks), by the network it was trained as; every other
network as before.

    python -m pytest -q tests/test_obs_league.py

What is held:

  * which checkpoints: "obs_settings" with main_line and exactly the main line's settings
    (checkpoint_league_blocks); M24381's own settings, a train-v2 run_cfg and no entry are not
    league networks; main_line with other settings is refused
  * the class: the MRO is the league's (AbilityObservationAssembler, OwnTeraObservationAssembler,
    BlocksBase, ObservationAssembler), with every fix of obs_abil and the own Tera type on
  * the layout: the first part is M24381's (the same offsets, 8,567 numbers), then the matchup block
    and the moveset block where obs_v2 puts them
  * on a fixture battle: the first part byte for byte M24381's observation, the blocks written, and
    make_model with the blocks runs on it

The bytes against the league branch's own code are bench/v2_exactness.py app against that branch's
bench/league_exactness.py ref on recorded battles (box c). No Showdown server is started.
"""

from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(REPO)          # vocab.json is opened by a relative path

pytest.importorskip("poke_env")

from obs_abil import AbilityObservationAssembler, SETTINGS, make_observation_assembler  # noqa: E402
from obs_assembler import ObservationAssembler  # noqa: E402
from obs_league import (BlocksBase, LINE, MAIN_LINE_DIM, MAIN_LINE_SETTINGS, checkpoint_league_blocks,  # noqa: E402
                        league_assembler, league_assembler_class)
from obs_tera import OwnTeraObservationAssembler  # noqa: E402
from obs_v2 import V2ObservationAssembler, checkpoint_v2_blocks  # noqa: E402

LEAGUE_OBS_SETTINGS = dict(MAIN_LINE_SETTINGS, main_line=True, matchup=True, movesets=True)
LEAGUE_RUN_CFG = {"obs": {"own_tera_type": False, "main_line": True, "matchup": True, "movesets": True}}


def test_which_checkpoints():
    assert LINE == "league"
    ck = {"obs_settings": dict(LEAGUE_OBS_SETTINGS), "run_cfg": LEAGUE_RUN_CFG}
    assert checkpoint_league_blocks(ck) == {"matchup": True, "movesets": True}
    assert checkpoint_v2_blocks(ck) is None          # its obs_settings entry keeps it off obs_v2
    no_sets = dict(LEAGUE_OBS_SETTINGS, movesets=False)
    assert checkpoint_league_blocks({"obs_settings": no_sets}) == {"matchup": True, "movesets": False}
    # M24381 itself, a train-v2 network, nothing: not league networks.
    assert checkpoint_league_blocks({"obs_settings": dict(MAIN_LINE_SETTINGS)}) is None
    assert checkpoint_league_blocks({"run_cfg": {"obs": {"matchup": True, "movesets": True}}}) is None
    assert checkpoint_league_blocks({}) is None
    assert checkpoint_league_blocks(None) is None


@pytest.mark.parametrize("change", [
    {"regenerator": False},                 # a fix missing
    {"own_tera_type": False},
    {"matchup": False},                     # main_line without the block
    {"something_new": True},                # a setting this code does not build
])
def test_other_main_line_settings_are_refused(change):
    with pytest.raises(RuntimeError):
        checkpoint_league_blocks({"obs_settings": dict(LEAGUE_OBS_SETTINGS, **change)})


def test_run_cfg_must_agree():
    bad = {"obs": dict(LEAGUE_RUN_CFG["obs"], movesets=False)}
    with pytest.raises(RuntimeError):
        checkpoint_league_blocks({"obs_settings": dict(LEAGUE_OBS_SETTINGS), "run_cfg": bad})


def test_class_and_settings():
    a = league_assembler(movesets=True)
    mro = type(a).__mro__
    order = [mro.index(c) for c in (AbilityObservationAssembler, OwnTeraObservationAssembler, BlocksBase, ObservationAssembler)]
    assert order == sorted(order)
    assert a.own_tera_type is True
    assert all(a.settings.get(n) for n in SETTINGS)
    assert league_assembler_class(True) is league_assembler_class(True)


def test_layout():
    native = make_observation_assembler(own_tera_type=True, abil={n: True for n in SETTINGS})
    v2 = V2ObservationAssembler(movesets=True)
    a = league_assembler(movesets=True)
    for name, span in native.offsets.items():
        assert a.offsets[name] == span, name
    assert native.total_dim == MAIN_LINE_DIM
    assert a.offsets["matchup"] == v2.offsets["matchup"]
    assert a.offsets["movesets"] == v2.offsets["movesets"]
    assert a.total_dim == v2.total_dim
    assert a.meta["matchup"] == v2.meta["matchup"] and a.meta["movesets"] == v2.meta["movesets"]
    b = league_assembler(movesets=False)
    assert "movesets" not in b.offsets and b.total_dim == a.offsets["movesets"][0]


def test_fixture_battle_first_part_and_network():
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    import test_own_tera as tot
    from config import RunConfig

    native = make_observation_assembler(own_tera_type=True, abil={n: True for n in SETTINGS})
    a = league_assembler(movesets=True)
    battle = tot.with_an_opponent(tot.make_battle())
    want = native.assemble(battle, events=[])
    got = a.assemble(tot.with_an_opponent(tot.make_battle()), events=[])
    assert got.dtype == want.dtype and got.shape == (a.total_dim,)
    assert np.array_equal(got[:MAIN_LINE_DIM].view(np.uint16), want.view(np.uint16))
    assert np.any(got[MAIN_LINE_DIM:] != 0)
    net = RunConfig.default().make_model(matchup=True, movesets=True).eval()
    with torch.no_grad():
        out = net(torch.from_numpy(got.astype(np.float32))[None])
    assert out[0].shape == (1, 14) and torch.isfinite(out[0]).all()
