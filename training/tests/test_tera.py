"""
Terastallization from the request to the order, without a server.

    python -m pytest tests/test_tera.py -q -p no:cacheprovider

A real move request (tests/fixtures/tera_request.json, as a local Showdown server sent it) is
given to a real poke-env 0.15 Battle, and the checks follow the Tera actions through the code
the ladder player runs: the action mask, the observation, hybrid.combine, map_index_to_order
and the text of the order. The same path was watched on a local server in real games
(bench/tera_usage.py, bench/notes/tera.md): 217 of 217, 368 of 368 Tera orders answered by the
server's "-terastallize" line.

What they pin down:
  * the path works: Tera is offered exactly when the request offers it, a Tera action becomes
    "/choose move <id> terastallize", and the observation's Tera flags follow the battle;
  * the hole: poke-env does not read "teraType", so the observation every checkpoint so far was
    trained on shows nothing of the player's own Tera types before it Terastallizes;
  * ObservationAssembler(own_tera_type=True) fills exactly that and nothing else, and is off
    unless asked for.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)          # vocab.json is opened by a relative path

import numpy as np  # noqa: E402
from poke_env.battle import Battle  # noqa: E402
from poke_env.player import Player  # noqa: E402

import obs_assembler  # noqa: E402
from obs_assembler import ObservationAssembler, own_tera_type_name  # noqa: E402

with open(os.path.join(REPO, "tests", "fixtures", "tera_request.json"), encoding="utf-8") as _f:
    REQUEST = json.load(_f)["request"]

TERA = slice(10, 14)


def make_battle(request=None) -> Battle:
    battle = Battle("battle-gen9randombattle-1", "tester", logging.getLogger("test_tera"), gen=9)
    battle.parse_request(copy.deepcopy(request if request is not None else REQUEST))
    return battle


@pytest.fixture(scope="module")
def plain() -> ObservationAssembler:
    return ObservationAssembler()


@pytest.fixture(scope="module")
def shown() -> ObservationAssembler:
    return ObservationAssembler(own_tera_type=True)


def tera_block(asm: ObservationAssembler, obs: np.ndarray, slot: int) -> np.ndarray:
    t0, t1 = asm.meta["feature_map"]["body"]["types_raw"]
    half = (t1 - t0) // 2
    start = asm.offsets["pokemon_body"][0] + slot * asm.pokemon_scalar_dim + t0 + half
    return obs[start:start + half]


def type_id(asm: ObservationAssembler, name: str) -> int:
    return asm.vocab_map["pokemon.type"][name.lower()] + 1


def request_tera_types() -> list:
    return [p["teraType"] for p in REQUEST["side"]["pokemon"]]


# ---------------------------------------------------------------------------
# The fixture is what the tests assume
# ---------------------------------------------------------------------------

def test_the_request_offers_tera_and_names_a_tera_type_for_every_pokemon():
    assert REQUEST["active"][0]["canTerastallize"]
    assert len(REQUEST["side"]["pokemon"]) == 6
    assert all(request_tera_types())
    assert len(set(request_tera_types())) > 1   # otherwise a mix-up between slots would pass


# ---------------------------------------------------------------------------
# Mask and order: the path a Tera action takes
# ---------------------------------------------------------------------------

def test_mask_offers_tera_for_the_moves_that_can_be_used(plain):
    battle = make_battle()
    assert battle.can_tera
    mask = plain.create_action_mask(battle, list(battle.team.values()))
    assert mask[:4].tolist() == [1, 1, 1, 1]
    assert mask[TERA].tolist() == mask[:4].tolist()


def test_mask_does_not_offer_tera_when_the_request_does_not(plain):
    request = copy.deepcopy(REQUEST)
    del request["active"][0]["canTerastallize"]
    battle = make_battle(request)
    assert not battle.can_tera
    mask = plain.create_action_mask(battle, list(battle.team.values()))
    assert mask[:4].tolist() == [1, 1, 1, 1]
    assert mask[TERA].tolist() == [0, 0, 0, 0]


def test_a_disabled_move_has_no_tera_action_either(plain):
    request = copy.deepcopy(REQUEST)
    request["active"][0]["moves"][2]["disabled"] = True
    battle = make_battle(request)
    mask = plain.create_action_mask(battle, list(battle.team.values()))
    assert mask[:4].tolist() == [1, 1, 0, 1]
    assert mask[TERA].tolist() == [1, 1, 0, 1]


def test_the_assembled_observation_carries_the_same_mask(plain):
    battle = make_battle()
    obs = plain.assemble(battle, events=[])
    m0, m1 = plain.offsets["action_mask"]
    assert obs[m0:m1].tolist() == plain.create_action_mask(battle, list(battle.team.values())).tolist()


@pytest.mark.parametrize("slot", [0, 1, 2, 3])
def test_a_tera_action_becomes_a_tera_order(plain, slot):
    battle = make_battle()
    move_id = REQUEST["active"][0]["moves"][slot]["id"]

    target, kwargs = plain.map_index_to_order(10 + slot, battle)
    assert target.id == move_id and kwargs == {"terastallize": True}
    order = Player.create_order(target, **kwargs)
    assert order.message == f"/choose move {move_id} terastallize"
    assert plain.map_order_to_index(order, battle) == 10 + slot

    target, kwargs = plain.map_index_to_order(slot, battle)
    order = Player.create_order(target, **kwargs)
    assert order.message == f"/choose move {move_id}"
    assert plain.map_order_to_index(order, battle) == slot


def test_combine_plays_the_tera_action_when_it_is_the_best_legal_one(plain):
    from hybrid import HybridConfig, combine

    cfg = HybridConfig(beta=0.0, veto_gap=None, policy_weight=1.0, greedy=True)   # ladder.py's settings
    battle = make_battle()
    mask = plain.create_action_mask(battle, list(battle.team.values()))
    logits = np.zeros(14, dtype=np.float32)
    logits[12] = 3.0
    assert combine(logits, None, mask, cfg, None) == 12
    logits[12], logits[2] = 0.0, 3.0
    assert combine(logits, None, mask, cfg, None) == 2
    # masked out: the same logits must not produce a Tera order
    logits[12], logits[2] = 3.0, 1.0
    mask[TERA] = 0.0
    assert combine(logits, None, mask, cfg, None) == 2


# ---------------------------------------------------------------------------
# What the observation shows of the own Tera type
# ---------------------------------------------------------------------------

def test_poke_env_does_not_read_the_tera_type_and_own_tera_type_name_does():
    battle = make_battle()
    team = list(battle.team.values())
    assert [mon.tera_type for mon in team] == [None] * 6
    assert [own_tera_type_name(mon) for mon in team] == [t.lower() for t in request_tera_types()]


def test_own_tera_type_name_without_a_request():
    class Bare:
        tera_type = None
        _last_request = None

    assert own_tera_type_name(Bare()) is None
    assert own_tera_type_name(object()) is None


def test_the_observation_the_checkpoints_were_trained_on_shows_no_own_tera_type(plain):
    assert plain.own_tera_type is False
    obs = plain.assemble(make_battle(), events=[])
    for slot in range(12):
        assert not tera_block(plain, obs, slot).any()


def test_own_tera_type_shows_each_pokemon_its_type_and_changes_nothing_else(plain, shown):
    battle = make_battle()
    off, on = plain.assemble(battle, events=[]), shown.assemble(battle, events=[])
    assert off.shape == on.shape
    for slot, name in enumerate(request_tera_types()):
        block = tera_block(shown, on, slot)
        assert np.flatnonzero(block).tolist() == [type_id(shown, name)]
        assert block[type_id(shown, name)] == 1.0
    for slot in range(6, 12):
        assert not tera_block(shown, on, slot).any()
    assert int((off != on).sum()) == 6


def test_the_default_did_not_change():
    assert ObservationAssembler().own_tera_type is False
    assert obs_assembler.ObservationAssembler.__init__.__defaults__[-1] is False


# ---------------------------------------------------------------------------
# After Terastallizing
# ---------------------------------------------------------------------------

def terastallized_battle() -> Battle:
    battle = make_battle()
    active = REQUEST["side"]["pokemon"][0]
    name = active["ident"].split(": ", 1)[1]
    battle.parse_message(["", "-terastallize", f"p1a: {name}", active["teraType"]])
    # the next request: Tera is no longer offered, the Pokemon is marked
    request = copy.deepcopy(REQUEST)
    del request["active"][0]["canTerastallize"]
    request["side"]["pokemon"][0]["terastallized"] = active["teraType"]
    request["rqid"] = int(REQUEST.get("rqid", 0)) + 2
    battle.parse_request(request)
    return battle


def test_after_terastallizing_the_flags_are_set_and_tera_is_not_offered(plain, shown):
    battle = terastallized_battle()
    assert battle.used_tera and not battle.can_tera and battle.active_pokemon.is_terastallized
    name = REQUEST["side"]["pokemon"][0]["teraType"]
    for asm in (plain, shown):
        obs = asm.assemble(battle, events=[])
        m0, m1 = asm.offsets["action_mask"]
        assert obs[m0:m1][TERA].tolist() == [0, 0, 0, 0]
        g0 = asm.offsets["global_scalars"][0]
        assert obs[g0 + 1] == 1.0 and obs[g0 + 2] == 0.0          # we used Tera, the opponent did not
        flags = asm.meta["feature_map"]["body"]["flags_raw"][0]
        assert obs[asm.offsets["pokemon_body"][0] + flags + 2] == 1.0   # the active Pokemon's Tera flag
        assert np.flatnonzero(tera_block(asm, obs, 0)).tolist() == [type_id(asm, name)]
    # the two observations now differ only in the five Pokemon that have not Terastallized, and in
    # the STAB flag of the active's moves of its original types (kept in gen 9, restored by the switch)
    off, on = plain.assemble(battle, events=[]), shown.assemble(battle, events=[])
    mon = battle.active_pokemon
    restored = sum(1 for mv in list(mon.moves.values())[:4]
                   if mv.type in set(mon.base_types) and mv.type != mon.tera_type)
    assert int((off != on).sum()) == 5 + restored


# ---------------------------------------------------------------------------
# The switch: config.ObsConfig.own_tera_type, recorded in checkpoints, followed by the players
# ---------------------------------------------------------------------------

def _cfg_with(flag: bool):
    import dataclasses
    from config import RunConfig

    cfg = RunConfig.default()
    return dataclasses.replace(cfg, obs=dataclasses.replace(cfg.obs, own_tera_type=flag))


def test_poke_env_upgrade_would_not_help_installed_version_reads_no_tera_type():
    # The installed poke-env's update_from_request leaves tera_type alone (0.16.1 does too). If this
    # fails, poke-env started reading "teraType" itself and obs_pokemon would light the block for
    # every checkpoint, switch or no switch; SimpleHeuristicsPlayer would start to Terastallize.
    battle = make_battle()
    assert all(mon.tera_type is None for mon in battle.team.values())


def test_config_switch_is_on_by_default_and_builds_the_assembler():
    from config import RunConfig

    cfg = RunConfig.default()
    assert cfg.obs.own_tera_type is True
    assert cfg.make_assembler().own_tera_type is True
    assert cfg.as_dict()["obs"]["own_tera_type"] is True
    off = _cfg_with(False)
    assert off.make_assembler().own_tera_type is False


def test_the_switch_changes_no_size_so_every_checkpoint_still_fits():
    # The Tera type block exists for every Pokemon; the switch only fills it earlier for the own
    # team. Same observation size, same network: no weight surgery is needed.
    off, on = _cfg_with(False), _cfg_with(True)
    a_off, a_on = off.make_assembler(), on.make_assembler()
    assert a_off.total_dim == a_on.total_dim and a_off.offsets == a_on.offsets
    net_off, net_on = off.make_model(), on.make_model()
    shapes = lambda net: {k: tuple(v.shape) for k, v in net.state_dict().items()}  # noqa: E731
    assert shapes(net_off) == shapes(net_on)
    missing, unexpected = net_on.load_state_dict(net_off.state_dict(), strict=False)
    assert not unexpected and not [k for k in missing if "attn_mask" not in k]


def test_checkpoint_own_tera_type_reads_the_run_cfg_and_old_checkpoints_are_off():
    from config import checkpoint_own_tera_type

    assert checkpoint_own_tera_type({"run_cfg": _cfg_with(True).as_dict()}) is True
    assert checkpoint_own_tera_type({"run_cfg": _cfg_with(False).as_dict()}) is False
    old = _cfg_with(False).as_dict()
    del old["obs"]["own_tera_type"]               # a checkpoint from before the switch
    assert checkpoint_own_tera_type({"run_cfg": old}) is False
    assert checkpoint_own_tera_type({"model": {}}) is False
    assert checkpoint_own_tera_type(None) is False


@pytest.mark.parametrize("flag", [False, True, None])
def test_policy_net_follows_the_checkpoint(tmp_path, flag):
    import torch
    from hybrid import HybridConfig, PolicyNet

    cfg = _cfg_with(bool(flag))
    run_cfg = cfg.as_dict()
    if flag is None:
        del run_cfg["obs"]["own_tera_type"]      # a checkpoint from before the switch
    path = tmp_path / "learner_update_000001.pt"
    torch.save({"model": cfg.make_model().state_dict(), "run_cfg": run_cfg}, path)
    pol = PolicyNet(HybridConfig(checkpoint=str(path), device="cpu"))
    assert pol.assembler.own_tera_type is bool(flag)
    obs = pol.assembler.assemble(make_battle(), events=[])
    assert np.isfinite(pol.logits("t", obs.astype(np.float32))).all()


def test_train_flag_sets_the_switch():
    pytest.importorskip("ray")
    import train

    assert train.build_config(train.parse_args([])).obs.own_tera_type is True
    assert train.build_config(train.parse_args(["--own-tera-type"])).obs.own_tera_type is True
    assert train.build_config(train.parse_args(["--no-own-tera-type"])).obs.own_tera_type is False


# ---------------------------------------------------------------------------
# STAB after Terastallizing: poke-env reports only the Tera type, the original types keep STAB
# ---------------------------------------------------------------------------

def stab_flags(asm: ObservationAssembler, obs: np.ndarray, slot: int) -> list:
    m = asm.meta["feature_map"]["move"]
    single = asm.move_scalar_dim // 4
    start = asm.offsets["move_scalars"][0] + slot * asm.move_scalar_dim
    return [float(obs[start + i * single + m["stab_flag"]]) for i in range(4)]


@pytest.mark.parametrize("fast", [True, False])
def test_with_the_switch_on_the_original_types_keep_stab_after_tera(fast):
    battle = terastallized_battle()
    mon = battle.active_pokemon
    base = set(mon.base_types)
    tera = mon.tera_type
    moves = list(mon.moves.values())[:4]
    expected_on = [1.0 if (mv.type in base or mv.type == tera) else 0.0 for mv in moves]
    expected_off = [1.0 if mv.type == tera else 0.0 for mv in moves]
    off = ObservationAssembler(fast=fast).assemble(battle, events=[])
    on = ObservationAssembler(fast=fast, own_tera_type=True).assemble(battle, events=[])
    assert stab_flags(plain_asm := ObservationAssembler(fast=fast), off, 0)[:len(moves)] == expected_off
    assert stab_flags(plain_asm, on, 0)[:len(moves)] == expected_on


def test_before_tera_the_stab_flag_is_the_same_either_way():
    battle = make_battle()
    off = ObservationAssembler().assemble(battle, events=[])
    on = ObservationAssembler(own_tera_type=True).assemble(battle, events=[])
    for slot in range(12):
        assert stab_flags(ObservationAssembler(), off, slot) == stab_flags(ObservationAssembler(), on, slot)
