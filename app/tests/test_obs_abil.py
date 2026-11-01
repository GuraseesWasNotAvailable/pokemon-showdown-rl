"""
The ability / item / switching fixes of the observation (obs_abil.py, abil_worker.py, train.py
--obs-abilities), the per-action-type weights (ppo_core, learner, train.py), and the ladder's
immunity guard (immunity_guard.py).

    python -m pytest tests/test_obs_abil.py -q -p no:cacheprovider

What is held:

  * off is what it was: make_observation_assembler() with nothing on is ObservationAssembler
    itself, with the own Tera type alone obs_tera's class; AbilityObservationAssembler with no
    setting gives the base class's bytes on every battle state here; a checkpoint without the
    entry loads as "off"; combat_bounds with both fixes off is get_combat_bounds
  * each fix writes the value the audit says, and only into the places it may write
  * train.py: no flag, no classes; with --obs-abilities the workers and the learner's
    obs_settings carry the settings (and the own Tera type with both flags)
  * the weights: absent or 1 is the loss of before, bit for bit; a weight scales the rows it names
  * the guard blocks what certainly does nothing and nothing else

No Showdown server and no Ray are started. Nothing is removed.
"""
from __future__ import annotations

import argparse
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

pytest.importorskip("numpy")
pytest.importorskip("poke_env")
torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402
from poke_env.battle import Battle  # noqa: E402
from poke_env.battle.move import Move  # noqa: E402
from poke_env.battle.pokemon import Pokemon  # noqa: E402

import obs_abil  # noqa: E402
from obs_abil import (SETTINGS, AbilityObservationAssembler, announcements, checkpoint_obs_settings,  # noqa: E402
                      combat_bounds, make_observation_assembler, parse_settings, settings_from_checkpoint)
from obs_assembler import ObservationAssembler  # noqa: E402
from obs_pokemon import get_combat_bounds  # noqa: E402
from obs_tera import OwnTeraObservationAssembler  # noqa: E402
from utils import get_id  # noqa: E402

with open(os.path.join(REPO, "tests", "fixtures", "tera_request.json"), encoding="utf-8") as _f:
    REQUEST = json.load(_f)["request"]


# --------------------------------------------------------------------------- battles

def make_battle() -> Battle:
    battle = Battle("battle-gen9randombattle-7", "tester", logging.getLogger("test_obs_abil"), gen=9)
    battle.parse_request(copy.deepcopy(REQUEST))
    return battle


OPPONENTS = [("Lanturn", "Lanturn, L88, F", "100/100"), ("Dondozo", "Dondozo, L80, M", "100/100"),
             ("Pelipper", "Pelipper, L86, F", "100/100"), ("Amoonguss", "Amoonguss, L88, F", "60/100"),
             ("Gyarados", "Gyarados, L80, M", "100/100")]


def with_opponents(battle: Battle, n: int = len(OPPONENTS)) -> Battle:
    for name, details, hp in OPPONENTS[:n]:
        battle.parse_message(["", "switch", f"p2a: {name}", details, hp])
    return battle


EVENTS = [
    ["", "-weather", "RainDance", "[from] ability: Drizzle", "[of] p2a: Pelipper"],
    ["", "-ability", "p2a: Gyarados", "Intimidate", "boost"],
    ["", "-status", "p2a: Amoonguss", "tox", "[from] item: Toxic Orb"],
    ["", "-damage", "p1a: Espeon", "80/100"],
]


def states():
    yield "first request", make_battle(), []
    yield "one opponent", with_opponents(make_battle(), 1), []
    yield "five opponents", with_opponents(make_battle()), []
    b = with_opponents(make_battle())
    for e in EVENTS:
        b.parse_message(e)
    yield "after the events", b, EVENTS


def ids(asm):
    return asm.vocab_map


# --------------------------------------------------------------------------- off is what it was

def test_settings_parse():
    assert parse_settings(None) == {} and parse_settings("") == {} and parse_settings([]) == {}
    assert list(parse_settings("all")) == list(SETTINGS)
    assert parse_settings("item_unknown, abil_sets") == {"abil_sets": True, "item_unknown": True}
    with pytest.raises(ValueError):
        parse_settings("abil_set")
    assert settings_from_checkpoint({"model": {}}) == {}
    assert settings_from_checkpoint({"obs_settings": {"own_tera_type": True}}) == {}
    assert settings_from_checkpoint({"obs_settings": {"abil_sets": True, "regenerator": False}}) == {"abil_sets": True}
    assert checkpoint_obs_settings(False, None) is None
    assert checkpoint_obs_settings(True, None) == {"own_tera_type": True}
    assert checkpoint_obs_settings(True, "abil_sets") == {"own_tera_type": True, "abil_sets": True}


def test_off_is_the_base_class():
    assert type(make_observation_assembler()) is ObservationAssembler
    assert type(make_observation_assembler(abil={})) is ObservationAssembler
    assert type(make_observation_assembler(own_tera_type=True)) is OwnTeraObservationAssembler
    assert isinstance(make_observation_assembler(abil="all"), AbilityObservationAssembler)


@pytest.mark.parametrize("name,battle,events", list(states()), ids=[s[0] for s in states()])
def test_no_setting_gives_the_bytes_of_before(name, battle, events):
    base = ObservationAssembler().assemble(copy.deepcopy(battle), events=copy.deepcopy(events))
    none = AbilityObservationAssembler(settings={}).assemble(copy.deepcopy(battle), events=copy.deepcopy(events))
    assert base.dtype == none.dtype and base.shape == none.shape
    assert base.tobytes() == none.tobytes()


def test_combat_bounds_off_is_get_combat_bounds():
    b = with_opponents(make_battle())
    mons = list(b.team.values()) + list(b.opponent_team.values())
    for mon in mons:
        for stat in ("atk", "def", "spa", "spd", "spe"):
            assert combat_bounds(mon, stat, mon.item, mon.ability, mon.possible_abilities, False, None) == \
                get_combat_bounds(mon, stat), (mon.species, stat)


# --------------------------------------------------------------------------- each fix

def _allowed_positions(asm) -> set:
    """Every place AbilityObservationAssembler may write (its docstring): per Pokemon the 4 ability
    ids, the item id, the two fog flags, the 15 stat bounds, the 2 HP numbers; the transition's ability ids."""
    body = asm.meta["feature_map"]["body"]
    out = set()
    for slot in range(12):
        b0 = asm.offsets["pokemon_body"][0] + slot * asm.pokemon_scalar_dim
        out.update(asm.offsets["ability_ids"][0] + slot * 4 + k for k in range(4))
        out.add(asm.offsets["pokemon_ids"][0] + slot * 2 + 1)
        out.update((b0 + body["mechanics_raw"][0], b0 + body["mechanics_raw"][0] + 1))
        out.update(b0 + body["stats_int"][0] + k for k in range(15))
        out.update((b0 + body["hp_int"], b0 + body["hp_abs_raw"][0]))
    t = asm.offsets["transition_ids"][0] + asm.meta["feature_map"]["transition"]["ability_ids"][0]
    out.update((t, t + 1))
    return out


def _opp_slot(battle, species):
    return 6 + [m.species for m in battle.opponent_team.values()].index(species)


def test_all_on_writes_only_its_places():
    b = with_opponents(make_battle())
    for e in EVENTS:
        b.parse_message(e)
    base = ObservationAssembler().assemble(copy.deepcopy(b), events=copy.deepcopy(EVENTS))
    asm = AbilityObservationAssembler(settings=parse_settings("all"))
    fixed = asm.assemble(copy.deepcopy(b), events=copy.deepcopy(EVENTS))
    changed = set(np.flatnonzero(base != fixed).tolist())
    assert changed, "nothing changed with every fix on"
    assert changed <= _allowed_positions(asm), sorted(changed - _allowed_positions(asm))[:10]


def test_abil_sets_fills_a_certain_ability_and_the_set_slots():
    b = with_opponents(make_battle())
    asm = AbilityObservationAssembler(settings=parse_settings("abil_sets"))
    base_asm = ObservationAssembler()
    base = base_asm.assemble(copy.deepcopy(b), events=[])
    obs = asm.assemble(copy.deepcopy(b), events=[])
    table = obs_abil.set_abilities_table()
    for species in ("lanturn", "dondozo"):
        sets = table[species]
        assert len(sets) == 1, (species, sets)          # the audit's examples: certain only from the sets
        slot = _opp_slot(b, species)
        a0 = asm.offsets["ability_ids"][0] + slot * 4
        assert obs[a0] == get_id(ids(asm), "pokemon.ability", sets[0])
        assert obs[a0 + 1] == get_id(ids(asm), "pokemon.ability", sets[0])
        assert list(obs[a0 + 2:a0 + 4]) == [0.0, 0.0]
        body = asm.offsets["pokemon_body"][0] + slot * asm.pokemon_scalar_dim
        assert obs[body + asm.meta["feature_map"]["body"]["mechanics_raw"][0] + 1] == 0.0   # ability not unknown
    # the base leaves Dondozo's slot 0 empty (poke-env's Pokedex has three abilities for it)
    slot = _opp_slot(b, "dondozo")
    assert base[base_asm.offsets["ability_ids"][0] + slot * 4] == 0.0


def test_a_hidden_forme_keeps_the_shown_species_ability_unknown():
    # Review of 2026-10-07: Greninja-Bond switches in as "Greninja", so a Greninja is either
    # (Protean or Battle Bond); with abil_sets its slot 0 showed Protean on 48 decisions of a
    # Greninja-Bond. The table now gives the shown species the hidden forme's abilities too.
    table = obs_abil.set_abilities_table()
    assert sorted(table["greninja"]) == ["battlebond", "protean"]
    assert table["greninjabond"] == ["battlebond"]


def test_abil_announced_reads_the_lines_poke_env_skips():
    ab, it = announcements(["", "-weather", "RainDance", "[from] ability: Drizzle", "[of] p2a: Pelipper"])
    assert ab[0] == ("p2: Pelipper", "drizzle") and it == []
    ab, it = announcements(["", "-status", "p2a: Amoonguss", "tox", "[from] item: Toxic Orb"])
    assert it == [("p2: Amoonguss", "toxicorb")] and ab == []
    assert announcements(["", "-ability", "p2a: Gyarados", "Intimidate", "boost"]) == ([], [])
    ab, _ = announcements(["", "-activate", "p2a: Lanturn", "ability: Volt Absorb"])
    assert ab == [("p2: Lanturn", "voltabsorb")]

    b = with_opponents(make_battle())
    asm = AbilityObservationAssembler(settings=parse_settings("abil_announced"))
    b.parse_message(EVENTS[0])
    b.parse_message(EVENTS[2])
    obs = asm.assemble(b, events=[EVENTS[0], EVENTS[2]])
    pel = _opp_slot(b, "pelipper")
    assert obs[asm.offsets["ability_ids"][0] + pel * 4] == get_id(ids(asm), "pokemon.ability", "drizzle")
    amo = _opp_slot(b, "amoonguss")
    if b.opponent_team["p2: Amoonguss"].item == obs_abil.UNKNOWN_ITEM:
        assert obs[asm.offsets["pokemon_ids"][0] + amo * 2 + 1] == get_id(ids(asm), "pokemon.item", "toxicorb")
    # remembered at the next decision, when the line is no longer among the events
    obs2 = asm.assemble(b, events=[])
    assert obs2[asm.offsets["ability_ids"][0] + pel * 4] == get_id(ids(asm), "pokemon.ability", "drizzle")


def test_abil_history_writes_a_bare_ability_line():
    b = with_opponents(make_battle())
    b.parse_message(EVENTS[1])
    t = None
    for setting in (None, "abil_history"):
        asm = AbilityObservationAssembler(settings=parse_settings(setting)) if setting else ObservationAssembler()
        obs = asm.assemble(copy.deepcopy(b), events=[EVENTS[1]])
        t = asm.offsets["transition_ids"][0] + asm.meta["feature_map"]["transition"]["ability_ids"][0]
        if setting is None:
            assert obs[t + 1] == 0.0                       # the audit: 3,352 of 3,368 such lines left it 0
        else:
            assert obs[t + 1] == get_id(ids(asm), "pokemon.ability", "intimidate")
            assert obs[t] == 0.0


def test_item_unknown_means_unknown_and_widens_speed():
    b = with_opponents(make_battle(), 1)
    mon = b.opponent_team["p2: Lanturn"]
    assert mon.item == obs_abil.UNKNOWN_ITEM
    base_asm, asm = ObservationAssembler(), AbilityObservationAssembler(settings=parse_settings("item_unknown"))
    base = base_asm.assemble(copy.deepcopy(b), events=[])
    obs = asm.assemble(copy.deepcopy(b), events=[])
    slot = _opp_slot(b, "lanturn")
    body = asm.offsets["pokemon_body"][0] + slot * asm.pokemon_scalar_dim
    fog = asm.meta["feature_map"]["body"]["mechanics_raw"][0]
    assert base[body + fog] == 0.0 and obs[body + fog] == 1.0
    spe = body + asm.meta["feature_map"]["body"]["stats_int"][0] + 12
    lo, est, hi = combat_bounds(mon, "spe", mon.item, mon.ability, mon.possible_abilities, False, None)
    assert obs[spe + 2] == min(int(hi * 1.5), 799) and obs[spe] == lo
    # a Pokemon known to hold nothing: not unknown, not widened
    lo2, _, hi2 = combat_bounds(mon, "spe", "", None, [], True, None)
    assert hi2 == lo2 or hi2 < int(hi * 1.5)


def test_spe_conditions_only_in_their_weather():
    mon = Pokemon(gen=9, species="kingdra")
    mon._ability = "swiftswim"
    off = combat_bounds(mon, "spe", "", "swiftswim", None, True, (frozenset(), frozenset()))
    rain = combat_bounds(mon, "spe", "", "swiftswim", None, True, (frozenset({"raindance"}), frozenset()))
    before = combat_bounds(mon, "spe", "", "swiftswim", None, True, None)
    assert rain == before and off[2] * 2 == rain[2] or off[2] * 2 >= 799


# --------------------------------------------------------------------------- train.py

def _args(**kw):
    ns = argparse.Namespace(obs_abilities=None, own_tera_type=False)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_train_without_the_flag_has_no_classes():
    pytest.importorskip("ray")
    import train
    assert train.obs_abilities_classes(_args()) is None
    assert train.obs_abilities_classes(argparse.Namespace()) is None


def test_train_with_the_flag():
    pytest.importorskip("ray")
    import train
    got = train.obs_abilities_classes(_args(obs_abilities="all"))
    assert got["settings"] == parse_settings("all")
    assert got["obs_settings"] == parse_settings("all")
    assert got["worker"]._obs_abilities == parse_settings("all") and got["worker"]._obs_own_tera_type is False
    both = train.obs_abilities_classes(_args(obs_abilities="abil_sets", own_tera_type=True))
    assert both["obs_settings"] == {"own_tera_type": True, "abil_sets": True}
    assert both["worker"]._obs_own_tera_type is True
    assert train.obs_abilities_classes(_args(obs_abilities="nonsense")) is None      # logged, the run goes on without


# --------------------------------------------------------------------------- the weights

def test_cross_entropy_weights():
    from ppo_core import action_type_weights, masked_smoothed_cross_entropy
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(32, 14, generator=g)
    mask = (torch.rand(32, 14, generator=g) > 0.3).float()
    act = torch.randint(0, 14, (32,), generator=g)
    plain = masked_smoothed_cross_entropy(logits, act, mask)
    ones = masked_smoothed_cross_entropy(logits, act, mask, weights=torch.ones(32))
    assert torch.allclose(plain, ones, atol=1e-6)
    w = action_type_weights(act, 3.0)
    assert set(w[(act >= 4) & (act <= 9)].tolist()) <= {3.0} and set(w[(act < 4) | (act > 9)].tolist()) <= {1.0}
    weighted = masked_smoothed_cross_entropy(logits, act, mask, weights=w)
    assert not torch.allclose(plain, weighted)


def _teacher_inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(16, 14, generator=g, requires_grad=True)
    mask = torch.ones(16, 14)
    keep = torch.zeros(16, dtype=torch.int32)
    keep[:8] = 1 << 0            # a move kept
    keep[8:12] = (1 << 5) | (1 << 1)   # a switch and a move kept: switching was close
    return logits, mask, keep


def test_teacher_switch_weight():
    from ppo_core import search_teacher_term
    logits, mask, keep = _teacher_inputs()

    def term(**kw):
        track = {k: 0.0 for k in ("lab_n", "aux_sum", "pK_sum", "agree_sum", "e0_n")}
        return search_teacher_term(logits, mask, keep, None, 0.5, 0.0, 16, track, True, **kw), track

    a, ta = term()
    b, tb = term(switch_weight=1.0)
    assert torch.equal(a, b) and ta == tb and "lab_switch_n" not in tb
    c, tc = term(switch_weight=3.0)
    assert tc["lab_switch_n"] == 4.0
    assert float(c) > float(a)


# --------------------------------------------------------------------------- the guard

def _guard_case(att, dfn, move, fields=(), att_ability=None, dfn_item=None):
    import types as _t
    import immunity_guard as ig
    A, D = Pokemon(gen=9, species=att), Pokemon(gen=9, species=dfn)
    if att_ability:
        A._ability = att_ability
    if dfn_item:
        D._item = dfn_item
    battle = _t.SimpleNamespace(active_pokemon=A, opponent_active_pokemon=D, fields={f: 1 for f in fields})
    return ig.move_blocked(Move(move, gen=9), battle, A, D)


@pytest.mark.parametrize("att,dfn,move,why", [
    ("cobalion", "sableye", "aurasphere", "type"),          # the audit's mix examples
    ("kommoo", "whimsicott", "scaleshot", "type"),
    ("zebstrika", "kilowattrel", "highhorsepower", "type"),
    ("metagross", "tsareena", "bulletpunch", "ability"),     # Queenly Majesty, certain from the sets
    ("alomomola", "lumineon", "scald", "ability"),           # Storm Drain
    ("garchomp", "lanturn", "thunderbolt", "ability"),       # Volt Absorb
    ("rillaboom", "gogoat", "woodhammer", "ability"),        # Sap Sipper
    ("garchomp", "skarmory", "earthquake", "type"),
])
def test_guard_blocks(att, dfn, move, why):
    got = _guard_case(att, dfn, move)
    assert got and got.startswith(why), got


def test_guard_leaves_alone():
    assert _guard_case("garchomp", "gengar", "earthquake") is None             # gen9 Gengar has no Levitate
    assert _guard_case("garchomp", "skarmory", "earthquake", fields=("gravity",)) is None
    assert _guard_case("haxorus", "lanturn", "thunderbolt", att_ability="moldbreaker") is None
    assert _guard_case("kangaskhan", "gengar", "doubleedge", att_ability="scrappy") is None
    assert _guard_case("garchomp", "gengar", "terablast") is None              # its type is not fixed
    assert _guard_case("garchomp", "gengar", "swordsdance") is None            # status moves are left to the search
    assert _guard_case("garchomp", "skarmory", "earthquake", dfn_item="ironball") is None


def test_guard_never_blocks_everything():
    import types as _t
    import immunity_guard as ig
    A, D = Pokemon(gen=9, species="cobalion"), Pokemon(gen=9, species="sableye")
    moves = [Move(m, gen=9) for m in ("aurasphere", "closecombat", "stoneedge", "swordsdance")]
    battle = _t.SimpleNamespace(active_pokemon=A, opponent_active_pokemon=D, fields={})
    mapper = lambda a, _b: (moves[a % 10], {})          # noqa: E731
    mask = [1, 1, 0, 0] + [0] * 10
    assert ig.blocked_actions(battle, mask, mapper) == {}                  # both legal moves blocked: nothing is
    mask = [1, 1, 1, 0] + [0] * 10
    assert sorted(ig.blocked_actions(battle, mask, mapper)) == [0, 1]


_REAL_CKPT = os.environ.get("PSPPO_OLD_CKPT", "")


@pytest.mark.skipif(not os.path.isfile(_REAL_CKPT), reason="PSPPO_OLD_CKPT not set (a learner checkpoint of the live run)")
def test_policynet_builds_the_observation_its_checkpoint_says(tmp_path):
    # Review of 2026-10-07: every player (evaluate, play, ladder) goes through hybrid.PolicyNet,
    # which must read obs_settings: without the entry the base ObservationAssembler itself, with it
    # the fixes (and the own Tera type) the checkpoint was trained with.
    import torch
    import hybrid
    from obs_tera import OwnTeraObservationAssembler
    ck = torch.load(_REAL_CKPT, map_location="cpu", weights_only=False)
    ck.pop("obs_settings", None)
    plain = tmp_path / "plain.pt"
    torch.save(ck, plain)
    ck["obs_settings"] = checkpoint_obs_settings(True, "all")
    fixed = tmp_path / "fixed.pt"
    torch.save(ck, fixed)
    ck["obs_settings"] = checkpoint_obs_settings(True, None)
    tera = tmp_path / "tera.pt"
    torch.save(ck, tera)
    p0 = hybrid.PolicyNet(hybrid.HybridConfig(checkpoint=str(plain), device="cpu"))
    assert type(p0.assembler) is ObservationAssembler and p0.obs_abilities == {} and p0.own_tera_type is False
    p1 = hybrid.PolicyNet(hybrid.HybridConfig(checkpoint=str(fixed), device="cpu"))
    assert isinstance(p1.assembler, AbilityObservationAssembler) and p1.assembler.settings == parse_settings("all")
    assert p1.own_tera_type is True and p1.assembler.own_tera_type is True
    p2 = hybrid.PolicyNet(hybrid.HybridConfig(checkpoint=str(tera), device="cpu"))
    assert type(p2.assembler) is OwnTeraObservationAssembler and p2.obs_abilities == {}


def test_an_absorb_heal_gives_the_ability_to_the_healed_pokemon_only():
    """The stage2 review (bench/STAGE2.md 6): '-heal|<ours>|...|[from] ability: Water Absorb|[of] <their attacker>'
    used to try the [of] first and gave THEIR attacker Water Absorb, over a correct ability."""
    ab, _ = announcements(["", "-heal", "p1a: Vaporeon", "100/100", "[from] ability: Water Absorb", "[of] p2a: Quagsire"])
    assert ab == [("p1: Vaporeon", "waterabsorb")]
    ab, _ = announcements(["", "-heal", "p2a: Lanturn", "100/100", "[from] ability: Volt Absorb", "[of] p1a: Espeon"])
    assert ab == [("p2: Lanturn", "voltabsorb")]
    ab, _ = announcements(["", "-damage", "p1a: Espeon", "90/100", "[from] ability: Rough Skin", "[of] p2a: Garchomp"])
    assert ab == [("p2: Garchomp", "roughskin")]                       # [of] is the holder, and only it
    ab, _ = announcements(["", "-item", "p2a: Weavile", "Leftovers", "[from] ability: Pickpocket", "[of] p1a: Espeon"])
    assert ab == [("p2: Weavile", "pickpocket")]

    table = obs_abil.set_abilities_table()
    asm = AbilityObservationAssembler(settings=parse_settings("all"))
    ab0 = asm.offsets["ability_ids"][0]
    wa = get_id(ids(asm), "pokemon.ability", "waterabsorb")
    for species, details, before in (("Quagsire", "Quagsire, L88, M", []),
                                     ("Politoed", "Politoed, L88, M", [["", "-weather", "RainDance", "[from] ability: Drizzle", "[of] p2a: Politoed"]])):
        b = make_battle()
        ours = list(b.team)[0].replace("p1: ", "p1a: ")
        ev0 = [["", "switch", f"p2a: {species}", details, "100/100"]] + before
        for e in ev0:
            b.parse_message(e)
        asm.assemble(b, events=ev0)
        ev = [["", "move", f"p2a: {species}", "Liquidation", ours],
              ["", "-heal", ours, "100/100", "[from] ability: Water Absorb", f"[of] p2a: {species}"]]
        for e in ev:
            b.parse_message(e)
        obs = asm.assemble(b, events=ev)
        slot = _opp_slot(b, species.lower())
        assert obs[ab0 + slot * 4] != wa, species
        if species == "Politoed":
            assert obs[ab0 + slot * 4] == get_id(ids(asm), "pokemon.ability", "drizzle")
        elif table.get("quagsire") == ["unaware"]:
            assert obs[ab0 + slot * 4] == get_id(ids(asm), "pokemon.ability", "unaware")


def test_an_announcement_the_random_battle_sets_rule_out_is_not_remembered():
    """A species the table knows takes only its set abilities from a line; the first announcement stands."""
    b = make_battle()
    b.parse_message(["", "switch", "p2a: Quagsire", "Quagsire, L88, M", "100/100"])
    asm = AbilityObservationAssembler(settings=parse_settings("abil_announced"))
    sets = obs_abil.set_abilities_of(b.opponent_team["p2: Quagsire"], asm.table)
    assert sets and "waterabsorb" not in sets and "damp" not in sets
    mem = asm._remember(b, [["", "-damage", "p1a: Espeon", "90/100", "[from] ability: Damp", "[of] p2a: Quagsire"]])
    assert "p2: Quagsire" not in mem["abilities"]
    mem = asm._remember(b, [["", "-damage", "p1a: Espeon", "90/100", f"[from] ability: {sets[0]}", "[of] p2a: Quagsire"]])
    assert mem["abilities"]["p2: Quagsire"] == sets[0]


def test_policynet_refuses_a_checkpoint_with_observation_settings_it_does_not_know(tmp_path):
    """The stage2 review's finding 2: code without an input a checkpoint was trained with plays it silently
    with the old observation. hybrid.PolicyNet refuses instead (before any weights are loaded)."""
    import torch
    import hybrid
    assert hybrid.unknown_obs_settings({"obs_settings": checkpoint_obs_settings(True, "all")}) == []
    assert hybrid.unknown_obs_settings({"model": {}}) == [] and hybrid.unknown_obs_settings(None) == []
    assert hybrid.unknown_obs_settings({"obs_settings": {"abil_sets": True, "illusion_fix": True, "off_one": False}}) == ["illusion_fix"]
    path = tmp_path / "future.pt"
    torch.save({"model": {"w": torch.zeros(1)}, "obs_settings": {"own_tera_type": True, "illusion_fix": True}}, path)
    with pytest.raises(RuntimeError, match="illusion_fix"):
        hybrid.PolicyNet(hybrid.HybridConfig(checkpoint=str(path), device="cpu"))
