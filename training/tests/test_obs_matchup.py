"""
The matchup and set-candidate block (obs_matchup.py), the model's support for it (ppo_core), and
the SetHead.

    python -m pytest tests/test_obs_matchup.py -q -p no:cacheprovider -s

Without the setting nothing may change:
  * the observation: replayed recorded battles through the worker's path, the working tree's
    ObservationAssembler(own_tera_type=True) against the reference code's (the commit before
    the block), byte for byte at every decision; the assembler with the block writes the same
    bytes in front of it;
  * the network: the working tree's PokeTransformer against the reference code's, same weights,
    same recorded observations: encode_features and forward bit for bit.
Those two need
    PSPPO_STREAMS       recorded player streams (bench/micro_showdown_sim.py --record N)
    PSPPO_V2OBS_REF     a directory holding the reference commit's copies of bench/micro_ref.MODULES
                        (here, where git is, the test exports BASE_COMMIT itself)
    PSPPO_V2OBS_CKPT    optional: a checkpoint whose weights the network comparison uses
and are skipped without them. The block's contents are checked on hand-made Pokemon.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from types import SimpleNamespace

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)          # vocab.json is opened by a relative path

import numpy as np  # noqa: E402

import obs_matchup as om  # noqa: E402
from obs_assembler import ObservationAssembler  # noqa: E402

# The commit the block was added on top of: what "unchanged" is measured against.
BASE_COMMIT = "c84a4ef"


# --------------------------------------------------------------------------- helpers


def _pokemon(species, moves=(), item=None, ability=None):
    from poke_env.battle import Pokemon
    m = Pokemon(gen=9, species=species)
    for mv in moves:
        m._add_move(mv)
    if item is not None:
        m.item = item
    if ability is not None:
        m.ability = ability
    return m


def _battle(own, opp):
    return SimpleNamespace(team={f"p1: {m.species}{i}": m for i, m in enumerate(own)},
                           opponent_team={f"p2: {m.species}{i}": m for i, m in enumerate(opp)})


@pytest.fixture(scope="module")
def asm():
    return ObservationAssembler(own_tera_type=True, matchup=True)


@pytest.fixture(scope="module")
def enc(asm):
    return om.MatchupEncoder(asm.vocab_map)


def _blk(enc, battle):
    """The block by the fast path, checked against the reference path on the same battle."""
    fast = enc.block(battle)
    slow = enc.block(battle, reference=True)
    assert fast.tobytes() == slow.tobytes()
    return fast


def _parts(block):
    b = np.asarray(block, dtype=np.float32)
    return b[:om.MOVE_BLOCK].reshape(12, 4, om.MOVE_FEATS), b[om.MOVE_BLOCK:].reshape(12, om.MON_FEATS)


def _eff(mv, slot, k):
    return mv[slot, k, om.MOVE_EFF[0]:om.MOVE_EFF[1]]


def _pim(mv, slot, k):
    return mv[slot, k, om.MOVE_POSSIMM[0]:om.MOVE_POSSIMM[1]]


def _mask(mv, slot, k):
    return mv[slot, k, om.MOVE_MASK[0]:om.MOVE_MASK[1]]


def _table_prob(species, kind, name, moves=()):
    """The probability of an ability or item from the pinned table, worked out here independently: roles that hold every move, equally likely."""
    with open(om.SETS_FILE, encoding="utf-8") as f:
        roles = json.load(f)["species"][species]
    kept = [r for r in roles if set(moves) <= set(r["moves"])] or roles
    return sum((1.0 / len(kept)) * (r[kind].count(name) / len(r[kind])) for r in kept if r[kind])


def M(mid, typ, dmg=True, flags=(), ab=None, stellar=False):
    return om.MoveInfo(id=mid, type=typ, damaging=dmg, flags=frozenset(flags), attacker_ability=ab, stellar=stellar)


def D(types, ability=None, ac=(), item=None, ic=(), tera=False):
    return om.DefenderInfo(types=tuple(types), terastallized=tera, ability=ability, ability_cands=tuple(ac), item=item, item_cands=tuple(ic))


# --------------------------------------------------------------------------- layout


def test_layout_appends_the_block_and_moves_nothing():
    a0 = ObservationAssembler(own_tera_type=True)
    a1 = ObservationAssembler(own_tera_type=True, matchup=True)
    assert [n for n, _s, _z in a0.layout] == ["pokemon_body", "pokemon_ids", "ability_ids", "move_ids", "move_scalars",
                                               "global_scalars", "transition_ids", "transition_scalars", "action_mask"]
    assert "matchup" not in a0.offsets and "matchup" not in a0.meta and a0.matchup is False
    for k, v in a0.offsets.items():
        assert a1.offsets[k] == v
    assert a1.offsets["matchup"] == (a0.total_dim, a0.total_dim + om.BLOCK_DIM)
    assert a1.total_dim == a0.total_dim + om.BLOCK_DIM == a0.total_dim + 1068
    assert om.MOVE_BLOCK == 864 and om.MON_BLOCK == 204
    assert a1.meta["matchup"] == om.block_meta()
    # every id the block can write is exact in float16 and inside the embedding tables
    assert len(a1.vocab_lists["pokemon.item"]) + 1 < 2048 and len(a1.vocab_lists["pokemon.ability"]) + 1 < 2048


def test_set_table_is_the_pinned_copy():
    t = om.SetTable()
    assert t.md5 == "e6385a98c69048be4f57e5207d4bf350" and len(t.species) == 509
    assert t.roles_for("palafinhero") == t.roles_for("palafin") is not None       # battle-only form
    assert t.roles_for("mimikyubusted") == t.roles_for("mimikyu") is not None
    assert t.roles_for("ogerponwellspringtera") == t.roles_for("ogerponwellspring") is not None
    assert t.roles_for("notapokemon") is None


# --------------------------------------------------------------------------- the matchup function


def test_effectiveness_rules():
    eff, pim = om.effectiveness, om.possible_immunity
    assert eff(M("icebeam", "ICE"), D(["DRAGON", "GROUND"])) == 4.0
    assert eff(M("earthquake", "GROUND"), D(["FLYING"])) == 0.0
    assert eff(M("thousandarrows", "GROUND"), D(["FLYING"])) == 1.0                       # grounds it
    assert eff(M("thousandarrows", "GROUND"), D(["ELECTRIC", "WATER"], ability="levitate")) == 2.0
    assert eff(M("earthquake", "GROUND"), D(["ELECTRIC", "WATER"], ability="levitate")) == 0.0
    assert eff(M("freezedry", "ICE"), D(["WATER", "GROUND"])) == 4.0
    assert eff(M("icebeam", "ICE"), D(["WATER", "GROUND"])) == 1.0
    assert eff(M("flyingpress", "FIGHTING"), D(["NORMAL"])) == 2.0
    assert eff(M("flyingpress", "FIGHTING"), D(["FIGHTING"])) == 2.0
    assert eff(M("flyingpress", "FIGHTING"), D(["GHOST"])) == 0.0
    assert eff(M("fakeout", "NORMAL"), D(["GHOST"])) == 0.0
    assert eff(M("fakeout", "NORMAL", ab="scrappy"), D(["GHOST"])) == 1.0
    assert eff(M("closecombat", "FIGHTING", ab="mindseye"), D(["GHOST", "NORMAL"])) == 2.0
    assert eff(M("uturn", "BUG", ab="tintedlens"), D(["FIRE"])) == 1.0
    assert eff(M("uturn", "BUG"), D(["FIRE"])) == 0.5
    assert eff(M("flamethrower", "FIRE"), D(["NORMAL"], ability="thickfat")) == 0.5
    assert eff(M("icebeam", "ICE"), D(["GRASS"], ability="thickfat")) == 1.0
    assert eff(M("flamethrower", "FIRE"), D(["STEEL"], ability="heatproof")) == 1.0
    assert eff(M("shadowball", "GHOST"), D(["ROCK"], ability="purifyingsalt")) == 0.5
    assert eff(M("surf", "WATER"), D(["BUG", "GHOST"], ability="wonderguard")) == 0.0
    assert eff(M("flamethrower", "FIRE"), D(["BUG", "GHOST"], ability="wonderguard")) == 2.0
    assert eff(M("shadowball", "GHOST", flags=("bullet",)), D(["PSYCHIC"], ability="bulletproof")) == 0.0
    assert eff(M("hypervoice", "NORMAL", flags=("sound",)), D(["NORMAL"], ability="soundproof")) == 0.0
    assert eff(M("hurricane", "FLYING", flags=("wind",)), D(["GRASS"], ability="windrider")) == 0.0
    for ab, t in (("waterabsorb", "WATER"), ("stormdrain", "WATER"), ("dryskin", "WATER"), ("voltabsorb", "ELECTRIC"),
                  ("lightningrod", "ELECTRIC"), ("motordrive", "ELECTRIC"), ("sapsipper", "GRASS"), ("eartheater", "GROUND"),
                  ("flashfire", "FIRE"), ("wellbakedbody", "FIRE"), ("levitate", "GROUND")):
        assert eff(M("x", t), D(["NORMAL"], ability=ab)) == 0.0, ab
        assert eff(M("x", t), D(["NORMAL"])) == 1.0
    assert eff(M("earthquake", "GROUND"), D(["STEEL"], item="airballoon")) == 0.0
    assert eff(M("terablast", "STELLAR", stellar=True), D(["NORMAL"], tera=True)) == 2.0
    assert eff(M("terablast", "STELLAR", stellar=True), D(["GHOST"], ability="levitate")) == 1.0
    assert eff(M("seismictoss", "FIGHTING"), D(["GHOST"])) == 0.0
    assert eff(M("seismictoss", "FIGHTING"), D(["NORMAL"])) == 1.0                      # fixed damage: no 2x
    assert eff(M("willowisp", "FIRE", dmg=False), D(["FIRE"])) == 1.0 and pim(M("willowisp", "FIRE", dmg=False), D(["FIRE"], ac=(("flashfire", 1.0),))) == 0.0
    # possible immunity: unrevealed candidates only, ability and item combined
    assert pim(M("earthquake", "GROUND"), D(["STEEL"], ac=(("levitate", 0.5), ("x", 0.5)), ic=(("airballoon", 0.4), ("y", 0.6)))) == pytest.approx(0.7)
    assert pim(M("earthquake", "GROUND"), D(["STEEL"], ability="levitate", ac=(("levitate", 0.5),))) == 0.0     # already immune
    assert pim(M("earthquake", "GROUND"), D(["FLYING"], ic=(("airballoon", 1.0),))) == 0.0                     # the type already is
    assert pim(M("surf", "WATER"), D(["GROUND"], ac=(("waterabsorb", 0.25), ("unaware", 0.75)))) == 0.25
    assert pim(M("thousandarrows", "GROUND"), D(["STEEL"], ac=(("levitate", 1.0),), ic=(("airballoon", 1.0),))) == 0.0
    assert [om.log2_code(x) for x in (0.0, 4.0, 0.25, 16.0, 1 / 16, 1.0)] == [-4.0, 2.0, -2.0, 3.0, -3.0, 0.0]


# --------------------------------------------------------------------------- the block on hand-made battles


def test_levitate_air_balloon_and_absorb_abilities(enc):
    own = [_pokemon("garchomp", ["earthquake", "firefang", "stoneedge", "swordsdance"], item="loadeddice", ability="roughskin"),
           _pokemon("greninja", ["surf", "flamethrower", "icebeam", "willowisp"], item="lifeorb", ability="protean"),
           _pokemon("magnezone", ["thunderbolt"], item="airballoon", ability="magnetpull")]
    rotom = _pokemon("rotomwash", ["hydropump"])                 # poke-env knows Levitate: its only Pokedex ability
    heatran = _pokemon("heatran", ["earthquake", "magmastorm"])  # sets: Flash Fire only; item Air Balloon or Assault Vest
    clod = _pokemon("clodsire", ["earthquake"])                  # sets: Unaware or Water Absorb
    b = _battle(own, [rotom, heatran, clod])
    assert rotom.ability == "levitate" and heatran.ability is None and clod.ability is None
    mv, mon = _parts(_blk(enc, b))

    # our Earthquake (slot 0, move 0): Levitate known -> immune; Heatran 4x with a 50% Air Balloon; Clodsire 2x
    assert _mask(mv, 0, 0).tolist() == [1, 1, 1, 0, 0, 0]
    assert _eff(mv, 0, 0).tolist() == [om.IMMUNE_LOG2, 2.0, 1.0, 0, 0, 0]
    p_balloon = _table_prob("heatran", "items", "airballoon")
    assert p_balloon == 0.5
    assert _pim(mv, 0, 0).tolist() == pytest.approx([0, p_balloon, 0, 0, 0, 0], abs=1e-3)
    # our Surf (slot 1, move 0) on Clodsire: 2x, Water Absorb possible
    p_wa = _table_prob("clodsire", "abilities", "waterabsorb")
    assert _eff(mv, 1, 0)[2] == 1.0 and _pim(mv, 1, 0)[2] == pytest.approx(p_wa, abs=1e-3) and p_wa == 0.5
    # our Flamethrower on Heatran: Flash Fire is the only ability its sets allow -> immune, though not revealed
    assert _eff(mv, 1, 1)[1] == om.IMMUNE_LOG2 and _pim(mv, 1, 1)[1] == 0
    # Will-O-Wisp: a status move -> no numbers, the mask still there
    assert _eff(mv, 1, 3).tolist() == [0] * 6 and _pim(mv, 1, 3).tolist() == [0] * 6 and _mask(mv, 1, 3).tolist() == [1, 1, 1, 0, 0, 0]
    # their Earthquake (Heatran = opponent slot 1 = token 7, move 0) against our Air Balloon Magnezone (our slot 2)
    assert _mask(mv, 7, 0).tolist() == [1, 1, 1, 0, 0, 0]
    assert _eff(mv, 7, 0).tolist() == [0.0, 0.0, om.IMMUNE_LOG2, 0, 0, 0]       # Garchomp and Greninja neutral
    # unseen opponents and empty move slots stay zero
    assert not mv[9:].any() and not mv[2, 1:].any() and not mon[9:].any()

    # Heatran shows its Air Balloon: Earthquake is now immune for certain
    heatran.item = "airballoon"
    mv, mon = _parts(_blk(enc, b))
    assert _eff(mv, 0, 0)[1] == om.IMMUNE_LOG2 and _pim(mv, 0, 0)[1] == 0
    assert mon[7, om.MON_ITEM_UNKNOWN] == 0 and mon[7, om.MON_ITEM_GONE] == 0
    # the balloon pops: 4x again, no doubt left, the item is gone
    heatran.end_item("airballoon")
    mv, mon = _parts(_blk(enc, b))
    assert _eff(mv, 0, 0)[1] == 2.0 and _pim(mv, 0, 0)[1] == 0
    assert mon[7, om.MON_ITEM_GONE] == 1 and mon[7, om.MON_ITEM_UNKNOWN] == 0 and not mon[7, om.MON_ITEM_IDS[0]:om.MON_ITEM_P[1]].any()

    # Clodsire reveals Water Absorb, or Unaware
    clod.ability = "waterabsorb"
    mv, _ = _parts(_blk(enc, b))
    assert _eff(mv, 1, 0)[2] == om.IMMUNE_LOG2 and _pim(mv, 1, 0)[2] == 0
    clod2 = _pokemon("clodsire", ["earthquake"], ability="unaware")
    mv, _ = _parts(_blk(enc, _battle(own, [rotom, heatran, clod2])))
    assert _eff(mv, 1, 0)[2] == 1.0 and _pim(mv, 1, 0)[2] == 0


def test_role_filter_narrows_flash_fire_on_chandelure(enc):
    own = [_pokemon("greninja", ["flamethrower"], item="lifeorb", ability="protean")]
    chandy = _pokemon("chandelure", ["shadowball"])
    mv, mon = _parts(_blk(enc, _battle(own, [chandy])))
    p_ff = _table_prob("chandelure", "abilities", "flashfire")
    assert p_ff == 0.75
    assert _eff(mv, 0, 0)[0] == -1.0 and _pim(mv, 0, 0)[0] == pytest.approx(p_ff, abs=1e-3)
    chandy._add_move("trick")                     # only the Choice Scarf role has Trick: Flash Fire for certain
    mv, mon = _parts(_blk(enc, _battle(own, [chandy])))
    assert _eff(mv, 0, 0)[0] == om.IMMUNE_LOG2 and _pim(mv, 0, 0)[0] == 0
    assert mon[6, om.MON_CHOICE_POSSIBLE] == 1 and mon[6, om.MON_ITEM_P[0]] == 1.0


def test_choice_scarf_candidate_species(enc, asm):
    vm = asm.vocab_map
    iid = lambda n: vm["pokemon.item"][n] + 1  # noqa: E731
    own = [_pokemon("garchomp", ["earthquake"], item="loadeddice", ability="roughskin")]
    rotom = _pokemon("rotomwash", ["hydropump", "painsplit"])        # both roles have these
    _mv, mon = _parts(_blk(enc, _battle(own, [rotom])))
    row = mon[6]
    p_scarf = _table_prob("rotomwash", "items", "choicescarf", ["hydropump", "painsplit"])
    assert p_scarf == 0.25
    assert row[om.MON_ITEM_IDS[0]:om.MON_ITEM_IDS[1]].tolist() == [iid("leftovers"), iid("choicescarf"), 0, 0]
    assert row[om.MON_ITEM_P[0]:om.MON_ITEM_P[1]].tolist() == pytest.approx([0.75, p_scarf, 0, 0], abs=1e-3)
    assert (row[om.MON_ITEM_UNKNOWN], row[om.MON_CHOICE_POSSIBLE], row[om.MON_ITEM_GONE]) == (1, 1, 0)
    assert row[om.MON_ABIL_IDS[0]] == vm["pokemon.ability"]["levitate"] + 1 and row[om.MON_ABIL_P[0]] == 1.0
    # Nasty Plot is only in the Choice Scarf role: an even split, names in order on the tie
    rotom._add_move("nastyplot")
    _mv, mon = _parts(_blk(enc, _battle(own, [rotom])))
    assert mon[6, om.MON_ITEM_IDS[0]:om.MON_ITEM_IDS[0] + 2].tolist() == [iid("choicescarf"), iid("leftovers")]
    assert mon[6, om.MON_ITEM_P[0]:om.MON_ITEM_P[0] + 2].tolist() == [0.5, 0.5]
    # the item is revealed: no doubt, no Choice item
    rotom.item = "leftovers"
    _mv, mon = _parts(_blk(enc, _battle(own, [rotom])))
    assert mon[6, om.MON_ITEM_IDS[0]:om.MON_ITEM_P[1]].tolist() == [iid("leftovers"), 0, 0, 0, 1, 0, 0, 0]
    assert (mon[6, om.MON_ITEM_UNKNOWN], mon[6, om.MON_CHOICE_POSSIBLE], mon[6, om.MON_ITEM_GONE]) == (0, 0, 0)
    # a species whose sets hold no Choice item
    _mv, mon = _parts(_blk(enc, _battle(own, [_pokemon("heatran")])))
    assert (mon[6, om.MON_ITEM_UNKNOWN], mon[6, om.MON_CHOICE_POSSIBLE]) == (1, 0)


def test_own_side_is_exact(enc, asm):
    vm = asm.vocab_map
    gyara = _pokemon("gyarados", ["waterfall"], item="choiceband", ability="intimidate")     # not its set item: exact anyway
    bare = _pokemon("garchomp", ["earthquake"], item="", ability="roughskin")              # an item lost (Knock Off)
    clod = _pokemon("clodsire", ["earthquake"])
    _mv, mon = _parts(_blk(enc, _battle([gyara, bare], [clod])))
    g = mon[0]
    assert g[om.MON_ABIL_IDS[0]:om.MON_ABIL_P[1]].tolist() == [vm["pokemon.ability"]["intimidate"] + 1, 0, 0, 1, 0, 0]
    assert g[om.MON_ITEM_IDS[0]:om.MON_ITEM_P[1]].tolist() == [vm["pokemon.item"]["choiceband"] + 1, 0, 0, 0, 1, 0, 0, 0]
    assert (g[om.MON_ITEM_UNKNOWN], g[om.MON_CHOICE_POSSIBLE], g[om.MON_ITEM_GONE]) == (0, 1, 0)
    assert (mon[1, om.MON_ITEM_UNKNOWN], mon[1, om.MON_CHOICE_POSSIBLE], mon[1, om.MON_ITEM_GONE]) == (0, 0, 1)
    assert not mon[1, om.MON_ITEM_IDS[0]:om.MON_ITEM_P[1]].any()
    # the opponent's unrevealed ability: both set abilities, half each, most likely first then by name
    c = mon[6]
    assert sorted(c[om.MON_ABIL_IDS[0]:om.MON_ABIL_IDS[0] + 2].tolist()) == sorted(
        [vm["pokemon.ability"]["unaware"] + 1, vm["pokemon.ability"]["waterabsorb"] + 1])
    assert c[om.MON_ABIL_P[0]:om.MON_ABIL_P[1]].tolist() == [0.5, 0.5, 0]
    assert c[om.MON_ITEM_P[0]] == 1.0 and c[om.MON_ITEM_UNKNOWN] == 1


def test_stellar_tera_keeps_original_types_for_defence(enc):
    own = [_pokemon("garchomp", ["earthquake", "terablast"], item="loadeddice", ability="roughskin")]
    heatran = _pokemon("heatran")
    heatran.item = "assaultvest"
    heatran.terastallize("stellar")
    mv, _ = _parts(_blk(enc, _battle(own, [heatran])))
    assert _eff(mv, 0, 0)[0] == 2.0                      # Fire/Steel: 4x, not the neutral poke-env's STELLAR type gives
    own[0].terastallize("stellar")
    mv, _ = _parts(_blk(enc, _battle(own, [heatran])))
    assert _eff(mv, 0, 1)[0] == 1.0                      # a Stellar Tera Blast on a Terastallized Pokemon: 2x


# --------------------------------------------------------------------------- SetHead


def test_set_labels_from_a_team(asm):
    vm = asm.vocab_map
    clod, rotom, chomp = _pokemon("clodsire"), _pokemon("rotomwash"), _pokemon("garchomp")
    chomp.item = "rockyhelmet"                                   # already seen: no item label
    b = _battle([], [clod, rotom, chomp])
    keys = list(b.opponent_team)
    truth = {keys[0]: {"species": "clodsire", "ability": "Water Absorb", "item": "Leftovers"},
             keys[1]: _pokemon("rotomwash", item="choicescarf", ability="levitate"),
             "p2: somethingelse": {"species": "garchomp", "ability": "roughskin", "item": "rockyhelmet"}}
    lab = om.build_set_labels(b, truth, vocab_map=vm)
    sp = lambda n: vm["pokemon.species"][n] + 1  # noqa: E731
    assert lab.shape == (6, 3) and lab.dtype == np.int32
    assert lab[0].tolist() == [sp("clodsire"), vm["pokemon.ability"]["waterabsorb"] + 1, vm["pokemon.item"]["leftovers"] + 1]
    assert lab[1].tolist() == [sp("rotomwash"), 0, vm["pokemon.item"]["choicescarf"] + 1]   # Levitate is known already
    assert lab[2].tolist() == [sp("garchomp"), vm["pokemon.ability"]["roughskin"] + 1, 0]   # matched by species
    assert not lab[3:].any()


def test_set_head_loss_is_restricted_to_the_sets():
    import torch
    from ppo_core import SetHead, compute_set_head_loss
    ab, it = om.set_candidate_tables()
    vocab = om.load_vocab_lists()
    vm = {k: {x.lower(): i for i, x in enumerate(v)} for k, v in vocab.items()}
    clod = vm["pokemon.species"]["clodsire"] + 1
    wa, un = vm["pokemon.ability"]["waterabsorb"] + 1, vm["pokemon.ability"]["unaware"] + 1
    assert ab[clod, wa] and ab[clod, un] and ab[clod].sum() == 2 and not ab[:, 0].any() and not it[:, 0].any()
    torch.manual_seed(0)
    head = SetHead(32, ab, it)
    net = SimpleNamespace(set_head=head, set_head_coef=0.0)
    z = torch.randn(2, 64, 32, requires_grad=True)
    lab = torch.zeros(2, 6, 3, dtype=torch.long)
    lab[0, 0] = torch.tensor([clod, wa, 0])
    term, st = compute_set_head_loss(net, z, lab)
    assert float(term) == 0.0 and int(st["set_ability_n"]) == 1 and int(st["set_item_n"]) == 0   # coefficient 0 by default
    term, st = compute_set_head_loss(net, z, lab, coef=1.0)
    lg = head.ability(z[0, 7].detach())
    want = -torch.log_softmax(lg[[wa, un]], dim=0)[0]
    assert float(term) == pytest.approx(float(want), rel=1e-5)
    term.backward()
    assert z.grad[0, 7].abs().sum() > 0 and z.grad[:, :7].abs().sum() == 0 and z.grad[1].abs().sum() == 0
    # an ability outside the species' sets changes nothing
    with torch.no_grad():
        head.ability.bias[vm["pokemon.ability"]["levitate"] + 1] += 100.0
    term2, _ = compute_set_head_loss(net, z, lab, coef=1.0)
    assert float(term2) == pytest.approx(float(term), rel=1e-6)
    # no labels at all: zero, no NaN
    term3, st3 = compute_set_head_loss(net, z, torch.zeros(2, 6, 3, dtype=torch.long), coef=1.0)
    assert float(term3) == 0.0 and int(st3["set_ability_n"]) == 0


def test_attach_set_head_on_the_network():
    import obs_matchup
    from config import RunConfig
    net = obs_matchup.build_model(RunConfig.default(), matchup=True, set_head=True, set_head_coef=0.1)
    names = [n for n, _p in net.named_parameters()]
    assert names[-4:] == ["set_head.ability.weight", "set_head.ability.bias", "set_head.item.weight", "set_head.item.bias"]
    zones = net.get_parameter_zones()
    assert {n for n, _p in zones["aux"]} == set(names[-4:])          # the auxiliary heads' zone: the learner's last groups
    sd = net.state_dict()
    assert "set_head.ability.weight" in sd and "set_head.ability_allowed" not in sd and net.set_head_coef == 0.1
    plain = RunConfig.default().make_model()
    assert plain.set_head is None and not any(k.startswith("set_head") for k in plain.state_dict())


def test_model_shapes_with_the_block():
    import obs_matchup
    from config import RunConfig
    plain = RunConfig.default().make_model()
    wide = obs_matchup.build_model(RunConfig.default(), matchup=True)
    assert plain.matchup is False and wide.matchup is True
    k_m, k_p = plain.move_net[0].weight.shape[1], plain.pokemon_net[0].weight.shape[1]
    assert wide.move_net[0].weight.shape[1] == k_m + 18 and wide.pokemon_net[0].weight.shape[1] == k_p + 64 + 32 + 3
    assert [n for n, _ in plain.named_parameters()] == [n for n, _ in wide.named_parameters()]


# --------------------------------------------------------------------------- recorded battles: nothing changes without the setting


@pytest.fixture(scope="module")
def ref():
    from bench import micro_ref
    given = os.environ.get("PSPPO_V2OBS_REF")
    try:
        if given:
            return micro_ref.Reference(given, source="PSPPO_V2OBS_REF")
        return micro_ref.Reference.export(commit=BASE_COMMIT, dest=os.path.join(tempfile.gettempdir(), f"psppo_ref_{BASE_COMMIT}"))
    except micro_ref.ReferenceUnavailable as e:
        pytest.skip(f"no reference code: {e}")


@pytest.fixture(scope="module")
def replayed(ref):
    import matchup_replay
    if not matchup_replay.stream_paths():
        pytest.skip("no recorded battles: set PSPPO_STREAMS")
    with ref.active():
        old_asm = ref.load("obs_assembler").ObservationAssembler(own_tera_type=True)
    mu_asm = ObservationAssembler(own_tera_type=True, matchup=True)
    slow = om.MatchupEncoder(mu_asm.vocab_map)
    reference_block = SimpleNamespace(assemble=lambda battle, events=None: slow.block(battle, reference=True))
    out = matchup_replay.replay({"ref": old_asm, "matchup": mu_asm, "block_reference": reference_block})
    out["dim0"] = old_asm.total_dim
    return out


def test_observation_without_the_setting_is_byte_for_byte_the_old_one(replayed):
    recs, dim0 = replayed["records"], replayed["dim0"]
    assert len(recs) >= 2000, f"only {len(recs)} decisions"
    bad, stats = [], {"immune": 0, "possible immunity": 0, "choice possible": 0, "item unknown": 0, "item gone": 0, "unseen": 0}
    for r in recs:
        d, o, mu = r["default"], r["ref"], r["matchup"]
        assert not isinstance(o, Exception) and not isinstance(mu, Exception), (o, mu)
        if d.dtype != o.dtype or d.shape != o.shape or d.tobytes() != o.tobytes():
            bad.append(("default != reference", r["key"], r["turn"]))
        if mu.shape != (dim0 + om.BLOCK_DIM,) or mu[:dim0].tobytes() != d.tobytes():
            bad.append(("prefix of the block observation != default", r["key"], r["turn"]))
        if mu[dim0:].tobytes() != r["block_reference"].tobytes():
            bad.append(("fast block != reference block", r["key"], r["turn"]))
        mv, mon = _parts(mu[dim0:])
        assert np.isfinite(mv).all() and np.isfinite(mon).all()
        e = mv[..., om.MOVE_EFF[0]:om.MOVE_EFF[1]]
        # every multiplier is a power of 2, or one times Dry Skin's 1.25 from Fire (matchup.py's damage multiplier)
        dry = {float(np.float16(min(3.0, max(-3.0, k + float(np.log2(1.25)))))) for k in range(-3, 3)}
        assert set(np.unique(e).tolist()) <= {-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0} | dry
        p = mv[..., om.MOVE_POSSIMM[0]:om.MOVE_POSSIMM[1]]
        assert (p >= 0).all() and (p <= 1.0005).all()
        for lo, hi in ((om.MON_ABIL_P[0], om.MON_ABIL_P[1]), (om.MON_ITEM_P[0], om.MON_ITEM_P[1])):
            assert (mon[:, lo:hi] >= 0).all() and (mon[:, lo:hi].sum(axis=1) <= 1.002).all()
        stats["immune"] += int((e == om.IMMUNE_LOG2).any())
        stats["possible immunity"] += int((p > 0).any())
        stats["choice possible"] += int(mon[6:, om.MON_CHOICE_POSSIBLE].any())
        stats["item unknown"] += int(mon[6:, om.MON_ITEM_UNKNOWN].any())
        stats["item gone"] += int(mon[:, om.MON_ITEM_GONE].any())
        stats["unseen"] += int((mv[:6, :, om.MOVE_MASK[0]:om.MOVE_MASK[1]].max(axis=(0, 1)) == 0).any())
        # our own side: no candidate is a guess
        assert (mon[:6, om.MON_ITEM_UNKNOWN] == 0).all()
    t = replayed["time"]
    print("\n" + json.dumps({"decisions": len(recs), **stats,
                             "assemble ms per decision": {k: round(1000 * v / len(recs), 4) for k, v in t.items()}}, indent=2))
    assert not bad, bad[:5]
    for k in ("immune", "possible immunity", "choice possible", "item unknown", "item gone", "unseen"):
        assert stats[k] > 0, k


def test_network_without_the_setting_is_bit_for_bit_the_old_one(ref, replayed):
    import torch
    import matchup_replay
    from config import RunConfig
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with ref.active():
        old_cfg = ref.load("config")
        old_net = old_cfg.RunConfig.default().make_model()
    new_net = RunConfig.default().make_model()
    assert type(old_net).__module__ == "ppo_core" and type(old_net) is not type(new_net)
    path = os.environ.get("PSPPO_V2OBS_CKPT")
    if path:
        state = {k: v for k, v in torch.load(path, map_location="cpu", weights_only=False)["model"].items() if "attn_mask" not in k}
    else:
        state = new_net.state_dict()
    old_net.load_state_dict(state, strict=True)
    new_net.load_state_dict(state, strict=True)
    seqs = matchup_replay.sequences(replayed["records"], "default", 1000)
    x = torch.from_numpy(np.concatenate(seqs).astype(np.float32))
    for bf16 in (False, True):
        a, b = old_net.to(dev).eval(), new_net.to(dev).eval()
        if bf16:
            a.enable_bf16_recurrent_path(), b.enable_bf16_recurrent_path()
        with torch.no_grad():
            for i in range(0, x.shape[0], 256):
                xb = x[i:i + 256].to(dev)
                fa, fb = a.encode_features(xb), b.encode_features(xb)
                for ta, tb in zip(fa, fb):
                    assert torch.equal(ta, tb)
                oa, ob = a(xb)[:3], b(xb)[:3]
                for ta, tb in zip(oa, ob):
                    assert torch.equal(ta, tb)
    print(f"\n{x.shape[0]} recorded observations, fp32 and bf16, on {dev}: encode_features and forward bit for bit equal"
          + (f" (weights of {path})" if path else " (random weights)"))
