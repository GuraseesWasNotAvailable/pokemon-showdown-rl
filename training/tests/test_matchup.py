"""
Tests for matchup.py: hand-made cases for every rule, and an exact agreement check against Showdown itself
on 20,000 random cases (tests/showdown_matchup_ref.js builds each case as a real battle in the engine).

    pytest tests/test_matchup.py -s          (the agreement test needs node and ~/pokemon-showdown, or PS_DIR)
"""

import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matchup import (Field, Mon, TYPES, dex, effectiveness, load_randbats_sets, matchup, move_type,  # noqa: E402
                     possible_immunity, set_abilities, set_items, species_types, to_id)


def tm(move, defender, attacker=None, fld=None):
    m = matchup(move, defender, attacker, fld)
    return "immune" if m.immune else m.type_mod


# --------------------------------------------------------------------------------------------- chart

def test_plain_chart():
    assert tm("icebeam", Mon("Garchomp")) == 2
    assert tm("icebeam", Mon("Heatran")) == -2
    assert tm("flamethrower", Mon("Ferrothorn")) == 2
    assert tm("thunderbolt", Mon("Garchomp")) == "immune"
    assert tm("shadowball", Mon("Blissey")) == "immune"
    assert tm("dracometeor", Mon("Clefable")) == "immune"
    assert tm("closecombat", Mon("Gengar")) == "immune"
    assert effectiveness("icebeam", Mon("Garchomp")) == 4.0
    assert effectiveness("surf", Mon("Kingdra")) == 0.25


def test_owner_example_water_fire_grass():
    # We have a Water type, they have a Fire type and switch to a Grass type; Ice Beam is resisted by the
    # Fire type and super effective on the Grass type.
    assert matchup("icebeam", Mon("Arcanine")).resisted
    assert matchup("icebeam", Mon("Meowscarada")).super_effective


# ------------------------------------------------------------------------------------------- immunities

@pytest.mark.parametrize("move,ability", [
    ("earthquake", "Levitate"), ("flamethrower", "Flash Fire"), ("flamethrower", "Well-Baked Body"),
    ("surf", "Water Absorb"), ("thunderbolt", "Volt Absorb"), ("surf", "Storm Drain"),
    ("thunderbolt", "Lightning Rod"), ("thunderbolt", "Motor Drive"), ("surf", "Dry Skin"),
    ("leafstorm", "Sap Sipper"), ("earthquake", "Earth Eater"), ("hurricane", "Wind Rider"),
    ("shadowball", "Bulletproof"), ("boomburst", "Soundproof"), ("bodyslam", "Wonder Guard"),
])
def test_ability_immunities(move, ability):
    # Mew: Psychic, so none of these moves is resisted or immune by type.
    m = matchup(move, Mon("Mew", ability=ability))
    assert m.immune, (move, ability)
    assert m.multiplier == 0.0
    # Mold Breaker ignores every one of them.
    assert not matchup(move, Mon("Mew", ability=ability), Mon("Excadrill", ability="Mold Breaker")).immune


def test_wonder_guard_lets_super_effective_through():
    assert not matchup("closecombat", Mon("Snorlax", ability="Wonder Guard")).immune
    assert matchup("surf", Mon("Snorlax", ability="Wonder Guard")).immune


def test_air_balloon_and_grounding():
    assert matchup("earthquake", Mon("Heatran", item="Air Balloon")).immune_by == "ground:airballoon"
    assert tm("earthquake", Mon("Heatran")) == 2
    assert tm("earthquake", Mon("Zapdos")) == "immune"
    # Iron Ball grounds a Flying type, and Ground then hits it neutrally (items.ts ironball onEffectiveness).
    assert tm("earthquake", Mon("Zapdos", item="Iron Ball")) == 0
    assert tm("earthquake", Mon("Zapdos"), fld=Field(gravity=True)) == 1
    assert tm("earthquake", Mon("Zapdos", volatiles=frozenset({"smackdown"}))) == 1
    assert tm("earthquake", Mon("Heatran", volatiles=frozenset({"magnetrise"}))) == "immune"
    # Thousand Arrows: an airborne Flying type takes neutral damage whatever its other type.
    assert tm("thousandarrows", Mon("Zapdos")) == 0
    assert tm("thousandarrows", Mon("Rotom-Wash", ability="Levitate")) == 1


def test_ring_target_and_foresight():
    assert tm("shadowball", Mon("Snorlax", item="Ring Target")) == 0
    assert tm("closecombat", Mon("Gengar", volatiles=frozenset({"foresight"}))) == -1


def test_scrappy_and_minds_eye():
    for ab in ("Scrappy", "Mind's Eye"):
        assert tm("closecombat", Mon("Gengar"), Mon("Kangaskhan", ability=ab)) == -1
        assert tm("bodyslam", Mon("Gholdengo"), Mon("Kangaskhan", ability=ab)) == -1


# ------------------------------------------------------------------------------------------ attacker side

def test_tera_blast_and_stellar():
    assert move_type("terablast", Mon("Clefable")) == "Normal"
    assert move_type("terablast", Mon("Clefable", tera="Fire")) == "Fire"
    assert tm("terablast", Mon("Scizor"), Mon("Clefable", tera="Fire")) == 2
    # Stellar Tera Blast: super effective on a Terastallized target, neutral otherwise.
    assert tm("terablast", Mon("Scizor", tera="Bug"), Mon("Clefable", tera="Stellar")) == 1
    assert tm("terablast", Mon("Scizor"), Mon("Clefable", tera="Stellar")) == 0
    assert tm("terablast", Mon("Gengar"), Mon("Clefable", tera="Stellar")) == 0
    # Pixilate does not touch a Terastallized Tera Blast.
    assert move_type("terablast", Mon("Sylveon", ability="Pixilate")) == "Fairy"
    assert move_type("terablast", Mon("Sylveon", ability="Pixilate", tera="Water")) == "Water"


def test_defender_tera():
    assert tm("thunderbolt", Mon("Gyarados")) == 2
    assert tm("thunderbolt", Mon("Gyarados", tera="Water")) == 1
    assert tm("thunderbolt", Mon("Gyarados", tera="Ground")) == "immune"
    assert tm("thunderbolt", Mon("Gyarados", tera="Stellar")) == 2


def test_freeze_dry_and_flying_press():
    assert tm("freezedry", Mon("Gyarados")) == 2
    assert tm("freezedry", Mon("Kingdra")) == 2
    assert tm("freezedry", Mon("Heatran")) == -2
    assert tm("flyingpress", Mon("Amoonguss")) == 0
    assert tm("flyingpress", Mon("Gengar")) == "immune"
    assert tm("flyingpress", Mon("Breloom")) == 2


def test_type_changing_moves_and_abilities():
    assert move_type("weatherball", Mon("Pelipper"), Field(weather="raindance")) == "Water"
    assert move_type("weatherball", Mon("Pelipper"), Field(weather="raindance"),
                     defender=Mon("Rayquaza", ability="Air Lock")) == "Normal"
    assert move_type("hypervoice", Mon("Sylveon", ability="Pixilate")) == "Fairy"
    assert move_type("hypervoice", Mon("Primarina", ability="Liquid Voice")) == "Water"
    assert move_type("ivycudgel", Mon("Ogerpon-Wellspring")) == "Water"
    assert move_type("ivycudgel", Mon("Ogerpon")) == "Grass"
    assert move_type("ragingbull", Mon("Tauros-Paldea-Blaze")) == "Fire"
    assert move_type("aurawheel", Mon("Morpeko-Hangry")) == "Dark"
    assert move_type("judgment", Mon("Arceus-Fire", item="Flame Plate")) == "Fire"
    assert move_type("revelationdance", Mon("Oricorio-Pom-Pom")) == "Electric"
    assert move_type("terastarstorm", Mon("Terapagos-Stellar")) == "Stellar"


def test_tinted_lens_and_halving():
    assert effectiveness("bugbuzz", Mon("Heatran"), Mon("Venomoth", ability="Tinted Lens")) == 0.5
    assert effectiveness("bugbuzz", Mon("Heatran")) == 0.25
    m = matchup("fireblast", Mon("Mamoswine", ability="Thick Fat"))
    assert m.super_effective and m.multiplier == 1.0
    assert effectiveness("fireblast", Mon("Bronzong", ability="Heatproof")) == 1.0
    assert effectiveness("shadowball", Mon("Garganacl", ability="Purifying Salt")) == 0.5
    assert effectiveness("fireblast", Mon("Toxicroak", ability="Dry Skin")) == 1.25


def test_disguise_ice_face_tera_shell():
    assert tm("shadowball", Mon("Mimikyu", ability="Disguise")) == 0
    assert tm("shadowball", Mon("Mimikyu-Busted", ability="Disguise")) == 1
    assert tm("closecombat", Mon("Eiscue", ability="Ice Face")) == 0
    assert tm("fireblast", Mon("Eiscue", ability="Ice Face")) == 1      # special: Ice Face does not stop it
    assert tm("closecombat", Mon("Terapagos-Terastal", ability="Tera Shell")) == -1
    assert tm("closecombat", Mon("Terapagos-Terastal", ability="Tera Shell", full_hp=False)) == 1


def test_fixed_damage():
    m = matchup("seismictoss", Mon("Alakazam"))
    assert m.fixed_damage and m.type_mod == 0 and m.multiplier == 1.0
    assert matchup("seismictoss", Mon("Gengar")).immune
    assert matchup("nightshade", Mon("Snorlax")).immune


def test_possible_immunity():
    sets = load_randbats_sets()
    if not sets:
        pytest.skip("heuristic/cache/gen9randombattle.json missing")
    # A species whose sets carry Air Balloon and which is grounded otherwise.
    balloon = next(s for s in sorted(sets) if "Air Balloon" in set_items(s, sets)
                   and "Flying" not in species_types(s) and "Levitate" not in set_abilities(s, sets))
    assert possible_immunity("earthquake", balloon, None, None, sets)
    assert not possible_immunity("earthquake", balloon, None, "Leftovers", sets)
    assert not possible_immunity("earthquake", balloon, None, "", sets)
    # A species with Levitate among two or more abilities.
    lev = next((s for s in sorted(sets) if "Levitate" in set_abilities(s, sets)
                and len(set_abilities(s, sets)) > 1 and "Flying" not in species_types(s)), None)
    if lev:
        assert possible_immunity("earthquake", lev, None, "", sets)
        other = next(a for a in set_abilities(lev, sets) if a != "Levitate")
        assert not possible_immunity("earthquake", lev, other, "", sets)
    # Already immune by type: not a "possible" immunity.
    assert not possible_immunity("earthquake", "Zapdos", None, None, sets)


# ------------------------------------------------------------------------- agreement with the engine

PS_DIR = os.environ.get("PS_DIR", os.path.expanduser("~/pokemon-showdown"))
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "showdown_matchup_ref.js")

DEF_SPECIAL_ABILITIES = ["Levitate", "Flash Fire", "Well-Baked Body", "Water Absorb", "Volt Absorb", "Storm Drain",
                         "Lightning Rod", "Motor Drive", "Dry Skin", "Sap Sipper", "Earth Eater", "Wind Rider",
                         "Bulletproof", "Soundproof", "Wonder Guard", "Thick Fat", "Heatproof", "Purifying Salt",
                         "Good as Gold", "Disguise", "Ice Face", "Tera Shell", "Air Lock", "Cloud Nine", "Trace"]
ATK_SPECIAL_ABILITIES = ["Scrappy", "Mind's Eye", "Tinted Lens", "Mold Breaker", "Teravolt", "Turboblaze",
                         "Pixilate", "Aerilate", "Refrigerate", "Galvanize", "Normalize", "Liquid Voice",
                         "Air Lock", "Cloud Nine"]
SPECIAL_SPECIES = ["Mimikyu", "Eiscue", "Terapagos", "Ogerpon", "Ogerpon-Wellspring", "Ogerpon-Hearthflame",
                   "Ogerpon-Cornerstone", "Arceus-Fire", "Arceus-Ground", "Arceus-Fairy", "Tauros-Paldea-Blaze",
                   "Tauros-Paldea-Aqua", "Tauros-Paldea-Combat", "Morpeko", "Oricorio", "Oricorio-Pom-Pom",
                   "Rotom-Wash", "Zapdos", "Gengar", "Heatran", "Gyarados", "Skarmory", "Bronzong"]
SPECIAL_MOVES = ["terablast", "freezedry", "flyingpress", "thousandarrows", "weatherball", "terrainpulse",
                 "judgment", "revelationdance", "ivycudgel", "ragingbull", "aurawheel", "terastarstorm",
                 "struggle", "seismictoss", "nightshade", "earthquake", "hurricane", "shadowball", "boomburst",
                 "hypervoice", "closecombat", "bodyslam", "thunderbolt", "surf", "flamethrower", "leafstorm"]
PLATES = ["Flame Plate", "Splash Plate", "Zap Plate", "Meadow Plate", "Earth Plate", "Pixie Plate", "Spooky Plate"]
VOLATILES = ["smackdown", "magnetrise", "ingrain", "telekinesis", "tarshot", "foresight", "miracleeye"]
OGERPON_TERA = {"Ogerpon": "Grass", "Ogerpon-Wellspring": "Water", "Ogerpon-Hearthflame": "Fire",
                "Ogerpon-Cornerstone": "Rock"}


def _set_moves(species, sets):
    out = []
    for r in (sets.get(species, {}).get("roles") or {}).values():
        out.extend(to_id(m) for m in r.get("moves", []))
    return out


def _tera_types(species, sets):
    out = []
    for r in (sets.get(species, {}).get("roles") or {}).values():
        out.extend(r.get("teraTypes", []))
    return out


def make_cases(n, seed=20261015):
    sets = load_randbats_sets()
    rng = random.Random(seed)
    d = dex()
    species = [s for s in sorted(sets) if s != "Ditto"]
    damaging = lambda m: m in d.moves and d.moves[m]["category"] != "Status" and m not in ("futuresight",)
    pool = sorted({m for s in species for m in _set_moves(s, sets) if damaging(m)})
    cases = []
    for i in range(n):
        def pick_species():
            return rng.choice(SPECIAL_SPECIES) if rng.random() < 0.08 else rng.choice(species)

        def tera_for(sp, own_sets_first):
            if sp in OGERPON_TERA:
                return OGERPON_TERA[sp]
            if sp == "Terapagos":
                return "Stellar"
            own = _tera_types(sp, sets)
            if own and own_sets_first and rng.random() < 0.5:
                return rng.choice(own)
            return rng.choice(list(TYPES) + ["Stellar"])

        dsp = pick_species()
        r = rng.random()
        dab = rng.choice(DEF_SPECIAL_ABILITIES) if r < 0.4 or not set_abilities(dsp, sets) \
            else rng.choice(set_abilities(dsp, sets))
        r = rng.random()
        items = set_items(dsp, sets)
        ditem = (rng.choice(items) if items and r < 0.6 else "Air Balloon" if r < 0.75 else "Iron Ball" if r < 0.8
                 else "Ring Target" if r < 0.85 else "")
        dtera = tera_for(dsp, True) if rng.random() < 0.35 else None

        asp = pick_species()
        r = rng.random()
        aab = rng.choice(ATK_SPECIAL_ABILITIES) if r < 0.35 or not set_abilities(asp, sets) \
            else rng.choice(set_abilities(asp, sets))
        r = rng.random()
        aitems = set_items(asp, sets)
        aitem = (rng.choice(aitems) if aitems and r < 0.7 else rng.choice(PLATES) if r < 0.8
                 else "Utility Umbrella" if r < 0.85 else "")
        atera = tera_for(asp, False) if rng.random() < 0.3 else None

        own = [m for m in _set_moves(asp, sets) if damaging(m)]
        r = rng.random()
        move = rng.choice(own) if own and r < 0.45 else rng.choice(SPECIAL_MOVES) if r < 0.65 else rng.choice(pool)

        weather = rng.choice([None, None, None, None, "sunnyday", "raindance", "sandstorm", "snowscape"])
        terrain = rng.choice([None, None, None, None, None, "electricterrain", "grassyterrain", "mistyterrain",
                              "psychicterrain"])
        if terrain == "psychicterrain" and d.moves[move].get("priority", 0) > 0:
            terrain = None   # Psychic Terrain blocks priority moves: a move failure, not a type matter
        vols = [rng.choice(VOLATILES)] if rng.random() < 0.1 else []
        cases.append({"id": i, "move": move,
                      "atk": {"species": asp, "ability": aab, "item": aitem, "tera": atera},
                      "def": {"species": dsp, "ability": dab, "item": ditem, "tera": dtera},
                      "weather": weather, "terrain": terrain, "gravity": rng.random() < 0.05,
                      "def_volatiles": vols})
    return cases


def python_verdict(c, res):
    """matchup.py given the state the engine reports the two Pokemon and the field in."""
    a, dfd = res["atk"], res["def"]
    atk = Mon(species=a["species"], ability=a["ability"] or None, item=a["item"], tera=a["tera"],
              volatiles=frozenset(a["volatiles"]), full_hp=a["full_hp"])
    de = Mon(species=dfd["species"], ability=dfd["ability"] or None, item=dfd["item"], tera=dfd["tera"],
             volatiles=frozenset(dfd["volatiles"]), full_hp=dfd["full_hp"])
    fld = Field(weather=res["weather"], terrain=res["terrain"], gravity=res["gravity"])
    return matchup(c["move"], de, atk, fld)


def compare(cases, results):
    by_id = {r["id"]: r for r in results}
    errors, mismatches, kinds = [], [], Counter()
    for c in cases:
        r = by_id.get(c["id"])
        if r is None or "error" in r:
            errors.append((c, r))
            continue
        m = python_verdict(c, r)
        got = (m.move_type, m.immune, m.raw_type_mod if not m.immune else 0, m.fixed_damage)
        want = (r["move_type"], r["immune"], r["type_mod"], r["fixed"])
        kinds["immune" if r["immune"] else "super" if r["type_mod"] > 0 else "resisted" if r["type_mod"] < 0
              else "neutral"] += 1
        if got != want:
            mismatches.append((c, r, got, want))
    return errors, mismatches, kinds


@pytest.mark.skipif(not (shutil.which("node") and os.path.isdir(os.path.join(PS_DIR, "dist", "sim"))),
                    reason="needs node and a built Showdown in PS_DIR (~/pokemon-showdown)")
def test_agrees_with_showdown_on_20000_cases(tmp_path):
    if not load_randbats_sets():
        pytest.skip("heuristic/cache/gen9randombattle.json missing")
    n = int(os.environ.get("MATCHUP_CASES", "20000"))
    cases = make_cases(n)
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in cases) + "\n")
    out = subprocess.run(["node", REF, str(path)], capture_output=True, text=True, timeout=1800,
                         env=dict(os.environ, PS_DIR=PS_DIR))
    assert out.returncode == 0, out.stderr[-2000:]
    results = [json.loads(line) for line in out.stdout.splitlines() if line.strip()]
    errors, mismatches, kinds = compare(cases, results)
    n_ok = n - len(errors) - len(mismatches)
    print(f"\nmatchup.py vs Showdown: {n_ok}/{n - len(errors)} agree "
          f"({100.0 * n_ok / max(1, n - len(errors)):.3f}%), {len(errors)} engine errors; verdicts {dict(kinds)}")
    for c, r, got, want in mismatches[:25]:
        print("MISMATCH", json.dumps(c), "\n   engine", json.dumps(r), "\n   python", got, "want", want)
    for c, r in errors[:5]:
        print("ENGINE ERROR", json.dumps(c), r)
    assert len(errors) <= n // 1000, f"{len(errors)} cases failed in the engine"
    assert not mismatches, f"{len(mismatches)} mismatches"
