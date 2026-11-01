"""
matchup.py: how effective a move is against a Pokemon in gen 9, decided the way Showdown decides it.

    m = matchup("earthquake", Mon("Rotom-Wash", ability="Levitate"))
    m.immune, m.type_mod, m.multiplier          # True, 0, 0.0

Stage 2 (read_reward.py) uses it to ask "would this move have been super effective on the Pokemon that
left?". Stage 4 (obs_matchup.py) and the app will use it for the effectiveness inputs.

Where the answer comes from:
  - data: the type chart, every move's type, category and flags (sound, bullet, wind), its
    ignoreImmunity / ignoreAbility / fixed damage, and every species' types come from poke-env's
    GenData, which is Showdown's own data. Nothing of the chart is typed in here.
  - code: what Showdown writes as handlers is mirrored rule by rule, in the engine's order
    (sim/battle-actions.ts useMoveInner, trySpreadMoveHit, getDamage; sim/pokemon.ts runImmunity,
    isGrounded, runEffectiveness):
      1. the move's type: the move's own ModifyType (Tera Blast, Tera Starstorm, Ivy Cudgel, Raging
         Bull, Revelation Dance, Aura Wheel, Judgment, Weather Ball, Terrain Pulse, Struggle), then the
         attacker's ability (Normalize, Pixilate, Aerilate, Refrigerate, Galvanize, Liquid Voice);
      2. ModifyMove: Scrappy / Mind's Eye (Normal and Fighting hit Ghost), Mold Breaker / Teravolt /
         Turboblaze and the moves with ignoreAbility (the defender's breakable abilities are ignored);
      3. TryHit, the defender's ability: Volt Absorb, Lightning Rod, Motor Drive, Water Absorb, Storm
         Drain, Dry Skin, Flash Fire, Well-Baked Body, Sap Sipper, Earth Eater, Wind Rider (wind moves),
         Bulletproof (ball and bomb moves), Soundproof (sound moves), Good as Gold (status moves),
         Wonder Guard;
      4. type immunity: the chart, Ground against Flying / Levitate / Air Balloon / Magnet Rise /
         Telekinesis, grounded again by Gravity / Ingrain / Smack Down / Iron Ball; Ring Target,
         Foresight and Miracle Eye negate it;
      5. typeMod: the chart per type, Freeze-Dry (super effective on Water), Flying Press (Fighting plus
         Flying), Thousand Arrows (neutral on an airborne Flying type), Disguise and Ice Face (neutral
         while intact), Iron Ball, Tar Shot, Delta Stream, a Stellar move against a Terastallized
         Pokemon (super effective), Tera Shell (resisted at full HP); Tera types replace the types,
         except Stellar, which keeps them;
      6. fixed-damage moves (Seismic Toss, Night Shade, Super Fang, Ruination, Endeavor, Mirror Coat,
         OHKO moves) take no typeMod: Showdown prints neither "super effective" nor "resisted" for them.
  - the damage multiplier on top of typeMod (not part of Showdown's typeMod, so not part of "super
    effective"): Tinted Lens doubles a resisted hit; Thick Fat (Fire, Ice), Heatproof (Fire) and
    Purifying Salt (Ghost) halve it; Dry Skin takes 1.25x from Fire.

Not modelled, on purpose (move-specific failures, not type matters): Endeavor's HP rule, Sucker Punch,
Poltergeist, Psychic Terrain's priority block, Protect, accuracy, Neutralizing Gas, Klutz / Embargo /
Magic Room, Electrify / Ion Deluge. Future Sight and Doom Desire are treated as the delayed hit lands
(immunity checked), not as the engine's use-turn shortcut (ignoreImmunity).

tests/test_matchup.py checks every result against the engine on 20,000 random cases
(tests/showdown_matchup_ref.js).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, replace
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple, Union

TYPES = ("Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison", "Ground",
         "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy")
STELLAR = "Stellar"
TYPELESS = "???"

_ID_RE = re.compile(r"[^a-z0-9]")


def to_id(x) -> str:
    """Showdown's toID: lower case, letters and digits only. Accepts objects with an .id (poke-env Move)."""
    if x is None:
        return ""
    if not isinstance(x, str):
        x = getattr(x, "id", None) or getattr(x, "name", None) or str(x)
    return _ID_RE.sub("", x.lower())


_TYPE_BY_ID = {t.lower(): t for t in TYPES}
_TYPE_BY_ID["stellar"] = STELLAR


def type_name(x) -> Optional[str]:
    """'FIRE', 'fire', PokemonType.FIRE -> 'Fire'; 'stellar' -> 'Stellar'; '???' stays; None/'' -> None."""
    if x is None:
        return None
    if not isinstance(x, str):
        x = getattr(x, "name", None) or str(x)
    if x == TYPELESS:
        return TYPELESS
    s = to_id(x.split(".")[-1])
    if not s:
        return None
    return _TYPE_BY_ID.get(s, x)


# ---------------------------------------------------------------------------------------------------
# Data: poke-env's copy of Showdown's data, loaded once.
# ---------------------------------------------------------------------------------------------------

class _Dex:
    _inst: Optional["_Dex"] = None

    def __init__(self):
        from poke_env.data import GenData
        g = GenData.from_gen(9)
        self.moves: Dict[str, dict] = g.moves
        self.pokedex: Dict[str, dict] = g.pokedex
        # Showdown's damageTaken codes, defender type -> attacking type: 0 neutral, 1 weak, 2 resists, 3 immune.
        self.taken: Dict[str, Dict[str, int]] = {}
        for d, row in g.type_chart.items():
            dt = type_name(d)
            codes = {}
            for a, mult in row.items():
                codes[type_name(a)] = 3 if mult == 0 else 1 if mult > 1 else 2 if mult < 1 else 0
            self.taken[dt] = codes
        self._species_cache: Dict[str, Optional[dict]] = {}

    @classmethod
    def get(cls) -> "_Dex":
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def move(self, move) -> dict:
        mid = to_id(move)
        md = self.moves.get(mid)
        if md is None:
            raise KeyError(f"unknown move {move!r}")
        return md

    def species(self, name: str) -> Optional[dict]:
        """The pokedex entry for a species or forme name ('Qwilfish-Hisui', 'Gastrodon-East'). Unknown
        cosmetic formes fall back to the base species."""
        if not name:
            return None
        hit = self._species_cache.get(name)
        if hit is not None or name in self._species_cache:
            return hit
        entry = self.pokedex.get(to_id(name))
        parts = name.split("-")
        while entry is None and len(parts) > 1:
            parts = parts[:-1]
            entry = self.pokedex.get(to_id("-".join(parts)))
        self._species_cache[name] = entry
        return entry


def dex() -> _Dex:
    return _Dex.get()


def species_types(species: str) -> Tuple[str, ...]:
    """The species' own types ('Rotom-Wash' -> ('Electric', 'Water')); () if the species is unknown."""
    entry = dex().species(species)
    if not entry:
        return ()
    return tuple(type_name(t) for t in entry.get("types", ()))


def species_name(species: str) -> str:
    """The canonical forme name ('gastrodoneast' -> 'Gastrodon'), or the input when unknown."""
    entry = dex().species(species)
    return entry.get("name", species) if entry else species


def _effectiveness_code(atk: str, def_type: str) -> int:
    return dex().taken.get(def_type, {}).get(atk, 0)


def get_effectiveness(atk: str, def_type: str) -> int:
    """Showdown's Dex.getEffectiveness for one defending type: +1, -1 or 0 (an immunity counts 0 here)."""
    code = _effectiveness_code(atk, def_type)
    return 1 if code == 1 else -1 if code == 2 else 0


def chart_immune(atk: str, def_types: Iterable[str]) -> bool:
    """Showdown's Dex.getImmunity, negated: True when any defending type takes no damage from atk."""
    return any(_effectiveness_code(atk, t) == 3 for t in def_types)


# ---------------------------------------------------------------------------------------------------
# The inputs.
# ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Mon:
    """One Pokemon as far as type matchups need it.

    species: the current species or forme ('Terapagos-Terastal', 'Mimikyu-Busted', 'Ogerpon-Wellspring-Tera').
    types:   the current types when they differ from the species' own (Soak, Protean, Transform); None = the species'.
    tera:    the Tera type if the Pokemon has Terastallized, else None. 'Stellar' keeps the original types.
    ability, item: as known; None = unknown or none (both act as "no effect").
    added_type: the type added by Forest's Curse / Trick-or-Treat.
    volatiles: ids such as 'smackdown', 'magnetrise', 'ingrain', 'telekinesis', 'tarshot', 'foresight',
               'miracleeye', 'roost'.
    full_hp: for Tera Shell.
    """
    species: str = ""
    types: Optional[Tuple[str, ...]] = None
    tera: Optional[str] = None
    ability: Optional[str] = None
    item: Optional[str] = None
    added_type: Optional[str] = None
    volatiles: FrozenSet[str] = frozenset()
    full_hp: bool = True

    def base_types(self) -> Tuple[str, ...]:
        if self.types is not None:
            return tuple(type_name(t) for t in self.types)
        if self.species.startswith("Arceus") and to_id(self.ability) == "multitype" and self.item is not None:
            # data/conditions.ts arceus.onType: with Multitype, Arceus is the type of the plate it holds
            # (Normal without one). An unknown item (None) leaves the forme's type, which is what a
            # random-battle Arceus holds anyway.
            return (_PLATES.get(to_id(self.item), "Normal"),)
        return species_types(self.species)

    def get_types(self) -> Tuple[str, ...]:
        """Pokemon.getTypes(): the Tera type alone once Terastallized (not Stellar), else the types plus
        an added type; a Pokemon with no types is Normal."""
        tera = type_name(self.tera)
        if tera and tera != STELLAR:
            return (tera,)
        t = self.base_types() or ("Normal",)
        if self.added_type:
            t = t + (type_name(self.added_type),)
        return t

    def has_type(self, t: str) -> bool:
        return t in self.get_types()


@dataclass(frozen=True)
class Field:
    weather: Optional[str] = None   # Showdown ids: sunnyday, raindance, sandstorm, snowscape, hail, desolateland, primordialsea, deltastream
    terrain: Optional[str] = None   # electricterrain, grassyterrain, mistyterrain, psychicterrain
    gravity: bool = False


NO_FIELD = Field()


@dataclass(frozen=True)
class Matchup:
    move_id: str
    move_type: str
    category: str
    immune: bool
    immune_by: str      # '' | 'type' | 'ability:<id>' | 'ground:<flying|levitate|airballoon|magnetrise|telekinesis>'
    type_mod: int       # Showdown's typeMod as it decides "-supereffective" / "-resisted"; 0 when immune or fixed damage
    raw_type_mod: int   # runEffectiveness before the fixed-damage rule (what Wonder Guard looks at)
    fixed_damage: bool
    multiplier: float   # 0 if immune; else 2**type_mod times Tinted Lens / Thick Fat / Heatproof / Purifying Salt / Dry Skin

    @property
    def super_effective(self) -> bool:
        return not self.immune and self.type_mod > 0

    @property
    def resisted(self) -> bool:
        return not self.immune and self.type_mod < 0

    @property
    def label(self) -> str:
        if self.immune:
            return "immune"
        return "super" if self.type_mod > 0 else "resisted" if self.type_mod < 0 else "neutral"

    @property
    def log2(self) -> float:
        """log2 of the type effectiveness, with immune as -3 (for observation inputs)."""
        return -3.0 if self.immune else float(self.type_mod)


# ---------------------------------------------------------------------------------------------------
# The rules.
# ---------------------------------------------------------------------------------------------------

_ATE = {"pixilate": "Fairy", "aerilate": "Flying", "refrigerate": "Ice", "galvanize": "Electric"}
_ATE_SKIP = {"judgment", "multiattack", "naturalgift", "revelationdance", "technoblast", "terrainpulse", "weatherball"}
_NORMALIZE_SKIP = {"hiddenpower", "judgment", "multiattack", "naturalgift", "revelationdance", "struggle",
                   "technoblast", "terrainpulse", "weatherball"}
_MOLD_BREAKERS = {"moldbreaker", "teravolt", "turboblaze"}
_WEATHER_SUPPRESSORS = {"airlock", "cloudnine"}
_PLATES = {"dracoplate": "Dragon", "dreadplate": "Dark", "earthplate": "Ground", "fistplate": "Fighting",
           "flameplate": "Fire", "icicleplate": "Ice", "insectplate": "Bug", "ironplate": "Steel",
           "meadowplate": "Grass", "mindplate": "Psychic", "pixieplate": "Fairy", "skyplate": "Flying",
           "splashplate": "Water", "spookyplate": "Ghost", "stoneplate": "Rock", "toxicplate": "Poison",
           "zapplate": "Electric"}
# TryHit immunities of the defender's ability, by move type.
_ABSORB = {"voltabsorb": "Electric", "lightningrod": "Electric", "motordrive": "Electric",
           "waterabsorb": "Water", "stormdrain": "Water", "dryskin": "Water",
           "flashfire": "Fire", "wellbakedbody": "Fire", "sapsipper": "Grass", "eartheater": "Ground"}
# ... and by move flag.
_FLAG_IMMUNE = {"windrider": "wind", "bulletproof": "bullet", "soundproof": "sound"}
# Abilities that only change the damage, not typeMod.
_HALVE = {"thickfat": ("Fire", "Ice"), "heatproof": ("Fire",), "purifyingsalt": ("Ghost",)}
# Moves whose damage ignores typeMod (Showdown: move.damage, move.damageCallback, move.ohko).
_DELAYED = {"futuresight", "doomdesire"}


def is_grounded(mon: Mon, fld: Field = NO_FIELD, negate_immunity: bool = False,
                ability_suppressed: bool = False) -> Optional[bool]:
    """Pokemon.isGrounded(): True / False, or None for Levitate (immune, shown as the ability)."""
    v = mon.volatiles
    if fld.gravity:
        return True
    if "ingrain" in v or "smackdown" in v:
        return True
    item = to_id(mon.item)
    if item == "ironball":
        return True
    types = mon.get_types()
    if not negate_immunity and "Flying" in types and not (TYPELESS in types and "roost" in v):
        return False
    if to_id(mon.ability) in ("levitate", "eelevate") and not ability_suppressed:
        return None
    if "magnetrise" in v or "telekinesis" in v:
        return False
    return item != "airballoon"


def _weather(attacker: Optional[Mon], defender: Optional[Mon], fld: Field) -> str:
    """Pokemon.effectiveWeather() of the attacker: Air Lock / Cloud Nine on the field hides the weather,
    Utility Umbrella hides sun and rain from its holder."""
    w = to_id(fld.weather)
    if not w:
        return ""
    for m in (attacker, defender):
        if m is not None and to_id(m.ability) in _WEATHER_SUPPRESSORS:
            return ""
    if attacker is not None and to_id(attacker.item) == "utilityumbrella" and w in (
            "sunnyday", "raindance", "desolateland", "primordialsea"):
        return ""
    return w


def move_type(move, attacker: Optional[Mon] = None, fld: Field = NO_FIELD,
              defender: Optional[Mon] = None) -> str:
    """The type the move has when this attacker uses it (ModifyType of the move, then of the ability)."""
    mid = to_id(move)
    md = dex().move(mid)
    t = type_name(md["type"])
    atk = attacker or Mon()
    species = species_name(atk.species) if atk.species else ""
    atk_tera = type_name(atk.tera)
    # The move's own onModifyType (data/moves.ts).
    if mid == "terablast":
        if atk_tera:
            t = atk_tera
    elif mid == "terastarstorm":
        if species == "Terapagos-Stellar":
            t = STELLAR
    elif mid == "ivycudgel":
        for forme, ty in (("Wellspring", "Water"), ("Hearthflame", "Fire"), ("Cornerstone", "Rock")):
            if species in (f"Ogerpon-{forme}", f"Ogerpon-{forme}-Tera"):
                t = ty
    elif mid == "ragingbull":
        t = {"Tauros-Paldea-Combat": "Fighting", "Tauros-Paldea-Blaze": "Fire",
             "Tauros-Paldea-Aqua": "Water"}.get(species, t)
    elif mid == "revelationdance":
        types = atk.get_types()
        t = types[0] if types else t
    elif mid == "aurawheel":
        t = "Dark" if species == "Morpeko-Hangry" else "Electric"
    elif mid == "judgment":
        plate = _PLATES.get(to_id(atk.item))
        if plate:
            t = plate
        elif atk.item is None and species.startswith("Arceus-"):
            # The item is not known: the Arceus forme says which plate it holds.
            t = species_types(species)[0]
    elif mid == "weatherball":
        w = _weather(attacker, defender, fld)
        t = {"sunnyday": "Fire", "desolateland": "Fire", "raindance": "Water", "primordialsea": "Water",
             "sandstorm": "Rock", "hail": "Ice", "snowscape": "Ice", "snow": "Ice"}.get(w, t)
    elif mid == "terrainpulse":
        if is_grounded(atk, fld):
            t = {"electricterrain": "Electric", "grassyterrain": "Grass", "mistyterrain": "Fairy",
                 "psychicterrain": "Psychic"}.get(to_id(fld.terrain), t)
    elif mid == "struggle":
        return TYPELESS
    # The attacker's ability (runEvent ModifyType). Normalize runs before the -ate abilities, but a
    # Pokemon has one ability.
    ab = to_id(atk.ability)
    tera_blast_tera = mid == "terablast" and bool(atk_tera)
    if ab == "normalize":
        if mid not in _NORMALIZE_SKIP and not tera_blast_tera:
            t = "Normal"
    elif ab in _ATE:
        if t == "Normal" and mid not in _ATE_SKIP and not tera_blast_tera:
            t = _ATE[ab]
    elif ab == "liquidvoice":
        if md.get("flags", {}).get("sound"):
            t = "Water"
    return t


def _ignore_immunity(md: dict, mid: str, atk_ability: str):
    """move.ignoreImmunity after ModifyMove (Scrappy / Mind's Eye) and hitStepTypeImmunity's default."""
    ii = md.get("ignoreImmunity")
    if mid in _DELAYED:
        ii = None
    if isinstance(ii, dict):
        ii = dict(ii)
    if atk_ability in ("scrappy", "mindseye"):
        if not ii:
            ii = {}
        if ii is not True:
            ii["Fighting"] = True
            ii["Normal"] = True
    if ii is None:
        ii = md.get("category") == "Status"
    return ii


def _ignores(ii, t: str) -> bool:
    return ii is True or (isinstance(ii, dict) and bool(ii.get(t)))


def matchup(move, defender: Mon, attacker: Optional[Mon] = None, fld: Optional[Field] = None,
            type_override: Optional[str] = None) -> Matchup:
    """The full verdict for this move against this defender.

    type_override: the move's type as the caller already worked it out (obs_matchup.py does, from
    poke-env's view of the attacker); it replaces step 1, and a move id the data does not know is then
    taken as a plain physical move of that type. None: the type is worked out here (the default)."""
    fld = fld or NO_FIELD
    atk = attacker or Mon()
    mid = to_id(move)
    if type_override is None:
        md = dex().move(mid)
    else:
        md = dex().moves.get(mid) or {}
    category = md.get("category", "Physical")
    flags = md.get("flags", {}) or {}
    mtype = move_type(mid, atk, fld, defender) if type_override is None else (type_name(type_override) or TYPELESS)
    atk_ab = to_id(atk.ability)

    # ModifyMove: Mold Breaker and the ignoreAbility moves switch the defender's breakable abilities off.
    ignore_ability = bool(md.get("ignoreAbility")) or atk_ab in _MOLD_BREAKERS or (
        atk_ab == "myceliummight" and category == "Status")
    def_ab = "" if ignore_ability else to_id(defender.ability)
    ii = _ignore_immunity(md, mid, atk_ab)
    def_types = defender.get_types()
    def_item = to_id(defender.item)
    fixed = bool(md.get("damage") or md.get("damageCallback") or md.get("ohko"))

    def type_immune() -> Tuple[bool, str]:
        """Pokemon.runImmunity(move): (immune, why)."""
        if _ignores(ii, mtype):
            return False, ""
        if not mtype or mtype in (TYPELESS, STELLAR):
            return False, ""
        v = defender.volatiles
        negate = def_item == "ringtarget" or (
            "foresight" in v and "Ghost" in def_types and mtype in ("Normal", "Fighting")) or (
            "miracleeye" in v and "Dark" in def_types and mtype == "Psychic")
        if mtype == "Ground":
            g = is_grounded(defender, fld, negate, ability_suppressed=ignore_ability)
            if g:
                return False, ""
            if g is None:
                return True, "ground:levitate"
            if not negate and "Flying" in def_types:
                return True, "ground:flying"
            if "magnetrise" in v:
                return True, "ground:magnetrise"
            if "telekinesis" in v:
                return True, "ground:telekinesis"
            return True, "ground:airballoon"
        if negate:
            return False, ""
        return (True, "type") if chart_immune(mtype, def_types) else (False, "")

    def run_effectiveness() -> int:
        """Pokemon.runEffectiveness(move)."""
        tera = type_name(defender.tera)
        if tera and mtype == STELLAR:
            total = 1
        else:
            total = 0
            species = species_name(defender.species) if defender.species else ""
            for t in def_types:
                tm = get_effectiveness(mtype, t)
                # The move's own onEffectiveness.
                if mid == "freezedry" and t == "Water":
                    tm = 1
                elif mid == "flyingpress":
                    tm = tm + get_effectiveness("Flying", t)
                elif mid == "thousandarrows" and mtype == "Ground":
                    # target.runImmunity('Ground') with no move attached: only Ring Target negates it.
                    grounded = is_grounded(defender, fld, def_item == "ringtarget", ignore_ability)
                    if not grounded and "Flying" in def_types:
                        tm = 0
                # runEvent('Effectiveness'): the defender's ability and item (priority 0), Delta Stream (-1),
                # Tar Shot (-2).
                if def_ab == "disguise" and category != "Status" and species in ("Mimikyu", "Mimikyu-Totem") \
                        and not type_immune()[0]:
                    tm = 0
                elif def_ab == "iceface" and category == "Physical" and species == "Eiscue" \
                        and not type_immune()[0]:
                    tm = 0
                if def_item == "ironball" and mtype == "Ground" and "Flying" in def_types and not (
                        "ingrain" in defender.volatiles or "smackdown" in defender.volatiles or fld.gravity):
                    tm = 0
                if to_id(fld.weather) == "deltastream" and category != "Status" and t == "Flying" and tm > 0:
                    tm = 0
                if "tarshot" in defender.volatiles and mtype == "Fire" and t == def_types[0]:
                    tm = tm + 1
                total += tm
        if species_name(defender.species) == "Terapagos-Terastal" and def_ab == "terashell":
            if not (category == "Status" or mid == "struggle" or type_immune()[0] or total < 0
                    or not defender.full_hp):
                return -1
        return total

    immune, why = False, ""
    # 3. TryHit: the defender's ability.
    if def_ab in _ABSORB and mtype == _ABSORB[def_ab]:
        immune, why = True, f"ability:{def_ab}"
    elif def_ab in _FLAG_IMMUNE and flags.get(_FLAG_IMMUNE[def_ab]):
        immune, why = True, f"ability:{def_ab}"
    elif def_ab == "goodasgold" and category == "Status":
        immune, why = True, "ability:goodasgold"
    elif def_ab == "wonderguard" and category != "Status" and mid != "struggle":
        if run_effectiveness() <= 0 or type_immune()[0]:
            immune, why = True, "ability:wonderguard"
    # 4. Type immunity.
    if not immune:
        immune, why = type_immune()
    raw = 0
    if not immune:
        raw = max(-6, min(6, run_effectiveness()))
    type_mod = 0 if (immune or fixed) else raw

    if immune:
        mult = 0.0
    elif fixed:
        mult = 1.0
    else:
        mult = 2.0 ** type_mod
        if atk_ab == "tintedlens" and type_mod < 0:
            mult *= 2.0
        if def_ab in _HALVE and mtype in _HALVE[def_ab]:
            mult *= 0.5
        if def_ab == "dryskin" and mtype == "Fire":
            mult *= 1.25
    return Matchup(move_id=mid, move_type=mtype, category=category, immune=immune, immune_by=why,
                   type_mod=type_mod, raw_type_mod=raw, fixed_damage=fixed, multiplier=mult)


def effectiveness(move, defender: Mon, attacker: Optional[Mon] = None, fld: Optional[Field] = None) -> float:
    """The damage multiplier: 0 for an immunity, else 0.25 ... 4 (times the ability modifiers)."""
    return matchup(move, defender, attacker, fld).multiplier


# ---------------------------------------------------------------------------------------------------
# Random-battle set candidates: what an unrevealed ability or item could be.
# ---------------------------------------------------------------------------------------------------

DEFAULT_SETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heuristic", "cache",
                                 "gen9randombattle.json")
_SETS_CACHE: Dict[str, dict] = {}


def load_randbats_sets(path: Optional[str] = None) -> dict:
    """heuristic/cache/gen9randombattle.json (fetched by heuristic/setup.js), or {} when it is missing."""
    path = path or os.environ.get("PSPPO_RANDBATS_SETS") or DEFAULT_SETS_PATH
    if path not in _SETS_CACHE:
        try:
            with open(path, encoding="utf-8") as f:
                _SETS_CACHE[path] = json.load(f)
        except OSError:
            _SETS_CACHE[path] = {}
    return _SETS_CACHE[path]


def _set_entry(species: str, sets: dict) -> Optional[dict]:
    if not sets or not species:
        return None
    if species in sets:
        return sets[species]
    name = species_name(species)
    if name in sets:
        return sets[name]
    # Battle-only formes ('Terapagos-Terastal', 'Mimikyu-Busted', 'Ogerpon-Wellspring-Tera') take the
    # base species' or the out-of-battle forme's sets.
    entry = dex().species(species)
    for key in ("baseSpecies", "changesFrom", "battleOnly"):
        alt = entry.get(key) if entry else None
        if isinstance(alt, str) and alt in sets:
            return sets[alt]
    parts = species.split("-")
    while len(parts) > 1:
        parts = parts[:-1]
        if "-".join(parts) in sets:
            return sets["-".join(parts)]
    return None


def set_abilities(species: str, sets: Optional[dict] = None) -> List[str]:
    """Every ability the species' random-battle sets allow (empty when unknown)."""
    e = _set_entry(species, sets if sets is not None else load_randbats_sets())
    if not e:
        return []
    out = list(e.get("abilities", []))
    for r in (e.get("roles") or {}).values():
        for a in r.get("abilities", []):
            if a not in out:
                out.append(a)
    return out


def set_items(species: str, sets: Optional[dict] = None) -> List[str]:
    e = _set_entry(species, sets if sets is not None else load_randbats_sets())
    if not e:
        return []
    out = list(e.get("items", []))
    for r in (e.get("roles") or {}).values():
        for a in r.get("items", []):
            if a not in out:
                out.append(a)
    return out


def known_or_only_ability(species: str, revealed: Optional[str], sets: Optional[dict] = None) -> Optional[str]:
    """The revealed ability, else the only one the sets allow, else None."""
    if revealed:
        return revealed
    cands = set_abilities(species, sets)
    return cands[0] if len(cands) == 1 else None


def possible_immunity(move, species: str, revealed_ability: Optional[str] = None,
                      revealed_item: Optional[str] = None, randbats_sets: Optional[dict] = None,
                      attacker: Optional[Mon] = None, fld: Optional[Field] = None,
                      tera: Optional[str] = None) -> bool:
    """True when the move would not be immune with what is known, but some ability or item the species'
    sets still allow (an unrevealed one) would make it immune: Levitate, Flash Fire, Air Balloon...

    revealed_ability / revealed_item: None = not revealed. An item known to be gone is ''.
    """
    sets = randbats_sets if randbats_sets is not None else load_randbats_sets()
    base = Mon(species=species, ability=revealed_ability, item=revealed_item or None, tera=tera)
    if matchup(move, base, attacker, fld).immune:
        return False
    abilities = [revealed_ability] if revealed_ability else (set_abilities(species, sets) or [None])
    items = [revealed_item or None] if revealed_item is not None else (set_items(species, sets) or [None])
    for ab in abilities:
        for it in items:
            if matchup(move, replace(base, ability=ab, item=it), attacker, fld).immune:
                return True
    return False
