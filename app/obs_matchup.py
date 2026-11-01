"""
The matchup and set-candidate block: one block appended at the END of the flat observation, so
no existing offset moves. On only with the observation setting "matchup" (run_cfg["obs"]["matchup"]
in a checkpoint, ObservationAssembler(matchup=True)); off, the observation is byte for byte what it
was.

What it holds (bench/notes/OBS_MATCHUP.md has the offsets):

  * For each of the 48 move tokens (12 Pokemon x 4 moves, the order of the observation's move
    tokens): against each of the 6 Pokemon on the OTHER side (our moves against the opponent's
    six, the opponent's moves against our six):
      - log2 of the damage multiplier from the type chart and the known abilities and items
        (IMMUNE_LOG2 for an immunity), from effectiveness();
      - the probability that an ability or item the defender may have but has not shown makes
        it immune ("possible immunity", from possible_immunity());
      - a mask: 1 where that Pokemon is known (seen), 0 for a slot not seen yet (its two numbers
        are 0 too).
    Status moves get 0 for both numbers (the mask is still written).
  * For each of the 12 Pokemon: the ability and item candidates as vocabulary ids with their
    probabilities (top 3 abilities, top 4 items, from the random battle sets; exact for our own
    side, which the request names), a corrected item-unknown flag, a "Choice item possible" flag
    and an "item gone" flag. The old item flag (obs_pokemon.py, "not mon.item", which is 0 for an
    unknown item and 1 for a removed one) and the Pokedex ability slots stay as they are.

The set data: randbats_sets_gen9.json, a pinned copy of heuristic/cache/gen9randombattle.json
(509 species; the md5 of the source is in the file), so the observation cannot change when the
cache is fetched again. `python obs_matchup.py --build-sets <gen9randombattle.json>` rewrites it.

Matchup function: effectiveness(move, defender) -> float and possible_immunity() call matchup.py
(Showdown's own rules, checked against the engine on 20,000 cases), with the move's type as worked
out here (move_type: Tera Blast, Ivy Cudgel, Raging Bull, Judgment; with the moveset block on,
MatchupEncoder(full_types=True), also the attacker's ability, form and the weather, as
matchup.move_type has them). The local chart helpers
(type_multiplier and the tables above it) are kept for reference only; nothing in the block uses them.

Also here, for the SetHead (ppo_core.SetHead): set_candidate_tables() (which abilities and items a
species' sets allow, by vocabulary id) and build_set_labels() (the true ability and item of each
opposing Pokemon, from a team that knows them).

Importing this module pulls in numpy only. poke-env is imported when a battle is encoded.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SETS_FILE = os.path.join(HERE, "randbats_sets_gen9.json")
VOCAB_FILE = os.path.join(HERE, "vocab.json")

# The observation setting's name in run_cfg["obs"].
SETTING = "matchup"

# ---------------------------------------------------------------------------------- layout

N_SIDE = 6                       # Pokemon per side
N_MOVE_TOKENS = 48               # 12 Pokemon x 4 moves
N_ABIL_CAND = 3                  # every species has at most 3 set abilities
N_ITEM_CAND = 4                  # 98% of species have at most 4 set items

# One move token: [log2 effectiveness x6 | possible immunity x6 | defender known x6]
MOVE_EFF = (0, 6)
MOVE_POSSIMM = (6, 12)
MOVE_MASK = (12, 18)
MOVE_FEATS = 18

# One Pokemon: [ability ids x3 | ability p x3 | item ids x4 | item p x4 | item unknown | choice possible | item gone]
MON_ABIL_IDS = (0, 3)
MON_ABIL_P = (3, 6)
MON_ITEM_IDS = (6, 10)
MON_ITEM_P = (10, 14)
MON_ITEM_UNKNOWN = 14
MON_CHOICE_POSSIBLE = 15
MON_ITEM_GONE = 16
MON_FEATS = 17
MON_FLAGS = (14, 17)

MOVE_BLOCK = N_MOVE_TOKENS * MOVE_FEATS      # 864
MON_BLOCK = 12 * MON_FEATS                   # 204
BLOCK_DIM = MOVE_BLOCK + MON_BLOCK           # 1068

IMMUNE_LOG2 = -4.0                           # an immunity; every other value is in [-3, 3]
LOG2_CLAMP = 3.0


def block_meta() -> Dict[str, Any]:
    """What ObservationAssembler(matchup=True) puts in meta["matchup"]; the model reads its sizes from it."""
    return {
        "dim": BLOCK_DIM, "move_block": MOVE_BLOCK, "mon_block": MON_BLOCK,
        "move_feats": MOVE_FEATS, "mon_feats": MON_FEATS,
        "n_abil_cand": N_ABIL_CAND, "n_item_cand": N_ITEM_CAND,
        "move": {"eff": MOVE_EFF, "possimm": MOVE_POSSIMM, "mask": MOVE_MASK},
        "mon": {"ability_ids": MON_ABIL_IDS, "ability_p": MON_ABIL_P, "item_ids": MON_ITEM_IDS,
                "item_p": MON_ITEM_P, "flags": MON_FLAGS},
    }


def checkpoint_obs_matchup(ckpt: Any) -> bool:
    """Whether a learner checkpoint's network sees the block (run_cfg["obs"]["matchup"]). No key: it does not."""
    run_cfg = ckpt.get("run_cfg") if isinstance(ckpt, dict) else None
    obs = run_cfg.get("obs") if isinstance(run_cfg, dict) else None
    return bool(obs.get(SETTING, False)) if isinstance(obs, dict) else False


def checkpoint_set_head(ckpt: Any) -> bool:
    """Whether a checkpoint's network has a SetHead (its weights are in the state dict)."""
    model = ckpt.get("model") if isinstance(ckpt, dict) else None
    return isinstance(model, dict) and any(str(k).startswith("set_head.") for k in model)


# ---------------------------------------------------------------------------------- names

_NON_ID = re.compile(r"[^a-z0-9]")


def to_id(name: Any) -> str:
    """Showdown's id of a name: lower case letters and digits only ("Choice Scarf" -> "choicescarf")."""
    return _NON_ID.sub("", str(name).lower())


CHOICE_ITEMS = frozenset({"choiceband", "choicespecs", "choicescarf"})
UNKNOWN_ITEM = "unknown_item"     # poke-env's item before it is revealed

# ---------------------------------------------------------------------------------- set data


def build_sets_table(src: str) -> Dict[str, Any]:
    """The pinned table from heuristic/cache/gen9randombattle.json: per species id, its roles' abilities, items, moves, Tera types."""
    with open(src, "rb") as f:
        raw_bytes = f.read()
    raw = json.loads(raw_bytes.decode("utf-8"))

    def ids(xs: Iterable[Any]) -> List[str]:
        out: List[str] = []
        for x in xs or []:
            i = to_id(x)
            if i and i not in out:
                out.append(i)
        return out

    species: Dict[str, List[Dict[str, List[str]]]] = {}
    for name, entry in raw.items():
        roles = []
        for _role_name, r in (entry.get("roles") or {}).items():
            roles.append({"abilities": ids(r.get("abilities")), "items": ids(r.get("items")),
                          "moves": ids(r.get("moves")), "tera": ids(r.get("teraTypes"))})
        species[to_id(name)] = roles
    return {"source": "heuristic/cache/gen9randombattle.json (Showdown client data/random-battles/gen9 sets)",
            "md5": hashlib.md5(raw_bytes).hexdigest(), "n_species": len(species), "species": species}


class SetTable:
    """The random battle sets of each species, from the pinned randbats_sets_gen9.json."""

    _default: Optional["SetTable"] = None

    def __init__(self, path: str = SETS_FILE):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        self.path = path
        self.md5 = d.get("md5")
        self.species: Dict[str, List[Dict[str, List[str]]]] = d["species"]
        self._resolved: Dict[str, Optional[List[Dict[str, List[str]]]]] = {}

    @classmethod
    def default(cls) -> "SetTable":
        if cls._default is None:
            cls._default = cls()
        return cls._default

    def roles_for(self, species: Any) -> Optional[List[Dict[str, List[str]]]]:
        """
        The roles of a species, or None. A form the sets do not list (a battle-only form such as
        Palafin-Hero or Mimikyu-Busted, a cosmetic form) falls back to the form it comes from:
        the Pokedex's battleOnly, then changesFrom, then baseSpecies.
        """
        sid = to_id(species)
        if sid in self._resolved:
            return self._resolved[sid]
        found = self.species.get(sid)
        if found is None:
            try:
                from poke_env.data import GenData
                entry = GenData.from_gen(9).pokedex.get(sid) or {}
            except Exception:  # noqa: BLE001
                entry = {}
            for key in ("battleOnly", "changesFrom", "baseSpecies"):
                val = entry.get(key)
                for cand in (val if isinstance(val, list) else [val]):
                    if cand and to_id(cand) in self.species:
                        found = self.species[to_id(cand)]
                        break
                if found is not None:
                    break
        self._resolved[sid] = found
        return found


# ---------------------------------------------------------------------------------- candidates


class SetGuess(NamedTuple):
    """What is known or guessed of one Pokemon's ability and item."""
    abilities: Tuple[Tuple[str, float], ...]   # (ability id, probability), most likely first
    items: Tuple[Tuple[str, float], ...]       # (item id, probability), most likely first; () when gone
    known_ability: Optional[str]               # revealed, our own, or the only one the sets allow
    known_item: Optional[str]                  # held and known; None when unknown or gone
    item_unknown: bool
    item_gone: bool
    choice_possible: bool


def _ranked(weights: Dict[str, float]) -> Tuple[Tuple[str, float], ...]:
    return tuple(sorted(((k, float(v)) for k, v in weights.items() if v > 0), key=lambda kv: (-kv[1], kv[0])))


def _item_state(item: Any) -> Tuple[Optional[str], bool, bool]:
    """(known item id, unknown, gone) from poke-env's Pokemon.item."""
    if item == UNKNOWN_ITEM:
        return None, True, False
    if not item:
        return None, False, True
    return to_id(item), False, False


def own_guess(mon: Any) -> SetGuess:
    """Our own Pokemon: the request names its ability and item, so they are exact."""
    ability = to_id(mon.ability) if getattr(mon, "ability", None) else None
    item, unknown, gone = _item_state(getattr(mon, "item", None))
    return SetGuess(abilities=((ability, 1.0),) if ability else (), items=((item, 1.0),) if item else (),
                    known_ability=ability, known_item=item, item_unknown=unknown, item_gone=gone,
                    choice_possible=item in CHOICE_ITEMS)


# Moves poke-env can list that no set holds.
_NOT_SET_MOVES = frozenset({"struggle", "recharge"})


def shown_moves(mon: Any) -> Tuple[str, ...]:
    """The move ids an opposing Pokemon has shown, sorted, without Struggle and Recharge; () while it is Transformed."""
    if getattr(mon, "transformed", False):
        return ()
    return tuple(sorted(i for i in (to_id(m) for m in (getattr(mon, "moves", None) or {})) if i not in _NOT_SET_MOVES))


def shown_tera(mon: Any) -> Optional[str]:
    """The Tera type id of a Pokemon that has Terastallized, else None."""
    if getattr(mon, "is_terastallized", False) and getattr(mon, "tera_type", None) is not None:
        return to_id(getattr(mon.tera_type, "name", mon.tera_type))
    return None


def kept_roles(roles: Sequence[Dict[str, List[str]]], moves: Sequence[str], ability: Optional[str],
               item: Optional[str], tera: Optional[str]) -> List[Dict[str, List[str]]]:
    """
    The roles of a species that fit what a Pokemon has shown (opponent_guess's filter, equally
    likely each): everything shown first; then drop what can come from outside the sets (a Traced
    or Tricked ability or item, a Tera type), then the moves (Transform, Mimic), then nothing.
    """
    def fits(r, use_ability: bool, use_item: bool, use_tera: bool, use_moves: bool) -> bool:
        if use_moves and moves and not set(moves) <= set(r["moves"]):
            return False
        if use_ability and ability and ability not in r["abilities"]:
            return False
        if use_item and item and item not in r["items"]:
            return False
        if use_tera and tera and r["tera"] and tera not in r["tera"]:
            return False
        return True

    kept: List[Dict[str, List[str]]] = []
    for rule in ((True, True, True, True), (False, False, True, True), (False, False, False, True), (False, False, False, False)):
        kept = [r for r in roles if fits(r, *rule)]
        if kept:
            break
    return kept


def opponent_guess(mon: Any, table: SetTable) -> SetGuess:
    """An opposing Pokemon: what it has shown, and for the rest its species' random battle sets."""
    ability = to_id(mon.ability) if getattr(mon, "ability", None) else None
    item, unknown, gone = _item_state(getattr(mon, "item", None))
    roles = table.roles_for(getattr(mon, "species", ""))
    moves = shown_moves(mon)
    tera = shown_tera(mon)

    ab_w: Dict[str, float] = {}
    it_w: Dict[str, float] = {}
    if roles:
        kept = kept_roles(roles, moves, ability, item, tera)
        w = 1.0 / len(kept)
        for r in kept:
            for a in r["abilities"]:
                ab_w[a] = ab_w.get(a, 0.0) + w / len(r["abilities"])
            for it in r["items"]:
                it_w[it] = it_w.get(it, 0.0) + w / len(r["items"])
    else:
        poss = [to_id(a) for a in (getattr(mon, "possible_abilities", None) or []) if a]
        for a in poss:
            ab_w[a] = ab_w.get(a, 0.0) + 1.0 / len(poss)

    if ability:
        abilities: Tuple[Tuple[str, float], ...] = ((ability, 1.0),)
        known_ability: Optional[str] = ability
    else:
        abilities = _ranked(ab_w)
        known_ability = abilities[0][0] if len(abilities) == 1 else None
    if item:
        items: Tuple[Tuple[str, float], ...] = ((item, 1.0),)
        choice = item in CHOICE_ITEMS
    elif unknown:
        items = _ranked(it_w)
        choice = any(it in CHOICE_ITEMS for it, _p in items)
    else:
        items, choice = (), False
    return SetGuess(abilities=abilities, items=items, known_ability=known_ability, known_item=item,
                    item_unknown=unknown, item_gone=gone, choice_possible=choice)


# ---------------------------------------------------------------------------------- matchup

# Defender abilities that make a move's type do nothing, or a move with a flag do nothing.
ABILITY_IMMUNE_TYPES: Dict[str, frozenset] = {
    "levitate": frozenset({"GROUND"}),
    "flashfire": frozenset({"FIRE"}), "wellbakedbody": frozenset({"FIRE"}),
    "waterabsorb": frozenset({"WATER"}), "stormdrain": frozenset({"WATER"}), "dryskin": frozenset({"WATER"}),
    "voltabsorb": frozenset({"ELECTRIC"}), "lightningrod": frozenset({"ELECTRIC"}), "motordrive": frozenset({"ELECTRIC"}),
    "sapsipper": frozenset({"GRASS"}),
    "eartheater": frozenset({"GROUND"}),
}
ABILITY_IMMUNE_FLAGS: Dict[str, str] = {"bulletproof": "bullet", "soundproof": "sound", "windrider": "wind"}
# Defender abilities that halve a type.
ABILITY_HALVES: Dict[str, frozenset] = {
    "thickfat": frozenset({"FIRE", "ICE"}), "heatproof": frozenset({"FIRE"}), "purifyingsalt": frozenset({"GHOST"}),
}
SCRAPPY = frozenset({"scrappy", "mindseye"})
# Damage that does not scale with effectiveness: only an immunity counts.
FIXED_DAMAGE = frozenset({"seismictoss", "nightshade", "superfang", "naturesmadness", "ruination", "finalgambit",
                          "endeavor", "counter", "mirrorcoat", "metalburst", "comeuppance", "bide", "psywave"})
MOVE_FLAGS = ("bullet", "sound", "wind")


class MoveInfo(NamedTuple):
    id: str
    type: str                  # poke-env PokemonType name, after Tera Blast and form-dependent moves
    damaging: bool
    flags: frozenset           # the subset of MOVE_FLAGS the move has
    attacker_ability: Optional[str]
    stellar: bool              # a Stellar Tera Blast


class DefenderInfo(NamedTuple):
    types: Tuple[str, ...]     # current defensive types (poke-env PokemonType names)
    terastallized: bool
    ability: Optional[str]     # known: revealed, our own, or the only one the sets allow
    ability_cands: Tuple[Tuple[str, float], ...]   # only while the ability is unknown
    item: Optional[str]        # known held item; None when unknown or gone
    item_cands: Tuple[Tuple[str, float], ...]      # only while the item is unknown


_CHART: Optional[Dict[str, Dict[str, float]]] = None


def type_chart() -> Dict[str, Dict[str, float]]:
    """poke-env's gen 9 chart: chart[defending type][attacking type] -> multiplier."""
    global _CHART
    if _CHART is None:
        from poke_env.data import GenData
        _CHART = GenData.from_gen(9).type_chart
    return _CHART


_NO_CHART = frozenset({"STELLAR", "THREE_QUESTION_MARKS"})


def type_multiplier(move: MoveInfo, d: DefenderInfo) -> float:
    """The type chart's multiplier, with Freeze-Dry, Flying Press, Thousand Arrows, Scrappy and a Stellar Tera Blast."""
    if move.stellar:
        return 2.0 if d.terastallized else 1.0
    if move.type in _NO_CHART:
        return 1.0
    chart = type_chart()
    m = 1.0
    for t in d.types:
        if t in _NO_CHART or t not in chart:
            continue
        x = chart[t].get(move.type, 1.0)
        if move.id == "freezedry" and t == "WATER":
            x = 2.0
        if x == 0.0:
            if t == "GHOST" and move.type in ("NORMAL", "FIGHTING") and move.attacker_ability in SCRAPPY:
                x = 1.0
            elif t == "FLYING" and move.type == "GROUND" and move.id == "thousandarrows":
                x = 1.0
        m *= x
        if move.id == "flyingpress":
            m *= chart[t].get("FLYING", 1.0)
    return m


def _ability_makes_immune(ability: Optional[str], move: MoveInfo, type_mult: float) -> bool:
    if not ability:
        return False
    types = ABILITY_IMMUNE_TYPES.get(ability)
    if types is not None and move.type in types and not move.stellar:
        return not (ability == "levitate" and move.id == "thousandarrows")
    flag = ABILITY_IMMUNE_FLAGS.get(ability)
    if flag is not None and flag in move.flags:
        return True
    if ability == "wonderguard":
        return type_mult <= 1.0
    return False


def _item_makes_immune(item: Optional[str], move: MoveInfo) -> bool:
    return item == "airballoon" and move.type == "GROUND" and not move.stellar and move.id != "thousandarrows"


_VERDICT_CACHE: Dict[Tuple[Any, ...], Tuple[bool, float]] = {}


def _verdict(move: MoveInfo, types: Tuple[str, ...], terastallized: bool, ability: Optional[str],
             item: Optional[str]) -> Tuple[bool, float]:
    """
    (immune, damage multiplier) of a move on a defender with these current types, Tera state, ability
    and item, from matchup.py. A pure function of its arguments, kept in one cache for the process:
    every encoder and every candidate ability or item of every defender asks the same few questions.
    """
    key = (move, types, terastallized, ability, item)
    hit = _VERDICT_CACHE.get(key)
    if hit is not None:
        return hit
    import matchup as mu
    mtypes = tuple(mu.type_name(t) or mu.TYPELESS for t in types) or (mu.TYPELESS,)
    tera = None
    if terastallized:
        tera = mtypes[0] if len(mtypes) == 1 else mu.STELLAR    # only the "is it Terastallized" matters past the types
    t = mu.STELLAR if move.stellar else (mu.TYPELESS if move.type in _NO_CHART else move.type)
    m = mu.matchup(move.id, mu.Mon(types=mtypes, tera=tera, ability=ability, item=item, full_hp=False),
                   mu.Mon(ability=move.attacker_ability), type_override=t)
    if len(_VERDICT_CACHE) > _VERDICT_CACHE_LIMIT:
        _VERDICT_CACHE.clear()
    hit = _VERDICT_CACHE[key] = (bool(m.immune), float(m.multiplier))
    return hit


_VERDICT_CACHE_LIMIT = 500_000


def effectiveness(move: MoveInfo, defender: DefenderInfo) -> float:
    """
    The damage multiplier of a damaging move on a defender, from matchup.py (Showdown's rules,
    checked against the engine on 20,000 cases) with the move's type as worked out here and what is
    known of the defender: 0 for an immunity (type, a known ability, a known Air Balloon), the
    halving abilities, Dry Skin's 1.25 from Fire, Tinted Lens on the attacker, Wonder Guard, the
    fixed-damage moves (1 unless immune). Status moves: 1.
    """
    if not move.damaging:
        return 1.0
    return _verdict(move, defender.types, defender.terastallized, defender.ability, defender.item)[1]


def possible_immunity(move: MoveInfo, defender: DefenderInfo) -> float:
    """The probability that an ability or item the defender may have but has not shown makes it immune. 0 when it is known."""
    if not move.damaging:
        return 0.0
    d = defender
    if _verdict(move, d.types, d.terastallized, d.ability, d.item)[0]:
        return 0.0
    pa = sum(p for a, p in d.ability_cands
             if _verdict(move, d.types, d.terastallized, a, d.item)[0]) if d.ability is None else 0.0
    pi = sum(p for it, p in d.item_cands
             if _verdict(move, d.types, d.terastallized, d.ability, it)[0]) if d.item is None else 0.0
    return float(1.0 - (1.0 - min(1.0, pa)) * (1.0 - min(1.0, pi)))


def log2_code(mult: float) -> float:
    """What the block stores for a multiplier: IMMUNE_LOG2 for 0, otherwise log2 clamped to [-3, 3]."""
    if mult <= 0.0:
        return IMMUNE_LOG2
    return float(max(-LOG2_CLAMP, min(LOG2_CLAMP, math.log2(mult))))


_TYPE_NAMES: Dict[Any, str] = {}


def _type_name(t: Any) -> Optional[str]:
    """A type's upper-case name (poke-env's PokemonType names) from the enum or a string, remembered per value."""
    if t is None:
        return None
    try:
        return _TYPE_NAMES[t]
    except KeyError:
        name = _TYPE_NAMES[t] = str(getattr(t, "name", t)).upper()
        return name
    except TypeError:        # not hashable
        return str(getattr(t, "name", t)).upper()


def defender_types(mon: Any) -> Tuple[str, ...]:
    """poke-env's current types, except that a Stellar Tera keeps the original types for defence."""
    if getattr(mon, "is_terastallized", False) and _type_name(getattr(mon, "tera_type", None)) == "STELLAR":
        t1, t2 = getattr(mon, "_type_1", None), getattr(mon, "_type_2", None)
    else:
        t1, t2 = getattr(mon, "type_1", None), getattr(mon, "type_2", None)
    n1, n2 = _type_name(t1), _type_name(t2)
    if n1 and n2:
        return (n1, n2)
    return (n1,) if n1 else ((n2,) if n2 else ())


def defender_info(mon: Any, guess: SetGuess, own: bool) -> DefenderInfo:
    return DefenderInfo(
        types=defender_types(mon),
        terastallized=bool(getattr(mon, "is_terastallized", False)),
        ability=guess.known_ability,
        ability_cands=() if (own or guess.known_ability is not None) else guess.abilities,
        item=guess.known_item,
        item_cands=guess.items if (guess.item_unknown and not own) else (),
    )


# Weather Ball's type in each weather (data/moves.ts, weatherball.onModifyType), by poke-env's Weather names as ids.
WEATHER_BALL_TYPES: Dict[str, str] = {"sunnyday": "FIRE", "desolateland": "FIRE", "raindance": "WATER",
                                      "primordialsea": "WATER", "sandstorm": "ROCK", "hail": "ICE", "snowscape": "ICE",
                                      "snow": "ICE"}
_SUN_RAIN = frozenset({"sunnyday", "desolateland", "raindance", "primordialsea"})
# Moves whose full type (move_type(full=True)) reads more of the attacker than its species, Tera state and
# ability: the weather (Weather Ball), its current types (Revelation Dance). The caches key on type_context().
WEATHER_MOVES = frozenset({"weatherball"})
TYPE_MOVES = frozenset({"revelationdance"})
CONTEXT_MOVES = WEATHER_MOVES | TYPE_MOVES


def battle_weather(battle: Any) -> Optional[str]:
    """The battle's weather as Weather Ball reads it (a WEATHER_BALL_TYPES key), or None."""
    w = getattr(battle, "weather", None)
    if not w:
        return None
    for k in w:
        name = to_id(getattr(k, "name", k))
        return name if name in WEATHER_BALL_TYPES else None
    return None


def attacker_weather(attacker: Any, weather: Optional[str]) -> Optional[str]:
    """The weather a move of this attacker sees: a known Utility Umbrella hides sun and rain from its holder."""
    if weather in _SUN_RAIN and to_id(getattr(attacker, "item", None) or "") == "utilityumbrella":
        return None
    return weather


def type_context(attacker: Any, move_ids: Iterable[str], weather: Optional[str]) -> Tuple[Optional[str], Optional[Tuple[str, ...]]]:
    """
    What move_type(full=True) reads of an attacker, for these move ids, beyond its species, Tera state and
    ability: (its weather if Weather Ball is among them, its current types if Revelation Dance is). For cache keys.
    """
    if CONTEXT_MOVES.isdisjoint(move_ids):
        return None, None
    ids = move_ids if isinstance(move_ids, (set, frozenset)) else frozenset(move_ids)
    w = attacker_weather(attacker, weather) if not WEATHER_MOVES.isdisjoint(ids) else None
    ty = defender_types(attacker) if not TYPE_MOVES.isdisjoint(ids) else None
    return w, ty


def move_type(move: Any, attacker: Any, attacker_ability: Optional[str] = None, weather: Optional[str] = None,
              full: bool = False) -> Tuple[str, bool]:
    """
    (type name, Stellar) of a move used by `attacker`.

    full=False: efe43f7's rules (Tera Blast, Ivy Cudgel, Raging Bull, Judgment), which every matchup-only
    checkpoint was trained on. full=True (the observation with the moveset block): also what matchup.move_type
    does (Showdown's onModifyType of the move, then of the attacker's ability): Tera Starstorm of a
    Terastallized Terapagos (Stellar), Revelation Dance (the attacker's first current type), Aura Wheel,
    Weather Ball (`weather`: battle_weather(); attacker_weather() applies Utility Umbrella), and the
    attacker's ability `attacker_ability` (Normalize, Pixilate, Aerilate, Refrigerate, Galvanize, Liquid
    Voice). Not modelled: Air Lock / Cloud Nine on the field, Morpeko's Hangry form (poke-env keeps the
    species "morpeko" through Hunger Switch, so Aura Wheel reads Electric).
    """
    mid = getattr(move, "id", "")
    t = _type_name(getattr(move, "type", None)) or "THREE_QUESTION_MARKS"
    if mid == "terablast" and getattr(attacker, "is_terastallized", False):
        tera = _type_name(getattr(attacker, "tera_type", None))
        if tera == "STELLAR":
            return "STELLAR", True
        if tera:
            return tera, False
    elif mid == "ivycudgel":
        sp = to_id(getattr(attacker, "species", ""))
        for prefix, tt in (("ogerponwellspring", "WATER"), ("ogerponhearthflame", "FIRE"), ("ogerponcornerstone", "ROCK")):
            if sp.startswith(prefix):
                return tt, False
    elif mid == "ragingbull":
        sp = to_id(getattr(attacker, "species", ""))
        for prefix, tt in (("taurospaldeacombat", "FIGHTING"), ("taurospaldeablaze", "FIRE"), ("taurospaldeaaqua", "WATER")):
            if sp.startswith(prefix):
                return tt, False
    elif mid in ("judgment", "multiattack"):
        base = _type_name(getattr(attacker, "_type_1", None))
        if base:
            return base, False
    if not full:
        return t, False
    return _full_type(mid, t, move, attacker, attacker_ability, weather)


def _full_type(mid: str, t: str, move: Any, attacker: Any, ability: Optional[str],
               weather: Optional[str]) -> Tuple[str, bool]:
    """move_type(full=True) past efe43f7's rules: matchup.move_type's other branches, on poke-env's view of the attacker."""
    import matchup as mu
    sp = to_id(getattr(attacker, "species", ""))
    if mid == "terastarstorm":
        # Showdown: Stellar for Terapagos-Stellar, the form a Terapagos takes when it Terastallizes (poke-env
        # may keep the species it switched in as)
        if sp.startswith("terapagos") and (sp == "terapagosstellar" or getattr(attacker, "is_terastallized", False)):
            return "STELLAR", True
    elif mid == "revelationdance":
        types = defender_types(attacker)            # Pokemon.getTypes(): the Tera type, except Stellar, replaces them
        if types and types[0] not in _NO_CHART:
            t = types[0]
    elif mid == "aurawheel":
        t = "DARK" if sp == "morpekohangry" else "ELECTRIC"
    elif mid == "weatherball":
        t = WEATHER_BALL_TYPES.get(attacker_weather(attacker, weather) or "", t)
    # The attacker's ability (ModifyType). A Terastallized Tera Blast returned above, untouched by these.
    ab = to_id(ability) if ability else ""
    if ab == "normalize":
        if mid not in mu._NORMALIZE_SKIP:
            t = "NORMAL"
    elif ab in mu._ATE:
        if t == "NORMAL" and mid not in mu._ATE_SKIP:
            t = mu._ATE[ab].upper()
    elif ab == "liquidvoice":
        if "sound" in (getattr(move, "flags", None) or ()):
            t = "WATER"
    return t, False


def move_info(move: Any, attacker: Any, attacker_ability: Optional[str], weather: Optional[str] = None,
              full: bool = False) -> MoveInfo:
    """The move as the multipliers see it; full / weather: move_type's."""
    cat = str(getattr(getattr(move, "category", None), "name", "STATUS")).upper()
    flags = getattr(move, "flags", None) or ()
    t, stellar = move_type(move, attacker, attacker_ability, weather, full)
    return MoveInfo(id=getattr(move, "id", ""), type=t, damaging=cat != "STATUS",
                    flags=frozenset(f for f in MOVE_FLAGS if f in flags),
                    attacker_ability=attacker_ability, stellar=stellar)


# ---------------------------------------------------------------------------------- the encoder

_CACHE_LIMIT = 200_000


class MatchupEncoder:
    """
    Writes the block for one battle. One per assembler; it remembers what it has worked out.

    encode() is the fast path: each defender and each move is interned to a small integer, and the
    move tokens of a Pokemon are kept per (its moves, the six defenders), so a decision is mostly
    a few dictionary lookups and one slice write per Pokemon. encode_reference() works every number out again; the
    two write the same bytes (tests/test_obs_matchup.py).

    full_types: the move tokens use move_type(full=True) (abilities, forms, weather). On only with the
    moveset block (ObservationAssembler(movesets=True)), so the matchup-only observation stays byte for
    byte efe43f7's, which every matchup checkpoint so far was trained on.
    """

    def __init__(self, vocab_map: Dict[str, Dict[str, int]], table: Optional[SetTable] = None, full_types: bool = False):
        self.full_types = bool(full_types)
        self.table = table or SetTable.default()
        self._abil = vocab_map.get("pokemon.ability", {})
        self._item = vocab_map.get("pokemon.item", {})
        self._guesses: Dict[Any, SetGuess] = {}
        self._own_guesses: Dict[Any, SetGuess] = {}      # own_guess reads only the ability and the item
        self._pairs: Dict[Tuple[MoveInfo, DefenderInfo], Tuple[float, float]] = {}
        self._clear_fast()

    def _clear_fast(self) -> None:
        self._def_ids: Dict[DefenderInfo, int] = {}
        self._defs: List[DefenderInfo] = []
        self._move_keys: Dict[Any, Tuple[int, ...]] = {}  # (move ids, species, Tera, ability) -> interned MoveInfos
        self._move_info_ids: Dict[MoveInfo, int] = {}
        self._moves: List[MoveInfo] = []
        self._rows: Dict[Any, np.ndarray] = {}           # (move ids of a Pokemon, defender ids, dtype) -> its move tokens
        self._mon_rows: Dict[Any, np.ndarray] = {}       # (SetGuess, dtype) -> its 17 numbers

    def ability_id(self, name: Optional[str]) -> int:
        """The vocabulary id the observation uses for an ability (0: none or not in the vocabulary)."""
        return self._abil.get(name, -1) + 1 if name else 0

    def item_id(self, name: Optional[str]) -> int:
        return self._item.get(name, -1) + 1 if name else 0

    def guess(self, mon: Any, own: bool) -> SetGuess:
        if own:
            okey = (getattr(mon, "ability", None), getattr(mon, "item", None))
            hit = self._own_guesses.get(okey)
            if hit is None:
                hit = self._own_guesses[okey] = own_guess(mon)
            return hit
        moves = () if getattr(mon, "transformed", False) else tuple(sorted(getattr(mon, "moves", None) or {}))
        tera = getattr(mon, "tera_type", None) if getattr(mon, "is_terastallized", False) else None
        key = (getattr(mon, "species", None), moves, getattr(mon, "ability", None), getattr(mon, "item", None),
               _type_name(tera), tuple(getattr(mon, "possible_abilities", None) or ()))
        hit = self._guesses.get(key)
        if hit is None:
            if len(self._guesses) > _CACHE_LIMIT:
                self._guesses.clear()
            hit = self._guesses[key] = opponent_guess(mon, self.table)
        return hit

    def pair(self, move: MoveInfo, d: DefenderInfo) -> Tuple[float, float]:
        key = (move, d)
        hit = self._pairs.get(key)
        if hit is None:
            if len(self._pairs) > _CACHE_LIMIT:
                self._pairs.clear()
            hit = self._pairs[key] = (log2_code(effectiveness(move, d)), possible_immunity(move, d))
        return hit

    def mons(self, battle: Any) -> List[Optional[Any]]:
        """The 12 Pokemon in the observation's order: list(battle.team.values()), then the opponent's, 6 each."""
        own = list(battle.team.values())
        opp = list(battle.opponent_team.values())
        return [own[i] if i < len(own) else None for i in range(N_SIDE)] + \
               [opp[i] if i < len(opp) else None for i in range(N_SIDE)]

    def block(self, battle: Any, reference: bool = False) -> np.ndarray:
        out = np.zeros(BLOCK_DIM, dtype=np.float16)
        (self.encode_reference if reference else self.encode)(battle, out, 0)
        return out

    # ------------------------------------------------------------------ the rows

    def move_row(self, mi: MoveInfo, foes: Sequence[Optional[DefenderInfo]]) -> np.ndarray:
        """The 18 numbers of one move token against the six Pokemon of the other side (None: not seen)."""
        row = np.zeros(MOVE_FEATS, dtype=np.float32)
        for j, d in enumerate(foes):
            if d is None:
                continue
            row[MOVE_MASK[0] + j] = 1.0
            if not mi.damaging:
                continue
            code, pim = self.pair(mi, d)
            row[MOVE_EFF[0] + j] = code
            row[MOVE_POSSIMM[0] + j] = pim
        return row

    def mon_row(self, g: SetGuess) -> np.ndarray:
        """The 17 numbers of one Pokemon."""
        row = np.zeros(MON_FEATS, dtype=np.float32)
        for k, (name, p) in enumerate(g.abilities[:N_ABIL_CAND]):
            row[MON_ABIL_IDS[0] + k] = self.ability_id(name)
            row[MON_ABIL_P[0] + k] = p
        for k, (name, p) in enumerate(g.items[:N_ITEM_CAND]):
            row[MON_ITEM_IDS[0] + k] = self.item_id(name)
            row[MON_ITEM_P[0] + k] = p
        row[MON_ITEM_UNKNOWN] = 1.0 if g.item_unknown else 0.0
        row[MON_CHOICE_POSSIBLE] = 1.0 if g.choice_possible else 0.0
        row[MON_ITEM_GONE] = 1.0 if g.item_gone else 0.0
        return row

    def _infos(self, mons: List[Optional[Any]]) -> Tuple[List[Optional[SetGuess]], List[Optional[DefenderInfo]]]:
        guesses = [None if m is None else self.guess(m, i < N_SIDE) for i, m in enumerate(mons)]
        defenders = [None if m is None else defender_info(m, guesses[i], i < N_SIDE) for i, m in enumerate(mons)]
        return guesses, defenders

    # ------------------------------------------------------------------ writing

    def encode_reference(self, battle: Any, buffer: np.ndarray, start: int) -> None:
        """Writes the block into buffer[start:start + BLOCK_DIM], which the caller has zeroed, working every row out."""
        mons = self.mons(battle)
        guesses, defenders = self._infos(mons)
        weather = battle_weather(battle) if self.full_types else None
        for i, mon in enumerate(mons):
            if mon is None:
                continue
            foes = defenders[N_SIDE:] if i < N_SIDE else defenders[:N_SIDE]
            for k, mv in enumerate(list(mon.moves.values())[:4]):
                if mv is None:
                    continue
                base = start + (i * 4 + k) * MOVE_FEATS
                mi = move_info(mv, mon, guesses[i].known_ability, weather, self.full_types)
                buffer[base:base + MOVE_FEATS] = self.move_row(mi, foes)
        for i, g in enumerate(guesses):
            if g is not None:
                s = start + MOVE_BLOCK + i * MON_FEATS
                buffer[s:s + MON_FEATS] = self.mon_row(g)

    def _move_id(self, mv: Any, mon: Any, att_ability: Optional[str], weather: Optional[str]) -> int:
        """The interned MoveInfo of one move of one attacker (-1 for an empty slot)."""
        if mv is None:
            return -1
        mi = move_info(mv, mon, att_ability, weather, self.full_types)
        mid = self._move_info_ids.get(mi)
        if mid is None:
            mid = self._move_info_ids[mi] = len(self._moves)
            self._moves.append(mi)
        return mid

    def encode(self, battle: Any, buffer: np.ndarray, start: int) -> Tuple[List[Optional[Any]], List[Optional[SetGuess]],
                                                                         List[Optional[DefenderInfo]], List[int]]:
        """
        Writes the block into buffer[start:start + BLOCK_DIM], which the caller has zeroed. Returns
        what it worked out on the way, (mons, guesses, defenders, interned defender ids), for the
        moveset block (obs_movesets.py), which reads the same twelve Pokemon.
        """
        if max(len(self._rows), len(self._def_ids), len(self._move_keys), len(self._mon_rows)) > _CACHE_LIMIT:
            self._clear_fast()
        dt = buffer.dtype.char
        mons = self.mons(battle)
        guesses, defenders = self._infos(mons)
        full = self.full_types
        weather = battle_weather(battle) if full else None
        def_ids = self._def_ids
        dids: List[int] = []
        for d in defenders:
            if d is None:
                dids.append(-1)
                continue
            x = def_ids.get(d)
            if x is None:
                x = def_ids[d] = len(self._defs)
                self._defs.append(d)
            dids.append(x)
        own_side, opp_side = tuple(dids[:N_SIDE]), tuple(dids[N_SIDE:])
        rows, move_keys = self._rows, self._move_keys
        for i, mon in enumerate(mons):
            if mon is None:
                continue
            moves = list(mon.moves.values())[:4]
            if not moves:
                continue
            att_ability = guesses[i].known_ability
            terastallized = getattr(mon, "is_terastallized", False)
            # everything move_info reads: the moves (an id fixes type, category and flags), and the
            # attacker's species (form), Tera and ability; with full types, its weather and current
            # types where a move reads them (type_context)
            ids = tuple(getattr(mv, "id", None) for mv in moves)
            mkey = (ids, getattr(mon, "species", None), terastallized,
                    _type_name(getattr(mon, "tera_type", None)) if terastallized else None, att_ability,
                    type_context(mon, ids, weather) if full else None)
            mids = move_keys.get(mkey)
            if mids is None:
                mids = move_keys[mkey] = tuple(self._move_id(mv, mon, att_ability, weather) for mv in moves)
            foe_ids = opp_side if i < N_SIDE else own_side
            rkey = (mids, foe_ids, dt)
            row = rows.get(rkey)
            if row is None:
                foes = [None if x < 0 else self._defs[x] for x in foe_ids]
                row = rows[rkey] = np.concatenate([
                    self.move_row(self._moves[m], foes) if m >= 0 else np.zeros(MOVE_FEATS, dtype=np.float32)
                    for m in mids]).astype(buffer.dtype)
            b = start + i * 4 * MOVE_FEATS
            buffer[b:b + row.shape[0]] = row
        mon_rows = self._mon_rows
        for i, g in enumerate(guesses):
            if g is None:
                continue
            gkey = (g, dt)
            row = mon_rows.get(gkey)
            if row is None:
                row = mon_rows[gkey] = self.mon_row(g).astype(buffer.dtype)
            s = start + MOVE_BLOCK + i * MON_FEATS
            buffer[s:s + MON_FEATS] = row
        return mons, guesses, defenders, dids


# ---------------------------------------------------------------------------------- SetHead support


def load_vocab_lists(path: str = VOCAB_FILE) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _vocab_map(vocab_lists: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
    # The same map ObservationAssembler builds.
    return {category: {item.lower(): i for i, item in enumerate(items)} for category, items in vocab_lists.items()}


def set_candidate_tables(vocab_lists: Optional[Dict[str, List[str]]] = None,
                         table: Optional[SetTable] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    (abilities [n_species + 1, n_abilities + 1], items [n_species + 1, n_items + 1]), bool: which
    vocabulary ids a species' random battle sets allow, rows and columns by the observation's ids
    (0 = none / not in the vocabulary, never allowed). A species without sets has an empty row.
    """
    vocab_lists = vocab_lists or load_vocab_lists()
    table = table or SetTable.default()
    vm = _vocab_map(vocab_lists)
    species = vocab_lists.get("pokemon.species", [])
    ab = np.zeros((len(species) + 1, len(vocab_lists.get("pokemon.ability", [])) + 1), dtype=bool)
    it = np.zeros((len(species) + 1, len(vocab_lists.get("pokemon.item", [])) + 1), dtype=bool)
    for i, name in enumerate(species):
        roles = table.roles_for(name) or []
        for r in roles:
            for a in r["abilities"]:
                j = vm["pokemon.ability"].get(a, -1) + 1
                if j > 0:
                    ab[i + 1, j] = True
            for x in r["items"]:
                j = vm["pokemon.item"].get(x, -1) + 1
                if j > 0:
                    it[i + 1, j] = True
    return ab, it


def _species_id(vm: Dict[str, Dict[str, int]], species: Any) -> int:
    # utils.get_id's lookup (normalize_name), without counting misses as unknown names.
    from utils import normalize_name
    clean = normalize_name(species)
    return vm.get("pokemon.species", {}).get(clean, -1) + 1 if clean else 0


def _true_parts(truth: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """(species, base ability, item) of a Pokemon whose set is known: a poke-env Pokemon or a dict."""
    if isinstance(truth, dict):
        return truth.get("species"), truth.get("ability"), truth.get("item")
    ability = getattr(truth, "base_ability", None) or getattr(truth, "_ability", None) or getattr(truth, "ability", None)
    return getattr(truth, "species", None), ability, getattr(truth, "item", None)


LABEL_SPECIES, LABEL_ABILITY, LABEL_ITEM = 0, 1, 2
# With the moveset block (obs_movesets.py) a row has 8 numbers: the three above, then the true moves
# (4) and the true Tera type (1).
LABEL_MOVES = (3, 7)
LABEL_TERA = 7
LABEL_WIDTH, LABEL_WIDTH_MOVESETS = 3, 8


def _true_moves_tera(truth: Any) -> Tuple[List[str], Optional[str]]:
    """(move ids, Tera type id) of a Pokemon whose set is known: a poke-env Pokemon (the partner seat's own) or a dict."""
    if isinstance(truth, dict):
        moves = [to_id(m) for m in (truth.get("moves") or [])]
        tera = truth.get("tera") or truth.get("teraType") or truth.get("tera_type")
        return [m for m in moves if m], (to_id(tera) if tera else None)
    moves = [to_id(m) for m in (getattr(truth, "moves", None) or {})]
    from obs_tera import own_tera_type_name   # branch decide keeps it there (train-v2: obs_assembler; the same function)
    tera = own_tera_type_name(truth)
    return [m for m in moves if m], (to_id(tera) if tera else None)


def build_set_labels(battle: Any, true_team: Any, vocab_lists: Optional[Dict[str, List[str]]] = None,
                     vocab_map: Optional[Dict[str, Dict[str, int]]] = None, movesets: bool = False) -> np.ndarray:
    """
    The SetHead's labels for one decision: int32 [6, 3], one row per opposing Pokemon in the
    observation's order (tokens 7-12): [species id, true ability id, true item id], vocabulary ids
    as the observation writes them. An ability or item is labelled only while the observing side
    has not seen it (ability None; item still "unknown_item"); 0 means no label.

    movesets=True (the moveset block, obs_movesets.py): int32 [6, 8], each row followed by
      - the true moves (4 numbers, the set's order): +id (move vocabulary id + 1) for a move the
        observing side has not seen, -id for one it has (left out of the loss: not a label, and not
        a candidate either), 0 for none;
      - the true Tera type: obs_movesets.TERA_TYPES index + 1 while the Pokemon has not
        Terastallized, 0 after.

    true_team: the opposing side's real team. In self-play, the partner seat's battle.team (a
    dict "p2: Name" -> poke-env Pokemon, the same keys battle.opponent_team uses); for the
    rebuilt top-player games, a dict of the same keys -> {"species", "ability", "item"} (and
    "moves", "tera" for the moveset labels). A key that is missing is matched by species.
    """
    if vocab_map is None:
        vocab_map = _vocab_map(vocab_lists or load_vocab_lists())
    out = np.zeros((N_SIDE, LABEL_WIDTH_MOVESETS if movesets else LABEL_WIDTH), dtype=np.int32)
    truth_by_key = dict(true_team.items()) if hasattr(true_team, "items") else {i: t for i, t in enumerate(true_team or [])}
    by_species: Dict[str, Any] = {}
    for t in truth_by_key.values():
        sp = _true_parts(t)[0]
        if sp:
            by_species.setdefault(to_id(sp), t)
    for j, (key, mon) in enumerate(list(battle.opponent_team.items())[:N_SIDE]):
        out[j, LABEL_SPECIES] = _species_id(vocab_map, getattr(mon, "species", None))
        truth = truth_by_key.get(key)
        if truth is None:
            truth = by_species.get(to_id(getattr(mon, "species", "")))
        if truth is None:
            continue
        _sp, t_ability, t_item = _true_parts(truth)
        if getattr(mon, "ability", None) is None and t_ability:
            out[j, LABEL_ABILITY] = vocab_map.get("pokemon.ability", {}).get(to_id(t_ability), -1) + 1
        if getattr(mon, "item", None) == UNKNOWN_ITEM and t_item and t_item != UNKNOWN_ITEM:
            out[j, LABEL_ITEM] = vocab_map.get("pokemon.item", {}).get(to_id(t_item), -1) + 1
        if movesets:
            t_moves, t_tera = _true_moves_tera(truth)
            seen = {to_id(m) for m in (getattr(mon, "moves", None) or {})}
            mv_vocab = vocab_map.get("move.id", {})
            k = LABEL_MOVES[0]
            for m in t_moves[:4]:
                mid = mv_vocab.get(m, -1) + 1
                if mid <= 0:
                    continue
                out[j, k] = -mid if m in seen else mid
                k += 1
            if t_tera and not getattr(mon, "is_terastallized", False):
                from obs_movesets import TERA_INDEX
                out[j, LABEL_TERA] = TERA_INDEX.get(t_tera, -1) + 1
    return out


# ---------------------------------------------------------------------------------- model helper


def build_model(cfg: Any, matchup: bool = True, set_head: bool = False, set_head_coef: float = 0.0,
                movesets: bool = False, opp_action_head: bool = False) -> Any:
    """
    cfg.make_model() with the block on or off (and the moveset block, obs_movesets.py, when asked;
    it needs the matchup block), and a SetHead (then an OppActionHead) attached when asked, in the
    learner's order.
    """
    import dataclasses
    if movesets and not matchup:
        raise ValueError("the moveset block needs the matchup block")
    obs = dataclasses.replace(cfg.obs, matchup=bool(matchup), **({"movesets": bool(movesets)} if hasattr(cfg.obs, "movesets") else {}))
    net = dataclasses.replace(cfg, obs=obs).make_model()
    if bool(getattr(net, "matchup", False)) != bool(matchup):
        raise RuntimeError(f"make_model gave a network with matchup={getattr(net, 'matchup', None)}, wanted {matchup}")
    if bool(getattr(net, "movesets", False)) != bool(movesets):
        raise RuntimeError(f"make_model gave a network with movesets={getattr(net, 'movesets', None)}, wanted {movesets}")
    if set_head:
        net.attach_set_head(set_head_coef)
    if opp_action_head:
        net.attach_opp_action_head(0.0)
    return net


def _main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Rewrite randbats_sets_gen9.json from the client's gen9randombattle.json.")
    ap.add_argument("--build-sets", metavar="GEN9RANDOMBATTLE_JSON", required=True)
    ap.add_argument("--out", default=SETS_FILE)
    args = ap.parse_args()
    t = build_sets_table(args.build_sets)
    head = json.dumps({k: v for k, v in t.items() if k != "species"}, sort_keys=True)
    body = ",\n".join(json.dumps(k) + ": " + json.dumps(v, separators=(",", ":")) for k, v in sorted(t["species"].items()))
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        # One species per line, so a new fetch of the sets shows as a readable diff.
        f.write(head[:-1] + ', "species": {\n' + body + "\n}}\n")
    with open(args.out, "r", encoding="utf-8") as f:
        if json.load(f) != t:
            raise SystemExit(f"{args.out} does not read back as the table")
    print(f"{args.out}: {t['n_species']} species from {args.build_sets} (md5 {t['md5']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
