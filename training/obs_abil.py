"""
The opponent's abilities and items, and what a switch costs, as the observation should show them.

bench/notes/abilities_obs.md (this branch) measured the observation against the server's raw
requests at 42,058 decisions and found the gaps below. Each fix here only puts the right value
into a slot the observation already has (the layout and the size do not change), and each is a
setting that is OFF by default:

    abil_sets        the opponent's ability slot 0 is filled when the gen9 random battle sets give
                     its species exactly one ability (441 of 509 species), and slots 1-3 of every
                     Pokemon hold the abilities its random battle sets can have instead of the
                     Pokedex ones (the Pokedex names 1 to 3 abilities a set never has 71% of the
                     time). The ability-unknown flag follows slot 0. randbats_abilities_gen9.json
                     is the table, taken from Showdown's data/random-battles/gen9/sets.json.
    abil_announced   abilities and items the server announced on lines poke-env does not read
                     ("[from] ability:" on -weather, -fieldstart, -heal, -curestatus, -start,
                     -damage ...; "[from] item:" on -status: Toxic Orb, Flame Orb ...): 9,893 and
                     2,677 (decision, Pokemon) pairs of the audit showed "unknown" after them.
    abil_history     the transition block (the field token the 64-decision history is made of)
                     writes the ability of a bare "-ability|POKEMON|Intimidate" line too: 3,352 of
                     3,368 such lines left the ability id 0.
    item_unknown     the "item unknown" flag means unknown (poke-env's "unknown_item"), not "has no
                     item", and the stat bounds treat "unknown_item" as unknown, so a possible
                     Choice Scarf widens the opponent's maximum Speed (and Band, Specs, Vest,
                     Eviolite the other maximums) and a Pokemon known to hold nothing is not widened.
    spe_conditions   Swift Swim, Chlorophyll, Sand Rush, Slush Rush, Surge Surfer double the Speed
                     bounds only while their weather or terrain is up, Unburden only once the item
                     is gone (before: always).
    regenerator      an opposing benched Pokemon whose ability is certainly Regenerator (shown,
                     announced or the only one in its sets) shows the third it healed on switching
                     out; poke-env applies it only when it knows the ability (1,726 pairs too low).

Not here: repairing a broken Illusion (30 breaks in 600 games, 98 wrong HP pairs of 175,818; it
needs poke-env's objects rebuilt, not an observation overlay), an opponent choice-lock input and
durations for screens and Tailwind (those are new inputs, not right values in old slots).

    AbilityObservationAssembler  ObservationAssembler (with the own Tera type option of
                                 obs_tera.py) that writes the fixes after the base has assembled
    make_observation_assembler() the assembler for a set of settings: ObservationAssembler itself
                                 when all are off, so "off" is the code of before
    settings_from_checkpoint()   which of them a learner checkpoint's obs_settings says
    parse_settings()             "all" / "abil_sets,item_unknown" / [] -> the settings

As with the own Tera type, a network gains from these only after it is trained with them: a
checkpoint trained without them reads the new numbers as it read the old ones (it has learned,
for example, that slot 0 empty plus "Volt Absorb" in slots 1-3 means "maybe"), so switching a
setting on for a trained checkpoint at play time without training is a change of input, not a
fix. train.py --obs-abilities switches them on for the rollout workers, the learner writes them
into obs_settings, and hybrid.PolicyNet builds the observation a checkpoint says.

Nothing of this is in obs_assembler.py, obs_pokemon.py, obs_abilities.py or obs_transitions.py,
on purpose (tests/test_ppo_health.py holds obs_assembler.py to release r1; with every setting off
no line of this module runs). tests/test_obs_abil.py holds the bytes.
"""

from __future__ import annotations

import json
import os
import re
from collections import OrderedDict
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from obs_assembler import ObservationAssembler
from obs_pokemon import estimate_stat, _ATK_BOOST_ITEMS, _ATK_DOUBLERS, _DEF_DOUBLERS, _SPA_BOOST_ITEMS, \
    _SPD_BOOST_ITEMS, _SPE_DOUBLERS
from obs_tera import CHECKPOINT_KEY, OwnTeraObservationAssembler
from obs_transitions import _parse_abs_side, _split_args_and_tags, _to_relative_side
from utils import get_id

SETTINGS: Tuple[str, ...] = ("abil_sets", "abil_announced", "abil_history", "item_unknown", "spe_conditions", "regenerator")
TABLE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "randbats_abilities_gen9.json")
UNKNOWN_ITEM = "unknown_item"
_MEMORY_BATTLES = 1024          # battles remembered at once by one assembler (a player plays far fewer)

# The condition under which each Speed-doubling ability is on.
_RAIN = frozenset({"raindance", "primordialsea"})
_SUN = frozenset({"sunnyday", "desolateland"})
_SAND = frozenset({"sandstorm"})
_SNOW = frozenset({"snow", "snowscape", "hail"})


def to_id(s: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


# ---------------------------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------------------------

def parse_settings(value: Any) -> Dict[str, bool]:
    """
    "all", "abil_sets,item_unknown", a list of names, a dict, None or "" -> {name: True} for the
    settings that are on (an empty dict when none). An unknown name raises ValueError: a run
    must never believe a setting is on that is not.
    """
    if value is None or value is False or value == "":
        return {}
    if isinstance(value, dict):
        names = [k for k, v in value.items() if v is True]
    elif isinstance(value, str):
        names = [p.strip() for p in value.split(",") if p.strip()]
    else:
        names = [str(p).strip() for p in value]
    if "all" in names:
        names = [n for n in names if n != "all"] + list(SETTINGS)
    unknown = sorted(set(names) - set(SETTINGS))
    if unknown:
        raise ValueError(f"unknown observation setting(s) {unknown}; known: {', '.join(SETTINGS)} or all")
    return {n: True for n in SETTINGS if n in names}


def settings_from_checkpoint(ckpt: Any) -> Dict[str, bool]:
    """The settings of this module a loaded learner checkpoint says it was trained with ({} for none)."""
    if not isinstance(ckpt, dict):
        return {}
    saved = ckpt.get(CHECKPOINT_KEY)
    if not isinstance(saved, dict):
        return {}
    return {n: True for n in SETTINGS if saved.get(n) is True}


def checkpoint_obs_settings(own_tera_type: bool, abil: Optional[Dict[str, bool]] = None) -> Optional[Dict[str, Any]]:
    """What a run tells its learner to store, or None when nothing is on (the learner saves as before)."""
    out: Dict[str, Any] = {}
    if own_tera_type:
        out["own_tera_type"] = True
    out.update(parse_settings(abil))
    return out or None


def make_observation_assembler(own_tera_type: bool = False, abil: Optional[Dict[str, bool]] = None,
                               **kwargs: Any) -> ObservationAssembler:
    """
    The assembler for a set of settings. Nothing on: ObservationAssembler(**kwargs), the class every
    checkpoint so far was trained on. Only the own Tera type: obs_tera's class. Anything of this
    module: AbilityObservationAssembler (which also does the own Tera type when asked).
    """
    settings = parse_settings(abil)
    if settings:
        return AbilityObservationAssembler(own_tera_type=own_tera_type, settings=settings, **kwargs)
    if own_tera_type:
        return OwnTeraObservationAssembler(own_tera_type=True, **kwargs)
    return ObservationAssembler(**kwargs)


def assembler_class_for(own_tera_type: bool, abil: Dict[str, bool]) -> type:
    """
    A class that worker.py can build with no arguments (it builds every assembler as
    ObservationAssembler(), or (vocab, fast)) and that has these settings.
    """
    settings = parse_settings(abil)
    tera = bool(own_tera_type)

    class ConfiguredAbilityObservationAssembler(AbilityObservationAssembler):
        def __init__(self, vocab: Optional[Dict[str, List[str]]] = None, fast: bool = True):
            super().__init__(vocab, fast, own_tera_type=tera, settings=settings)

    return ConfiguredAbilityObservationAssembler


# ---------------------------------------------------------------------------------------------
# the random battle sets
# ---------------------------------------------------------------------------------------------

_TABLE: Optional[Dict[str, List[str]]] = None

# Formes the protocol does not show: a Greninja-Bond switches in as "Greninja", so poke-env's
# species is "greninja" for both, and the abilities of the hidden forme belong to the shown one
# too (the review's probe: 48 decisions showed Greninja-Bond's ability as Protean with abil_sets).
HIDDEN_FORMES: Dict[str, str] = {"greninjabond": "greninja"}


def _with_hidden_formes(table: Dict[str, List[str]]) -> Dict[str, List[str]]:
    for hidden, shown in HIDDEN_FORMES.items():
        if hidden in table and shown in table:
            table[shown] = table[shown] + [a for a in table[hidden] if a not in table[shown]]
    return table


def set_abilities_table(path: str = TABLE_FILE) -> Dict[str, List[str]]:
    """species id -> the abilities its gen9 random battle sets can have (read once per process)."""
    global _TABLE
    if _TABLE is None or path != TABLE_FILE:
        with open(path, encoding="utf-8") as f:
            table = _with_hidden_formes({to_id(k): [to_id(a) for a in v] for k, v in json.load(f)["abilities"].items()})
        if path != TABLE_FILE:
            return table
        _TABLE = table
    return _TABLE


def set_abilities_of(mon: Any, table: Dict[str, List[str]]) -> Optional[List[str]]:
    """The set abilities of a Pokemon's species, or None when the table does not have the species."""
    species = to_id(getattr(mon, "species", None))
    return table.get(species) if species else None


# ---------------------------------------------------------------------------------------------
# the stat bounds, with the two fixes
# ---------------------------------------------------------------------------------------------

def _spe_doubler_on(ability: str, weather: FrozenSet[str], fields: FrozenSet[str], item: Any) -> bool:
    if ability == "swiftswim":
        return bool(weather & _RAIN)
    if ability == "chlorophyll":
        return bool(weather & _SUN)
    if ability == "sandrush":
        return bool(weather & _SAND)
    if ability == "slushrush":
        return bool(weather & _SNOW)
    if ability == "surgesurfer":
        return "electricterrain" in fields
    if ability == "unburden":
        return not item              # None or "": the item is gone ("unknown_item" is not gone)
    return True


def combat_bounds(mon: Any, stat_name: str, item: Any, ability: Any, possible: Optional[Sequence[str]],
                  item_fix: bool, spe_ctx: Optional[Tuple[FrozenSet[str], FrozenSet[str]]]) -> Tuple[int, int, int]:
    """
    obs_pokemon.get_combat_bounds with what it reads given as arguments (item, confirmed ability,
    possible abilities) and the two fixes as switches. item_fix False, spe_ctx None and the
    Pokemon's own item, ability and possible_abilities give get_combat_bounds' answer
    (tests/test_obs_abil.py holds that).
    """
    base = mon.base_stats.get(stat_name, 100)
    level = mon.level
    min_iv = est_iv = max_iv = 31
    min_ev = est_ev = max_ev = 85
    if stat_name == "atk":
        min_iv, min_ev = 0, 0
        if mon.moves:
            for m in mon.moves.values():
                cat = getattr(m, 'category', None)
                if cat and cat.name == 'PHYSICAL' and m.id not in ('foulplay', 'bodypress', 'rapidspin'):
                    min_iv, min_ev = 31, 85
                    break
    elif stat_name == "spe":
        min_iv, min_ev = 0, 0
        if mon.moves:
            for m in mon.moves.values():
                if m.id in ("trickroom", "gyroball"):
                    max_iv = max_ev = 0
                    est_iv = est_ev = 0
                    break
    base_x2 = 2 * base
    min_stat = int(((base_x2 + min_iv + (min_ev // 4)) * level) / 100) + 5
    est_stat = int(((base_x2 + est_iv + (est_ev // 4)) * level) / 100) + 5
    max_stat = int(((base_x2 + max_iv + (max_ev // 4)) * level) / 100) + 5

    item_unknown = (item == UNKNOWN_ITEM) if item_fix else (not bool(item))
    mult_min = mult_est = mult_max = 1.0

    def doubler(names: FrozenSet[str]) -> None:
        nonlocal mult_min, mult_est, mult_max
        if ability in names:
            if spe_ctx is None or _spe_doubler_on(ability, spe_ctx[0], spe_ctx[1], item):
                mult_min *= 2.0; mult_est *= 2.0; mult_max *= 2.0
        elif possible:
            cands = [a.lower().replace(" ", "") for a in possible]
            if spe_ctx is None:
                if any(a in names for a in cands):
                    mult_max *= 2.0
            elif any(a in names and _spe_doubler_on(a, spe_ctx[0], spe_ctx[1], item) for a in cands):
                mult_max *= 2.0

    if stat_name == "spe":
        if item == "choicescarf":
            mult_min *= 1.5; mult_est *= 1.5; mult_max *= 1.5
        elif item_unknown:
            mult_max *= 1.5
        doubler(_SPE_DOUBLERS)
    elif stat_name == "atk":
        if item in _ATK_BOOST_ITEMS:
            mult_min *= 1.5; mult_est *= 1.5; mult_max *= 1.5
        elif item_unknown:
            mult_max *= 1.5
        if ability in _ATK_DOUBLERS:
            mult_min *= 2.0; mult_est *= 2.0; mult_max *= 2.0
        elif possible and any(a.lower().replace(" ", "") in _ATK_DOUBLERS for a in possible):
            mult_max *= 2.0
    elif stat_name == "spa":
        if item in _SPA_BOOST_ITEMS:
            mult_min *= 1.5; mult_est *= 1.5; mult_max *= 1.5
        elif item_unknown:
            mult_max *= 1.5
    elif stat_name == "spd":
        if item in _SPD_BOOST_ITEMS:
            mult_min *= 1.5; mult_est *= 1.5; mult_max *= 1.5
        elif item_unknown:
            mult_max *= 1.5
    elif stat_name == "def":
        if item == "eviolite":
            mult_min *= 1.5; mult_est *= 1.5; mult_max *= 1.5
        elif item_unknown:
            mult_max *= 1.5
        if ability in _DEF_DOUBLERS:
            mult_min *= 2.0; mult_est *= 2.0; mult_max *= 2.0
        elif possible and any(a.lower().replace(" ", "") in _DEF_DOUBLERS for a in possible):
            mult_max *= 2.0
    return (min(int(min_stat * mult_min), 799), min(int(est_stat * mult_est), 799), min(int(max_stat * mult_max), 799))


# ---------------------------------------------------------------------------------------------
# the lines poke-env does not read
# ---------------------------------------------------------------------------------------------

_IDENT = re.compile(r"^(p[1-4])[a-d]?: ")


def _team_key(ident: Any) -> Optional[Tuple[str, str]]:
    """"p2a: Pelipper" -> ("p2", "p2: Pelipper"), the key poke-env files the Pokemon under."""
    if not isinstance(ident, str):
        return None
    m = _IDENT.match(ident.strip())
    if not m:
        return None
    name = ident.strip()[m.end():]
    return m.group(1), f"{m.group(1)}: {name}"


# "-item|<holder>|<item>|[from] ability: Pickpocket|[of] <robbed>": the line's own Pokemon has the ability.
_OWN_HOLDER_ITEM_ABILITIES = frozenset({"pickpocket", "magician"})


def announcements(parts: Sequence[str]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    One protocol line (split on "|") -> (abilities, items) it announces, each a list of
    (candidate ident's team key, id), the candidates in the order they are tried. Lines poke-env
    reads itself give nothing: "-ability" (whose "[from] ability: Trace" names the Tracer's ability
    and whose [of] is the Pokemon traced), and the items of "-item" and "-enditem" (their "[from]
    ability:" is still read: Frisk, Pickpocket, Magician).

    "[from] ability: X": ONE holder, never a second guess. The [of] Pokemon when the line has one
    (Drizzle, Rough Skin, Frisk), else the line's own Pokemon (Poison Heal). Two kinds of line
    name the holder as their own Pokemon even with an [of]: "-heal" (Water Absorb, Volt Absorb, Dry
    Skin, Earth Eater heal the holder; the [of] is the attacker whose move was absorbed) and
    "-item" from Pickpocket or Magician (the [of] is the Pokemon robbed). The review of stage2 found
    the earlier "[of], then own" order giving their attacker our Water Absorb (bench/STAGE2.md 6).
    "ability: X" as an argument of -activate: the line's own Pokemon.
    "[from] item: X": the [of] Pokemon when there is one (Rocky Helmet), else the line's own
    (Toxic Orb, Life Orb).
    """
    abilities: List[Tuple[str, str]] = []
    items: List[Tuple[str, str]] = []
    if len(parts) < 3:
        return abilities, items
    cmd = parts[1]
    if cmd in ("-ability", "switch", "drag", "replace", "detailschange", "move", "request"):
        return abilities, items
    items_read_by_poke_env = cmd in ("-item", "-enditem")
    own = _team_key(parts[2])
    frm = of = None
    plain: List[str] = []
    for p in parts[3:]:
        if p.startswith("[from]"):
            frm = p[len("[from]"):].strip()
        elif p.startswith("[of]"):
            of = _team_key(p[len("[of]"):].strip())
        elif not p.startswith("["):
            plain.append(p)
    if frm:
        low = frm.lower()
        if low.startswith("ability:"):
            name = to_id(frm.split(":", 1)[1])
            holder = of if of is not None else own
            if cmd == "-heal" or (cmd == "-item" and name in _OWN_HOLDER_ITEM_ABILITIES):
                holder = own
            if holder is not None and name:
                abilities.append((holder[1], name))
        elif low.startswith("item:") and not items_read_by_poke_env:
            name = to_id(frm.split(":", 1)[1])
            holder = of if of is not None else own
            if holder is not None and name:
                items.append((holder[1], name))
    if cmd == "-activate" and own is not None and plain:
        head = plain[0].strip()
        if head.lower().startswith("ability:"):
            abilities.append((own[1], to_id(head.split(":", 1)[1])))
        elif head.lower().startswith("item:"):
            items.append((own[1], to_id(head.split(":", 1)[1])))
    return abilities, items


# ---------------------------------------------------------------------------------------------
# the assembler
# ---------------------------------------------------------------------------------------------

class AbilityObservationAssembler(OwnTeraObservationAssembler):
    """
    ObservationAssembler that writes the fixes switched on in `settings` after the base class (and
    the own Tera type, when own_tera_type) has assembled the observation. Only these places of an
    observation are ever written: a Pokemon's 4 ability ids, its item id, its two fog-of-war flags
    (item unknown, ability unknown), its 15 stat bounds, its 2 HP numbers (Regenerator), and the
    two ability ids of the transition block. Everything else is the base class's bytes.

    abil_announced keeps, per battle, what the lines told so far (the assembler sees each line
    once: every player passes the lines since its last decision); a finished battle is forgotten.

    counts: how often each fix changed something, for a report.
    """

    def __init__(self, vocab: Optional[Dict[str, List[str]]] = None, fast: bool = True, own_tera_type: bool = False,
                 settings: Optional[Dict[str, bool]] = None, table: Optional[Dict[str, List[str]]] = None):
        super().__init__(vocab, fast, own_tera_type=own_tera_type)
        self.settings = parse_settings(settings)
        on = self.settings
        self.abil_sets = bool(on.get("abil_sets"))
        self.abil_announced = bool(on.get("abil_announced"))
        self.abil_history = bool(on.get("abil_history"))
        self.item_unknown = bool(on.get("item_unknown"))
        self.spe_conditions = bool(on.get("spe_conditions"))
        self.regenerator = bool(on.get("regenerator"))
        needs_table = self.abil_sets or self.abil_announced or self.regenerator
        self.table = table if table is not None else (set_abilities_table() if needs_table else {})
        body = self.meta["feature_map"]["body"]
        self._body0 = self.offsets["pokemon_body"][0]
        self._dim = self.pokemon_scalar_dim
        self._hp = body["hp_int"]
        self._hp_abs = body["hp_abs_raw"][0]
        self._stats = body["stats_int"][0]
        self._fog = body["mechanics_raw"][0]
        self._ids0 = self.offsets["pokemon_ids"][0]
        self._abil0 = self.offsets["ability_ids"][0]
        tmeta = self.meta["feature_map"]["transition"]["ability_ids"]
        self._trans_abil = self.offsets["transition_ids"][0] + tmeta[0]
        self._memory: "OrderedDict[str, Dict[str, Dict[str, str]]]" = OrderedDict()
        self.counts = {k: 0 for k in ("observations", "slot0_from_sets", "slots_from_sets", "ability_announced",
                                      "item_announced", "history_ability", "item_flag", "bounds", "regenerator")}

    # -- the per-battle memory of announcements --------------------------------------------------

    def _remember(self, battle: Any, events: Sequence[Sequence[str]]) -> Dict[str, Dict[str, str]]:
        tag = getattr(battle, "battle_tag", None) or id(battle)
        mem = self._memory.get(tag)
        if mem is None:
            mem = {"abilities": {}, "items": {}}
            self._memory[tag] = mem
            while len(self._memory) > _MEMORY_BATTLES:
                self._memory.popitem(last=False)
        else:
            self._memory.move_to_end(tag)
        role = getattr(battle, "player_role", None)
        opp = "p2" if role != "p2" else "p1"
        team = getattr(battle, "opponent_team", {}) or {}
        for parts in events or ():
            if len(parts) < 3:
                continue
            abilities, items = announcements(parts)
            for key, name in abilities:
                mon = team.get(key) if key.startswith(opp + ":") else None
                if mon is None or key in mem["abilities"]:
                    continue                       # not theirs, or already announced: the first announcement stands
                # What the Pokemon can have: its random-battle sets when the table knows the species
                # (a Quagsire whose only set ability is Unaware never takes Water Absorb), else the Pokedex.
                sets = set_abilities_of(mon, self.table)
                allowed = set(sets) if sets else {to_id(a) for a in (getattr(mon, "possible_abilities", None) or ())}
                if name in allowed:
                    mem["abilities"][key] = name
            for key, name in items:
                if key.startswith(opp + ":") and key in team:
                    mem["items"][key] = name
        if getattr(battle, "finished", False):
            self._memory.pop(tag, None)
        return mem

    # -- assembling ------------------------------------------------------------------------------

    def assemble(self, battle: Any, events: List[List[str]] = None):
        events = events if events is not None else []
        obs = super().assemble(battle, events)
        try:
            self._fix(battle, events, obs)
        except Exception:                 # a fix never costs a decision: the base observation stands
            import logging
            logging.getLogger(__name__).exception("obs_abil: a fix failed in %s; the base observation is used",
                                                  getattr(battle, "battle_tag", "?"))
        return obs

    def _fix(self, battle: Any, events: List[List[str]], obs: np.ndarray) -> None:
        counts = self.counts
        counts["observations"] += 1
        mem = self._remember(battle, events) if self.abil_announced else None
        own = list(battle.team.values())[:6]
        opp = list(battle.opponent_team.values())[:6]
        spe_ctx = None
        if self.spe_conditions:
            weather = frozenset(to_id(getattr(w, "name", w)) for w in (getattr(battle, "weather", None) or {}))
            fields = frozenset(to_id(getattr(f, "name", f)) for f in (getattr(battle, "fields", None) or {}))
            spe_ctx = (weather, fields)
        neutralizing = any(to_id(getattr(f, "name", f)) == "neutralizinggas" for f in (getattr(battle, "fields", None) or {}))
        ids = self.vocab_map
        for slot in range(12):
            mon = own[slot] if slot < 6 and slot < len(own) else (opp[slot - 6] if slot >= 6 and slot - 6 < len(opp) else None)
            if mon is None:
                continue
            theirs = slot >= 6
            body = self._body0 + slot * self._dim
            known = getattr(mon, "ability", None)
            item = getattr(mon, "item", None)
            ability = known
            possible = getattr(mon, "possible_abilities", None)
            sets = set_abilities_of(mon, self.table) if self.table else None
            key = None
            if theirs and mem is not None:
                key = next((k for k, v in battle.opponent_team.items() if v is mon), None)
                if not ability and key in mem["abilities"]:
                    ability = mem["abilities"][key]
                    counts["ability_announced"] += 1
                if item == UNKNOWN_ITEM and key in mem["items"]:
                    item = mem["items"][key]
                    obs[self._ids0 + slot * 2 + 1] = get_id(ids, "pokemon.item", item)
                    counts["item_announced"] += 1
            if self.abil_sets and sets:
                if theirs and not ability and len(sets) == 1:
                    ability = sets[0]
                    counts["slot0_from_sets"] += 1
                possible = list(sets)
                start = self._abil0 + slot * 4
                obs[start + 1:start + 4] = 0.0
                for i, name in enumerate(possible[:3]):
                    obs[start + 1 + i] = get_id(ids, "pokemon.ability", name)
                counts["slots_from_sets"] += 1
            if ability != known:
                obs[self._abil0 + slot * 4] = get_id(ids, "pokemon.ability", ability)
                obs[body + self._fog + 1] = 1.0 if not ability else 0.0
            if self.item_unknown:
                flag = 1.0 if item == UNKNOWN_ITEM else 0.0
                if obs[body + self._fog] != flag:
                    counts["item_flag"] += 1
                obs[body + self._fog] = flag
            if self.item_unknown or self.spe_conditions or ability != known or (self.abil_sets and sets) or item != getattr(mon, "item", None):
                at = body + self._stats
                for k, stat in enumerate(("atk", "def", "spa", "spd", "spe")):
                    lo, est, hi = combat_bounds(mon, stat, item, ability, possible, self.item_unknown, spe_ctx)
                    if (obs[at], obs[at + 1], obs[at + 2]) != (lo, est, hi):
                        counts["bounds"] += 1
                    obs[at], obs[at + 1], obs[at + 2] = lo, est, hi
                    at += 3
            if (self.regenerator and theirs and not getattr(mon, "active", False) and not getattr(mon, "fainted", False)
                    and known != "regenerator" and not neutralizing):
                certain = ability if ability else (sets[0] if sets and len(sets) == 1 else None)
                if certain == "regenerator":
                    hp = getattr(mon, "current_hp", None)
                    max_hp = getattr(mon, "max_hp", None)
                    if hp is not None and max_hp:
                        healed = min(int(hp + max_hp / 3), max_hp)
                        if healed != hp:
                            frac = healed / max_hp
                            obs[body + self._hp] = int(frac * 100)
                            obs[body + self._hp_abs] = int(estimate_stat(mon, 'hp') * frac)
                            counts["regenerator"] += 1
        if self.abil_history and events:
            self._history_abilities(battle, events, obs)

    def _history_abilities(self, battle: Any, events: List[List[str]], obs: np.ndarray) -> None:
        """
        obs_transitions writes an ability id only from "ability: X" text; "-ability|POKEMON|X" lines
        carry the bare name. The last ability line of each side wins, as there, and the side and the
        arguments are read with obs_transitions' own helpers.
        """
        role = getattr(battle, "player_role", None)
        self_is_p1 = role not in ("p2",)          # obs_assembler: anything but p1/p2 counts as p1
        last: Dict[int, str] = {}
        for event in events:
            if len(event) < 2 or event[1] not in ("-ability", "-activate"):
                continue
            side = _to_relative_side(_parse_abs_side(event[2] if len(event) > 2 else ""), self_is_p1)
            if side is None:
                continue
            plain, _tags = _split_args_and_tags(event[3:] if len(event) > 3 else [])
            if not plain or not isinstance(plain[0], str):
                continue
            s = plain[0].strip()
            if s.lower().startswith("ability:"):
                s = s.split(":", 1)[1].strip()
            elif event[1] != "-ability":
                continue
            if s:
                last[side] = s
        for side, name in last.items():
            at = self._trans_abil + side
            value = get_id(self.vocab_map, "pokemon.ability", name.lower())
            if obs[at] != value:
                self.counts["history_ability"] += 1
            obs[at] = value
