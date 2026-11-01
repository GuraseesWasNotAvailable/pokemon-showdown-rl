# obs_pokemon.py

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Final, Tuple

import numpy as np
from obs_abilities import IdLookup, count_unknowns, recording_unknowns
from utils import get_id

# Configure logger
logger = logging.getLogger(__name__)

# Constants for coordinate mapping
TEAM_SIZE: Final[int] = 6
TOTAL_SLOTS: Final[int] = 12

def get_pokemon_metadata(vocab_lists: Dict[str, List[str]]) -> Dict[str, Any]:
    v_type = len(vocab_lists.get("pokemon.type", [])) + 1
    v_effect = len(vocab_lists.get("pokemon.effect", [])) + 1
    v_status = len(vocab_lists.get("pokemon.status", [])) + 1
    v_gender = len(vocab_lists.get("pokemon.gender", [])) + 1
    
    curr = 0
    p_map = {
        "hp_int": curr,  # Fractional HP
    }
    curr += 1
    p_map["hp_abs_raw"] = (curr, curr + 2) # Current, Max
    curr += 2
    p_map["stats_int"] = (curr, curr + 15) # 5 stats * (min, est, max)
    curr += 15
    p_map["boosts_raw"] = (curr, curr + 91) # 7 stats * 13 bins
    curr += 91
    p_map["level_int"] = curr
    curr += 1
    p_map["weight_int"] = curr
    curr += 1
    p_map["height_int"] = curr
    curr += 1
    p_map["flags_raw"] = (curr, curr + 12) # Active, Fainted, Tera, Status Counter
    curr += 12
    faint_internal_idx = curr + 1
    p_map["mechanics_raw"] = (curr, curr + 5) # Fog of War block
    curr += 5
    p_map["types_raw"] = (curr, curr + (v_type * 2))
    curr += (v_type * 2)
    p_map["effects_raw"] = (curr, curr + v_effect)
    curr += v_effect
    p_map["status_raw"] = (curr, curr + v_status)
    curr += v_status
    p_map["gender_raw"] = (curr, curr + v_gender)
    curr += v_gender
    p_map["pos_raw"] = (curr, curr + 12) # Mon Identity
    curr += 12

    return {
        "scalar_dim": curr,
        "feature_map": p_map,
        "faint_internal_idx": faint_internal_idx
    }

def estimate_stat(mon: Any, stat_name: str) -> int:
    """
    Estimates raw stats for Random Battles where exact IVs/EVs are hidden.
    Includes Showdown's hardcoded HP hazard/recoil optimizations.
    """
    base = mon.base_stats.get(stat_name, 100)
    level = mon.level
    
    iv, ev = 31, 84
    nature_mult = 1.0 

    if stat_name == "spe":
        for move_id in (mon.moves or {}):
            if move_id in ["trickroom", "gyroball"]:
                iv, ev, nature_mult = 0, 0, 0.9
                break

    if stat_name == "hp":
        return int(((2 * base + iv + (ev // 4)) * level) / 100) + level + 10

    raw_stat = int(((2 * base + iv + (ev // 4)) * level) / 100) + 5
    return int(raw_stat * nature_mult)

# ---------------------------------------------------------
# MODULE-LEVEL CONSTANTS
# ---------------------------------------------------------
_EXCLUDED_PHYSICAL = frozenset({'foulplay', 'bodypress', 'rapidspin'})
_SPE_DOUBLERS = frozenset({"swiftswim", "chlorophyll", "sandrush", "slushrush", "surgesurfer", "unburden"})
_ATK_DOUBLERS = frozenset({"hugepower", "purepower"})
_DEF_DOUBLERS = frozenset({"furcoat"})
_ATK_BOOST_ITEMS = frozenset({"choiceband", "thickclub", "lightball"})
_SPA_BOOST_ITEMS = frozenset({"choicespecs", "lightball"})
_SPD_BOOST_ITEMS = frozenset({"assaultvest", "eviolite"})

def get_combat_bounds(mon: Any, stat_name: str) -> Tuple[int, int, int]:
    base = mon.base_stats.get(stat_name, 100)
    level = mon.level
    
    # ---------------------------------------------------------
    # 1. BASE ASSUMPTIONS (Gen 9 Randbats Determinism)
    # ---------------------------------------------------------
    min_iv = est_iv = max_iv = 31
    min_ev = est_ev = max_ev = 85

    # ---------------------------------------------------------
    # 2. EVIDENCE-BASED TIGHTENING (Short-circuited for speed)
    # ---------------------------------------------------------
    if stat_name == "atk":
        min_iv, min_ev = 0, 0 
        
        # TIGHTEN: Exclude physical moves that don't scale with the user's Attack
        if mon.moves:
            for m in mon.moves.values():
                cat = getattr(m, 'category', None)
                if cat and cat.name == 'PHYSICAL' and m.id not in _EXCLUDED_PHYSICAL:
                    min_iv, min_ev = 31, 85
                    break  # Found evidence, stop looping

    elif stat_name == "spe":
        min_iv, min_ev = 0, 0 
        
        # TIGHTEN: If we see a slow move, we rule out standard Speed.
        if mon.moves:
            for m in mon.moves.values():
                if m.id in ("trickroom", "gyroball"):
                    max_iv = max_ev = 0
                    est_iv = est_ev = 0
                    break  # Found evidence, stop looping

    # ---------------------------------------------------------
    # 3. RAW INTEGER CALCULATION (Inlined to avoid function overhead)
    # ---------------------------------------------------------
    base_x2 = 2 * base
    min_stat = int(((base_x2 + min_iv + (min_ev // 4)) * level) / 100) + 5
    est_stat = int(((base_x2 + est_iv + (est_ev // 4)) * level) / 100) + 5
    max_stat = int(((base_x2 + max_iv + (max_ev // 4)) * level) / 100) + 5

    # ---------------------------------------------------------
    # 4. VOLATILE MULTIPLIERS (Items & Abilities Grouped by Stat)
    # ---------------------------------------------------------
    item = getattr(mon, 'item', None)
    item_unknown = not bool(item) 
    
    confirmed_ability = getattr(mon, 'ability', None)

    mult_min = mult_est = mult_max = 1.0

    # Grouped logic prevents double-checking the 'stat_name' variable
    if stat_name == "spe":
        if item == "choicescarf": 
            mult_min *= 1.5; mult_est *= 1.5; mult_max *= 1.5
        elif item_unknown: 
            mult_max *= 1.5
            
        if confirmed_ability in _SPE_DOUBLERS: 
            mult_min *= 2.0; mult_est *= 2.0; mult_max *= 2.0
        else:
            poss_abs = getattr(mon, 'possible_abilities', None)
            if poss_abs and any(a.lower().replace(" ", "") in _SPE_DOUBLERS for a in poss_abs): 
                mult_max *= 2.0

    elif stat_name == "atk":
        if item in _ATK_BOOST_ITEMS: 
            mult_min *= 1.5; mult_est *= 1.5; mult_max *= 1.5
        elif item_unknown: 
            mult_max *= 1.5
            
        if confirmed_ability in _ATK_DOUBLERS: 
            mult_min *= 2.0; mult_est *= 2.0; mult_max *= 2.0
        else:
            poss_abs = getattr(mon, 'possible_abilities', None)
            if poss_abs and any(a.lower().replace(" ", "") in _ATK_DOUBLERS for a in poss_abs): 
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
            
        if confirmed_ability in _DEF_DOUBLERS: 
            mult_min *= 2.0; mult_est *= 2.0; mult_max *= 2.0
        else:
            poss_abs = getattr(mon, 'possible_abilities', None)
            if poss_abs and any(a.lower().replace(" ", "") in _DEF_DOUBLERS for a in poss_abs): 
                mult_max *= 2.0

    return (
        min(int(min_stat * mult_min), 799), 
        min(int(est_stat * mult_est), 799), 
        min(int(max_stat * mult_max), 799)
    )

def encode_pokemon_body_inplace(
    mon: Optional[Any], 
    buffer: np.ndarray, 
    mon_idx: int, 
    scalar_dim: int, 
    offsets: Dict[str, Tuple[int, int]], 
    vocab: Dict[str, Dict[str, int]], 
    vocab_lists: Dict[str, List[str]], 
) -> None:
    """
    Encodes a Pokémon's full state directly into the pre-allocated calculation buffer.
    """
    body_base, _ = offsets["pokemon_body"]
    body_start = body_base + (mon_idx * scalar_dim)
    
    pos_start = body_start + (scalar_dim - TOTAL_SLOTS)
    if 0 <= mon_idx < TOTAL_SLOTS:
        buffer[pos_start + mon_idx] = 1.0
    
    id_base, _ = offsets["pokemon_ids"]
    id_start = id_base + (mon_idx * 2)

    if mon is None:
        return

    # 1. Identity Mappings
    buffer[id_start] = get_id(vocab, "pokemon.species", mon.species)
    buffer[id_start + 1] = get_id(vocab, "pokemon.item", mon.item)

    # 2. Health Representation
    curr = body_start
    buffer[curr] = int(mon.current_hp_fraction * 100)
    curr += 1

    # 3. Absolute Health
    max_hp = estimate_stat(mon, 'hp')
    buffer[curr] = int(max_hp * mon.current_hp_fraction)  # Current Absolute HP
    buffer[curr + 1] = max_hp                             # Max Absolute HP
    curr += 2

    # 4. Estimated Base Stats (Combat Bounds)
    combat_stats = ['atk', 'def', 'spa', 'spd', 'spe']
    for key in combat_stats:
        min_s, est_s, max_s = get_combat_bounds(mon, key)
        buffer[curr] = min_s
        buffer[curr + 1] = est_s
        buffer[curr + 2] = max_s
        curr += 3

    # 4. Stat Boosts (7 stats * 13 bins)
    boost_keys = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
    for key in boost_keys:
        stage = mon.boosts.get(key, 0)
        idx = stage + 6 
        buffer[curr + idx] = 1.0
        curr += 13
        
    # 5. Numerical Metadata
    buffer[curr] = int(mon.level) 
    buffer[curr + 1] = int(np.clip(np.log10(max(0.1, mon.weight)) * 5, 0, 20)) 
    buffer[curr + 2] = int(np.clip(mon.height * 10, 0, 200)) 
    curr += 3

    # 6. Boolean Flags & Status Counter (12-bin block)
    buffer[curr] = 1.0 if mon.active else 0.0
    buffer[curr + 1] = 1.0 if mon.fainted else 0.0
    buffer[curr + 2] = 1.0 if (getattr(mon, 'terastallized', False) or getattr(mon, 'is_terastallized', False)) else 0.0
    sc = int(np.clip(mon.status_counter, 0, 8))
    buffer[curr + 3 + sc] = 1.0
    curr += 12
    
    # 6.5 Fog of War & Turn Mechanics
    buffer[curr] = 1.0 if not mon.item else 0.0
    buffer[curr + 1] = 1.0 if not mon.ability else 0.0
    buffer[curr + 2] = float(max(0, 4 - len(mon.moves or {})))
    buffer[curr + 3] = float(getattr(mon, 'protect_counter', 0))
    buffer[curr + 4] = 1.0 if getattr(mon, 'first_turn', False) else 0.0
    curr += 5

    # 7. Type Encoding
    type_list = vocab_lists.get("pokemon.type", []) # Safely fetch list
    type_block_size = len(type_list) + 1 
    
    # Block 1: Active Types
    for t in [mon.type_1, mon.type_2]:
        if t:
            t_idx = get_id(vocab, "pokemon.type", t.name.lower())
            if t_idx > 0:
                buffer[curr + t_idx] = 1.0
    curr += type_block_size
    
    # Block 2: Tera Type
    if mon.tera_type:
        t_idx = get_id(vocab, "pokemon.type", mon.tera_type.name.lower())
        if t_idx > 0:
            buffer[curr + t_idx] = 1.0
    curr += type_block_size

    # 8. Volatile Effects
    v_vocab = vocab_lists.get("pokemon.effect", []) # Safely fetch list
    if mon.effects:
        for effect in mon.effects:
            v_idx = get_id(vocab, "pokemon.effect", effect)
            if v_idx > 0:
                buffer[curr + v_idx] = 1.0
    curr += (len(v_vocab) + 1)
    
    # 9. Persistent Status
    s_vocab = vocab_lists.get("pokemon.status", []) # Safely fetch list
    if mon.status:
        s_idx = get_id(vocab, "pokemon.status", mon.status.name.lower())
        if s_idx > 0:
            buffer[curr + s_idx] = 1.0
    curr += (len(s_vocab) + 1)

    # 10. Gender Encoding - NEW
    g_vocab = vocab_lists.get("pokemon.gender", [])
    if mon.gender:
        g_idx = get_id(vocab, "pokemon.gender", getattr(mon.gender, 'name', str(mon.gender)).lower())
        if g_idx > 0:
            buffer[curr + g_idx] = 1.0


# ---------------------------------------------------------------------------
# THE SAME ENCODING WITHOUT THE REPEATED WORK
# ---------------------------------------------------------------------------

_FLOAT16 = np.dtype(np.float16)
_COMBAT_STATS: Final[Tuple[str, ...]] = ('atk', 'def', 'spa', 'spd', 'spe')

# Answers of the helpers above, kept by everything the answer depends on. Each is filled by
# running the original expression, so what comes out of the table is what the expression gave.
_WEIGHT_BIN: Dict[Any, int] = {}          # weight -> int(np.clip(np.log10(max(0.1, weight)) * 5, 0, 20))
_HEIGHT_BIN: Dict[Any, int] = {}          # height -> int(np.clip(height * 10, 0, 200))
_STATUS_COUNTER_BIN: Dict[Any, int] = {}  # counter -> int(np.clip(counter, 0, 8))
_MOVE_EVIDENCE: Dict[Any, Tuple[bool, bool]] = {}    # move -> (shows a physical attacker, shows a slow one)
_STAT_BLOCK: Dict[Any, Tuple[Any, np.ndarray]] = {}  # see PokemonBodyEncoder._stats
_STAT_BLOCK_LIMIT: Final[int] = 50_000    # states of a Pokemon are many but not endless; start over when full


def _move_evidence(move: Any) -> Tuple[bool, bool]:
    """
    What get_combat_bounds reads off one revealed move: whether it is a physical move that
    scales with the user's Attack, and whether it is one only a slow Pokemon runs.
    """
    key = (move.__class__, move.id, move.gen)
    hit = _MOVE_EVIDENCE.get(key)
    if hit is None:
        cat = getattr(move, 'category', None)
        hit = (bool(cat and cat.name == 'PHYSICAL' and move.id not in _EXCLUDED_PHYSICAL),
               move.id in ("trickroom", "gyroball"))
        _MOVE_EVIDENCE[key] = hit
    return hit


class PokemonBodyEncoder:
    """
    encode_pokemon_body_inplace for one vocabulary and one buffer layout.

    It writes the same numbers to the same places. The difference is that each property of
    the Pokemon is read once, and that the parts which come out the same decision after
    decision are looked up, not recomputed:

      * the maximum HP and the fifteen stat bounds, kept under everything estimate_stat and
        get_combat_bounds read (the six base stats, the level, the item, the ability, the
        possible abilities, and the two things they look for in the revealed moves). On a
        miss the two functions above are called on the Pokemon itself.
      * the weight, height and status-counter bins, kept by their input, computed with the
        expressions of the function above (np.clip on a scalar costs 3 microseconds).
      * the vocabulary ids of species, item, types, status, gender and effects.

    The buffer must be float16, as the assembler's is. Any other buffer goes to the
    function above.
    """

    def __init__(self, scalar_dim: int, offsets: Dict[str, Tuple[int, int]],
                 vocab: Dict[str, Dict[str, int]], vocab_lists: Dict[str, List[str]]):
        self._scalar_dim = scalar_dim
        self._offsets = offsets
        self._vocab = vocab
        self._vocab_lists = vocab_lists
        body_base = offsets["pokemon_body"][0]
        id_base = offsets["pokemon_ids"][0]
        self._body = [body_base + i * scalar_dim for i in range(TOTAL_SLOTS)]
        self._pos = [body_base + i * scalar_dim + (scalar_dim - TOTAL_SLOTS) + i for i in range(TOTAL_SLOTS)]
        self._ids = [id_base + i * 2 for i in range(TOTAL_SLOTS)]
        self._type_block = len(vocab_lists.get("pokemon.type", [])) + 1
        self._effect_block = len(vocab_lists.get("pokemon.effect", [])) + 1
        self._status_block = len(vocab_lists.get("pokemon.status", [])) + 1
        self._species_id = IdLookup(vocab, "pokemon.species")
        self._item_id = IdLookup(vocab, "pokemon.item")
        self._effect_id = IdLookup(vocab, "pokemon.effect")
        self._type_ids: Dict[Any, Tuple[int, Tuple[str, ...]]] = {}
        self._status_ids: Dict[Any, Tuple[int, Tuple[str, ...]]] = {}
        self._gender_ids: Dict[Any, Tuple[int, Tuple[str, ...]]] = {}

    def _named_id(self, cache: Dict[Any, Tuple[int, Tuple[str, ...]]], path: str, thing: Any) -> int:
        """get_id(vocab, path, <the lower-case name of an enum member>), remembered per member."""
        hit = cache.get(thing)
        if hit is None:
            with recording_unknowns() as seen:
                answer = get_id(self._vocab, path, thing.name.lower())
            cache[thing] = (answer, tuple(seen.elements()))
            return answer
        if hit[1]:
            count_unknowns(hit[1])
        return hit[0]

    def _stats(self, mon: Any, moves: Any, item: Any, ability: Any) -> Tuple[Any, np.ndarray]:
        """
        (maximum HP, [maximum HP and the min/estimate/max of the five combat stats] as float16).

        The key is every input of estimate_stat(mon, 'hp') and get_combat_bounds(mon, stat):
        nothing else about the Pokemon reaches those two functions.
        """
        bs = mon.base_stats
        level = mon.level
        physical = slow = False
        if moves:
            for move in moves.values():
                p, s = _move_evidence(move)
                physical = physical or p
                slow = slow or s
        possible = getattr(mon, 'possible_abilities', None)
        key = (bs.get('hp', 100), bs.get('atk', 100), bs.get('def', 100), bs.get('spa', 100), bs.get('spd', 100),
               bs.get('spe', 100), level, physical, slow, item, ability,
               tuple(possible) if possible.__class__ is list else possible)
        hit = _STAT_BLOCK.get(key)
        if hit is None:
            max_hp = estimate_stat(mon, 'hp')
            block = np.zeros(16, dtype=np.float16)
            block[0] = max_hp
            at = 1
            for stat in _COMBAT_STATS:
                min_s, est_s, max_s = get_combat_bounds(mon, stat)
                block[at] = min_s
                block[at + 1] = est_s
                block[at + 2] = max_s
                at += 3
            hit = (max_hp, block)
            if len(_STAT_BLOCK) >= _STAT_BLOCK_LIMIT:
                _STAT_BLOCK.clear()
            _STAT_BLOCK[key] = hit
        return hit

    def encode(self, mon: Optional[Any], buffer: np.ndarray, mon_idx: int) -> None:
        if not 0 <= mon_idx < TOTAL_SLOTS or getattr(buffer, "dtype", None) != _FLOAT16:
            return encode_pokemon_body_inplace(mon, buffer, mon_idx, self._scalar_dim, self._offsets, self._vocab, self._vocab_lists)

        buffer[self._pos[mon_idx]] = 1.0
        if mon is None:
            return

        # 1. Identity Mappings
        id_start = self._ids[mon_idx]
        item = mon.item
        buffer[id_start] = self._species_id(mon.species)
        buffer[id_start + 1] = self._item_id(item)

        # 2, 3. Health, 4. Estimated stats
        curr = self._body[mon_idx]
        hp_fraction = mon.current_hp_fraction
        moves = mon.moves
        ability = getattr(mon, 'ability', None)
        try:
            max_hp, block = self._stats(mon, moves, item, ability)
        except (TypeError, AttributeError):
            # Something in the key cannot be hashed, or a move lacks what a poke-env move has.
            # Not a Pokemon as poke-env makes them: the function above works it out the long
            # way (and raises what it always raised if the Pokemon itself is broken).
            return encode_pokemon_body_inplace(mon, buffer, mon_idx, self._scalar_dim, self._offsets, self._vocab, self._vocab_lists)
        buffer[curr] = int(hp_fraction * 100)
        buffer[curr + 1] = int(max_hp * hp_fraction)
        buffer[curr + 2:curr + 18] = block
        curr += 18

        # 4. Stat Boosts (7 stats * 13 bins)
        boosts = mon.boosts
        buffer[curr + boosts.get("atk", 0) + 6] = 1.0
        buffer[curr + 13 + boosts.get("def", 0) + 6] = 1.0
        buffer[curr + 26 + boosts.get("spa", 0) + 6] = 1.0
        buffer[curr + 39 + boosts.get("spd", 0) + 6] = 1.0
        buffer[curr + 52 + boosts.get("spe", 0) + 6] = 1.0
        buffer[curr + 65 + boosts.get("accuracy", 0) + 6] = 1.0
        buffer[curr + 78 + boosts.get("evasion", 0) + 6] = 1.0
        curr += 91

        # 5. Numerical Metadata
        buffer[curr] = int(mon.level)
        weight = mon.weight
        w_bin = _WEIGHT_BIN.get(weight)
        if w_bin is None:
            w_bin = _WEIGHT_BIN[weight] = int(np.clip(np.log10(max(0.1, weight)) * 5, 0, 20))
        buffer[curr + 1] = w_bin
        height = mon.height
        h_bin = _HEIGHT_BIN.get(height)
        if h_bin is None:
            h_bin = _HEIGHT_BIN[height] = int(np.clip(height * 10, 0, 200))
        buffer[curr + 2] = h_bin
        curr += 3

        # 6. Boolean Flags & Status Counter (12-bin block)
        buffer[curr] = 1.0 if mon.active else 0.0
        buffer[curr + 1] = 1.0 if mon.fainted else 0.0
        buffer[curr + 2] = 1.0 if (getattr(mon, 'terastallized', False) or getattr(mon, 'is_terastallized', False)) else 0.0
        counter = mon.status_counter
        sc = _STATUS_COUNTER_BIN.get(counter)
        if sc is None:
            sc = _STATUS_COUNTER_BIN[counter] = int(np.clip(counter, 0, 8))
        buffer[curr + 3 + sc] = 1.0
        curr += 12

        # 6.5 Fog of War & Turn Mechanics
        buffer[curr] = 1.0 if not item else 0.0
        buffer[curr + 1] = 1.0 if not ability else 0.0
        buffer[curr + 2] = float(max(0, 4 - len(moves or {})))
        buffer[curr + 3] = float(getattr(mon, 'protect_counter', 0))
        buffer[curr + 4] = 1.0 if getattr(mon, 'first_turn', False) else 0.0
        curr += 5

        # 7. Type Encoding
        type_ids = self._type_ids
        for t in (mon.type_1, mon.type_2):
            if t:
                t_idx = self._named_id(type_ids, "pokemon.type", t)
                if t_idx > 0:
                    buffer[curr + t_idx] = 1.0
        curr += self._type_block

        tera_type = mon.tera_type
        if tera_type:
            t_idx = self._named_id(type_ids, "pokemon.type", tera_type)
            if t_idx > 0:
                buffer[curr + t_idx] = 1.0
        curr += self._type_block

        # 8. Volatile Effects
        effects = mon.effects
        if effects:
            effect_id = self._effect_id
            for effect in effects:
                v_idx = effect_id(effect)
                if v_idx > 0:
                    buffer[curr + v_idx] = 1.0
        curr += self._effect_block

        # 9. Persistent Status
        status = mon.status
        if status:
            s_idx = self._named_id(self._status_ids, "pokemon.status", status)
            if s_idx > 0:
                buffer[curr + s_idx] = 1.0
        curr += self._status_block

        # 10. Gender Encoding
        gender = mon.gender
        if gender:
            hit = self._gender_ids.get((gender.__class__, gender))
            if hit is None:
                with recording_unknowns() as seen:
                    g_idx = get_id(self._vocab, "pokemon.gender", getattr(gender, 'name', str(gender)).lower())
                self._gender_ids[(gender.__class__, gender)] = (g_idx, tuple(seen.elements()))
            else:
                if hit[1]:
                    count_unknowns(hit[1])
                g_idx = hit[0]
            if g_idx > 0:
                buffer[curr + g_idx] = 1.0
