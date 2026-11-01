"""
Move Encoder for Pokémon Showdown Reinforcement Learning.

This module encodes the 4-move set of a Pokémon into a structured numerical format.
It captures discrete properties (ID, Type, Category, Target) and continuous/ordinal 
properties (Power, Accuracy, PP, Priority).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Final, Tuple

import numpy as np
from obs_abilities import IdLookup, count_unknowns, recording_unknowns
from utils import get_id

# Configure logger
logger = logging.getLogger(__name__)

# Constants for schema layout
MOVES_PER_MON: Final[int] = 4
PRIORITY_OFFSET: Final[int] = 6  # Shifts priority range [-6, +6] to [0, 12]
PRIORITY_BINS: Final[int] = 13   # Total discrete priority levels

def get_move_metadata(vocab_lists: Dict[str, List[str]]) -> Dict[str, Any]:
    """Source of truth for move encoding layout."""
    v_type = len(vocab_lists.get("pokemon.type", [])) + 1
    v_cat = len(vocab_lists.get("move.category", [])) + 1
    v_target = len(vocab_lists.get("move.target", [])) + 1
    v_status = len(vocab_lists.get("pokemon.status", [])) + 1
    
    curr = 19
    m_map = {
        "acc_int": 0, "pwr_int": 1, "pp_int": 2,
        "onehots_raw": (3, 19),
        "type_raw": (curr, curr + v_type)
    }
    curr += v_type
    m_map["category_raw"] = (curr, curr + v_cat)
    curr += v_cat
    m_map["target_raw"] = (curr, curr + v_target)
    curr += v_target
    m_map["stab_flag"] = curr
    curr += 1
    m_map["status_raw"] = (curr, curr + v_status)
    curr += v_status
    m_map["status_prob"] = curr
    curr += 1
    m_map["owner_raw"] = (curr, curr + 12)
    curr += 12
    m_map["slot_raw"] = (curr, curr + 4)
    curr += 4
    m_map["is_available"] = curr
    curr += 1
    m_map["expected_hits"] = curr
    curr += 1
    m_map["self_boost_sum"] = curr
    curr += 1
    
    m_map["flags_end"] = curr
    
    return {
        "single_move_dim": curr,
        "feature_map": m_map
    }

def get_move_scalar_dim(vocab_lists: Dict[str, List[str]]) -> int:
    """Wrapper for legacy compatibility; uses modular metadata."""
    return get_move_metadata(vocab_lists)["single_move_dim"]

def get_accuracy_int(move: Any) -> int:
    """
    Normalizes move accuracy to a 0-100 integer.
    """
    acc = getattr(move, "accuracy", 100)
    
    if acc is True:
        return 100
    
    if isinstance(acc, (int, float)):
        return int(acc) if acc > 1.0 else int(acc * 100)
        
    return 100

def encode_moves_inplace(
    mon: Optional[Any], 
    buffer: Any, 
    mon_idx: int, 
    scalar_dim: int, 
    offsets: Dict[str, tuple[int, int]], 
    vocab: Dict[str, Dict[str, int]], 
    vocab_lists: Dict[str, List[str]], 
    is_self: bool = True, 
    available_move_ids: Optional[List[str]] = None
) -> None:
    # 1. Initialize metadata and global offsets
    m_meta = get_move_metadata(vocab_lists)["feature_map"]
    single_move_dim = get_move_metadata(vocab_lists)["single_move_dim"]
    
    id_base, _ = offsets["move_ids"]
    sc_base, _ = offsets["move_scalars"]
    id_start_mon = id_base + (mon_idx * MOVES_PER_MON)
    sc_start_mon = sc_base + (mon_idx * scalar_dim)

    # Pre-extract revealed moves to maintain slot order (0-3)
    revealed_moves = list(mon.moves.values()) if mon else []

    # 2. Iterate exactly 4 slots to guarantee Transformer sequence stability
    for m_idx in range(MOVES_PER_MON):
        s = sc_start_mon + (m_idx * single_move_dim)
        move = revealed_moves[m_idx] if m_idx < len(revealed_moves) else None

        # --- A. Positional Identity (MANDATORY) ---
        # Set these first so even empty/unknown slots have coordinates
        buffer[s + m_meta["owner_raw"][0] + mon_idx] = 1.0
        buffer[s + m_meta["slot_raw"][0] + m_idx] = 1.0

        # --- B. Categorical ID Decision ---
        if move:
            buffer[id_start_mon + m_idx] = get_id(vocab, "move.id", move.id)
        elif mon:
            buffer[id_start_mon + m_idx] = get_id(vocab, "move.id", "unknown")
            continue # Remaining scalars for unrevealed move stay 0.0
        else:
            buffer[id_start_mon + m_idx] = 0 # Empty slot
            continue # Remaining scalars stay 0.0

        # --- C. Absolute Offset Scalar Encoding ---
        # 1. Core Stats
        buffer[s + m_meta["acc_int"]] = get_accuracy_int(move)
        buffer[s + m_meta["pwr_int"]] = int(move.base_power)
        buffer[s + m_meta["pp_int"]] = int(move.current_pp)
        
        buffer[s + m_meta["expected_hits"]] = float(getattr(move, "expected_hits", 1.0))
        
        boost_sum = 0
        if move.boosts and move.target in ("self", "allAlly"):
            boost_sum = sum(move.boosts.values())
        buffer[s + m_meta["self_boost_sum"]] = float(max(0, min(10, boost_sum)))

        # 2. Priority One-Hot (s + 6 + offset)
        prio_idx = int(move.priority + PRIORITY_OFFSET)
        if 0 <= prio_idx < PRIORITY_BINS:
            buffer[s + 6 + prio_idx] = 1.0

        # 3. Dynamic Type, Category, Target
        if move.type:
            t_idx = get_id(vocab, "pokemon.type", move.type.name.lower())
            if t_idx > 0: buffer[s + m_meta["type_raw"][0] + t_idx] = 1.0
        if move.category:
            c_idx = get_id(vocab, "move.category", move.category.name.lower())
            if c_idx > 0: buffer[s + m_meta["category_raw"][0] + c_idx] = 1.0
        if move.target:
            tg_idx = get_id(vocab, "move.target", move.target.name.lower())
            if tg_idx > 0: buffer[s + m_meta["target_raw"][0] + tg_idx] = 1.0

        # 4. STAB & Status Engine (Fixed extraction)
        stab_pool = [mon.type_1, mon.type_2]
        
        if getattr(mon, 'terastallized', False) and getattr(mon, 'tera_type', None):
            # Defensive types are overridden, but offensive STAB expands.
            stab_pool.append(mon.tera_type)
            
        buffer[s + m_meta["stab_flag"]] = 1.0 if move.type in stab_pool else 0.0

        status_id, chance = None, 0
        if move.status:
            status_id, chance = move.status.name.lower(), 100
        elif move.secondary:
            for effect in move.secondary:
                if "status" in effect:
                    st_obj = effect["status"]
                    status_id = st_obj.name.lower() if hasattr(st_obj, "name") else str(st_obj).lower()
                    chance = effect.get("chance", 100)
                    break
        
        if status_id:
            st_idx = get_id(vocab, "pokemon.status", status_id)
            if st_idx > 0: buffer[s + m_meta["status_raw"][0] + st_idx] = 1.0
        buffer[s + m_meta["status_prob"]] = float(chance)

        # 5. Active Availability Flag
        is_active = False
        if getattr(mon, 'active', False):
            if not is_self: is_active = True
            elif available_move_ids and move.id in available_move_ids: is_active = True
        buffer[s + m_meta["is_available"]] = 1.0 if is_active else 0.0


# ---------------------------------------------------------------------------
# THE SAME ENCODING FROM A TABLE
# ---------------------------------------------------------------------------

class _OneMove:
    """
    A stand-in Pokemon that knows one move and nothing else: no types, not active. Running
    encode_moves_inplace on it gives what that function writes for the move whoever holds it.
    """
    __slots__ = ("moves",)
    type_1 = None
    type_2 = None
    active = False

    def __init__(self, move: Any):
        self.moves = {0: move}


_FLOAT16 = np.dtype(np.float16)


class MoveEncoder:
    """
    encode_moves_inplace for one vocabulary and one buffer layout, with a table in place of
    the per-decision lookups.

    Of the 85 numbers a move takes, all but five depend on the move alone: accuracy, power,
    priority, type, category, target, status and its chance, expected hits, the boost sum.
    The function above reads each of them from poke-env's move table through a property, for
    every one of up to 48 moves, at every decision: 620 lookups a decision that return what
    they returned the decision before. Here that row is produced once per move, by running
    the function above on that move alone, and copied from then on. Left to do per decision
    are the five numbers that do change: PP, the STAB flag, whether the move can be chosen
    now, and the owner and slot markers.

    What makes two moves the same row is the key in encode(): the move's class, its id, the
    generation, and its base power as poke-env reports it. The last is there because poke-env
    keeps a power of its own on Hidden Power variants, which the id does not show.

    The buffer's move block must be zero on entry (the assembler clears its buffer first),
    and the buffer must be float16: a row holds float16 and a wider buffer would be given
    the rounded value where the function above gives it the exact one. Any other buffer, and
    anything else this class is not sure about, goes to the function above.
    """

    def __init__(self, scalar_dim: int, offsets: Dict[str, Tuple[int, int]],
                 vocab: Dict[str, Dict[str, int]], vocab_lists: Dict[str, List[str]]):
        self._scalar_dim = scalar_dim
        self._offsets = offsets
        self._vocab = vocab
        self._vocab_lists = vocab_lists

        meta = get_move_metadata(vocab_lists)
        m = meta["feature_map"]
        single = meta["single_move_dim"]
        self._single = single
        self._block = MOVES_PER_MON * single
        self._id_start = [offsets["move_ids"][0] + i * MOVES_PER_MON for i in range(12)]
        self._sc_start = [offsets["move_scalars"][0] + i * scalar_dim for i in range(12)]
        owner_start, owner_end = m["owner_raw"]
        slot_start, slot_end = m["slot_raw"]
        self._head = owner_start            # a row's numbers before the owner marker
        self._tail = slot_end               # and after the slot marker
        self._pp, self._stab, self._avail = m["pp_int"], m["stab_flag"], m["is_available"]

        # The copy below writes [0, head) and [tail, single) of a slot and leaves the markers
        # in between alone. That is only right while the layout keeps the changing numbers
        # where they are today, and while the three slots the stand-in leaves empty do not
        # count "unknown" as a name missing from the vocabulary.
        self._fast = (
            owner_end == slot_start and self._pp < owner_start and self._stab < owner_start
            and slot_end <= self._avail < single and self._block <= scalar_dim
            and "unknown" in vocab.get("move.id", {})
        )

        # What a Pokemon with no known move gets: the owner and slot markers of its four slots.
        self._blank = []
        for mon_idx in range(12):
            blank = np.zeros(self._block, dtype=np.float16)
            for k in range(MOVES_PER_MON):
                blank[k * single + owner_start + mon_idx] = 1.0
                blank[k * single + slot_start + k] = 1.0
            self._blank.append(blank)

        self._rows: Dict[Any, tuple] = {}
        self._move_id = IdLookup(vocab, "move.id")
        self._probe_offsets = {"move_ids": (0, MOVES_PER_MON), "move_scalars": (MOVES_PER_MON, MOVES_PER_MON + scalar_dim)}
        self._probe = np.zeros(MOVES_PER_MON + scalar_dim, dtype=np.float16)

    def _slow(self, mon, buffer, mon_idx, is_self, available_move_ids) -> None:
        encode_moves_inplace(mon, buffer, mon_idx, self._scalar_dim, self._offsets, self._vocab, self._vocab_lists,
                             is_self=is_self, available_move_ids=available_move_ids)

    def _row(self, move: Any, key: Any) -> tuple:
        """(id, numbers before the markers, numbers after them, type, unknown names counted) for a move."""
        probe = self._probe
        probe.fill(0)
        with recording_unknowns() as seen:
            encode_moves_inplace(_OneMove(move), probe, 0, self._scalar_dim, self._probe_offsets,
                                 self._vocab, self._vocab_lists, is_self=True, available_move_ids=None)
        row = probe[MOVES_PER_MON:MOVES_PER_MON + self._single]
        head = row[:self._head].copy()
        tail = row[self._tail:].copy()
        # The stand-in holds full PP, has no types and is not active: these three are written
        # per decision.
        head[self._pp] = 0
        head[self._stab] = 0
        tail[self._avail - self._tail] = 0
        entry = (probe[0], head, tail, move.type, tuple(seen.elements()))
        self._rows[key] = entry
        return entry

    def encode(self, mon: Optional[Any], buffer: Any, mon_idx: int, is_self: bool = True,
               available_move_ids: Optional[List[str]] = None) -> None:
        if not self._fast or not 0 <= mon_idx < 12 or getattr(buffer, "dtype", None) != _FLOAT16:
            return self._slow(mon, buffer, mon_idx, is_self, available_move_ids)

        sc = self._sc_start[mon_idx]
        ids = self._id_start[mon_idx]
        buffer[sc:sc + self._block] = self._blank[mon_idx]
        if not mon:
            buffer[ids:ids + MOVES_PER_MON] = 0
            return

        moves = list(mon.moves.values())
        known = len(moves) if len(moves) < MOVES_PER_MON else MOVES_PER_MON
        rows = self._rows
        single, head, tail = self._single, self._head, self._tail
        stab_pool = None
        active = False
        for m_idx in range(known):
            move = moves[m_idx]
            if not move:
                # Not a move object as poke-env makes them. The function above knows what to do.
                return self._slow(mon, buffer, mon_idx, is_self, available_move_ids)
            try:
                key = (move.__class__, move.id, move.gen, move.base_power)
            except AttributeError:
                return self._slow(mon, buffer, mon_idx, is_self, available_move_ids)
            entry = rows.get(key)
            if entry is None:
                entry = self._row(move, key)
            elif entry[4]:
                count_unknowns(entry[4])

            s = sc + m_idx * single
            buffer[ids + m_idx] = entry[0]
            buffer[s:s + head] = entry[1]
            buffer[s + tail:s + single] = entry[2]
            buffer[s + self._pp] = int(move.current_pp)

            if stab_pool is None:
                stab_pool = [mon.type_1, mon.type_2]
                if getattr(mon, 'terastallized', False) and getattr(mon, 'tera_type', None):
                    stab_pool.append(mon.tera_type)
                active = getattr(mon, 'active', False)
            if entry[3] in stab_pool:
                buffer[s + self._stab] = 1.0
            if active and (not is_self or (available_move_ids and move.id in available_move_ids)):
                buffer[s + self._avail] = 1.0

        if known < MOVES_PER_MON:
            unknown = self._move_id("unknown")
            for m_idx in range(known, MOVES_PER_MON):
                buffer[ids + m_idx] = unknown
