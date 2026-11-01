"""
Ability Encoder for Pokémon Showdown Observations.

This module handles the encoding of Pokémon abilities into categorical IDs. 
It distinguishes between the 'confirmed' ability (Slot 0) and 'possible' 
abilities (Slots 1-3) to account for hidden information in competitive play.
"""

from __future__ import annotations

import contextlib
import logging
from collections import Counter
from typing import Any, Callable, Dict, Iterator, Optional, Final, Tuple

import numpy as np
import utils
from utils import get_id

# Configure logger
logger = logging.getLogger(__name__)

# Constants for the ability sub-block per Pokémon
ABILITY_SLOTS_PER_MON: Final[int] = 4

def encode_ability_inplace(
    mon: Optional[Any], 
    buffer: np.ndarray, 
    mon_idx: int, 
    offsets: Dict[str, tuple[int, int]], 
    vocab: Dict[str, Dict[str, int]]
) -> None:
    """
    Writes a block of 4 ability IDs directly into the workspace buffer.

    The first slot (index 0) is reserved for the ability confirmed by the 
    game engine. Remaining slots are populated with 'possible' abilities 
    defined by the Pokémon's species data.

    Args:
        mon: The poke-env Pokemon object (or None if the slot is empty).
        buffer: The main NumPy calculation buffer (float16/float32).
        mon_idx: Slot index (0-5 for self, 6-11 for opponent).
        offsets: Dictionary containing start/end indices for 'ability_ids'.
        vocab: The vocabulary map for categorical ability IDs.
    """
    # 1. Calculate the starting position for this Pokémon's 4-slot ability block
    try:
        base_start, _ = offsets["ability_ids"]
    except KeyError:
        logger.error("Offset 'ability_ids' not found in provided offsets dictionary.")
        return

    start = base_start + (mon_idx * ABILITY_SLOTS_PER_MON)

    # If the Pokémon is None (empty team slot), the buffer remains zeroed.
    # In our vocab, ID 0 is typically reserved for 'None' or 'Padding'.
    if mon is None:
        return

    # 2. Encode the Confirmed Ability (Slot 0)
    # This is the ability currently active or revealed in battle.
    try:
        buffer[start] = get_id(vocab, "pokemon.ability", mon.ability)
    except Exception as e:
        logger.warning(f"Failed to encode confirmed ability for {mon.species}: {e}")

    # 3. Handle Possible Abilities (Slots 1-3)
    # This provides the model with context about what the opponent *could* have.
    # We use manual iteration to maintain high performance in the assembly loop.
    possible_abilities = getattr(mon, 'possible_abilities', None)
    
    if possible_abilities:
        ability_iter = iter(possible_abilities)
        # We attempt to fill the remaining 3 slots (indices 1, 2, 3)
        for i in range(1, ABILITY_SLOTS_PER_MON):
            try:
                ability_name = next(ability_iter)
                buffer[start + i] = get_id(vocab, "pokemon.ability", ability_name)
            except StopIteration:
                # No more possible abilities to encode; remaining slots stay 0
                break
            except Exception as e:
                logger.debug(f"Error encoding possible ability slot {i} for {mon.species}: {e}")
                continue


# ---------------------------------------------------------------------------
# REMEMBERED LOOKUPS
# ---------------------------------------------------------------------------
#
# Most of what the encoders compute for one decision they computed for the decision before:
# the id of an ability, the features of a move, the stat bounds of a Pokemon nothing new was
# learned about. The helpers below let them keep such an answer under a key that holds every
# input it depends on, and hand it back the next time. The answer is always produced by the
# original code, once, so it is the same number it always was.
#
# One thing the original lookups do besides answering: utils.get_id counts every name that is
# not in the vocabulary in utils.UNKNOWN_ENTITIES, each time it is asked. A remembered answer
# skips get_id, so the names it would have counted are kept with the answer and counted again
# on every use. The counter ends up exactly where it would have.

@contextlib.contextmanager
def recording_unknowns() -> Iterator[Counter]:
    """
    Collects what utils.get_id counts as unknown inside the block, and still counts it.
    The encoders run on one thread (poke-env's loop), which is what makes the swap safe.
    """
    real = utils.UNKNOWN_ENTITIES
    seen: Counter = Counter()
    utils.UNKNOWN_ENTITIES = seen
    try:
        yield seen
    finally:
        utils.UNKNOWN_ENTITIES = real
        real.update(seen)


def count_unknowns(names: Tuple[str, ...]) -> None:
    """Counts names a remembered answer would have counted had it been worked out again."""
    counter = utils.UNKNOWN_ENTITIES
    for name in names:
        counter[name] += 1


def remembered(cache: Dict[Any, Tuple[Any, Tuple[str, ...]]], key: Any, compute: Callable[[], Any]) -> Any:
    """compute() the first time a key is seen, the stored answer afterwards. The key must be hashable."""
    hit = cache.get(key)
    if hit is None:
        with recording_unknowns() as seen:
            value = compute()
        cache[key] = (value, tuple(seen.elements()))
        return value
    if hit[1]:
        count_unknowns(hit[1])
    return hit[0]


class IdLookup:
    """
    get_id(vocab, path, value) with the answer for each value remembered.

    Keyed by the value's class as well as the value: get_id works on the text of a value, and
    values that compare equal (1, 1.0 and True) do not have the same text.
    """
    __slots__ = ("_vocab", "_path", "_strings", "_others")

    def __init__(self, vocab: Dict[str, Dict[str, int]], path: str):
        self._vocab = vocab
        self._path = path
        self._strings: Dict[str, Tuple[int, Tuple[str, ...]]] = {}
        self._others: Dict[Any, Tuple[int, Tuple[str, ...]]] = {}

    def __call__(self, value: Any) -> int:
        if value.__class__ is str:
            hit = self._strings.get(value)
            if hit is None:
                with recording_unknowns() as seen:
                    answer = get_id(self._vocab, self._path, value)
                self._strings[value] = (answer, tuple(seen.elements()))
                return answer
        else:
            try:
                hit = self._others.get((value.__class__, value))
            except TypeError:        # not hashable: nothing to remember it under
                return get_id(self._vocab, self._path, value)
            if hit is None:
                with recording_unknowns() as seen:
                    answer = get_id(self._vocab, self._path, value)
                self._others[(value.__class__, value)] = (answer, tuple(seen.elements()))
                return answer
        if hit[1]:
            count_unknowns(hit[1])
        return hit[0]


class AbilityEncoder:
    """encode_ability_inplace for one vocabulary and one buffer layout, with the ids remembered."""

    def __init__(self, offsets: Dict[str, Tuple[int, int]], vocab: Dict[str, Dict[str, int]]):
        self._offsets = offsets
        self._vocab = vocab
        base = offsets.get("ability_ids", (None, None))[0]
        self._starts = None if base is None else [base + i * ABILITY_SLOTS_PER_MON for i in range(12)]
        self._id = IdLookup(vocab, "pokemon.ability")

    def encode(self, mon: Optional[Any], buffer: np.ndarray, mon_idx: int) -> None:
        if self._starts is None or not 0 <= mon_idx < 12:
            return encode_ability_inplace(mon, buffer, mon_idx, self._offsets, self._vocab)
        if mon is None:
            return
        start = self._starts[mon_idx]
        lookup = self._id
        try:
            buffer[start] = lookup(mon.ability)
        except Exception as e:
            logger.warning(f"Failed to encode confirmed ability for {mon.species}: {e}")

        possible_abilities = getattr(mon, 'possible_abilities', None)
        if possible_abilities:
            ability_iter = iter(possible_abilities)
            for i in range(1, ABILITY_SLOTS_PER_MON):
                try:
                    ability_name = next(ability_iter)
                    buffer[start + i] = lookup(ability_name)
                except StopIteration:
                    break
                except Exception as e:
                    logger.debug(f"Error encoding possible ability slot {i} for {mon.species}: {e}")
                    continue
