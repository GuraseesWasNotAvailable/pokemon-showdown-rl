"""
Which moves set up, put to sleep, or do nothing to the Pokemon across the field, and the names
to print for move and species ids. Read from poke-env's move data (gen 9), so no hand-kept table
of boosts can go out of date; the few moves whose effect is code and not data are listed here.

    import setup_moves as sm
    sm.raised_stats("swordsdance")          {"atk": 2}
    sm.is_setup("curse", ["WATER"])         True (a Ghost's Curse is not a set-up move)
    sm.is_sleep("spore"), sm.is_passive("stealthrock"), sm.fails_on_status("suckerpunch")
    sm.move_name("sleeppowder"), sm.species_name("ursalunabloodmoon")

Used by the search player's guard rule (search_player.Threat), its printed predictions, and the
set-up test opponent (bench/setup_player.py). Nothing here raises: an unknown id is "not set-up,
not sleep, not passive", and its name is the id.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional

__all__ = ["STATS", "raised_stats", "is_setup", "is_sleep", "is_passive", "fails_on_status", "is_status_move",
           "move_name", "species_name", "to_id", "largest_boost", "setup_share", "move_entry"]

STATS = ("atk", "def", "spa", "spd", "spe")

# Set-up moves whose boosts are code in Showdown's data, not a "boosts" field.
_CODE_BOOSTS: Dict[str, Dict[str, int]] = {
    "curse": {"atk": 1, "def": 1, "spe": -1},          # not for a Ghost (see is_setup)
    "bellydrum": {"atk": 12},                          # to +6, for half its HP
    "tidyup": {"atk": 1, "spe": 1},
    "acupressure": {"def": 2},                         # a random stat by 2: counted as one
}
# Sleep: the status "slp" in the data, and Yawn (sleep at the end of the next turn).
_SLEEP_EXTRA = {"yawn"}
# Status moves that hit the Pokemon across the field in effect though the data says "all" or
# "self": they are answers to a set-up move, not passive.
_NOT_PASSIVE = {"haze", "perishsong", "sleeptalk"}
# Status moves aimed at the opponent that still do nothing to its active Pokemon.
_PASSIVE_EXTRA = {"defog"}
# Fail when the target does not use a damaging move this turn (a set-up or a sleep move, say).
_FAILS_ON_STATUS = {"suckerpunch", "thunderclap"}
_PASSIVE_TARGETS = {"self", "allySide", "allyTeam", "foeSide", "all", "allies", "adjacentAllyOrSelf", "adjacentAlly"}

_MOVES: Optional[Dict[str, Any]] = None
_DEX: Optional[Dict[str, Any]] = None


def to_id(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def _data() -> None:
    global _MOVES, _DEX
    if _MOVES is None:
        try:
            from poke_env.data import GenData

            data = GenData.from_gen(9)
            _MOVES, _DEX = data.moves, data.pokedex
        except Exception:       # no poke-env: every move is "unknown", every name its id
            _MOVES, _DEX = {}, {}


def _entry(move_id: str) -> Dict[str, Any]:
    _data()
    return (_MOVES or {}).get(to_id(move_id)) or {}


def move_entry(move_id: Any) -> Dict[str, Any]:
    """The move's entry in poke-env's gen 9 data ({} when unknown): category, type, basePower, flags, target... Not to be changed."""
    return _entry(str(move_id or ""))


def raised_stats(move_id: str, user_types: Optional[Iterable[str]] = None) -> Dict[str, int]:
    """The stat stages a status move raises on its user ({} for anything else, a Ghost's Curse included)."""
    move = to_id(move_id)
    if move == "curse" and any(str(t).upper() == "GHOST" for t in (user_types or ()) if t):
        return {}
    if move in _CODE_BOOSTS:
        return {s: v for s, v in _CODE_BOOSTS[move].items() if v > 0}
    entry = _entry(move)
    if entry.get("category") != "Status" or entry.get("target") not in ("self", "adjacentAllyOrSelf", "allies"):
        return {}
    return {s: int(v) for s, v in (entry.get("boosts") or {}).items() if s in STATS and int(v) > 0}


def is_setup(move_id: str, user_types: Optional[Iterable[str]] = None) -> bool:
    """A status move that raises the user's attack, defences or speed: Swords Dance, Calm Mind, Curse, Shell Smash..."""
    return bool(raised_stats(move_id, user_types))


def is_sleep(move_id: str) -> bool:
    """Spore, Sleep Powder, Hypnosis, Sing, Lovely Kiss, Grass Whistle, Dark Void, Yawn."""
    move = to_id(move_id)
    if move in _SLEEP_EXTRA:
        return True
    entry = _entry(move)
    return entry.get("category") == "Status" and entry.get("status") == "slp"


def is_status_move(move_id: str) -> bool:
    return _entry(move_id).get("category") == "Status"


def is_passive(move_id: str) -> bool:
    """
    A status move that does nothing to the opponent's active Pokemon: our own set-up, recovery,
    protection, Substitute, hazards, screens, weather, Defog. Haze, Perish Song, phazing,
    Encore, Taunt, burns and paralysis are not passive: they are what answers a set-up move.
    """
    move = to_id(move_id)
    if move in _NOT_PASSIVE:
        return False
    if move in _PASSIVE_EXTRA or move in _CODE_BOOSTS:
        return True
    entry = _entry(move)
    return entry.get("category") == "Status" and entry.get("target") in _PASSIVE_TARGETS


def fails_on_status(move_id: str) -> bool:
    """Sucker Punch and Thunderclap: they fail against a Pokemon that uses a status move."""
    return to_id(move_id) in _FAILS_ON_STATUS


def largest_boost(pokemon: Any) -> int:
    """The largest positive stat stage of a poke-env Pokemon in attack, defences or speed (0 when none, or no Pokemon)."""
    try:
        boosts = dict(getattr(pokemon, "boosts", None) or {})
        return max([int(v) for k, v in boosts.items() if k in STATS and int(v) > 0] or [0])
    except Exception:
        return 0


def setup_share(options: Iterable[Any], user_types: Optional[Iterable[str]] = None) -> float:
    """
    The share a read of the opponent's action puts on its set-up moves, 0 to 1. `options` is
    (name, share) pairs or (name, share, seen) triples in the search's names ("move swordsdance",
    "move swordsdance terastallize", "switch garchomp"); the shares are normalised over all options.
    """
    total = setup = 0.0
    types = list(user_types or ())
    for option in options or ():
        try:
            name, share = str(option[0]), max(0.0, float(option[1]))
        except Exception:
            continue
        total += share
        if name.startswith("move ") and is_setup(name[5:].replace(" terastallize", ""), types):
            setup += share
    return setup / total if total > 0 else 0.0


def move_name(move_id: str) -> str:
    """"sleeppowder" -> "Sleep Powder" (the id itself when unknown)."""
    entry = _entry(move_id)
    return str(entry.get("name") or move_id)


def species_name(species_id: str) -> str:
    """"ursalunabloodmoon" -> "Ursaluna-Bloodmoon" (the id itself when unknown)."""
    _data()
    entry = (_DEX or {}).get(to_id(species_id)) or {}
    return str(entry.get("name") or species_id)
