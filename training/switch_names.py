"""
A switch the server would give to the active Pokemon instead.

poke-env 0.15 sends a switch as "/choose switch <name>" (SingleBattleOrder.message). Showdown
resolves that text in Side.chooseSwitch (sim/side.ts) by walking the side's Pokemon in their
order, the active one first, and taking the FIRST whose name is the text or whose species id is
toID(text). A Pokemon that has Transformed (Imposter Ditto, or the move) has the species it
copied. So when our active Ditto has become, say, the foe's Garchomp and our own Garchomp is on
the bench, "/choose switch Garchomp" names the active Ditto and the server answers

    [Invalid choice] Can't switch: You can't switch to an active Pokemon

The switch itself is legal ("/choose switch 2" is taken), and poke-env's available_switches is
right: it is the name that cannot reach that Pokemon. The live run saw it about once in 100,000
decisions, always with Ditto active (2026-10-06, trainer_0005: 30 refusals). Reproduced on the
box with the server's own simulator, tests/fixtures/ditto_transform_switch.json.

switch_name_taken(active, mon) says whether that is the case for a switch to mon, so that the
action mask (ObservationAssembler.create_action_mask, foulplay_bridge.legal_mask) does not offer
an action whose order the server refuses. It is False for every state but an active Pokemon that
has Transformed into the species mon is named after, so nothing else of the mask changes.

poke-env keeps no "transformed into" species: Pokemon.transform(into) keeps the copied species'
base stats, height and weight (_temporary_base_stats, _heightm, _weightkg). The name's species is
taken as the copied one when all three are its Pokedex entry's. (Types are not compared: a
Transformed Pokemon that Terastallizes loses its _temporary_types.)
"""

from __future__ import annotations

from typing import Any


def switch_name_taken(active: Any, mon: Any) -> bool:
    """True when "/choose switch <mon.name>" would name the active Pokemon (it has Transformed into mon's species)."""
    if active is None or mon is None or mon is active:
        return False
    try:
        if not active.transformed:
            return False
        stats = active._temporary_base_stats
        if not stats:
            return False
        from poke_env.data import GenData
        from poke_env.data.normalize import to_id_str
        entry = GenData.from_gen(active.gen).pokedex.get(to_id_str(mon.name))
        if not entry:
            return False
        return (dict(entry.get("baseStats") or {}) == dict(stats)
                and entry.get("heightm") == getattr(active, "_heightm", None)
                and entry.get("weightkg") == getattr(active, "_weightkg", None))
    except Exception:
        return False
