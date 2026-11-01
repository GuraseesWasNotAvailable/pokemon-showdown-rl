"""
The set-up rules of the setupnet opponent seat (opponent_seats.py).

They are rules 2 to 5 of branch decide's bench/setup_player.py (as of 364f225). That player is
poke-env's SimpleHeuristicsPlayer with set-up rules on top. Rule 1 (switch out when
SimpleHeuristics would) and rule 6 (SimpleHeuristics' own move) are left out: in the setupnet
seat the network's own choice stands in for both. On a move request in singles, in this order:

  2. Asleep with Sleep Talk: Sleep Talk.
  3. Boosted (+1 or more in anything) and below half its HP: a recovery move (Recover, Roost,
     Slack Off, Rest...) when it has one.
  4. A sleep move (Spore, Sleep Powder, Hypnosis, Sing, Lovely Kiss, Grass Whistle, Dark Void,
     Yawn) when the target can fall asleep: no status, nobody on its side asleep already (Sleep
     Clause), no Substitute, not drowsy already, no Safeguard, no Electric or Misty Terrain, no
     Insomnia, Vital Spirit, Sweet Veil, Comatose or Purifying Salt known (or the only ones it can
     have), and for a powder move not a Grass type, no Overcoat, no Safety Goggles.
  5. A set-up move (setup_moves.is_setup: Swords Dance, Dragon Dance, Quiver Dance, Calm Mind,
     Nasty Plot, Curse, Shell Smash, Bulk Up, Belly Drum, Agility...) when it is not about to be
     knocked out (in_danger) and something the move raises is still below max_stage. Of several,
     the one that raises most.

in_danger: at half its HP or less; or the opponent's types hit it four times over; or twice over
and it is below 85%; or the opponent is at +2 or more in attack or special attack and it is below
85%. A rough guess: it sees types, HP and boosts, not damage.

Every move a rule picks is one of battle.available_moves, so it is always legal.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import setup_moves as sm

RECOVERY = ("recover", "roost", "slackoff", "softboiled", "milkdrink", "moonlight", "morningsun", "synthesis", "shoreup",
            "rest", "lunarblessing", "junglehealing")
SLEEP_BLOCKERS = {"insomnia", "vitalspirit", "sweetveil", "comatose", "purifyingsalt"}
MAX_STAGE = 4                     # rule 5 stops raising a stat once it is at this stage (decide's default)
KINDS = ("sleeptalk", "recover", "sleep", "setup")   # which rule picked a move, in rule order


def _types(mon: Any) -> list:
    return [getattr(t, "name", str(t)) for t in (getattr(mon, "types", None) or ()) if t is not None]


def in_danger(battle: Any) -> bool:
    """About to be knocked out, roughly: see the module's text."""
    me, them = battle.active_pokemon, battle.opponent_active_pokemon
    if me is None or them is None:
        return True
    hp = float(me.current_hp_fraction or 0.0)
    if hp <= 0.5:
        return True
    hits = max([me.damage_multiplier(t) for t in (getattr(them, "types", None) or ()) if t is not None] or [1.0])
    if hits >= 4 or (hits >= 2 and hp < 0.85):
        return True
    boosts = getattr(them, "boosts", None) or {}
    return max(int(boosts.get("atk", 0)), int(boosts.get("spa", 0))) >= 2 and hp < 0.85


def can_sleep(battle: Any, move: Any) -> bool:
    """Whether the target of a sleep move can fall asleep now (as far as the battle shows)."""
    from poke_env.battle import Effect, Field, SideCondition, Status

    them = battle.opponent_active_pokemon
    if them is None or them.fainted or them.status is not None:
        return False
    if any(m.status == Status.SLP for m in battle.opponent_team.values() if not m.fainted):
        return False                                        # Sleep Clause: one of its side is asleep already
    effects = getattr(them, "effects", None) or {}
    if Effect.SUBSTITUTE in effects or (move.id == "yawn" and Effect.YAWN in effects):
        return False
    if SideCondition.SAFEGUARD in (battle.opponent_side_conditions or {}):
        return False
    if Field.ELECTRIC_TERRAIN in (battle.fields or {}) or Field.MISTY_TERRAIN in (battle.fields or {}):
        return False
    ability = sm.to_id(getattr(them, "ability", None))
    possible = {sm.to_id(a) for a in (getattr(them, "possible_abilities", None) or [])}
    if ability in SLEEP_BLOCKERS or (not ability and possible and possible <= SLEEP_BLOCKERS):
        return False
    if (move.entry.get("flags") or {}).get("powder"):
        if "GRASS" in _types(them) or ability == "overcoat" or sm.to_id(getattr(them, "item", None)) == "safetygoggles":
            return False
    return True


def kind_of(move_id: str) -> str:
    """Which rule a move belongs to: "sleeptalk", "recover", "sleep" or "setup"."""
    move_id = sm.to_id(move_id)
    if move_id == "sleeptalk":
        return "sleeptalk"
    if move_id in RECOVERY:
        return "recover"
    if sm.is_sleep(move_id):
        return "sleep"
    return "setup"


def setup_rule(battle: Any, max_stage: int = MAX_STAGE, recover: bool = True) -> Optional[Tuple[str, Any]]:
    """
    (kind, move) for the first of rules 2 to 5 that applies, the move one of battle.available_moves;
    None when none applies (the network's own choice then stands).
    """
    from poke_env.battle import Status

    me, them = battle.active_pokemon, battle.opponent_active_pokemon
    moves = list(battle.available_moves or [])
    if me is None or them is None or not moves:
        return None
    by_id = {m.id: m for m in moves}
    if me.status == Status.SLP:
        move = by_id.get("sleeptalk")                                                  # 2
        return ("sleeptalk", move) if move is not None else None
    boosts = getattr(me, "boosts", None) or {}
    hp = float(me.current_hp_fraction or 0.0)
    if recover and hp < 0.5 and max([int(boosts.get(s, 0)) for s in sm.STATS] or [0]) >= 1:
        for move_id in RECOVERY:                                                       # 3
            if move_id in by_id:
                return "recover", by_id[move_id]
    for move in moves:                                                                 # 4
        if sm.is_sleep(move.id) and can_sleep(battle, move):
            return "sleep", move
    if in_danger(battle):
        return None
    best, gain = None, 0
    for move in moves:                                                                 # 5
        raised = sm.raised_stats(move.id, _types(me))
        if not raised:
            continue
        if move.id == "bellydrum" and hp <= 0.55:
            continue
        room = sum(min(v, max(0, int(max_stage) - int(boosts.get(s, 0)))) for s, v in raised.items())
        if room > gain:
            best, gain = move, room
    return ("setup", best) if best is not None else None
