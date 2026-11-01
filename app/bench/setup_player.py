"""
A set-up test opponent: poke-env's SimpleHeuristicsPlayer that sets up and puts to sleep, the way
the ladder opponents that beat the search player did (bench/notes/predict_live.md: Swords Dance,
Calm Mind x5 + Recover, Curse + Rest + Sleep Talk, Quiver Dance + Sleep Powder, Dragon Dance).

    from bench.setup_player import setup_player
    Setup = setup_player()                    # the class, built on first use (poke-env is imported then)
    p = Setup(account_configuration=..., server_configuration=..., battle_format="gen9randombattle")

    python bench/ladder_opp.py --kind setup --port P ...          on a local ladder
    python bench/gauntlet.py opp --kind setup --pairs ...         in a head to head measurement

It plays random-battle sets as they are dealt and sends only legal orders: every move it picks is
one of battle.available_moves, everything else is SimpleHeuristicsPlayer's own choice. On a move
request in singles, in this order:

  1. When SimpleHeuristics would switch out (a bad matchup, stats dropped by 3), it does.
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
  6. SimpleHeuristics: hazards, its strongest move by its own estimate (boosts included), Tera.

in_danger: at half its HP or less; or the opponent's types hit it four times over; or twice over
and it is below 85%; or the opponent is at +2 or more in attack or special attack and it is below
85%. A rough guess, as SimpleHeuristics' own estimates are: it sees types, HP and boosts, not
damage.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import setup_moves as sm  # noqa: E402

RECOVERY = ("recover", "roost", "slackoff", "softboiled", "milkdrink", "moonlight", "morningsun", "synthesis", "shoreup",
            "rest", "lunarblessing", "junglehealing")
SLEEP_BLOCKERS = {"insomnia", "vitalspirit", "sweetveil", "comatose", "purifyingsalt"}
MAX_STAGE = 4

_CLASS = None


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


def setup_choice(battle: Any, max_stage: int = MAX_STAGE, recover: bool = True) -> Optional[Any]:
    """The move rules 2 to 5 pick (a poke-env Move from battle.available_moves), or None for SimpleHeuristics' own choice."""
    from poke_env.battle import Status
    from poke_env.player import SimpleHeuristicsPlayer

    me, them = battle.active_pokemon, battle.opponent_active_pokemon
    moves = list(battle.available_moves or [])
    if me is None or them is None or not moves:
        return None
    if battle.available_switches and SimpleHeuristicsPlayer._should_switch_out(battle):
        return None                                                                    # 1
    by_id = {m.id: m for m in moves}
    if me.status == Status.SLP:
        return by_id.get("sleeptalk")                                                  # 2
    boosts = getattr(me, "boosts", None) or {}
    hp = float(me.current_hp_fraction or 0.0)
    if recover and hp < 0.5 and max([int(boosts.get(s, 0)) for s in sm.STATS] or [0]) >= 1:
        for move_id in RECOVERY:                                                       # 3
            if move_id in by_id:
                return by_id[move_id]
    for move in moves:                                                                 # 4
        if sm.is_sleep(move.id) and can_sleep(battle, move):
            return move
    if in_danger(battle):
        return None
    best, gain = None, 0
    for move in moves:                                                                 # 5
        raised = sm.raised_stats(move.id, _types(me))
        if not raised:
            continue
        if move.id == "bellydrum" and hp <= 0.55:
            continue
        room = sum(min(v, max(0, max_stage - int(boosts.get(s, 0)))) for s, v in raised.items())
        if room > gain:
            best, gain = move, room
    return best


def setup_player() -> type:
    """The class (SimpleHeuristicsPlayer with the rules above). Built on first use."""
    global _CLASS
    if _CLASS is not None:
        return _CLASS
    from poke_env.battle import DoubleBattle
    from poke_env.player import SimpleHeuristicsPlayer

    class SetupPlayer(SimpleHeuristicsPlayer):
        """SimpleHeuristics that sets up while it is safe, puts the target to sleep when it can, and heals when boosted."""

        def __init__(self, *args: Any, max_stage: int = MAX_STAGE, recover: bool = True, **kwargs: Any):
            self.max_stage = int(max_stage)
            self.recover = bool(recover)
            self.setup_counts = {"setup": 0, "sleep": 0, "recover": 0, "sleeptalk": 0, "heuristic": 0}
            super().__init__(*args, **kwargs)

        def choose_move(self, battle: Any):
            if not isinstance(battle, DoubleBattle):
                try:
                    move = setup_choice(battle, self.max_stage, self.recover)
                except Exception:       # a test opponent must keep playing: SimpleHeuristics decides
                    move = None
                if move is not None:
                    kind = ("sleeptalk" if move.id == "sleeptalk" else "recover" if move.id in RECOVERY
                            else "sleep" if sm.is_sleep(move.id) else "setup")
                    self.setup_counts[kind] += 1
                    return self.create_order(move)
                self.setup_counts["heuristic"] += 1
            return super().choose_move(battle)

    _CLASS = SetupPlayer
    return _CLASS
