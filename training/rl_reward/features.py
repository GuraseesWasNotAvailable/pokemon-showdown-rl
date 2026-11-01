"""
What the reward's potential function looks at.

One function, `battle_features`, turns a poke-env battle into a short vector of differences
between the two sides: Pokemon left, HP left, status, boosts, hazards, screens, Tera. Every
entry is "mine minus theirs", so the vector seen from the other side is the same vector with
the sign flipped. That is what makes the potential built on it zero-sum.

The same function is used in two places, on purpose:

  * `fit_potential.py` runs recorded games through poke-env's own protocol parser and calls it
    on every turn, to fit the weights against who went on to win.
  * `worker.py` calls it on the live battle at every decision during training.

Because both go through the same parser and the same function, the potential the agent is
trained on is exactly the one that was fitted and checked.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:  # poke-env >= 0.15
    from poke_env.battle import Battle, SideCondition, Status
except ImportError:  # older layouts
    from poke_env.environment import Battle, SideCondition, Status  # type: ignore

FEATURES: List[str] = [
    "alive_diff",       # (mine alive - theirs alive) / 6
    "hp_diff",          # (sum of my HP fractions - sum of theirs) / 6, unseen Pokemon count as full
    "alive_share",      # mine / (mine + theirs) - 0.5: one Pokemon matters more at 2 v 1 than at 6 v 5
    "hp_share",         # the same for HP
    "active_hp_diff",   # HP fraction of my active minus theirs
    "slpfrz_diff",      # Pokemon asleep or frozen, mine - theirs, / 6
    "par_diff",         # paralysed
    "brn_diff",         # burned
    "psn_diff",         # poisoned or badly poisoned
    "off_boost_diff",   # positive Atk + SpA + Spe stages on the active, / 6
    "def_boost_diff",   # positive Def + SpD stages on the active, / 6
    "neg_boost_diff",   # negative stages on the active (a negative number), / 6
    "rocks_diff",       # Stealth Rock on my side minus on theirs
    "spikes_diff",      # Spikes layers / 3
    "tspikes_diff",     # Toxic Spikes layers / 2
    "web_diff",         # Sticky Web
    "screens_diff",     # Reflect, Light Screen, Aurora Veil up, / 2
    "tera_diff",        # I still have my Tera minus they still have theirs
    "hidden_diff",      # Pokemon the other side has not seen yet, mine - theirs, / 6
]
N_FEATURES = len(FEATURES)

_SLEEPY = (Status.SLP, Status.FRZ)
_POISON = (Status.PSN, Status.TOX)
_SCREENS = (SideCondition.REFLECT, SideCondition.LIGHT_SCREEN, SideCondition.AURORA_VEIL)


def _side(team: Dict, active, conditions: Dict, tera_left: bool, size: int) -> Dict[str, float]:
    """Totals for one side. Pokemon not yet seen are alive at full HP."""
    mons = list(team.values())
    fainted = sum(1 for p in mons if p.fainted)
    size = max(size or 6, len(mons))
    # Not yet sent out: the other side knows nothing about these.
    hidden = size - sum(1 for p in mons if getattr(p, "revealed", True) or p.fainted or p.active)
    alive = size - fainted
    hp = float(size - len(mons))
    slpfrz = par = brn = psn = 0
    for p in mons:
        if p.fainted:
            continue
        hp += max(0.0, min(1.0, p.current_hp_fraction or 0.0))
        st = p.status
        if st in _SLEEPY:
            slpfrz += 1
        elif st == Status.PAR:
            par += 1
        elif st == Status.BRN:
            brn += 1
        elif st in _POISON:
            psn += 1
    off = dfn = neg = 0.0
    act_hp = 0.0
    if active is not None and not active.fainted:
        act_hp = max(0.0, min(1.0, active.current_hp_fraction or 0.0))
        b = active.boosts or {}
        off = sum(max(0, b.get(k, 0)) for k in ("atk", "spa", "spe"))
        dfn = sum(max(0, b.get(k, 0)) for k in ("def", "spd"))
        neg = sum(min(0, b.get(k, 0)) for k in ("atk", "spa", "spe", "def", "spd"))
    return {
        "alive": float(alive), "hp": hp, "act_hp": act_hp,
        "slpfrz": float(slpfrz), "par": float(par), "brn": float(brn), "psn": float(psn),
        "off": float(off), "def": float(dfn), "neg": float(neg),
        "rocks": 1.0 if SideCondition.STEALTH_ROCK in conditions else 0.0,
        "spikes": min(3.0, float(conditions.get(SideCondition.SPIKES, 0) or 0)),
        "tspikes": min(2.0, float(conditions.get(SideCondition.TOXIC_SPIKES, 0) or 0)),
        "web": 1.0 if SideCondition.STICKY_WEB in conditions else 0.0,
        "screens": float(sum(1 for c in _SCREENS if c in conditions)),
        "tera": 1.0 if tera_left else 0.0,
        "hidden": float(max(0, hidden)),
    }


def _team_size(battle, role: Optional[str]) -> int:
    try:
        return int(battle.team_size.get(role, 6)) if role else 6
    except Exception:
        return 6


def battle_features(battle) -> np.ndarray:
    """The feature vector for `battle` from the point of view of the player it belongs to."""
    gen9 = getattr(battle, "gen", 9) == 9
    me = _side(battle.team, battle.active_pokemon, battle.side_conditions,
               gen9 and not battle.used_tera, _team_size(battle, battle.player_role))
    op = _side(battle.opponent_team, battle.opponent_active_pokemon, battle.opponent_side_conditions,
               gen9 and not battle.opponent_used_tera, _team_size(battle, battle.opponent_role))
    d = lambda k: me[k] - op[k]  # noqa: E731
    alive_tot = me["alive"] + op["alive"]
    hp_tot = me["hp"] + op["hp"]
    f = np.array([
        d("alive") / 6.0,
        d("hp") / 6.0,
        (me["alive"] / alive_tot - 0.5) if alive_tot > 0 else 0.0,
        (me["hp"] / hp_tot - 0.5) if hp_tot > 0 else 0.0,
        d("act_hp"),
        d("slpfrz") / 6.0,
        d("par") / 6.0,
        d("brn") / 6.0,
        d("psn") / 6.0,
        d("off") / 6.0,
        d("def") / 6.0,
        d("neg") / 6.0,
        d("rocks"),
        d("spikes") / 3.0,
        d("tspikes") / 2.0,
        d("web"),
        d("screens") / 2.0,
        d("tera"),
        d("hidden") / 6.0,
    ], dtype=np.float64)
    return f


# ---------------------------------------------------------------------------
# Running a recorded game through poke-env, for fitting and for checks
# ---------------------------------------------------------------------------

_SKIP = {"request", "t:", "uhtmlchange", "expire", "c", "c:", "j", "J", "l", "L", "n", "N", "raw", "html",
         "uhtml", "inactive", "inactiveoff", "error", "bigerror", "chat", "debug", "message", "showteam",
         "teampreview", "rated", "badge", "seed", "deinit", "init", "title", "join", "leave", "unlink", "hidelines"}

_quiet = logging.getLogger("rl_reward.replay")
_quiet.setLevel(logging.CRITICAL)


def replay_states(lines: Iterable[str], me: str, tag: str = "battle-replay-0") -> Tuple[List[np.ndarray], List[int], Optional[str]]:
    """
    Feeds a game's protocol lines to a poke-env Battle as the player named `me` and returns the
    feature vector at the start of every turn, the turn numbers, and the winner's name (None for
    a tie or an unfinished log).
    """
    battle = Battle(tag, me, _quiet, gen=9)
    feats: List[np.ndarray] = []
    turns: List[int] = []
    winner: Optional[str] = None
    for raw in lines:
        if not raw or raw[0] != "|":
            continue
        parts = raw.rstrip("\n").split("|")
        if len(parts) < 2:
            continue
        cmd = parts[1]
        if cmd == "win":
            winner = parts[2] if len(parts) > 2 else None
            break
        if cmd == "tie":
            break
        if cmd in _SKIP or cmd == "":
            continue
        try:
            battle.parse_message(parts)
        except Exception:
            continue
        if cmd == "turn":
            try:
                feats.append(battle_features(battle))
                turns.append(int(parts[2]))
            except Exception:
                pass
    return feats, turns, winner
