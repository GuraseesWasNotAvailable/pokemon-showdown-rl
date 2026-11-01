"""
What the two counter rules of the search player know of the board (round 3 of the set-up work,
branch counterfix): which of our moves answers a set-up move and whether it can work now, which
benched Pokemon of ours could come in to answer it, and how hard each of our attacks hits.

    import setup_counter as sc
    sc.counter_kind("dragontail")                       "phaze"
    sc.counter_check("dragontail", sc.Foe(types=("FAIRY",), boosts={"spa": 2}), sc.Scene())
                                                        ("phaze", "Fairy: Dragon Tail does nothing to it")
    sc.damage_share("earthquake", attacker, defender)   0.62 (of the defender's whole HP)
    board = sc.build_board(battle, actions, threat, advice)    (the player's view, from poke-env)

The rules themselves are search_player.mix()'s (MixSettings.setup_counter, setup_other_side):
they read a Board and nothing else, so they are tested without a battle. Everything here that
reads a battle is wrapped by the player: a failure leaves mix()'s choice as it is.

When something is not known (an ability not yet shown, an item, whether it floats) the counter is
taken to work, as the owner asked ("when unknown, assume it works"). What is known comes from
poke-env's view of the battle, else from the search's sampled sets of the opponent's active
Pokemon when every sampled world agrees (two worlds at least).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import setup_moves as sm

__all__ = ["PHAZE", "SLEEP_EXTRA", "Foe", "Scene", "Fighter", "Board", "counter_kind", "counter_check", "damage_share",
           "hits_defence", "two_hit_ko", "build_board", "stage", "estimated_stats", "effectiveness", "swap_doubt", "speed",
           "full_hp_guard", "SLEEP_ACCURACY"]

PHAZE = ("roar", "whirlwind", "dragontail", "circlethrow")
HAZE = ("haze", "clearsmog")
ENCORE = ("encore",)
TAUNT = ("taunt",)
PERISH = ("perishsong",)
SLEEP_EXTRA = ("yawn",)                         # setup_moves.is_sleep adds the moves whose status is "slp"

# Abilities and items that stop a counter (only when known: else it is taken to work).
_NO_PHAZE_ABILITIES = ("guarddog", "suctioncups")
_NO_SLEEP_ABILITIES = ("insomnia", "vitalspirit", "sweetveil", "comatose", "purifyingsalt")
# A sleep move counts only at this accuracy or more (Spore, Sleep Powder, Lovely Kiss, Yawn; not
# Hypnosis, Sing or Grass Whistle: round 3's review).
SLEEP_ACCURACY = 75
# What lets a Pokemon at full HP live through any one hit (or halves it): a KO estimate at full HP
# counts for "use the counter" only when these are ruled out, or at twice its HP (round 3's review).
_FULL_HP_ABILITIES = ("sturdy", "multiscale", "shadowshield", "disguise", "iceface")
# Moves whose category (and so the defence they hit) depends on the position: never judged by
# "don't hit the boosted defence", never swapped in.
_VARIABLE_CATEGORY = ("shellsidearm", "photongeyser", "lightthatburnsthesky", "terablast", "terastarstorm")
_NO_TAUNT_ABILITIES = ("oblivious", "aromaveil")
_NO_ENCORE_ABILITIES = ("aromaveil",)
_NO_POWDER_ABILITIES = ("overcoat",)
_NO_POWDER_ITEMS = ("safetygoggles",)
_FIELD_TARGETS = ("all", "self", "allySide", "allyTeam", "foeSide", "allies")

# Damage: abilities of the defender that stop or halve a type, and the like.
_IMMUNE: Dict[str, Tuple[str, ...]] = {
    "levitate": ("GROUND",), "eartheater": ("GROUND",), "flashfire": ("FIRE",), "wellbakedbody": ("FIRE",),
    "waterabsorb": ("WATER",), "stormdrain": ("WATER",), "dryskin": ("WATER",), "voltabsorb": ("ELECTRIC",),
    "lightningrod": ("ELECTRIC",), "motordrive": ("ELECTRIC",), "sapsipper": ("GRASS",),
}
_HALVED: Dict[str, Tuple[str, ...]] = {"heatproof": ("FIRE",), "thickfat": ("FIRE", "ICE"), "waterbubble": ("FIRE",),
                                       "purifyingsalt": ("GHOST",)}
_FLAG_BOOST: Dict[str, Tuple[str, float]] = {"ironfist": ("punch", 1.2), "strongjaw": ("bite", 1.5), "megalauncher": ("pulse", 1.5),
                                             "sharpness": ("slicing", 1.5), "toughclaws": ("contact", 1.3), "punkrock": ("sound", 1.3)}
_TYPE_ABILITY: Dict[str, Tuple[str, float]] = {"transistor": ("ELECTRIC", 1.3), "dragonsmaw": ("DRAGON", 1.5),
                                               "steelworker": ("STEEL", 1.5), "waterbubble": ("WATER", 2.0),
                                               "rockypayload": ("ROCK", 1.5)}


def stage(n: int) -> float:
    """The multiplier of a stat stage (-6 to +6)."""
    n = max(-6, min(6, int(n)))
    return (2 + n) / 2.0 if n >= 0 else 2.0 / (2 - n)


def _stat(base: float, level: int, hp: bool = False) -> int:
    """A random battle stat: 84 EVs, 31 IVs, a neutral nature (as search_player._stat)."""
    core = int((2 * float(base) + 31 + 21) * int(level) / 100)
    return core + int(level) + 10 if hp else core + 5


def estimated_stats(base_stats: Dict[str, Any], level: int) -> Dict[str, float]:
    """The stats of a random battle set from its base stats and level (hp included)."""
    out = {k: float(_stat(v, level)) for k, v in (base_stats or {}).items() if k != "hp"}
    if "hp" in (base_stats or {}):
        out["hp"] = float(_stat(base_stats["hp"], level, hp=True))
    return out


def _entry(move: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(move, dict):
        return move
    return sm.move_entry(move)


def _types(values: Iterable[Any]) -> Tuple[str, ...]:
    out = []
    for t in values or ():
        if t is None:
            continue
        name = str(getattr(t, "name", t)).upper()
        if name and name not in out:
            out.append(name)
    return tuple(out)


_CHART: Optional[Dict[str, Dict[str, float]]] = None


def effectiveness(move_type: str, defender_types: Sequence[str]) -> float:
    """The type multiplier of a move of move_type on a Pokemon of these types (1 when the chart cannot be read)."""
    global _CHART
    if _CHART is None:
        try:
            from poke_env.data import GenData

            _CHART = GenData.from_gen(9).type_chart
        except Exception:
            _CHART = {}
    mult = 1.0
    for t in defender_types or ():
        mult *= float((_CHART.get(str(t).upper()) or {}).get(str(move_type).upper(), 1.0))
    return mult


# =============================================================================================
# Counters: which of our moves answer a set-up move, and whether one can work now
# =============================================================================================

@dataclass(frozen=True)
class Foe:
    """
    What the counter rules know of the opponent's active Pokemon.

    types        its types now (the Tera type alone once it has terastallized), upper case
    boosts       its stat stages ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
    ability      its ability when known (shown, the only one it can have, or every sampled world's), else ""
    item         its item when known, "none" when known to have none, "" when unknown
    status       "slp", "brn", "par", "psn", "tox", "frz" or ""
    effects      its volatile effects that matter here: "substitute", "ingrain", "encore", "taunt", "yawn", "perish"
    grounded     False when it may float (Flying, Levitate possible, Air Balloon, Magnet Rise): terrain does not guard it
    last_setup   its last action was a set-up move (Threat.last_setup)
    others       how many of its Pokemon could be dragged in (phazing fails with none)
    """
    species: str = ""
    types: Tuple[str, ...] = ()
    boosts: Dict[str, int] = field(default_factory=dict)
    ability: str = ""
    item: str = ""
    status: str = ""
    effects: Tuple[str, ...] = ()
    grounded: bool = True
    last_setup: bool = False
    others: int = 1

    @property
    def boosted(self) -> bool:
        """Any positive stat stage (evasion included): something for Haze or a phazing move to take away."""
        return any(int(v) > 0 for k, v in self.boosts.items() if k != "accuracy")


@dataclass(frozen=True)
class Scene:
    """
    The field as far as a counter cares.

    terrain      "electric", "misty" or "" (others do not stop sleep)
    sleep_used   one of the opponent's Pokemon is asleep already: Sleep Clause stops another sleep move
                 (a Rest is counted too: poke-env does not say which sleep was whose, so this errs on "used")
    """
    terrain: str = ""
    sleep_used: bool = False
    safeguard: bool = False         # the opponent's side has Safeguard up: no sleep
    weather: str = ""               # "sun": Leaf Guard stops sleep


def counter_kind(move: str) -> str:
    """"phaze", "haze", "encore", "taunt", "sleep", "perish", or "" for a move that is not a counter."""
    move = sm.to_id(move)
    if move in PHAZE:
        return "phaze"
    if move in HAZE:
        return "haze"
    if move in ENCORE:
        return "encore"
    if move in TAUNT:
        return "taunt"
    if move in PERISH:
        return "perish"
    if move in SLEEP_EXTRA or sm.is_sleep(move):
        return "sleep"
    return ""


def counter_check(move: str, foe: Foe, scene: Scene, read: float = 0.0, now: bool = True, user_ability: str = "",
                  read_floor: float = 0.50, outspeeds: bool = True) -> Tuple[str, str]:
    """
    (kind, why it does not work) of one of our moves against the opponent's active Pokemon; the
    second is "" when it works. kind is "" for a move that is not a counter.

    now     the move is our active Pokemon's, to be used this turn: Haze, Clear Smog, phazing and
            Perish Song need a boost to take away; Encore needs the opponent's last action to have
            been a set-up move; Taunt is for before the boost lands (a set-up read of read_floor or
            more, nothing boosted yet). False: the move of a benched Pokemon that would come in
            first (the boost will be there by then): those timing conditions are dropped, and Taunt
            is no counter (too late after the switch).
    read    the read of a set-up move this turn (Threat.setup_read).
    outspeeds  our active moves before the opponent's (build_board: speed with stages, Choice Scarf,
            paralysis, Trick Room). Encore now counts only when we move first or the read puts
            read_floor or more on a set-up move again: else it locks in the attack made this turn.
    """
    move = sm.to_id(move)
    kind = counter_kind(move)
    if not kind:
        return "", "not a counter"
    e = sm.move_entry(move)
    flags = set((e.get("flags") or {}).keys())
    target = str(e.get("target") or "normal")
    category = str(e.get("category") or "")
    name = sm.move_name(move)
    ability, item = foe.ability, foe.item
    if now:
        if kind in ("phaze", "haze", "perish") and not foe.boosted:
            return kind, "nothing boosted to take away yet"
        if kind == "encore" and not foe.last_setup:
            return kind, "its last action was not a set-up move"
        if kind == "encore" and not outspeeds and read < read_floor:
            return kind, "it moves first and may attack: Encore would lock in the attack"
        if kind == "taunt" and foe.boosted:
            return kind, "the boost has landed already"
        if kind == "taunt" and read < read_floor:
            return kind, "no set-up read of %d%% or more" % round(100 * read_floor)
    elif kind == "taunt":
        return kind, "too late after a switch"
    aimed = target not in _FIELD_TARGETS
    # What stops any move of its kind.
    if ability == "goodasgold" and category == "Status" and aimed:
        return kind, "Good as Gold"
    if ability == "magicbounce" and "reflectable" in flags:
        return kind, "Magic Bounce"
    if ability == "soundproof" and "sound" in flags:
        return kind, "Soundproof"
    if ability == "windrider" and "wind" in flags:
        return kind, "Wind Rider"
    if sm.to_id(user_ability) == "prankster" and category == "Status" and aimed and "DARK" in foe.types:
        return kind, "Dark: a Prankster status move fails on it"
    if ("substitute" in foe.effects and aimed and not ({"bypasssub", "sound"} & flags)
            and sm.to_id(user_ability) != "infiltrator"):
        return kind, "Substitute"
    if kind == "phaze":
        if ability in _NO_PHAZE_ABILITIES:
            return kind, "Guard Dog" if ability == "guarddog" else "Suction Cups"
        if "ingrain" in foe.effects:
            return kind, "Ingrain"
        if foe.others <= 0:
            return kind, "nothing left to drag in"
        if move == "dragontail" and "FAIRY" in foe.types:
            return kind, "Fairy: %s does nothing to it" % name
        if move == "circlethrow" and "GHOST" in foe.types:
            return kind, "Ghost: %s does nothing to it" % name
    elif kind == "haze":
        if move == "clearsmog" and "STEEL" in foe.types:
            return kind, "Steel: %s does nothing to it" % name
    elif kind == "encore":
        if "encore" in foe.effects:
            return kind, "encored already"
        if ability in _NO_ENCORE_ABILITIES:
            return kind, "Aroma Veil"
    elif kind == "taunt":
        if "taunt" in foe.effects:
            return kind, "taunted already"
        if ability in _NO_TAUNT_ABILITIES:
            return kind, "Oblivious" if ability == "oblivious" else "Aroma Veil"
    elif kind == "sleep":
        accuracy = e.get("accuracy", True)
        if accuracy is not True and isinstance(accuracy, (int, float)) and accuracy < SLEEP_ACCURACY:
            return kind, "%s hits only %d%% of the time" % (name, accuracy)
        if foe.status:
            return kind, "it has a status already (%s)" % foe.status
        if move == "yawn" and "yawn" in foe.effects:
            return kind, "drowsy already"
        if scene.sleep_used:
            return kind, "Sleep Clause"
        if foe.grounded and scene.terrain in ("electric", "misty"):
            return kind, "%s Terrain" % scene.terrain.capitalize()
        if ability in _NO_SLEEP_ABILITIES:
            return kind, {"insomnia": "Insomnia", "vitalspirit": "Vital Spirit", "sweetveil": "Sweet Veil",
                          "comatose": "Comatose", "purifyingsalt": "Purifying Salt"}[ability]
        if scene.safeguard:
            return kind, "Safeguard"
        if ability == "leafguard" and scene.weather == "sun":
            return kind, "Leaf Guard in the sun"
        if ability == "flowerveil" and "GRASS" in foe.types:
            return kind, "Flower Veil"
        if "powder" in flags:
            if "GRASS" in foe.types:
                return kind, "Grass: powder does nothing to it"
            if ability in _NO_POWDER_ABILITIES:
                return kind, "Overcoat"
            if item in _NO_POWDER_ITEMS:
                return kind, "Safety Goggles"
    elif kind == "perish":
        if any(str(x).startswith("perish") for x in foe.effects):
            return kind, "its count runs already"
    return kind, ""


# =============================================================================================
# Damage
# =============================================================================================

@dataclass(frozen=True)
class Fighter:
    """
    One side of a damage estimate.

    stats        "hp" (the whole HP), "atk", "def", "spa", "spd", "spe": ours from the request, theirs estimated
    boosts       stat stages
    types        types now (the Tera type alone once terastallized)
    base_types   types before Tera (STAB with a Tera that has not happened yet)
    tera         the Tera type it can use ("" when none or unknown)
    terastallized  it has used it
    hp           current HP as a share of the whole
    """
    level: int = 100
    stats: Dict[str, float] = field(default_factory=dict)
    boosts: Dict[str, int] = field(default_factory=dict)
    types: Tuple[str, ...] = ()
    base_types: Tuple[str, ...] = ()
    tera: str = ""
    terastallized: bool = False
    ability: str = ""
    item: str = ""
    status: str = ""
    hp: float = 1.0


def hits_defence(move: Union[str, Dict[str, Any]]) -> str:
    """
    The defence a move's damage is reckoned against, boosts included: "def" or "spd"; "" for a
    move whose damage does not depend on the target's defensive stages (fixed damage, Sacred
    Sword and the like) or that does no damage. Psyshock, Psystrike and Secret Sword: "def".
    """
    e = _entry(move)
    category = str(e.get("category") or "").upper()
    if category not in ("PHYSICAL", "SPECIAL"):
        return ""
    if e.get("damage") or e.get("ignoreDefensive") or e.get("ohko"):
        return ""
    if sm.to_id(e.get("name") or (move if isinstance(move, str) else "")) in _VARIABLE_CATEGORY:
        return ""
    over = str(e.get("overrideDefensiveStat") or "")
    if over in ("def", "spd"):
        return over
    return "def" if category == "PHYSICAL" else "spd"


# Attacks "don't hit the boosted defence" never swaps IN: they knock our own Pokemon out, lose a
# turn, lock it in, switch it out, or fail in the very positions the rule acts in (Sucker Punch
# against a set-up move). Found on the replay of this morning's games (bench/counter_replay.py):
# the first version swapped a Psychic for an Explosion.
_NO_SWAP_IDS = ("fakeout", "firstimpression", "focuspunch", "dreameater", "belch", "steelbeam", "mindblown", "lastresort",
                "synchronoise", "burnup", "doubleshock", "poltergeist", "suckerpunch", "thunderclap", "upperhand", "rollout",
                "iceball", "futuresight", "doomdesire", "counter", "mirrorcoat", "metalburst", "comeuppance", "bide",
                "finalgambit", "endeavor", "skydrop")


def swap_doubt(move: Union[str, Dict[str, Any]]) -> str:
    """Why an attack is never swapped in by "don't hit the boosted defence" ("knocks itself out", ...), or ""."""
    e = _entry(move)
    move_id = sm.to_id(e.get("name") or (move if isinstance(move, str) else ""))
    if e.get("selfdestruct"):
        return "knocks itself out"
    status = str((e.get("self") or {}).get("volatileStatus") or "")
    if status == "mustrecharge":
        return "must recharge"
    if status == "lockedmove":
        return "locks it in"
    if "charge" in (e.get("flags") or {}):
        return "charges a turn"
    if e.get("selfSwitch"):
        return "switches out"
    if move_id in _NO_SWAP_IDS:
        return "works only at times"
    if move_id in _VARIABLE_CATEGORY:
        return "its category depends on the position"
    return ""


def _hits(e: Dict[str, Any], ability: str, item: str) -> float:
    multihit = e.get("multihit")
    if isinstance(multihit, (list, tuple)) and len(multihit) == 2:
        lo, hi = int(multihit[0]), int(multihit[1])
        if lo == hi:
            return float(lo)
        if ability == "skilllink":
            return float(hi)
        if item == "loadeddice":
            return 4.0 if hi >= 5 else float(hi)
        return 3.1 if (lo, hi) == (2, 5) else (lo + hi) / 2.0
    if isinstance(multihit, (int, float)) and multihit:
        return float(multihit)
    return 1.0


def damage_share(move: Union[str, Dict[str, Any]], att: Fighter, dfn: Fighter, tera: bool = False, roll: float = 0.925,
                 weather: str = "") -> Optional[float]:
    """
    The damage of `move` (an id, or a data entry) from att to dfn as a share of dfn's whole HP,
    at the given damage roll; None when it cannot be told (a status move, a power the data does
    not give, a stat missing). tera: att terastallizes with it (STAB on its Tera type).

    Base power, the two stats with their stages (Unaware on either side, moves that ignore the
    target's stages, Psyshock's Defence, Body Press, Foul Play), STAB (Tera, Adaptability), type,
    burn, multi-hit, weather, the common items and abilities when known (Choice Band and Specs,
    Life Orb, Expert Belt, Assault Vest, Eviolite, Huge Power, Technician, Fur Coat, Ice Scales,
    Fluffy, Multiscale, Thick Fat, the immunities...). No critical hits, screens or spreads.
    """
    try:
        e = _entry(move)
        category = str(e.get("category") or "").upper()
        if category not in ("PHYSICAL", "SPECIAL"):
            return None
        move_type = str(e.get("type") or "").upper()
        flags = set((e.get("flags") or {}).keys())
        hp_total = float(dfn.stats.get("hp") or 0)
        if hp_total <= 0:
            return None
        eff = effectiveness(move_type, dfn.types)
        if att.ability == "scrappy" and move_type in ("NORMAL", "FIGHTING") and "GHOST" in dfn.types and eff == 0:
            eff = effectiveness(move_type, [t for t in dfn.types if t != "GHOST"])
        if eff <= 0:
            return 0.0
        if move_type in _IMMUNE.get(dfn.ability, ()):
            return 0.0
        if dfn.ability == "wonderguard" and eff <= 1.0:
            return 0.0
        fixed = e.get("damage")
        if fixed == "level":
            return float(att.level) / hp_total
        if isinstance(fixed, (int, float)) and fixed:
            return float(fixed) / hp_total
        power = float(e.get("basePower") or 0)
        if power <= 0:
            return None
        physical = category == "PHYSICAL"
        off_key = str(e.get("overrideOffensiveStat") or ("atk" if physical else "spa"))
        source = dfn if str(e.get("overrideOffensivePokemon") or "") == "target" else att
        a = float(source.stats.get(off_key) or 0)
        d_key = str(e.get("overrideDefensiveStat") or ("def" if physical else "spd"))
        d = float(dfn.stats.get(d_key) or 0)
        if a <= 0 or d <= 0:
            return None
        a_stage = int(source.boosts.get(off_key, 0) or 0)
        if dfn.ability == "unaware" and source is att:
            a_stage = 0
        d_stage = int(dfn.boosts.get(d_key, 0) or 0)
        if e.get("ignoreDefensive") or att.ability == "unaware":
            d_stage = 0
        a *= stage(a_stage)
        d *= stage(d_stage)
        if physical:
            if att.ability in ("hugepower", "purepower"):
                a *= 2.0
            if att.ability in ("hustle", "gorillatactics") or (att.ability == "guts" and att.status):
                a *= 1.5
            if att.item == "choiceband":
                a *= 1.5
        elif att.item == "choicespecs":
            a *= 1.5
        if dfn.item == "assaultvest" and d_key == "spd":
            d *= 1.5
        if dfn.item == "eviolite":
            d *= 1.5
        if dfn.ability == "furcoat" and d_key == "def":
            d *= 2.0
        if dfn.ability == "marvelscale" and dfn.status and d_key == "def":
            d *= 1.5
        if weather == "sand" and "ROCK" in dfn.types and d_key == "spd":
            d *= 1.5
        if weather == "snow" and "ICE" in dfn.types and d_key == "def":
            d *= 1.5
        if att.ability == "technician" and power <= 60:
            power *= 1.5
        flag_boost = _FLAG_BOOST.get(att.ability)
        if flag_boost and flag_boost[0] in flags:
            power *= flag_boost[1]
        if att.ability == "sheerforce" and (e.get("secondary") or e.get("secondaries")):
            power *= 1.3
        type_boost = _TYPE_ABILITY.get(att.ability)
        if type_boost and type_boost[0] == move_type:
            power *= type_boost[1]
        base = math.floor(math.floor(math.floor(2 * int(att.level) / 5 + 2) * power * a / d) / 50) + 2
        # STAB: the types before Tera, and the Tera type once used or with this move.
        before = att.base_types or att.types
        tera_type = att.tera if (tera or att.terastallized) else ""
        in_base = move_type in before
        stab = 1.0
        if tera_type and move_type == tera_type and in_base:
            stab = 2.25 if att.ability == "adaptability" else 2.0
        elif in_base or (tera_type and move_type == tera_type):
            stab = 2.0 if att.ability == "adaptability" else 1.5
        mods = stab * eff
        if physical and att.status == "brn" and att.ability != "guts" and sm.to_id(e.get("name") or "") != "facade":
            mods *= 0.5
        if weather == "rain":
            mods *= 1.5 if move_type == "WATER" else (0.5 if move_type == "FIRE" else 1.0)
        elif weather == "sun":
            mods *= 1.5 if move_type == "FIRE" else (0.5 if move_type == "WATER" else 1.0)
        if att.item == "lifeorb":
            mods *= 1.3
        if att.item == "expertbelt" and eff > 1.0:
            mods *= 1.2
        if att.item == "muscleband" and physical:
            mods *= 1.1
        if att.item == "wiseglasses" and not physical:
            mods *= 1.1
        if att.ability == "tintedlens" and eff < 1.0:
            mods *= 2.0
        if move_type in _HALVED.get(dfn.ability, ()):
            mods *= 0.5
        if dfn.ability == "fluffy":
            mods *= (0.5 if "contact" in flags else 1.0) * (2.0 if move_type == "FIRE" else 1.0)
        if dfn.ability == "icescales" and not physical:
            mods *= 0.5
        if dfn.ability in ("multiscale", "shadowshield") and dfn.hp >= 1.0:
            mods *= 0.5
        if dfn.ability in ("filter", "solidrock", "prismarmor") and eff > 1.0:
            mods *= 0.75
        return base * mods * float(roll) * _hits(e, att.ability, att.item) / hp_total
    except Exception:
        return None


def two_hit_ko(threat: Fighter, moves: Sequence[str], ours: Fighter, weather: str = "") -> Tuple[bool, float]:
    """
    (whether the opponent's Pokemon `threat`, with its boosts, is estimated to knock `ours` out in
    two hits at the highest damage roll, its best hit as a share of our whole HP). `moves` are
    its known attacks; with none known, a STAB 90 base power hit of each of its types, physical
    or special by its better attacking stat with boosts. An Unaware Pokemon of ours does not
    see the boosts (damage_share).
    """
    entries: List[Dict[str, Any]] = []
    for m in moves or ():
        e = sm.move_entry(m)
        if str(e.get("category") or "") in ("Physical", "Special"):
            entries.append(e)
    if not entries:
        atk = float(threat.stats.get("atk") or 0) * stage(int(threat.boosts.get("atk", 0) or 0))
        spa = float(threat.stats.get("spa") or 0) * stage(int(threat.boosts.get("spa", 0) or 0))
        category = "Physical" if atk >= spa else "Special"
        entries = [{"category": category, "basePower": 90, "type": t.capitalize(), "flags": {}} for t in (threat.types or ("NORMAL",))]
    best = 0.0
    for e in entries:
        share = damage_share(e, threat, ours, roll=1.0, weather=weather)
        if share is not None and share > best:
            best = share
    return 2.0 * best >= float(ours.hp), best


# =============================================================================================
# The board: what mix() is given
# =============================================================================================

@dataclass(frozen=True)
class Board:
    """
    What the counter rules know at one decision, on our 14 actions (built by build_board).

    forced        the request is a forced switch (our Pokemon fainted, or U-turn): the rules do nothing
    boosts        the opponent's active Pokemon's stat stages
    top_boost     its largest positive stage in attack, defences, special attack, speed or evasion
    counters      our active's WORKING counters: plain move action (0-3) -> "kind move" ("haze haze")
    failed        our active's counters that cannot work now: action -> why
    unaware       our active Pokemon has Unaware (the boosts do nothing against it)
    holders       legal switches to a benched Pokemon that could answer: action -> (its HP share, what it has)
                  (at least MixSettings.counter_hp of its HP, not two-hit KO'd by the boosted Pokemon, a
                  working counter move or Unaware)
    passed        the other benched Pokemon with a counter: action -> why not
    hits          our legal attacks (0-3, 10-13) -> the defence they hit ("def", "spd", "" for neither)
    damage        our legal attacks -> estimated damage as a share of the opponent's whole HP (Tera forms with Tera)
    no_swap       our legal attacks that "don't hit the boosted defence" never swaps in -> why (swap_doubt: Explosion...)
    accuracy      our legal attacks -> the share of the time they hit (1.0 for those that cannot miss)
    priority      our legal attacks -> their priority (only those not 0)
    outspeeds     our active moves before the opponent's at the same priority (None: not known)
    threat_hit    the opponent's best hit on our active, as a share of its whole HP (two_hit_ko), at the top roll
    names         every legal action -> its name, for the record
    """
    forced: bool = False
    boosts: Dict[str, int] = field(default_factory=dict)
    top_boost: int = 0
    counters: Dict[int, str] = field(default_factory=dict)
    failed: Dict[int, str] = field(default_factory=dict)
    unaware: bool = False
    holders: Dict[int, Tuple[float, str]] = field(default_factory=dict)
    passed: Dict[int, str] = field(default_factory=dict)
    hits: Dict[int, str] = field(default_factory=dict)
    damage: Dict[int, float] = field(default_factory=dict)
    no_swap: Dict[int, str] = field(default_factory=dict)
    accuracy: Dict[int, float] = field(default_factory=dict)
    priority: Dict[int, int] = field(default_factory=dict)
    outspeeds: Optional[bool] = None
    threat_hit: float = 0.0
    names: Dict[int, str] = field(default_factory=dict)


def _status(pokemon: Any) -> str:
    status = getattr(pokemon, "status", None)
    return str(getattr(status, "name", status) or "").lower() if status is not None else ""


def _effects(pokemon: Any) -> Tuple[str, ...]:
    out = []
    for effect in (getattr(pokemon, "effects", None) or {}):
        name = str(getattr(effect, "name", effect)).lower()
        if name.startswith("perish"):
            out.append("perish")
        elif name in ("substitute", "ingrain", "encore", "taunt", "yawn"):
            out.append(name)
    return tuple(sorted(set(out)))


def _unanimous(worlds: Sequence[Dict[str, Any]], key: str) -> str:
    values = [sm.to_id((w.get("active") or {}).get(key)) for w in worlds or () if isinstance(w, dict)]
    values = [v for v in values if v]
    if len(values) >= 2 and len(set(values)) == 1:
        return values[0]
    return ""


def foe_ability(pokemon: Any, worlds: Sequence[Dict[str, Any]] = ()) -> str:
    """The opponent's ability when known: shown or the only one it can have (poke-env), else every sampled world's; "" otherwise."""
    known = sm.to_id(getattr(pokemon, "ability", None) or "")
    if known:
        return known
    possible = [sm.to_id(a) for a in (getattr(pokemon, "possible_abilities", None) or ()) if a]
    if len(set(possible)) == 1:
        return possible[0]
    return _unanimous(worlds, "ability")


def foe_item(pokemon: Any, worlds: Sequence[Dict[str, Any]] = ()) -> str:
    """The opponent's item when known, "none" when known to have none, else every sampled world's; "" otherwise."""
    item = getattr(pokemon, "item", "unknown_item")
    if item is None or item == "":
        return "none"
    item = sm.to_id(item)
    if item and item != "unknownitem":
        return item
    return _unanimous(worlds, "item")


def _weather(battle: Any) -> str:
    for w in (getattr(battle, "weather", None) or {}):
        name = str(getattr(w, "name", w)).lower()
        for word, out in (("rain", "rain"), ("primordial", "rain"), ("sun", "sun"), ("desolate", "sun"), ("sand", "sand"),
                          ("snow", "snow"), ("hail", "snow")):
            if word in name:
                return out
    return ""


def _terrain(battle: Any) -> str:
    for f in (getattr(battle, "fields", None) or {}):
        name = str(getattr(f, "name", f)).lower()
        if name.startswith("electric"):
            return "electric"
        if name.startswith("misty"):
            return "misty"
    return ""


def fighter_of(pokemon: Any, ours: bool, ability: Optional[str] = None, item: Optional[str] = None) -> Fighter:
    """A poke-env Pokemon as a Fighter: ours with the stats of the request, theirs estimated as a random battle set."""
    level = int(getattr(pokemon, "level", 0) or 100)
    if ours:
        stats = {k: float(v) for k, v in (getattr(pokemon, "stats", None) or {}).items() if v}
        stats["hp"] = float(getattr(pokemon, "max_hp", 0) or 0) or estimated_stats(getattr(pokemon, "base_stats", {}), level).get("hp", 0.0)
        for k, v in estimated_stats(getattr(pokemon, "base_stats", {}), level).items():
            stats.setdefault(k, v)
    else:
        stats = estimated_stats(getattr(pokemon, "base_stats", {}), level)
    base_types = _types([getattr(pokemon, "_type_1", None), getattr(pokemon, "_type_2", None)]) or _types(getattr(pokemon, "types", ()))
    tera = getattr(pokemon, "tera_type", None)
    return Fighter(level=level, stats=stats, boosts={k: int(v) for k, v in (getattr(pokemon, "boosts", None) or {}).items()},
                   types=_types(getattr(pokemon, "types", ())), base_types=base_types,
                   tera=str(getattr(tera, "name", tera) or "").upper() if tera is not None else "",
                   terastallized=bool(getattr(pokemon, "is_terastallized", False)),
                   ability=sm.to_id(ability if ability is not None else (getattr(pokemon, "ability", None) or "")),
                   item=sm.to_id(item if item is not None else (getattr(pokemon, "item", None) or "")),
                   status=_status(pokemon), hp=float(getattr(pokemon, "current_hp_fraction", 1.0) or 0.0))


_SPEED_WEATHER = {"swiftswim": "rain", "chlorophyll": "sun", "sandrush": "sand", "slushrush": "snow"}


def speed(f: Fighter, weather: str = "") -> float:
    """A Fighter's speed with its stage, Choice Scarf, paralysis and the weather abilities when known (0 when not known)."""
    v = float(f.stats.get("spe") or 0) * stage(int(f.boosts.get("spe", 0) or 0))
    if f.item == "choicescarf":
        v *= 1.5
    if f.status == "par" and f.ability != "quickfeet":
        v *= 0.5
    if f.ability == "quickfeet" and f.status:
        v *= 1.5
    if weather and _SPEED_WEATHER.get(f.ability) == weather:
        v *= 2.0
    return v


def full_hp_guard(pokemon: Any, worlds: Sequence[Dict[str, Any]] = ()) -> str:
    """
    What could let the opponent's Pokemon live through any one hit at full HP (Sturdy, Focus Sash,
    Multiscale, Shadow Shield, Disguise, Ice Face), or "" when that is ruled out: by its known
    ability (else every sampled world's, at least two of them, else every possible one) and its
    known item (else at least two sampled worlds, none with a Focus Sash).
    """
    try:
        ws = [w for w in worlds or () if isinstance(w, dict) and isinstance(w.get("active"), dict)]
        known = sm.to_id(getattr(pokemon, "ability", None) or "")
        if known:
            abilities = [known]
        else:
            seen = [sm.to_id(w["active"].get("ability")) for w in ws]
            if len(ws) >= 2 and all(seen):
                abilities = sorted(set(seen))
            else:
                abilities = [sm.to_id(a) for a in (getattr(pokemon, "possible_abilities", None) or ()) if a]
        for ab in abilities:
            if ab in _FULL_HP_ABILITIES:
                return ab
        item = getattr(pokemon, "item", "unknown_item")
        if item is None or item == "":
            return ""
        item = sm.to_id(item)
        if item == "focussash":
            return "focussash"
        if item and item != "unknownitem":
            return ""
        items = [sm.to_id(w["active"].get("item")) for w in ws]
        if len(ws) >= 2 and all(items) and "focussash" not in items:
            return ""
        return "focussash?"
    except Exception:
        return "unknown"


def _positive(boosts: Dict[str, Any]) -> int:
    return sum(max(0, int(v or 0)) for k, v in (boosts or {}).items() if k != "accuracy")


def _move_id(target: Any) -> str:
    return sm.to_id(getattr(target, "id", None) or "")


def build_board(battle: Any, actions: Dict[int, Tuple[Any, Dict[str, Any]]], threat: Any = None, advice: Any = None,
                read_floor: float = 0.50, holder_hp: float = 0.50) -> Board:
    """
    The Board of one decision from poke-env's battle. `actions` maps every legal action to what
    the player's assembler orders for it (map_index_to_order: a Move, or a Pokemon to switch to,
    and its keywords). threat is the decision's search_player.Threat (last_setup, setup_read),
    advice the search's answer (its sampled sets of the opponent's active Pokemon).
    """
    forced = bool(getattr(battle, "force_switch", False))
    theirs = getattr(battle, "opponent_active_pokemon", None)
    ours = getattr(battle, "active_pokemon", None)
    worlds = [w for w in (getattr(advice, "opponent_worlds", None) or []) if isinstance(w, dict)]
    names: Dict[int, str] = {}
    for a, (target, kw) in actions.items():
        if hasattr(target, "current_hp"):
            names[a] = "switch " + sm.species_name(str(getattr(target, "species", target)))
        else:
            names[a] = sm.move_name(_move_id(target)) + (" + Tera" if (kw or {}).get("terastallize") else "")
    if theirs is None:
        return Board(forced=forced, names=names)
    boosts = {k: int(v) for k, v in (getattr(theirs, "boosts", None) or {}).items()}
    top = max([v for k, v in boosts.items() if k in ("atk", "def", "spa", "spd", "spe", "evasion") and v > 0] or [0])
    ability = foe_ability(theirs, worlds)
    item = foe_item(theirs, worlds)
    team = dict(getattr(battle, "opponent_team", None) or {})
    size = 6
    try:
        role = getattr(battle, "opponent_role", None) or ("p2" if getattr(battle, "player_role", "p1") == "p1" else "p1")
        size = int((getattr(battle, "_team_size", None) or {}).get(role) or 6)
    except Exception:
        pass
    others = max(0, size - len(team)) + sum(1 for m in team.values() if m is not theirs and not getattr(m, "fainted", False))
    try:
        grounded = bool(battle.is_grounded(theirs))
    except Exception:
        grounded = False            # unknown: taken to float, so terrain does not stop the sleep move
    foe = Foe(species=str(getattr(theirs, "species", "") or ""), types=_types(getattr(theirs, "types", ())), boosts=boosts,
              ability=ability, item=item, status=_status(theirs), effects=_effects(theirs), grounded=grounded,
              last_setup=bool(getattr(threat, "last_setup", False)), others=others)
    weather = _weather(battle)
    safeguard = any("safeguard" in str(getattr(c, "name", c)).lower() for c in (getattr(battle, "opponent_side_conditions", None) or {}))
    scene = Scene(terrain=_terrain(battle),
                  sleep_used=any(_status(m) == "slp" and not getattr(m, "fainted", False) for m in team.values()),
                  safeguard=safeguard, weather=weather)
    read = float(getattr(threat, "setup_read", 0.0) or 0.0)
    our_ability = sm.to_id(getattr(ours, "ability", None) or "") if ours is not None else ""
    counters: Dict[int, str] = {}
    failed: Dict[int, str] = {}
    hits: Dict[int, str] = {}
    damage: Dict[int, float] = {}
    no_swap: Dict[int, str] = {}
    foe_fighter = fighter_of(theirs, ours=False, ability=ability, item=item if item != "none" else "")
    our_fighter = fighter_of(ours, ours=True) if ours is not None else None
    threat_moves = sorted({sm.to_id(m) for m in (getattr(theirs, "moves", None) or {})}
                          | {sm.to_id(m) for w in worlds for m in ((w.get("active") or {}).get("moves") or [])})
    accuracy: Dict[int, float] = {}
    priority: Dict[int, int] = {}
    outspeeds: Optional[bool] = None
    threat_hit = 0.0
    our_status = ""
    our_hp = 1.0
    if our_fighter is not None:
        ours_speed, theirs_speed = speed(our_fighter, weather), speed(foe_fighter, weather)
        if ours_speed > 0 and theirs_speed > 0:
            trick_room = any("trick" in str(getattr(f, "name", f)).lower() for f in (getattr(battle, "fields", None) or {}))
            outspeeds = (ours_speed < theirs_speed) if trick_room else (ours_speed > theirs_speed)
        _ko2, threat_hit = two_hit_ko(foe_fighter, threat_moves, our_fighter, weather=weather)
        our_status = our_fighter.status
        our_hp = float(our_fighter.hp)
    for a, (target, kw) in actions.items():
        if hasattr(target, "current_hp") or a in range(4, 10):
            continue
        move = _move_id(target)
        if not move:
            continue
        e = sm.move_entry(move)
        prio = int(e.get("priority") or 0)
        if our_ability == "prankster" and str(e.get("category") or "") == "Status":
            prio += 1
        if a < 4:
            kind, why = counter_check(move, foe, scene, read=read, now=True, user_ability=our_ability, read_floor=read_floor,
                                      outspeeds=outspeeds is not False)
            if kind and not why:
                # Our side of it: asleep or frozen we cannot use it; a hit that knocks us out first wastes it;
                # Haze that would clear more of our boosts than of theirs.
                if our_status in ("slp", "frz"):
                    why = "our Pokemon is %s" % ("asleep" if our_status == "slp" else "frozen")
                elif prio < 0 and threat_hit >= our_hp:
                    why = "it moves last, and a hit of %d%% would knock ours out first (%d%% left)" % (round(100 * threat_hit), round(100 * our_hp))
                elif prio == 0 and outspeeds is not True and threat_hit >= our_hp:
                    why = "it moves first, and a hit of %d%% would knock ours out (%d%% left)" % (round(100 * threat_hit), round(100 * our_hp))
                elif kind == "haze" and ours is not None and _positive(getattr(ours, "boosts", None) or {}) > _positive(boosts):
                    why = "Haze would clear more of our boosts than of theirs"
            if kind and not why:
                counters[a] = "%s %s" % (kind, move)
            elif kind:
                failed[a] = "%s: %s" % (sm.move_name(move), why)
        side = hits_defence(move)
        if str(e.get("category") or "") in ("Physical", "Special"):
            hits[a] = side
            acc = e.get("accuracy", True)
            accuracy[a] = 1.0 if acc is True or not isinstance(acc, (int, float)) else max(0.0, min(1.0, float(acc) / 100.0))
            if prio:
                priority[a] = prio
            doubt = swap_doubt(move)
            if doubt:
                no_swap[a] = doubt
            if our_fighter is not None:
                share = damage_share(move, our_fighter, foe_fighter, tera=a >= 10, weather=weather)
                if share is not None:
                    damage[a] = round(share, 4)
    # Benched Pokemon that could come in and answer.
    holders: Dict[int, Tuple[float, str]] = {}
    passed: Dict[int, str] = {}
    for a, (target, _kw) in actions.items():
        if not hasattr(target, "current_hp"):
            continue
        mon_ability = sm.to_id(getattr(target, "ability", None) or "")
        has = []
        for m in (getattr(target, "moves", None) or {}):
            kind, why = counter_check(m, foe, scene, read=read, now=False, user_ability=mon_ability, read_floor=read_floor)
            if kind and not why:
                has.append(sm.move_name(m))
        if mon_ability == "unaware":
            has.append("Unaware")
        if not has:
            continue
        hp = float(getattr(target, "current_hp_fraction", 0.0) or 0.0)
        if hp < holder_hp:
            passed[a] = "%s at %d%% HP" % (names.get(a, "?"), round(100 * hp))
            continue
        ko2, best = two_hit_ko(foe_fighter, threat_moves, fighter_of(target, ours=True), weather=weather)
        if ko2:
            passed[a] = "%s: two hits of %d%% each would knock it out" % (names.get(a, "?"), round(100 * best))
            continue
        holders[a] = (round(hp, 4), ", ".join(has))
    return Board(forced=forced, boosts=boosts, top_boost=int(top), counters=counters, failed=failed,
                 unaware=our_ability == "unaware", holders=holders, passed=passed, hits=hits, damage=damage, no_swap=no_swap,
                 accuracy=accuracy, priority=priority, outspeeds=outspeeds, threat_hit=round(float(threat_hit), 4), names=names)
