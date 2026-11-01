"""
What real opponents do: one row per opponent decision, from the recorded ladder games.

    python bench/opp_dataset.py                     # writes runs/opp_pred/dataset.jsonl and prints the counts
    python bench/opp_dataset.py --battles <dir> --out <dir>

Source (READ ONLY): C:/Users/you/showdown-bot/battles/<date>/battle-*.json, the rated games an
older bot of the owner played against people. A recording holds the battle log as OUR side saw it
(our HP exact, theirs in percent), our team, and what our bot decided at each request.

A row is one decision of the OPPONENT:

    kind   "turn"    what it chose at the start of a turn (a move or a switch)
           "forced"  whom it sent in after a faint
           "pivot"   whom it sent in after its own U-turn, Baton Pass, Eject Button and the like
    label  {"action": "move earthquake" | "switch garchomp" | None, "type", "name", "tera", "known",
            "why_unknown", "locked", "target_revealed", "move_revealed"}
           known = False when the log cannot say what it chose: it fainted before it moved, it could
           not move (sleep, paralysis, flinch) or the game ended in that turn. locked = True when it
           had no choice (the second turn of Outrage or of a charging move, a recharge turn).
    ctx    what was known to us BEFORE that decision: both actives, both benches as revealed, the
           opponent's revealed moves with how often each was used, its last action, the field, and
           for the opponent's active the move pool of its species in random battles.
    our_action   what WE did in that turn. NOT context (it is simultaneous): kept for analysis.

Names: a game is "g" + 12 hex of a salted hash of its id, an opponent "opp" + 10 hex. The real
names stay in the source files and are written nowhere by this script.

The split is by GAME: sorted game hashes, shuffled with a fixed seed, the first 70% train.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from typing import Any, Dict, Iterator, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATTLES_DIR = os.environ.get("OPP_BATTLES_DIR") or r"C:\Users\you\showdown-bot\battles"
if not os.path.isdir(BATTLES_DIR) and os.path.isdir("/mnt/c/Users/you/showdown-bot/battles"):
    BATTLES_DIR = "/mnt/c/Users/you/showdown-bot/battles"
OUT_DIR = os.path.join(REPO, "runs", "opp_pred")
SPLIT_SEED = 20261006
TRAIN_FRACTION = 0.70
# The salt is private: set OPP_DATASET_SALT to get the same hashes across runs. Unset, a random salt is
# drawn per process, so a hashed name cannot be reversed by hashing known ladder names.
_SALT = (os.environ.get("OPP_DATASET_SALT") or os.urandom(16).hex()) + "|opp-dataset|v1|"

PIVOT_MOVES = {"uturn", "voltswitch", "flipturn", "partingshot", "batonpass", "teleport", "chillyreception", "shedtail"}
_SET_FILES = [os.path.join(REPO, "heuristic", "cache", "gen9randombattle.json"),
              r"C:\Users\you\ps-ppo\heuristic\cache\gen9randombattle.json",
              "/mnt/c/Users/you/ps-ppo/heuristic/cache/gen9randombattle.json"]


def to_id(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def game_hash(game_id: str) -> str:
    return "g" + hashlib.sha1((_SALT + "game|" + game_id).encode("utf-8")).hexdigest()[:12]


def name_hash(name: str) -> str:
    return "opp" + hashlib.sha1((_SALT + "name|" + to_id(name)).encode("utf-8")).hexdigest()[:10]


# =============================================================================================
# Static data: types, moves, the random battle move pools
# =============================================================================================

class Dex:
    """Types and moves from poke-env's data when it is installed, the random battle sets from the cache."""

    def __init__(self) -> None:
        self.pokedex: Dict[str, Any] = {}
        self.moves: Dict[str, Any] = {}
        self.type_chart: Dict[str, Dict[str, float]] = {}
        self.pools: Dict[str, List[str]] = {}
        self.sets_file: Optional[str] = None
        try:
            from poke_env.data import GenData
            data = GenData.from_gen(9)
            self.pokedex, self.moves, self.type_chart = data.pokedex, data.moves, data.type_chart
        except Exception as exc:  # the dataset is still built, without types and move data
            sys.stderr.write("opp_dataset: poke-env data not available (%r): no types in the rows\n" % (exc,))
        for path in _SET_FILES:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as fh:
                    sets = json.load(fh)
                for species, entry in sets.items():
                    pool: List[str] = []
                    for role in (entry.get("roles") or {}).values():
                        for move in role.get("moves") or []:
                            if to_id(move) not in pool:
                                pool.append(to_id(move))
                    self.pools[to_id(species)] = pool
                self.sets_file = path
                break

    def types(self, species: str) -> List[str]:
        entry = self.pokedex.get(to_id(species)) or self.pokedex.get(to_id(str(species).split("-")[0])) or {}
        return [str(t).upper() for t in entry.get("types") or []]

    def pool(self, species: str) -> List[str]:
        return list(self.pools.get(to_id(species)) or self.pools.get(to_id(str(species).split("-")[0])) or [])

    def effectiveness(self, attack_type: str, defender_types: List[str]) -> float:
        out = 1.0
        for defender in defender_types:
            out *= float((self.type_chart.get(str(defender).upper()) or {}).get(str(attack_type).upper(), 1.0))
        return out

    def move(self, move_id: str) -> Dict[str, Any]:
        return self.moves.get(to_id(move_id)) or {}


_DEX: Optional[Dex] = None


def dex() -> Dex:
    global _DEX
    if _DEX is None:
        _DEX = Dex()
    return _DEX


# =============================================================================================
# The recordings, and the log cut into the batches the server sent
# =============================================================================================

def list_games(battles_dir: str = BATTLES_DIR) -> List[str]:
    return sorted(glob.glob(os.path.join(battles_dir, "*", "battle-*.json")))


def load_game(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        game = json.load(fh)
    game["_path"] = path
    return game


def split_batches(log: List[str]) -> List[List[str]]:
    """
    The log as the server sent it: batch 0 is everything up to "|turn|1", every later batch begins
    with a "|t:|" line (the server writes one each time it resolves choices). After every batch
    but the last the server sent each player a request, and the request ids count up by two per
    batch: batch k carries rqid 2k+2 for p1 and 2k+3 for p2 (checked against the recorded rqids).
    """
    first_turn = next((i for i, line in enumerate(log) if line.startswith("|turn|")), None)
    if first_turn is None:
        return [list(log)]
    batches: List[List[str]] = [list(log[:first_turn + 1])]
    for line in log[first_turn + 1:]:
        if line.startswith("|t:|"):
            if batches[-1] and batches[-1][-1] == "|":
                batches[-1].pop()
                batches.append(["|", line])
            else:
                batches.append([line])
        else:
            batches[-1].append(line)
    return batches


def batch_rqid(index: int, side: str) -> int:
    return 2 * index + 2 + (1 if side == "p2" else 0)


def batch_kind(batch: List[str]) -> str:
    """"end": the battle is over. "move": a new turn begins. "mid": somebody has to switch first."""
    if any(line.startswith("|win|") or line == "|tie" or line.startswith("|tie|") for line in batch):
        return "end"
    if any(line.startswith("|turn|") for line in batch):
        return "move"
    return "mid"


def alignment(game: Dict[str, Any]) -> Dict[str, Any]:
    """Do the recorded decisions sit where the log says a request was sent? (needed for a replay)"""
    side = game["me"]["side"]
    batches = split_batches(game["log"])
    kinds = {batch_rqid(k, side): batch_kind(b) for k, b in enumerate(batches)}
    problems: List[str] = []
    seen: Counter = Counter()
    for decision in game.get("decisions") or []:
        seen[decision.get("rqid")] += 1
        want = "mid" if decision.get("forced") else "move"
        got = kinds.get(decision.get("rqid"))
        if got != want:
            problems.append("rqid %s: a %s decision where the log has %s" % (decision.get("rqid"), want, got))
    for rqid, kind in kinds.items():
        if kind == "move" and rqid not in seen:
            problems.append("rqid %s: a new turn without a recorded decision" % rqid)
    if any(count > 1 for count in seen.values()):
        problems.append("a request was decided twice")
    if game.get("resumed"):
        problems.append("the bot joined this battle midway")
    return {"ok": not problems, "problems": problems, "batches": len(batches)}


# =============================================================================================
# Following a battle from the log
# =============================================================================================

def _hp(condition: str) -> Tuple[float, str, bool]:
    parts = str(condition).split()
    status = parts[1] if len(parts) > 1 else ""
    if status == "fnt" or parts[0] == "0":
        return 0.0, "", True
    try:
        a, b = parts[0].split("/")
        return max(0.0, min(1.0, float(a) / float(b))), status, False
    except Exception:
        return 1.0, status, False


class Mon:
    def __init__(self, name: str, details: str) -> None:
        fields = [f.strip() for f in details.split(",")]
        self.name = name
        self.species = fields[0]
        self.level = next((int(f[1:]) for f in fields[1:] if re.fullmatch(r"L\d+", f)), 100)
        self.hp, self.status, self.fainted = 1.0, "", False
        self.moves: Dict[str, int] = {}
        self.item: Optional[str] = None
        self.item_lost = False
        self.ability: Optional[str] = None
        self.tera: Optional[str] = None
        self.last_move: Optional[str] = None

    def public(self) -> Dict[str, Any]:
        return {"species": self.species, "id": to_id(self.species), "level": self.level, "hp": round(self.hp, 4),
                "status": self.status, "fainted": self.fainted, "moves": dict(self.moves), "item": self.item,
                "item_lost": self.item_lost, "ability": self.ability, "tera": self.tera, "last_move": self.last_move}


class Side:
    def __init__(self) -> None:
        self.mons: Dict[str, Mon] = {}
        self.active: Optional[Mon] = None
        self.boosts: Dict[str, int] = {}
        self.volatiles: List[str] = []
        self.conditions: Dict[str, int] = {}
        self.tera_used = False
        self.teamsize = 6
        self.turns_on_field = 0          # turn starts the active Pokemon has already been on the field for
        self.entered_by = "lead"
        self.last_action: Optional[str] = None
        self.streak = 0                  # how many turns in a row it has clicked the move in last_action
        self.rating: Optional[int] = None
        # this turn
        self.action: Optional[Dict[str, Any]] = None
        self.tera_now: Optional[str] = None
        self.ejecting = False


class GameRows:
    """Walks one recording and yields its rows."""

    def __init__(self, game: Dict[str, Any]) -> None:
        self.game = game
        self.me = game["me"]["side"]
        self.opp = "p2" if self.me == "p1" else "p1"
        self.sides = {"p1": Side(), "p2": Side()}
        self.field: Dict[str, Any] = {"weather": "", "terrain": "", "trick_room": False}
        self.turn = 0
        self.start_ctx: Optional[Dict[str, Any]] = None
        self.gid = game_hash(game["id"])
        self.my_team = {to_id(m.get("speciesForme") or m.get("name") or m["details"].split(",")[0]): m for m in game.get("myTeam") or []}
        self.rows: List[Dict[str, Any]] = []
        self.over = False

    # ---- context ------------------------------------------------------------------------------

    def _side_ctx(self, key: str) -> Dict[str, Any]:
        side = self.sides[key]
        d = dex()
        active = None
        if side.active is not None:
            mon = side.active
            active = mon.public()
            active.update({"boosts": dict(side.boosts), "volatiles": list(side.volatiles), "turns_on_field": side.turns_on_field,
                           "entered_by": side.entered_by, "types": [mon.tera.upper()] if mon.tera else d.types(mon.species),
                           "base_types": d.types(mon.species)})
            if key == self.opp:
                active["pool"] = d.pool(mon.species)
            else:
                mine = self.my_team.get(to_id(mon.species)) or self.my_team.get(to_id(mon.species.split("-")[0])) or {}
                active["own_moves"] = list(mine.get("moves") or [])
                active["own_item"] = mine.get("item")
                active["own_ability"] = mine.get("ability")
                active["own_tera_type"] = mine.get("teraType")
        bench = [m.public() for m in side.mons.values() if m is not side.active]
        revealed = len(side.mons)
        return {"active": active, "bench": bench, "revealed": revealed, "unrevealed": max(0, side.teamsize - revealed),
                "alive": side.teamsize - sum(1 for m in side.mons.values() if m.fainted), "tera_used": side.tera_used,
                "side_conditions": dict(side.conditions), "last_action": side.last_action, "same_move_streak": side.streak,
                "rating": side.rating}

    def _matchup(self) -> Dict[str, Any]:
        d = dex()
        mine, theirs = self.sides[self.me].active, self.sides[self.opp].active
        if mine is None or theirs is None or not d.type_chart:
            return {}
        my_def = [mine.tera.upper()] if mine.tera else d.types(mine.species)
        their_def = [theirs.tera.upper()] if theirs.tera else d.types(theirs.species)
        my_atk = d.types(mine.species) + ([mine.tera.upper()] if mine.tera else [])
        their_atk = d.types(theirs.species) + ([theirs.tera.upper()] if theirs.tera else [])
        we = max([d.effectiveness(t, their_def) for t in my_atk] or [1.0])
        they = max([d.effectiveness(t, my_def) for t in their_atk] or [1.0])
        return {"we_threaten": we, "opp_threatens": they, "opp_disadvantage": we > they, "opp_advantage": they > we}

    def ctx(self) -> Dict[str, Any]:
        return {"turn": self.turn, "opp": self._side_ctx(self.opp), "us": self._side_ctx(self.me),
                "field": dict(self.field), "matchup": self._matchup()}

    # ---- rows ---------------------------------------------------------------------------------

    def _row(self, kind: str, ctx: Dict[str, Any], label: Dict[str, Any], our_action: Any = None) -> None:
        self.rows.append({"game": self.gid, "row": len(self.rows), "kind": kind, "turn": self.turn, "label": label,
                          "ctx": ctx, "our_action": our_action})

    def _switch_label(self, ctx: Dict[str, Any], species: str, tera: bool = False) -> Dict[str, Any]:
        known = {m["id"] for m in ctx["opp"]["bench"]}
        return {"type": "switch", "name": to_id(species), "action": "switch " + to_id(species), "tera": tera, "known": True,
                "why_unknown": None, "locked": False, "target_revealed": to_id(species) in known, "move_revealed": None}

    def _close_turn(self, ended: bool) -> None:
        if self.turn < 1 or self.start_ctx is None:
            return
        ctx = self.start_ctx
        self.start_ctx = None
        opp, me = self.sides[self.opp], self.sides[self.me]
        act = opp.action
        active = ctx["opp"]["active"] or {}
        if act is None:
            label = {"type": "unknown", "name": None, "action": None, "tera": bool(opp.tera_now), "known": False,
                     "why_unknown": "the game ended in this turn" if ended else "no action of the opponent in the log",
                     "locked": False, "target_revealed": None, "move_revealed": None}
        elif act["type"] == "switch":
            label = self._switch_label(ctx, act["name"])
        elif act["type"] == "move":
            label = {"type": "move", "name": act["name"], "action": "move " + act["name"], "tera": bool(opp.tera_now),
                     "tera_type": opp.tera_now, "known": True, "why_unknown": None, "locked": bool(act.get("locked")),
                     "target_revealed": None, "move_revealed": act["name"] in (active.get("moves") or {})}
        else:
            label = {"type": "unknown", "name": None, "action": None, "tera": bool(opp.tera_now), "known": False,
                     "why_unknown": act.get("why"), "locked": bool(act.get("locked")), "target_revealed": None,
                     "move_revealed": None}
        mine = me.action
        our = None
        if mine is not None and mine.get("type") in ("move", "switch"):
            our = {"type": mine["type"], "name": mine["name"], "tera": bool(me.tera_now)}
        self._row("turn", ctx, label, our)
        # bookkeeping for the next turn's context
        for side in (opp, me):
            a = side.action
            if a is not None and a.get("type") == "move":
                name = "move " + a["name"]
                side.streak = side.streak + 1 if side.last_action == name else 1
                side.last_action = name
            elif a is not None and a.get("type") == "switch":
                side.last_action, side.streak = "switch " + to_id(a["name"]), 0

    def _begin_turn(self, number: int) -> None:
        self.turn = number
        self.start_ctx = self.ctx()
        for side in self.sides.values():
            side.action, side.tera_now, side.ejecting = None, None, False
            side.turns_on_field += 1

    # ---- the log ------------------------------------------------------------------------------

    def _mon(self, ident: str) -> Tuple[Optional[str], Optional[Side], Optional[Mon]]:
        key = ident[:2]
        side = self.sides.get(key)
        if side is None:
            return None, None, None
        name = ident.split(":", 1)[1].strip() if ":" in ident else ""
        if len(ident) > 2 and ident[2] == "a":
            return key, side, side.active
        return key, side, side.mons.get(name)

    def _enter(self, key: str, side: Side, parts: List[str], dragged: bool) -> None:
        name = parts[2].split(":", 1)[1].strip()
        details = parts[3]
        note = parts[5] if len(parts) > 5 else ""
        previous = side.active
        acted = side.action is not None
        if self.turn >= 1 and not dragged:
            if key == self.opp:
                ctx = self.ctx() if self.start_ctx is None or acted or (previous is not None and previous.fainted) or side.ejecting else None
                species = details.split(",")[0].strip()
                if previous is not None and previous.fainted:
                    self._row("forced", self.ctx(), self._switch_label(self.ctx(), species))
                    how = "forced"
                elif acted or side.ejecting or "[from]" in note:
                    self._row("pivot", self.ctx(), self._switch_label(self.ctx(), species))
                    how = "pivot"
                elif self.start_ctx is not None:
                    side.action = {"type": "switch", "name": species}
                    how = "switch"
                else:
                    self._row("forced", self.ctx(), self._switch_label(self.ctx(), species))
                    how = "forced"
                del ctx
            else:
                if previous is not None and previous.fainted:
                    how = "forced"
                elif acted or side.ejecting or "[from]" in note:
                    how = "pivot"
                else:
                    side.action = {"type": "switch", "name": details.split(",")[0].strip()}
                    how = "switch"
        else:
            how = "drag" if dragged else "lead"
        mon = side.mons.get(name)
        if mon is None:
            mon = side.mons[name] = Mon(name, details)
        else:
            mon.species = details.split(",")[0].strip()
        if len(parts) > 4:
            mon.hp, mon.status, mon.fainted = _hp(parts[4])
        side.active = mon
        side.boosts, side.volatiles = {}, []
        side.turns_on_field, side.entered_by = 0, how

    def feed(self, line: str) -> None:
        parts = line.split("|")
        if len(parts) < 2:
            return
        kind = parts[1]
        if kind == "turn":
            self._close_turn(False)
            self._begin_turn(int(parts[2]))
            return
        if kind in ("win", "tie"):
            if not self.over:
                self._close_turn(True)
                self.over = True
            return
        if kind == "player" and len(parts) > 5 and parts[2] in self.sides:
            try:
                self.sides[parts[2]].rating = int(parts[5])
            except ValueError:
                pass
            return
        if kind == "teamsize" and len(parts) > 3 and parts[2] in self.sides:
            self.sides[parts[2]].teamsize = int(parts[3])
            return
        if kind in ("switch", "drag"):
            key = parts[2][:2]
            if key in self.sides and len(parts) > 3:
                self._enter(key, self.sides[key], parts, kind == "drag")
            return
        if kind == "-weather":
            self.field["weather"] = "" if parts[2] == "none" else to_id(parts[2])
            return
        if kind == "-fieldstart":
            name = to_id(parts[2].replace("move:", ""))
            if name == "trickroom":
                self.field["trick_room"] = True
            elif name.endswith("terrain"):
                self.field["terrain"] = name
            return
        if kind == "-fieldend":
            name = to_id(parts[2].replace("move:", ""))
            if name == "trickroom":
                self.field["trick_room"] = False
            elif name.endswith("terrain"):
                self.field["terrain"] = ""
            return
        if kind in ("-sidestart", "-sideend") and len(parts) > 3:
            side = self.sides.get(parts[2][:2])
            if side is not None:
                name = to_id(parts[3].replace("move:", ""))
                if kind == "-sidestart":
                    side.conditions[name] = side.conditions.get(name, 0) + 1
                else:
                    side.conditions.pop(name, None)
            return
        if len(parts) < 3:
            return
        key, side, mon = self._mon(parts[2])
        if side is None or mon is None:
            return
        on_field = len(parts[2]) > 2 and parts[2][2] == "a"
        if kind == "move":
            move = to_id(parts[3])
            extras = "|".join(parts[5:])
            called = "[from]" in extras
            locked = "lockedmove" in extras
            if not called or locked:
                if move != "struggle" and not locked:
                    mon.moves[move] = mon.moves.get(move, 0) + 1
                elif locked:
                    mon.moves[move] = mon.moves.get(move, 0)
                mon.last_move = move
                if side.action is None and self.start_ctx is not None:
                    side.action = {"type": "move", "name": move, "locked": locked}
            return
        if kind == "cant":
            reason = to_id(parts[3]) if len(parts) > 3 else ""
            if side.action is None and self.start_ctx is not None:
                side.action = {"type": "cant", "why": "could not move: " + reason, "locked": reason == "recharge"}
            return
        if kind == "faint":
            mon.hp, mon.fainted, mon.status = 0.0, True, ""
            if on_field and side.action is None and self.start_ctx is not None:
                side.action = {"type": "cant", "why": "fainted before it moved", "locked": False}
            return
        if kind in ("-damage", "-heal", "-sethp") and len(parts) > 3:
            mon.hp, status, fainted = _hp(parts[3])
            if len(str(parts[3]).split()) > 1 or fainted:
                mon.status = status
            mon.fainted = fainted
            if "[from] item:" in line and "[of]" not in line:
                mon.item = to_id(line.split("[from] item:")[1].split("|")[0])
            return
        if kind == "-status" and len(parts) > 3:
            mon.status = parts[3]
            return
        if kind == "-curestatus":
            mon.status = ""
            return
        if kind in ("-boost", "-unboost") and len(parts) > 4 and on_field:
            sign = 1 if kind == "-boost" else -1
            side.boosts[parts[3]] = max(-6, min(6, side.boosts.get(parts[3], 0) + sign * int(parts[4])))
            return
        if kind == "-setboost" and len(parts) > 4 and on_field:
            side.boosts[parts[3]] = int(parts[4])
            return
        if kind in ("-clearboost", "-clearallboost") and on_field:
            if kind == "-clearallboost":
                for s in self.sides.values():
                    s.boosts = {}
            else:
                side.boosts = {}
            return
        if kind == "-clearnegativeboost" and on_field:
            side.boosts = {k: v for k, v in side.boosts.items() if v > 0}
            return
        if kind == "-item" and len(parts) > 3:
            mon.item, mon.item_lost = to_id(parts[3]), False
            return
        if kind == "-enditem" and len(parts) > 3:
            if to_id(parts[3]) in ("ejectbutton", "ejectpack") and on_field:
                side.ejecting = True
            mon.item, mon.item_lost = None, True
            return
        if kind == "-ability" and len(parts) > 3:
            mon.ability = to_id(parts[3])
            return
        if kind == "-terastallize" and len(parts) > 3:
            mon.tera = parts[3]
            side.tera_used = True
            side.tera_now = parts[3]
            return
        if kind in ("-start", "-end") and len(parts) > 3 and on_field:
            name = to_id(parts[3].replace("move:", "").replace("ability:", ""))
            if kind == "-start" and name not in side.volatiles:
                side.volatiles.append(name)
            elif kind == "-end" and name in side.volatiles:
                side.volatiles.remove(name)
            return

    def run(self) -> List[Dict[str, Any]]:
        for line in self.game["log"]:
            if isinstance(line, str):
                self.feed(line)
        if not self.over:
            self._close_turn(True)
        opp_rating, my_rating = self.sides[self.opp].rating, self.sides[self.me].rating
        for row in self.rows:
            row["opp_rating"], row["our_rating"] = opp_rating, my_rating
            row["result"] = self.game.get("result")
        return self.rows


def game_rows(game: Dict[str, Any]) -> List[Dict[str, Any]]:
    return GameRows(game).run()


# =============================================================================================
# The split, and the file
# =============================================================================================

def split_of(game_hashes: List[str], seed: int = SPLIT_SEED, train_fraction: float = TRAIN_FRACTION) -> Dict[str, str]:
    order = sorted(set(game_hashes))
    random.Random(seed).shuffle(order)
    cut = int(round(len(order) * train_fraction))
    return {g: ("train" if i < cut else "test") for i, g in enumerate(order)}


def build(battles_dir: str = BATTLES_DIR, out_dir: str = OUT_DIR) -> Dict[str, Any]:
    paths = list_games(battles_dir)
    all_rows: List[Dict[str, Any]] = []
    index: List[Dict[str, Any]] = []
    for path in paths:
        game = load_game(path)
        rows = game_rows(game)
        gid = game_hash(game["id"])
        check = alignment(game)
        index.append({"game": gid, "file": os.path.relpath(path, battles_dir).replace("\\", "/"), "v": game.get("v"),
                      "bot": game.get("bot"), "side": game["me"]["side"], "turns": game.get("turns"), "result": game.get("result"),
                      "rows": len(rows), "decisions": len(game.get("decisions") or []), "has_req": any("req" in d for d in game.get("decisions") or []),
                      "aligned": check["ok"], "alignment_problems": check["problems"][:5], "opponent": name_hash(game["opponent"]["name"])})
        all_rows.extend(rows)
    split = split_of([entry["game"] for entry in index])
    for entry in index:
        entry["split"] = split[entry["game"]]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "dataset.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
        for row in all_rows:
            row["split"] = split[row["game"]]
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    with open(os.path.join(out_dir, "games.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
        for entry in index:
            fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
    return summarize(all_rows, index)


def summarize(rows: List[Dict[str, Any]], index: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"games": len(index), "rows": len(rows), "sets_file": dex().sets_file, "splits": {}}
    for name in ("train", "test"):
        part = [r for r in rows if r["split"] == name]
        turn = [r for r in part if r["kind"] == "turn"]
        scored = [r for r in turn if r["label"]["known"] and not r["label"]["locked"]]
        why = Counter(r["label"]["why_unknown"] for r in turn if not r["label"]["known"])
        out["splits"][name] = {
            "games": sum(1 for e in index if e["split"] == name),
            "rows": len(part), "turn_rows": len(turn),
            "turn_rows_with_a_known_free_choice": len(scored),
            "turn_rows_unknown": sum(1 for r in turn if not r["label"]["known"]),
            "turn_rows_locked": sum(1 for r in turn if r["label"]["locked"]),
            "forced_rows": sum(1 for r in part if r["kind"] == "forced"),
            "pivot_rows": sum(1 for r in part if r["kind"] == "pivot"),
            "moves": sum(1 for r in scored if r["label"]["type"] == "move"),
            "switches": sum(1 for r in scored if r["label"]["type"] == "switch"),
            "tera_turns": sum(1 for r in turn if r["label"]["tera"]),
            "why_unknown": dict(why.most_common()),
        }
    out["games_v2_with_request_parts"] = sum(1 for e in index if e["has_req"])
    out["games_aligned"] = sum(1 for e in index if e["aligned"])
    out["games_aligned_with_request_parts"] = sum(1 for e in index if e["aligned"] and e["has_req"])
    return out


def load_rows(out_dir: str = OUT_DIR, split: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    with open(os.path.join(out_dir, "dataset.jsonl"), "r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if split is None or row["split"] == split:
                yield row


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="One row per opponent decision from the recorded ladder games.")
    parser.add_argument("--battles", default=BATTLES_DIR)
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args(argv)
    summary = build(args.battles, args.out)
    print(json.dumps(summary, indent=1))
    with open(os.path.join(args.out, "dataset_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
