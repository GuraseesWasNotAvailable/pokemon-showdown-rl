"""
read_reward.py: the stage 2 read signal (plan: "Stage 2: reads").

One ReadTracker per battle and seat. It reads the Showdown protocol lines the seat receives, is told
when the seat decides, and labels every decision with:
  read_tier         what the decision earned by reading an opponent's switch (1.0 / 0.6 / 0)
  being_read_tier   the same rule from the opponent's side: what the opponent earned by reading ours
  opp_action        for an auxiliary head: what the opponent did in the turn that followed
                    0 = moved or stayed in, 1..6 = switched to its team slot 0..5 in the order the
                    observation uses (poke-env's battle.opponent_team at the decision), 7 = switched to a
                    Pokemon not seen before, -1 = not applicable (a forced switch, a game that ended
                    before the opponent acted, a switch an item or ability forced)

    tracker = ReadTracker(seat="p1")
    tracker.observe(lines)                 # the lines since the last decision (strings or split parts)
    i = tracker.decide(action, forced=...) # our decision: 0-3 moves, 4-9 switches, 10-13 Tera moves
    ...
    labels = tracker.finalize()            # one DecisionLabel per decide(), in order

A read. Credit goes to the decision a seat makes at the start of a turn; it is scored from that turn's
lines (from the decision to the turn's "upkeep", or the end of the game). All four must hold:
  1. the opponent switched by choice: its switch line comes before any move line of the turn, its
     outgoing Pokemon was still standing, and it made no move that turn. That leaves out faint
     replacements (they come after "upkeep", into an empty slot), pivots (U-turn, Volt Switch, Flip Turn,
     Parting Shot, Teleport, Chilly Reception, Shed Tail, Baton Pass: the switching side moved first),
     drags (Roar, Whirlwind, Dragon Tail, Red Card: "drag" lines), and switches forced by Eject Button,
     Eject Pack, Emergency Exit or Wimp Out (announced just before the switch);
  2. we attacked: our own move line of the turn (not one called by another move or a lock, such as Sleep
     Talk or Outrage's later turns), and our decision was a move (0-3 or 10-13). Tera Blast counts with the
     type it had;
  3. Showdown printed "-supereffective" on the Pokemon that came in, inside our move's lines. A miss,
     an immunity, Protect, a failed move or a status move prints none;
  4. the move would not have been super effective on the Pokemon that left, by matchup.py, with what
     both players could see: its current types or Tera type, its ability if revealed or the only one its
     random-battle sets allow, a revealed Air Balloon, its volatiles, the weather.
  Tier: resisted or immune on the outgoing Pokemon 1.0, neutral 0.6, also super effective 0 (the
  opponent misplayed, we did not read).
  Never counted: Illusion turns (a "replace" line in the turn, or the Pokemon that came in revealed
  later, while still on the field, as an Illusion user in disguise; a Zoroark whose Illusion broke in an
  earlier turn is known for what it is and counts normally), double switches (the reader switched too,
  so did not attack).

Zero-sum by construction: everything above uses only what both players see, and each turn is judged
from the same lines by both seats, so seat A's read_tier is exactly seat B's being_read_tier for the
same turn. The learner can add b * (read_tier - being_read_tier) and the two seats' bonuses cancel.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field as dc_field, replace as dc_replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from matchup import (STELLAR, Field, Mon, dex, known_or_only_ability, load_randbats_sets, matchup,
                     set_abilities, to_id, type_name)

TIER_RESISTED = 1.0     # resisted or immune on the outgoing Pokemon
TIER_NEUTRAL = 0.6
TIER_SUPER = 0.0        # also super effective on the outgoing Pokemon: not a read

OPP_MOVED = 0
OPP_UNSEEN = 7
OPP_NA = -1

SIDES = ("p1", "p2")
_EJECT_ITEMS = {"ejectbutton", "ejectpack"}
_EXIT_ABILITIES = {"emergencyexit", "wimpout"}
_VOLATILE_IDS = {"smackdown", "magnetrise", "ingrain", "telekinesis", "tarshot", "foresight", "miracleeye"}
# Abilities Neutralizing Gas does not switch off (the ones matchup.py looks at).
_CANT_SUPPRESS = {"disguise", "iceface", "multitype"}


def other(side: str) -> str:
    return "p2" if side == "p1" else "p1"


def tier_of(m) -> float:
    if m.immune or m.type_mod < 0:
        return TIER_RESISTED
    return TIER_NEUTRAL if m.type_mod == 0 else TIER_SUPER


def _parts(line) -> Optional[List[str]]:
    """A protocol line as poke-env splits it: ['', 'move', 'p1a: X', ...]."""
    if isinstance(line, (list, tuple)):
        return list(line)
    if not isinstance(line, str) or not line.startswith("|"):
        return None
    return line.rstrip("\r").split("|")


def _ident(ident: str) -> Tuple[Optional[str], Optional[str]]:
    """'p2a: Slowbro' -> ('p2', 'p2: Slowbro'); the key matches poke-env's team dict keys."""
    if not ident or len(ident) < 2 or ident[:2] not in SIDES:
        return None, None
    side = ident[:2]
    name = ident.split(": ", 1)[1] if ": " in ident else ident[3:].strip()
    return side, f"{side}: {name}"


def _species_of(details: str) -> str:
    return details.split(",")[0].strip()


def _hp(s: str) -> Tuple[Optional[int], Optional[int]]:
    s = s.split(" ")[0] if s else ""
    if "/" in s:
        a, b = s.split("/", 1)
        try:
            return int(a), int(b)
        except ValueError:
            return None, None
    if s == "0":
        return 0, None
    return None, None


def _kwargs(parts: Sequence[str]) -> Dict[str, str]:
    kw = {}
    for p in parts[2:]:
        if p.startswith("[") and "]" in p:
            k = p[1:p.index("]")]
            kw[k] = p[p.index("]") + 1:].strip()
    return kw


class _Poke:
    __slots__ = ("key", "side", "species", "form", "tera", "ability", "item", "hp", "maxhp", "fainted",
                 "volatiles", "types", "added_type", "transformed")

    def __init__(self, key: str, side: str, species: str):
        self.key, self.side, self.species = key, side, species
        self.form: Optional[str] = None          # a temporary forme (-formechange), gone on switching out
        self.tera: Optional[str] = None
        self.ability: Optional[str] = None       # revealed current ability
        self.item: Optional[str] = None          # None unknown, '' gone, else the item
        self.hp: Optional[int] = None
        self.maxhp: Optional[int] = None
        self.fainted = False
        self.volatiles: set = set()
        self.types: Optional[Tuple[str, ...]] = None   # typechange / transform
        self.added_type: Optional[str] = None
        self.transformed = False

    def leave_field(self):
        self.form = None
        self.volatiles = set()
        self.types = None
        self.added_type = None
        if self.transformed:
            self.transformed = False


@dataclass
class ReadEvent:
    """One judged opponent switch that the other side hit super effectively."""
    reader: str
    switcher: str
    tier: float
    label: str              # the move against the outgoing Pokemon: immune / resisted / neutral / super
    move: str
    move_type: str
    out_species: str
    in_species: str
    turn: Optional[int]
    region: int             # index of the turn in this tracker
    out_instance: int
    in_instance: int
    excluded: str = ""      # '' or why it does not count (illusion, unknown)
    lines: Optional[List[str]] = None

    @property
    def counts(self) -> bool:
        return not self.excluded


@dataclass
class DecisionLabel:
    index: int
    forced: bool
    action: Optional[int]
    turn: Optional[int]
    region: Optional[int]
    read_tier: float = 0.0
    being_read_tier: float = 0.0
    opp_action: int = OPP_NA
    read: Optional[ReadEvent] = None
    being_read: Optional[ReadEvent] = None


@dataclass
class _Region:
    """The action phase of one turn: from its start to 'upkeep' (or the end of the game)."""
    index: int
    owner: Optional[int]
    turn: Optional[int]
    lines: Optional[List[str]]
    move_seen: bool = False
    any_action: bool = False
    moved: Dict[str, bool] = dc_field(default_factory=lambda: {"p1": False, "p2": False})
    switches: Dict[str, int] = dc_field(default_factory=lambda: {"p1": 0, "p2": 0})
    vswitch: Dict[str, Optional[dict]] = dc_field(default_factory=lambda: {"p1": None, "p2": None})
    forced_switch: Dict[str, bool] = dc_field(default_factory=lambda: {"p1": False, "p2": False})
    pending: Dict[str, Optional[str]] = dc_field(default_factory=lambda: {"p1": None, "p2": None})
    attacks: List[dict] = dc_field(default_factory=list)
    block: Optional[dict] = None
    replace: bool = False
    closed: bool = False


class ReadTracker:
    def __init__(self, seat: Optional[str] = None, sets: Optional[dict] = None, keep_lines: bool = False):
        self.seat = seat
        self.sets = sets if sets is not None else load_randbats_sets()
        self.keep_lines = keep_lines
        self.pokes: Dict[str, _Poke] = {}
        self.active: Dict[str, Optional[str]] = {"p1": None, "p2": None}
        self.instance: Dict[str, Optional[int]] = {"p1": None, "p2": None}
        self.illusion_instances: set = set()
        self.seen: Dict[str, List[str]] = {"p1": [], "p2": []}
        self.names: Dict[str, str] = {}
        self.weather: Optional[str] = None
        self.terrain: Optional[str] = None
        self.gravity = False
        self.turn: Optional[int] = None
        self.started = False
        self.ended = False
        self._next_instance = 0
        self._skip_next = False
        self.regions: List[_Region] = []
        self.region: Optional[_Region] = None
        self.events: List[ReadEvent] = []
        self.decisions: List[DecisionLabel] = []
        self._opp_team: List[Optional[List[str]]] = []
        self.action_mismatch = 0

    # ------------------------------------------------------------------ public API

    def observe(self, lines: Iterable) -> None:
        """Feed protocol lines in order: raw strings ('|move|p1a: X|...', or a multi-line chunk) or
        poke-env's split parts. Requests, chat and timer lines are ignored."""
        if isinstance(lines, str):
            lines = lines.split("\n")
        for line in lines:
            if isinstance(line, str) and "\n" in line:
                self.observe(line.split("\n"))
                continue
            parts = _parts(line)
            if parts is None or len(parts) < 2:
                continue
            self._line(parts)

    def decide(self, action: Optional[int] = None, forced: Optional[bool] = None,
               opp_team: Optional[Sequence[str]] = None) -> int:
        """Our seat makes a decision now (after observing everything before it). forced: True for a
        forced switch (faint replacement, our own pivot or eject); None works it out from the lines.
        opp_team: the opponent's team keys in the observation's order ('p2: Slowbro', ...), as
        list(battle.opponent_team.keys()); None uses the order of first appearance. Returns its index."""
        seat = self.seat
        if forced is None:
            mid_turn = self.region is not None and not self.region.closed and self.region.any_action
            own_empty = seat is not None and self.started and self.active.get(seat) is None
            forced = bool(mid_turn or own_empty)
        idx = len(self.decisions)
        region_idx = None
        if not forced:
            r = self.region
            if r is not None and not r.closed and r.owner is None and not r.any_action:
                r.owner = idx
            else:
                if r is not None and not r.closed:
                    self._close_region()
                self._open_region(owner=idx)
            region_idx = self.region.index
        team = list(opp_team) if opp_team is not None else (
            list(self.seen[other(seat)]) if seat in SIDES else None)
        self.decisions.append(DecisionLabel(index=idx, forced=bool(forced), action=action, turn=self.turn,
                                            region=region_idx))
        self._opp_team.append(team)
        return idx

    def set_action(self, index: int, action: int) -> None:
        """The action of decision `index`, when it is known only after decide()."""
        self.decisions[index].action = action

    def labels(self) -> List[DecisionLabel]:
        """The labels so far (a turn still open is not judged yet; Illusion found later can still
        cancel a read)."""
        self.action_mismatch = 0
        for d in self.decisions:
            d.read_tier, d.being_read_tier, d.read, d.being_read = 0.0, 0.0, None, None
            d.opp_action = OPP_NA
            if d.forced or d.region is None:
                continue
            r = self.regions[d.region]
            if not r.closed or self.seat not in SIDES:
                continue
            d.opp_action = self._opp_action_label(r, other(self.seat), self._opp_team[d.index])
        for ev in self.events:
            # The Pokemon that came in was an Illusion user in disguise (revealed later, while it was
            # still on the field). The one that left cannot be revealed after leaving; if its Illusion
            # had broken earlier, its true species was already known when it left.
            if not ev.excluded and ev.in_instance in self.illusion_instances:
                ev.excluded = "illusion"
            owner = self.regions[ev.region].owner
            if owner is None or not ev.counts:
                continue
            d = self.decisions[owner]
            if ev.reader == self.seat:
                if d.action is not None and not (0 <= d.action <= 3 or 10 <= d.action <= 13):
                    self.action_mismatch += 1
                    continue
                d.read_tier, d.read = ev.tier, ev
            elif ev.switcher == self.seat:
                d.being_read_tier, d.being_read = ev.tier, ev
        return self.decisions

    def finalize(self) -> List[DecisionLabel]:
        """Close the last turn and return the final labels, one per decide() call."""
        if self.region is not None and not self.region.closed:
            self._close_region()
        return self.labels()

    def net(self) -> float:
        """sum(read_tier - being_read_tier) over this seat's decisions."""
        return sum(d.read_tier - d.being_read_tier for d in self.labels())

    def columns(self) -> Tuple[List[float], List[float], List[int]]:
        """read_tier, being_read_tier and opp_action per decision, for the trajectory."""
        ls = self.finalize()
        return [d.read_tier for d in ls], [d.being_read_tier for d in ls], [d.opp_action for d in ls]

    # ------------------------------------------------------------------ turns

    def _open_region(self, owner: Optional[int]) -> None:
        self.region = _Region(index=len(self.regions), owner=owner, turn=self.turn,
                              lines=[] if self.keep_lines else None)
        self.regions.append(self.region)

    def _close_region(self) -> None:
        r = self.region
        if r is None or r.closed:
            return
        r.closed = True
        r.block = None
        for x in SIDES:
            v = r.vswitch[x]
            if v is None:
                continue
            y = other(x)
            if r.moved[x]:
                continue                        # it switched and also moved: not a plain switch
            attack = next((a for a in r.attacks if a["side"] == y and not a["called"]), None)
            if attack is None or v["in_instance"] not in attack["se"]:
                continue
            excluded = ""
            if r.replace:
                excluded = "illusion"
            label, tier, mtype = "", 0.0, ""
            out_mon = v["out_mon"]
            if to_id(attack["attacker"].ability) == "neutralizinggas" and                     to_id(out_mon.ability) not in _CANT_SUPPRESS:
                out_mon = dc_replace(out_mon, ability=None)
            try:
                m = matchup(attack["move"], out_mon, attack["attacker"], attack["field"])
                label, tier, mtype = m.label, tier_of(m), m.move_type
            except (KeyError, ValueError, IndexError):
                excluded = excluded or "unknown"
            self.events.append(ReadEvent(
                reader=y, switcher=x, tier=tier, label=label, move=attack["move"], move_type=mtype,
                out_species=v["out_species"], in_species=v["in_species"], turn=r.turn, region=r.index,
                out_instance=v["out_instance"], in_instance=v["in_instance"], excluded=excluded,
                lines=list(r.lines) if r.lines is not None else None))

    def _opp_action_label(self, r: _Region, opp: str, team: Optional[List[str]]) -> int:
        v = r.vswitch[opp]
        if v is not None and not r.moved[opp]:
            if team is not None and v["in_key"] in team:
                i = team.index(v["in_key"])
                return 1 + i if i < 6 else OPP_UNSEEN
            return OPP_UNSEEN
        if r.forced_switch[opp]:
            return OPP_NA
        return OPP_MOVED if r.any_action else OPP_NA

    # ------------------------------------------------------------------ state

    def _poke(self, key: str, side: str, species: str = "") -> _Poke:
        p = self.pokes.get(key)
        if p is None:
            p = _Poke(key, side, species)
            self.pokes[key] = p
            self.seen[side].append(key)
        elif species and not p.species:
            p.species = species
        return p

    def _active_poke(self, side: str) -> Optional[_Poke]:
        k = self.active.get(side)
        return self.pokes.get(k) if k else None

    def _field(self) -> Field:
        return Field(weather=self.weather, terrain=self.terrain, gravity=self.gravity)

    def _mon(self, p: _Poke) -> Mon:
        """What both players know about p, for matchup.py."""
        species = p.form or p.species
        ability = known_or_only_ability(species, p.ability, self.sets)
        if "gastroacid" in p.volatiles:
            ability = None
        vols = frozenset(v for v in p.volatiles if v in _VOLATILE_IDS)
        full = p.hp is not None and p.maxhp is not None and p.hp >= p.maxhp
        return Mon(species=species, types=p.types, tera=p.tera, ability=ability,
                   item=p.item if p.item else None, added_type=p.added_type, volatiles=vols, full_hp=full)

    def _possible_abilities(self, p: _Poke) -> set:
        out = set(set_abilities(p.form or p.species, self.sets))
        entry = dex().species(p.form or p.species)
        if entry:
            out.update(entry.get("abilities", {}).values())
        return {to_id(a) for a in out}

    def _set_ability(self, name: str, subject: Optional[str], of: Optional[str]) -> None:
        """'[from] ability: X': the [of] Pokemon if it can have X, else the subject if it can, else as
        Showdown's client does ([of] first)."""
        cands = [k for k in (of, subject) if k and k in self.pokes]
        if not cands:
            return
        aid = to_id(name)
        owner = next((k for k in cands if aid in self._possible_abilities(self.pokes[k])), cands[0])
        self.pokes[owner].ability = name

    # ------------------------------------------------------------------ one line

    def _line(self, parts: List[str]) -> None:
        if self._skip_next:
            self._skip_next = False
            return
        cmd = parts[1]
        if cmd in ("request", "t:", "c", "chat", "j", "J", "l", "L", "n", "N", "raw", "html", "uhtml",
                   "inactive", "inactiveoff", "badge", "rule", "title", "gen", "tier", "gametype", "rated",
                   "seed", "teamsize", "timestamp", "init", "deinit", "debug", "message", "hint"):
            return
        if cmd == "split":
            self._skip_next = True          # the next line is the secret one, the one after the public one
            return
        r = self.region if (self.region is not None and not self.region.closed) else None
        if r is not None and r.lines is not None and cmd != "":
            r.lines.append("|".join(parts))

        if cmd == "player" and len(parts) > 3:
            self.names[parts[2]] = parts[3]
            return
        if cmd == "start":
            self.started = True
            return
        if cmd == "turn":
            if self.region is not None and not self.region.closed:
                self._close_region()
            try:
                self.turn = int(parts[2])
            except (IndexError, ValueError):
                self.turn = (self.turn or 0) + 1
            self._open_region(owner=None)
            return
        if cmd == "upkeep":
            self._close_region()
            return
        if cmd in ("win", "tie"):
            self.ended = True
            self._close_region()
            return
        if cmd == "":
            if r is not None:
                r.block = None
            return

        kw = _kwargs(parts)
        side, key = _ident(parts[2]) if len(parts) > 2 else (None, None)
        of_side, of_key = _ident(kw.get("of", ""))

        if cmd in ("switch", "drag"):
            self._switch(parts, side, key, cmd, r)
            return
        if cmd == "replace":
            self._replace(parts, side, key, r)
            return
        if cmd == "move":
            self._move(parts, side, key, kw, r)
            return
        if cmd == "cant":
            if r is not None and side:
                r.move_seen = True
                r.any_action = True
                r.moved[side] = True
                r.block = None
            return
        if cmd == "-supereffective":
            if r is not None and r.block is not None and side and side != r.block["side"]:
                inst = self.instance.get(side)
                if inst is not None:
                    r.block["se"].add(inst)
            return
        if cmd == "faint":
            p = self.pokes.get(key) if key else None
            if p:
                p.fainted = True
                p.hp = 0
            if side and self.active.get(side) == key:
                self.active[side] = None
            return

        p = self.pokes.get(key) if key else None
        if cmd in ("-damage", "-heal", "-sethp") and p and len(parts) > 3:
            cur, mx = _hp(parts[3])
            if cur is not None:
                p.hp = cur
                if mx is not None:
                    p.maxhp = mx
        elif cmd == "detailschange" and p and len(parts) > 3:
            p.species = _species_of(parts[3])
            p.form = None
            # A permanent forme change takes the forme's ability when it has only one (Tera Shift ->
            # Terapagos-Terastal's Tera Shell).
            entry = dex().species(p.species)
            abil = list((entry or {}).get("abilities", {}).values())
            if len(abil) == 1:
                p.ability = abil[0]
        elif cmd == "-formechange" and p and len(parts) > 3:
            p.form = _species_of(parts[3])
        elif cmd == "-terastallize" and p and len(parts) > 3:
            p.tera = type_name(parts[3])
        elif cmd == "-transform" and p and len(parts) > 3:
            _, tkey = _ident(parts[3])
            t = self.pokes.get(tkey) if tkey else None
            if t is not None:
                tm = self._mon(t)
                p.types = (tm.tera,) if tm.tera and tm.tera != STELLAR else (tm.base_types() or None)
                p.transformed = True
                if t.ability:
                    p.ability = t.ability
        elif cmd == "-ability" and p and len(parts) > 3:
            p.ability = parts[3]
        elif cmd == "-endability" and p:
            p.volatiles.add("gastroacid")
        elif cmd == "-item" and p and len(parts) > 3:
            p.item = parts[3]
        elif cmd == "-enditem" and p and len(parts) > 3:
            item = to_id(parts[3])
            p.item = ""
            if item in _EJECT_ITEMS and r is not None and side:
                r.pending[side] = item
        elif cmd == "-activate" and p and len(parts) > 3:
            eff = parts[3]
            if eff.startswith("ability: "):
                p.ability = eff[len("ability: "):]
                if to_id(p.ability) in _EXIT_ABILITIES and r is not None and side:
                    r.pending[side] = to_id(p.ability)
            elif eff.startswith("item: "):
                p.item = eff[len("item: "):]
        elif cmd == "-start" and p and len(parts) > 3:
            eff = parts[3]
            if eff == "typechange" and len(parts) > 4:
                if not parts[4].startswith("["):
                    p.types = tuple(type_name(t) for t in parts[4].split("/") if t)
                elif of_key in self.pokes:
                    # Reflect Type: '-start|X|typechange|[from] move: Reflect Type|[of] Y' copies Y's types.
                    tm = self._mon(self.pokes[of_key])
                    p.types = (tm.tera,) if tm.tera and tm.tera != STELLAR else (tm.base_types() or None)
            elif eff == "typeadd" and len(parts) > 4:
                p.added_type = type_name(parts[4])
            else:
                vid = to_id(eff.split(": ", 1)[-1])
                p.volatiles.add(vid)
                if vid == "smackdown":
                    p.volatiles.discard("magnetrise")
                if eff.startswith("ability: "):
                    p.ability = eff[len("ability: "):]
        elif cmd == "-end" and p and len(parts) > 3:
            eff = parts[3]
            vid = to_id(eff.split(": ", 1)[-1])
            p.volatiles.discard(vid)
            if vid == "typechange":
                p.types = None
            if vid in ("futuresight", "doomdesire") and r is not None:
                r.block = None
        elif cmd == "-weather" and len(parts) > 2:
            w = to_id(parts[2])
            self.weather = None if w in ("none", "") else w
        elif cmd == "-fieldstart" and len(parts) > 2:
            fid = to_id(parts[2].split(": ", 1)[-1])
            if fid == "gravity":
                self.gravity = True
            elif fid.endswith("terrain"):
                self.terrain = fid
        elif cmd == "-fieldend" and len(parts) > 2:
            fid = to_id(parts[2].split(": ", 1)[-1])
            if fid == "gravity":
                self.gravity = False
            elif fid.endswith("terrain") and self.terrain == fid:
                self.terrain = None

        # Abilities and items named in [from] tags.
        frm = kw.get("from", "")
        if frm.startswith("ability: ") and cmd not in ("-ability",):
            self._set_ability(frm[len("ability: "):], key, of_key)
        elif frm.startswith("item: ") and cmd not in ("-item", "-enditem"):
            owner = of_key if of_key in self.pokes else key
            if owner in self.pokes:
                self.pokes[owner].item = frm[len("item: "):]

    def _switch(self, parts, side, key, cmd, r: Optional[_Region]) -> None:
        if not side or len(parts) < 4:
            return
        species = _species_of(parts[3])
        prev = self._active_poke(side)
        if r is not None:
            r.any_action = True
            if cmd == "switch" and r.switches[side] == 0 and r.vswitch[side] is None and not r.moved[side]                     and not r.move_seen:
                if r.pending[side]:
                    # Eject Pack / Emergency Exit before anyone moved: what the side chose is unknown.
                    r.forced_switch[side] = True
                elif prev is not None and not prev.fainted:
                    r.vswitch[side] = {
                        "out_mon": self._mon(prev), "out_key": prev.key, "out_species": prev.form or prev.species,
                        "out_instance": self.instance[side], "in_key": key, "in_species": species,
                        "in_instance": self._next_instance,
                    }
            r.switches[side] += 1
            r.pending[side] = None
            r.block = None
        if prev is not None:
            prev.leave_field()
        p = self._poke(key, side, species)
        p.species = species
        p.fainted = False
        p.form = None
        for detail in parts[3].split(",")[1:]:
            detail = detail.strip()
            if detail.startswith("tera:"):
                p.tera = type_name(detail[len("tera:"):])
        cur, mx = _hp(parts[4]) if len(parts) > 4 else (None, None)
        if cur is not None:
            p.hp = cur
            if mx is not None:
                p.maxhp = mx
        self.active[side] = key
        self.instance[side] = self._next_instance
        self._next_instance += 1

    def _replace(self, parts, side, key, r: Optional[_Region]) -> None:
        """Illusion ends: the Pokemon on the field was someone else all along."""
        if not side:
            return
        inst = self.instance.get(side)
        if inst is not None:
            self.illusion_instances.add(inst)
        if r is not None:
            r.replace = True
        old = self._active_poke(side)
        if old is not None:
            old.leave_field()
        species = _species_of(parts[3]) if len(parts) > 3 else ""
        p = self._poke(key, side, species)
        if species:
            p.species = species
        p.fainted = False
        self.active[side] = key

    def _move(self, parts, side, key, kw, r: Optional[_Region]) -> None:
        if r is None or not side or len(parts) < 4:
            return
        # A move called by another move or a lock ([from]move: Sleep Talk, [from]lockedmove) is not a
        # choice made this turn; one copied by an ability ([from]ability: Dancer / Magic Bounce) is not
        # even the side's action.
        frm = kw.get("from")
        called = frm is not None
        r.move_seen = True
        r.any_action = True
        attacker = self._active_poke(side)
        block = {"side": side, "move": to_id(parts[3]), "called": called, "se": set(),
                 "attacker": self._mon(attacker) if attacker is not None else Mon(),
                 "field": self._field()}
        r.block = block
        if not (frm or "").startswith("ability:"):
            r.moved[side] = True
        if not called:
            r.attacks.append(block)


# ---------------------------------------------------------------------------------------------------
# Battle records: the ladder app's jsonl ("in" protocol chunks, "note" decisions) and self-play logs.
# ---------------------------------------------------------------------------------------------------

def _open(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


_STATE_CMDS = {"switch", "drag", "move", "faint", "turn", "upkeep", "detailschange", "replace", "cant", "win",
               "tie", "start"}


def _is_state_line(ln: str) -> bool:
    if not ln.startswith("|"):
        return False
    cmd = ln.split("|", 2)[1]
    return cmd.startswith("-") or cmd in _STATE_CMDS


def read_ladder_record(path: str) -> Tuple[Optional[str], List[tuple]]:
    """(our seat, steps) from a ladder battle record: steps are ('lines', [str...]) and
    ('decide', action, forced).

    After a reconnection the server sends the whole battle again (a second '|init|battle'); the lines
    already seen are dropped from that replay, so every turn is read once."""
    user, seat, steps = None, None, []
    seen_state: List[str] = []
    with _open(path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            d = json.loads(raw)
            data = d.get("data")
            if d.get("dir") == "note" and isinstance(data, dict):
                if "user" in data and user is None:
                    user = data["user"]
                dec = data.get("decision")
                if isinstance(dec, dict) and "played" in dec:
                    steps.append(("decide", int(dec["played"]), bool(dec.get("forced"))))
            elif d.get("dir") == "in" and isinstance(data, str):
                lines = data.split("\n")
                for ln in lines:
                    if ln.startswith("|player|") and user is not None:
                        ps = ln.split("|")
                        if len(ps) > 3 and to_id(ps[3]) == to_id(user):
                            seat = ps[2]
                if seen_state and any(ln.startswith("|init|") for ln in lines):
                    kept, i = [], 0
                    for ln in lines:
                        if i < len(seen_state):
                            if _is_state_line(ln):
                                if ln == seen_state[i]:
                                    i += 1
                                    continue
                                i = len(seen_state)      # diverged: keep the rest
                            else:
                                continue
                        kept.append(ln)
                    lines = kept
                seen_state.extend(ln for ln in lines if _is_state_line(ln))
                steps.append(("lines", lines))
    return seat, steps


def track_steps(steps: Iterable[tuple], seat: str, keep_lines: bool = False, sets: Optional[dict] = None,
                infer_forced: bool = False) -> ReadTracker:
    """Run a tracker over ('lines', ...) / ('decide', action, forced) steps."""
    t = ReadTracker(seat=seat, sets=sets, keep_lines=keep_lines)
    for st in steps:
        if st[0] == "lines":
            t.observe(st[1])
        elif st[0] == "decide":
            t.decide(st[1], forced=None if infer_forced else st[2])
    t.finalize()
    return t


def protocol_decisions(steps: Iterable[tuple], seat: str) -> List[tuple]:
    """The steps with the decisions of `seat` made up from the protocol alone: one turn-start decision
    after every '|turn|' line (the seat's action there is not known, so None). Forced decisions do not
    carry labels, so they are left out. Used to run the other seat over a one-sided record."""
    out = []
    for st in steps:
        if st[0] != "lines":
            continue
        buf = []
        for ln in st[1]:
            buf.append(ln)
            if ln.startswith("|turn|"):
                out.append(("lines", buf))
                out.append(("decide", None, False))
                buf = []
        if buf:
            out.append(("lines", buf))
    return out
