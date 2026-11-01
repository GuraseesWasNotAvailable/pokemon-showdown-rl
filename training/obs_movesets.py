"""
The moveset block: what each Pokemon can do, appended after the matchup block (obs_matchup.py), so
no existing offset moves. On only with the observation setting "movesets" (run_cfg["obs"]["movesets"]
in a checkpoint, ObservationAssembler(matchup=True, movesets=True)); it needs the matchup block.
Off, the observation is byte for byte what it was. bench/notes/OBS_MOVESETS.md has the offsets.

For each of the 12 Pokemon (ours, then theirs, the observation's order), 55 numbers:

  * the moves it may have but has not shown: top K_MOVES (8) move vocabulary ids (+1) and their
    probabilities, the marginal over the random battle roles that fit what it has shown (the
    roles obs_matchup.opponent_guess keeps, equally likely), each role's moves drawn as
    4 - (moves shown) of the role's moves not shown yet, all equally likely. Our own side is exact:
    its 4 moves with probability 1;
  * its Tera type: a probability over TERA_TYPES (18 types and Stellar), the same roles' Tera
    types equally likely within a role; ours from the request; all 0 once it has Terastallized;
  * its threat to each of the 6 Pokemon on the OTHER side: the expected best log2 damage
    multiplier of its damaging moves (shown and not shown, by the draw above; a status move
    counts as IMMUNE_LOG2, so a Pokemon with no damaging move scores IMMUNE_LOG2), and the
    probability that it has a super-effective move (multiplier above 1). The multipliers are
    obs_matchup.effectiveness (matchup.py) on what is known of the defender, with each move's full
    type (obs_matchup.move_type(full=True): the attacker's ability, form, Tera and the weather), exactly
    the numbers of the matchup block's move tokens, which use the full types when this block is on. A
    move not shown is typed and judged with its role's ability while the Pokemon's is not known (a
    Primarina's Psychic Noise is Water in its Liquid Voice role). 0 for a slot not seen yet;
  * the probability that it has: a set-up move, a priority attack, recovery, a pivot move, a
    status move (sleep, paralysis, burn, poison), a hazard, hazard removal, phazing (FLAG_NAMES).

Illusion (illusion_adjust): poke-env leaves the moves an Illusion user used in disguise on the
Pokemon it posed as, also after the Illusion breaks. Once an Illusion user is revealed on the
opposing side, a teammate's shown move that none of its species' sets hold is left out of that
teammate's numbers and counted as shown for the Illusion user when its sets hold it. A move both sets
hold stays where poke-env put it (exact attribution needs the |replace| line; this block reads the
battle only).

Also here, for the SetHead's move and Tera heads (ppo_core.SetHead): candidate_tables().

Importing this module pulls in numpy and obs_matchup only. poke-env is imported when a battle is encoded.
"""

from __future__ import annotations

import math
from typing import Any, Dict, FrozenSet, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

import obs_matchup as om
from obs_matchup import IMMUNE_LOG2, N_SIDE, to_id

# The observation setting's name in run_cfg["obs"].
SETTING = "movesets"

# ---------------------------------------------------------------------------------- layout

K_MOVES = 8
TERA_TYPES: Tuple[str, ...] = ("bug", "dark", "dragon", "electric", "fairy", "fighting", "fire", "flying", "ghost", "grass",
                               "ground", "ice", "normal", "poison", "psychic", "rock", "steel", "water", "stellar")
TERA_INDEX: Dict[str, int] = {t: i for i, t in enumerate(TERA_TYPES)}
N_TERA = len(TERA_TYPES)                      # 19
FLAG_NAMES: Tuple[str, ...] = ("setup", "priority", "recovery", "pivot", "status", "hazard", "hazard_removal", "phazing")
N_FLAGS = len(FLAG_NAMES)                     # 8

MS_MOVE_IDS = (0, 8)
MS_MOVE_P = (8, 16)
MS_TERA = (16, 35)
MS_THREAT = (35, 41)       # expected best log2 multiplier vs each of the 6 on the other side
MS_SE = (41, 47)           # probability of a super-effective move vs each of the 6
MS_FLAGS = (47, 55)
MS_DENSE = (16, 55)        # Tera, threat, super-effective, flags: what the model reads as numbers
MS_FEATS = 55
BLOCK_DIM = 12 * MS_FEATS  # 660


def block_meta() -> Dict[str, Any]:
    """What ObservationAssembler(movesets=True) puts in meta["movesets"]; the model reads its sizes from it."""
    return {"dim": BLOCK_DIM, "mon_feats": MS_FEATS, "k_moves": K_MOVES, "n_tera": N_TERA, "n_flags": N_FLAGS,
            "move_ids": MS_MOVE_IDS, "move_p": MS_MOVE_P, "dense": MS_DENSE,
            "tera": MS_TERA, "threat": MS_THREAT, "se": MS_SE, "flags": MS_FLAGS}


def checkpoint_obs_movesets(ckpt: Any) -> bool:
    """Whether a learner checkpoint's network sees the moveset block (run_cfg["obs"]["movesets"]). No key: it does not."""
    run_cfg = ckpt.get("run_cfg") if isinstance(ckpt, dict) else None
    obs = run_cfg.get("obs") if isinstance(run_cfg, dict) else None
    return bool(obs.get(SETTING, False)) if isinstance(obs, dict) else False


# ---------------------------------------------------------------------------------- what a move does

PIVOT = frozenset({"uturn", "voltswitch", "flipturn", "partingshot", "teleport", "chillyreception", "shedtail"})
HAZARD = frozenset({"stealthrock", "spikes", "toxicspikes", "stickyweb", "stoneaxe", "ceaselessedge"})
HAZARD_REMOVAL = frozenset({"rapidspin", "defog", "mortalspin", "tidyup", "courtchange"})
PHAZING = frozenset({"roar", "whirlwind", "dragontail", "circlethrow", "haze"})        # Haze: a reset, counted with them
# Reliable recovery of the user's own HP (the drain attacks are left out).
RECOVERY = frozenset({"recover", "roost", "slackoff", "softboiled", "milkdrink", "moonlight", "morningsun", "synthesis",
                      "shoreup", "strengthsap", "wish", "rest", "painsplit", "junglehealing", "lunarblessing", "lifedew",
                      "healorder"})
# Status moves whose effect is a major status on the target: sleep, paralysis, burn, poison (and Yawn).
_STATUSES = frozenset({"slp", "par", "brn", "tox", "psn"})

_FLAG_CACHE: Dict[Tuple[str, bool], int] = {}


def move_flags(move_id: str, ghost: bool = False) -> int:
    """The FLAG_NAMES a move has, as bits (bit k = FLAG_NAMES[k]). `ghost`: the user is a Ghost (its Curse is not set-up)."""
    key = (move_id, bool(ghost))
    hit = _FLAG_CACHE.get(key)
    if hit is not None:
        return hit
    import setup_moves as sm
    e = sm.move_entry(move_id)
    damaging = e.get("category") not in (None, "Status")
    bits = 0
    if sm.is_setup(move_id, ["GHOST"] if ghost else []):
        bits |= 1 << 0
    if damaging and int(e.get("priority", 0) or 0) > 0:
        bits |= 1 << 1
    if move_id in RECOVERY:
        bits |= 1 << 2
    if move_id in PIVOT:
        bits |= 1 << 3
    if move_id == "yawn" or (e.get("category") == "Status" and e.get("status") in _STATUSES):
        bits |= 1 << 4
    if move_id in HAZARD:
        bits |= 1 << 5
    if move_id in HAZARD_REMOVAL:
        bits |= 1 << 6
    if move_id in PHAZING:
        bits |= 1 << 7
    _FLAG_CACHE[key] = bits
    return bits


# ---------------------------------------------------------------------------------- draws

_TOP: Dict[Tuple[int, int], Tuple[float, ...]] = {}


def top_probs(n: int, s: int) -> Tuple[float, ...]:
    """
    s of n moves drawn, all subsets equally likely. With the moves ranked 1..n, the probability
    that move i is the best one drawn (drawn, and none above it): C(n - i, s - 1) / C(n, s). Sums
    to 1 when 1 <= s; with s >= n the first is drawn for certain.
    """
    key = (n, s)
    hit = _TOP.get(key)
    if hit is None:
        if n <= 0 or s <= 0:
            hit = tuple(0.0 for _ in range(max(n, 0)))
        elif s >= n:
            hit = (1.0,) + tuple(0.0 for _ in range(n - 1))
        else:
            total = math.comb(n, s)
            hit = tuple(math.comb(n - i, s - 1) / total for i in range(1, n + 1))
        _TOP[key] = hit
    return hit


def none_drawn(n: int, k: int, s: int) -> float:
    """The probability that none of k given moves is among s drawn of n."""
    if k <= 0 or s <= 0:
        return 1.0
    if s >= n or n - k < s:
        return 0.0
    return math.comb(n - k, s) / math.comb(n, s)


# ---------------------------------------------------------------------------------- one Pokemon


class Profile(NamedTuple):
    """What one Pokemon may do, independent of the other side: its moves as MoveInfos, by role."""
    moves: Tuple[om.MoveInfo, ...]                         # every move it may have, each once: the shown ones first
    n_shown: int                                           # moves[:n_shown] are shown (ours: all 4)
    roles: Tuple[Tuple[float, int, Tuple[int, ...]], ...]  # (weight, free slots, indexes into moves of the ones not shown)
    row: np.ndarray                                        # float32 [MS_FEATS], threat columns 0


_MOVES: Dict[str, Any] = {}


def _move_obj(move_id: str) -> Any:
    hit = _MOVES.get(move_id)
    if hit is None:
        from poke_env.battle import Move
        hit = _MOVES[move_id] = Move(move_id, gen=9)
    return hit


def _ghost(mon: Any) -> bool:
    return "GHOST" in (om._type_name(getattr(mon, "_type_1", None)), om._type_name(getattr(mon, "_type_2", None)))


def _flag_probs(revealed_ids: Sequence[str], roles: Sequence[Tuple[float, int, Tuple[str, ...]]], ghost: bool) -> np.ndarray:
    have = 0
    for m in revealed_ids:
        have |= move_flags(m, ghost)
    out = np.zeros(N_FLAGS, dtype=np.float64)
    for w, s, pool in roles:
        bits = [move_flags(m, ghost) for m in pool]
        for k in range(N_FLAGS):
            if have >> k & 1:
                out[k] += w
            else:
                kk = sum(1 for b in bits if b >> k & 1)
                out[k] += w * (1.0 - none_drawn(len(pool), kk, s))
    return out


def own_tera(mon: Any) -> Optional[str]:
    """Our own Pokemon's Tera type id (the request names it), None once it has Terastallized or when unknown."""
    if getattr(mon, "is_terastallized", False):
        return None
    from obs_assembler import own_tera_type_name
    name = own_tera_type_name(mon)
    return to_id(name) if name else None


def full_move_info(move: Any, attacker: Any, attacker_ability: Optional[str], weather: Optional[str]) -> om.MoveInfo:
    """obs_matchup.move_info with the full type (the block's, and the matchup block's when the block is on)."""
    return om.move_info(move, attacker, attacker_ability, weather, full=True)


Adjust = Tuple[FrozenSet[str], Tuple[str, ...]]      # (shown moves to leave out, moves to count as shown)

_SPECIES_INFO: Dict[str, Tuple[Any, Optional[FrozenSet[str]], bool, FrozenSet[str]]] = {}


def _species_info(table: om.SetTable, species: Any) -> Tuple[Optional[FrozenSet[str]], bool, FrozenSet[str]]:
    """(every move the species' sets hold, None without sets; whether they have Illusion; the CONTEXT_MOVES among them)."""
    hit = _SPECIES_INFO.get(species)
    if hit is None or hit[0] is not table:
        roles = table.roles_for(species or "")
        if roles:
            moves = frozenset(m for r in roles for m in r["moves"])
            hit = (table, moves, any("illusion" in r["abilities"] for r in roles), moves & om.CONTEXT_MOVES)
        else:
            hit = (table, None, False, frozenset())
        if len(_SPECIES_INFO) > 4096:
            _SPECIES_INFO.clear()
        _SPECIES_INFO[species] = hit
    return hit[1], hit[2], hit[3]


def illusion_adjust(mons: Sequence[Optional[Any]], table: om.SetTable) -> Optional[List[Optional[Adjust]]]:
    """
    The Illusion correction for the 12 Pokemon (the observation's order), or None when there is none
    (no Illusion user revealed on the opposing side: every battle without one). poke-env leaves the moves
    an Illusion user used in disguise on the Pokemon it posed as. Once one is revealed, each other
    opposing Pokemon (not Transformed, with sets) leaves out the shown moves its species' sets do not hold;
    the Illusion user counts as shown those of them its own sets hold.
    """
    opp = list(mons[N_SIDE:2 * N_SIDE])
    zs = [j for j, m in enumerate(opp) if m is not None and _species_info(table, getattr(m, "species", None))[1]]
    if not zs:
        return None
    out: List[Optional[Adjust]] = [None] * (2 * N_SIDE)
    foreign: set = set()
    for j, m in enumerate(opp):
        if m is None or j in zs or getattr(m, "transformed", False):
            continue
        legal = _species_info(table, getattr(m, "species", None))[0]
        if legal is None:
            continue
        drop = frozenset(x for x in om.shown_moves(m) if x not in legal)
        if drop:
            out[N_SIDE + j] = (drop, ())
            foreign |= drop
    if foreign:
        for j in zs:
            z = opp[j]
            if getattr(z, "transformed", False):
                continue
            legal = _species_info(table, getattr(z, "species", None))[0] or frozenset()
            have = set(om.shown_moves(z))
            extra = tuple(sorted(x for x in foreign if x in legal and x not in have))
            if extra:
                out[N_SIDE + j] = (frozenset(), extra)
    return out


def build_profile(mon: Any, own: bool, guess: om.SetGuess, table: om.SetTable, move_vocab: Dict[str, int],
                  move_info_of: Any, weather: Optional[str] = None, adjust: Optional[Adjust] = None) -> Profile:
    """
    One Pokemon's profile, worked out from scratch. move_info_of(move object, mon, attacker ability,
    weather) -> MoveInfo (full_move_info, or a cached stand-in that returns the same values). weather:
    obs_matchup.battle_weather(). adjust: illusion_adjust()'s entry for this Pokemon (theirs only).
    """
    att = guess.known_ability
    ghost = _ghost(mon)
    row = np.zeros(MS_FEATS, dtype=np.float32)
    if own:
        moves = [mv for mv in list((getattr(mon, "moves", None) or {}).values())[:4] if mv is not None]
        revealed = tuple(move_info_of(mv, mon, att, weather) for mv in moves)
        ids = [getattr(mv, "id", "") for mv in moves]
        cands = sorted(((m, 1.0) for m in set(ids) if move_vocab.get(m, -1) >= 0), key=lambda kv: (-kv[1], kv[0]))
        tera = own_tera(mon)
        tera_p = {tera: 1.0} if tera in TERA_INDEX else {}
        all_moves: Tuple[om.MoveInfo, ...] = revealed
        roles: Tuple[Tuple[float, int, Tuple[int, ...]], ...] = ((1.0, 0, ()),)
        flags = _flag_probs(ids, ((1.0, 0, ()),), ghost)
    else:
        drop, extra = adjust or (frozenset(), ())
        shown_objs = [mv for k, mv in (getattr(mon, "moves", None) or {}).items()
                      if mv is not None and to_id(k) not in om._NOT_SET_MOVES and to_id(k) not in drop]
        shown_objs += [_move_obj(m) for m in extra]
        shown_ids = [to_id(getattr(mv, "id", "")) for mv in shown_objs]
        shown_set = set(shown_ids)
        revealed = tuple(move_info_of(mv, mon, att, weather) for mv in shown_objs)
        s = max(0, 4 - len(shown_ids))
        ability = to_id(mon.ability) if getattr(mon, "ability", None) else None
        item, _unknown, _gone = om._item_state(getattr(mon, "item", None))
        all_roles = table.roles_for(getattr(mon, "species", ""))
        role_moves = om.shown_moves(mon)
        if adjust:
            role_moves = tuple(sorted((set(role_moves) - drop) | set(extra)))
        kept = om.kept_roles(all_roles, role_moves, ability, item, om.shown_tera(mon)) if all_roles else []
        id_roles: List[Tuple[float, int, Tuple[str, ...]]] = []
        # the ability that types and judges a role's moves not shown: the Pokemon's if known, else the role's
        # only one (a Primarina's Psychic Noise is Water in its Liquid Voice role)
        role_atts: List[Optional[str]] = []
        cand_p: Dict[str, float] = {}
        tera_p = {}
        if kept:
            w = 1.0 / len(kept)
            for r in kept:
                pool = tuple(m for m in r["moves"] if m not in shown_set)
                id_roles.append((w, s, pool))
                role_atts.append(att if att is not None else (r["abilities"][0] if len(r["abilities"]) == 1 else None))
                if pool and s > 0:
                    q = min(1.0, s / len(pool))
                    for m in pool:
                        cand_p[m] = cand_p.get(m, 0.0) + w * q
                if r["tera"]:
                    for t in r["tera"]:
                        tera_p[t] = tera_p.get(t, 0.0) + w / len(r["tera"])
        else:
            id_roles.append((1.0, 0, ()))
            role_atts.append(att)
        if getattr(mon, "is_terastallized", False):
            tera_p = {}
        cands = sorted(((m, p) for m, p in cand_p.items() if p > 0 and move_vocab.get(m, -1) >= 0), key=lambda kv: (-kv[1], kv[0]))
        flags = _flag_probs(shown_ids, id_roles, ghost)
        # every (move, ability) once, the shown ones first; the roles point into it
        index: Dict[Tuple[str, Optional[str]], int] = {}
        listed: List[om.MoveInfo] = list(revealed)
        role_idx: List[Tuple[float, int, Tuple[int, ...]]] = []
        for (w, s_r, pool), ra in zip(id_roles, role_atts):
            idx: List[int] = []
            for m in pool:
                k = (m, ra)
                if k not in index:
                    index[k] = len(listed)
                    listed.append(move_info_of(_move_obj(m), mon, ra, weather))
                idx.append(index[k])
            role_idx.append((w, s_r, tuple(idx)))
        all_moves = tuple(listed)
        roles = tuple(role_idx)
    for k, (m, p) in enumerate(cands[:K_MOVES]):
        row[MS_MOVE_IDS[0] + k] = move_vocab[m] + 1
        row[MS_MOVE_P[0] + k] = min(1.0, p)
    for t, p in tera_p.items():
        j = TERA_INDEX.get(t)
        if j is not None:
            row[MS_TERA[0] + j] = p
    row[MS_FLAGS[0]:MS_FLAGS[1]] = np.minimum(flags, 1.0)
    return Profile(moves=all_moves, n_shown=len(revealed), roles=roles, row=row)


def threat_from_codes(prof: Profile, codes: Sequence[float]) -> Tuple[float, float]:
    """
    (expected best log2 multiplier, probability of a super-effective move) of one profile against
    one defender, from the log2 code of each of prof.moves on it (IMMUNE_LOG2 for a status move).
    """
    rev = codes[:prof.n_shown]
    base = max(rev) if rev else IMMUNE_LOG2
    base_se = any(c > 0.0 for c in rev)
    e = se = 0.0
    for w, s, pool in prof.roles:
        n = len(pool)
        if s <= 0 or n == 0:
            e += w * base
            se += w if base_se else 0.0
            continue
        cs = sorted((codes[k] for k in pool), reverse=True)
        e += w * sum(p * (c if c > base else base) for p, c in zip(top_probs(n, s), cs))
        if base_se:
            se += w
        else:
            se += w * (1.0 - none_drawn(n, sum(1 for c in cs if c > 0.0), s))
    return e, min(1.0, se)


def threat(prof: Profile, foes: Sequence[Optional[om.DefenderInfo]], code: Any) -> np.ndarray:
    """
    The 12 threat numbers of one profile against the six Pokemon of the other side (None: not seen):
    [expected best log2 multiplier x6 | probability of a super-effective move x6]. code(mi, d) is
    obs_matchup.log2_code(effectiveness(mi, d)) for a damaging move.
    """
    out = np.zeros(12, dtype=np.float64)
    for j, d in enumerate(foes):
        if d is None:
            continue
        out[j], out[N_SIDE + j] = threat_from_codes(prof, [code(mi, d) if mi.damaging else IMMUNE_LOG2 for mi in prof.moves])
    return out


# ---------------------------------------------------------------------------------- the encoder

# The attacker abilities matchup.matchup reads once the move's type is given (obs_matchup._verdict passes
# type_override): Mold Breaker and kin, Mycelium Might, Scrappy, Mind's Eye, Tinted Lens. Any other ability
# of the attacker gives the same verdict as none, so the multiplier caches below leave it out of the key
# (tests/test_obs_movesets.py holds the result byte for byte against the reference path, which keeps it).
VERDICT_ATTACKER_ABILITIES = frozenset({"moldbreaker", "teravolt", "turboblaze", "myceliummight", "scrappy", "mindseye",
                                        "tintedlens"})

# How many entries each of the store's caches may hold before it is emptied (alone, when nothing else
# refers to its ids). About 100 to 300 bytes an entry: some 600 MB per process at the very most.
_LIMITS = {"prof_ids": 200_000, "def_ids": 200_000, "mi_ids": 200_000, "move_infos": 200_000,
           "codes": 1_000_000, "thr": 1_000_000, "rows": 400_000}
_K = 1 << 24          # codes and thr are keyed by one int: (MoveInfo or profile id) * _K + defender id


class _Store:
    """
    What the encoders of one process have worked out, shared by all of them (a rollout worker runs
    one assembler per player, 8 or more). Every entry is a pure function of its key (the set table
    and the move vocabulary are fixed per store).
    """

    def __init__(self) -> None:
        self.gen = 0
        self.clear()

    def clear(self) -> None:
        self.gen += 1                                   # the encoders' maps of ids go with it
        self.prof_ids: Dict[Any, int] = {}              # profile key -> profile id
        self.profs: List[Profile] = []
        self.prof_mids: List[Tuple[int, ...]] = []      # per profile: its moves as MoveInfo ids (-1: a status move)
        self.mi_ids: Dict[om.MoveInfo, int] = {}        # a move as its multipliers see it (_mi_id)
        # A defender by what a multiplier reads of it (obs_matchup._verdict): types, Tera state, known ability,
        # known item. Its candidate lists (which change as its roles narrow) do not matter here.
        self.def_ids: Dict[Tuple[Any, ...], int] = {}
        self.defs: List[om.DefenderInfo] = []           # one DefenderInfo with that key, for effectiveness()
        self.codes: Dict[int, float] = {}               # MoveInfo id * _K + defender id -> log2 code
        self.thr: Dict[int, Tuple[float, float]] = {}   # profile id * _K + defender id -> threat numbers
        self.rows: Dict[Any, np.ndarray] = {}           # (profile id, six defender ids, dtype) -> the 55 numbers
        self.move_infos: Dict[Any, om.MoveInfo] = {}

    def trim(self) -> None:
        """Empties what has grown past its limit, and whatever refers to its ids."""
        if len(self.def_ids) > _LIMITS["def_ids"] or len(self.mi_ids) > _LIMITS["mi_ids"]:
            self.clear()
            return
        if len(self.prof_ids) > _LIMITS["prof_ids"]:
            self.gen += 1
            self.prof_ids, self.profs, self.prof_mids, self.thr, self.rows = {}, [], [], {}, {}
        for name in ("codes", "thr", "rows", "move_infos"):
            d = getattr(self, name)
            if len(d) > _LIMITS[name]:
                d.clear()


_STORES: Dict[Any, _Store] = {}


def _store_for(table: om.SetTable, move_vocab: Dict[str, int]) -> _Store:
    key = (id(table), hash(frozenset(move_vocab.items())))
    st = _STORES.get(key)
    if st is None:
        st = _STORES[key] = _Store()
    return st


class MovesetEncoder:
    """
    Writes the moveset block for one battle, next to the assembler's MatchupEncoder (whose guesses
    and defenders it reads). One per assembler; what it works out goes in a store shared by every
    encoder of the process with the same set table and move vocabulary: a profile per (Pokemon,
    what it has shown), its threat numbers per (profile, defender), and a row per (profile, the
    six defenders), all by small interned integers; a defender is interned by what the multipliers
    read of it (types, Tera state, known ability, known item).

    encode() is the fast path; encode_reference() works every number out again from the battle,
    with no cache; the two write the same bytes (tests/test_obs_movesets.py).
    """

    def __init__(self, vocab_map: Dict[str, Dict[str, int]], mu: om.MatchupEncoder):
        self.mu = mu
        self.table = mu.table
        self._move_vocab = vocab_map.get("move.id", {})
        self.store = _store_for(self.table, self._move_vocab)
        self._reset_map()

    def _reset_map(self) -> None:
        self._gen = self.store.gen
        self._defs_seen = self.mu._defs        # the matchup encoder's defender ids are only good while this list is
        self._l2g: List[int] = []              # matchup encoder's defender id -> the store's
        # An opposing Pokemon's profile by the matchup encoder's guess for it: MatchupEncoder.guess
        # keeps one SetGuess per (species, moves shown, ability, item, Tera type, possible abilities),
        # which is everything a profile reads of a Pokemon that is not Transformed, with the weather and types
        # (_context) and the Illusion correction it was built for. id -> (guess, (context, adjust), profile id).
        self._g2p: Dict[int, Tuple[om.SetGuess, Any, int]] = {}

    # ------------------------------------------------------------------ helpers

    def _move_info(self, mv: Any, mon: Any, att: Optional[str], weather: Optional[str]) -> om.MoveInfo:
        terastallized = bool(getattr(mon, "is_terastallized", False))
        mid = getattr(mv, "id", None)
        key = (mid, getattr(mon, "species", None), terastallized,
               om._type_name(getattr(mon, "tera_type", None)) if terastallized else None, att,
               om.type_context(mon, (mid,), weather) if mid in om.CONTEXT_MOVES else None)
        cache = self.store.move_infos
        hit = cache.get(key)
        if hit is None:
            hit = cache[key] = full_move_info(mv, mon, att, weather)
        return hit

    def _context(self, mon: Any, own: bool, weather: Optional[str], adjust: Optional[Adjust]) -> Any:
        """What a profile reads beyond _key: type_context of the moves it may type (shown, counted as shown, its sets')."""
        cm = om.CONTEXT_MOVES
        ids = {k for k in (getattr(mon, "moves", None) or ()) if k in cm}
        if not own:
            ids |= _species_info(self.table, getattr(mon, "species", None))[2]
            if adjust:
                ids |= cm.intersection(adjust[1])
        return om.type_context(mon, ids, weather) if ids else None

    @staticmethod
    def _key(mon: Any, own: bool) -> Any:
        """Everything a profile reads from the Pokemon."""
        terastallized = bool(getattr(mon, "is_terastallized", False))
        moves = tuple(getattr(mon, "moves", None) or ())
        if own:
            req = getattr(mon, "_last_request", None)
            return (True, getattr(mon, "species", None), moves, getattr(mon, "ability", None), terastallized,
                    om._type_name(getattr(mon, "tera_type", None)), req.get("teraType") if isinstance(req, dict) else None)
        return (False, getattr(mon, "species", None), moves, bool(getattr(mon, "transformed", False)), getattr(mon, "ability", None),
                getattr(mon, "item", None), terastallized, om._type_name(getattr(mon, "tera_type", None)) if terastallized else None,
                tuple(getattr(mon, "possible_abilities", None) or ()))

    def block(self, battle: Any, reference: bool = False) -> np.ndarray:
        """The block alone (float16 [BLOCK_DIM]), for tests and tools."""
        out = np.zeros(BLOCK_DIM, dtype=np.float16)
        if reference:
            self.encode_reference(battle, out, 0)
        else:
            scratch = np.zeros(om.BLOCK_DIM, dtype=np.float16)
            self.encode(battle, out, 0, self.mu.encode(battle, scratch, 0))
        return out

    # ------------------------------------------------------------------ writing

    def encode_reference(self, battle: Any, buffer: np.ndarray, start: int) -> None:
        """Writes the block into buffer[start:start + BLOCK_DIM] (zeroed by the caller), every number worked out again."""
        mons = self.mu.mons(battle)
        guesses = [None if m is None else (om.own_guess(m) if i < N_SIDE else om.opponent_guess(m, self.table))
                   for i, m in enumerate(mons)]
        defenders = [None if m is None else om.defender_info(m, guesses[i], i < N_SIDE) for i, m in enumerate(mons)]
        weather = om.battle_weather(battle)
        adjust = illusion_adjust(mons, self.table) or [None] * len(mons)

        def code(mi: om.MoveInfo, d: om.DefenderInfo) -> float:
            return om.log2_code(om.effectiveness(mi, d))

        for i, mon in enumerate(mons):
            if mon is None:
                continue
            prof = build_profile(mon, i < N_SIDE, guesses[i], self.table, self._move_vocab, full_move_info,
                                 weather, adjust[i])
            row = prof.row.copy()
            t = threat(prof, defenders[N_SIDE:] if i < N_SIDE else defenders[:N_SIDE], code)
            row[MS_THREAT[0]:MS_THREAT[1]] = t[:N_SIDE]
            row[MS_SE[0]:MS_SE[1]] = t[N_SIDE:]
            s = start + i * MS_FEATS
            buffer[s:s + MS_FEATS] = row

    def _global_def(self, did: int, d: om.DefenderInfo) -> int:
        """The store's id of the matchup encoder's defender `did`: of its (types, Tera state, known ability, known item)."""
        l2g = self._l2g
        if did < len(l2g):
            g = l2g[did]
            if g >= 0:
                return g
        else:
            l2g.extend([-1] * (did + 1 - len(l2g)))
        st = self.store
        key = (d.types, d.terastallized, d.ability, d.item)
        g = st.def_ids.get(key)
        if g is None:
            g = st.def_ids[key] = len(st.defs)
            st.defs.append(d)
        l2g[did] = g
        return g

    def _pid(self, mon: Any, own: bool, guess: om.SetGuess, weather: Optional[str], adjust: Optional[Adjust]) -> int:
        st = self.store
        ctx = (self._context(mon, own, weather, adjust), adjust)
        memo = None
        if not own and not getattr(mon, "transformed", False) and \
                (getattr(mon, "tera_type", None) is not None or not getattr(mon, "is_terastallized", False)):
            memo = self._g2p.get(id(guess))
            if memo is not None and memo[0] is guess and memo[1] == ctx:
                return memo[2]
        key = (self._key(mon, own), ctx)
        pid = st.prof_ids.get(key)
        if pid is None:
            prof = build_profile(mon, own, guess, self.table, self._move_vocab, self._move_info, weather, adjust)
            pid = st.prof_ids[key] = len(st.profs)
            st.profs.append(prof)
            st.prof_mids.append(tuple(self._mi_id(mi) if mi.damaging else -1 for mi in prof.moves))
        if not own and not getattr(mon, "transformed", False) and \
                (getattr(mon, "tera_type", None) is not None or not getattr(mon, "is_terastallized", False)):
            if len(self._g2p) > _LIMITS["prof_ids"]:
                self._g2p.clear()
            self._g2p[id(guess)] = (guess, ctx, pid)
        return pid

    def encode(self, battle: Any, buffer: np.ndarray, start: int, infos: Tuple[Any, Any, Any, Any]) -> None:
        """
        Writes the block into buffer[start:start + BLOCK_DIM], which the caller has zeroed. `infos`:
        what MatchupEncoder.encode returned for this battle a moment ago (mons, guesses, defenders,
        defender ids).
        """
        mons, guesses, defenders, dids = infos
        st = self.store
        st.trim()
        if self._gen != st.gen or self.mu._defs is not self._defs_seen:
            self._reset_map()
        dt = buffer.dtype.char
        gids = [-1 if x < 0 else self._global_def(x, defenders[k]) for k, x in enumerate(dids)]
        own_side, opp_side = tuple(gids[:N_SIDE]), tuple(gids[N_SIDE:])
        weather = om.battle_weather(battle)
        adjust = illusion_adjust(mons, self.table)
        rows = st.rows
        for i, mon in enumerate(mons):
            if mon is None:
                continue
            own = i < N_SIDE
            pid = self._pid(mon, own, guesses[i], weather, adjust[i] if adjust else None)
            foe_ids = opp_side if own else own_side
            rkey = (pid, foe_ids, dt)
            row = rows.get(rkey)
            if row is None:
                full = st.profs[pid].row.copy()
                for j, g in enumerate(foe_ids):
                    if g >= 0:
                        full[MS_THREAT[0] + j], full[MS_SE[0] + j] = self._threat_one(pid, g)
                row = rows[rkey] = full.astype(buffer.dtype)
            s = start + i * MS_FEATS
            buffer[s:s + MS_FEATS] = row

    def _mi_id(self, mi: om.MoveInfo) -> int:
        """The store's id of a move for its multipliers: the attacker's ability is kept only where matchup.py reads it."""
        if mi.attacker_ability is not None and mi.attacker_ability not in VERDICT_ATTACKER_ABILITIES:
            mi = mi._replace(attacker_ability=None)
        ids = self.store.mi_ids
        x = ids.get(mi)
        if x is None:
            x = ids[mi] = len(ids)
        return x

    def _threat_one(self, pid: int, g: int) -> Tuple[float, float]:
        """threat_from_codes of one profile on one defender (the store's ids), remembered."""
        st = self.store
        key = pid * _K + g
        hit = st.thr.get(key)
        if hit is None:
            prof, mids, d = st.profs[pid], st.prof_mids[pid], st.defs[g]
            codes = st.codes
            cs: List[float] = []
            for mi, m in zip(prof.moves, mids):
                if m < 0:
                    cs.append(IMMUNE_LOG2)
                    continue
                ck = m * _K + g
                c = codes.get(ck)
                if c is None:
                    # pair()[0] without the possible-immunity half pair() also works out; with the move
                    # as the matchup block has it, so the verdict the block cached is found
                    c = codes[ck] = om.log2_code(om.effectiveness(mi, d))
                cs.append(c)
            hit = st.thr[key] = threat_from_codes(prof, cs)
        return hit


# ---------------------------------------------------------------------------------- SetHead support


def candidate_tables(vocab_lists: Optional[Dict[str, List[str]]] = None,
                     table: Optional[om.SetTable] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    (moves [n_species + 1, n_moves + 1], tera [n_species + 1, N_TERA + 1]), bool: which move
    vocabulary ids (+1) and which Tera types (TERA_TYPES index + 1) a species' random battle sets
    allow. Column 0 is never allowed; a species without sets has an empty row.
    """
    vocab_lists = vocab_lists or om.load_vocab_lists()
    table = table or om.SetTable.default()
    vm = om._vocab_map(vocab_lists)
    species = vocab_lists.get("pokemon.species", [])
    mv = np.zeros((len(species) + 1, len(vocab_lists.get("move.id", [])) + 1), dtype=bool)
    te = np.zeros((len(species) + 1, N_TERA + 1), dtype=bool)
    for i, name in enumerate(species):
        for r in table.roles_for(name) or []:
            for m in r["moves"]:
                j = vm["move.id"].get(m, -1) + 1
                if j > 0:
                    mv[i + 1, j] = True
            for t in r["tera"]:
                j = TERA_INDEX.get(t, -1) + 1
                if j > 0:
                    te[i + 1, j] = True
    return mv, te
