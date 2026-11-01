"""
The moveset block (obs_movesets.py), the model's support for it (ppo_core), the SetHead's move and
Tera parts, and the wiring (config, train.py, check_restart, the worker's label column, hybrid.PolicyNet).

    PSPPO_STREAMS=<streams.jsonl> PSPPO_MOVESETS_BASE=<copy of efe43f7> CUDA_VISIBLE_DEVICES=3 \
        python -m pytest tests/test_obs_movesets.py -q -p no:cacheprovider -s

    PSPPO_STREAMS        recorded player streams (bench/micro_showdown_sim.py --record N)
    PSPPO_MOVESETS_BASE  a directory holding the files of efe43f7 (the commit this branch starts from,
                         `git archive efe43f7`): what "unchanged with the flag off" is measured against
    PSPPO_MOVESETS_CKPT  optional: a matchup checkpoint for hybrid.PolicyNet (widened here)

Checked:
  * the block on hand-made battles: our own side exact (its 4 moves at 1, its Tera type, its roles
    and threat exact); an opponent's candidates and roles narrow as it shows moves and match a brute
    force over every draw of its random battle roles; the Tera vector is 0 after the Tera; the
    threat numbers are matchup.py's;
  * on recorded battles: the fast path writes the reference path's bytes; the opponents' numbers
    match the brute force; our side's threat is the best of the matchup block's own move tokens;
  * with the setting off: the observation (plain and matchup) byte for byte what efe43f7 writes,
    the network bit for bit, and a whole PPO update with the SetHead bit for bit;
  * the SetHead's move and Tera parts, the 48-integer label column, an update with them;
  * the wiring: config, train.py's flag, check_restart's refusal, the learner's refusal,
    hybrid.PolicyNet and evaluate's checks on a widened checkpoint.
"""
from __future__ import annotations

import asyncio
import dataclasses
import itertools
import json
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(REPO, "tests")
for _p in (REPO, TESTS, os.path.join(REPO, "tools"), os.path.join(REPO, "cloud")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(REPO)          # vocab.json is opened by a relative path

import numpy as np  # noqa: E402

import obs_matchup as om  # noqa: E402
import obs_movesets as oms  # noqa: E402
from obs_assembler import ObservationAssembler  # noqa: E402

BASE_DIR = os.path.expanduser(os.environ.get("PSPPO_MOVESETS_BASE", ""))
MS_CKPT = os.environ.get("PSPPO_MOVESETS_CKPT", "")


# --------------------------------------------------------------------------- helpers


def _pokemon(species, moves=(), item=None, ability=None, tera=None):
    from poke_env.battle import Pokemon
    m = Pokemon(gen=9, species=species)
    for mv in moves:
        m._add_move(mv)
    if item is not None:
        m.item = item
    if ability is not None:
        m.ability = ability
    if tera is not None:
        m._last_request = {"teraType": tera}
    return m


def _battle(own, opp, weather=None):
    b = SimpleNamespace(team={f"p1: {m.species}{i}": m for i, m in enumerate(own)},
                        opponent_team={f"p2: {m.species}{i}": m for i, m in enumerate(opp)})
    if weather is not None:
        from poke_env.battle import Weather
        b.weather = {Weather[weather.upper()]: 1}
    return b


@pytest.fixture(scope="module")
def asm():
    return ObservationAssembler(own_tera_type=True, matchup=True, movesets=True)


@pytest.fixture(scope="module")
def enc(asm):
    return oms.MovesetEncoder(asm.vocab_map, om.MatchupEncoder(asm.vocab_map, full_types=True))


def _blk(enc, battle):
    """The block by the fast path, checked against the reference path; as float32 [12, 55]."""
    fast = enc.block(battle)
    slow = enc.block(battle, reference=True)
    assert fast.tobytes() == slow.tobytes()
    return fast.astype(np.float32).reshape(12, oms.MS_FEATS)


def _mid(asm, name):
    return asm.vocab_map["move.id"][name] + 1


def _cands(row):
    ids = row[oms.MS_MOVE_IDS[0]:oms.MS_MOVE_IDS[1]]
    ps = row[oms.MS_MOVE_P[0]:oms.MS_MOVE_P[1]]
    return {int(i): float(p) for i, p in zip(ids, ps) if i > 0}


def _flags(row):
    return dict(zip(oms.FLAG_NAMES, row[oms.MS_FLAGS[0]:oms.MS_FLAGS[1]].tolist()))


def _code(mi, d):
    return om.log2_code(om.effectiveness(mi, d))


def illusion_view(mon, team):
    """
    The move ids an opposing Pokemon counts as shown, with the Illusion rule, written here apart from
    obs_movesets: once an Illusion user is on its side (`team`, its six), every other Pokemon (not
    Transformed, with sets) leaves out the moves its sets do not hold, and the Illusion user gets those
    of them its sets hold.
    """
    tbl = om.SetTable.default()
    shown = [om.to_id(k) for k in mon.moves if om.to_id(k) not in ("struggle", "recharge")]
    team = [m for m in (team or []) if m is not None]

    def legal(m):
        return {x for r in (tbl.roles_for(m.species) or []) for x in r["moves"]}

    def illusion(m):
        return any("illusion" in r["abilities"] for r in (tbl.roles_for(m.species) or []))

    zs = [m for m in team if illusion(m)]
    if not zs or mon.transformed:
        return shown

    def foreign(m):
        if illusion(m) or m.transformed or not tbl.roles_for(m.species):
            return set()
        ok = legal(m)
        return {om.to_id(k) for k in m.moves if om.to_id(k) not in ("struggle", "recharge") and om.to_id(k) not in ok}

    if any(z is mon for z in zs):
        got = set().union(*(foreign(m) for m in team)) & legal(mon)
        return shown + sorted(got - set(shown))
    bad = foreign(mon)
    return [x for x in shown if x not in bad]


def brute_force(mon, foes, vocab, team=None, weather=None):
    """
    An opposing Pokemon's numbers from the pinned sets file, by enumerating every draw of every role
    that fits what it has shown (opponent_guess's filter): {move: p}, {tera: p}, flags, threat [12].
    Independent of obs_movesets' formulas (top_probs, none_drawn, the caches). Every move gets its full
    type (obs_matchup.move_type(full=True), in `weather`); a drawn move its role's ability while the
    Pokemon's is not known. team: its side's six, for the Illusion rule (illusion_view).
    """
    from poke_env.battle import Move
    with open(om.SETS_FILE, encoding="utf-8") as f:
        table = json.load(f)["species"]
    g = om.opponent_guess(mon, om.SetTable.default())
    roles = om.SetTable.default().roles_for(mon.species) or []
    ability = om.to_id(mon.ability) if mon.ability else None
    item = None if mon.item in (om.UNKNOWN_ITEM, None, "") else om.to_id(mon.item)
    view = illusion_view(mon, team)
    kept = om.kept_roles(roles, () if mon.transformed else tuple(sorted(set(view))), ability, item, om.shown_tera(mon)) if roles else []
    shown = {m: (mon.moves.get(m) or Move(m, gen=9)) for m in view}
    s = max(0, 4 - len(shown))
    ghost = "GHOST" in (om._type_name(mon._type_1), om._type_name(mon._type_2))
    moves_p, tera_p, flags, thr = {}, {}, np.zeros(oms.N_FLAGS), np.zeros(12)
    draws = []                                  # (weight, the moves of that draw: shown + drawn, the role's ability)
    for r in kept:
        pool = [m for m in r["moves"] if m not in shown]
        k = min(s, len(pool))
        combos = list(itertools.combinations(pool, k)) if pool else [()]
        ra = g.known_ability if g.known_ability else (r["abilities"][0] if len(r["abilities"]) == 1 else None)
        for c in combos:
            draws.append((1.0 / len(kept) / len(combos), list(c), ra))
        for t in r["tera"]:
            tera_p[t] = tera_p.get(t, 0.0) + 1.0 / len(kept) / len(r["tera"])
    if not kept:
        draws = [(1.0, [], g.known_ability)]
    for w, c, ra in draws:
        for m in c:
            moves_p[m] = moves_p.get(m, 0.0) + w
        bits = 0
        for m in list(shown) + c:
            bits |= oms.move_flags(m, ghost)
        flags += w * np.array([(bits >> k) & 1 for k in range(oms.N_FLAGS)], dtype=np.float64)
        infos = [om.move_info(mv, mon, g.known_ability, weather, full=True) for mv in shown.values()] + \
                [om.move_info(Move(m, gen=9), mon, ra, weather, full=True) for m in c]
        for j, d in enumerate(foes):
            if d is None:
                continue
            codes = [_code(mi, d) if mi.damaging else om.IMMUNE_LOG2 for mi in infos]
            best = max(codes) if codes else om.IMMUNE_LOG2
            thr[j] += w * best
            thr[6 + j] += w * (1.0 if any(cd > 0 and mi.damaging for cd, mi in zip(codes, infos)) else 0.0)
    if mon.is_terastallized:
        tera_p = {}
    del table, vocab
    return moves_p, tera_p, flags, thr


def _check_against_brute(asm, row, mon, foes, team=None, weather=None):
    moves_p, tera_p, flags, thr = brute_force(mon, foes, asm.vocab_map, team, weather)
    ranked = sorted(((m, p) for m, p in moves_p.items() if p > 0 and m in asm.vocab_map["move.id"]), key=lambda kv: (-kv[1], kv[0]))
    want = {asm.vocab_map["move.id"][m] + 1: p for m, p in ranked[:oms.K_MOVES]}
    got = _cands(row)
    assert set(got) == set(want), (mon.species, got, want)
    for k, p in want.items():
        assert got[k] == pytest.approx(p, abs=2e-3)
    tv = row[oms.MS_TERA[0]:oms.MS_TERA[1]]
    for t, j in oms.TERA_INDEX.items():
        assert tv[j] == pytest.approx(tera_p.get(t, 0.0), abs=2e-3), (mon.species, t)
    assert row[oms.MS_FLAGS[0]:oms.MS_FLAGS[1]] == pytest.approx(flags, abs=2e-3)
    assert row[oms.MS_THREAT[0]:oms.MS_SE[1]] == pytest.approx(thr, abs=4e-3), (mon.species, row[oms.MS_THREAT[0]:oms.MS_SE[1]], thr)


# --------------------------------------------------------------------------- layout


def test_layout_appends_the_block_after_the_matchup_block():
    a0 = ObservationAssembler(own_tera_type=True)
    a1 = ObservationAssembler(own_tera_type=True, matchup=True)
    a2 = ObservationAssembler(own_tera_type=True, matchup=True, movesets=True)
    assert "movesets" not in a1.offsets and "movesets" not in a1.meta and a1.movesets is False and a0.movesets is False
    for k, v in a1.offsets.items():
        assert a2.offsets[k] == v
    assert a2.offsets["movesets"] == (a1.total_dim, a1.total_dim + oms.BLOCK_DIM)
    assert a2.total_dim == a1.total_dim + 660 and oms.MS_FEATS == 55 and oms.N_TERA == 19 and oms.N_FLAGS == 8
    assert a2.meta["movesets"] == oms.block_meta()
    assert oms.MS_DENSE == (oms.MS_TERA[0], oms.MS_FLAGS[1])
    with pytest.raises(ValueError):
        ObservationAssembler(movesets=True)                       # needs the matchup block
    # every move id the block writes is exact in float16
    assert len(a2.vocab_lists["move.id"]) + 1 < 2048
    assert set(oms.TERA_TYPES) <= set(a2.vocab_lists["pokemon.type"])


def test_draw_probabilities():
    assert oms.top_probs(5, 2) == pytest.approx([4 / 10, 3 / 10, 2 / 10, 1 / 10, 0.0])
    assert sum(oms.top_probs(7, 3)) == pytest.approx(1.0)
    assert oms.top_probs(3, 4) == (1.0, 0.0, 0.0) and oms.top_probs(3, 0) == (0.0, 0.0, 0.0)
    assert oms.none_drawn(5, 2, 2) == pytest.approx(3 / 10) and oms.none_drawn(5, 0, 2) == 1.0 and oms.none_drawn(4, 1, 4) == 0.0
    # brute force
    for n, s, k in ((6, 2, 2), (8, 3, 1), (7, 4, 3), (5, 1, 2)):
        combos = list(itertools.combinations(range(n), s))
        assert oms.none_drawn(n, k, s) == pytest.approx(sum(1 for c in combos if not set(c) & set(range(k))) / len(combos))
        for i in range(n):
            assert oms.top_probs(n, s)[i] == pytest.approx(sum(1 for c in combos if min(c) == i) / len(combos))


def test_move_roles():
    f = lambda m, ghost=False: {n for k, n in enumerate(oms.FLAG_NAMES) if oms.move_flags(m, ghost) >> k & 1}  # noqa: E731
    assert f("swordsdance") == {"setup"} and f("dragondance") == {"setup"} and f("curse") == {"setup"} and f("curse", True) == set()
    assert f("extremespeed") == {"priority"} and f("suckerpunch") == {"priority"} and f("protect") == set()
    assert f("recover") == {"recovery"} and f("wish") == {"recovery"} and f("gigadrain") == set()
    assert f("uturn") == {"pivot"} and f("shedtail") == {"pivot"} and f("chillyreception") == {"pivot"}
    for m in ("spore", "thunderwave", "willowisp", "toxic", "yawn", "glare"):
        assert f(m) == {"status"}, m
    assert f("nuzzle") == set()                                    # an attack with a chance to paralyse
    assert f("stealthrock") == {"hazard"} and f("ceaselessedge") == {"hazard"} and f("stoneaxe") == {"hazard"}
    assert f("rapidspin") == {"hazard_removal"} and f("defog") == {"hazard_removal"} and f("mortalspin") == {"hazard_removal"}
    assert f("tidyup") == {"setup", "hazard_removal"} and f("courtchange") == {"hazard_removal"}
    assert f("roar") == {"phazing"} and f("haze") == {"phazing"} and f("dragontail") == {"phazing"}


# --------------------------------------------------------------------------- hand-made battles


def test_own_side_is_exact(asm, enc):
    gar = _pokemon("garchomp", ["earthquake", "outrage", "swordsdance", "stealthrock"], item="loadeddice", ability="roughskin", tera="Steel")
    rotom = _pokemon("rotomwash", ["hydropump"])                  # Levitate (its only ability)
    heatran = _pokemon("heatran")
    b = _battle([gar], [rotom, heatran])
    ms = _blk(enc, b)
    row = ms[0]
    assert _cands(row) == {_mid(asm, m): 1.0 for m in ("earthquake", "outrage", "swordsdance", "stealthrock")}
    tv = row[oms.MS_TERA[0]:oms.MS_TERA[1]]
    assert tv[oms.TERA_INDEX["steel"]] == 1.0 and tv.sum() == 1.0
    assert _flags(row) == {"setup": 1, "priority": 0, "recovery": 0, "pivot": 0, "status": 0, "hazard": 1,
                           "hazard_removal": 0, "phazing": 0}
    # threat: Earthquake does nothing to Levitate Rotom, Outrage 1x (Water/Electric): best 0, no super-effective move;
    # Heatran: Earthquake 4x
    thr, se = row[oms.MS_THREAT[0]:oms.MS_THREAT[1]], row[oms.MS_SE[0]:oms.MS_SE[1]]
    assert thr.tolist() == [0.0, 2.0, 0, 0, 0, 0] and se.tolist() == [0, 1, 0, 0, 0, 0]
    # the matchup block's own move tokens say the same (tests/test_obs_matchup.py holds those against matchup.py)
    mu = om.MatchupEncoder(asm.vocab_map, full_types=True).block(b).astype(np.float32)[:om.MOVE_BLOCK].reshape(12, 4, om.MOVE_FEATS)
    eff = mu[0, :2, om.MOVE_EFF[0]:om.MOVE_EFF[0] + 2]
    assert eff.max(axis=0).tolist() == thr[:2].tolist()
    # unseen opponents and empty slots stay zero
    assert not ms[1:6].any() and not ms[8:].any()
    # it Terastallizes: no Tera type left to guess
    gar.terastallize("steel")
    row = _blk(enc, b)[0]
    assert not row[oms.MS_TERA[0]:oms.MS_TERA[1]].any() and _cands(row) == {_mid(asm, m): 1.0 for m in ("earthquake", "outrage", "swordsdance", "stealthrock")}


def test_opponent_candidates_narrow_as_it_shows_moves(asm, enc):
    own = [_pokemon("garchomp", ["earthquake", "outrage", "swordsdance", "stealthrock"], item="loadeddice", ability="roughskin", tera="Steel"),
           _pokemon("greninja", ["surf", "icebeam", "uturn", "gunkshot"], item="lifeorb", ability="protean", tera="Water")]
    with open(om.SETS_FILE, encoding="utf-8") as f:
        sets = json.load(f)["species"]
    # a species with several roles and more than four moves in them
    species = "rotomwash"
    assert len(sets[species]) >= 2 and any(len(r["moves"]) > 4 for r in sets[species])
    opp = _pokemon(species)
    b = _battle(own, [opp])
    foes = [om.defender_info(m, om.own_guess(m), True) for m in own] + [None] * 4

    before = _blk(enc, b)[6]
    _check_against_brute(asm, before, opp, foes)
    n0 = len(_cands(before))
    assert sum(_cands(before).values()) == pytest.approx(4.0, abs=0.01) or n0 == oms.K_MOVES
    tv = before[oms.MS_TERA[0]:oms.MS_TERA[1]]
    assert tv.sum() == pytest.approx(1.0, abs=2e-3)

    revealed = []
    for mv in ("hydropump", "voltswitch", "willowisp"):
        if not any(mv in r["moves"] for r in sets[species]):
            continue
        opp._add_move(mv)
        revealed.append(mv)
        row = _blk(enc, b)[6]
        _check_against_brute(asm, row, opp, foes)
        got = _cands(row)
        assert _mid(asm, mv) not in got                            # a move it has shown is no longer a candidate
        assert sum(got.values()) <= 4 - len(revealed) + 0.01
        assert len(got) <= len(_cands(before)) <= n0
        if mv == "voltswitch":
            assert _flags(row)["pivot"] == 1.0
        before = row
    assert len(revealed) >= 2

    # all four shown: nothing left to guess, and the threat is the best of its own four
    for mv in sets[species][0]["moves"]:
        if len(opp.moves) >= 4:
            break
        if mv not in opp.moves:
            opp._add_move(mv)
    row = _blk(enc, b)[6]
    assert not _cands(row)
    _check_against_brute(asm, row, opp, foes)

    # it Terastallizes: the Tera vector goes to zero
    opp.terastallize("steel")
    row = _blk(enc, b)[6]
    assert not row[oms.MS_TERA[0]:oms.MS_TERA[1]].any()
    _check_against_brute(asm, row, opp, foes)


def test_tiny_cache_limits_change_nothing(asm, monkeypatch):
    """Every cache of the shared store emptied over and over (limits of 3): the same bytes as the reference."""
    monkeypatch.setattr(oms, "_LIMITS", {k: 3 for k in oms._LIMITS})
    encs = [oms.MovesetEncoder(asm.vocab_map, om.MatchupEncoder(asm.vocab_map, full_types=True)) for _ in range(2)]
    assert encs[0].store is encs[1].store                         # one store per process (same sets, same vocabulary)
    own = [_pokemon("garchomp", ["earthquake", "outrage", "swordsdance", "stealthrock"], item="loadeddice", ability="roughskin", tera="Steel"),
           _pokemon("corviknight", ["bravebird", "roost", "uturn", "defog"], item="leftovers", ability="pressure", tera="Dragon")]
    opp = [_pokemon("rotomwash"), _pokemon("heatran"), _pokemon("clodsire")]
    b = _battle(own, opp)
    gen0 = encs[0].store.gen
    for step, mv in enumerate(["hydropump", "magmastorm", "earthquake", "voltswitch", "toxic", "willowisp"]):
        opp[step % 3]._add_move(mv)
        for e in encs:
            _blk(e, b)
    assert encs[0].store.gen > gen0                               # the store was emptied on the way


def test_several_species_against_the_brute_force(asm, enc):
    own = [_pokemon("garchomp", ["earthquake", "outrage", "swordsdance", "stealthrock"], item="loadeddice", ability="roughskin", tera="Steel"),
           _pokemon("corviknight", ["bravebird", "roost", "uturn", "defog"], item="leftovers", ability="pressure", tera="Dragon"),
           _pokemon("gholdengo", ["makeitrain", "shadowball", "nastyplot", "recover"], item="choicescarf", ability="goodasgold", tera="Steel")]
    foes = [om.defender_info(m, om.own_guess(m), True) for m in own] + [None] * 3
    n = 0
    for sp, shown in (("garchomp", []), ("garchomp", ["earthquake"]), ("heatran", ["magmastorm"]), ("clodsire", []),
                      ("dragonite", ["extremespeed"]), ("gengar", []), ("toxapex", ["toxic"]), ("kingambit", ["suckerpunch", "swordsdance"])):
        opp = _pokemon(sp, shown)
        row = _blk(enc, _battle(own, [opp]))[6]
        _check_against_brute(asm, row, opp, foes)
        n += 1
    assert n == 8


def _thr(row):
    return row[oms.MS_THREAT[0]:oms.MS_THREAT[1]].tolist(), row[oms.MS_SE[0]:oms.MS_SE[1]].tolist()


def _mu_tokens(vocab_map, battle, full):
    """The matchup block's move tokens (fast path, checked against its reference path), float32 [12, 4, 18]."""
    e = om.MatchupEncoder(vocab_map, full_types=full)
    fast, slow = e.block(battle), e.block(battle, reference=True)
    assert fast.tobytes() == slow.tobytes()
    return fast.astype(np.float32)[:om.MOVE_BLOCK].reshape(12, 4, om.MOVE_FEATS)


def test_full_move_types_in_the_threat(asm, enc):
    """Abilities, forms, Tera and the weather set a move's type (move_type(full=True)), shown or not."""
    own = [_pokemon("gengar", ["shadowball", "sludgewave", "focusblast", "nastyplot"], item="lifeorb", ability="cursedbody", tera="Dark"),
           _pokemon("garchomp", ["earthquake", "outrage", "swordsdance", "stealthrock"], item="loadeddice", ability="roughskin", tera="Steel"),
           _pokemon("dragonite", ["dragondance", "earthquake", "outrage", "roost"], item="heavydutyboots", ability="multiscale", tera="Steel")]
    foes = [om.defender_info(m, om.own_guess(m), True) for m in own] + [None] * 3

    def row_of(opp, battle_own=own, weather=None, team_foes=foes):
        row = _blk(enc, _battle(battle_own, [opp], weather))[6]
        _check_against_brute(asm, row, opp, team_foes, None, weather)
        return row

    # Pixilate: Sylveon's only attack, Hyper Voice, is Fairy: resisted by Gengar (Poison), super effective on the Dragons
    assert _thr(row_of(_pokemon("sylveon"))) == ([-1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0])
    # Liquid Voice: Psychic Noise is Water, so 1x on Gengar (Moonblast, Draining Kiss resisted)
    assert _thr(row_of(_pokemon("primarina", ["psychicnoise"]))) == ([0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0])
    # ... and before it is shown, only in the Liquid Voice role (the Torrent roles have Psychic: 2x on Gengar)
    thr, se = _thr(row_of(_pokemon("primarina")))
    assert thr[0] == pytest.approx(2 / 3, abs=1e-3) and se[0] == pytest.approx(2 / 3, abs=1e-3)
    # Tera Starstorm of a Terastallized Terapagos is Stellar: 1x on Gengar; Earth Power / Dark Pulse drawn 3 in 4
    for sp in ("terapagosstellar", "terapagos"):
        tp = _pokemon(sp, ["terastarstorm"])
        tp.terastallize("stellar")
        thr, se = _thr(row_of(tp))
        assert thr[0] == pytest.approx(0.75, abs=1e-3) and se[0] == pytest.approx(0.75, abs=1e-3), sp
    # Revelation Dance takes the user's first type: Ghost for Oricorio-Sensu (2x on Gengar), its Tera type after
    ori = _pokemon("oricoriosensu", ["revelationdance"])
    assert _thr(row_of(ori))[0][0] == 1.0 and _thr(row_of(ori))[1][0] == 1.0
    ori.terastallize("fighting")
    assert _thr(row_of(ori))[0][0] == 0.0 and _thr(row_of(ori))[1][0] == 0.0     # Fighting: Gengar is immune; Hurricane 1x
    # Galvanize: Double-Edge (and Explosion) are Electric, 2x on Corviknight
    corv = [_pokemon("corviknight", ["bravebird", "roost", "uturn", "defog"], item="leftovers", ability="pressure", tera="Dragon")]
    corv_foes = [om.defender_info(m, om.own_guess(m), True) for m in corv] + [None] * 5
    assert _thr(row_of(_pokemon("golemalola", ["doubleedge"]), corv, None, corv_foes)) == ([1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0])
    # Weather Ball, the same Sunflora as the weather changes: Fire in the sun (2x on Corviknight), Normal without
    # weather (resisted), Water in rain, Rock in sand (1x); a known Utility Umbrella hides the sun
    sunflora = _pokemon("sunflora", ["weatherball"])
    for weather, want in (("sunnyday", 1.0), (None, -1.0), ("raindance", 0.0), ("sandstorm", 0.0), ("sunnyday", 1.0)):
        thr, se = _thr(row_of(sunflora, corv, weather, corv_foes))
        assert thr[0] == want and se[0] == (1.0 if want > 0 else 0.0), weather
        tok = _mu_tokens(asm.vocab_map, _battle(corv, [sunflora], weather), True)
        assert tok[6, 0, om.MOVE_EFF[0]] == want, weather
    sunflora.item = "utilityumbrella"
    assert _thr(row_of(sunflora, corv, "sunnyday", corv_foes))[0][0] == -1.0


def test_full_types_reach_the_matchup_tokens_only_with_the_block(asm):
    """With the block the move tokens use the full types (the threat is their best); without it, efe43f7's."""
    assert asm._matchup_encoder.full_types
    assert not ObservationAssembler(own_tera_type=True, matchup=True)._matchup_encoder.full_types
    sylveon = _pokemon("sylveon", ["hypervoice", "calmmind", "protect", "wish"], item="leftovers", ability="pixilate", tera="Steel")
    b = _battle([sylveon], [_pokemon("gengar")])
    assert _mu_tokens(asm.vocab_map, b, True)[0, 0, om.MOVE_EFF[0]] == -1.0              # Fairy on Ghost/Poison
    assert _mu_tokens(asm.vocab_map, b, False)[0, 0, om.MOVE_EFF[0]] == om.IMMUNE_LOG2   # efe43f7: Normal on a Ghost
    enc = oms.MovesetEncoder(asm.vocab_map, om.MatchupEncoder(asm.vocab_map, full_types=True))
    assert _thr(_blk(enc, b)[0]) == ([-1, 0, 0, 0, 0, 0], [0] * 6)
    # the whole observation: the matchup-only one keeps efe43f7's token, the one with the block has the full type
    a1 = ObservationAssembler(own_tera_type=True, matchup=True)
    m0 = asm.offsets["matchup"][0]
    assert a1._matchup_encoder.block(b)[0] == om.IMMUNE_LOG2
    full = np.zeros(om.BLOCK_DIM, dtype=np.float16)
    asm._matchup_encoder.encode(b, full, 0)
    assert full[0] == -1.0 and m0 == a1.offsets["matchup"][0]


def test_illusion_moves_go_back_to_the_illusion_user(asm, enc):
    """
    A real poke-env battle: Zoroark, disguised as Garchomp, uses Nasty Plot, Dark Pulse and Focus Blast; poke-env
    keeps them on Garchomp after the Illusion breaks. Once Zoroark is revealed the block counts them as Zoroark's.
    """
    import logging
    from poke_env.battle import Battle
    pb = Battle("battle-gen9randombattle-1", "p1user", logging.getLogger("illusion-test"), gen=9)
    pb._player_role = "p1"

    def feed(lines):
        for ln in lines:
            pb.parse_message(ln.split("|"))

    feed(["|player|p1|p1user|", "|player|p2|p2user|",
          "|switch|p1a: Dragonite|Dragonite, L74, M|100/100", "|switch|p2a: Garchomp|Garchomp, L77, M|100/100", "|turn|1",
          "|move|p2a: Garchomp|Nasty Plot|p2a: Garchomp", "|turn|2",
          "|move|p2a: Garchomp|Dark Pulse|p1a: Dragonite", "|turn|3",
          "|move|p2a: Garchomp|Focus Blast|p1a: Dragonite|[miss]"])
    own = _pokemon("dragonite", ["extremespeed", "dragondance", "earthquake", "roost"], item="heavydutyboots", ability="multiscale", tera="Normal")
    battle = SimpleNamespace(team={"p1: Dragonite": own}, opponent_team=pb.opponent_team)
    foes = [om.defender_info(own, om.own_guess(own), True)] + [None] * 5
    disguise = ["nastyplot", "darkpulse", "focusblast"]
    # in disguise, nothing tells Garchomp's moves from Zoroark's: the block reads what poke-env shows
    before = _blk(enc, battle)
    assert before.tobytes() == _blk(enc, _battle([own], [_pokemon("garchomp", disguise)])).tobytes()
    assert oms.illusion_adjust(enc.mu.mons(battle), enc.table) is None

    feed(["|move|p1a: Dragonite|Extreme Speed|p2a: Garchomp", "|replace|p2a: Zoroark|Zoroark, L84, M",
          "|-end|p2a: Zoroark|Illusion", "|-damage|p2a: Zoroark|40/100", "|turn|4"])
    gar, zor = list(pb.opponent_team.values())
    assert gar.species == "garchomp" and zor.species == "zoroark" and sorted(gar.moves) == sorted(disguise) and not zor.moves
    after = _blk(enc, battle)
    clean = _blk(enc, _battle([own], [_pokemon("garchomp"), _pokemon("zoroark", disguise)]))
    assert after.tobytes() == clean.tobytes()                       # Garchomp as if it had shown nothing, Zoroark with the three
    assert after[6].tobytes() != before[6].tobytes()
    for m in (gar, zor):
        _check_against_brute(asm, after[6 + (m is zor)], m, foes, [gar, zor])
    assert oms.illusion_adjust(enc.mu.mons(battle), enc.table)[6:8] == [(frozenset(disguise), ()), (frozenset(), tuple(sorted(disguise)))]

    # the real Garchomp comes back and shows two moves of its own
    feed(["|switch|p2a: Garchomp|Garchomp, L77, M|100/100", "|move|p2a: Garchomp|Earthquake|p1a: Dragonite", "|turn|5",
          "|move|p2a: Garchomp|Stealth Rock|p1a: Dragonite", "|turn|6"])
    later = _blk(enc, battle)
    want = _blk(enc, _battle([own], [_pokemon("garchomp", ["earthquake", "stealthrock"]), _pokemon("zoroark", disguise)]))
    assert later.tobytes() == want.tobytes()
    assert _flags(later[6])["setup"] < 1.0 and _flags(later[6])["hazard"] == 1.0


# --------------------------------------------------------------------------- recorded battles


@pytest.fixture(scope="module")
def replayed(asm):
    import matchup_replay
    if not matchup_replay.stream_paths():
        pytest.skip("no recorded battles: set PSPPO_STREAMS")
    mu_asm = ObservationAssembler(own_tera_type=True, matchup=True)
    assert asm._matchup_encoder.full_types and not mu_asm._matchup_encoder.full_types
    ref_enc = oms.MovesetEncoder(asm.vocab_map, om.MatchupEncoder(asm.vocab_map, full_types=True))
    rng = np.random.default_rng(0)
    checks = {"brute": 0, "tera after": 0, "tera before": 0}

    def inspect(battle, events=None):
        """What the battle holds at this decision, and every 7th decision the brute force on its opponents."""
        own = list(battle.team.values())
        opp = list(battle.opponent_team.values())
        weather = om.battle_weather(battle)
        table = om.SetTable.default()
        info = {"opp": [(sorted(illusion_view(m, opp)), bool(m.is_terastallized)) for m in opp],
                "own_tera": [None if m.is_terastallized else oms.own_tera(m) for m in own]}
        # the move tokens whose full type (with the moveset block) is not efe43f7's: the only ones the
        # block may change in the matchup block
        retyped = np.zeros((12, 4), dtype=bool)
        for i, m in enumerate(own[:6] + [None] * (6 - len(own[:6])) + opp[:6]):
            if m is None:
                continue
            ab = (om.own_guess(m) if i < 6 else om.opponent_guess(m, table)).known_ability
            for k, mv in enumerate(list(m.moves.values())[:4]):
                if mv is not None:
                    retyped[i, k] = om.move_type(mv, m) != om.move_type(mv, m, ab, weather, full=True)
        info["retyped"] = retyped
        if rng.random() < 1 / 7:
            foes = [om.defender_info(m, om.own_guess(m), True) for m in own] + [None] * (6 - len(own))
            info["brute"] = [(j, brute_force(m, foes, asm.vocab_map, opp, weather)) for j, m in enumerate(opp)]
            # our side: the best log2 multiplier of its damaging moves on each opponent, by matchup.py
            theirs = [om.defender_info(m, om.opponent_guess(m, table), False) for m in opp] + [None] * (6 - len(opp))
            own_thr = []
            for m in own:
                infos = [om.move_info(mv, m, om.own_guess(m).known_ability, weather, full=True) for mv in list(m.moves.values())[:4]]
                t = np.zeros(12)
                for j, d in enumerate(theirs):
                    if d is None:
                        continue
                    codes = [_code(mi, d) for mi in infos if mi.damaging]
                    t[j] = max(codes) if codes else om.IMMUNE_LOG2
                    t[6 + j] = 1.0 if any(c > 0 for c in codes) else 0.0
                own_thr.append(t)
            info["own_thr"] = own_thr
        return info

    # A second assembler with the block that sees every other decision of every battle: it shares the
    # process-wide store with the first (as the players of one rollout worker do) and must write the same bytes.
    asm2 = ObservationAssembler(own_tera_type=True, matchup=True, movesets=True)
    seen = {"n": 0}

    def second(battle, events=None):
        seen["n"] += 1
        return asm2.assemble(battle, events=events) if seen["n"] % 2 else None

    out = matchup_replay.replay({"movesets": asm, "matchup": mu_asm, "inspect": SimpleNamespace(assemble=inspect),
                                 "ms_reference": SimpleNamespace(assemble=lambda battle, events=None: ref_enc.block(battle, reference=True)),
                                 "movesets2": SimpleNamespace(assemble=second)})
    out["checks"] = checks
    return out


def test_recorded_battles(asm, replayed):
    recs = replayed["records"]
    assert len(recs) >= 2000, f"only {len(recs)} decisions"
    s0, s1 = asm.offsets["movesets"]
    m0, m1 = asm.offsets["matchup"]
    stats = {"brute force Pokemon": 0, "opp Terastallized": 0, "opp not Terastallized": 0, "own Terastallized": 0,
             "candidates by moves shown": {k: [] for k in range(5)}, "a super-effective move possible, not certain": 0}
    bad = []
    for r in recs:
        ms, mu, d = r["movesets"], r["matchup"], r["default"]
        assert not isinstance(ms, Exception) and not isinstance(mu, Exception) and not isinstance(r["inspect"], Exception), r
        ins = r["inspect"]
        # before the block: the plain observation and the matchup block's Pokemon rows as the matchup-only
        # observation has them; its move tokens too, except a move whose full type is not efe43f7's
        if ms[:m0].tobytes() != d.tobytes() or mu[:m0].tobytes() != d.tobytes() or \
                ms[m0 + om.MOVE_BLOCK:s0].tobytes() != mu[m0 + om.MOVE_BLOCK:].tobytes():
            bad.append(("prefix", r["key"], r["turn"]))
        tok_ms = ms[m0:m0 + om.MOVE_BLOCK].view(np.uint16).reshape(12, 4, om.MOVE_FEATS)
        tok_mu = mu[m0:m0 + om.MOVE_BLOCK].view(np.uint16).reshape(12, 4, om.MOVE_FEATS)
        differ = (tok_ms != tok_mu).any(axis=2)
        if (differ & ~ins["retyped"]).any():
            bad.append(("a move token changed with its type unchanged", r["key"], r["turn"]))
        stats["move tokens retyped"] = stats.get("move tokens retyped", 0) + int(ins["retyped"].sum())
        stats["move tokens retyped, numbers changed"] = stats.get("move tokens retyped, numbers changed", 0) + int(differ.sum())
        if ms[s0:s1].tobytes() != r["ms_reference"].tobytes():
            bad.append(("fast != reference", r["key"], r["turn"]))
        if r["movesets2"] is not None:
            stats["second assembler"] = stats.get("second assembler", 0) + 1
            if r["movesets2"].tobytes() != ms.tobytes():
                bad.append(("second assembler (shared store) != first", r["key"], r["turn"]))
        blk = ms[s0:s1].astype(np.float32).reshape(12, oms.MS_FEATS)
        assert np.isfinite(blk).all()
        mv_tok = ms[m0:m1].astype(np.float32)[:om.MOVE_BLOCK].reshape(12, 4, om.MOVE_FEATS)
        # our side: exact
        for i in range(6):
            row = blk[i]
            if not row.any():
                continue
            p = row[oms.MS_MOVE_P[0]:oms.MS_MOVE_P[1]]
            assert set(np.unique(p).tolist()) <= {0.0, 1.0}
            assert set(np.unique(row[oms.MS_FLAGS[0]:oms.MS_FLAGS[1]]).tolist()) <= {0.0, 1.0}
            tv = row[oms.MS_TERA[0]:oms.MS_TERA[1]]
            want_tera = ins["own_tera"][i] if i < len(ins["own_tera"]) else None
            if want_tera is None:
                assert not tv.any()
                stats["own Terastallized"] += 1
            else:
                assert tv[oms.TERA_INDEX[want_tera]] == 1.0 and tv.sum() == 1.0
            # threat: the best code of its own damaging move tokens against each seen opponent (status tokens hold 0)
            seen = np.arange(6) < len(ins["opp"])
            se = row[oms.MS_SE[0]:oms.MS_SE[1]]
            thr = row[oms.MS_THREAT[0]:oms.MS_THREAT[1]]
            assert not thr[~seen].any() and not se[~seen].any()
            assert set(np.unique(se).tolist()) <= {0.0, 1.0}
            assert ((se > 0) == ((thr > 0) & seen)).all()
            # every token code is at most the threat (a status move's 0 aside)
            eff = mv_tok[i, :, om.MOVE_EFF[0]:om.MOVE_EFF[1]]
            assert (eff[:, seen] <= np.maximum(thr[seen], 0.0)[None, :] + 1e-3).all()
            # and exactly matchup.py's best over its damaging moves (worked out from the battle every 7th decision)
            if "own_thr" in ins and i < len(ins["own_thr"]):
                assert row[oms.MS_THREAT[0]:oms.MS_SE[1]] == pytest.approx(ins["own_thr"][i], abs=1e-3)
                stats["own threat checked"] = stats.get("own threat checked", 0) + 1
        # theirs
        for j, (shown, tera) in enumerate(ins["opp"]):
            row = blk[6 + j]
            ids = row[oms.MS_MOVE_IDS[0]:oms.MS_MOVE_IDS[1]]
            p = row[oms.MS_MOVE_P[0]:oms.MS_MOVE_P[1]]
            shown_ids = {asm.vocab_map["move.id"].get(m, -1) + 1 for m in shown}
            assert not (set(ids[ids > 0].astype(int).tolist()) & shown_ids)
            n_shown = min(4, len([m for m in shown if m not in ("struggle", "recharge")]))
            assert (p >= 0).all() and (p <= 1.0).all() and p.sum() <= 4 - n_shown + 0.02
            stats["candidates by moves shown"][n_shown].append(int((ids > 0).sum()))
            tv = row[oms.MS_TERA[0]:oms.MS_TERA[1]]
            if tera:
                assert not tv.any()
                stats["opp Terastallized"] += 1
            else:
                assert tv.sum() == pytest.approx(1.0, abs=5e-3) or not tv.any()
                stats["opp not Terastallized"] += 1
            se = row[oms.MS_SE[0]:oms.MS_SE[1]]
            stats["a super-effective move possible, not certain"] += int(((se > 0.01) & (se < 0.99)).any())
        for j, (moves_p, tera_p, flags, thr) in ins.get("brute", []):
            row = blk[6 + j]
            ranked = sorted(((m, p) for m, p in moves_p.items() if p > 0 and m in asm.vocab_map["move.id"]), key=lambda kv: (-kv[1], kv[0]))
            want = {asm.vocab_map["move.id"][m] + 1: p for m, p in ranked[:oms.K_MOVES]}
            got = _cands(row)
            # ties at the 8th place may be cut differently by float16: compare what is above the cut
            cut = ranked[oms.K_MOVES - 1][1] if len(ranked) >= oms.K_MOVES else -1.0
            assert {k for k, v in want.items() if v > cut + 1e-6} <= set(got), (r["key"], r["turn"], got, want)
            for k, v in got.items():
                assert v == pytest.approx(want.get(k, cut), abs=2e-3)
            tv = row[oms.MS_TERA[0]:oms.MS_TERA[1]]
            assert tv == pytest.approx(np.array([tera_p.get(t, 0.0) for t in oms.TERA_TYPES]), abs=2e-3)
            assert row[oms.MS_FLAGS[0]:oms.MS_FLAGS[1]] == pytest.approx(flags, abs=2e-3)
            assert row[oms.MS_THREAT[0]:oms.MS_SE[1]] == pytest.approx(thr, abs=4e-3), (r["key"], r["turn"], j)
            stats["brute force Pokemon"] += 1
    means = {k: (round(float(np.mean(v)), 2) if v else None, len(v)) for k, v in stats["candidates by moves shown"].items()}
    stats["candidates by moves shown"] = means
    t = replayed["time"]
    print("\n" + json.dumps({"decisions": len(recs), **stats,
                             "assemble ms per decision": {k: round(1000 * v / len(recs), 4) for k, v in t.items()}}, indent=2))
    assert not bad, bad[:5]
    assert stats["brute force Pokemon"] >= 300 and stats["opp Terastallized"] > 0 and stats["own Terastallized"] > 0
    assert stats.get("own threat checked", 0) >= 300 and stats.get("second assembler", 0) >= 1000
    assert stats["a super-effective move possible, not certain"] > 0
    m = [means[k][0] for k in range(5) if means[k][0] is not None]
    assert m == sorted(m, reverse=True) and means[4][0] in (0.0, None)          # fewer candidates as more moves are shown


# --------------------------------------------------------------------------- with the setting off: nothing changes

_BASE_REPLAY = r"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "tests"))
import matchup_replay
from obs_assembler import ObservationAssembler
out = matchup_replay.replay({"matchup": ObservationAssembler(own_tera_type=True, matchup=True)})
recs = out["records"]
np.savez(sys.argv[1], default=np.stack([r["default"] for r in recs]), matchup=np.stack([r["matchup"] for r in recs]),
         keys=np.array([json.dumps([list(r["key"]), r["turn"]]) for r in recs]))
"""


@pytest.fixture(scope="module")
def base_obs():
    import matchup_replay
    if not matchup_replay.stream_paths():
        pytest.skip("no recorded battles: set PSPPO_STREAMS")
    if not BASE_DIR or not os.path.isfile(os.path.join(BASE_DIR, "obs_matchup.py")):
        pytest.skip("PSPPO_MOVESETS_BASE names no copy of efe43f7")
    assert not os.path.exists(os.path.join(BASE_DIR, "obs_movesets.py")), "PSPPO_MOVESETS_BASE holds this branch's code"
    path = os.path.join(tempfile.mkdtemp(prefix="psppo_ms_base_"), "base.npz")
    r = subprocess.run([sys.executable, "-c", _BASE_REPLAY, path], cwd=BASE_DIR, capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=""))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    return dict(np.load(path))


def test_observation_with_the_setting_off_is_byte_for_byte_efe43f7s(base_obs, replayed):
    recs = replayed["records"]
    assert len(recs) == base_obs["default"].shape[0]
    keys = [json.dumps([list(r["key"]), r["turn"]]) for r in recs]
    assert keys == base_obs["keys"].tolist()
    default = np.stack([r["default"] for r in recs])
    matchup = np.stack([r["matchup"] for r in recs])
    assert default.dtype == base_obs["default"].dtype and default.tobytes() == base_obs["default"].tobytes()
    assert matchup.dtype == base_obs["matchup"].dtype and matchup.tobytes() == base_obs["matchup"].tobytes()
    print(f"\n{len(recs)} recorded decisions: the plain and the matchup observation byte for byte efe43f7's")


@pytest.fixture(scope="module")
def base_code():
    if not BASE_DIR or not os.path.isfile(os.path.join(BASE_DIR, "ppo_core.py")):
        pytest.skip("PSPPO_MOVESETS_BASE names no copy of efe43f7")
    import ppo_core as new_core
    from bench import micro_ref
    ref = micro_ref.Reference(BASE_DIR, source="PSPPO_MOVESETS_BASE")
    core, lrn, cfg = ref.load("ppo_core"), ref.load("learner"), ref.load("config")
    assert lrn.ppo_core is core and core is not new_core and not hasattr(core, "_move_bce")
    import torch
    if torch.cuda.is_available():
        micro_ref.shared_compiled_flex_attention(core, new_core)
    core._learner_flex_compiled = True
    new_core._learner_flex_compiled = True
    return SimpleNamespace(core=core, learner=lrn, config=cfg, ref=ref)


def _obs_seqs(replayed, name, n):
    import matchup_replay
    return matchup_replay.sequences(replayed["records"], name, n)


def test_network_with_the_setting_off_is_bit_for_bit_efe43f7s(base_code, replayed):
    import torch
    from config import RunConfig
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for matchup in (False, True):
        with base_code.ref.active():
            oc = base_code.config.RunConfig.default()
            if matchup:
                oc = dataclasses.replace(oc, obs=dataclasses.replace(oc.obs, matchup=True))
            old_net = oc.make_model()
        nc = RunConfig.default()
        if matchup:
            nc = dataclasses.replace(nc, obs=dataclasses.replace(nc.obs, matchup=True))
        new_net = nc.make_model()
        assert type(old_net) is not type(new_net) and new_net.movesets is False
        new_net.load_state_dict(old_net.state_dict(), strict=True)
        seqs = _obs_seqs(replayed, "matchup" if matchup else "default", 1000)
        x = torch.from_numpy(np.concatenate(seqs).astype(np.float32))
        for bf16 in (False, True):
            a, b = old_net.to(dev).eval(), new_net.to(dev).eval()
            if bf16:
                a.enable_bf16_recurrent_path(), b.enable_bf16_recurrent_path()
            with torch.no_grad():
                for i in range(0, x.shape[0], 256):
                    xb = x[i:i + 256].to(dev)
                    for ta, tb in zip(a.encode_features(xb), b.encode_features(xb)):
                        assert torch.equal(ta, tb)
                    for ta, tb in zip(a(xb)[:3], b(xb)[:3]):
                        assert torch.equal(ta, tb)
        print(f"\nmatchup={matchup}: {x.shape[0]} recorded observations, fp32 and bf16 on {dev}: bit for bit efe43f7's network")


def _labels_for(obs: np.ndarray, asm, n: int, seed: int, width: int):
    """Plausible SetHead labels for recorded observations: the opponent species of each row, abilities, items (and moves, Tera) its sets allow."""
    rng = np.random.default_rng(seed)
    ab, it = om.set_candidate_tables()
    mv, te = oms.candidate_tables()
    p0 = asm.offsets["pokemon_ids"][0]
    out = np.zeros((n, 6, width), dtype=np.int32)
    for r in range(n):
        for j in range(6):
            sp = int(obs[r, p0 + (6 + j) * 2])
            if sp <= 0 or rng.random() < 0.3:
                continue
            out[r, j, 0] = sp
            for col, table in ((1, ab), (2, it)):
                cands = np.flatnonzero(table[sp])
                if len(cands) and rng.random() < 0.7:
                    out[r, j, col] = rng.choice(cands)
            if width >= 8:
                cands = np.flatnonzero(mv[sp])
                if len(cands):
                    pick = rng.choice(cands, size=min(4, len(cands)), replace=False)
                    signs = np.where(rng.random(len(pick)) < 0.3, -1, 1)
                    out[r, j, 3:3 + len(pick)] = pick * signs
                tc = np.flatnonzero(te[sp])
                if len(tc) and rng.random() < 0.8:
                    out[r, j, 7] = rng.choice(tc)
    return torch_from(out.reshape(n, 6 * width))


def torch_from(a):
    import torch
    return torch.from_numpy(a)


def _learner_cfg(config_mod, mode, matchup, movesets=False, **over):
    import test_speed_update as tsu
    cfg = tsu._learner_cfg(config_mod, mode, **over)
    obs = dataclasses.replace(cfg.obs, matchup=matchup, **({"movesets": True} if movesets else {}))
    return dataclasses.replace(cfg, obs=obs)


def _recorded_update_data(replayed, name, n, seed):
    import matchup_replay
    import test_speed_update as tsu
    seqs = matchup_replay.sequences(replayed["records"], name, n)
    obs = np.concatenate(seqs)[:n]
    ep = np.concatenate([np.full(len(s), i, dtype=np.int64) for i, s in enumerate(seqs)])[:n]
    return obs, tsu._update_data(obs, ep, n, seed)


def test_a_ppo_update_with_the_setting_off_is_bit_for_bit_efe43f7s(base_code, replayed, asm):
    import torch
    import test_speed_update as tsu
    from bench import micro_ref
    if not torch.cuda.is_available():
        pytest.skip("flex_attention has no backward on a CPU")
    n = 1536
    obs, cols = _recorded_update_data(replayed, "matchup", n, seed=31)
    labels = _labels_for(obs, asm, n, seed=32, width=3)
    for mode, over in (("ppo", {"set_head_coef": 0.1}), ("ppo", {"set_head_coef": 0.1, "opp_action_aux_coef": 0.1}), ("imitation", {})):
        with base_code.ref.active():
            old = tsu._make_learner(base_code.learner, base_code.core,
                                    _learner_cfg(base_code.config, mode, True, save_every_updates=10**9, **over))
        new = tsu._make_learner(sys.modules["learner"], sys.modules["ppo_core"],
                                _learner_cfg(sys.modules["config"], mode, True, save_every_updates=10**9, **over))
        assert getattr(new.net.set_head, "has_sets", False) is False
        new.net.load_state_dict(old.net.state_dict(), strict=True)
        new.target_net.load_state_dict(old.target_net.state_dict(), strict=True)
        new.opt.sync_masters_from_model()
        extras = {"set_labels": labels} if over.get("set_head_coef") else None
        if over.get("opp_action_aux_coef"):
            extras["opp_action"] = torch.from_numpy(np.random.default_rng(3).integers(-1, 8, n).astype(np.int8))
        lines = []
        for la in (old, new):
            if extras is None:
                la.dataset.add_steps(*cols)
            else:
                la.dataset.add_steps(*cols, extras=extras)
            la.total_steps += n
            torch.manual_seed(79)
            asyncio.run(la._perform_update())
            assert la.update_idx == 1 and len(la.dataset) == 0
        want, got = tsu._state(old), tsu._state(new)
        assert want.keys() == got.keys()
        diff = [k for k in want if not micro_ref.same_bits(want[k], got[k])]
        # Weights, fp32 masters, rates and schedule: bit for bit. Adam's moments too, except that the
        # GPU sums an embedding's gradient over repeated ids in no fixed order (move_emb: seen once in
        # six runs of this test, with the setting off, so not this change), which moves their last bits: then they
        # must agree to float rounding. At the first step Adam's update is sign(g), so the weights do not move.
        assert not [k for k in diff if not k.startswith("adam/")], (mode, over, diff[:5])
        for k in diff:
            a, b = want[k].double(), got[k].double()
            rel = float((a - b).abs().max() / a.abs().max().clamp(min=1e-30))
            assert rel < 1e-5, (mode, over, k, rel)
        print(f"\n{mode} {over}: a whole update on {n} recorded matchup decisions, bit for bit efe43f7's"
              + (f" (Adam moments {diff} within float rounding)" if diff else ""))
        del old, new
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- the model


def test_model_shapes_with_the_block():
    from config import RunConfig
    mu = om.build_model(RunConfig.default(), matchup=True)
    ms = om.build_model(RunConfig.default(), matchup=True, movesets=True)
    assert ms.movesets is True and mu.movesets is False
    assert ms.pokemon_net[0].weight.shape[1] == mu.pokemon_net[0].weight.shape[1] + 64 + 39 and ms.ms_pok_in == 103
    assert ms.move_net[0].weight.shape == mu.move_net[0].weight.shape
    assert [n for n, _ in mu.named_parameters()] == [n for n, _ in ms.named_parameters()]
    with pytest.raises(ValueError):
        dataclasses.replace(RunConfig.default().obs, movesets=True)          # needs matchup


def test_the_block_reaches_the_pokemon_tokens(asm):
    import torch
    from config import RunConfig
    torch.manual_seed(0)
    net = om.build_model(RunConfig.default(), matchup=True, movesets=True).eval()
    gar = _pokemon("garchomp", ["earthquake", "outrage", "swordsdance", "stealthrock"], item="loadeddice", ability="roughskin", tera="Steel")
    b = _battle([gar], [_pokemon("rotomwash", ["hydropump"])])
    s0, s1 = asm.offsets["movesets"]
    x = np.zeros((2, asm.total_dim), dtype=np.float32)
    enc = oms.MovesetEncoder(asm.vocab_map, om.MatchupEncoder(asm.vocab_map))
    x[0, s0:s1] = enc.block(b)
    with torch.no_grad():
        _f, p, _m = net.encode_features(torch.from_numpy(x))
    assert not torch.allclose(p[0, 0], p[1, 0]) and not torch.allclose(p[0, 6], p[1, 6])
    assert torch.allclose(p[0, 3], p[1, 3])                         # an empty slot: no block, the same token


# --------------------------------------------------------------------------- the SetHead's move and Tera parts


def test_set_labels_with_moves_and_tera(asm):
    vm = asm.vocab_map
    clod, rotom = _pokemon("clodsire", ["earthquake"]), _pokemon("rotomwash", ["hydropump", "voltswitch"])
    rotom.terastallize("electric")
    b = _battle([], [clod, rotom])
    keys = list(b.opponent_team)
    truth_rotom = _pokemon("rotomwash", ["hydropump", "voltswitch", "willowisp", "painsplit"], item="leftovers", ability="levitate", tera="Electric")
    truth = {keys[0]: {"species": "clodsire", "ability": "Water Absorb", "item": "Leftovers",
                       "moves": ["Earthquake", "Toxic", "Recover", "Gunk Shot"], "tera": "Flying"},
             keys[1]: truth_rotom}
    lab3 = om.build_set_labels(b, truth, vocab_map=vm)
    lab = om.build_set_labels(b, truth, vocab_map=vm, movesets=True)
    assert lab3.shape == (6, 3) and lab.shape == (6, 8) and lab.dtype == np.int32
    assert (lab[:, :3] == lab3).all()
    mid = lambda n: vm["move.id"][n] + 1  # noqa: E731
    assert lab[0, 3:7].tolist() == [-mid("earthquake"), mid("toxic"), mid("recover"), mid("gunkshot")]
    assert lab[0, 7] == oms.TERA_INDEX["flying"] + 1
    assert lab[1, 3:7].tolist() == [-mid("hydropump"), -mid("voltswitch"), mid("willowisp"), mid("painsplit")]
    assert lab[1, 7] == 0                                          # Terastallized: nothing left to learn
    assert not lab[2:].any()


def test_set_head_move_and_tera_loss():
    import torch
    from ppo_core import SetHead, compute_set_head_loss
    ab, it = om.set_candidate_tables()
    mv, te = oms.candidate_tables()
    vocab = om.load_vocab_lists()
    vm = {k: {x.lower(): i for i, x in enumerate(v)} for k, v in vocab.items()}
    sp = vm["pokemon.species"]["rotomwash"] + 1
    cand = np.flatnonzero(mv[sp])
    assert len(cand) >= 6 and te[sp].sum() >= 2 and not mv[:, 0].any() and not te[:, 0].any()
    torch.manual_seed(0)
    head = SetHead(32, ab, it, mv, te)
    assert [n for n, _ in head.named_parameters()] == ["ability.weight", "ability.bias", "item.weight", "item.bias",
                                                      "moves.weight", "moves.bias", "tera.weight", "tera.bias"]
    assert set(head.state_dict()) == {n for n, _ in head.named_parameters()}             # the tables are not saved
    net = SimpleNamespace(set_head=head, set_head_coef=1.0)
    z = torch.randn(2, 64, 32, requires_grad=True)
    lab = torch.zeros(2, 6, 8, dtype=torch.long)
    true_ids = [int(cand[0]), int(cand[1])]
    lab[0, 0, 0] = sp
    lab[0, 0, 3:7] = torch.tensor([true_ids[0], true_ids[1], -int(cand[2]), 0])
    tera_ok = int(np.flatnonzero(te[sp])[0])
    lab[0, 0, 7] = tera_ok
    term, st = compute_set_head_loss(net, z, lab)
    assert int(st["set_move_n"]) == 1 and int(st["set_tera_n"]) == 1 and int(st["set_ability_n"]) == 0
    # by hand: BCE over the species' candidates without the shown one, and the Tera cross-entropy over its candidates
    lg = head.moves(z[0, 7].detach())
    c = [int(i) for i in cand if int(i) != int(cand[2])]
    y = torch.tensor([1.0 if i in true_ids else 0.0 for i in c])
    bce = torch.nn.functional.binary_cross_entropy_with_logits(lg[c], y)
    tl = head.tera(z[0, 7].detach())
    tc = [int(i) for i in np.flatnonzero(te[sp])]
    ce = -torch.log_softmax(tl[tc], dim=0)[tc.index(tera_ok)]
    assert float(term) == pytest.approx(float((bce + ce) / 2), rel=1e-5)
    term.backward()
    assert z.grad[0, 7].abs().sum() > 0 and z.grad[:, :7].abs().sum() == 0 and z.grad[1].abs().sum() == 0
    # precision at k: put the two true moves on top
    with torch.no_grad():
        head.moves.bias[true_ids] += 50.0
    _t, st = compute_set_head_loss(net, z, lab)
    assert float(st["set_move_hits"]) == pytest.approx(1.0)
    with torch.no_grad():
        head.moves.bias[true_ids] -= 100.0
    _t, st = compute_set_head_loss(net, z, lab)
    assert float(st["set_move_hits"]) == 0.0
    # a move outside the sets changes nothing, nor do 3-wide labels reach the move part
    outside = int(np.flatnonzero(~mv[sp])[5])
    t1, _ = compute_set_head_loss(net, z, lab)
    with torch.no_grad():
        head.moves.bias[outside] += 100.0
    t2, _ = compute_set_head_loss(net, z, lab)
    assert float(t1) == pytest.approx(float(t2), rel=1e-6)
    _t, st3 = compute_set_head_loss(net, z, lab[..., :3])
    assert "set_move_n" not in st3
    # no labels at all: zero, no NaN
    t0, st0 = compute_set_head_loss(net, z, torch.zeros(2, 6, 8, dtype=torch.long))
    assert float(t0) == 0.0 and int(st0["set_move_n"]) == 0 and int(st0["set_tera_n"]) == 0


def test_attach_set_head_with_the_block():
    from config import RunConfig
    net = om.build_model(RunConfig.default(), matchup=True, movesets=True, set_head=True, set_head_coef=0.1)
    names = [n for n, _p in net.named_parameters()]
    assert names[-8:] == ["set_head.ability.weight", "set_head.ability.bias", "set_head.item.weight", "set_head.item.bias",
                          "set_head.moves.weight", "set_head.moves.bias", "set_head.tera.weight", "set_head.tera.bias"]
    assert {n for n, _p in net.get_parameter_zones()["aux"]} == set(names[-8:])
    assert net.set_head.moves.weight.shape == (len(om.load_vocab_lists()["move.id"]) + 1, net.d_model)
    plain = om.build_model(RunConfig.default(), matchup=True, set_head=True)
    assert not plain.set_head.has_sets and not any("moves" in n for n, _ in plain.named_parameters())


def test_an_update_with_the_block_trains_the_head_and_reports_it(replayed, asm, capsys):
    import torch
    import test_speed_update as tsu
    if not torch.cuda.is_available():
        pytest.skip("flex_attention has no backward on a CPU")
    import ppo_core
    ppo_core._learner_flex_compiled = True
    n = 1024
    obs, cols = _recorded_update_data(replayed, "movesets", n, seed=41)
    labels = _labels_for(obs, asm, n, seed=42, width=8)
    la = tsu._make_learner(sys.modules["learner"], ppo_core,
                           _learner_cfg(sys.modules["config"], "ppo", True, movesets=True, save_every_updates=10**9, set_head_coef=0.1))
    assert la.net.movesets and la.net.set_head.has_sets and getattr(la.target_net, "set_head", None) is None
    before = {k: v.detach().clone() for k, v in la.net.set_head.state_dict().items()}
    la.dataset.add_steps(*cols, extras={"set_labels": labels})
    la.total_steps += n
    asyncio.run(la._perform_update())
    assert la.update_idx == 1
    after = la.net.set_head.state_dict()
    assert all(not torch.equal(before[k], after[k]) for k in ("moves.weight", "tera.weight", "ability.weight"))
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[learner] upd="))
    for key in (" set_ab_acc=", " set_it_acc=", " set_mv_n=", " set_mv_acc=", " set_tera_n=", " set_tera_acc="):
        assert key in line, line
    print("\n" + line)
    del la
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- the wiring


def test_config_and_train_flags():
    import train
    from config import RunConfig
    cfg = train.build_config(train.parse_args(["--obs-matchup", "--obs-movesets", "--set-head-coef", "0.1"]))
    assert cfg.obs.matchup and cfg.obs.movesets and cfg.learner.set_head_coef == 0.1
    assert "obs_movesets=True" in train.features_line(cfg)
    assert cfg.make_assembler().movesets and cfg.make_model().movesets
    assert train.build_config(train.parse_args([])) == RunConfig.default() and RunConfig.default().obs.movesets is False
    assert "obs_movesets" not in train.features_line(RunConfig.default())
    with pytest.raises(SystemExit):
        train.parse_args(["--obs-movesets"])                        # needs --obs-matchup
    with pytest.raises(SystemExit):
        train.parse_args(["--no-obs-matchup", "--obs-movesets"])
    assert cfg.as_dict()["obs"]["movesets"] is True


def test_the_worker_writes_48_labels_with_the_block():
    import worker as new_worker
    from config import RunConfig
    import queue
    from poke_env.player import Player
    cfg = RunConfig.default()
    ms = dataclasses.replace(cfg, obs=dataclasses.replace(cfg.obs, matchup=True, movesets=True),
                             learner=dataclasses.replace(cfg.learner, set_head_coef=0.1))
    orig = Player.__init__
    Player.__init__ = lambda self, *a, **k: None
    try:
        p = new_worker.SyncBridgePlayer(cfg=ms, event_queue=queue.Queue(), action_futures={}, ep_sem=None)
    finally:
        Player.__init__ = orig
    assert p.set_labels_on and p.set_labels_movesets
    b = SimpleNamespace(player_role="p1", force_switch=False, opponent_team={}, battle_tag="b1", opponent_active_pokemon=None)
    assert p._set_labels(b).shape == (48,)
    opp = _pokemon("clodsire", ["earthquake"])
    b.opponent_team = {"p2: Clodsire": opp}
    truth = _pokemon("clodsire", ["earthquake", "toxic", "recover", "gunkshot"], item="leftovers", ability="waterabsorb", tera="Flying")
    p.partner = SimpleNamespace(_battles={"b1": SimpleNamespace(team={"p2: Clodsire": truth})})
    got = p._set_labels(b)
    assert got.shape == (48,) and got.dtype == np.int32 and got[3] < 0 and got[7] == oms.TERA_INDEX["flying"] + 1
    p.set_labels_movesets = False
    assert p._set_labels(b).shape == (18,)


def test_check_restart_refuses_a_movesets_change(tmp_path, monkeypatch):
    import check_restart
    import test_check_restart as tcr
    ck = dict(tcr._ckpt(409_000), obs_settings={"own_tera_type": True, "matchup": True})
    code, lines = tcr._run(tmp_path, monkeypatch, ["--obs-matchup", "--obs-movesets"], ck, allow_obs_change=True)
    assert code == 2 and any("--movesets" in ln and "widen" in ln.lower() for ln in lines), lines
    widened = dict(ck, obs_settings={"own_tera_type": True, "matchup": True, "movesets": True})
    code, lines = tcr._run(tmp_path, monkeypatch, ["--obs-matchup", "--obs-movesets"], widened)
    assert code == 0, lines
    code, lines = tcr._run(tmp_path, monkeypatch, ["--obs-matchup"], widened, allow_obs_change=True)
    assert code == 2, lines
    assert check_restart.checkpoint_obs_settings({"run_cfg": {"obs": {"own_tera_type": True, "matchup": True, "movesets": True}}}) == \
        {"own_tera_type": True, "matchup": True, "movesets": True}
    assert check_restart.checkpoint_obs_settings({"run_cfg": {"obs": {"own_tera_type": True, "matchup": True}}}) == \
        {"own_tera_type": True, "matchup": True}


@pytest.fixture(scope="module")
def ms_ckpt(tmp_path_factory):
    if not MS_CKPT or not os.path.isfile(MS_CKPT):
        pytest.skip("PSPPO_MOVESETS_CKPT names no matchup checkpoint")
    import torch
    import widen_checkpoint as wc
    ck = torch.load(MS_CKPT, map_location="cpu", weights_only=False)
    out = wc.widen(ck, movesets=True, source=MS_CKPT)
    path = str(tmp_path_factory.mktemp("ms") / "learner_update_000001.pt")
    torch.save(out, path)
    return path, out


def test_the_learner_refuses_a_checkpoint_across_the_setting(ms_ckpt):
    import torch
    import test_speed_update as tsu
    if not torch.cuda.is_available():
        pytest.skip("the learner builds on the GPU here")
    path, _out = ms_ckpt
    la = tsu._make_learner(sys.modules["learner"], sys.modules["ppo_core"],
                           _learner_cfg(sys.modules["config"], "imitation", True, save_every_updates=10**9))
    with pytest.raises(RuntimeError, match="movesets"):
        la._load_checkpoint(path)
    lb = tsu._make_learner(sys.modules["learner"], sys.modules["ppo_core"],
                           _learner_cfg(sys.modules["config"], "imitation", True, movesets=True, save_every_updates=10**9))
    lb._load_checkpoint(path)
    assert torch.equal(lb.net.pokemon_net[0].weight.detach().float().cpu(), _out["model"]["pokemon_net.0.weight"].float())
    del la, lb
    torch.cuda.empty_cache()


def test_policy_net_and_evaluate_load_a_movesets_checkpoint(ms_ckpt, replayed):
    import torch
    import evaluate
    import hybrid
    if not torch.cuda.is_available():
        pytest.skip("the CPU path of flex attention does not compile on this box")
    path, out = ms_ckpt
    pol = hybrid.PolicyNet(hybrid.HybridConfig(checkpoint=path, device="cuda"))
    assert pol.assembler.movesets and pol.assembler.matchup and pol.net.movesets
    meta = evaluate.load_checkpoint_meta(path)
    names = [str(n) for n in pol.net.state_dict().keys()]
    learned = [n for n, _ in pol.net.named_parameters()]
    tensor_names = frozenset(k for k in out["model"] if not k.startswith(("set_head.", "opp_action_head.")))
    assert frozenset(meta["tensor_names"]) == tensor_names
    assert evaluate.weights_mismatch(names, tensor_names, learned) is None
    # the player's logits are the learner network's on the same recorded observations
    from config import RunConfig
    net = om.build_model(RunConfig.default(), matchup=True, movesets=True)
    net.load_state_dict({k: v for k, v in out["model"].items() if not k.startswith(("set_head.", "opp_action_head."))}, strict=True)
    net = net.float().to("cuda").eval()
    seq = _obs_seqs(replayed, "movesets", 1)[0][:20]
    for t, o in enumerate(seq):
        got = pol.logits("t", o)
        assert np.isfinite(got).all()
    with torch.no_grad():
        want = net(torch.from_numpy(seq[:1].astype(np.float32)).cuda())[0][0].cpu().numpy()
    pol.forget("t")
    assert np.allclose(pol.logits("t", seq[0]), want, atol=1e-5)
