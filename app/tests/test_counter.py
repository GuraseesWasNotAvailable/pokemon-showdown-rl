"""
Round 3's counter rules (branch counterfix): "use the counter" (search_player.MixSettings.setup_counter,
ladder.py --setup-counter) and "don't hit the boosted defence" (setup_other_side, --setup-other-side),
the arm "mix-counter" with both, what they know of the board (setup_counter.py), and the one-line
login failure of ladder.py (log_in_checked, watch_messages; more in tests/test_ladder_login.py).

    python -m pytest tests/test_counter.py -q -p no:cacheprovider

The fuzz test compares mix() with both rules off against mix() of the commit before them (a3b1cec,
read with git show, or the file named by $SEARCH_PLAYER_A3B1CEC where there is no git): the same
action on every random position, with a random Board given and with none. The counter and damage
tests read poke-env's move data; the player tests replay the recorded battles of
tests/fixtures/foulplay_bridge_trace.jsonl.gz through the real player (tests/test_search_player.py).
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import random
import subprocess
import sys
import tempfile
from dataclasses import replace
from types import SimpleNamespace

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (REPO, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

import ladder  # noqa: E402
import search_player as sp  # noqa: E402
import setup_counter as sc  # noqa: E402
from search_player import MixSettings, Threat, counter_reason, mix  # noqa: E402
from setup_counter import Board, Fighter, Foe, Scene, counter_check, damage_share  # noqa: E402
from test_search_player import (FakeBridge, chosen, classes, load_fixture, mask_of, needs_player,  # noqa: E402,F401
                                orders_of, play, searcher, spread, view_of)

BEFORE = "a3b1cec"          # setupfix-play: the play of the ladder today, before the counter rules

try:
    from poke_env.data import GenData

    HAVE_DATA = bool(GenData.from_gen(9).moves.get("roar"))
except Exception:  # pragma: no cover
    HAVE_DATA = False
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="needs poke-env's move data")

COUNTER = MixSettings(setup_counter=True)
OTHER = MixSettings(setup_other_side=True)
BOTH = MixSettings(setup_counter=True, setup_other_side=True)
COUNTER_SW = replace(COUNTER, counter_switch=True)          # d573f7e's switches to a benched holder (off since the review)
BOTH_SW = replace(BOTH, counter_switch=True)


def old_search_player():
    """search_player.py as it was at BEFORE ($SEARCH_PLAYER_A3B1CEC, else git show), as its own module; None without either."""
    given = os.environ.get("SEARCH_PLAYER_A3B1CEC")
    try:
        if given:
            src = open(given, encoding="utf-8").read()
        else:
            src = subprocess.run(["git", "show", BEFORE + ":search_player.py"], cwd=REPO, capture_output=True, text=True,
                                 check=True).stdout
    except Exception:
        return None
    path = os.path.join(tempfile.mkdtemp(), "search_player_a3b1cec.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("search_player_a3b1cec", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["search_player_a3b1cec"] = mod
    spec.loader.exec_module(mod)
    return mod


def random_case(rng):
    legal = sorted(rng.sample(range(14), rng.randint(2, 9)))
    mask = mask_of(*legal)
    raw = [rng.random() ** 3 if a in legal else 0.0 for a in range(14)]
    probs = [x / sum(raw) for x in raw]
    shares = [0.001 + rng.random() ** 2 if a in legal and rng.random() < 0.8 else 0.0 for a in range(14)]
    if not any(shares):
        shares[legal[0]] = 1.0
    moves = [a for a in legal if a < 4 or a >= 10]
    threat = dict(boost=rng.randint(0, 3), last_setup=rng.random() < 0.3, setup_read=rng.random(), sleep_read=rng.random() * 0.6,
                  status_read=rng.random(), passive=tuple(a for a in moves if rng.random() < 0.3),
                  fails=tuple(a for a in moves if rng.random() < 0.15), ko=tuple(a for a in moves if rng.random() < 0.2))
    return legal, mask, probs, shares, threat


def random_board(rng, legal):
    """A Board with counters, holders, defences and damage on random actions (some of them not legal)."""
    boosts = {k: rng.choice([-1, 0, 0, 1, 2, 3]) for k in ("atk", "def", "spa", "spd", "spe", "evasion")}
    attacks = [a for a in range(14) if a < 4 or a >= 10]
    return Board(forced=rng.random() < 0.1, boosts=boosts,
                 top_boost=max([v for v in boosts.values() if v > 0] or [0]),
                 counters={a: "haze haze" for a in range(4) if rng.random() < 0.3},
                 holders={a: (round(rng.uniform(0.5, 1.0), 3), "Haze") for a in range(4, 10) if rng.random() < 0.3},
                 unaware=rng.random() < 0.1,
                 hits={a: rng.choice(["def", "spd", ""]) for a in attacks if rng.random() < 0.8},
                 damage={a: round(rng.random(), 3) for a in attacks if rng.random() < 0.8},
                 no_swap={a: "knocks itself out" for a in attacks if rng.random() < 0.1})


# =============================================================================================
# Flags off: exactly today's play
# =============================================================================================

def test_with_both_rules_off_mix_plays_what_it_played_at_a3b1cec():
    old = old_search_player()
    if old is None:
        pytest.skip("git show %s is not available and $SEARCH_PLAYER_A3B1CEC is not set" % BEFORE)
    rng = random.Random(20261008)
    base_settings = [{}, {"guard": True}, {"no_third": True, "tera_own": True, "sure_attack": 0.7}, {"setup_defer": True},
                     {"guard": True, "no_third": True}, {"rule": "pool"}, {"rule": "shortlist"}, {"setup_blend": 8.0},
                     {"setup_net_trust": 0.5, "guard": True}, {"setup_keep": 0.1}, {"setup_switch_trust": 0.6}]
    positions = 0
    for _ in range(4500):
        legal, mask, probs, shares, t = random_case(rng)
        board = random_board(rng, legal)
        view_new = view_of(shares)
        view_old = old.SearchView(view_new.share, view_new.visits, view_new.score, view_new.best, view_new.iterations)
        for kw in base_settings:
            before = old.mix(probs, mask, view_old, old.MixSettings(**kw), threat=old.Threat(**t))
            assert mix(probs, mask, view_new, MixSettings(**kw), threat=Threat(**t)) == before
            assert mix(probs, mask, view_new, MixSettings(**kw), threat=Threat(**t), board=board) == before, (kw, legal, board)
            assert mix(probs, mask, view_new, MixSettings(**kw)) == old.mix(probs, mask, view_old, old.MixSettings(**kw))
            # The rules on but no board (it could not be built): today's play too.
            assert mix(probs, mask, view_new, replace(MixSettings(**kw), setup_counter=True, setup_other_side=True),
                       threat=Threat(**t)) == before
        positions += 1
    assert positions == 4500


def test_with_the_rules_on_every_action_is_legal_and_every_change_has_its_reason():
    rng = random.Random(3)
    changed = {"counter": 0, "switch": 0, "other": 0}
    for _ in range(4000):
        legal, mask, probs, shares, t = random_case(rng)
        board = random_board(rng, legal)
        view = view_of(shares)
        moves = [a for a in legal if a < 4 or a >= 10]
        threat = Threat(**t, counter_ko=tuple(a for a in moves if rng.random() < 0.15))
        for s in (COUNTER, OTHER, BOTH, COUNTER_SW, BOTH_SW, replace(BOTH, guard=True), replace(BOTH_SW, rule="pool")):
            base = mix(probs, mask, view, replace(s, setup_counter=False, setup_other_side=False), threat=threat)
            explain = {}
            got = mix(probs, mask, view, s, threat=threat, explain=explain, board=board)
            assert mask[got] == 1
            if board.forced:
                assert got == base and "board" not in explain
                continue
            if got == base:
                continue
            note = explain["board"]
            counter = note.get("counter") or {}
            if counter.get("decided") == "played the counter":
                assert s.setup_counter and counter_reason(threat, board, s) and got in board.counters and got <= 3
                assert not any(a in legal for a in threat.counter_ko) and not board.unaware
                changed["counter"] += 1
            elif counter.get("decided") == "switched to a counter":
                assert s.setup_counter and s.counter_switch and got in board.holders and not any(a in board.counters for a in legal if a <= 3)
                assert board.holders[got][0] == max(board.holders[a][0] for a in board.holders if a in legal)
                changed["switch"] += 1
            else:
                swap = note["other_side"]
                assert s.setup_other_side and swap["mix"] == base and swap["pick"] == got
                side = board.hits[base]
                other = "spd" if side == "def" else "def"
                assert board.boosts[side] >= 1 and board.boosts[side] > board.boosts[other]
                assert board.hits[got] != side and board.damage[got] >= board.damage[base] and (got >= 10) == (base >= 10)
                assert got not in board.no_swap and sp.base_action(base) not in board.counters
                assert got in threat.counter_ko or base not in threat.counter_ko
                changed["other"] += 1
    assert all(n > 50 for n in changed.values()), changed


# =============================================================================================
# "Use the counter" in mix()
# =============================================================================================

BOOSTED = Threat(boost=2)
MASK = mask_of(0, 1, 2, 3, 5, 6)
PROBS = spread(a0=0.55, a1=0.05, a2=0.25, a3=0.05, a5=0.05, a6=0.05)        # the network: attack 0
VIEW = view_of(spread(a0=0.5, a1=0.1, a2=0.2, a3=0.1, a5=0.05, a6=0.05))    # the search: attack 0 too


def board(**kw):
    kw.setdefault("boosts", {"spa": 2})
    kw.setdefault("top_boost", max([v for v in kw["boosts"].values() if v > 0] or [0]))
    return Board(**kw)


def test_the_active_plays_its_counter_in_a_set_up_position():
    b = board(counters={1: "haze haze"})
    assert mix(PROBS, MASK, VIEW, MixSettings(), threat=BOOSTED, board=b) == 0
    explain = {}
    assert mix(PROBS, MASK, VIEW, COUNTER, threat=BOOSTED, board=b, explain=explain) == 1
    assert explain["board"]["counter"] == {"trigger": "boosted +2", "mix": 0, "decided": "played the counter",
                                           "counter": "haze haze", "pick": 1}
    # Two counters: the network's likeliest of them.
    assert mix(PROBS, MASK, VIEW, COUNTER, threat=BOOSTED, board=board(counters={1: "haze haze", 3: "phaze roar"})) == 1
    probs = spread(a0=0.55, a1=0.05, a2=0.25, a3=0.10, a5=0.05)
    assert mix(probs, MASK, VIEW, COUNTER, threat=BOOSTED, board=board(counters={1: "haze haze", 3: "phaze roar"})) == 3


@pytest.mark.parametrize("threat,top,fires", [
    (Threat(boost=2), 0, True),                       # +2
    (Threat(boost=1, last_setup=True), 0, True),      # +1 right after a set-up move
    (Threat(boost=1), 0, False),                      # +1 alone
    (Threat(setup_read=0.55), 0, True),               # the set-up read at 50% or more
    (Threat(setup_read=0.45), 0, False),              # 45%: the guard's 40% would fire, this rule's 50% does not
    (Threat(), 2, True),                              # +2 evasion (Acupressure, Double Team): the board's top boost
    (None, 2, True),
    (None, 1, False),
])
def test_the_trigger(threat, top, fires):
    b = Board(counters={1: "taunt taunt"}, top_boost=top, boosts={"evasion": top})
    assert bool(counter_reason(threat, b, COUNTER)) is fires
    assert (mix(PROBS, MASK, VIEW, COUNTER, threat=threat, board=b) == 1) is fires


def test_an_attack_in_ko_range_keeps_mixs_choice():
    b = board(counters={1: "haze haze"})
    explain = {}
    assert mix(PROBS, MASK, VIEW, COUNTER, threat=Threat(boost=2, counter_ko=(2,)), board=b, explain=explain) == 0
    assert explain["board"]["counter"]["decided"].startswith("kept: an attack") and explain["board"]["counter"]["ko"] == [2]
    # A KO estimate on an action that is not legal does not count.
    assert mix(PROBS, MASK, VIEW, COUNTER, threat=Threat(boost=2, counter_ko=(12,)), board=b) == 1
    # The rule reads its own KO list (counter_ko, which counts a fresh sweeper too), not sure_attack's.
    assert mix(PROBS, MASK, VIEW, COUNTER, threat=Threat(boost=2, ko=(2,)), board=b) == 1


def test_mixs_own_counter_unaware_and_no_counter_keep_the_choice():
    explain = {}
    assert mix(PROBS, MASK, VIEW, COUNTER, threat=BOOSTED, board=board(counters={0: "phaze dragontail"}), explain=explain) == 0
    assert explain["board"]["counter"]["decided"] == "kept: mix plays a counter"
    # Its Tera form counts as the counter too.
    mask = mask_of(0, 1, 10, 11)
    probs = spread(a0=0.1, a10=0.6, a1=0.2, a11=0.1)
    view = view_of(spread(a0=0.1, a10=0.6, a1=0.2, a11=0.1))
    assert mix(probs, mask, view, COUNTER, threat=BOOSTED, board=board(counters={0: "phaze dragontail"})) == 10
    explain = {}
    assert mix(PROBS, MASK, VIEW, COUNTER, threat=BOOSTED, board=board(counters={1: "haze haze"}, unaware=True), explain=explain) == 0
    assert explain["board"]["counter"]["decided"] == "kept: our active Pokemon has Unaware"
    explain = {}
    assert mix(PROBS, MASK, VIEW, COUNTER, threat=BOOSTED, board=board(), explain=explain) == 0
    assert explain["board"]["counter"]["decided"] == "kept: no counter can work"


def test_no_switch_to_a_holder_unless_the_switches_are_on():
    b = board(holders={5: (0.6, "Haze"), 6: (0.9, "Unaware")})
    explain = {}
    # Since the review: the switches are off (counter_switch), mix's choice stays.
    assert mix(PROBS, MASK, VIEW, COUNTER, threat=BOOSTED, board=b, explain=explain) == 0
    assert explain["board"]["counter"]["decided"] == "kept: no counter on the active (switches are off)"
    assert mix(PROBS, MASK, VIEW, BOTH, threat=BOOSTED, board=b) == 0
    explain = {}
    assert mix(PROBS, MASK, VIEW, COUNTER_SW, threat=BOOSTED, board=b, explain=explain) == 6
    assert explain["board"]["counter"]["decided"] == "switched to a counter" and explain["board"]["counter"]["holder"] == "Unaware"
    # Mix switching to a holder already: kept.
    probs = spread(a0=0.1, a5=0.8, a6=0.1)
    view = view_of(spread(a0=0.3, a5=0.5, a6=0.2))
    assert mix(probs, MASK, view, COUNTER_SW, threat=BOOSTED, board=b) == 5
    # The active's counter comes first.
    assert mix(PROBS, MASK, VIEW, COUNTER_SW, threat=BOOSTED, board=board(counters={3: "sleep spore"}, holders={6: (1.0, "Haze")})) == 3
    # A holder whose switch is not legal (trapped) is not used.
    assert mix(PROBS, mask_of(0, 1, 2, 3), VIEW, COUNTER_SW, threat=BOOSTED, board=b) == 0


def test_nothing_in_a_forced_switch_and_nothing_illegal():
    mask = mask_of(4, 5, 6)
    probs = spread(a4=0.7, a5=0.2, a6=0.1)
    view = view_of(spread(a4=0.6, a5=0.3, a6=0.1))
    forced = board(forced=True, holders={6: (1.0, "Haze")})
    explain = {}
    assert mix(probs, mask, view, BOTH, threat=BOOSTED, board=forced, explain=explain) == 4 and "board" not in explain
    # Not forced, the same switches: the holder only with the switches on.
    assert mix(probs, mask, view, BOTH, threat=BOOSTED, board=replace(forced, forced=False)) == 4
    assert mix(probs, mask, view, BOTH_SW, threat=BOOSTED, board=replace(forced, forced=False)) == 6
    # Counters and holders on actions that are not legal are never played.
    assert mix(PROBS, mask_of(0, 2), VIEW, COUNTER, threat=BOOSTED, board=board(counters={1: "haze haze"}, holders={7: (1.0, "Haze")})) == 0


def test_the_counter_comes_after_the_guard_and_the_fixes():
    b = board(counters={1: "haze haze"})
    for s in (replace(COUNTER, guard=True), replace(COUNTER, no_third=True, tera_own=True, sure_attack=0.7), replace(COUNTER, setup_blend=8.0)):
        assert mix(PROBS, MASK, VIEW, s, threat=BOOSTED, board=b) == 1


# =============================================================================================
# "Don't hit the boosted defence" in mix()
# =============================================================================================

def other_board(boosts, damage, hits=None, **kw):
    hits = hits or {0: "def", 1: "spd", 2: "def", 3: "", 10: "def", 11: "spd"}
    return Board(boosts=boosts, top_boost=max([v for v in boosts.values() if v > 0] or [0]), hits=hits, damage=damage, **kw)


def test_an_attack_into_the_boosted_defence_gives_way_to_one_that_does_as_much():
    mask = mask_of(0, 1, 2, 4)
    probs = spread(a0=0.7, a1=0.1, a2=0.1, a4=0.1)
    view = view_of(spread(a0=0.6, a1=0.2, a2=0.1, a4=0.1))
    explain = {}
    b = other_board({"def": 2}, {0: 0.50, 1: 0.50, 2: 0.45})
    assert mix(probs, mask, view, OTHER, board=b, explain=explain) == 1                   # as much: the other side
    assert explain["board"]["other_side"] == {"boosted": "def +2", "mix": 0, "pick": 1, "damage": [0.5, 0.5], "why": ""}
    explain = {}
    assert mix(probs, mask, view, OTHER, board=other_board({"def": 2}, {0: 0.50, 1: 0.30}), explain=explain) == 0
    assert explain["board"]["other_side_kept"]["why"] == "the best of the other kind does 60% of it"
    assert mix(probs, mask, view, OTHER, board=other_board({"def": 2}, {0: 0.50, 1: 0.499, 2: 0.45})) == 0   # less: it stays
    assert mix(probs, mask, view, replace(OTHER, other_side_ratio=0.8), board=other_board({"def": 2}, {0: 0.50, 1: 0.40})) == 1
    assert mix(probs, mask, view, MixSettings(), board=b) == 0                            # off
    # Not boosted more than the other defence, or not at all: no change.
    assert mix(probs, mask, view, OTHER, board=other_board({"def": 1, "spd": 1}, {0: 0.5, 1: 0.5})) == 0
    assert mix(probs, mask, view, OTHER, board=other_board({"def": 0, "spd": -1}, {0: 0.5, 1: 0.5})) == 0
    assert mix(probs, mask, view, OTHER, board=other_board({"def": 1}, {0: 0.5, 1: 0.55})) == 1          # +1 is enough
    # Special into boosted Special Defence: the physical one.
    probs_s = spread(a0=0.1, a1=0.7, a2=0.1, a4=0.1)
    assert mix(probs_s, mask, view, OTHER, board=other_board({"spd": 2}, {0: 0.36, 1: 0.35, 2: 0.29})) == 0
    # The strongest of the other kind; a move that hits neither defence (fixed damage) counts as the other kind.
    assert mix(probs, mask_of(0, 1, 3), view, OTHER, board=other_board({"def": 2}, {0: 0.5, 1: 0.5, 3: 0.55})) == 3


def test_the_other_side_rule_counts_accuracy_and_keeps_ko_and_priority_moves():
    mask = mask_of(0, 1, 4)
    probs = spread(a0=0.7, a1=0.2, a4=0.1)
    view = view_of(spread(a0=0.6, a1=0.3, a4=0.1))
    # Boomburst 53% for Gunk Shot 50% at 80% (the review's game 278462): the expected damage decides.
    b = other_board({"def": 2}, {0: 0.50, 1: 0.60}, accuracy={1: 0.8})
    assert mix(probs, mask, view, OTHER, board=b) == 0                                     # 0.48 expected < 0.50
    b = other_board({"def": 2}, {0: 0.50, 1: 0.45}, accuracy={0: 0.8, 1: 1.0})
    assert mix(probs, mask, view, OTHER, board=b) == 1                                     # 0.45 >= 0.40 expected
    # Mix's attack has priority: it stays.
    explain = {}
    assert mix(probs, mask, view, OTHER, board=other_board({"def": 2}, {0: 0.5, 1: 0.9}, priority={0: 1}), explain=explain) == 0
    assert explain["board"]["other_side_kept"]["why"] == "mix's attack has priority"
    # Mix's attack in KO range where the other is not: it stays; both in range: the other side.
    explain = {}
    b = other_board({"def": 2}, {0: 0.5, 1: 0.9})
    assert mix(probs, mask, view, OTHER, threat=Threat(counter_ko=(0,)), board=b, explain=explain) == 0
    assert explain["board"]["other_side_kept"]["why"] == "mix's attack is in KO range and the other kind is not"
    assert mix(probs, mask, view, OTHER, threat=Threat(counter_ko=(0, 1)), board=b) == 1
    assert mix(probs, mask, view, OTHER, threat=Threat(counter_ko=(1,)), board=b) == 1


def test_the_other_side_rule_keeps_the_tera_form_and_never_touches_status_moves_or_a_forced_switch():
    mask = mask_of(0, 1, 10, 11)
    probs = spread(a0=0.05, a1=0.05, a10=0.8, a11=0.1)
    view = view_of(spread(a0=0.1, a1=0.1, a10=0.6, a11=0.2))
    b = other_board({"def": 2}, {0: 0.3, 1: 0.9, 10: 0.5, 11: 0.55})
    assert mix(probs, mask, view, OTHER, board=b) == 11                     # Tera Body Slam -> Tera Psychic, not the plain one
    b2 = other_board({"def": 2}, {0: 0.3, 1: 0.9, 10: 0.5, 11: 0.3})
    assert mix(probs, mask, view, OTHER, board=b2) == 10                    # its Tera form too weak; the plain form is not taken
    # A status move played (not in hits): nothing.
    mask_st = mask_of(0, 1, 2)
    probs_st = spread(a0=0.1, a1=0.1, a2=0.8)
    view_st = view_of(spread(a0=0.2, a1=0.2, a2=0.6))
    assert mix(probs_st, mask_st, view_st, OTHER, board=other_board({"def": 2}, {0: 0.5, 1: 0.5}, hits={0: "def", 1: "spd"})) == 2
    # A forced switch: nothing.
    assert mix(spread(a4=0.9, a5=0.1), mask_of(4, 5), view_of(spread(a4=0.9, a5=0.1)), OTHER,
               board=other_board({"def": 2}, {}, forced=True)) == 4
    # No damage estimate for the played move or for the other: nothing.
    assert mix(spread(a0=0.8, a1=0.2), mask_of(0, 1), view_of(spread(a0=0.8, a1=0.2)), OTHER,
               board=other_board({"def": 2}, {1: 0.5})) == 0


def test_the_other_side_rule_never_swaps_in_explosion_and_the_like():
    mask = mask_of(0, 1, 2, 3)
    probs = spread(a0=0.7, a1=0.1, a2=0.1, a3=0.1)
    view = view_of(spread(a0=0.6, a1=0.2, a2=0.1, a3=0.1))
    hits = {0: "spd", 1: "def", 2: "def", 3: "def"}
    # Psychic into +1 Special Defence; Explosion would do 89%: not it, the next best physical attack at 0.8 or more.
    b = other_board({"spd": 1}, {0: 0.33, 1: 0.89, 2: 0.34, 3: 0.27}, hits=hits, no_swap={1: "knocks itself out"})
    assert mix(probs, mask, view, OTHER, board=b) == 2
    b2 = other_board({"spd": 1}, {0: 0.33, 1: 0.89, 2: 0.20}, hits=hits, no_swap={1: "knocks itself out"})
    assert mix(probs, mask, view, OTHER, board=b2) == 0


@needs_data
@pytest.mark.parametrize("move,why", [("explosion", "knocks itself out"), ("mistyexplosion", "knocks itself out"),
                                      ("hyperbeam", "must recharge"), ("gigaimpact", "must recharge"), ("outrage", "locks it in"),
                                      ("solarbeam", "charges a turn"), ("meteorbeam", "charges a turn"), ("uturn", "switches out"),
                                      ("flipturn", "switches out"), ("voltswitch", "switches out"), ("suckerpunch", "works only at times"),
                                      ("fakeout", "works only at times"), ("focuspunch", "works only at times"),
                                      ("earthquake", ""), ("bodypress", ""), ("closecombat", ""), ("dracometeor", "")])
def test_what_is_never_swapped_in(move, why):
    assert sc.swap_doubt(move) == why


def test_the_other_side_rule_does_not_undo_a_counter():
    # The counter played is Dragon Tail into +2 Defence: it stays (it is there to phaze, not to hit).
    mask = mask_of(0, 1)
    probs = spread(a0=0.2, a1=0.8)
    view = view_of(spread(a0=0.3, a1=0.7))
    b = other_board({"def": 2}, {0: 0.1, 1: 0.5}, hits={0: "def", 1: "spd"}, counters={0: "phaze dragontail"})
    explain = {}
    assert mix(probs, mask, view, BOTH, threat=BOOSTED, board=b, explain=explain) == 0
    assert "other_side" not in explain["board"]
    # Mix plays the Dragon Tail itself (the review's game 278391: swapped for Shadow Ball six turns running): it stays,
    # with the counter rule on or off.
    probs_dt = spread(a0=0.8, a1=0.2)
    view_dt = view_of(spread(a0=0.7, a1=0.3))
    for s in (BOTH, OTHER):
        explain = {}
        assert mix(probs_dt, mask, view_dt, s, threat=BOOSTED, board=b, explain=explain) == 0
        assert "other_side" not in explain.get("board", {})


# =============================================================================================
# Which counters work (poke-env's move data)
# =============================================================================================

FOE = Foe(species="garchomp", types=("DRAGON", "GROUND"), boosts={"atk": 2}, last_setup=True, others=3)
SCENE = Scene()


@needs_data
@pytest.mark.parametrize("move,kind", [("roar", "phaze"), ("whirlwind", "phaze"), ("dragontail", "phaze"), ("circlethrow", "phaze"),
                                       ("haze", "haze"), ("clearsmog", "haze"), ("encore", "encore"), ("yawn", "sleep"),
                                       ("spore", "sleep"), ("sleeppowder", "sleep"), ("lovelykiss", "sleep"), ("perishsong", "perish")])
def test_every_counter_kind_works_in_the_plain_case(move, kind):
    assert counter_check(move, FOE, SCENE) == (kind, "")
    assert counter_check(move, FOE, SCENE, now=False) == (kind, "")
    assert sc.counter_kind(move) == kind


@needs_data
@pytest.mark.parametrize("move", ["hypnosis", "sing", "grasswhistle", "darkvoid"])
def test_a_sleep_move_below_75_percent_is_no_counter(move):
    assert sc.counter_kind(move) == "sleep"
    assert "of the time" in counter_check(move, FOE, SCENE)[1] and "of the time" in counter_check(move, FOE, SCENE, now=False)[1]


@needs_data
def test_encore_only_when_we_move_first_or_it_sets_up_again():
    assert counter_check("encore", FOE, SCENE, outspeeds=True)[1] == ""
    assert counter_check("encore", FOE, SCENE, outspeeds=False)[1].startswith("it moves first")
    assert counter_check("encore", FOE, SCENE, outspeeds=False, read=0.6)[1] == ""        # it sets up again: locked in that
    assert counter_check("encore", FOE, SCENE, outspeeds=False, now=False)[1] == ""      # a benched holder: no timing


@needs_data
def test_the_reviews_smaller_gaps():
    assert counter_check("whirlwind", replace(FOE, ability="windrider"), SCENE)[1] == "Wind Rider"
    assert counter_check("roar", replace(FOE, ability="windrider"), SCENE)[1] == ""
    dark = replace(FOE, types=("DARK",))
    assert counter_check("encore", dark, SCENE, user_ability="prankster")[1].startswith("Dark: a Prankster")
    assert counter_check("haze", dark, SCENE, user_ability="prankster")[1] == ""          # Haze hits the field
    assert counter_check("encore", dark, SCENE)[1] == ""
    assert counter_check("spore", FOE, Scene(safeguard=True))[1] == "Safeguard"
    assert counter_check("spore", replace(FOE, ability="leafguard"), Scene(weather="sun"))[1] == "Leaf Guard in the sun"
    assert counter_check("yawn", replace(FOE, ability="leafguard"), SCENE)[1] == ""
    assert counter_check("yawn", replace(FOE, types=("GRASS",), ability="flowerveil"), SCENE)[1] == "Flower Veil"
    assert sc.hits_defence("shellsidearm") == "" and sc.swap_doubt("shellsidearm") == "its category depends on the position"
    assert sc.swap_doubt("terablast") == "its category depends on the position"


@needs_data
def test_taunt_only_before_the_boost_lands():
    quiet = replace(FOE, boosts={}, last_setup=False)
    assert counter_check("taunt", quiet, SCENE, read=0.6) == ("taunt", "")
    assert counter_check("taunt", quiet, SCENE, read=0.5) == ("taunt", "")
    assert counter_check("taunt", quiet, SCENE, read=0.49)[1].startswith("no set-up read")
    assert counter_check("taunt", FOE, SCENE, read=0.9)[1] == "the boost has landed already"
    assert counter_check("taunt", quiet, SCENE, read=0.9, now=False)[1] == "too late after a switch"
    assert counter_check("taunt", replace(quiet, effects=("taunt",)), SCENE, read=0.9)[1] == "taunted already"
    assert counter_check("taunt", replace(quiet, ability="oblivious"), SCENE, read=0.9)[1] == "Oblivious"
    assert counter_check("taunt", replace(quiet, ability="aromaveil"), SCENE, read=0.9)[1] == "Aroma Veil"
    assert counter_check("earthquake", FOE, SCENE) == ("", "not a counter")


@needs_data
def test_the_timing_of_the_active_and_of_a_benched_pokemon():
    quiet = replace(FOE, boosts={"atk": 0, "spe": -1}, last_setup=False)
    for move in ("haze", "roar", "dragontail", "clearsmog", "perishsong"):
        assert counter_check(move, quiet, SCENE)[1] == "nothing boosted to take away yet"
        assert counter_check(move, quiet, SCENE, now=False)[1] == ""                     # the boost will be there by then
    assert counter_check("haze", replace(quiet, boosts={"evasion": 2}), SCENE)[1] == ""    # evasion is a boost to take away
    assert counter_check("encore", quiet, SCENE)[1] == "its last action was not a set-up move"
    assert counter_check("encore", quiet, SCENE, now=False)[1] == ""


@needs_data
def test_phazing_failures_and_immunities():
    assert counter_check("dragontail", replace(FOE, types=("FAIRY",)), SCENE)[1].startswith("Fairy")
    assert counter_check("roar", replace(FOE, types=("FAIRY",)), SCENE)[1] == ""
    assert counter_check("circlethrow", replace(FOE, types=("GHOST", "DRAGON")), SCENE)[1].startswith("Ghost")
    for move in ("roar", "whirlwind", "dragontail", "circlethrow"):
        assert counter_check(move, replace(FOE, ability="guarddog"), SCENE)[1] == "Guard Dog"
        assert counter_check(move, replace(FOE, ability="suctioncups"), SCENE)[1] == "Suction Cups"
        assert counter_check(move, replace(FOE, effects=("ingrain",)), SCENE)[1] == "Ingrain"
        assert counter_check(move, replace(FOE, others=0), SCENE)[1] == "nothing left to drag in"
    # Substitute stops Dragon Tail and Circle Throw, not Roar (sound) or Whirlwind (it goes through).
    sub = replace(FOE, effects=("substitute",))
    assert counter_check("dragontail", sub, SCENE)[1] == "Substitute" and counter_check("circlethrow", sub, SCENE)[1] == "Substitute"
    assert counter_check("roar", sub, SCENE)[1] == "" and counter_check("whirlwind", sub, SCENE)[1] == ""
    assert counter_check("dragontail", sub, SCENE, user_ability="infiltrator")[1] == ""
    # Status moves aimed at it: Good as Gold and Magic Bounce; Soundproof for Roar only.
    assert counter_check("roar", replace(FOE, ability="goodasgold"), SCENE)[1] == "Good as Gold"
    assert counter_check("dragontail", replace(FOE, ability="goodasgold"), SCENE)[1] == ""
    assert counter_check("whirlwind", replace(FOE, ability="magicbounce"), SCENE)[1] == "Magic Bounce"
    assert counter_check("roar", replace(FOE, ability="soundproof"), SCENE)[1] == "Soundproof"
    assert counter_check("whirlwind", replace(FOE, ability="soundproof"), SCENE)[1] == ""
    # An ability not known: it works.
    assert counter_check("roar", replace(FOE, ability=""), SCENE)[1] == ""


@needs_data
def test_haze_clear_smog_encore_and_perish_song_failures():
    assert counter_check("clearsmog", replace(FOE, types=("STEEL", "FLYING")), SCENE)[1].startswith("Steel")
    assert counter_check("haze", replace(FOE, types=("STEEL",)), SCENE)[1] == ""
    sub = replace(FOE, effects=("substitute",))
    assert counter_check("haze", sub, SCENE)[1] == "" and counter_check("clearsmog", sub, SCENE)[1] == "Substitute"
    assert counter_check("haze", replace(FOE, ability="goodasgold"), SCENE)[1] == ""                # Haze hits the field
    assert counter_check("encore", replace(FOE, effects=("encore",)), SCENE)[1] == "encored already"
    assert counter_check("encore", replace(FOE, ability="aromaveil"), SCENE)[1] == "Aroma Veil"
    assert counter_check("encore", replace(FOE, ability="goodasgold"), SCENE)[1] == "Good as Gold"
    assert counter_check("encore", replace(FOE, ability="magicbounce"), SCENE)[1] == "Magic Bounce"
    assert counter_check("encore", sub, SCENE)[1] == ""                                              # Encore goes through it
    assert counter_check("perishsong", replace(FOE, effects=("perish",)), SCENE)[1] == "its count runs already"
    assert counter_check("perishsong", replace(FOE, ability="soundproof"), SCENE)[1] == "Soundproof"
    assert counter_check("perishsong", sub, SCENE)[1] == ""


@needs_data
def test_sleep_failures_and_immunities():
    assert counter_check("spore", replace(FOE, status="par"), SCENE)[1].startswith("it has a status")
    assert counter_check("yawn", replace(FOE, effects=("yawn",)), SCENE)[1] == "drowsy already"
    assert counter_check("spore", FOE, Scene(sleep_used=True))[1] == "Sleep Clause"
    assert counter_check("spore", FOE, Scene(terrain="electric"))[1] == "Electric Terrain"
    assert counter_check("yawn", FOE, Scene(terrain="misty"))[1] == "Misty Terrain"
    assert counter_check("spore", replace(FOE, grounded=False), Scene(terrain="electric"))[1] == ""     # it floats
    for ability, why in (("insomnia", "Insomnia"), ("vitalspirit", "Vital Spirit"), ("sweetveil", "Sweet Veil"),
                         ("comatose", "Comatose"), ("purifyingsalt", "Purifying Salt")):
        assert counter_check("spore", replace(FOE, ability=ability), SCENE)[1] == why
    grass = replace(FOE, types=("GRASS", "POISON"))
    assert counter_check("spore", grass, SCENE)[1].startswith("Grass") and counter_check("sleeppowder", grass, SCENE)[1].startswith("Grass")
    assert counter_check("lovelykiss", grass, SCENE)[1] == "" and counter_check("yawn", grass, SCENE)[1] == ""
    assert counter_check("spore", replace(FOE, ability="overcoat"), SCENE)[1] == "Overcoat"
    assert counter_check("spore", replace(FOE, item="safetygoggles"), SCENE)[1] == "Safety Goggles"
    assert counter_check("lovelykiss", replace(FOE, item="safetygoggles"), SCENE)[1] == ""             # not a powder
    sub = replace(FOE, effects=("substitute",))
    assert counter_check("spore", sub, SCENE)[1] == "Substitute" and counter_check("yawn", sub, SCENE)[1] == "Substitute"
    assert counter_check("sing", sub, SCENE)[1].endswith("of the time")                                # Sing: 55%
    assert counter_check("spore", replace(FOE, ability="goodasgold"), SCENE)[1] == "Good as Gold"
    assert counter_check("spore", replace(FOE, ability="magicbounce"), SCENE)[1] == "Magic Bounce"
    assert counter_check("perishsong", replace(FOE, ability="soundproof"), SCENE)[1] == "Soundproof"


# =============================================================================================
# Damage, the defence a move hits, two hits
# =============================================================================================

def fighter(**kw):
    base = dict(level=80, stats={"hp": 300, "atk": 200, "def": 200, "spa": 200, "spd": 200, "spe": 200}, types=("NORMAL",))
    base.update(kw)
    return Fighter(**base)


@needs_data
def test_the_defence_each_move_hits():
    assert sc.hits_defence("earthquake") == "def" and sc.hits_defence("thunderbolt") == "spd"
    assert sc.hits_defence("psyshock") == "def" and sc.hits_defence("secretsword") == "def"
    assert sc.hits_defence("seismictoss") == "" and sc.hits_defence("sacredsword") == "" and sc.hits_defence("swordsdance") == ""


@needs_data
def test_damage_share_reads_boosts_stab_types_items_and_abilities():
    att, dfn = fighter(types=("GROUND",)), fighter(types=("STEEL",))
    plain = damage_share("earthquake", att, dfn)
    expected = (int(int(int(2 * 80 / 5 + 2) * 100 * 200 / 200) / 50) + 2) * 1.5 * 2.0 * 0.925 / 300
    assert plain == pytest.approx(expected)
    assert damage_share("earthquake", att, replace(dfn, boosts={"def": 2})) < 0.55 * plain          # +2 Defence
    assert damage_share("earthquake", replace(att, ability="unaware"), replace(dfn, boosts={"def": 2})) == pytest.approx(plain)
    assert damage_share("sacredsword", att, replace(dfn, boosts={"def": 2})) == pytest.approx(damage_share("sacredsword", att, dfn))
    assert damage_share("psyshock", att, replace(dfn, boosts={"def": 2})) < damage_share("psyshock", att, dfn)
    assert damage_share("psyshock", att, replace(dfn, boosts={"spd": 2})) == pytest.approx(damage_share("psyshock", att, dfn))
    assert damage_share("earthquake", att, replace(dfn, ability="levitate")) == 0.0
    assert damage_share("earthquake", att, replace(dfn, types=("FLYING",))) == 0.0
    assert damage_share("earthquake", replace(att, item="choiceband"), dfn) > 1.4 * plain
    assert damage_share("thunderbolt", att, replace(dfn, item="assaultvest")) < damage_share("thunderbolt", att, dfn)
    assert damage_share("earthquake", att, replace(dfn, ability="furcoat")) < 0.6 * plain
    assert damage_share("earthquake", replace(att, status="brn"), dfn) < 0.6 * plain
    # Tera: STAB on the Tera type with this move.
    att_t = replace(att, tera="ELECTRIC")
    assert damage_share("thunderbolt", att_t, dfn, tera=True) == pytest.approx(1.5 * damage_share("thunderbolt", att_t, dfn), rel=0.02)
    assert damage_share("swordsdance", att, dfn) is None
    assert damage_share("seismictoss", att, dfn) == pytest.approx(80 / 300)
    assert damage_share("bulletseed", att, fighter(types=("WATER",))) > 2.5 * damage_share(
        {"category": "Physical", "basePower": 25, "type": "Grass", "flags": {}}, att, fighter(types=("WATER",)))


@needs_data
def test_two_hit_ko_with_the_boosts_and_without_them_for_unaware():
    threat = fighter(types=("DRAGON",), boosts={"atk": 2})
    ours = fighter(types=("WATER",), hp=1.0)
    ko, best = sc.two_hit_ko(threat, ["outrage"], ours)
    assert ko and best > 0.5
    ko_u, best_u = sc.two_hit_ko(threat, ["outrage"], replace(ours, ability="unaware"))
    assert best_u == pytest.approx(best / 2.0, rel=0.02) and not ko_u
    assert not sc.two_hit_ko(threat, ["outrage"], replace(ours, types=("FAIRY",)))[0]           # immune
    # No move known: a STAB 90 hit of each of its types.
    ko_none, best_none = sc.two_hit_ko(threat, [], ours)
    assert best_none > 0 and sc.two_hit_ko(threat, ["swordsdance"], ours)[1] == pytest.approx(best_none)


# =============================================================================================
# The board from a battle
# =============================================================================================

class _T:
    def __init__(self, name):
        self.name = name


def poke(species, types, ability=None, moves=(), hp=1.0, base=None, stats=None, boosts=None, status=None, effects=(), item="unknown_item",
         possible=(), level=80, fainted=False, max_hp=300):
    return SimpleNamespace(species=species, types=[_T(t) for t in types], _type_1=_T(types[0]), _type_2=_T(types[1]) if len(types) > 1 else None,
                           ability=ability, possible_abilities=list(possible), moves={m: SimpleNamespace(id=m) for m in moves},
                           current_hp_fraction=hp, current_hp=int(hp * max_hp), max_hp=max_hp, level=level,
                           base_stats=base or {"hp": 100, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100},
                           stats=stats or {"atk": 200, "def": 200, "spa": 200, "spd": 200, "spe": 200}, boosts=boosts or {},
                           status=_T(status) if status else None, effects={_T(e): 1 for e in effects}, item=item,
                           tera_type=None, is_terastallized=False, fainted=fainted)


def battle_of(ours, theirs, bench=(), their_team=None, forced=False, fields=None):
    team = their_team if their_team is not None else {"p2: " + theirs.species: theirs}
    return SimpleNamespace(force_switch=forced, active_pokemon=ours, opponent_active_pokemon=theirs, opponent_team=team,
                           is_grounded=lambda mon: True, fields=fields or {}, weather={}, _team_size={"p2": 6}, player_role="p1",
                           opponent_role="p2")


@needs_data
def test_build_board_from_a_battle():
    theirs = poke("raikou", ["ELECTRIC"], boosts={"spa": 2, "spd": 2}, possible=("pressure", "innerfocus"), moves=("calmmind", "thunderbolt"))
    ours = poke("milotic", ["WATER"], ability="competitive", moves=("dragontail", "scald", "recover", "icebeam"), max_hp=600)
    altaria = poke("altaria", ["DRAGON", "FLYING"], ability="naturalcure", moves=("haze", "roost"), hp=0.88, max_hp=340,
                   stats={"atk": 200, "def": 250, "spa": 200, "spd": 300, "spe": 200})
    weak = poke("quagsire", ["WATER", "GROUND"], ability="unaware", moves=("earthquake",), hp=0.40)
    nothing = poke("garchomp", ["DRAGON", "GROUND"], ability="roughskin", moves=("earthquake",))
    frail = poke("politoed", ["WATER"], ability="drizzle", moves=("haze", "scald"), max_hp=200,
                 stats={"atk": 150, "def": 150, "spa": 150, "spd": 100, "spe": 150})       # +2 Thunderbolt: about 380% of its HP
    actions = {0: (ours.moves["dragontail"], {}), 1: (ours.moves["scald"], {}), 2: (ours.moves["recover"], {}),
               3: (ours.moves["icebeam"], {}), 10: (ours.moves["dragontail"], {"terastallize": True}),
               4: (altaria, {}), 5: (weak, {}), 6: (nothing, {}), 7: (frail, {})}
    b = sc.build_board(battle_of(ours, theirs), actions, threat=Threat(boost=2), advice=None)
    assert b.counters == {0: "phaze dragontail"} and not b.failed
    assert b.top_boost == 2 and b.boosts == {"spa": 2, "spd": 2} and not b.forced and not b.unaware
    assert set(b.holders) == {4} and b.holders[4] == (0.88, "Haze")
    assert "40% HP" in b.passed[5] and 6 not in b.passed and 6 not in b.holders
    assert "two hits" in b.passed[7] and 7 not in b.holders
    assert b.hits == {0: "def", 1: "spd", 3: "spd", 10: "def"} and set(b.damage) == {0, 1, 3, 10} and b.no_swap == {}
    assert b.names[4] == "switch Altaria" and b.names[10].endswith("+ Tera")
    assert b.outspeeds is False and 0.5 < b.threat_hit < 1.0 and b.accuracy[1] == 1.0 and b.priority == {0: -6, 10: -6}
    # Ours at half HP: +2 Thunderbolt knocks it out before Dragon Tail (priority -6) moves.
    low = poke("milotic", ["WATER"], ability="competitive", moves=("dragontail", "scald", "recover", "icebeam"), max_hp=600, hp=0.5)
    acts_low = {0: (low.moves["dragontail"], {}), 1: (low.moves["scald"], {})}
    b_low = sc.build_board(battle_of(low, theirs), acts_low, threat=Threat(boost=2))
    assert b_low.counters == {} and "moves last" in b_low.failed[0]
    # Asleep: no counter.
    asleep_ours = poke("milotic", ["WATER"], ability="competitive", moves=("dragontail",), max_hp=600, status="SLP")
    b_slp = sc.build_board(battle_of(asleep_ours, theirs), {0: (asleep_ours.moves["dragontail"], {})}, threat=Threat(boost=2))
    assert b_slp.counters == {} and b_slp.failed[0].endswith("our Pokemon is asleep")
    # Haze that would clear more of ours than of theirs.
    hazer = poke("altaria", ["DRAGON", "FLYING"], moves=("haze",), max_hp=600, boosts={"atk": 2, "spe": 2, "def": 1})
    b_hz = sc.build_board(battle_of(hazer, theirs), {0: (hazer.moves["haze"], {})}, threat=Threat(boost=2))
    assert b_hz.counters == {} and "more of our boosts" in b_hz.failed[0]
    # Their Fairy type: Dragon Tail does nothing, so it is a failed counter.
    fairy = poke("clefable", ["FAIRY"], boosts={"spa": 2}, ability="magicguard", moves=("calmmind",))
    b2 = sc.build_board(battle_of(ours, fairy), actions, threat=Threat(boost=2))
    assert b2.counters == {} and b2.failed[0].startswith("Dragon Tail: Fairy")
    # The last Pokemon of its team: nothing to drag in.
    dead = {"p2: x%d" % i: poke("x%d" % i, ["NORMAL"], fainted=True) for i in range(5)}
    dead["p2: raikou"] = theirs
    b3 = sc.build_board(battle_of(ours, theirs, their_team=dead), actions, threat=Threat(boost=2))
    assert b3.failed[0].endswith("nothing left to drag in")
    # One of its Pokemon asleep already: Sleep Clause; Electric Terrain on the ground.
    spore = poke("amoonguss", ["GRASS", "POISON"], ability="regenerator", moves=("spore",))
    acts = {0: (spore.moves["spore"], {})}
    asleep = {"p2: raikou": theirs, "p2: snorlax": poke("snorlax", ["NORMAL"], status="SLP")}
    assert "Sleep Clause" in sc.build_board(battle_of(spore, theirs, their_team=asleep), acts).failed[0]
    assert "Electric Terrain" in sc.build_board(battle_of(spore, theirs, fields={_T("ELECTRIC_TERRAIN"): 1}), acts).failed[0]
    # A forced switch says so; our Unaware active says so.
    assert sc.build_board(battle_of(ours, theirs, forced=True), {4: (altaria, {})}).forced
    assert sc.build_board(battle_of(weak, theirs), {0: (weak.moves["earthquake"], {})}).unaware


@needs_data
def test_build_board_reads_the_searchs_sampled_sets_when_they_agree():
    theirs = poke("gholdengo", ["STEEL", "GHOST"], boosts={"spa": 2}, possible=("goodasgold",))
    ours = poke("skarmory", ["STEEL", "FLYING"], ability="sturdy", moves=("whirlwind",))
    acts = {0: (ours.moves["whirlwind"], {})}
    assert sc.build_board(battle_of(ours, theirs), acts).failed[0].endswith("Good as Gold")      # its only ability
    two = poke("hatterene", ["PSYCHIC", "FAIRY"], boosts={"spa": 2}, possible=("healer", "magicbounce"))
    advice = SimpleNamespace(opponent_worlds=[{"active": {"ability": "magicbounce"}}, {"active": {"ability": "Magic Bounce"}}])
    assert sc.build_board(battle_of(ours, two), acts, advice=advice).failed[0].endswith("Magic Bounce")
    split = SimpleNamespace(opponent_worlds=[{"active": {"ability": "magicbounce"}}, {"active": {"ability": "healer"}}])
    assert sc.build_board(battle_of(ours, two), acts, advice=split).counters == {0: "phaze whirlwind"}   # not known: it works
    one = SimpleNamespace(opponent_worlds=[{"active": {"ability": "magicbounce"}}])
    assert sc.build_board(battle_of(ours, two), acts, advice=one).counters == {0: "phaze whirlwind"}     # one world is not enough


# =============================================================================================
# In play: the arm, the record, the counts
# =============================================================================================

def test_the_arm_and_the_flags():
    assert "mix-counter" in sp.MODES and "mix-counter" in ladder.PLAYER_MODES
    assert sp.split_mode("mix-counter") == ("mix", "veto")
    s = MixSettings(**sp.ARM_FIXES["mix-counter"][0])
    assert s.setup_counter and s.setup_other_side and s.board_rules and not s.fixes and not s.setup_rules
    assert not sp.ARM_FIXES["mix-counter"][1]
    assert not MixSettings().board_rules
    assert MixSettings(setup_counter=True).without_fixes().board_rules is False
    args = ladder.argparse.Namespace(mix_rule="veto", guard=False, veto_fix=False, sure_attack=0.7, setup_defer=False,
                                     setup_switch_sure=0.0, think_longer=0, setup_switch_trust=0.0, setup_net_trust=0.0,
                                     setup_blend=0.0, setup_keep=0.0, guard_sucker=0.0, setup_counter=False, setup_other_side=False)
    assert ladder.mix_settings(args) == MixSettings() and ladder.setup_fix_text(args) == ""
    args.setup_counter = True
    assert ladder.mix_settings(args) == MixSettings(setup_counter=True) and "use the counter" in ladder.setup_fix_text(args)
    args.setup_other_side = True
    assert ladder.mix_settings(args) == BOTH and "boosted defence" in ladder.setup_fix_text(args)


def test_the_flags_parse(tmp_path, monkeypatch):
    for name in ("OUT", "GRAPH", "BATTLES"):
        monkeypatch.setattr(ladder, name, getattr(ladder, name))     # main() sets them from --out: put back after
    out = str(tmp_path / "results.jsonl")
    assert ladder.main(["--history", "--server", "local:1", "--out", out, "--player", "mix", "--setup-counter", "--setup-other-side"]) == 0
    assert ladder.main(["--history", "--server", "local:1", "--out", out, "--arms", "mix,mix-counter"]) == 0


def test_the_end_of_run_summary_counts_each_rule_per_arm():
    rows = [{"arm": "mix", "won": True}, {"arm": "mix", "won": False},
            {"arm": "mix-counter", "won": True, "counter_looked": 3, "counter_played": 1, "counter_switched": 1, "counter_ko_kept": 1,
             "counter_had": 0, "other_side_looked": 4, "other_side": 2},
            {"arm": "mix-counter", "won": False, "counter_looked": 2, "counter_played": 1, "counter_switched": 0, "counter_ko_kept": 0,
             "counter_had": 1, "other_side_looked": 1, "other_side": 0}]
    text = ladder.arms_text(rows)
    lines = text.splitlines()
    assert "use the counter" not in lines[0]
    assert "use the counter: looked 5, played 2, switched 1, kept for KO 1, mix had one 1; boosted defence: hit 5, swapped 2" in lines[1]
    assert sp.counter_text({"counter_looked": 5, "other_side": 2}).startswith("use the counter looked at 5 set-up positions")
    assert sp._counter_counts({"counter": {"decided": "played the counter"}, "other_side": None}) == {
        "counter_looked": 1, "counter_played": 1, "counter_switched": 0, "counter_ko_kept": 0, "counter_had": 0, "other_side_looked": 0,
        "other_side": 0}
    assert sp._counter_counts({"other_side_kept": {"why": "no attack of the other kind"}})["other_side_looked"] == 1


@needs_player
def test_the_arm_in_play_records_every_change_and_counts_it(classes, monkeypatch):
    battles = load_fixture()
    bridge = FakeBridge(battles)
    import setup_counter

    real = setup_counter.build_board
    made = []

    def rigged(battle, actions, **kw):
        # The real board, with a counter put on the first legal move of every third decision.
        b = real(battle, actions, **kw)
        made.append(b)
        moves = sorted(a for a in actions if a <= 3)
        if moves and len(made) % 3 == 0:
            b = replace(b, counters={moves[-1]: "haze haze"}, boosts=dict(b.boosts, atk=2), top_boost=2)
        return b

    monkeypatch.setattr(setup_counter, "build_board", rigged)
    played = changed = 0
    try:
        for events in battles:
            room = events[0]["room"]
            player = searcher(classes, events[-1]["username"], bridge, mode="mix-counter")
            plain = searcher(classes, events[-1]["username"], FakeBridge([events]), mode="mix")
            play(player, events)
            play(plain, events)
            decisions = player.decisions_of(room)
            assert len(decisions) == len(orders_of(events)) == len(chosen(player))
            assert player.search_totals["counter_errors"] == 0
            for d, p in zip(decisions, plain.decisions_of(room)):
                assert d["mask"][d["played"]] == 1
                if "counter" not in d:
                    if not d["forced"]:
                        assert d["played"] == p["played"] or d["search"] is None
                    continue
                c = d["counter"]
                assert c["net"] == d["net"] and c["search"] == d["search"] and c["pick"] == d["played"]
                assert c["mix"] == p["played"] and set(c["text"]) == {"net", "search", "mix", "pick"} and c["why"]
                played += int(c.get("counter", {}).get("decided") == "played the counter")
                changed += int(c["changed"])
            summary = player.game_summary(room)
            for name in sp.COUNTER_COUNTS:
                assert name in summary
            assert summary["counter_played"] + summary["counter_switched"] + summary["other_side"] == sum(
                1 for d in decisions if d.get("counter", {}).get("changed"))
            assert "use the counter looked at" in player.search_summary() or not summary["counter_looked"]
        assert played > 0 and changed >= played
    finally:
        bridge.close()


@needs_player
def test_a_board_that_cannot_be_built_costs_nothing(classes, monkeypatch):
    battles = load_fixture()
    events = min(battles, key=lambda e: len(orders_of(e)))
    room = events[0]["room"]
    import setup_counter

    def broken(*a, **kw):
        raise RuntimeError("no board")

    monkeypatch.setattr(setup_counter, "build_board", broken)
    bridge = FakeBridge([events])
    try:
        player = searcher(classes, events[-1]["username"], bridge, mode="mix", mix=MixSettings(min_iterations=0, setup_counter=True,
                                                                                               setup_other_side=True))
        plain = searcher(classes, events[-1]["username"], FakeBridge([events]), mode="mix")
        play(player, events)
        play(plain, events)
        assert [d["played"] for d in player.decisions_of(room)] == [d["played"] for d in plain.decisions_of(room)]
        assert player.search_totals["counter_errors"] == sum(1 for d in player.decisions_of(room) if d["search"] is not None)
        assert "raised: mix played" in player.search_summary()
    finally:
        bridge.close()


# =============================================================================================
# The login: one line for a failed login request
# =============================================================================================

class FakeWebsocket:
    def __init__(self):
        self.sent = []

    async def send(self, text):
        self.sent.append(text)


class FakeClient:
    """What ladder.log_in_checked reads of poke-env's PSClient: the account, the login server's address, the websocket."""

    def __init__(self, password=None, username="counterbot"):
        self.account_configuration = SimpleNamespace(username=username, password=password)
        self.server_configuration = SimpleNamespace(authentication_url="http://127.0.0.1:9/action.php?")
        self.username = username
        self.websocket = FakeWebsocket()
        self.logged_in = asyncio.Event()
        self.passed = []
        self.logger = logging.getLogger("test_counter.fake_client")

    async def _handle_message(self, message):
        self.passed.append(message)


def run_like_poke_env(client, messages):
    """Each message handled in a task nobody waits for, as poke-env's listen() does; the loop's unhandled errors."""
    errors = []
    loop = asyncio.new_event_loop()
    loop.set_exception_handler(lambda _loop, ctx: errors.append(ctx))

    async def listen():
        tasks = [asyncio.ensure_future(client._handle_message(m)) for m in messages]
        await asyncio.wait(tasks)
        return tasks

    try:
        tasks = loop.run_until_complete(listen())
    finally:
        loop.close()
    import gc

    gc.collect()
    return tasks, errors


def failures():
    import requests

    return [requests.exceptions.ReadTimeout("HTTPSConnectionPool(host='play.pokemonshowdown.com', port=443): Read timed out. "
                                            "(read timeout=10.0)"),
            requests.exceptions.ConnectionError("Connection aborted."), OSError("Network is unreachable"),
            RuntimeError("Expected bot to be logged in.")]


@pytest.mark.parametrize("exc", failures(), ids=lambda e: type(e).__name__)
def test_a_failed_login_request_is_one_line_and_no_task_exception(exc, monkeypatch, capsys):
    import requests

    asked = []

    def post(url, data=None, timeout=None):
        asked.append((url, sorted(data)))
        raise exc

    monkeypatch.setattr(requests, "post", post)
    client = FakeClient(password="not-a-real-password")
    player = SimpleNamespace()
    ladder.watch_messages(client, player)
    tasks, errors = run_like_poke_env(client, ["|challstr|4|abcdef"])
    assert all(t.exception() is None for t in tasks) and errors == []
    out = capsys.readouterr()
    lines = [line for line in (out.out + out.err).splitlines() if line.strip()]
    assert len(lines) == 1 and "login request failed: %s" % type(exc).__name__ in lines[0] and "Traceback" not in out.err
    assert asked == [("http://127.0.0.1:9/action.php?", ["act", "challstr", "name", "pass"])] and client.passed == []
    assert client.websocket.sent == [] and "not-a-real-password" not in out.out + out.err      # nothing sent to Showdown
    assert ladder.login_state(client).failed and not ladder.login_state(client).refused


def test_the_login_and_the_other_messages_pass_as_before(capsys):
    client = FakeClient()                     # no password (a local server): no request, the name alone
    player = SimpleNamespace()
    ladder.watch_messages(client, player)
    search = '|updatesearch|{"searching":["gen9randombattle"],"games":{"battle-gen9randombattle-1":"[Gen 9]"}}'
    tasks, errors = run_like_poke_env(client, ["|challstr|4|abc", search, ">battle-gen9randombattle-1\n|turn|3"])
    assert errors == [] and all(t.exception() is None for t in tasks)
    assert client.websocket.sent == ["|/trn counterbot,0,"]
    assert client.passed == [search, ">battle-gen9randombattle-1\n|turn|3"]
    assert player.server_games == {"battle-gen9randombattle-1"} and player.server_searching == ["gen9randombattle"]
    out = capsys.readouterr().out
    assert out.count("\n") == 1 and "name counterbot sent to Showdown" in out
    tasks, errors = run_like_poke_env(client, ["|updateuser| counterbot|1|1|{}"])
    assert errors == [] and client.logged_in.is_set() and client.passed[-1] == "|updateuser| counterbot|1|1|{}"
    assert "logged in as counterbot" in capsys.readouterr().out


def test_the_real_poke_env_login_that_times_out_is_one_line(monkeypatch, capsys):
    try:
        import requests
        from poke_env import AccountConfiguration, ServerConfiguration
        from poke_env.ps_client import ps_client as psc
    except Exception:  # pragma: no cover
        pytest.skip("needs poke-env and requests")
    client = psc.PSClient(AccountConfiguration("counterbot", "not-a-real-password"), log_level=40, start_listening=False,
                          server_configuration=ServerConfiguration("ws://127.0.0.1:9/showdown/websocket", "http://127.0.0.1:9/action.php?"))
    asked = []

    def post(url, data=None, timeout=None):
        asked.append(url)
        raise requests.exceptions.ReadTimeout("Read timed out. (read timeout=10.0)")

    monkeypatch.setattr(psc.requests, "post", post)
    ladder.watch_messages(client, SimpleNamespace())
    tasks, errors = run_like_poke_env(client, ["|challstr|4|abc"])
    assert asked and errors == [] and all(t.exception() is None for t in tasks)
    out = capsys.readouterr()
    assert "Traceback" not in out.err + out.out and (out.out + out.err).count("login request failed: ReadTimeout") == 1
    assert "not-a-real-password" not in out.out + out.err
    assert not client.logged_in.is_set()


# =============================================================================================
# After round 3's review: the KO exception at full HP, and an A/B's balance on the ladder
# =============================================================================================

def test_full_hp_guard_rules_out_sturdy_sash_and_multiscale_only_when_known():
    plain = poke("raikou", ["ELECTRIC"], possible=("pressure", "innerfocus"))
    assert sc.full_hp_guard(plain) == "focussash?"                                   # item unknown, no sampled sets
    worlds = [{"active": {"ability": "pressure", "item": "leftovers"}}, {"active": {"ability": "pressure", "item": "lifeorb"}}]
    assert sc.full_hp_guard(plain, worlds) == ""                                     # two sets, no Sash: ruled out
    assert sc.full_hp_guard(plain, worlds[:1]) == "focussash?"                       # one set is not enough
    assert sc.full_hp_guard(plain, worlds + [{"active": {"ability": "pressure", "item": "focussash"}}]) == "focussash?"
    assert sc.full_hp_guard(poke("raikou", ["ELECTRIC"], possible=("pressure",), item="leftovers")) == ""
    assert sc.full_hp_guard(poke("raikou", ["ELECTRIC"], possible=("pressure",), item="focussash")) == "focussash"
    assert sc.full_hp_guard(poke("raikou", ["ELECTRIC"], possible=("pressure",), item="")) == ""     # no item (knocked off)
    assert sc.full_hp_guard(poke("dragonite", ["DRAGON", "FLYING"], possible=("innerfocus", "multiscale"), item="leftovers"),
                            worlds) == ""                                            # the sets say Pressure... and so not Multiscale
    assert sc.full_hp_guard(poke("dragonite", ["DRAGON", "FLYING"], possible=("innerfocus", "multiscale"), item="leftovers")) == "multiscale"
    assert sc.full_hp_guard(poke("magnezone", ["ELECTRIC", "STEEL"], ability="sturdy", item="leftovers")) == "sturdy"


def test_ko_doubt_without_the_full_hp_doubt():
    move = SimpleNamespace(category=_T("PHYSICAL"), type=_T("GROUND"))
    full = poke("raikou", ["ELECTRIC"], possible=("pressure",), item="leftovers")
    assert sp.ko_doubt(move, full) == "full HP" and sp.ko_doubt(move, full, full_hp=False) == ""
    floats = poke("rotom", ["ELECTRIC", "GHOST"], ability="levitate", item="leftovers")
    assert sp.ko_doubt(move, floats, full_hp=False) == "levitate"


def test_an_ab_counts_only_its_own_games_for_the_balance():
    ck = "learner_update_011850.pt"
    games = [{"arm": "mix", "checkpoint": ck} for _ in range(220)]                   # the ordinary runs before the A/B
    games += [{"arm": "mix", "checkpoint": ck, "experiment": "mix,mix-counter"}] * 2
    games += [{"arm": "mix-counter", "checkpoint": ck, "experiment": "mix,mix-counter"}]
    games += [{"arm": "mix-counter", "checkpoint": "other.pt", "experiment": "mix,mix-counter"}]
    assert ladder.arm_counts(games, ck) == {"mix": 222, "mix-counter": 1}
    assert ladder.arm_counts(games, ck, experiment="mix,mix-counter") == {"mix": 2, "mix-counter": 1}
    picker = sp.ArmPicker(["mix", "mix-counter"], counts=ladder.arm_counts(games, ck, experiment="mix,mix-counter"))
    assert picker("battle-1") == "mix-counter" and picker("battle-2") in ("mix", "mix-counter")
