"""
Round 2's set-up rules of the search player (search_player.MixSettings.setup_switch_trust,
setup_net_trust, setup_blend, setup_keep, guard_status) and their ladder flags.
bench/setup_puzzles/ROUND2_PLAN.md and ROUND2.md on branch setupfix say what they are for.

    python -m pytest tests/test_setup_round2.py -q -p no:cacheprovider

The fuzz test compares mix() with every new setting off against mix() of the commit before them
(478d82a, read with git show): the same action on every random position, with and without a Threat.
"""

from __future__ import annotations

import importlib.util
import os
import random
import subprocess
import sys
import tempfile
from dataclasses import replace

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (REPO, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

import ladder  # noqa: E402
from search_player import MixSettings, Threat, mix, setup_reason  # noqa: E402
from test_search_player import mask_of, spread, view_of  # noqa: E402

BEFORE = "478d82a"          # setupfix-play before round 2: the rules as the ladder plays them today


def old_search_player():
    """search_player.py as it was at BEFORE (the file named by $SEARCH_PLAYER_BEFORE, else git show), as its own module; None without either."""
    given = os.environ.get("SEARCH_PLAYER_BEFORE")
    try:
        if given:
            src = open(given, encoding="utf-8").read()
        else:
            src = subprocess.run(["git", "show", BEFORE + ":search_player.py"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    except Exception:
        return None
    path = os.path.join(tempfile.mkdtemp(), "search_player_before.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("search_player_before", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["search_player_before"] = mod
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
                  fails=tuple(a for a in moves if rng.random() < 0.15), ko=tuple(a for a in moves if rng.random() < 0.3))
    return legal, mask, probs, shares, threat


# =============================================================================================
# Flags off: exactly today's play
# =============================================================================================

def test_with_every_new_rule_off_mix_plays_what_it_played_before():
    old = old_search_player()
    if old is None:
        pytest.skip("git show %s is not available" % BEFORE)
    rng = random.Random(20261008)
    base_settings = [{}, {"guard": True}, {"no_third": True, "tera_own": True, "sure_attack": 0.7},
                     {"setup_defer": True}, {"guard": True, "no_third": True}, {"rule": "pool"}, {"rule": "shortlist"}]
    n = 0
    for _ in range(4000):
        legal, mask, probs, shares, t = random_case(rng)
        view_new = view_of(shares)
        view_old = old.SearchView(view_new.share, view_new.visits, view_new.score, view_new.best, view_new.iterations)
        for kw in base_settings:
            new = mix(probs, mask, view_new, MixSettings(**kw), threat=Threat(**t))
            before = old.mix(probs, mask, view_old, old.MixSettings(**kw), threat=old.Threat(**t))
            assert new == before, (kw, legal, probs, shares, t)
            assert mix(probs, mask, view_new, MixSettings(**kw)) == old.mix(probs, mask, view_old, old.MixSettings(**kw))
            n += 1
    assert n == 4000 * len(base_settings)


def test_the_new_rules_act_only_in_a_set_up_position():
    rng = random.Random(11)
    rules = [{"setup_switch_trust": 0.6}, {"setup_net_trust": 0.5}, {"setup_blend": 2.0}, {"setup_keep": 0.1},
             {"setup_net_trust": 0.5, "guard": True}]
    for _ in range(3000):
        legal, mask, probs, shares, t = random_case(rng)
        threat = Threat(**t)
        view = view_of(shares)
        for kw in rules:
            s = MixSettings(**kw)
            got = mix(probs, mask, view, s, threat=threat)
            assert mask[got] == 1
            if not setup_reason(threat, s):
                assert got == mix(probs, mask, view, MixSettings(guard=s.guard), threat=threat)
            assert mix(probs, mask, view, s) == mix(probs, mask, view, MixSettings(guard=s.guard))     # no threat: no set-up position


# =============================================================================================
# Each rule
# =============================================================================================

SETUP = Threat(boost=2)
QUIET = Threat(boost=0, setup_read=0.1)


def test_switch_trust_keeps_a_sure_switch_the_search_vetoed():
    mask = mask_of(0, 1, 5)
    probs = spread(a0=0.15, a1=0.05, a5=0.80)
    view = view_of(spread(a0=0.80, a1=0.15, a5=0.05))        # the switch is vetoed (0.05 < 0.25 x 0.80)
    assert mix(probs, mask, view, MixSettings(), threat=SETUP) == 0
    assert mix(probs, mask, view, MixSettings(setup_switch_trust=0.8), threat=SETUP) == 5
    assert mix(probs, mask, view, MixSettings(setup_switch_trust=0.9), threat=SETUP) == 0     # below the threshold
    assert mix(probs, mask, view, MixSettings(setup_switch_trust=0.8), threat=QUIET) == 0     # not a set-up position
    probs_move = spread(a0=0.05, a1=0.85, a5=0.10)
    assert mix(probs_move, mask, view, MixSettings(setup_switch_trust=0.6), threat=SETUP) == 0   # a sure MOVE is not a switch: the veto plays


def test_net_trust_plays_the_networks_sure_action_of_any_kind():
    mask = mask_of(0, 1, 5)
    view = view_of(spread(a0=0.80, a1=0.15, a5=0.05))
    probs = spread(a0=0.10, a1=0.75, a5=0.15)                  # the network's move 1, vetoed (0.15 < 0.20)
    assert mix(probs, mask, view, MixSettings(), threat=SETUP) == 0
    assert mix(probs, mask, view, MixSettings(setup_net_trust=0.7), threat=SETUP) == 1
    assert mix(probs, mask, view, MixSettings(setup_net_trust=0.8), threat=SETUP) == 0
    # A move and its Tera form count together; the search decides between the two forms as before.
    mask_t = mask_of(0, 1, 10, 11)
    probs_t = spread(a0=0.1, a1=0.4, a10=0.1, a11=0.4)
    view_t = view_of(spread(a0=0.7, a1=0.05, a10=0.05, a11=0.2))
    assert mix(probs_t, mask_t, view_t, MixSettings(setup_net_trust=0.7), threat=SETUP) in (1, 11)
    assert mix(probs_t, mask_t, view_t, MixSettings(setup_net_trust=0.7), threat=SETUP) == 11      # Tera form: larger share, >= 0.20


def test_blend_weighs_the_network_against_the_search():
    mask = mask_of(0, 1, 2)
    probs = spread(a0=0.6, a1=0.3, a2=0.1)
    view = view_of(spread(a0=0.1, a1=0.3, a2=0.6))
    # log p + lambda * share / largest share: lambda 1 -> 0 (-0.34 against -0.70 and -1.30), lambda 8 -> 2 (5.70 against 2.80 and 0.82).
    import math
    def best(lam):
        return max((0, 1, 2), key=lambda a: math.log(probs[a]) + lam * view.share[a] / 0.6)
    for lam in (1.0, 2.0, 4.0, 8.0):
        assert mix(probs, mask, view, MixSettings(setup_blend=lam), threat=SETUP) == best(lam)
    assert best(1.0) == 0 and best(8.0) == 2
    assert mix(probs, mask, view, MixSettings(setup_blend=8.0), threat=QUIET) == mix(probs, mask, view, MixSettings())


def test_setup_keep_vetoes_less_in_a_set_up_position():
    mask = mask_of(0, 1)
    probs = spread(a0=0.3, a1=0.7)
    view = view_of(spread(a0=0.85, a1=0.15))                  # 0.15 < 0.25 x 0.85 = 0.21: vetoed; 0.15 >= 0.10 x 0.85: kept
    assert mix(probs, mask, view, MixSettings(), threat=SETUP) == 0
    assert mix(probs, mask, view, MixSettings(setup_keep=0.10), threat=SETUP) == 1
    assert mix(probs, mask, view, MixSettings(setup_keep=0.10), threat=QUIET) == 0


def test_guard_sucker_threshold_and_the_combination_with_the_guard():
    mask = mask_of(0, 1, 2)
    probs = spread(a0=0.7, a1=0.2, a2=0.1)                    # 0 is Sucker Punch, 2 our Swords Dance
    view = view_of(spread(a0=0.5, a1=0.3, a2=0.2))
    t = Threat(boost=2, status_read=0.3, fails=(0,), passive=(2,))
    assert mix(probs, mask, view, MixSettings(guard=True), threat=t) == 0                     # 0.30 < 0.40: Sucker Punch stays
    assert mix(probs, mask, view, MixSettings(guard=True, guard_status=0.25), threat=t) == 1  # 0.30 >= 0.25: it goes
    # Net trust with the guard: the network's likeliest of the actions the guard leaves (1 at 0.2: below 0.5, the guard decides).
    assert mix(probs, mask, view, MixSettings(guard=True, guard_status=0.25, setup_net_trust=0.5), threat=t) == 1
    assert mix(probs, mask, view, MixSettings(guard=True, guard_status=0.25, setup_net_trust=0.15), threat=t) == 1
    probs2 = spread(a0=0.05, a1=0.15, a2=0.80)                # the network wants our Swords Dance: the guard drops it
    view2 = view_of(spread(a0=0.2, a1=0.2, a2=0.6))
    got = mix(probs2, mask, view2, MixSettings(guard=True, setup_net_trust=0.5), threat=Threat(boost=2, passive=(2,)))
    assert got in (0, 1) and got == mix(probs2, mask, view2, MixSettings(guard=True), threat=Threat(boost=2, passive=(2,)))


def test_ladder_flags_for_the_new_rules():
    args = ladder.argparse.Namespace(mix_rule="veto", guard=False, veto_fix=False, sure_attack=0.7, setup_defer=False,
                                     setup_switch_sure=0.0, think_longer=0, setup_switch_trust=0.0, setup_net_trust=0.0,
                                     setup_blend=0.0, setup_keep=0.0, guard_sucker=0.0)
    assert ladder.mix_settings(args) == MixSettings() and ladder.setup_fix_text(args) == ""
    args.setup_net_trust, args.guard, args.guard_sucker = 0.7, True, 0.25
    s = ladder.mix_settings(args)
    assert s == MixSettings(guard=True, setup_net_trust=0.7, guard_status=0.25) and s.setup_rules and not s.fixes
    assert "0.70 is played" in ladder.setup_fix_text(args) and "0.25" in ladder.setup_fix_text(args)
    assert replace(s, setup_net_trust=0.0).setup_rules is False
    assert MixSettings(setup_blend=2.0, rule="pool").setup_rules is False                     # the veto rule only
