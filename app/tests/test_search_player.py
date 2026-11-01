"""
The search player (search_player.py), the arms in ladder.py and the arms in ladder_graph.py.

    python -m pytest tests/test_search_player.py -q -p no:cacheprovider

No Foul Play, no Showdown server and no checkpoint are needed. The player tests replay the seven
recorded battles of tests/fixtures/foulplay_bridge_trace.jsonl.gz through the real player class
with real poke-env battle objects. The search is a fake bridge that serves, for every request,
the answer the real helper gave when the battle was recorded (or no answer, a late one, an
error, a move that is not legal). The network is a stub that gives the same logits for the same
observation, so that two players that see the same battle can be compared move for move.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import gzip
import importlib.util
import json
import os
import queue
import random
import subprocess
import sys
import threading
import time
import zlib

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import foulplay_bridge as fb  # noqa: E402
import ladder  # noqa: E402
import ladder_graph  # noqa: E402
import search_player as sp  # noqa: E402
from search_player import ArmPicker, MixSettings, SearchView, mix, read_advice  # noqa: E402

FIXTURE = os.path.join(REPO, "tests", "fixtures", "foulplay_bridge_trace.jsonl.gz")

try:
    import numpy as np
    import poke_env  # noqa: F401
    import torch  # noqa: F401

    HAVE_PLAYER = os.path.isfile(FIXTURE)
except Exception:  # pragma: no cover
    HAVE_PLAYER = False

needs_player = pytest.mark.skipif(not HAVE_PLAYER, reason="needs poke-env, torch, numpy and the recorded battles")


# =============================================================================================
# The combination, as a function
# =============================================================================================

def mask_of(*legal):
    return [1.0 if a in legal else 0.0 for a in range(14)]


def view_of(shares, visits=None, iterations=100_000):
    shares = [float(x) for x in shares]
    visits = visits or [int(round(x * iterations)) for x in shares]
    legal = [a for a in range(14) if visits[a] > 0]
    best = max(legal, key=lambda a: (shares[a], visits[a], -a))
    return SearchView(shares, visits, [0.5 if v else None for v in visits], best, iterations)


def spread(**kw):
    """14 numbers from a0=..., a4=... (the rest 0)."""
    out = [0.0] * 14
    for key, value in kw.items():
        out[int(key[1:])] = value
    return out


def test_modes_and_rules():
    assert sp.MODES == (("net", "search", "mix", "mix-veto", "mix-pool", "mix-shortlist", "predict", "mix-guard") + tuple(sp.ARM_FIXES)
                        + (sp.DEEP_ARM, sp.LUNA_ARM, sp.SOL_ARM) + sp.LINES_ARMS + sp.SOLO_ARMS)
    assert sp.split_mode("mix-guard") == ("mix", "veto")              # the guard is a change of the veto rule
    assert ladder.PLAYER_MODES == sp.MODES            # ladder.py lists them without importing the player
    assert len(sp.RULES) <= 3                          # at most three variants of the combination
    assert sp.split_mode("net") == ("net", None) and sp.split_mode("search") == ("search", None)
    assert sp.split_mode("mix") == ("mix", None) and sp.split_mode("mix-pool") == ("mix", "pool")
    with pytest.raises(ValueError):
        sp.split_mode("mix-magic")
    with pytest.raises(ValueError):
        sp.check_modes([])
    with pytest.raises(ValueError):
        MixSettings(rule="magic")
    assert sp.check_modes([" net", "mix "]) == ["net", "mix"]


def test_legal_probabilities_are_the_softmax_over_the_legal_actions():
    logits = [0.0] * 14
    logits[2], logits[5], logits[9] = 2.0, 1.0, 50.0     # 9 is not legal: it must not count
    p = sp.legal_probabilities(logits, mask_of(0, 2, 5))
    assert abs(sum(p) - 1.0) < 1e-9 and p[9] == 0.0 and p[1] == 0.0
    assert p[2] > p[5] > p[0] > 0
    assert abs(p[2] / p[5] - 2.718281828) < 1e-6
    assert sp.legal_probabilities(logits, [0.0] * 14) == [0.0] * 14


def test_merged_prior_adds_a_move_and_its_tera_form():
    probs = spread(a0=0.1, a10=0.5, a1=0.2, a4=0.2)
    q = sp.merged_prior(probs, mask_of(0, 1, 4, 10))          # Tera is offered with move 0 only
    assert q[0] == q[10] == pytest.approx(0.6)
    assert q[1] == pytest.approx(0.2) and q[11] == 0.0 and q[4] == pytest.approx(0.2)
    assert sp.base_action(12) == 2 and sp.base_action(7) == 7 and sp.base_action(3) == 3


def test_veto_keeps_the_networks_move_unless_the_search_turned_away_from_it():
    mask = mask_of(0, 1, 2, 4)
    probs = spread(a0=0.6, a1=0.3, a2=0.05, a4=0.05)
    settings = MixSettings(rule="veto", keep=0.25)
    # The search prefers move 1, but looked at move 0 a third as often: the network's move stands.
    assert mix(probs, mask, view_of(spread(a0=0.2, a1=0.6, a2=0.1, a4=0.1)), settings) == 0
    # The search hardly looked at move 0 (a tenth of its favourite): the network's next choice that is sound.
    assert mix(probs, mask, view_of(spread(a0=0.06, a1=0.6, a2=0.04, a4=0.3)), settings) == 1
    # Neither of the network's two likeliest is sound: the likeliest of the sound ones, and on a tie
    # between those the one the search looked at more.
    assert mix(spread(a0=0.6, a1=0.3, a2=0.06, a4=0.04), mask, view_of(spread(a0=0.02, a1=0.03, a2=0.45, a4=0.5)), settings) == 2
    assert mix(probs, mask, view_of(spread(a0=0.02, a1=0.03, a2=0.45, a4=0.5)), settings) == 4
    # keep 0 is the network alone, keep 1 the search alone.
    assert mix(probs, mask, view_of(spread(a0=0.01, a1=0.6, a2=0.1, a4=0.29)), MixSettings(rule="veto", keep=0.0)) == 0
    assert mix(probs, mask, view_of(spread(a0=0.2, a1=0.3, a2=0.1, a4=0.4)), MixSettings(rule="veto", keep=1.0)) == 4


def test_trust_lets_the_search_choose_only_among_moves_the_network_finds_plausible():
    # mix-trust (reports/2026-10-15/LOSS_STUDY.md): the veto works among the moves the network gives 3% or more.
    mask = mask_of(0, 1, 2, 4)
    probs = spread(a0=0.80, a1=0.18, a2=0.01, a4=0.01)
    plain = MixSettings(rule="veto", keep=0.25)
    trust = MixSettings(rule="veto", keep=0.25, trust=0.03)
    view = view_of(spread(a0=0.05, a1=0.05, a2=0.1, a4=0.8))
    assert mix(probs, mask, view, plain) == 4            # the plain veto plays the search's move the network gives 1%
    assert mix(probs, mask, view, trust) == 0            # among 0 and 1 neither is vetoed: the network's move stands
    # The search can still turn the network from its first choice, to a plausible one.
    assert mix(probs, mask, view_of(spread(a0=0.02, a1=0.6, a2=0.08, a4=0.3)), trust) == 1
    assert sp.ARM_FIXES["mix-trust"][0] == {"trust": 0.03}
    assert sp.split_mode("mix-trust")[0] == "mix"


def test_trust_never_plays_a_move_the_network_all_but_rules_out():
    import random
    rng = random.Random(7)
    s = MixSettings(rule="veto", keep=0.25, trust=0.03)
    for _ in range(2000):
        legal = sorted(rng.sample(range(10), rng.randint(2, 8)))
        mask = mask_of(*legal)
        raw = [rng.random() ** 3 if a in legal else 0.0 for a in range(14)]
        probs = [x / sum(raw) for x in raw]
        sh = [rng.random() ** 2 if a in legal and rng.random() > 0.1 else 0.0 for a in range(14)]
        if sum(sh) == 0:
            continue
        shares = [x / sum(sh) for x in sh]
        choice = mix(probs, mask, view_of(shares), s)
        q = sp.merged_prior(probs, mask)
        top = max(legal, key=lambda a: (q[a], -a))
        assert choice in legal and (q[choice] >= 0.03 or choice == top)


def test_the_trust_flag_gives_the_players_every_mode_the_guarded_search():
    # ladder.py --trust P: MixSettings.trust for every battle of the player. The app's guarded mode
    # searched wider is --player mix-deep --trust 0.03 (mix-deep is no ARM_FIXES arm).
    from dataclasses import replace

    plain = ladder.parse_args(["--player", "mix-deep"])
    args = ladder.parse_args(["--player", "mix-deep", "--trust", "0.03"])
    assert plain.trust == 0.0 and ladder.mix_settings(plain).trust == 0.0
    assert ladder.mix_settings(args) == replace(ladder.mix_settings(plain), trust=0.03)
    # the same settings as a battle of the arm mix-trust gets
    assert ladder.mix_settings(args) == replace(ladder.mix_settings(plain), rule="veto", **sp.ARM_FIXES["mix-trust"][0])
    assert ladder.search_problems(args) == []
    for bad in ("-0.1", "1", "1.5", "nan"):
        with pytest.raises(SystemExit):
            ladder.parse_args(["--trust", bad])
    pool = ladder.parse_args(["--player", "mix", "--mix-rule", "pool", "--trust", "0.03"])
    assert any("--trust 0.03 works with the veto rule only" in p for p in ladder.search_problems(pool))
    # search plays the search's own pick, and predict's record has no "trust": the flag is refused there
    for player in ("search", "predict"):
        other = ladder.parse_args(["--player", player, "--mix-rule", "veto", "--trust", "0.03"])
        assert any("--trust 0.03 guards the mix modes only, not " + player in p for p in ladder.search_problems(other))
    arms = ladder.parse_args(["--arms", "net,mix,search", "--trust", "0.03"])
    assert any("guards the mix modes only, not search" in p for p in ladder.search_problems(arms))
    assert ladder.search_problems(ladder.parse_args(["--arms", "net,mix", "--trust", "0.03"])) == []


APP_THINK_TABLE = os.path.join(REPO, "tests", "fixtures", "app_think_table.txt")


def app_think_rows():
    """The lines of tests/fixtures/app_think_table.txt (PSBot Lookahead.exe --test-think) as dicts."""
    rows = []
    with open(APP_THINK_TABLE, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            words = line.split()
            seconds, style, mode, player, teams, ms, about, cap, floor = words[:9]
            rows.append(dict(seconds=float(seconds), style=style, mode=mode, player=player, teams=int(teams), ms=int(ms),
                             about=float(about), cap=float(cap), floor=float(floor), argv=words[9:]))
    return rows


def test_every_thinking_setting_of_the_app_can_be_played():
    """
    The window's thinking slider (0.5 to 7 s a move) and search style (deeper, wider) in both search
    modes, as "PSBot Lookahead.exe --test-think" printed them (tests/fixtures/app_think_table.txt; print
    it again after a change of Thinking.Plan in app/PSBotLookahead.cs): every line passes ladder.py's
    checks next to the window's other arguments, its cap leaves room under the battle timer's 10 s for
    the order spacing's longest wait and the trip to the server, and 1, 2 and 5 s are the window's
    three settings from before the slider, byte for byte.
    """
    rows = app_think_rows()
    assert len(rows) == 14 * 2 * 2
    assert sorted({r["seconds"] for r in rows}) == [0.5 * i for i in range(1, 15)]
    for r in rows:
        args = ladder.parse_args(["--games", "1", "--parallel", "2"] + r["argv"])
        assert ladder.search_problems(args) == [], r
        assert (args.player, args.mix_rule, args.helpers) == (r["player"], "veto", 3), r
        wide = args.player == sp.DEEP_ARM
        budget = ladder.deep_budget(args) if wide else None
        worlds = budget.worlds_x if wide else 1
        ms, cap = (budget.search_ms, budget.deadline) if wide else (args.search_ms, args.deadline)
        assert (ms, cap, r["teams"]) == (r["ms"], r["cap"], 2 * worlds), r
        assert r["floor"] == pytest.approx(sp.minimum_deadline(ms, worlds), abs=0.005), r
        assert sp.minimum_deadline(ms, worlds) <= cap < 10.0, r
        # after the cap: OrderGuard may hold the order up to --order-max-wait-s, and it travels to the server (about 0.3 s both ways)
        assert cap + args.order_max_wait_s + 0.3 <= 10.0 + 1e-9, r
        if wide:
            assert cap < budget.max_decision_s, r     # a decision at its deadline never sends the game back to today's budget
        assert r["about"] == pytest.approx(2.26 * ms * worlds / 1000.0, abs=0.005), r
        assert sp.minimum_deadline(args.search_ms) <= args.deadline < 10.0, r
        guarded = r["mode"] == "guarded"
        if wide:
            assert r["style"] == "wider" and worlds >= 2 and (budget.threads, budget.workers, ms) == (1, 1, 500), r
            assert (args.search_ms, args.deadline) == (500, 2.5), r          # what a game falls back to: the 1 s setting
            assert ladder.mix_settings(args).trust == (sp.ARM_FIXES["mix-trust"][0]["trust"] if guarded else 0.0), r
        else:
            assert args.player == ("mix-trust" if guarded else "mix") and args.trust == 0.0, r
    by = {(r["seconds"], r["style"], r["mode"]): r for r in rows}
    for seconds, ms, cap in ((1.0, 500, "2.5"), (2.0, 900, "2.7"), (5.0, 2200, "6.0")):
        for mode, player in (("lookahead", "mix"), ("guarded", "mix-trust")):
            assert by[(seconds, "deeper", mode)]["argv"] == ["--player", player, "--mix-rule", "veto", "--search-ms", str(ms),
                                                             "--helpers", "3", "--deadline", cap]
    for mode in ("lookahead", "guarded"):
        for seconds in (0.5, 1.0, 1.5):        # no time for more teams: Wider is Deeper
            assert by[(seconds, "wider", mode)]["argv"] == by[(seconds, "deeper", mode)]["argv"]
        for style in ("deeper", "wider"):      # more time is never less search
            line = [by[(0.5 * i, style, mode)] for i in range(1, 15)]
            assert all(a["ms"] * a["teams"] <= b["ms"] * b["teams"] for a, b in zip(line, line[1:])), style
        assert [by[(s, "wider", mode)]["teams"] for s in (2.0, 3.0, 4.0, 5.0, 6.0, 6.5, 7.0)] == [4, 6, 8, 8, 10, 12, 12]
    source = open(os.path.join(REPO, "app", "PSBotLookahead.cs"), encoding="utf-8").read()
    assert "public const double GuardTrust = %s;" % sp.ARM_FIXES["mix-trust"][0]["trust"] in source


def test_veto_does_not_veto_what_the_search_never_looked_at():
    mask = mask_of(0, 1, 4)
    probs = spread(a0=0.2, a1=0.1, a4=0.7)
    # The search has no option "switch" at all (it believes we are trapped): no opinion, the network's switch stands.
    view = view_of(spread(a0=0.7, a1=0.3), visits=[700, 300] + [0] * 12)
    assert mix(probs, mask, view, MixSettings(rule="veto")) == 4


def test_pool_is_a_weighted_sum_of_log_probabilities_with_floors():
    mask = mask_of(0, 1, 4)
    probs = spread(a0=0.70, a1=0.25, a4=0.05)
    view = view_of(spread(a0=0.10, a1=0.50, a4=0.40))
    # 0.70 * 0.10 = 0.07, 0.25 * 0.50 = 0.125, 0.05 * 0.40 = 0.02
    assert mix(probs, mask, view, MixSettings(rule="pool")) == 1
    assert mix(probs, mask, view, MixSettings(rule="pool", search_weight=0.0)) == 0      # the network alone
    assert mix(probs, mask, view, MixSettings(rule="pool", net_weight=0.0)) == 1         # the search alone
    # Floors: an action the search never visited is not ruled out, it counts as search_floor.
    probs = spread(a0=0.98, a1=0.01, a4=0.01)
    view = view_of(spread(a1=0.6, a4=0.4), visits=[0, 600, 0, 0, 400] + [0] * 9)
    assert mix(probs, mask, view, MixSettings(rule="pool", search_floor=0.05)) == 0      # 0.98 * 0.05 > 0.01 * 0.6
    assert mix(probs, mask, view, MixSettings(rule="pool", search_floor=1e-4)) == 1      # 0.98 * 0.0001 < 0.01 * 0.6
    # And a move the network gave almost nothing counts as net_floor, so a strong search view can carry it.
    probs = spread(a0=0.9999, a1=0.00005, a4=0.00005)
    view = view_of(spread(a0=0.01, a1=0.98, a4=0.01))
    assert mix(probs, mask, view, MixSettings(rule="pool", net_floor=5e-2)) == 1
    assert mix(probs, mask, view, MixSettings(rule="pool", net_floor=1e-9)) == 0


def test_shortlist_lets_the_search_choose_among_the_networks_likeliest():
    mask = mask_of(0, 1, 2, 4, 5)
    probs = spread(a0=0.50, a1=0.30, a2=0.15, a4=0.04, a5=0.01)
    view = view_of(spread(a0=0.05, a1=0.15, a2=0.10, a4=0.10, a5=0.60))
    # The search's favourite (5) is not among the network's top three: the best of those for the search is 1.
    assert mix(probs, mask, view, MixSettings(rule="shortlist", top=3, ratio=0.1)) == 1
    assert mix(probs, mask, view, MixSettings(rule="shortlist", top=1)) == 0             # a list of one is the network
    assert mix(probs, mask, view, MixSettings(rule="shortlist", top=5, ratio=0.0)) == 5  # a list of all is the search
    # ratio: with 0.4 only the moves with at least 0.4 * 0.50 are listed (0 and 1).
    assert mix(probs, mask, view_of(spread(a0=0.05, a1=0.10, a2=0.60, a4=0.25)), MixSettings(rule="shortlist", ratio=0.4)) == 1
    # The search saw none of the listed moves: it has nothing to say, the network's first.
    view = view_of(spread(a5=1.0), visits=[0, 0, 0, 0, 0, 1000] + [0] * 8)
    assert mix(probs, mask, view, MixSettings(rule="shortlist", top=3)) == 0


@pytest.mark.parametrize("rule", sp.RULES)
def test_the_search_decides_between_a_move_and_its_tera_form(rule):
    mask = mask_of(0, 1, 4, 10, 11)
    settings = MixSettings(rule=rule, tera_share=0.20)
    # The network puts everything on the Tera form of move 0. It does not see the Tera type, so
    # that says "move 0", not "Terastallize": the search prefers the plain move, and plain it is.
    probs = spread(a10=0.9, a1=0.05, a4=0.05)
    assert mix(probs, mask, view_of(spread(a0=0.45, a10=0.25, a1=0.2, a4=0.1)), settings) == 0
    # The network wants the plain move, the search wants it with Tera and clearly so: with Tera.
    probs = spread(a0=0.9, a1=0.05, a4=0.05)
    assert mix(probs, mask, view_of(spread(a0=0.25, a10=0.45, a1=0.2, a4=0.1)), settings) == 10
    # The Tera form is ahead of the plain form but is not one of the search's main lines: no Tera.
    assert mix(probs, mask, view_of(spread(a0=0.14, a10=0.16, a1=0.4, a4=0.3)), settings) == 0


def test_veto_plays_the_tera_form_when_the_search_turned_away_from_the_plain_one():
    mask = mask_of(0, 1, 4, 10, 11)
    probs = spread(a0=0.9, a1=0.05, a4=0.05)
    # Plain move 0 has a twentieth of the favourite's visits (vetoed), its Tera form is the favourite.
    assert mix(probs, mask, view_of(spread(a0=0.03, a10=0.6, a1=0.2, a4=0.17)), MixSettings(rule="veto")) == 10


def test_mix_never_returns_an_illegal_action():
    rng = random.Random(7)
    for _ in range(3000):
        legal = sorted(rng.sample(range(14), rng.randint(1, 14)))
        mask = mask_of(*legal)
        raw = [rng.random() if a in legal else 0.0 for a in range(14)]
        probs = [x / sum(raw) for x in raw]
        seen = [a for a in legal if rng.random() < 0.8] or legal[:1]
        visits = [rng.randint(1, 5000) if a in seen else 0 for a in range(14)]
        shares = [v / sum(visits) for v in visits]
        view = view_of(shares, visits=visits)
        for rule in sp.RULES:
            settings = MixSettings(rule=rule, keep=rng.random(), top=rng.randint(1, 4), ratio=rng.random(),
                                   tera_share=rng.random() * 0.5)
            assert mix(probs, mask, view, settings) in legal
    assert mix([0.0] * 14, [0.0] * 14, view_of(spread(a0=1.0))) == 0      # nothing legal: action 0, as hybrid.combine


def advice_of(policy, visits=None, iterations=50_000, unmapped=()):
    policy = [float(x) for x in policy]
    visits = visits or [int(round(x * 1000)) for x in policy]
    advice = fb.Advice(choice=None, best=None, iterations=iterations, worlds=2)
    advice.policy, advice.visits = policy, visits
    advice.score = [0.5 if v else None for v in visits]
    advice.unmapped = [fb.Option(choice=None, fp="x", visits=int(p * 1000), score=0.5, total_score=0.0, policy=p) for p in unmapped]
    return advice


def test_read_advice_says_why_an_answer_cannot_be_used():
    mask = mask_of(0, 1, 4)
    view, why = read_advice(advice_of(spread(a0=0.2, a1=0.5, a4=0.3)), mask)
    assert why == "" and view.best == 1 and view.iterations == 50_000 and view.share[4] == pytest.approx(0.3)
    assert read_advice(None, mask) == (None, "error")
    broken = advice_of(spread(a0=1.0))
    broken.policy = [1.0]                                      # fill_actions() was not run: no 14 numbers
    assert read_advice(broken, mask) == (None, "illegal")
    # Everything the search looked at is something we cannot play.
    assert read_advice(advice_of(spread(a2=0.6, a5=0.4)), mask) == (None, "illegal")
    assert read_advice(advice_of([0.0] * 14, visits=[0] * 14, unmapped=[1.0]), mask) == (None, "illegal")
    # Its favourite is not offered by the request, though something else it looked at is.
    assert read_advice(advice_of(spread(a0=0.2, a1=0.1), unmapped=[0.7]), mask) == (None, "illegal")
    view, why = read_advice(advice_of(spread(a0=0.5, a1=0.3), unmapped=[0.2]), mask)
    assert why == "" and view.best == 0                        # a side line that is not offered does no harm
    # One option where we have several: the search believes we are locked into it.
    assert read_advice(advice_of(spread(a1=1.0)), mask) == (None, "narrow")
    assert read_advice(advice_of(spread(a1=1.0)), mask_of(1))[1] == ""       # one legal action, one option: fine
    # Too few iterations to mean anything.
    assert read_advice(advice_of(spread(a0=0.2, a1=0.5, a4=0.3), iterations=900), mask, min_iterations=1000) == (None, "weak")
    assert read_advice(advice_of(spread(a0=0.2, a1=0.5, a4=0.3), iterations=1000), mask, min_iterations=1000)[1] == ""


def test_minimum_deadline_is_what_the_helper_needs_plus_the_wait():
    assert sp.minimum_deadline(500) == pytest.approx(2.5 * 0.5 + 0.02 + sp.RESERVE_S + 0.3)
    assert sp.minimum_deadline(500) < 2.5 < sp.minimum_deadline(1000)     # 500 ms fits the default deadline, 1000 does not


# =============================================================================================
# Arms
# =============================================================================================

def test_arm_picker_keeps_the_arms_level():
    picker = ArmPicker(["net", "mix", "search"], rng=random.Random(3))
    given = [picker("battle-%d" % i) for i in range(31)]
    counts = {arm: given.count(arm) for arm in picker.arms}
    assert max(counts.values()) - min(counts.values()) <= 1 and sum(counts.values()) == 31
    for start in range(0, 30, 3):
        assert sorted(given[start:start + 3]) == ["mix", "net", "search"]      # every round has each arm once
    assert [picker("battle-%d" % i) for i in range(31)] == given               # a battle keeps its arm
    assert picker.counts == counts


def test_arm_picker_starts_from_the_games_already_played():
    picker = ArmPicker(["net", "mix"], counts={"net": 5, "mix": 2, "search": 9}, rng=random.Random(1))
    assert [picker("b%d" % i) for i in range(3)] == ["mix", "mix", "mix"]      # mix catches up first
    assert sorted(picker("c%d" % i) for i in range(4)) == ["mix", "mix", "net", "net"]
    assert picker.counts == {"net": 7, "mix": 7}


def test_arm_picker_refuses_bad_lists():
    with pytest.raises(ValueError):
        ArmPicker(["net", "net"])
    with pytest.raises(ValueError):
        ArmPicker(["net", "tree"])
    with pytest.raises(ValueError):
        ArmPicker([])


def test_arm_counts_and_arm_fields_and_arms_text():
    games = [
        {"arm": "mix", "checkpoint": "a.pt", "won": True, "decisions": 30, "searched": 28, "fallbacks": 1, "overruled": 4, "tera": True},
        {"arm": "mix", "checkpoint": "a.pt", "won": False, "decisions": 20, "searched": 20, "fallbacks": 0, "overruled": 2, "tera": False},
        {"arm": "net", "checkpoint": "a.pt", "won": True, "decisions": 25, "searched": 0, "fallbacks": 0, "overruled": 0, "tera": True},
        {"arm": "mix", "checkpoint": "b.pt", "won": True},
        {"checkpoint": "a.pt", "won": True},                                   # a game from before there were arms
    ]
    assert ladder.arm_counts(games, "a.pt") == {"mix": 2, "net": 1}
    assert ladder.arm_counts(games, "b.pt") == {"mix": 1}
    assert ladder.arm_counts(games[4:], "a.pt") == {}
    text = ladder.arms_text(games[:3]).splitlines()
    assert len(text) == 2 and text[0].strip().startswith("mix") and "1-1 (50%)" in text[0] and "Tera in 1" in text[0]
    assert "search answered 48 of 50 decisions, 1 fell back to the network, the network was overruled in 6" in text[0]
    assert text[1].strip().startswith("net") and "1-0 (100%)" in text[1] and "search answered" not in text[1]
    assert ladder.arms_text(games[4:]) == ""

    summary = {"arm": "mix", "decisions": 30, "searched": 28, "fallbacks": 1, "overruled": 4, "tera": True, "tera_turn": 9}
    line = ladder.arm_fields(summary, "p2", 500, "veto")
    assert line == dict(summary, role="p2", search_ms=500, mix_rule="veto")
    assert ladder.arm_fields({"arm": "mix-pool"}, "p1", 500, "veto")["mix_rule"] is None     # the arm's name says the rule
    assert ladder.arm_fields({"arm": "net"}, "p1", None, "veto") == {"arm": "net", "role": "p1", "search_ms": None, "mix_rule": None}
    assert ladder.arm_fields(None, None, None, "veto") == {"role": None, "search_ms": None, "mix_rule": None}


def test_ladder_arguments(tmp_path, capsys, monkeypatch):
    for name in ("OUT", "GRAPH", "BATTLES"):
        monkeypatch.setattr(ladder, name, getattr(ladder, name))      # main() moves them with --out: put back afterwards
    out = str(tmp_path / "results.jsonl")
    # --history needs no player: it shows that the new options parse and that --out is honoured.
    assert ladder.main(["--history", "--server", "local:1", "--out", out, "--arms", "net,mix", "--search-ms", "250", "--helpers", "9"]) == 0
    assert "no games recorded" in capsys.readouterr().out
    assert ladder.OUT == out and ladder.BATTLES == str(tmp_path / "battles")
    for bad in ("net,net", "net,tree", ","):
        with pytest.raises(SystemExit):
            ladder.main(["--history", "--server", "local:1", "--out", out, "--arms", bad])
    with pytest.raises(SystemExit):
        ladder.main(["--history", "--player", "tree"])


def test_battle_recorder_keeps_battle_rooms_only(tmp_path):
    recorder = ladder.BattleRecorder(str(tmp_path / "battles"))
    recorder.write("", "out", "/trn someone,0,assertion-that-must-not-be-kept")
    recorder.write("lobby", "in", "|c|someone|hello")
    recorder.write(None, "in", "x")
    recorder.write("battle-gen9randombattle-7", "note", {"arm": "mix"})
    recorder.write("battle-gen9randombattle-7", "in", "|request|{}\n|turn|1")
    recorder.write("battle-gen9randombattle-7", "out", "/choose move tackle")
    recorder.close("battle-gen9randombattle-7")
    recorder.write("battle-gen9randombattle-7", "in", "|raw|a late line")     # after the close: the same file again
    assert recorder.flush()
    files = sorted(os.listdir(tmp_path / "battles"))
    assert files == ["battle-gen9randombattle-7.jsonl"]
    with open(tmp_path / "battles" / files[0], encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert [r["dir"] for r in rows] == ["note", "in", "out", "in"]
    assert rows[1]["data"] == "|request|{}\n|turn|1" and rows[2]["data"] == "/choose move tackle"
    assert all(isinstance(r["t"], float) for r in rows)
    assert "assertion" not in open(tmp_path / "battles" / files[0], encoding="utf-8").read()
    recorder.close_all()

    # A server that numbers its battles from 1 again: the old file is left alone.
    again = ladder.BattleRecorder(str(tmp_path / "battles"))
    again.write("battle-gen9randombattle-7", "in", "|init|battle")
    again.close_all()
    assert sorted(os.listdir(tmp_path / "battles")) == ["battle-gen9randombattle-7.2.jsonl", "battle-gen9randombattle-7.jsonl"]
    assert recorder.failed == 0 and again.failed == 0


def test_battle_recorder_never_raises(tmp_path):
    blocked = tmp_path / "a-file"
    blocked.write_text("x")
    recorder = ladder.BattleRecorder(str(blocked))            # a folder that cannot be made
    recorder.write("battle-gen9randombattle-1", "in", "|init|battle")
    recorder.close("battle-gen9randombattle-1")
    assert recorder.flush()
    assert recorder.failed >= 1


# =============================================================================================
# The graph
# =============================================================================================

def games_for_graph(arms=None, n=24, checkpoint="learner_update_001475.pt"):
    rng = random.Random(11)
    games, rating = [], 1500
    for i in range(n):
        won = rng.random() < 0.6
        after = rating + (18 if won else -17)
        d = {"time": "2026-10-06T0%d:%02d:00Z" % (i // 60, i % 60), "battle": "battle-gen9randombattle-%d" % i, "won": won,
             "user": "tester", "turns": 20 + i, "opponent": "opp<%d>" % i, "opponent_rating": 1480 + 7 * (i % 9),
             "rating_before": rating, "rating_after": after, "checkpoint": checkpoint, "server": "local:1"}
        if arms:
            arm = arms[i % len(arms)]
            if arm:
                d.update({"arm": arm, "decisions": 30, "searched": 0 if arm == "net" else 27, "fallbacks": 0 if arm == "net" else 2,
                          "overruled": 0 if arm == "net" else 5, "iterations_per_decision": 0 if arm == "net" else 200_000,
                          "tera": i % 4 != 0, "tera_turn": 5 + i % 7 if i % 4 != 0 else None})
        games.append(d)
        rating = after
    return games


def old_ladder_graph():
    """ladder_graph.py as it was before arms (commit 2680577), or None when git cannot give it."""
    try:
        done = subprocess.run(["git", "-C", REPO, "show", "2680577:ladder_graph.py"], capture_output=True, timeout=60)
    except Exception:
        return None
    if done.returncode != 0 or not done.stdout:
        return None
    module = type(sys)("ladder_graph_before_arms")
    exec(compile(done.stdout.decode("utf-8"), "ladder_graph_before_arms.py", "exec"), module.__dict__)
    return module


def test_a_page_without_arms_reads_as_before(monkeypatch):
    games = games_for_graph()
    page = ladder_graph.render(games, "local:1", "tester")
    for word in ("By arm", "dots by arm", "class=\"arm", "--a0", "hollow"):
        assert word not in page
    assert page.count('<circle class="win"') + page.count('<circle class="loss"') == len(games)
    assert ladder_graph.arm_names(games) == [] and ladder_graph.arm_rows(games) == []
    old = old_ladder_graph()
    if old is None:
        pytest.skip("git could not show ladder_graph.py as it was")
    monkeypatch.setattr(time, "strftime", lambda *a, **k: "06 Oct 08:00:00" if len(a) == 1 else "06 Oct 08:00")
    assert ladder_graph.render(games, "local:1", "tester") == old.render(games, "local:1", "tester")
    assert ladder_graph.render(games, "public", None) == old.render(games, "public", None)
    assert ladder_graph.render([], "public", None) == old.render([], "public", None)


def test_arm_rows_count_each_arm_by_itself():
    games = games_for_graph(arms=["mix", "search", "net"])
    rows = ladder_graph.arm_rows(games)
    assert [r["arm"] for r in rows] == ["mix", "search", "net"] and [r["colour"] for r in rows] == [0, 1, 2]
    for r in rows:
        own = [d for d in games if d["arm"] == r["arm"]]
        assert r["games"] == len(own) == 8 and r["wins"] == sum(d["won"] for d in own) and r["wins"] + r["losses"] == 8
        level = ladder_graph.performance(own)
        assert r["level"] == level and level is not None and level[2] == 8
        assert r["tera_games"] == sum(1 for d in own if d["tera"])
        assert r["decisions"] == 240
    mix_row, net_row = rows[0], rows[2]
    assert (mix_row["searched"], mix_row["fallbacks"], mix_row["overruled"], mix_row["iterations_per_decision"]) == (216, 16, 40, 200_000)
    assert (net_row["searched"], net_row["fallbacks"], net_row["overruled"], net_row["iterations_per_decision"]) == (0, 0, 0, None)
    turns = sorted(d["tera_turn"] for d in games if d["arm"] == "mix" and d["tera"])
    assert mix_row["tera_turn"] == turns[len(turns) // 2]
    # An arm with too few rated games has no level yet.
    few = ladder_graph.arm_rows(games[:6])
    assert all(r["games"] == 2 and r["level"] is None for r in few)


def test_a_page_with_arms_shows_each_arm_and_colours_its_dots():
    games = games_for_graph(arms=["mix", "search", "net"])
    page = ladder_graph.render(games, "local:1", "tester")
    assert "<h2>By arm</h2>" in page and "dots by arm:" in page and "--a0" in page
    for index, arm in enumerate(["mix", "search", "net"]):
        own = [d for d in games if d["arm"] == arm]
        wins = sum(d["won"] for d in own)
        assert f'<u class="a{index}"></u>{arm}' in page                               # in the legend and in the table
        assert page.count(f'<circle class="arm a{index}"') == wins                   # filled: a win
        assert page.count(f'<circle class="arm a{index} lost"') == len(own) - wins   # hollow: a loss
        level, se, _ = ladder_graph.performance(own)
        assert f"<td>{len(own)}</td><td>{wins} - {len(own) - wins}</td><td>{100.0 * wins / len(own):.0f}%</td><td>{level:.0f} &#177; {se:.0f}</td>" in page
    assert '<circle class="win"' not in page and '<circle class="loss"' not in page
    assert "played as search" in page and "green and red" not in page
    assert "216 of 240 decisions" in page and "200,000" in page
    assert "opp&lt;3&gt;" in page and "opp<3>" not in page                           # names are still escaped


def test_a_page_with_some_games_from_before_the_arms_keeps_their_dots():
    games = games_for_graph(arms=[None, "mix"])
    page = ladder_graph.render(games, "local:1", "tester")
    plain = [d for d in games if not d.get("arm")]
    assert page.count('<circle class="win"') + page.count('<circle class="loss"') == len(plain) == 12
    assert page.count('<circle class="arm a0') == 12
    assert "green and red: games without an arm" in page
    assert [r["arm"] for r in ladder_graph.arm_rows(games)] == ["mix"]


def test_the_arm_table_is_of_the_newest_network_and_the_dots_of_every_game():
    old = games_for_graph(arms=["search"], n=10, checkpoint="learner_update_001000.pt")
    new = games_for_graph(arms=["mix"], n=12)
    page = ladder_graph.render(old + new, "local:1", "tester")
    table = page.split("<h2>By arm</h2>", 1)[1].split("</table>", 1)[0]
    assert '<u class="a1"></u>mix</td>' in table and "search</td>" not in table      # the colour it has in the graph
    assert page.count('<circle class="arm a0') == 10 and page.count('<circle class="arm a1') == 12   # search came first


def test_guarded_games_of_another_arm_are_shown_apart():
    # The window's guarded mode searched wider records "mix-deep" and "trust" 0.03 (ladder.py --trust):
    # the graph and the arms text keep it apart from the unguarded mix-deep games; mix-trust says it already.
    games = games_for_graph(arms=["mix-deep", "mix-deep", "mix-trust"], n=12)
    for i, d in enumerate(games):
        if d["arm"] == "mix-trust" or (d["arm"] == "mix-deep" and i % 3 == 1):
            d["trust"] = 0.03
    assert ladder_graph.arm_of({"arm": "mix-deep", "trust": 0.03}) == "mix-deep+trust"
    assert ladder_graph.arm_of({"arm": "mix-trust", "trust": 0.03}) == "mix-trust"
    assert ladder_graph.arm_of({"arm": "mix"}) == "mix" and ladder_graph.arm_of({"won": True}) == ""
    rows = ladder_graph.arm_rows(games)
    assert [r["arm"] for r in rows] == ["mix-deep", "mix-deep+trust", "mix-trust"]
    assert [r["games"] for r in rows] == [4, 4, 4]
    page = ladder_graph.render(games, "local:1", "tester")
    assert "played as mix-deep+trust" in page and page.count('<circle class="arm a1') == 4
    text = ladder.arms_text(games).splitlines()
    assert [line.split()[0] for line in text] == ["mix-deep", "mix-deep+trust", "mix-trust"]


def test_more_arms_than_colours_start_again_at_the_first():
    arms = ["net", "search", "mix", "mix-veto", "mix-pool", "mix-shortlist", "extra"]
    rows = ladder_graph.arm_rows(games_for_graph(arms=arms, n=14))
    assert [r["colour"] for r in rows] == [0, 1, 2, 3, 4, 5, 0]


def test_write_page_with_arms(tmp_path):
    path = ladder_graph.write_page(games_for_graph(arms=["net", "mix"]), str(tmp_path / "elo.html"), "local:1", "tester")
    text = open(path, encoding="utf-8").read()
    assert text.startswith("<!doctype html>") and "By arm" in text and not os.path.exists(path + ".tmp")


# =============================================================================================
# The player, on recorded battles, with a fake search and a stub network
# =============================================================================================

def load_fixture():
    battles, current = [], None
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as fh:
        for line in fh:
            event = json.loads(line)
            if event["t"] == "msg" and event["data"].startswith("|init|"):
                current = []
                battles.append(current)
            current.append(event)
    return battles


def split(room: str, data: str):
    return [[">" + room]] + [line.split("|") for line in data.split("\n")]


def key_of(events):
    return "%s|%s" % (events[-1]["username"], events[0]["room"])


def orders_of(events):
    return [e for e in events if e["t"] == "order"]


class FakeBridge:
    """
    Stands in for foulplay_bridge.FoulPlayBridge. It answers a request with what the real helper
    answered when the battle was recorded, after `search_s` seconds, one question at a time per
    fake helper (as a real helper does: the engine holds the interpreter lock).

    behave   "answer"     the recorded answer
             "never"      no answer at all
             "error"      the helper failed
             "expired"    the helper did not start the search (the caller's time was up)
             "raise"      advice_future itself raises (a bridge that is broken in a way nobody planned)
             "closed"     the bridge was closed under the player
             "illegal"    an answer whose every option is something we cannot play
             "elsewhere"  an answer whose favourite is something we cannot play
             "narrow"     an answer with one option only
             "weak"       the recorded answer, with next to no iterations
             "garbage"    not an Advice at all
    """

    def __init__(self, battles, behave="answer", search_s=0.0, helpers=1):
        self.replies = {}
        for events in battles:
            for order in orders_of(events):
                self.replies[(key_of(events), order["rqid"])] = order["fp_reply"]
        self.behave, self.search_s, self.search_ms, self._closed = behave, float(search_s), 20, False
        self.block_loop_s = 0.0          # > 0: every question also holds the player's event loop up for this long
        self.fed, self.notes, self.forgotten, self.asked = [], [], [], []
        self.answered = self.expired = 0
        self._helper_of = {}
        self._queues = [queue.Queue() for _ in range(helpers)]
        self._threads = [threading.Thread(target=self._serve, args=(q,), daemon=True) for q in self._queues]
        for thread in self._threads:
            thread.start()

    # -- what the mixin and the player call -------------------------------------------------------
    def feed(self, key, user, room, data):
        self._helper_of.setdefault(key, len(self._helper_of) % len(self._queues))
        self.fed.append((key, data))

    def note(self, key, choice):
        self.notes.append((key, choice))

    def forget(self, key):
        self.forgotten.append(key)

    def advice_future(self, key, search_ms=None, rqid=None, ttl=None, **more):
        self.asked.append((key, rqid, ttl, search_ms, more))
        if self.block_loop_s > 0:
            asyncio.get_running_loop().call_later(0.05, time.sleep, self.block_loop_s)
        if self.behave == "raise":
            raise RuntimeError("a bridge that is broken")
        future = concurrent.futures.Future()
        if self.behave == "closed":
            future.set_exception(fb.FoulPlayBridgeError("the Foul Play bridge is closed"))
        elif self.behave == "never":
            pass
        elif self.search_s <= 0:
            self._answer(future, key, rqid, 0.0)
        else:
            self._queues[self._helper_of.get(key, 0)].put((future, key, rqid, time.monotonic(), ttl))
        return future

    def stats(self):
        return {"decisions": self.answered, "expired": self.expired, "recent_iterations_per_decision": 0, "helper_exits": 0}

    def close(self):
        self._closed = True
        for q in self._queues:
            q.put(None)

    # -- the fake helper --------------------------------------------------------------------------
    def _serve(self, q):
        while True:
            item = q.get()
            if item is None:
                return
            future, key, rqid, asked, ttl = item
            if future.done():                 # the player stopped waiting: a real helper still searches
                time.sleep(self.search_s)
                continue
            if ttl is not None and time.monotonic() + self.search_s > asked + ttl:
                self.expired += 1
                error = fb.FoulPlayBridgeError("not searched: nobody waits for this decision any more")
                error.expired = True
                try:
                    future.set_exception(error)
                except concurrent.futures.InvalidStateError:
                    pass
                continue
            time.sleep(self.search_s)
            self._answer(future, key, rqid, (time.monotonic() - asked) * 1000.0)

    def _answer(self, future, key, rqid, wall_ms):
        if future.done():
            return
        reply = dict(self.replies.get((key, rqid)) or {"ok": False, "error": "nothing recorded for this request"})
        behave = self.behave
        try:
            if not reply.get("ok"):
                error = fb.FoulPlayBridgeError(str(reply.get("error")))
                error.revive = bool(reply.get("revive"))
                future.set_exception(error)
                return
            if behave == "error":
                future.set_exception(fb.FoulPlayBridgeError("Foul Play's decision failed: KeyError: ''"))
                return
            if behave == "expired":
                error = fb.FoulPlayBridgeError("not searched")
                error.expired = True
                future.set_exception(error)
                return
            if behave == "garbage":
                future.set_result("not an advice")
                return
            if behave == "illegal":
                reply["options"] = [dict(o, choice="move 9") for o in reply["options"]]
            elif behave == "elsewhere":
                options = [dict(o, policy=0.01) for o in reply["options"]]
                options.append({"choice": "switch 9", "fp": "switch nobody", "visits": 99000, "total_score": 50000.0, "policy": 0.9})
                reply["options"] = options
            elif behave == "narrow":
                reply["options"] = [dict(reply["options"][0], policy=1.0)]
            elif behave == "weak":
                reply["iterations"] = 40
            self.answered += 1
            future.set_result(fb._advice_from(reply, wall_ms, 0))
        except concurrent.futures.InvalidStateError:
            pass


def stub_logits(row):
    """The stub network: the same 14 logits for the same observation."""
    rng = np.random.default_rng(zlib.crc32(np.ascontiguousarray(row, dtype=np.float32).tobytes()))
    return rng.normal(0.0, 2.0, 14).astype(np.float32)


class StubPolicy:
    """hybrid.PolicyNet without the network: the assembler and the per-battle cache, nothing to load."""

    def __init__(self, cfg):
        import hybrid
        from obs_assembler import ObservationAssembler

        with hybrid._in_repo_root():
            self.assembler = ObservationAssembler()
        self.device, self.path, self.net, self._cache = "cpu", None, None, {}

    def forget(self, tag):
        self._cache.pop(tag, None)


@pytest.fixture(scope="module")
def classes():
    """(the plain policy player, the search player), both with the stub network."""
    import hybrid
    from players import batched_policy_player

    real = hybrid.PolicyNet
    hybrid.PolicyNet = StubPolicy
    made = {}

    def stubbed(base):
        class Stubbed(base):
            forward_calls = 0

            def _forward_batch(self, tags, obs):
                type(self).forward_calls += 1
                for tag in tags:
                    self.policy._cache[tag] = True
                self.batch_stats["forwards"] += 1
                self.batch_stats["rows"] += len(tags)
                return np.stack([stub_logits(row) for row in obs])

        return Stubbed

    made["plain"] = stubbed(batched_policy_player())
    made["search"] = stubbed(sp.search_player())
    yield made
    hybrid.PolicyNet = real


def offline(cls, username, **kwargs):
    """A real poke-env player that never connects: its message handler is called by hand."""
    from poke_env import AccountConfiguration, ServerConfiguration

    from hybrid import HybridConfig

    hcfg = HybridConfig(checkpoint=None, beta=0.0, veto_gap=None, policy_weight=1.0, greedy=True, device="cpu", random_init=True)
    kwargs.setdefault("max_concurrent_battles", 3)
    player = cls(hybrid=hcfg, account_configuration=AccountConfiguration(username, None),
                 server_configuration=ServerConfiguration("ws://127.0.0.1:9/showdown/websocket", "http://127.0.0.1:9/action.php?"),
                 battle_format="gen9randombattle", start_listening=False, log_level=50, **kwargs)
    player.sent = []
    player.DEFAULT_CHOICE_CHANCE = 0.0           # poke-env's one-in-a-thousand "/choose default" on a retry

    async def send_message(message, room="", message_2=None):
        player.sent.append((room, message))

    if hasattr(player, "_order_send"):
        player._order_send = send_message        # behind the search player's own watch on what is sent
    else:
        player.ps_client.send_message = send_message
    return player


def searcher(classes, username, bridge, mode="search", deadline=0.5, **kwargs):
    kwargs.setdefault("mix", MixSettings(min_iterations=0))
    return offline(classes["search"], username, mode=mode, bridge=bridge, search_ms=20, deadline=deadline, warm_up=False, **kwargs)


def on_loop(player, coro, timeout=180.0):
    return asyncio.run_coroutine_threadsafe(coro, player.ps_client.loop).result(timeout=timeout)


async def feed(player, events):
    for event in events:
        if event["t"] == "msg":
            await player._handle_battle_message(split(event["room"], event["data"]))


def play(player, *battles):
    async def together():
        await asyncio.gather(*(feed(player, events) for events in battles))

    on_loop(player, together())


def chosen(player, room=None):
    return [m for r, m in player.sent if m.startswith("/choose") and (room is None or r == room)]


def check_notes(player, bridge, events):
    """Whatever was sent in this battle, the search was told, order for order."""
    key = key_of(events)
    sent = chosen(player, events[0]["room"])
    noted = [choice for k, choice in bridge.notes if k == key]
    assert len(noted) == len(sent) == len(orders_of(events))
    for message, choice in zip(sent, noted):
        words, told = message.split(), choice.split()
        assert words[1] == told[0], (message, choice)                          # a move noted as a move, a switch as a switch
        assert ("terastallize" in words) == ("terastallize" in told), (message, choice)


@needs_player
def test_net_mode_plays_what_the_policy_player_plays_and_tells_the_search_nothing(classes):
    battles = load_fixture()
    bridge = FakeBridge(battles)
    try:
        for events in battles:
            name = events[-1]["username"]
            plain = offline(classes["plain"], name)
            ours = searcher(classes, name, bridge, mode="net")
            play(plain, events)
            play(ours, events)
            assert chosen(ours) == chosen(plain) and len(chosen(ours)) == len(orders_of(events))
            assert plain.counts == ours.counts
            summary = ours.game_summary(events[0]["room"])
            assert summary["arm"] == "net" and summary["decisions"] == len(orders_of(events))
            assert (summary["asked"], summary["searched"], summary["fallbacks"], summary["overruled"]) == (0, 0, 0, 0)
            assert all(d["source"] == "net" and d["played"] == d["net"] and not d["asked"] for d in ours.decisions_of(events[0]["room"]))
        assert bridge.fed == [] and bridge.asked == [] and bridge.notes == [] and bridge.forgotten == []
    finally:
        bridge.close()


@needs_player
def test_search_mode_plays_the_largest_share_of_the_search(classes):
    battles = load_fixture()
    bridge = FakeBridge(battles)
    seen = {"searched": 0, "tera": 0, "switch": 0, "overruled": 0, "revive": 0, "reasons": {}}
    try:
        for events in battles:
            name, room = events[-1]["username"], events[0]["room"]
            player = searcher(classes, name, bridge, mode="search")
            play(player, events)
            decisions = player.decisions_of(room)
            assert len(decisions) == len(orders_of(events)) == len(chosen(player))
            assert player.n_finished_battles == 1
            for d in decisions:
                assert d["mask"][d["played"]] == 1 and sum(d["mask"]) == d["legal"]   # a legal action, always
                if d["search"] is not None:
                    assert d["source"] == "search" and d["played"] == d["search"]
                    assert d["share_played"] == max(d["shares"]) > 0                 # the largest share
                    assert d["iterations"] > 0 and isinstance(d["opponent"], list)      # (recorded before the helper sent the opponent's side)
                    assert d["net_overruled"] == (sp.base_action(d["net"]) != sp.base_action(d["played"]))
                    seen["searched"] += 1
                    seen["tera"] += d["played"] >= 10
                    seen["switch"] += 4 <= d["played"] <= 9
                    seen["overruled"] += d["net_overruled"]
                else:
                    assert d["source"] == "net" and d["played"] == d["net"] and d["reason"]
                    seen["reasons"][d["reason"]] = seen["reasons"].get(d["reason"], 0) + 1
            # A request that asks whom Revival Blessing revives: the search has no answer to it and
            # is not asked (with one fainted Pokemon there is nothing to choose either).
            for order in orders_of(events):
                if order["fp_reply"].get("revive"):
                    d = next(d for d in decisions if d["rqid"] == order["rqid"])
                    assert d["reason"] in ("revive", "single") and not d["asked"] and d["played"] == d["net"] and d["forced"]
                    seen["revive"] += 1
            check_notes(player, bridge, events)
            assert key_of(events) in bridge.forgotten
            summary = player.game_summary(room)
            assert summary["arm"] == "search" and summary["decisions"] == len(decisions)
            assert summary["searched"] + sum(summary["fallback_reasons"].values()) == summary["decisions"]
            assert summary["searched"] == sum(d["search"] is not None for d in decisions)
            assert summary["overruled"] == sum(bool(d.get("net_overruled")) for d in decisions)
            assert summary["search_overruled"] == 0 and summary["late"] == 0
            assert summary["fallbacks"] == sum(n for why, n in summary["fallback_reasons"].items() if why not in ("single", "revive"))
            assert player.search_totals["exceptions"] == 0 and player.search_totals["noted_at_send"] == 0
            assert "search:" in player.search_summary()
        assert seen["revive"] == 1                                # the recording holds one such request
        assert set(seen["reasons"]) <= {"revive", "single", "narrow", "illegal"}, seen["reasons"]
        assert seen["searched"] >= 140 and seen["tera"] >= 1 and seen["switch"] >= 10 and seen["overruled"] >= 30
    finally:
        bridge.close()


@needs_player
@pytest.mark.parametrize("mode", ["mix", "mix-veto", "mix-pool", "mix-shortlist"])
def test_mix_modes_play_legal_actions_and_record_both_sides(classes, mode):
    battles = load_fixture()
    bridge = FakeBridge(battles)
    counts = {"searched": 0, "disagree": 0, "net_overruled": 0, "search_overruled": 0}
    try:
        for events in battles:
            name, room = events[-1]["username"], events[0]["room"]
            player = searcher(classes, name, bridge, mode=mode, mix=MixSettings(rule="pool", min_iterations=0))
            play(player, events)
            decisions = player.decisions_of(room)
            assert len(decisions) == len(orders_of(events)) == len(chosen(player))
            rule = "pool" if mode == "mix" else mode[4:]               # "mix" alone takes the rule of the settings
            for d in decisions:
                assert d["mode"] == mode and d["mask"][d["played"]] == 1
                if d["search"] is None:
                    assert d["played"] == d["net"]
                    continue
                assert d["source"] == "mix"
                merged = sp.merged_prior(d["probs"], d["mask"])
                if rule == "veto" and d["shares"][d["net"]] >= 0.25 * max(d["shares"]) + 1e-4 and merged[d["net"]] > max(
                        merged[a] for a in range(14) if sp.base_action(a) != sp.base_action(d["net"])) + 1e-4:
                    assert not d["net_overruled"], d                    # sound for the search and the network's clear favourite
                counts["searched"] += 1
                for name_ in ("disagree", "net_overruled", "search_overruled"):
                    counts[name_] += bool(d[name_])
            check_notes(player, bridge, events)
            summary = player.game_summary(room)
            assert summary["overruled"] == sum(bool(d.get("net_overruled")) for d in decisions)
            assert summary["search_overruled"] == sum(bool(d.get("search_overruled")) for d in decisions)
            assert summary["disagreed"] == sum(bool(d.get("disagree")) for d in decisions)
        assert counts["searched"] >= 140
        # Where the two disagree on the move, exactly one of them is overruled, unless a third move was played.
        assert counts["net_overruled"] + counts["search_overruled"] >= counts["disagree"] > 0
        assert counts["net_overruled"] > 0 and counts["search_overruled"] > 0      # neither side always wins
    finally:
        bridge.close()


@needs_player
def test_a_search_that_never_answers_costs_the_deadline_and_no_more(classes):
    events = min(load_fixture(), key=lambda e: len(orders_of(e)))
    bridge = FakeBridge([events], behave="never")
    try:
        plain = offline(classes["plain"], events[-1]["username"])
        play(plain, events)
        player = searcher(classes, events[-1]["username"], bridge, mode="search", deadline=0.45)
        began = time.monotonic()
        play(player, events)
        took = time.monotonic() - began
        decisions = player.decisions_of(events[0]["room"])
        asked = [d for d in decisions if d["asked"]]
        assert len(asked) >= 10 and len(decisions) == len(orders_of(events))
        assert all(d["reason"] == "timeout" and d["played"] == d["net"] and d["source"] == "net" for d in asked)
        # Waited for until deadline - RESERVE_S after the request, then the network's move at once.
        assert all(400.0 - 60.0 <= d["ms"] <= 450.0 + 150.0 for d in asked), [d["ms"] for d in asked]
        assert took <= len(asked) * 0.6 + 3.0
        assert chosen(player) == chosen(plain)                    # the network's moves, every one
        assert all(ttl is not None and 0.2 < ttl <= 0.40 for _k, _r, ttl, _ms, _more in bridge.asked)   # the helper is told how long
        check_notes(player, bridge, events)
        summary = player.game_summary(events[0]["room"])
        assert summary["searched"] == 0 and summary["fallbacks"] == len(asked) == summary["fallback_reasons"]["timeout"]
        assert summary["late"] <= 1 and summary["overruled"] == 0
        assert "timeout %d" % len(asked) in player.search_summary()
    finally:
        bridge.close()


@needs_player
@pytest.mark.parametrize("behave,reason", [("error", "error"), ("expired", "expired"), ("raise", "error"), ("closed", "error"),
                                           ("illegal", "illegal"), ("elsewhere", "illegal"), ("narrow", "narrow"),
                                           ("weak", "weak"), ("garbage", "error")])
def test_every_failure_of_the_search_plays_the_networks_move_and_is_counted(classes, behave, reason):
    events = min(load_fixture(), key=lambda e: len(orders_of(e)))
    bridge = FakeBridge([events], behave=behave)
    try:
        plain = offline(classes["plain"], events[-1]["username"])
        play(plain, events)
        settings = MixSettings(min_iterations=10_000 if behave == "weak" else 0)
        for mode in ("search", "mix"):
            bridge.notes.clear()
            player = searcher(classes, events[-1]["username"], bridge, mode=mode, mix=settings)
            play(player, events)
            decisions = player.decisions_of(events[0]["room"])
            hit = [d for d in decisions if d["reason"] == reason]
            assert len(hit) >= 10
            assert all(d["reason"] in (reason, "single", "revive") for d in decisions), [d["reason"] for d in decisions]
            assert all(d["played"] == d["net"] and d["source"] == "net" and d["search"] is None for d in decisions)
            assert all(d["asked"] == (behave != "raise") for d in hit)
            assert chosen(player) == chosen(plain) and player.search_totals["exceptions"] == 0
            check_notes(player, bridge, events)
            summary = player.game_summary(events[0]["room"])
            assert summary["fallbacks"] == len(hit) and summary["fallback_reasons"][reason] == len(hit)
            assert summary["searched"] == 0 and summary["iterations_per_decision"] == 0 and summary["late"] == 0
            assert player.n_finished_battles == 1
    finally:
        bridge.close()


@needs_player
def test_without_a_bridge_a_searched_battle_is_the_network_and_says_so(classes):
    events = min(load_fixture(), key=lambda e: len(orders_of(e)))
    plain = offline(classes["plain"], events[-1]["username"])
    play(plain, events)
    player = offline(classes["search"], events[-1]["username"], mode="mix", bridge=None, warm_up=False)
    play(player, events)
    assert chosen(player) == chosen(plain)
    summary = player.game_summary(events[0]["room"])
    assert summary["arm"] == "mix" and summary["fallback_reasons"] == {"off": len(orders_of(events))} and summary["asked"] == 0


@needs_player
def test_three_battles_on_one_helper_keep_the_deadline(classes):
    """Decisions queue behind each other in a helper: each waits for its turn, no longer than the deadline."""
    battles = [e for e in load_fixture() if e[-1]["username"] == "recA3ace"]
    assert len(battles) == 3
    bridge = FakeBridge(battles, search_s=0.15, helpers=1)
    try:
        player = searcher(classes, "recA3ace", bridge, mode="search", deadline=0.45)
        play(player, *battles)                                   # three battles at the same time
        assert player.n_finished_battles == 3
        reasons = {}
        slowest = 0.0
        for events in battles:
            decisions = player.decisions_of(events[0]["room"])
            assert len(decisions) == len(orders_of(events)) == len(chosen(player, events[0]["room"]))
            for d in decisions:
                slowest = max(slowest, d["ms"])
                if d["search"] is None:
                    assert d["played"] == d["net"]
                reasons[d["reason"] or "searched"] = reasons.get(d["reason"] or "searched", 0) + 1
            check_notes(player, bridge, events)
        assert slowest <= 450.0 + 150.0, slowest                 # the deadline, plus what a busy test machine adds
        assert reasons.get("searched", 0) >= 20                  # the helper answers as many as it can in time
        assert reasons.get("timeout", 0) + reasons.get("expired", 0) >= 5, reasons    # and the rest are not waited for
        assert set(reasons) <= {"searched", "timeout", "expired", "single", "narrow", "illegal"}, reasons
        totals = player.search_totals
        assert totals["decisions"] == sum(len(orders_of(e)) for e in battles) and totals["exceptions"] == 0
        assert totals["late"] <= 2
        assert player.loop_lag["checks"] > 0 and "the event loop ran" in player.search_summary()
    finally:
        bridge.close()


@needs_player
def test_an_answer_that_was_there_is_used_when_only_the_event_loop_was_late(classes):
    """
    A busy machine holds the event loop up for seconds (seen in real games). The helper answers in
    time, in its own thread, but the loop reads the answer after the deadline, when the timer is
    due as well. The answer is in hand: it is played, and the record says how late the loop was.
    """
    whole = min(load_fixture(), key=lambda e: len(orders_of(e)))
    events = whole[:1 + [i for i, e in enumerate(whole) if e["t"] == "order"][4]] + [whole[-1]]   # the first five decisions
    bridge = FakeBridge([events], search_s=0.1)
    bridge.block_loop_s = 0.7
    try:
        player = searcher(classes, events[-1]["username"], bridge, mode="search", deadline=0.45)
        play(player, events)
        decisions = player.decisions_of(events[0]["room"])
        asked = [d for d in decisions if d["asked"]]
        assert len(asked) >= 4 and len(decisions) == len(orders_of(events)) == len(chosen(player))
        for d in asked:
            assert d["reason"] in ("", "narrow", "illegal") and d["reason"] != "timeout"
            assert d["ms"] > 600.0 and d["wait_over_ms"] > 200.0, d            # over the deadline: the loop was held up
        assert sum(1 for d in asked if d["search"] is not None and d["played"] == d["search"]) >= 3
        assert player.search_totals["late"] == len(asked)                      # late, and counted as late
        assert player.loop_lag["over_500_ms"] >= 1 and player.loop_lag["max_ms"] > 400.0
        check_notes(player, bridge, events)
    finally:
        bridge.close()


@needs_player
def test_arms_per_battle_records_and_the_decision_callback(classes):
    battles = [e for e in load_fixture() if e[-1]["username"] == "recA3ace"]
    bridge = FakeBridge(battles)
    records = []
    try:
        picker = ArmPicker(["net", "search"], counts={"net": 1}, rng=random.Random(5))
        player = searcher(classes, "recA3ace", bridge, mode="net", mode_picker=picker, on_decision=records.append)
        for events in battles:
            play(player, events)
        rooms = [e[0]["room"] for e in battles]
        arms = [player.battle_mode(room) for room in rooms]
        assert arms[0] == "search" and sorted(arms) == ["net", "search", "search"]        # search was a game behind
        assert picker.counts == {"net": 2, "search": 2}
        for events, arm in zip(battles, arms):
            room = events[0]["room"]
            summary = player.game_summary(room)
            assert summary["arm"] == arm and summary["decisions"] == len(orders_of(events))
            fed = [k for k, _d in bridge.fed if k == key_of(events)]
            if arm == "net":
                assert fed == [] and summary["asked"] == 0 and key_of(events) not in bridge.forgotten
            else:
                assert len(fed) == sum(1 for e in events if e["t"] == "msg") and summary["searched"] > 0
                check_notes(player, bridge, events)
        assert len(records) == sum(len(orders_of(e)) for e in battles)
        assert {r["battle"] for r in records} == set(rooms) and {r["mode"] for r in records} == {"net", "search"}
        json.dumps(records)                                       # ladder.py writes them as JSON lines
        assert set(player.search_totals["by_mode"]) == {"net", "search"}
        player.forget_decisions(rooms[0])
        assert player.decisions_of(rooms[0]) == [] and player.game_summary(rooms[0])["arm"] == arms[0]
        assert player.game_summary("battle-gen9randombattle-0") is None
    finally:
        bridge.close()


@needs_player
def test_a_picker_that_fails_gives_the_network(classes):
    events = min(load_fixture(), key=lambda e: len(orders_of(e)))
    bridge = FakeBridge([events])
    try:
        def broken(tag):
            raise RuntimeError("no arm today")

        player = searcher(classes, events[-1]["username"], bridge, mode="search", mode_picker=broken)
        play(player, events)
        assert player.battle_mode(events[0]["room"]) == "net" and bridge.asked == []
        assert len(chosen(player)) == len(orders_of(events))
    finally:
        bridge.close()


@needs_player
def test_the_game_summary_says_whether_and_when_it_terastallized(classes):
    events = [e for e in load_fixture() if e[-1]["username"] == "rvA93ff"][0]
    bridge = FakeBridge([events])
    try:
        player = searcher(classes, "rvA93ff", bridge, mode="search")
        role = None
        tera_turn, turn = None, 0
        for event in events:
            if event["t"] != "msg":
                continue
            for line in event["data"].split("\n"):
                parts = line.split("|")
                if len(parts) > 3 and parts[1] == "player" and parts[3] == "rvA93ff":
                    role = parts[2]
                if len(parts) > 3 and parts[1] == "-terastallize" and role and parts[2].startswith(role) and tera_turn is None:
                    tera_turn = turn
                if len(parts) > 2 and parts[1] == "turn":
                    turn = int(parts[2])
        play(player, events)
        summary = player.game_summary(events[0]["room"])
        assert role in ("p1", "p2") and tera_turn is not None           # the recorded player did Terastallize
        assert summary["tera"] is True and summary["tera_turn"] == tera_turn and summary["tera_type"]
        other = min(load_fixture(), key=lambda e: len(orders_of(e)))    # a battle without any Terastallization
        player2 = searcher(classes, other[-1]["username"], FakeBridge([other]), mode="search")
        play(player2, other)
        summary2 = player2.game_summary(other[0]["room"])
        assert summary2["tera"] is False and summary2["tera_turn"] is None and summary2["tera_by"] is None
    finally:
        bridge.close()


@needs_player
def test_a_decision_that_raises_is_played_at_random_counted_and_noted(classes):
    events = min(load_fixture(), key=lambda e: len(orders_of(e)))
    bridge = FakeBridge([events])
    try:
        player = searcher(classes, events[-1]["username"], bridge, mode="mix")
        real = player.assembler.assemble
        calls = {"n": 0}

        def sometimes(battle, events=None):
            calls["n"] += 1
            if calls["n"] in (2, 5):
                raise RuntimeError("the observation could not be built")
            return real(battle, events=events)

        player.assembler.assemble = sometimes
        play(player, events)
        player.assembler.assemble = real
        assert player.search_totals["exceptions"] == 2
        assert len(chosen(player)) == len(orders_of(events)) and player.n_finished_battles == 1
        check_notes(player, bridge, events)                       # the random orders were noted as well
        assert "2 decisions raised" in player.search_summary()
    finally:
        bridge.close()


@needs_player
def test_an_order_the_player_did_not_choose_is_noted_when_it_is_sent(classes):
    """players.ChoiceAuditMixin forces a random order after repeated rejections, past choose_move."""
    from players import audited

    events = min(load_fixture(), key=lambda e: len(orders_of(e)))
    bridge = FakeBridge([events])
    try:
        cls = audited(classes["search"])
        player = offline(cls, events[-1]["username"], mode="search", bridge=bridge, search_ms=20, deadline=0.5,
                         warm_up=False, mix=MixSettings(min_iterations=0))
        room = events[0]["room"]
        messages = [e for e in events if e["t"] == "msg"]
        cut = next(i for i, e in enumerate(messages) if "|request|" in e["data"] and '"active"' in e["data"]) + 1

        async def go():
            for event in messages[:cut]:
                await player._handle_battle_message(split(event["room"], event["data"]))
            first = len(bridge.notes)
            assert first == 1 and len(chosen(player)) == 1
            # The server rejects that order three times: twice the player is asked again, then a random order is forced.
            for _ in range(3):
                await player._handle_battle_message(split(room, "|error|[Invalid choice] Can't move: something"))
            return first

        first = on_loop(player, go())
        assert player.audit["invalid_choices"] == 3 and player.audit["forced_random"] == 1
        assert len(chosen(player)) == 4 and len(bridge.notes) == first + 3          # every order sent was noted
        assert player.search_totals["noted_at_send"] == 1
        forced, told = chosen(player)[-1].split(), bridge.notes[-1][1].split()
        assert forced[1] == told[0] and ("terastallize" in forced) == ("terastallize" in told)
        # "/choose default" from poke-env itself is noted as such, and a message that is no order is not noted.
        on_loop(player, player.ps_client.send_message("/choose default", room))
        on_loop(player, player.ps_client.send_message("/timer on", room))
        on_loop(player, player.ps_client.send_message("/choose default", "battle-gen9randombattle-999999"))
        assert bridge.notes[-1] == (key_of(events), "default") and len(bridge.notes) == first + 4
    finally:
        bridge.close()


@needs_player
def test_a_request_after_the_battle_has_ended_sends_nothing(classes):
    events = min(load_fixture(), key=lambda e: len(orders_of(e)))
    bridge = FakeBridge([events])
    try:
        player = searcher(classes, events[-1]["username"], bridge, mode="search")
        play(player, events)
        battle = player.battles[events[0]["room"]]
        assert battle.finished
        before = (len(player.sent), len(bridge.notes), len(bridge.asked))
        order = on_loop(player, player.choose_move(battle))
        assert order.message == ""                                # poke-env sends nothing for an empty message
        on_loop(player, player._handle_battle_request(battle, maybe_default_order=True))
        assert (len(player.sent), len(bridge.notes), len(bridge.asked)) == before
        assert player.search_totals["after_end"] >= 1
    finally:
        bridge.close()


@needs_player
def test_warm_up_runs_the_network_once_and_leaves_no_trace(classes):
    cls = classes["search"]
    before = cls.forward_calls
    player = offline(cls, "warmupper", mode="net", bridge=None, warm_up=True, max_concurrent_battles=4)
    assert cls.forward_calls == before + 2                         # one battle, then four at once
    assert player.warm_up_ms is not None and player.warm_up_ms >= 0
    assert player.batch_stats == {"forwards": 0, "rows": 0, "largest": 0} and player.policy._cache == {}
    quiet = offline(cls, "nowarmup", mode="net", bridge=None, warm_up=False)
    assert cls.forward_calls == before + 2 and quiet.warm_up_ms is None


@needs_player
def test_the_trust_flag_guards_a_deep_battle_as_the_mix_trust_arm_does(classes):
    # The app's guarded mode searched wider: --player mix-deep with --trust 0.03 plays every recorded
    # battle move for move as the arm mix-trust (the fake search answers the same at any budget), and
    # its game summary says so.
    battles = load_fixture()
    budget = sp.SearchBudget(search_ms=40, worlds_x=2, deadline=1.0)
    played = {}
    for name, mode, settings in (("arm", "mix-trust", MixSettings(min_iterations=0)),
                                 ("flag", sp.DEEP_ARM, MixSettings(min_iterations=0, trust=0.03)),
                                 ("plain", sp.DEEP_ARM, MixSettings(min_iterations=0))):
        bridge = FakeBridge(battles)
        try:
            orders, summaries = [], []
            for events in battles:
                player = searcher(classes, events[-1]["username"], bridge, mode=mode, mix=settings, budgets={sp.DEEP_ARM: budget})
                play(player, events)
                orders.append(chosen(player))
                summaries.append(player.game_summary(events[0]["room"]))
            played[name] = (orders, summaries)
        finally:
            bridge.close()
    assert played["flag"][0] == played["arm"][0]
    assert played["flag"][0] != played["plain"][0]          # the guard changed moves in these battles
    assert all(s["arm"] == sp.DEEP_ARM and s["trust"] == 0.03 and s["budget"]["worlds_x"] == 2 for s in played["flag"][1])
    assert all(s["arm"] == "mix-trust" and s["trust"] == 0.03 for s in played["arm"][1])
    assert all("trust" not in s for s in played["plain"][1])


@needs_player
def test_a_deadline_too_short_for_the_search_time_is_refused(classes):
    bridge = FakeBridge([])
    try:
        with pytest.raises(ValueError, match="deadline"):
            offline(classes["search"], "tooshort", mode="search", bridge=bridge, search_ms=1000, deadline=2.5, warm_up=False)
        with pytest.raises(ValueError):
            offline(classes["search"], "badmode", mode="tree", bridge=bridge, warm_up=False)
        offline(classes["search"], "fits", mode="search", bridge=bridge, search_ms=500, deadline=2.5, warm_up=False)
    finally:
        bridge.close()


def test_importing_the_module_does_not_import_torch():
    code = "import sys; import search_player, ladder, ladder_graph; print('torch' in sys.modules, 'poke_env' in sys.modules)"
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split()[0] == "False"
