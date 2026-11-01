"""
The set-up fixes of the search player (search_player.MixSettings.no_third, sure_attack, tera_own,
setup_defer; search_player.ThinkLonger) and their ladder flags. bench/notes/setup_fix.md says
what they are for.

    python -m pytest tests/test_setup_fix.py -q -p no:cacheprovider

The rule tests need nothing. The player tests replay the recorded battles of
tests/fixtures/foulplay_bridge_trace.jsonl.gz through the real player with the fake bridge and
the stub network of tests/test_search_player.py.
"""

from __future__ import annotations

import concurrent.futures
import os
import random
import sys
import time
from types import SimpleNamespace

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (REPO, os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

import ladder  # noqa: E402
import search_player as sp  # noqa: E402
import setup_moves as sm  # noqa: E402
from search_player import MixSettings, Threat, ThinkLonger, mix  # noqa: E402
from test_search_player import (FakeBridge, chosen, classes, load_fixture, mask_of, needs_player,  # noqa: E402,F401
                                orders_of, play, searcher, spread, view_of)

FIX = MixSettings(no_third=True, tera_own=True, sure_attack=0.7)


# =============================================================================================
# The rule
# =============================================================================================

def test_with_every_fix_off_mix_plays_what_it_played_before():
    rng = random.Random(7)
    for _ in range(3000):
        legal = sorted(rng.sample(range(14), rng.randint(2, 9)))
        mask = mask_of(*legal)
        raw = [rng.random() ** 3 if a in legal else 0.0 for a in range(14)]
        probs = [x / sum(raw) for x in raw]
        shares = [rng.random() ** 2 if a in legal and rng.random() < 0.8 else 0.0 for a in range(14)]
        if not any(shares):
            shares[legal[0]] = 1.0
        view = view_of(shares)
        threat = Threat(boost=rng.randint(0, 3), setup_read=rng.random(), passive=tuple(a for a in legal if a < 4 and rng.random() < 0.3),
                        ko=tuple(a for a in legal if a < 4 and rng.random() < 0.3))
        plain = mix(probs, mask, view, MixSettings())
        assert mix(probs, mask, view, MixSettings(), threat=threat) == plain
        assert not MixSettings().fixes and MixSettings(no_third=True, rule="pool").fixes is False
        fixed = mix(probs, mask, view, MixSettings(no_third=True, tera_own=True, sure_attack=0.7, setup_defer=True), threat=threat)
        assert mask[fixed] == 1                                              # always legal


def test_no_third_plays_the_searchs_best_when_the_networks_move_is_vetoed():
    mask = mask_of(0, 1, 2, 4)
    probs = spread(a0=0.6, a1=0.3, a2=0.05, a4=0.05)           # the network wants 0, then 1
    view = view_of(spread(a0=0.05, a1=0.3, a2=0.6, a4=0.05))   # the search turns away from 0, likes 2, keeps 1
    assert mix(probs, mask, view, MixSettings()) == 1           # the plain veto: a third action neither side picked
    explain = {}
    assert mix(probs, mask, view, MixSettings(no_third=True), explain=explain) == 2
    assert explain["fix"] == "no third action"
    # The network's own move not vetoed: no change.
    probs = spread(a0=0.2, a1=0.7, a2=0.05, a4=0.05)
    assert mix(probs, mask, view, MixSettings(no_third=True)) == mix(probs, mask, view, MixSettings()) == 1


def test_a_sure_attack_in_ko_range_is_not_vetoed_by_our_own_set_up_move():
    mask = mask_of(0, 1, 2, 3)
    probs = spread(a0=0.8, a1=0.1, a2=0.05, a3=0.05)           # 0: Shadow Sneak, the network is sure
    view = view_of(spread(a0=0.1, a1=0.7, a2=0.1, a3=0.1))     # 1: Swords Dance, the search's favourite
    threat = Threat(passive=(1,), ko=(0,))
    assert mix(probs, mask, view, MixSettings(), threat=threat) == 1
    explain = {}
    assert mix(probs, mask, view, FIX, threat=threat, explain=explain) == 0
    assert explain["fix"] == "sure attack in KO range"
    # Not in KO range, not sure enough, or the search's favourite an attack: the veto as before.
    assert mix(probs, mask, view, FIX, threat=Threat(passive=(1,), ko=())) == 1
    assert mix(spread(a0=0.6, a1=0.3, a2=0.05, a3=0.05), mask, view, FIX, threat=threat) == 1
    assert mix(probs, mask, view, FIX, threat=Threat(passive=(3,), ko=(0,))) == 1
    # A sure attack whose Tera form is the one in KO range counts too.
    mask = mask_of(0, 1, 10, 11)
    probs = spread(a0=0.45, a10=0.4, a1=0.1, a11=0.05)
    view = view_of(spread(a0=0.03, a10=0.05, a1=0.8, a11=0.12))
    assert sp.base_action(mix(probs, mask, view, FIX, threat=Threat(passive=(1, 11), ko=(10,)))) == 0


def test_tera_only_when_one_side_picked_the_tera_form():
    mask = mask_of(0, 1, 10, 11)
    probs = spread(a0=0.5, a10=0.1, a1=0.3, a11=0.1)           # the network: move 0, plain
    view = view_of(spread(a0=0.2, a10=0.25, a1=0.5, a11=0.05)) # the search: move 1; of move 0's forms it likes Tera
    assert mix(probs, mask, view, MixSettings()) == 10          # the plain veto adds Tera that neither side picked
    assert mix(probs, mask, view, MixSettings(tera_own=True)) == 0
    # The network's own pick is the Tera form: kept.
    probs = spread(a0=0.2, a10=0.4, a1=0.3, a11=0.1)
    assert mix(probs, mask, view, MixSettings(tera_own=True)) == 10
    # The search's best is the Tera form: kept.
    view = view_of(spread(a0=0.2, a10=0.6, a1=0.15, a11=0.05))
    assert mix(spread(a0=0.5, a10=0.1, a1=0.3, a11=0.1), mask, view, MixSettings(tera_own=True)) == 10


def test_setup_defer_plays_the_searchs_best_in_a_set_up_position():
    mask = mask_of(0, 1, 4, 5)
    probs = spread(a0=0.1, a1=0.05, a4=0.85)                    # the network switches
    view = view_of(spread(a0=0.6, a1=0.1, a4=0.2, a5=0.1))      # the search attacks, and keeps the switch
    settings = MixSettings(setup_defer=True)
    assert mix(probs, mask, view, settings, threat=Threat()) == 4             # no set-up: the veto as before
    explain = {}
    assert mix(probs, mask, view, settings, threat=Threat(boost=2), explain=explain) == 0
    assert explain["fix"] == "set-up: the search's best"
    assert mix(probs, mask, view, settings, threat=Threat(boost=1, last_setup=True)) == 0
    assert mix(probs, mask, view, settings, threat=Threat(setup_read=0.5)) == 0
    assert mix(probs, mask, view, settings, threat=Threat(boost=1)) == 4      # +1 alone, no set-up seen: not a set-up position
    # A switch the network is sure of is played.
    assert mix(probs, mask, view, MixSettings(setup_defer=True, setup_switch_sure=0.8), threat=Threat(boost=2)) == 4
    assert mix(probs, mask, view, MixSettings(setup_defer=True, setup_switch_sure=0.9), threat=Threat(boost=2)) == 0


def test_the_guard_still_decides_when_it_fires():
    mask = mask_of(0, 1, 2)
    probs = spread(a0=0.2, a1=0.7, a2=0.1)
    view = view_of(spread(a0=0.5, a1=0.3, a2=0.2))
    threat = Threat(boost=3, passive=(1,))
    guarded = MixSettings(guard=True)
    assert mix(probs, mask, view, replace_fixes(guarded), threat=threat) == mix(probs, mask, view, guarded, threat=threat) == 0


def replace_fixes(settings):
    from dataclasses import replace
    return replace(settings, no_third=True, tera_own=True, sure_attack=0.7, setup_defer=True)


# =============================================================================================
# The pieces
# =============================================================================================

class _Type:
    def __init__(self, name):
        self.name = name


def mon(level=80, stats=None, base=None, boosts=None, types=("NORMAL",), mult=1.0, hp=1.0, status=None):
    out = SimpleNamespace(level=level, stats=stats or {}, base_stats=base or {}, boosts=boosts or {}, types=[_Type(t) for t in types],
                          current_hp_fraction=hp, status=status, is_terastallized=False, tera_type=None)
    out.damage_multiplier = lambda move: mult
    return out


def move(power=80, category="PHYSICAL", type_="NORMAL", damage=None):
    return SimpleNamespace(base_power=power, category=_Type(category), type=_Type(type_), damage=damage)


def test_ko_estimate_is_the_damage_formula_at_the_lowest_roll():
    base = {"hp": 100, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100}
    ours = mon(stats={"atk": 250, "spa": 150}, types=("GHOST",))
    theirs = mon(base=base)
    # Their stats as a random battle set at level 80 (84 EVs, 31 IVs): HP 291, Defence 206.
    assert sp._stat(100, 80, hp=True) == 291 and sp._stat(100, 80) == 206
    expected = (int(int(int(2 * 80 / 5 + 2) * 40 * 250 / 206) / 50) + 2) * 1.5 * 0.85 / 291     # Shadow Sneak with STAB
    assert sp.ko_estimate(move(40, type_="GHOST"), ours, theirs) == pytest.approx(expected)
    assert sp.ko_estimate(move(40, type_="NORMAL"), ours, theirs) == pytest.approx(expected / 1.5)
    # Their +2 Defence halves it about; our burn halves a physical move; immune is 0; status moves and unknown stats say nothing.
    assert sp.ko_estimate(move(40, type_="GHOST"), ours, mon(base=base, boosts={"def": 2})) < 0.55 * expected
    assert sp.ko_estimate(move(40, type_="GHOST"), mon(stats={"atk": 250}, types=("GHOST",), status=_Type("BRN")), theirs) == pytest.approx(expected / 2, rel=0.05)
    assert sp.ko_estimate(move(40), ours, mon(base=base, mult=0.0)) == 0.0
    assert sp.ko_estimate(move(0, "STATUS"), ours, theirs) is None
    assert sp.ko_estimate(move(80), mon(stats={}), theirs) is None
    assert sp.ko_estimate(move(80), ours, mon(base={})) is None
    assert sp.ko_estimate(move(0, "SPECIAL", damage="level"), ours, theirs) == pytest.approx(80 / 291)
    assert sp.ko_estimate(None, None, None) is None


def test_ko_doubt_leaves_out_kos_that_items_abilities_or_full_hp_can_stop():
    base = {"hp": 100, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100}

    def them(hp=0.4, item="", ability="", possible=(), evos=None, species="x"):
        out = mon(base=base, hp=hp)
        out.item, out.ability, out.possible_abilities, out.species = item, ability, list(possible), species
        out._data = SimpleNamespace(pokedex={species: {"evos": evos} if evos else {}})
        return out

    assert sp.ko_doubt(move(80), them()) == ""
    assert sp.ko_doubt(move(80), them(hp=1.0)) == "full HP"                             # Sturdy, Sash, Multiscale
    assert sp.ko_doubt(move(80, "SPECIAL"), them(item="assaultvest")) == "Assault Vest"
    assert sp.ko_doubt(move(80, "PHYSICAL"), them(item="assaultvest")) == ""
    assert sp.ko_doubt(move(80), them(item="eviolite")) == "Eviolite"
    assert sp.ko_doubt(move(80), them(item="unknown_item", evos=["y"])) == "Eviolite"    # it still evolves, item not seen
    assert sp.ko_doubt(move(80), them(item="leftovers", evos=["y"])) == ""
    assert sp.ko_doubt(move(80, type_="GROUND"), them(possible=("levitate", "overcoat"))) == "levitate"
    assert sp.ko_doubt(move(80, type_="GROUND"), them(ability="overcoat", possible=("levitate", "overcoat"))) == ""   # known
    assert sp.ko_doubt(move(80, type_="FIRE"), them(possible=("thickfat",))) == "thickfat"
    assert sp.ko_doubt(move(80, "PHYSICAL"), them(possible=("furcoat",))) == "furcoat"
    assert sp.ko_doubt(move(80), them(ability="wonderguard")) == "wonderguard"
    assert sp.ko_doubt(None, None) in ("", "unknown") and sp.ko_doubt(move(80), None) == ""


def test_time_left_and_set_up_shares():
    assert sp.parse_time_left("Time left: 150 sec this turn | 150 sec total") == 150.0
    assert sp.parse_time_left("Time left: 40 sec this turn | 95 sec total | 30 sec grace") == 40.0
    assert sp.parse_time_left("Battle timer is ON: inactive players will automatically lose when time's up.") is None
    assert sp.parse_time_left("someplayer has 60 seconds to reconnect!") is None
    assert sm.setup_share([("move swordsdance", 0.5), ("move earthquake", 0.3), ("switch garchomp", 0.2)]) == pytest.approx(0.5)
    assert sm.setup_share([("move swordsdance terastallize", 1.0, True), ("move earthquake", 1.0, True)]) == pytest.approx(0.5)
    assert sm.setup_share([("move curse", 1.0)], ["GHOST"]) == 0.0 and sm.setup_share([("move curse", 1.0)], ["WATER"]) == 1.0
    assert sm.setup_share([]) == 0.0 and sm.setup_share([("bad",)]) == 0.0
    assert sm.largest_boost(SimpleNamespace(boosts={"atk": 2, "spe": -1, "accuracy": 3})) == 2
    assert sm.largest_boost(SimpleNamespace(boosts={"def": -1})) == 0 and sm.largest_boost(None) == 0


def test_think_longer_waits_long_enough_for_its_search():
    assert ThinkLonger().wait_s() == pytest.approx(sp.minimum_deadline(2000) + 1.0)
    assert ThinkLonger(ms=1500, deadline=5.0).wait_s() == 5.0


def test_ladder_flags(tmp_path, monkeypatch):
    args = ladder.argparse.Namespace(mix_rule="veto", guard=False, veto_fix=False, sure_attack=0.7, setup_defer=False,
                                     setup_switch_sure=0.0, think_longer=0, think_longer_read=0.4, think_longer_boost=1,
                                     think_longer_threads=2)
    assert ladder.mix_settings(args) == MixSettings() and ladder.think_longer(args) is None and ladder.setup_fix_text(args) == ""
    args.veto_fix, args.setup_defer, args.setup_switch_sure, args.think_longer = True, True, 0.8, 2000
    s = ladder.mix_settings(args)
    assert s.no_third and s.tera_own and s.sure_attack == 0.7 and s.setup_defer and s.setup_switch_sure == 0.8 and s.fixes
    assert ladder.think_longer(args) == ThinkLonger(ms=2000, read=0.4, boost=1, threads=2)
    assert "veto fix on" in ladder.setup_fix_text(args) and "2000 ms" in ladder.setup_fix_text(args)
    for name in ("OUT", "GRAPH", "BATTLES"):
        monkeypatch.setattr(ladder, name, getattr(ladder, name))
    out = str(tmp_path / "results.jsonl")
    assert ladder.main(["--history", "--server", "local:1", "--out", out, "--player", "mix", "--veto-fix", "--setup-defer",
                        "--think-longer", "2000", "--think-longer-read", "0.5"]) == 0


# =============================================================================================
# The player
# =============================================================================================

class SlowSecondBridge(FakeBridge):
    """The recorded answer to the ordinary question; a long question (search_ms above 20) answered after `long_s`, or never."""

    def __init__(self, battles, long_s=None, helpers=3):
        super().__init__(battles)
        self.long_s = long_s
        self.helpers = helpers

    def advice_future(self, key, search_ms=None, rqid=None, ttl=None, **more):
        if search_ms and search_ms > 20 and self.long_s is None:
            self.asked.append((key, rqid, ttl, search_ms, more))
            return concurrent.futures.Future()               # never answers
        future = super().advice_future(key, search_ms, rqid, ttl, **more)
        if search_ms and search_ms > 20 and self.long_s:
            slow = concurrent.futures.Future()

            def later():
                time.sleep(self.long_s)
                try:
                    slow.set_result(future.result())
                except Exception as exc:        # noqa: BLE001
                    slow.set_exception(exc)

            import threading
            threading.Thread(target=later, daemon=True).start()
            return slow
        return future


def _first_battle():
    return min(load_fixture(), key=lambda e: len(orders_of(e)))


@needs_player
def test_the_long_search_is_asked_before_on_a_set_up_turn_and_used(classes, monkeypatch):
    events = _first_battle()
    bridge = SlowSecondBridge([events], long_s=0.05)
    try:
        player = searcher(classes, events[-1]["username"], bridge, mode="mix", deadline=0.5,
                          think_longer=ThinkLonger(ms=60, deadline=0.8, threads=2))
        calls = []
        monkeypatch.setattr(player, "_long_before", lambda battle: calls.append(battle.turn) or ("boosted +2" if battle.turn % 2 else ""))
        play(player, events)
        decisions = player.decisions_of(events[0]["room"])
        assert len(decisions) == len(orders_of(events)) == len(chosen(player))
        longs = [d for d in decisions if "long" in d]
        assert longs and all(d["long"]["when"] == "before" and d["long"]["ms"] == 60 and d["turn"] % 2 for d in longs)
        assert all(d["long"]["used"] for d in longs if d["search"] is not None)
        assert all(0.8 <= d.get("deadline_s", 0) < 0.95 for d in longs)          # the long wait, from when it was asked
        asked_long = [a for a in bridge.asked if a[3] == 60]
        assert len(asked_long) == len(longs)
        # One battle on three helpers: a helper is free, so the long search has the second thread.
        assert all(a[4].get("threads") == 2 for a in asked_long) and all(d["long"]["threads"] == 2 for d in longs)
        assert player._long_threads_used == 0                                          # given back
        summary = player.game_summary(events[0]["room"])
        assert summary["long_asked"] == len(longs) and summary["long_used"] == sum(bool(d["long"]["used"]) for d in longs)
        assert "longer search on set-up turns" in player.search_summary()
    finally:
        bridge.close()


@needs_player
def test_a_second_thread_is_given_back_when_the_network_raises(classes, monkeypatch):
    events = _first_battle()
    bridge = SlowSecondBridge([events], long_s=0.05)
    try:
        player = searcher(classes, events[-1]["username"], bridge, mode="mix", deadline=0.5,
                          think_longer=ThinkLonger(ms=60, deadline=0.8, threads=2))
        monkeypatch.setattr(player, "_long_before", lambda battle: "boosted +2" if battle.turn % 2 else "")
        real = player._queue_inference
        raised = []

        async def flaky(tag, obs):
            if player._long_threads_used and not raised:
                raised.append(True)
                raise RuntimeError("inference failed")
            return await real(tag, obs)

        monkeypatch.setattr(player, "_queue_inference", flaky)
        play(player, events)
        assert raised and player.search_totals["exceptions"] == 1                       # a random legal order, counted
        assert player._long_threads_used == 0                                          # the thread came back
        asked_long = [a for a in bridge.asked if a[3] == 60]
        assert len(asked_long) >= 2 and asked_long[-1][4].get("threads") == 2           # later long searches still get it
    finally:
        bridge.close()


@needs_player
def test_a_set_up_read_asks_again_and_a_late_long_answer_costs_nothing(classes, monkeypatch):
    events = _first_battle()
    bridge = SlowSecondBridge([events], long_s=None, helpers=1)          # the long search never answers; no helper free
    try:
        player = searcher(classes, events[-1]["username"], bridge, mode="mix", deadline=0.5,
                          think_longer=ThinkLonger(ms=60, deadline=0.6, threads=2))
        monkeypatch.setattr(player, "_long_before", lambda battle: "")
        monkeypatch.setattr(player, "_read_of_setup", lambda battle, advice: 0.9)
        began = time.monotonic()
        play(player, events)
        took = time.monotonic() - began
        decisions = player.decisions_of(events[0]["room"])
        assert len(decisions) == len(orders_of(events)) == len(chosen(player))
        searched = [d for d in decisions if d["search"] is not None]
        assert searched and all("long" in d and d["long"]["when"] == "after" for d in searched)
        for d in searched:
            assert d["long"]["used"] is False and d["long"]["why_not"] == "timeout" and d["long"]["threads"] == 1
            assert d["ms"] <= 1000.0 * (0.5 + 0.6) + 300.0                       # the ordinary deadline, then the long one
        # The ordinary answer was played: what a player without the longer search plays.
        plain = searcher(classes, events[-1]["username"], FakeBridge([events]), mode="mix", deadline=0.5)
        play(plain, events)
        assert [d["played"] for d in decisions] == [d["played"] for d in plain.decisions_of(events[0]["room"])]
        assert took < 60.0
    finally:
        bridge.close()


@needs_player
def test_a_long_answer_after_the_ordinary_one_replaces_it(classes, monkeypatch):
    events = _first_battle()
    bridge = SlowSecondBridge([events], long_s=0.02)
    try:
        player = searcher(classes, events[-1]["username"], bridge, mode="search", deadline=0.5,
                          think_longer=ThinkLonger(ms=60, deadline=0.6))
        monkeypatch.setattr(player, "_long_before", lambda battle: "")
        monkeypatch.setattr(player, "_read_of_setup", lambda battle, advice: 0.9)
        play(player, events)
        searched = [d for d in player.decisions_of(events[0]["room"]) if d["search"] is not None]
        assert searched and all(d["long"]["used"] and "short_action" in d["long"] and "changed" in d["long"] for d in searched)
        assert all(d["played"] == d["search"] for d in searched)
    finally:
        bridge.close()


@needs_player
def test_no_long_search_when_the_timer_is_short(classes, monkeypatch):
    events = _first_battle()
    bridge = SlowSecondBridge([events], long_s=0.01)
    try:
        player = searcher(classes, events[-1]["username"], bridge, mode="mix", deadline=0.5,
                          think_longer=ThinkLonger(ms=60, deadline=0.6, min_time_left=30.0))
        monkeypatch.setattr(player, "_long_before", lambda battle: "boosted +2")
        room = events[0]["room"]
        monkeypatch.setattr(player, "_time_left", type(player._time_left)())
        real = player._long_allowed

        def short_timer(battle):
            player._time_left[battle.battle_tag] = 20.0          # as if the server said 20 s are left
            return real(battle)

        monkeypatch.setattr(player, "_long_allowed", short_timer)
        play(player, events)
        assert not any("long" in d for d in player.decisions_of(room))
        assert not any(a[3] == 60 for a in bridge.asked)
    finally:
        bridge.close()


@needs_player
def test_the_fixes_in_play_record_what_the_plain_veto_would_have_played(classes):
    battles = load_fixture()
    bridge = FakeBridge(battles)
    changed = 0
    try:
        for events in battles:
            player = searcher(classes, events[-1]["username"], bridge, mode="mix",
                              mix=MixSettings(min_iterations=0, no_third=True, tera_own=True, sure_attack=0.7))
            play(player, events)
            decisions = player.decisions_of(events[0]["room"])
            assert len(decisions) == len(orders_of(events)) == len(chosen(player))
            for d in decisions:
                if "fix" in d:
                    changed += 1
                    assert d["fix"]["veto"] != d["played"] and d["fix"]["fix"] in ("no third action", "sure attack in KO range", "tera")
                if d["search"] is not None:
                    # Never a third action: the move played is the network's (its likeliest move, plain and
                    # Tera forms together) or the search's.
                    merged = sp.merged_prior(d["probs"], d["mask"])
                    net_move = max((a for a in range(14) if d["mask"][a]), key=lambda a: (merged[a], d["shares"][a], -a))
                    assert sp.base_action(d["played"]) in (sp.base_action(net_move), sp.base_action(d["search"])), d
            assert player.game_summary(events[0]["room"])["fix_changed"] == sum("fix" in d for d in decisions)
        assert changed > 0
    finally:
        bridge.close()


def test_arm_fixes_modes():
    """The measurement arms (search_player.ARM_FIXES): the veto rule with the fixes, the long flag per arm."""
    import search_player as sp

    for mode in sp.ARM_FIXES:
        assert mode in sp.MODES
        assert sp.split_mode(mode) == ("mix", "veto")
    assert sp.check_modes(["mix", "mix-fixdefer"]) == ["mix", "mix-fixdefer"]
    kw, long_ = sp.ARM_FIXES["mix-fixdefer-long"]
    assert long_ and kw["setup_defer"] and kw["no_third"] and kw["tera_own"] and kw["sure_attack"] == 0.70
    assert not sp.ARM_FIXES["mix-fix"][1] and "setup_defer" not in sp.ARM_FIXES["mix-fix"][0]
    s = sp.MixSettings(**sp.ARM_FIXES["mix-fixdefer"][0])
    assert s.fixes and s.setup_defer
    import ladder

    assert set(sp.ARM_FIXES) <= set(ladder.PLAYER_MODES)
