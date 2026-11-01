"""
Live predictions, the guard against set-up moves and sleep, and the set-up test opponent.

    python -m pytest tests/test_predict_live.py -q -p no:cacheprovider

No Foul Play, no Showdown server and no checkpoint: the pure parts on hand-made numbers, the
player on the seven recorded battles of tests/fixtures/foulplay_bridge_trace.jsonl.gz with the
fake bridge of tests/test_opponent_model.py (an opponent's side in every answer).
"""

from __future__ import annotations

import importlib.util
import os
import random
import sys
from types import SimpleNamespace

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import search_player as sp  # noqa: E402
import setup_moves as sm  # noqa: E402
from search_player import MixSettings, SearchView, Threat  # noqa: E402

_spec = importlib.util.spec_from_file_location("opponent_model_tests", os.path.join(REPO, "tests", "test_opponent_model.py"))
t_om = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t_om)
h = t_om.h
classes = h.classes
needs_player = h.needs_player


def view_of(shares):
    return SearchView(share=list(shares), visits=[1 if s > 0 else 0 for s in shares], score=[None] * 14,
                      best=max(range(14), key=lambda a: shares[a]), iterations=100_000)


def numbers(**kw):
    """probs and shares over 14 actions from a0=..., s0=... (the rest 0), and the mask of the actions given."""
    probs, shares, mask = [0.0] * 14, [0.0] * 14, [0.0] * 14
    for key, value in kw.items():
        a = int(key[1:])
        (probs if key[0] == "p" else shares)[a] = value
        mask[a] = 1.0
    return probs, shares, mask


# =============================================================================================
# The moves
# =============================================================================================

def test_setup_sleep_passive_and_names_come_from_the_move_data():
    assert sm.raised_stats("swordsdance") == {"atk": 2} and sm.raised_stats("shellsmash") == {"atk": 2, "spa": 2, "spe": 2}
    for move in ("swordsdance", "dragondance", "quiverdance", "calmmind", "nastyplot", "curse", "shellsmash", "bulkup", "bellydrum",
                 "agility", "tidyup", "filletaway", "victorydance", "cosmicpower"):
        assert sm.is_setup(move), move
    assert not sm.is_setup("curse", ["GHOST", "FIRE"]) and sm.is_setup("curse", ["WATER"])
    for move in ("closecombat", "earthquake", "willowisp", "haze", "roost", "stealthrock", "spore"):
        assert not sm.is_setup(move), move
    for move in ("spore", "sleeppowder", "hypnosis", "sing", "lovelykiss", "grasswhistle", "darkvoid", "yawn"):
        assert sm.is_sleep(move), move
    assert not sm.is_sleep("toxic") and not sm.is_sleep("relicsong")
    for move in ("swordsdance", "roost", "recover", "protect", "substitute", "stealthrock", "toxicspikes", "stickyweb", "reflect",
                 "trickroom", "defog", "rest", "wish"):
        assert sm.is_passive(move), move
    for move in ("haze", "perishsong", "whirlwind", "roar", "encore", "taunt", "willowisp", "thunderwave", "toxic", "sleeptalk",
                 "earthquake", "clearsmog", "suckerpunch"):
        assert not sm.is_passive(move), move
    assert sm.fails_on_status("suckerpunch") and sm.fails_on_status("thunderclap") and not sm.fails_on_status("aquajet")
    assert sm.move_name("sleeppowder") == "Sleep Powder" and sm.species_name("ursalunabloodmoon") == "Ursaluna-Bloodmoon"
    assert sm.move_name("notamove") == "notamove" and not sm.is_setup("notamove") and not sm.is_passive("")


# =============================================================================================
# The words
# =============================================================================================

def test_the_prediction_line_and_the_read_after_the_turn():
    assert sp.short_tag("battle-gen9randombattle-2694021408") == "1408"
    assert sp.short_tag("battle-gen9randombattle-2694004575-0123456789abcdefghijklmnopqrstuvpw") == "4575"
    top = [["move sleeppowder", 0.581], ["switch lurantis", 0.22], ["move sludgebomb", 0.12]]
    line = sp.prediction_line(7, "Venusaur", top, "Meteor Mash")
    assert line == "T7 Venusaur: predict Sleep Powder (sleep) 58%, switch Lurantis 22%, Sludge Bomb 12% | we play Meteor Mash"
    guarded = sp.prediction_line(9, "Dondozo", [["move curse", 0.6]], "Volt Switch",
                                 {"guard": "boosted +2", "changed": True, "veto_text": "Toxic Spikes"})
    assert guarded.endswith("| we play Volt Switch [guard: boosted +2, instead of Toxic Spikes]") and "Curse (set-up) 60%" in guarded
    assert sp.option_text("switch ?") == "switch to one not seen yet" and sp.option_text("move gunkshot terastallize") == "Gunk Shot + Tera"

    names = ["move sleeppowder", "switch lurantis", "move sludgebomb"]
    move = {"type": "move", "name": "sludgebomb", "action": "move sludgebomb", "tera": False, "known": True, "locked": False}
    assert sp.outcome_text(names, "move sleeppowder", dict(move, name="sleeppowder", action="move sleeppowder")) == "read: right (Sleep Powder (sleep))"
    assert sp.outcome_text(names, "move sludgebomb", move) == "read: wrong, it played Sludge Bomb (our 3rd guess)"
    switch = {"type": "switch", "name": "lurantis", "action": "switch lurantis", "tera": False, "known": True}
    assert sp.outcome_text(names, "switch lurantis", switch) == "read: wrong, it switched to Lurantis (our 2nd guess)"
    unseen = {"type": "switch", "name": "garchomp", "action": "switch garchomp", "tera": False, "known": True}
    assert sp.outcome_text(names, "switch ?", unseen) == "read: wrong, it switched to Garchomp"
    assert sp.outcome_text(names, "move earthquake", dict(move, name="earthquake", action="move earthquake", tera=True)) == \
        "read: wrong, it played Earthquake + Tera"
    assert sp.outcome_text(names, None, {"known": False, "why_unknown": "the game ended in this turn"}) == \
        "read: not scored (the game ended in this turn)"
    assert sp.outcome_text(names, None, dict(move, locked=True, name="outrage", action="move outrage")) == \
        "read: not scored, it was locked into Outrage"


# =============================================================================================
# The guard
# =============================================================================================

def test_the_guard_is_off_unless_asked_for_and_fires_on_boosts_reads_and_a_sure_network():
    assert not MixSettings().guard
    assert sp.guarded("mix-guard", MixSettings()) and not sp.guarded("mix", MixSettings())
    assert sp.guarded("mix", MixSettings(guard=True)) and not sp.guarded("mix-pool", MixSettings(guard=True))
    assert not sp.guarded("predict", MixSettings(guard=True)) and not sp.guarded("search", MixSettings(guard=True))
    s = MixSettings(guard=True)
    assert sp.guard_reason(None, s) == "" and sp.guard_reason(Threat(), s) == ""
    assert sp.guard_reason(Threat(boost=1), s) == ""                               # +1 alone: not yet
    assert sp.guard_reason(Threat(boost=1, last_setup=True), s) == "set up last turn"
    assert sp.guard_reason(Threat(boost=2), s) == "boosted +2"
    assert sp.guard_reason(Threat(setup_read=0.45), s) == "set-up read 45%"
    assert sp.guard_reason(Threat(sleep_read=0.61), s) == "sleep read 61%"
    assert sp.guard_reason(Threat(setup_read=0.39, sleep_read=0.39), s) == ""

    # With the guard off the threat changes nothing, whatever it says.
    probs, shares, mask = numbers(p0=0.79, p2=0.2, p3=0.01, s0=0.08, s2=0.14, s3=0.66, s8=0.12)
    threat = Threat(boost=4, passive=(3,))
    assert sp.mix(probs, mask, view_of(shares), MixSettings(), threat=threat) == sp.mix(probs, mask, view_of(shares)) == 3


def test_the_guard_on_the_positions_of_the_lost_games():
    on = MixSettings(guard=True)
    # 2694007454 T42: the network sure of Shadow Sneak (0), the search on our own Swords Dance (3).
    probs, shares, mask = numbers(p0=0.7907, p2=0.2078, p3=0.0015, p8=0.0, s0=0.0829, s2=0.1357, s3=0.6615, s8=0.12)
    assert sp.mix(probs, mask, view_of(shares)) == 3                                   # the plain veto: Swords Dance
    explain = {}
    assert sp.mix(probs, mask, view_of(shares), on, threat=Threat(passive=(3,)), explain=explain) == 0
    assert explain["guard"] == "network sure 79%" and explain["dropped"] == [3] and explain["by"] == "net"

    # 2693998716 T3, a forced switch in front of a +2 Fezandipiti: the network wanted Dedenne (5), the
    # search Spiritomb (7), the veto played Milotic (6), which neither wanted.
    probs, shares, mask = numbers(p5=0.6, p6=0.3, p7=0.1, s5=0.05, s6=0.25, s7=0.7)
    assert sp.mix(probs, mask, view_of(shares)) == 6
    explain = {}
    assert sp.mix(probs, mask, view_of(shares), on, threat=Threat(boost=2), explain=explain) == 7
    assert explain["by"] == "search" and explain["net_pick"] == 5

    # 2694007454 T7: Tera Toxic Spikes (12, a hazard) against a +2 Dondozo; Volt Switch (1) is the search's choice.
    probs, shares, mask = numbers(p1=0.3, p2=0.1, p11=0.0, p12=0.6, s1=0.6, s2=0.11, s11=0.0, s12=0.21)
    assert sp.mix(probs, mask, view_of(shares)) == 12
    assert sp.mix(probs, mask, view_of(shares), on, threat=Threat(boost=2, passive=(2, 12))) == 1

    # Sucker Punch (0) into a likely Agility: left out on a status read, kept on a boost alone.
    probs, shares, mask = numbers(p0=0.8, p1=0.2, s0=0.6, s1=0.4)
    assert sp.mix(probs, mask, view_of(shares), on, threat=Threat(boost=2, status_read=0.5, fails=(0,))) == 1
    assert sp.mix(probs, mask, view_of(shares), on, threat=Threat(boost=2, status_read=0.2, fails=(0,))) == 0

    # A sleep read alone leaves our passive moves in (Substitute blocks sleep).
    probs, shares, mask = numbers(p0=0.3, p1=0.7, s0=0.5, s1=0.5)
    assert sp.mix(probs, mask, view_of(shares), on, threat=Threat(sleep_read=0.7, passive=(1,))) == 1
    assert sp.mix(probs, mask, view_of(shares), on, threat=Threat(setup_read=0.7, passive=(1,))) == 0

    # Everything passive: nothing is left out.
    probs, shares, mask = numbers(p0=0.3, p1=0.7, s0=0.5, s1=0.5)
    assert sp.mix(probs, mask, view_of(shares), on, threat=Threat(boost=3, passive=(0, 1))) == 1


def test_the_guard_never_plays_an_illegal_action_or_one_neither_side_picked():
    rng = random.Random(7)
    on = MixSettings(guard=True)
    for _ in range(3000):
        mask = [1.0 if rng.random() < 0.6 else 0.0 for _ in range(14)]
        if sum(mask) < 1:
            mask[rng.randrange(14)] = 1.0
        legal = [a for a in range(14) if mask[a]]
        raw = [rng.random() ** 3 if mask[a] else 0.0 for a in range(14)]
        probs = [x / sum(raw) for x in raw]
        shares = [rng.random() ** 2 if mask[a] and rng.random() < 0.8 else 0.0 for a in range(14)]
        if not any(shares[a] > 0 for a in legal):
            shares[legal[0]] = 1.0
        threat = Threat(boost=rng.choice([0, 1, 2, 4]), last_setup=rng.random() < 0.3, setup_read=rng.random(), sleep_read=rng.random(),
                        status_read=rng.random(), passive=tuple(a for a in legal if a < 4 and rng.random() < 0.4),
                        fails=tuple(a for a in legal if a < 4 and rng.random() < 0.2))
        explain = {}
        action = sp.mix(probs, mask, view_of(shares), on, threat=threat, explain=explain)
        assert mask[action] == 1.0
        if explain.get("guard"):
            # The network's pick among what was left, or the search's: in their plain or Tera form.
            assert sp.base_action(action) in picks_of(explain)


def picks_of(guard):
    """The moves (Tera or not) or switches the guard may play: the network's pick and the search's."""
    return {sp.base_action(a) for a in (guard["net_pick"], guard["search_pick"]) if a is not None}


# =============================================================================================
# The player
# =============================================================================================

@needs_player
def test_mix_with_live_predictions_plays_the_same_moves_and_prints_and_scores_a_read_every_turn(classes):
    battles = h.load_fixture()
    bridge = t_om.PredictBridge(battles, opponent=[("move tackle", 0.6, None), ("move growl", 0.3, None), ("switch ?", 0.1, False)])
    made = scored = 0
    try:
        for events in battles:
            name, room = events[-1]["username"], events[0]["room"]
            plain = h.searcher(classes, name, bridge, mode="mix", mix=MixSettings(min_iterations=0))
            h.play(plain, events)
            bridge.notes.clear()
            outcomes, records = [], []
            player = h.searcher(classes, name, bridge, mode="mix", mix=MixSettings(min_iterations=0), live_predictions=True,
                                on_outcome=outcomes.append, on_decision=records.append)
            h.play(player, events)
            assert h.chosen(player) == h.chosen(plain)                      # predictions change no move
            h.check_notes(player, bridge, events)
            assert not any("payoff_ms" in a[4] for a in bridge.asked)       # and cost no extra search
            lines = [r for r in records if (r.get("predict") or {}).get("line") and not r["predict"].get("repeat")]
            for r in records:
                if r["search"] is None or r["forced"]:
                    assert "predict" not in r                                # no answer, or nothing for the opponent to choose
            for r in lines:
                line = r["predict"]["line"]
                assert line.startswith("T%d " % r["turn"]) and ": predict " in line and " | we play " in line
                assert 1 <= len(r["predict"]["top"]) <= 3 and r["predict"]["top"][0][1] >= r["predict"]["top"][-1][1]
            summary = player.game_summary(room)
            assert summary["pred_made"] == len(lines) > 0
            assert summary["pred_top1_right"] <= summary["pred_top3_right"] <= summary["pred_scored"] <= summary["pred_made"]
            assert "pred_reads" not in summary and "guard_fired" not in summary
            got = [o for o in outcomes if o["predict"].get("outcome")]
            assert len(got) == len(outcomes) and len(lines) - len(outcomes) <= 1
            for o in got:
                assert o["mode"] == "mix" and o["predict"]["outcome"]["text"].startswith("read: ")
            assert summary["pred_scored"] == sum(1 for o in got if o["predict"]["outcome"]["name"] is not None)
            assert room not in player._trackers and room not in player._predictions      # dropped when the battle ended
            made += summary["pred_made"]
            scored += summary["pred_scored"]
        assert made > 100 and scored > 0.8 * made
        assert player.search_totals["predict_errors"] == 0 and player.search_totals["trackers_broken"] == 0
        text = player.search_summary()
        assert "opponent predicted on" in text and "in the top 3" in text and "confident reads" not in text
    finally:
        bridge.close()


@needs_player
def test_without_live_predictions_mix_records_none(classes):
    battles = h.load_fixture()
    bridge = t_om.PredictBridge(battles)
    try:
        events = battles[0]
        player = h.searcher(classes, events[-1]["username"], bridge, mode="mix", mix=MixSettings(min_iterations=0))
        h.play(player, events)
        assert not any("predict" in d for d in player.decisions_of(events[0]["room"]))
        assert "pred_made" not in player.game_summary(events[0]["room"])
        assert not player._trackers
    finally:
        bridge.close()


@needs_player
def test_mix_guard_plays_legal_moves_and_counts_what_it_did(classes):
    battles = h.load_fixture()
    bridge = t_om.PredictBridge(battles, opponent=[("move swordsdance", 0.7, None), ("move tackle", 0.3, None)])
    fired = 0
    try:
        for events in battles:
            name, room = events[-1]["username"], events[0]["room"]
            player = h.searcher(classes, name, bridge, mode="mix-guard", mix=MixSettings(min_iterations=0), opponent_model=t_om.believer())
            h.play(player, events)
            h.check_notes(player, bridge, events)
            decisions = player.decisions_of(room)
            for d in decisions:
                assert d["mask"][d["played"]] == 1
                g = d.get("guard")
                if g:
                    assert g["guard"] and sp.base_action(d["played"]) in picks_of(g)
                    assert g["changed"] == (d["played"] != g["veto"])
            summary = player.game_summary(room)
            assert summary["guard_fired"] == sum(1 for d in decisions if d.get("guard"))
            assert summary["guard_changed"] == sum(1 for d in decisions if (d.get("guard") or {}).get("changed"))
            fired += summary["guard_fired"]
        assert fired > 20                                       # a read of 70% Swords Dance fires it on most move turns
        assert player.search_totals["guard_errors"] == 0
    finally:
        bridge.close()


def test_the_ladder_takes_the_new_settings_and_counts_predictions():
    import ladder

    assert ladder.PLAYER_MODES == sp.MODES and "mix-guard" in ladder.PLAYER_MODES
    rows = [{"arm": "mix", "won": True, "pred_scored": 20, "pred_top1_right": 9, "pred_top3_right": 17},
            {"arm": "mix", "won": False, "pred_scored": 5, "pred_top1_right": 3, "pred_top3_right": 4},
            {"arm": "mix-guard", "won": True, "guard_fired": 4, "guard_changed": 1}]
    assert ladder.predictions_text(rows) == ", predictions: likeliest right 12 of 25 (48%), in the top 3 21 (84%)"
    text = ladder.arms_text(rows)
    assert "likeliest right 12 of 25" in text and "guard fired 4 times and changed the move 1 times" in text
    assert ladder.predictions_text([{"arm": "net"}]) == ""


def test_console_lines_come_out_in_order_and_a_held_window_does_not_hold_the_caller(monkeypatch):
    """A window that holds every write (conhost with a selection in it) holds only the printing thread."""
    import threading
    import time

    import ladder

    held, out = threading.Event(), []

    class HeldWindow:
        def write(self, data):
            held.wait(10.0)
            out.append(data)
            return len(data)

        def flush(self):
            pass

    monkeypatch.setattr(sys, "stdout", HeldWindow())
    lines = ladder.ConsoleLines()
    began = time.monotonic()
    for i in range(200):
        lines.say(f"[1408] T{i} predict Swords Dance 47% (set-up)")
    assert time.monotonic() - began < 0.5                    # the event loop goes on: the order is sent
    assert lines.flush(timeout=0.2) is False                 # the window is still held
    held.set()
    assert lines.flush(timeout=10.0) is True
    assert "".join(out).splitlines() == [f"[1408] T{i} predict Swords Dance 47% (set-up)" for i in range(200)]
    assert lines.dropped == 0


def test_an_illusion_that_ends_gives_the_moves_and_the_faint_to_the_revealed_pokemon():
    """|replace|: the lines after it name Zoroark, not the disguise (review of 2026-10-07, finding 2)."""
    from bench.opp_dataset import GameRows

    rows = GameRows({"me": {"side": "p1"}, "id": "battle-gen9randombattle-1", "myTeam": []})
    for line in ["|player|p1|me|1|1500", "|player|p2|them|2|1500", "|teamsize|p1|6", "|teamsize|p2|6",
                 "|switch|p1a: Pikachu|Pikachu, L90|100/100", "|switch|p2a: Venusaur|Venusaur, L84|100/100", "|turn|1",
                 "|move|p2a: Venusaur|Sludge Bomb|p1a: Pikachu", "|move|p1a: Pikachu|Thunderbolt|p2a: Venusaur",
                 "|-damage|p2a: Venusaur|60/100", "|turn|2",
                 "|switch|p2a: Zapdos|Zapdos-Galar, L80|100/100",            # Zoroark-Hisui, looking like Zapdos-Galar
                 "|move|p1a: Pikachu|Thunderbolt|p2a: Zapdos", "|-damage|p2a: Zapdos|50/100", "|turn|3",
                 "|move|p2a: Zapdos|Hyper Voice|p1a: Pikachu", "|-damage|p1a: Pikachu|40/100",
                 "|move|p1a: Pikachu|Close Combat|p2a: Zapdos",
                 "|replace|p2a: Zoroark|Zoroark-Hisui, L80, F", "|-damage|p2a: Zoroark|10/100", "|-end|p2a: Zoroark|Illusion",
                 "|turn|4"]:
        rows.feed(line)
    opp = rows.ctx()["opp"]
    assert opp["active"]["species"] == "Zoroark-Hisui" and opp["active"]["moves"] == {"hypervoice": 1}
    assert abs(opp["active"]["hp"] - 0.10) < 1e-9
    assert {m["species"] for m in opp["bench"]} == {"Venusaur"}      # Zapdos-Galar was never seen itself
    assert opp["revealed"] == 2
    rows.feed("|move|p2a: Zoroark|Hyper Voice|p1a: Pikachu")
    rows.feed("|faint|p2a: Zoroark")
    assert rows.sides["p2"].mons["Zoroark"].fainted and "Zapdos" not in rows.sides["p2"].mons
    # A disguise that had been seen gets back its own HP and moves.
    rows = GameRows({"me": {"side": "p1"}, "id": "battle-gen9randombattle-2", "myTeam": []})
    for line in ["|switch|p1a: Pikachu|Pikachu, L90|100/100", "|switch|p2a: Venusaur|Venusaur, L84|100/100", "|turn|1",
                 "|move|p2a: Venusaur|Sludge Bomb|p1a: Pikachu", "|-damage|p2a: Venusaur|60/100", "|turn|2",
                 "|switch|p2a: Zapdos|Zapdos-Galar, L80|100/100", "|turn|3",
                 "|switch|p2a: Venusaur|Venusaur, L84|100/100",              # Zoroark again, now as Venusaur
                 "|turn|4", "|move|p2a: Venusaur|Sludge Bomb|p1a: Pikachu", "|move|p2a: Venusaur|Flamethrower|p1a: Pikachu",
                 "|replace|p2a: Zoroark|Zoroark, L84, M", "|-damage|p2a: Zoroark|70/100", "|turn|5"]:
        rows.feed(line)
    venusaur, zoroark = rows.sides["p2"].mons["Venusaur"], rows.sides["p2"].mons["Zoroark"]
    assert venusaur.moves == {"sludgebomb": 1} and abs(venusaur.hp - 0.60) < 1e-9
    assert zoroark.moves == {"sludgebomb": 1, "flamethrower": 1} and rows.sides["p2"].active is zoroark


# =============================================================================================
# The set-up opponent
# =============================================================================================

def _poke_env():
    try:
        from poke_env.battle import Move, PokemonType, Status  # noqa: F401
        return True
    except Exception:
        return False


needs_poke_env = pytest.mark.skipif(not _poke_env(), reason="poke-env is not installed")


def fake_battle(moves, hp=1.0, boosts=None, my_types=("NORMAL",), their_types=("WATER",), their_status=None, asleep_on_their_side=False,
                my_status=None, hits=1.0, their_boosts=None, ability=None, effects=()):
    from poke_env.battle import Move, PokemonType, Status

    me = SimpleNamespace(current_hp_fraction=hp, boosts=dict({s: 0 for s in sm.STATS}, **(boosts or {})), status=my_status,
                         types=[PokemonType[t] for t in my_types], damage_multiplier=lambda t: hits)
    them = SimpleNamespace(fainted=False, status=their_status, types=[PokemonType[t] for t in their_types], effects={e: 1 for e in effects},
                           ability=ability, possible_abilities=[], item=None, boosts=dict({s: 0 for s in sm.STATS}, **(their_boosts or {})))
    team = {"p2: a": SimpleNamespace(status=Status.SLP if asleep_on_their_side else None, fainted=False)}
    return SimpleNamespace(active_pokemon=me, opponent_active_pokemon=them, available_moves=[Move(m, gen=9) for m in moves],
                           available_switches=[], opponent_team=team, opponent_side_conditions={}, fields={})


@needs_poke_env
def test_the_setup_opponent_sets_up_puts_to_sleep_and_heals_by_its_rules():
    from poke_env.battle import Effect, Status

    from bench.setup_player import setup_choice, setup_player

    pick = lambda b, **kw: getattr(setup_choice(b, **kw), "id", None)  # noqa: E731
    assert pick(fake_battle(["swordsdance", "earthquake"])) == "swordsdance"
    assert pick(fake_battle(["swordsdance", "earthquake"], boosts={"atk": 4})) is None             # at max_stage: attack
    assert pick(fake_battle(["swordsdance", "earthquake"], boosts={"atk": 4}), max_stage=6) == "swordsdance"
    assert pick(fake_battle(["swordsdance", "earthquake"], hp=0.45)) is None                        # about to be knocked out
    assert pick(fake_battle(["swordsdance", "earthquake"], hp=0.8, hits=2.0)) is None
    assert pick(fake_battle(["swordsdance", "earthquake"], hp=0.8, their_boosts={"spa": 2})) is None
    assert pick(fake_battle(["calmmind", "agility", "earthquake"])) in ("calmmind", "agility")
    assert pick(fake_battle(["shellsmash", "calmmind"])) == "shellsmash"                            # raises most
    assert pick(fake_battle(["bellydrum", "earthquake"], hp=0.55)) is None
    assert pick(fake_battle(["earthquake", "tackle"])) is None                                     # SimpleHeuristics decides
    # Sleep first, when the target can fall asleep.
    assert pick(fake_battle(["spore", "swordsdance"])) == "spore"
    assert pick(fake_battle(["spore", "swordsdance"], their_status=Status.PAR)) == "swordsdance"
    assert pick(fake_battle(["spore", "swordsdance"], asleep_on_their_side=True)) == "swordsdance"   # Sleep Clause
    assert pick(fake_battle(["spore", "swordsdance"], their_types=("GRASS",))) == "swordsdance"      # powder on a Grass type
    assert pick(fake_battle(["hypnosis", "tackle"], their_types=("GRASS",))) == "hypnosis"
    assert pick(fake_battle(["yawn", "tackle"], effects=(Effect.YAWN,))) is None
    assert pick(fake_battle(["spore", "tackle"], ability="insomnia")) is None
    # Asleep: Sleep Talk. Boosted and hurt: heal.
    assert pick(fake_battle(["sleeptalk", "curse", "rest"], my_status=Status.SLP)) == "sleeptalk"
    assert pick(fake_battle(["calmmind", "recover", "psychic"], hp=0.4, boosts={"spa": 2})) == "recover"
    assert pick(fake_battle(["calmmind", "recover", "psychic"], hp=0.4)) is None                     # not boosted: no heal rule
    assert setup_player() is setup_player() and setup_player().__name__ == "SetupPlayer"
