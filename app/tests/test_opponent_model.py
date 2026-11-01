"""
The opponent model (opponent_model.py) and the "predict" mode of the search player.

    python -m pytest tests/test_opponent_model.py -q -p no:cacheprovider

No Foul Play, no Showdown server and no checkpoint are needed. The model's parts are tested on
hand-made situations, the choice on hand-made payoff tables, the tracker and the player on the
seven recorded battles of tests/fixtures/foulplay_bridge_trace.jsonl.gz (the harness of
tests/test_search_player.py: real poke-env battles, a stub network, a fake bridge). The recorded
answers are older than the opponent's side of the bridge, so the fake bridge here adds one: an
opponent block made from the battle itself, and a payoff table made up for the test.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import opponent_model as om  # noqa: E402
import search_player as sp  # noqa: E402
from opponent_model import OpponentModel, OpponentTracker, Prediction, PredictSettings, choose  # noqa: E402

_spec = importlib.util.spec_from_file_location("search_player_tests", os.path.join(REPO, "tests", "test_search_player.py"))
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)
classes = h.classes                 # the module's fixture: the player classes with the stub network
needs_player = h.needs_player


# =============================================================================================
# A situation by hand
# =============================================================================================

def ctx_of(moves=None, last=None, bench=("garchomp", "blissey"), unrevealed=3, turns=2, entered="switch", disadvantage=False,
           advantage=False, we_switched=False, boosts=None, tera_used=False, pool=("earthquake", "stoneedge", "swordsdance", "uturn", "toxic")):
    moves = dict(moves or {})
    return {
        "turn": 5,
        "opp": {"active": {"species": "Landorus-Therian", "id": "landorustherian", "hp": 0.8, "status": "", "fainted": False,
                           "moves": moves, "last_move": last, "boosts": dict(boosts or {}), "turns_on_field": turns,
                           "entered_by": entered, "types": ["GROUND", "FLYING"], "base_types": ["GROUND", "FLYING"], "pool": list(pool)},
                "bench": [{"id": b, "species": b, "fainted": False} for b in bench],
                "revealed": 1 + len(bench), "unrevealed": unrevealed, "alive": 1 + len(bench) + unrevealed, "tera_used": tera_used},
        "us": {"active": {"species": "Heatran", "id": "heatran", "types": ["FIRE", "STEEL"], "base_types": ["FIRE", "STEEL"]},
               "last_action": "switch heatran" if we_switched else "move magmastorm"},
        "matchup": {"opp_disadvantage": disadvantage, "opp_advantage": advantage},
    }


def search_of(**shares):
    """A search view from name=share ("move_earthquake", "switch_garchomp", "unseen_kyogre", "tera_earthquake")."""
    options = []
    for key, share in shares.items():
        kind, name = key.split("_", 1)
        if kind == "move":
            options.append(("move " + name, share, None))
        elif kind == "tera":
            options.append(("move " + name + " terastallize", share, None))
        elif kind == "switch":
            options.append(("switch " + name, share, True))
        else:
            options.append(("switch " + name, share, False))
    return {"options": options, "sets": [["earthquake", "uturn", "stealthrock", "stoneedge"]]}


def test_search_view_reads_an_advice_a_helper_block_and_a_replay_row():
    import foulplay_bridge as fb

    advice = fb._advice_from({"ok": True, "options": [], "opponent": {
        "active": "landorustherian", "revealed": ["garchomp"],
        "options": [{"name": "move earthquake", "share": 0.6, "visits": 6, "score": 0.5},
                    {"name": "switch garchomp", "share": 0.3, "visits": 3, "score": 0.5, "seen": True},
                    {"name": "switch kyogre", "share": 0.1, "visits": 1, "score": 0.5, "seen": False}],
        "worlds": [{"chance": 1.0, "active": {"species": "landorustherian", "moves": ["earthquake", "uturn"]}}]}}, 1.0, 0)
    view = om.search_view(advice)
    assert view["options"] == [("move earthquake", 0.6, None), ("switch garchomp", 0.3, True), ("switch kyogre", 0.1, False)]
    assert view["sets"] == [["earthquake", "uturn"]]
    assert om.search_view(advice.raw["opponent"])["options"] == view["options"]
    row = {"options": [["move earthquake", 0.6, None, 0.5, 1, 6]], "sets": [{"moves": ["earthquake"]}]}
    assert om.search_view(row) == {"options": [("move earthquake", 0.6, None)], "sets": [["earthquake"]]}
    assert om.search_view(om.search_view(row)) == om.search_view(row)           # a view is read as itself
    assert om.search_view(None) is None and om.search_view({}) is None
    assert om.search_view(fb._advice_from({"ok": True, "options": []}, 1.0, 0)) is None     # an answer of an old helper


def test_fold_search_folds_tera_into_the_move_and_unshown_switches_into_one():
    view = search_of(move_earthquake=0.4, tera_earthquake=0.1, move_uturn=0.1, switch_garchomp=0.2, unseen_kyogre=0.15, unseen_zapdos=0.05)
    folded = om.fold_search(view, ["garchomp", "blissey"], True)
    assert folded["moves"] == pytest.approx({"earthquake": 0.5, "uturn": 0.1})
    assert folded["tera"] == pytest.approx({"earthquake": 0.1})
    assert folded["switches"] == pytest.approx({"garchomp": 0.2}) and folded["unseen"] == pytest.approx(0.2)
    assert folded["switch_share"] == pytest.approx(0.4)
    assert "stealthrock" in folded["pool"]
    # Nobody hidden: a switch to an invented Pokemon is a switch all the same, to nobody we can name.
    assert om.fold_search(view, ["garchomp"], False)["unseen"] == 0.0
    assert om.fold_search(None, [], False) is None and om.fold_search({"options": [("none", 1.0, None)], "sets": []}, [], False) is None


def test_truth_name_is_the_scorers():
    assert om.truth_name({"known": True, "locked": False, "type": "move", "action": "move earthquake"}) == "move earthquake"
    assert om.truth_name({"known": True, "locked": False, "type": "switch", "action": "switch kyogre", "target_revealed": False}) == "switch ?"
    assert om.truth_name({"known": True, "locked": False, "type": "switch", "action": "switch garchomp", "target_revealed": True}) == "switch garchomp"
    assert om.truth_name({"known": False}) is None and om.truth_name({"known": True, "locked": True, "action": "move outrage"}) is None


# =============================================================================================
# The model
# =============================================================================================

@pytest.mark.parametrize("model", [OpponentModel(None), OpponentModel.load()])
def test_a_prediction_is_a_distribution_over_what_it_can_do(model):
    rng = random.Random(3)
    for _ in range(200):
        moves = {m: rng.randint(1, 3) for m in rng.sample(["earthquake", "uturn", "stoneedge", "swordsdance"], rng.randint(0, 4))}
        ctx = ctx_of(moves=moves, last=rng.choice(list(moves)) if moves else None, bench=rng.sample(["garchomp", "blissey", "zapdos"], rng.randint(0, 3)),
                     unrevealed=rng.randint(0, 3), turns=rng.randint(0, 3), entered=rng.choice(["switch", "forced", "lead"]),
                     disadvantage=rng.random() < 0.4, we_switched=rng.random() < 0.3, tera_used=rng.random() < 0.3)
        search = None
        if rng.random() < 0.7:
            search = search_of(move_earthquake=rng.random(), tera_earthquake=rng.random() * 0.2, move_stealthrock=rng.random(),
                               switch_garchomp=rng.random() * 0.5, unseen_kyogre=rng.random() * 0.3)
        p = model.predict(ctx, search)
        assert sum(p.dist.values()) == pytest.approx(1.0)
        assert all(v > 0 and (n.startswith("move ") or n.startswith("switch ")) for n, v in p.dist.items())
        assert 0.0 <= p.p_switch <= 1.0 and 0.0 <= p.p_tera <= 1.0
        assert sum(v for n, v in p.dist.items() if n.startswith("switch ")) == pytest.approx(p.p_switch, abs=1e-9)
        bench = {b["id"] for b in ctx["opp"]["bench"]}
        for name in p.dist:
            if name.startswith("switch "):
                assert name[7:] in bench or (name == "switch ?" and ctx["opp"]["unrevealed"] > 0)
        if ctx["opp"]["tera_used"]:
            assert p.p_tera == 0.0
        assert p.with_search == (search is not None)
        assert p.ranked()[0][1] == max(p.dist.values()) and p.top(2) == p.ranked()[:2]


def test_nobody_to_switch_to_means_it_stays_and_a_move_only_the_search_names_is_a_candidate():
    model = OpponentModel(None)
    p = model.predict(ctx_of(moves={"earthquake": 1}, bench=(), unrevealed=0), search_of(move_earthquake=0.5, move_stealthrock=0.5, unseen_kyogre=0.3))
    assert p.p_switch == 0.0 and not any(n.startswith("switch ") for n in p.dist)
    assert "move stealthrock" in p.dist           # not in its pool here, but the search sampled a set with it


def test_the_situation_moves_the_switch_probability_the_way_people_do():
    model = OpponentModel.load()
    assert model.fitted, "opponent_model.json is missing: python opponent_model.py fit --replay ..."
    search = search_of(move_earthquake=0.6, switch_garchomp=0.3, unseen_kyogre=0.1)
    base = model.predict(ctx_of(moves={"earthquake": 1}), search).p_switch
    assert model.predict(ctx_of(moves={"earthquake": 1}, disadvantage=True), search).p_switch > base
    assert model.predict(ctx_of(moves={"earthquake": 1}, we_switched=True), search).p_switch > base
    assert model.predict(ctx_of(moves={"earthquake": 1}, turns=0, entered="forced"), search).p_switch < base / 2      # just in after a faint
    assert model.predict(ctx_of(moves={"earthquake": 1}, boosts={"atk": 2}), search).p_switch < base
    more = search_of(move_earthquake=0.3, switch_garchomp=0.6, unseen_kyogre=0.1)
    assert model.predict(ctx_of(moves={"earthquake": 1}), more).p_switch > base                # and the search's own view counts
    # Without the search the same directions hold.
    assert model.predict(ctx_of(disadvantage=True), None).p_switch > model.predict(ctx_of(), None).p_switch


def test_the_search_and_its_last_move_both_show_in_the_move_distribution():
    model = OpponentModel.load()
    ctx = ctx_of(moves={"earthquake": 2, "uturn": 1}, last="uturn")
    p = model.predict(ctx, search_of(move_earthquake=0.7, move_uturn=0.1, move_stoneedge=0.2))
    assert p.top(1)[0][0] == "move earthquake"
    q = model.predict(ctx_of(moves={"earthquake": 2, "uturn": 1}, last="earthquake"), search_of(move_earthquake=0.7, move_uturn=0.1, move_stoneedge=0.2))
    assert q.dist["move earthquake"] > p.dist["move earthquake"] and q.dist["move uturn"] < p.dist["move uturn"]
    assert p.dist["move toxic"] > 0          # a move of its pool nobody has seen keeps a little


def test_fit_recovers_what_the_rows_say():
    """Made-up games: it switches on 60% of the turns at a disadvantage and 5% otherwise, whatever the search says."""
    rng = random.Random(11)
    rows, replay = [], {}
    for i in range(1200):
        disadvantage = rng.random() < 0.4
        switches = rng.random() < (0.6 if disadvantage else 0.05)
        ctx = ctx_of(moves={"earthquake": 1, "uturn": 1}, last="uturn", disadvantage=disadvantage)
        ctx["turn"] = i
        if switches:
            label = {"type": "switch", "name": "garchomp", "action": "switch garchomp", "tera": False, "known": True, "locked": False,
                     "target_revealed": True, "move_revealed": None}
        else:
            label = {"type": "move", "name": "earthquake" if rng.random() < 0.8 else "uturn", "action": None, "tera": rng.random() < 0.1,
                     "known": True, "locked": False, "target_revealed": None, "move_revealed": True}
            label["action"] = "move " + label["name"]
        rows.append({"game": "g%d" % (i % 40), "turn": i, "kind": "turn", "label": label, "ctx": ctx, "split": "train"})
        replay[("g%d" % (i % 40), i)] = {"iterations": 50000, "opp": {"options": [["move earthquake", 0.5, None], ["move uturn", 0.3, None],
                                                                                 ["switch garchomp", 0.15, True], ["switch blissey", 0.05, True]],
                                                                     "sets": [{"moves": ["earthquake", "uturn"]}]}}
    params = om.fit(rows, replay)
    model = OpponentModel(params)
    search = replay[("g0", 0)]["opp"]
    assert model.predict(ctx_of(moves={"earthquake": 1, "uturn": 1}, last="uturn", disadvantage=True), search).p_switch == pytest.approx(0.6, abs=0.07)
    assert model.predict(ctx_of(moves={"earthquake": 1, "uturn": 1}, last="uturn"), search).p_switch == pytest.approx(0.05, abs=0.03)
    stays = model.predict(ctx_of(moves={"earthquake": 1, "uturn": 1}, last="uturn", bench=(), unrevealed=0), search)
    assert stays.dist["move earthquake"] == pytest.approx(0.8, abs=0.08)
    target = model.targets(["garchomp", "blissey"], True, om.fold_search(om.search_view(search), ["garchomp", "blissey"], True))
    assert target["garchomp"] > 0.8 > 0.75                # always Garchomp in the rows, three in four for the search: its shares are sharpened
    assert params["tera"]["base_rate"] == pytest.approx(0.1, abs=0.03)
    assert params["fitted"]["replayed_rows"] == 1200 and params["fitted"]["iterations_per_decision_median"] == 50000
    json.dumps(params)                                    # it is written as JSON


def test_reliability_bins_what_was_said_against_what_happened():
    pairs = [(0.25, 0.0)] * 3 + [(0.25, 1.0)] + [(0.85, 1.0)] * 4 + [(0.85, 0.0)]
    out = om.reliability(pairs)
    low, high = out["bins"]
    assert (low["n"], low["said"], low["happened"]) == (4, 0.25, 0.25)
    assert (high["n"], high["said"], high["happened"]) == (5, 0.85, 0.8)
    assert out["ece"] == pytest.approx(5 * 0.05 / 9, abs=1e-4)


# =============================================================================================
# Acting on it
# =============================================================================================

COLUMNS = ["move earthquake", "move earthquake terastallize", "move uturn", "switch garchomp", "switch kyogre"]
VIEW = {"options": [("move earthquake", 0.5, None), ("move earthquake terastallize", 0.05, None), ("move uturn", 0.1, None),
                    ("switch garchomp", 0.25, True), ("switch kyogre", 0.1, False)], "sets": []}
MASK = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
SHARES = [0.4, 0.2, 0.1, 0.05, 0.15, 0.1, 0, 0, 0, 0, 0, 0, 0, 0]


def table_of(rows):
    """A payoff table from {action: [value per column]}; everything else unknown."""
    table = [[None] * len(COLUMNS) for _ in range(14)]
    for action, values in rows.items():
        table[action] = list(values)
    return table


def prediction_of(**dist):
    dist = {k.replace("_", " ", 1): v for k, v in dist.items()}
    p_switch = sum(v for k, v in dist.items() if k.startswith("switch "))
    return Prediction(dist=dist, p_switch=p_switch, p_tera=0.1, tera={}, with_search=True)


def test_column_probabilities_split_tera_and_spread_the_unshown():
    p = prediction_of(move_earthquake=0.5, move_uturn=0.1, switch_garchomp=0.3, **{"switch_?": 0.1})
    probs, names = om.column_probabilities(p, COLUMNS, VIEW)
    assert names == ["move earthquake", "move earthquake", "move uturn", "switch garchomp", "switch ?"]
    assert probs == pytest.approx([0.45, 0.05, 0.1, 0.3, 0.1])
    assert sum(probs) == pytest.approx(1.0)
    # A move the table has no column for is on no column.
    q = prediction_of(move_earthquake=0.5, move_toxic=0.5)
    assert sum(om.column_probabilities(q, COLUMNS, VIEW)[0]) == pytest.approx(0.5)


def test_an_unsure_prediction_changes_nothing():
    p = prediction_of(move_earthquake=0.4, move_uturn=0.3, switch_garchomp=0.3)
    table = table_of({0: [0] * 5, 4: [500] * 5})                       # the other action looks far better everywhere
    action, info = choose(p, table, COLUMNS, VIEW, 0, MASK, SHARES, PredictSettings(min_confidence=0.6))
    assert action == 0 and info["why"] == "unsure" and not info["changed"] and info["confidence"] == 0.4


def test_a_confident_read_with_a_clear_gain_and_no_risk_leaves_the_mix_choice():
    # It very probably switches to Garchomp. Our move 0 is fine against what is there, action 4 (a
    # switch of ours) is far better against Garchomp coming in and no worse if it stays: the double switch.
    p = prediction_of(switch_garchomp=0.7, move_earthquake=0.2, move_uturn=0.1)
    table = table_of({0: [50, 50, 50, 0, 0], 4: [45, 45, 45, 120, 100]})
    action, info = choose(p, table, COLUMNS, VIEW, 0, MASK, SHARES, PredictSettings())
    assert action == 4 and info["changed"] and info["why"] == ""
    assert info["read"] == "switch" and info["confidence"] == pytest.approx(0.7) and info["second"] == "move earthquake"
    assert info["gain"] > 0.08 and info["risk"] <= 0.08 and info["coverage"] == pytest.approx(1.0)


def test_a_wrong_read_must_not_cost_much():
    # The same read, but the answer to Garchomp loses badly to the Earthquake it uses if it stays.
    p = prediction_of(switch_garchomp=0.7, move_earthquake=0.2, move_uturn=0.1)
    table = table_of({0: [50, 50, 50, 0, 0], 4: [-150, -150, 45, 220, 100]})
    action, info = choose(p, table, COLUMNS, VIEW, 0, MASK, SHARES, PredictSettings())
    assert action == 0 and info["why"] == "risk" and info["candidate"] == 4 and info["gain"] > 0.08 and info["risk"] > 0.08
    # A setting: somebody who accepts the risk gets the move.
    action, info = choose(p, table, COLUMNS, VIEW, 0, MASK, SHARES, PredictSettings(max_risk=0.5))
    assert action == 4 and info["changed"]


def test_no_gain_no_table_no_cell_for_the_second_and_too_little_of_the_table():
    p = prediction_of(switch_garchomp=0.7, move_earthquake=0.2, move_uturn=0.1)
    same = table_of({0: [50, 50, 50, 0, 0], 4: [50, 50, 50, 10, 0]})
    assert choose(p, same, COLUMNS, VIEW, 0, MASK, SHARES)[1]["why"] == "no gain"
    for nothing in (None, [], table_of({})):
        action, info = choose(p, nothing, COLUMNS, VIEW, 0, MASK, SHARES)
        assert action == 0 and info["why"] == "no table"
    blind = table_of({0: [50, 50, 50, 0, 0], 4: [None, None, 45, 120, 100]})           # nothing known against its Earthquake
    action, info = choose(p, blind, COLUMNS, VIEW, 0, MASK, SHARES, PredictSettings(min_coverage=0.5))
    assert action == 0 and info["why"] == "blind"
    thin = table_of({0: [50, 50, 50, 0, 0], 4: [None, None, None, None, 100]})
    assert choose(p, thin, COLUMNS, VIEW, 0, MASK, SHARES)[1]["why"] == "coverage"
    assert choose(None, same, COLUMNS, VIEW, 0, MASK, SHARES)[1]["why"] == "no prediction"


def test_a_confident_move_read_and_the_form_of_the_move_stays_the_mix_rules():
    p = prediction_of(move_earthquake=0.8, move_uturn=0.1, switch_garchomp=0.1)
    mask = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0]
    shares = [0.3, 0.1, 0, 0, 0.1, 0, 0, 0, 0, 0, 0.1, 0.4, 0, 0]
    table = table_of({0: [0, 0, 0, 0, 0], 10: [0, 0, 0, 0, 0], 1: [10, 10, 10, 10, 10], 11: [200, 200, 150, 150, 150], 4: [60, 60, 0, 0, 0]})
    # The mix choice is move 0 with Tera (action 10). Move 1 is compared in the form the search prefers: with Tera (0.4 against 0.1).
    action, info = choose(p, table, COLUMNS, VIEW, 10, mask, shares, PredictSettings())
    assert action == 11 and info["read"] == "move earthquake" and info["changed"]
    # Tera alone is never what this rule changes: with the mix choice on move 1 already, nothing better is left.
    action, info = choose(p, table, COLUMNS, VIEW, 11, mask, shares, PredictSettings())
    assert action == 11 and not info["changed"]
    # An action the search did not look at is no candidate.
    shares2 = list(shares)
    shares2[11], shares2[1] = 0.0, 0.0
    action, info = choose(p, table, COLUMNS, VIEW, 10, mask, shares2, PredictSettings())
    assert action != 1 and action != 11


def test_choose_never_returns_an_illegal_action_and_only_leaves_mix_on_a_confident_gain():
    rng = random.Random(5)
    settings = PredictSettings()
    changed = 0
    for _ in range(3000):
        mask = [1 if rng.random() < 0.6 else 0 for _ in range(14)]
        for k in range(4):
            mask[10 + k] = mask[10 + k] and mask[k]
        legal = [a for a in range(14) if mask[a]]
        if not legal:
            continue
        mix_action = rng.choice(legal)
        shares = [rng.random() if mask[a] else 0.0 for a in range(14)]
        weights = [rng.random() ** 3 for _ in range(4)]
        total = sum(weights)
        p = prediction_of(move_earthquake=weights[0] / total, move_uturn=weights[1] / total, switch_garchomp=weights[2] / total,
                          **{"switch_?": weights[3] / total})
        table = [[(rng.uniform(-200, 200) if rng.random() < 0.7 else None) for _ in COLUMNS] if mask[a] else [None] * 5 for a in range(14)]
        action, info = choose(p, table, COLUMNS, VIEW, mix_action, mask, shares, settings)
        assert mask[action] == 1
        if info["changed"]:
            changed += 1
            assert action != mix_action and info["confidence"] >= settings.min_confidence
            assert info["gain"] >= settings.min_gain and info["risk"] <= settings.max_risk and info["coverage"] >= settings.min_coverage
        else:
            assert action == mix_action and info["why"]
    assert changed > 20


# =============================================================================================
# The tracker on the recorded battles
# =============================================================================================

@needs_player
def test_the_tracker_gives_a_context_every_turn_and_says_what_the_opponent_did():
    battles = h.load_fixture()
    turns = known = 0
    for events in battles:
        me = events[-1]["username"]
        tracker = OpponentTracker(events[0]["room"])
        role = None
        seen_turn = 0
        closed = []
        for event in events:
            if event["t"] != "msg":
                continue
            for line in event["data"].split("\n"):
                parts = line.split("|")
                if len(parts) > 3 and parts[1] == "player" and parts[3] == me:
                    role = parts[2]
                tracker.feed(line, role)
                if len(parts) > 2 and parts[1] == "turn":
                    seen_turn = int(parts[2])
                    ctx = tracker.context(seen_turn)
                    assert ctx is not None and ctx["turn"] == seen_turn
                    assert ctx["opp"]["active"]["species"] and ctx["us"]["active"]["species"]
                    assert tracker.context(seen_turn + 1) is None            # not that turn's context
            closed += tracker.closed()
        assert role in ("p1", "p2") and tracker.broken is None
        assert [t for t, _label, _name in closed] == list(range(1, seen_turn + 1))      # every turn once, in order
        for _turn, label, name in closed:
            turns += 1
            if name is not None:
                known += 1
                assert name.startswith("move ") or name.startswith("switch ")
                assert name == om.truth_name(label)
        assert tracker.closed() == []
    assert turns > 100 and known > 0.8 * turns


def test_a_tracker_without_a_side_waits_and_a_line_it_cannot_read_stops_it_quietly():
    tracker = OpponentTracker("battle-x")
    tracker.feed("|switch|p1a: Heatran|Heatran, L80|100/100")
    tracker.feed("|switch|p2a: Landorus|Landorus-Therian, L76|100/100")
    assert tracker.context() is None and tracker.closed() == []
    tracker.feed("|turn|1", "p1")                                   # now we know: the lines kept are read first
    ctx = tracker.context(1)
    assert ctx["opp"]["active"]["id"] == "landorustherian" and ctx["us"]["active"]["id"] == "heatran"
    tracker.feed("|move|p2a: Landorus|Earthquake|p1a: Heatran")
    tracker.feed("|turn|2")
    assert [(t, n) for t, _l, n in tracker.closed()] == [(1, "move earthquake")]
    tracker.feed("|turn|not a number")
    assert tracker.broken and tracker.context() is None
    assert [(t, n) for t, _l, n in tracker.closed()] == [(2, None)]     # the turn it broke in ends as "unknown"
    tracker.feed("|turn|3")                                        # nothing more is read, nothing is raised
    assert tracker.closed() == []


# =============================================================================================
# The player in "predict" mode
# =============================================================================================

class PredictBridge(h.FakeBridge):
    """
    The fake bridge of the search player's tests with an opponent's side in every answer, which
    the recorded answers do not have: the opponent "moves" with the share given here, and a
    payoff table in which `favourite` (a function of our options) is worth `bonus` more than
    every other action of ours against everything.
    """

    def __init__(self, battles, opponent=None, bonus=None, risky=False, **kwargs):
        super().__init__(battles, **kwargs)
        self.opponent = opponent or [("move tackle", 0.9, None), ("move growl", 0.1, None)]
        self.bonus, self.risky = bonus, risky
        for key, reply in list(self.replies.items()):
            if not reply.get("ok"):
                continue
            reply = dict(reply)
            reply["opponent"] = {"active": "somebody", "revealed": [],
                                 "options": [{"name": n, "share": s, "seen": seen, "visits": int(s * 1000), "score": 0.5} for n, s, seen in self.opponent],
                                 "worlds": [{"chance": 1.0, "iterations": 1000, "active": {"species": "somebody", "moves": ["tackle", "growl"]}}]}
            self.replies[key] = reply

    def _answer(self, future, key, rqid, wall_ms):
        asked = [a for a in self.asked if a[0] == key and a[1] == rqid]
        reply = self.replies.get((key, rqid))
        if reply and reply.get("ok") and self.bonus is not None and asked and asked[-1][4].get("payoff_ms"):
            options = sorted(reply["options"], key=lambda o: -o["policy"])
            theirs = [n for n, _s, _seen in self.opponent]
            rows = []
            for rank, option in enumerate(options):
                row = [self.bonus if rank == 1 else 0.0] * len(theirs)      # the search's second choice is the one the table likes
                if self.risky and rank == 1:
                    row[1] = -1000.0                                        # and it is terrible against the opponent's second action
                rows.append(row)
            reply = dict(reply, payoff={"mode": "full", "ms": 1.0, "worlds": [{
                "chance": 1.0, "depth": 2, "choices": [o["choice"] for o in options], "theirs": theirs, "matrix": rows,
                "cells": len(rows) * len(theirs), "known": len(rows) * len(theirs)}]})
            self.replies[(key, rqid)] = reply
        super()._answer(future, key, rqid, wall_ms)


def believer():
    """A model that takes the search's shares for the opponent's moves as they are, and never expects a switch."""
    params = om.default_params()
    params["move"].update({"T": 1.0, "weights": {"search": 1.0, "last": 0.0, "strongest": 0.0, "shown": 0.0, "pool": 0.0}})
    params["switch"]["with_search"] = [-12.0] + [0.0] * (len(om.SWITCH_FEATURES) - 1)
    return OpponentModel(params)


def predictor(classes, username, bridge, **kwargs):
    kwargs.setdefault("predict", PredictSettings(min_confidence=0.6))
    kwargs.setdefault("deadline", 1.0)             # the search (20 ms in these tests) and the table's 400 ms must fit
    return h.searcher(classes, username, bridge, mode="predict", mix=sp.MixSettings(min_iterations=0), **kwargs)


def test_predict_is_a_mode_of_the_player_and_of_the_ladder():
    import ladder

    assert "predict" in sp.MODES and sp.split_mode("predict") == ("predict", None)
    assert ladder.PLAYER_MODES == sp.MODES
    assert ladder.arm_fields({"arm": "predict", "pred_made": 3}, "p1", 500, "veto") == {
        "arm": "predict", "pred_made": 3, "role": "p1", "search_ms": 500, "mix_rule": "veto"}


@needs_player
def test_without_a_payoff_table_predict_plays_what_mix_plays_and_still_scores_its_predictions(classes):
    battles = h.load_fixture()
    bridge = PredictBridge(battles)                    # no table in any answer
    made = scored = 0
    try:
        for events in battles:
            name, room = events[-1]["username"], events[0]["room"]
            mixer = h.searcher(classes, name, bridge, mode="mix", mix=sp.MixSettings(min_iterations=0))
            h.play(mixer, events)
            bridge.notes.clear()
            bridge.asked.clear()
            player = predictor(classes, name, bridge)
            outcomes = []
            player.on_outcome = outcomes.append
            h.play(player, events)
            assert h.chosen(player) == h.chosen(mixer)                 # move for move
            h.check_notes(player, bridge, events)
            decisions = player.decisions_of(room)
            assert len(decisions) == len(h.orders_of(events))
            summary = player.game_summary(room)
            assert summary["arm"] == "predict" and summary["pred_changed"] == 0
            predicted = [d for d in decisions if d.get("predict") and "top" in d["predict"]]
            assert summary["pred_made"] == len(predicted)
            for d in decisions:
                if d["search"] is None:
                    assert "predict" not in d                           # the search did not answer: nothing to predict from
                    continue
                p = d["predict"]
                assert p["mix"] == d["played"] and not p["changed"]
                if d["forced"]:
                    assert p["why"] == "forced" and "top" not in p      # the opponent chooses nothing at a forced switch
                    continue
                assert p["why"] in ("unsure", "no table"), p
                assert p["top"][0][1] >= p["top"][1][1] and 0 <= p["p_switch"] <= 1
            # What the opponent did is read from the battle: every prediction but the one of a turn the game ended in.
            got = [d["predict"]["outcome"] for d in predicted if "outcome" in d["predict"]]
            assert len(got) == len(outcomes) and len(predicted) - len(got) <= 1
            right = [o for o in got if o["name"] is not None]
            assert summary["pred_scored"] == len(right)
            assert summary["pred_top1_right"] == sum(1 for o in right if o["top1_right"])
            assert summary["pred_reads_scored"] == sum(1 for o in right if o["read_right"] is not None)
            assert summary["pred_reads_right"] == sum(1 for o in right if o["read_right"])
            assert summary["pred_p_true"] == pytest.approx(sum(o["p_true"] for o in right), abs=1e-3)
            assert sum(summary["pred_kept"].values()) == summary["pred_made"]
            made += summary["pred_made"]
            scored += summary["pred_scored"]
            # The questions of a predict battle ask for the table, those of a forced switch do not.
            asked = [a for a in bridge.asked if a[0] == h.key_of(events)]
            assert any(a[4].get("payoff_ms") == 400.0 and a[4].get("payoff") == "full" for a in asked)
            assert sum(1 for a in asked if not a[4].get("payoff_ms")) == sum(1 for d in decisions if d["forced"] and d["asked"])
        assert made > 100 and scored > 0.8 * made
        assert player.search_totals["predict_errors"] == 0 and player.search_totals["trackers_broken"] == 0
        assert "opponent predicted on" in player.search_summary()
    finally:
        bridge.close()


@needs_player
def test_with_a_table_predict_leaves_the_mix_choice_only_when_the_read_is_confident(classes):
    battles = h.load_fixture()
    sure = PredictBridge(battles, bonus=400.0)                                         # "move tackle" at 0.9
    unsure = PredictBridge(battles, opponent=[("move tackle", 0.4, None), ("move growl", 0.35, None), ("move leer", 0.25, None)], bonus=400.0)
    risky = PredictBridge(battles, bonus=400.0, risky=True)
    try:
        changed = 0
        reasons = {"unsure": 0, "risk": 0}
        for events in battles[:3]:
            name, room = events[-1]["username"], events[0]["room"]
            player = predictor(classes, name, sure, opponent_model=believer(), predict=PredictSettings(once_bitten=False))
            h.play(player, events)
            h.check_notes(player, sure, events)
            for d in player.decisions_of(room):
                assert d["mask"][d["played"]] == 1
                p = d.get("predict")
                if p and p["changed"]:
                    changed += 1
                    assert d["source"] == "predict" and d["played"] != p["mix"]
                    assert p["confidence"] >= 0.6 and p["gain"] >= 0.08 and p["risk"] <= 0.08
                    assert sp.base_action(d["played"]) != sp.base_action(p["mix"])
                    where = p["position"]                   # the position in words, for whoever reads the record
                    assert where["played"] != where["mix_would_play"] and where["played"].split()[0] in ("move", "switch")
                    assert where["ours"].endswith("%") and where["theirs"].endswith("%")
                elif p:
                    assert d["played"] == p["mix"] and d["source"] == "mix"
            summary = player.game_summary(room)
            assert summary["pred_changed"] == sum(1 for d in player.decisions_of(room) if (d.get("predict") or {}).get("changed"))
            assert summary["pred_kept"].get("changed", 0) == summary["pred_changed"]

            for bridge, why in ((unsure, "unsure"), (risky, "risk")):
                other = predictor(classes, name, bridge, opponent_model=believer())
                h.play(other, events)
                kept = other.game_summary(room)
                assert kept["pred_changed"] == 0, kept
                reasons[why] += kept["pred_kept"].get(why, 0)
                if why == "unsure":
                    assert set(kept["pred_kept"]) == {"unsure"}, kept        # never confident: nothing else is looked at
        assert changed > 10
        assert reasons["unsure"] > 30 and reasons["risk"] > 0, reasons
    finally:
        for bridge in (sure, unsure, risky):
            bridge.close()


@needs_player
def test_a_read_that_failed_against_a_pokemon_is_not_acted_on_again(classes):
    """The fake search always expects "move tackle", which nobody in the recorded battles ever uses: every read is wrong."""
    battles = h.load_fixture()
    bridge = PredictBridge(battles, bonus=400.0)
    totals = [0, 0]
    try:
        for events in battles[:3]:
            name, room = events[-1]["username"], events[0]["room"]
            free = predictor(classes, name, bridge, opponent_model=believer(), predict=PredictSettings(once_bitten=False))
            h.play(free, events)
            bridge.notes.clear()
            player = predictor(classes, name, bridge, opponent_model=believer())          # once_bitten is the default
            h.play(player, events)
            h.check_notes(player, bridge, events)
            failed, changed, bitten = set(), 0, 0
            for d in player.decisions_of(room):
                p = d.get("predict")
                if not p or "top" not in p:
                    continue
                key = (p["against"], p["read"])
                if p["changed"]:
                    changed += 1
                    assert key not in failed, d                 # never twice on a read that was wrong against this Pokemon
                elif p["why"] == "bitten":
                    bitten += 1
                    assert key in failed and d["played"] == p["mix"] and d["source"] == "mix"
                if (p.get("outcome") or {}).get("read_right") is False:
                    failed.add(key)
            summary = player.game_summary(room)
            assert summary["pred_changed"] == changed and summary["pred_kept"].get("bitten", 0) == bitten
            assert changed + bitten == free.game_summary(room)["pred_changed"]     # the same turns, some of them now kept
            assert summary["pred_changed_right"] == 0           # "move tackle" never happens
            totals[0] += changed
            totals[1] += bitten
        assert totals[0] > 5 and totals[1] > 3, totals
    finally:
        bridge.close()


@needs_player
def test_a_prediction_that_raises_plays_the_mix_choice_and_is_counted(classes):
    battles = h.load_fixture()
    bridge = PredictBridge(battles, bonus=400.0)

    class Broken(OpponentModel):
        def predict(self, ctx, search=None):
            raise RuntimeError("no idea")

    try:
        events = battles[0]
        name, room = events[-1]["username"], events[0]["room"]
        mixer = h.searcher(classes, name, bridge, mode="mix", mix=sp.MixSettings(min_iterations=0))
        h.play(mixer, events)
        bridge.notes.clear()
        player = predictor(classes, name, bridge, opponent_model=Broken(None))
        h.play(player, events)
        assert h.chosen(player) == h.chosen(mixer)
        assert player.search_totals["predict_errors"] > 0
        assert all(d["predict"]["why"] in ("error", "forced") for d in player.decisions_of(room) if "predict" in d)
        h.check_notes(player, bridge, events)
    finally:
        bridge.close()


@needs_player
def test_a_deadline_too_short_for_the_search_and_the_table_is_refused(classes):
    bridge = PredictBridge([])
    try:
        least = sp.minimum_deadline(500) + 0.4
        with pytest.raises(ValueError):
            h.offline(classes["search"], "TooShort", mode="predict", bridge=bridge, search_ms=500, deadline=least - 0.05, warm_up=False)
        h.offline(classes["search"], "LongEnough", mode="predict", bridge=bridge, search_ms=500, deadline=least + 0.05, warm_up=False)
    finally:
        bridge.close()
