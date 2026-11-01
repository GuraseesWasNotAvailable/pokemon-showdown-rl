"""
Opponent seats (opponent_seats.py, setup_rules.py).

    python -m pytest tests/test_opponent_seats.py -q -p no:cacheprovider

What is held:

  * the plan: each kind gets round(share x all pairs) of the run's pairs (at least one), dealt
    round the workers; the opponent's seat alternates so that the network is p1 in half of a
    kind's pairs; Foul Play pairs go to at most max_helpers workers; the same arguments always
    give the same plan; the flags and their checks
  * setupnet's rules (rules 2 to 5 of decide's set-up player) on poke-env Battle objects built
    from a real request: set up when safe, up to max_stage, the move that raises most; sleep when
    the target can fall asleep (status, Sleep Clause, powder on Grass, Insomnia, Substitute,
    Electric Terrain); Sleep Talk when asleep; recover when boosted and hurt; nothing when in danger
  * setupnet's override: with p = 1 the rule's move replaces the network's, with p = 0 never; a
    switch the network chose is kept; the network's own pick of the same move stands; p is the
    share of the draws; a seed gives the same draws again
  * an opponent seat never hands the learner anything: on recorded battles the setupnet seat
    puts one STEP per decision and one CLEANUP, PSBot's and Foul Play's seats put nothing while
    they answer and STEP + CLEANUP when the network has to move for them; never a DONE
  * through worker.RolloutWorker's own rollout loop: the trained seat's battle reaches the
    learner, the setupnet seat's same battle does not, and its steps, history and slot are given
    back
  * a kind that cannot be used in a worker is made as self-play pairs; the heartbeat and the
    log line say so

No Showdown server, no Ray actor and no Foul Play are needed: the players never connect and are
fed recorded battles (tests/fixtures/foulplay_bridge_trace.jsonl.gz), Foul Play's helper is the
fake one of tests/test_foulplay_bridge.py, and PSBot's engine is a stand-in. bench/seats_smoke.py
is the live check on a server.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import dataclasses
import gzip
import json
import logging
import os
import pickle
import queue
import sys
import threading
import time
from types import SimpleNamespace

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("numpy")
pytest.importorskip("poke_env")
pytest.importorskip("ray")
pytest.importorskip("torch")

import numpy as np  # noqa: E402
from poke_env.battle import Battle, Field, Status  # noqa: E402

import config  # noqa: E402
import opponent_seats as os_  # noqa: E402
import setup_rules  # noqa: E402
import test_foulplay_bridge as tfb  # noqa: E402  (the fake Foul Play helper)
import worker  # noqa: E402
from opponent_seats import FoulPlaySeatSettings, OpponentSeatsConfig, SetupNetSettings  # noqa: E402

fake_helper = tfb.fake_helper          # the module-scoped fixture, used here under the same name
FIXTURE = os.path.join(REPO, "tests", "fixtures", "foulplay_bridge_trace.jsonl.gz")
REQUEST = json.load(open(os.path.join(REPO, "tests", "fixtures", "tera_request.json"), encoding="utf-8"))["request"]


def _cfg(**shares):
    fp = FoulPlaySeatSettings(iterations=500, **{k[3:]: v for k, v in shares.items() if k.startswith("fp_")})
    return OpponentSeatsConfig(shares=tuple((k, v) for k, v in shares.items() if not k.startswith("fp_")), foulplay=fp)


# --------------------------------------------------------------------------- the plan


def _seats(plan):
    return [[(s["kind"], s["seat"]) for s in w["seats"]] for w in plan["workers"]]


def test_the_shares_are_of_all_the_runs_pairs_and_the_seats_alternate():
    plan = os_.plan_seats([4] * 8, 32, _cfg(setupnet=0.15, psbot=0.05, foulplay=0.05))
    assert plan["pairs_total"] == 32
    assert plan["wanted"] == plan["placed"] == {"setupnet": 5, "psbot": 2, "foulplay": 2}
    for kind, n in plan["placed"].items():
        assert plan["net_p1"][kind] + plan["net_p2"][kind] == n and abs(plan["net_p1"][kind] - plan["net_p2"][kind]) <= 1
    for w, seats in zip(plan["workers"], _seats(plan)):
        assert len(seats) <= 4
        assert [k for k, _s in seats] == sorted((k for k, _s in seats), key=os_.SEAT_KINDS.index), "kinds in a fixed order"
    # spread thinly: no worker has more than one more opponent pair than another
    counts = [len(s) for s in _seats(plan)]
    assert max(counts) - min(counts) <= 1 and sum(counts) == 9
    # the Foul Play hosts have helpers, the others none
    for w in plan["workers"]:
        assert (w["helpers"] > 0) == any(s["kind"] == "foulplay" for s in w["seats"])
    assert "setupnet 5 of 32 pairs" in os_.describe_plan(plan)


def test_the_plan_is_the_same_every_time():
    args = ([4, 4, 3, 3, 2], 32, _cfg(setupnet=0.3, psbot=0.1, foulplay=0.1))
    first = os_.plan_seats(*args)
    for _ in range(5):
        again = os_.plan_seats(*args)
        assert _seats(again) == _seats(first) and [w["helpers"] for w in again["workers"]] == [w["helpers"] for w in first["workers"]]
    # the order the shares are given in does not matter
    swapped = os_.plan_seats(args[0], 32, _cfg(foulplay=0.1, psbot=0.1, setupnet=0.3))
    assert _seats(swapped) == _seats(first)


def test_small_runs_get_at_least_one_pair_and_never_more_than_they_have():
    plan = os_.plan_seats([1, 1], 32, _cfg(setupnet=0.05))
    assert plan["placed"]["setupnet"] == 1 and sum(len(s) for s in _seats(plan)) == 1
    plan = os_.plan_seats([1, 1], 32, _cfg(setupnet=0.4, psbot=0.3, foulplay=0.3))
    assert plan["placed"] == {"setupnet": 1, "psbot": 1, "foulplay": 0}, "Foul Play, the costliest, gives back first"
    plan = os_.plan_seats([0, 3, 0], 32, _cfg(setupnet=1.0))
    assert _seats(plan)[0] == [] and _seats(plan)[2] == [] and len(_seats(plan)[1]) == 3
    assert os_.plan_seats([4, 4], 32, None)["placed"] == {k: 0 for k in os_.SEAT_KINDS}
    assert os_.plan_seats([], 32, _cfg(setupnet=0.5))["placed"]["setupnet"] == 0


def test_foul_play_pairs_go_to_at_most_max_helpers_workers_and_the_helpers_are_capped():
    plan = os_.plan_seats([4] * 8, 32, _cfg(foulplay=0.5, fp_max_helpers=2))
    hosts = [i for i, w in enumerate(plan["workers"]) if w["seats"]]
    assert len(hosts) == 2 and plan["placed"]["foulplay"] == 8 < plan["wanted"]["foulplay"] == 16
    assert plan["helpers"] <= 2 and plan["helpers_capped"] == 2
    assert "FEWER THAN WANTED" in os_.describe_plan(plan)
    # sized by the rule: enough helpers for a decision of every side about every ROUND_TARGET_S
    plan = os_.plan_seats([4] * 4, 8, _cfg(foulplay=0.25, fp_max_helpers=100))
    for w in plan["workers"]:
        sides = 8 * sum(1 for s in w["seats"] if s["kind"] == "foulplay")
        assert w["helpers"] == os_.default_helpers(sides, 200)
    # asked for: dealt to the hosts with the most sides per helper
    plan = os_.plan_seats([4] * 4, 8, _cfg(foulplay=0.5, fp_helpers=6, fp_max_helpers=100))
    assert plan["helpers"] == 6 and all(w["helpers"] >= 1 for w in plan["workers"] if w["seats"])


def test_the_flags():
    parser = argparse.ArgumentParser()
    os_.add_arguments(parser)
    assert os_.from_args(parser.parse_args([])) is None
    assert os_.from_args(parser.parse_args(["--opponent-seats", "setupnet:0"])) is None
    cfg = os_.from_args(parser.parse_args(["--opponent-seats", "setupnet:0.15, psbot:0.05", "--setupnet-p", "0.3",
                                           "--setupnet-max-stage", "6"]))
    assert cfg.share("setupnet") == 0.15 and cfg.share("psbot") == 0.05 and cfg.share("foulplay") == 0.0
    assert cfg.setupnet == SetupNetSettings(p=0.3, max_stage=6)
    cfg = os_.from_args(parser.parse_args(["--opponent-seats", "foulplay:0.05", "--seat-foulplay-iterations", "4000"]))
    assert cfg.foulplay.iterations == 4000
    for bad in (["--opponent-seats", "foulplay:0.05"],                         # no iterations
                ["--opponent-seats", "bogus:0.1"], ["--opponent-seats", "bogus:0"],
                ["--opponent-seats", "setupnet"], ["--opponent-seats", "setupnet:x"],
                ["--opponent-seats", "setupnet:0.6,psbot:0.6"],
                ["--opponent-seats", "setupnet:0.1,setupnet:0.1"],
                ["--opponent-seats", "setupnet:0.1", "--setupnet-p", "1.5"],
                ["--opponent-seats", "setupnet:0.1", "--setupnet-max-stage", "0"]):
        with pytest.raises(ValueError):
            os_.from_args(parser.parse_args(bad))


def test_what_train_py_hands_ray():
    plan = os_.plan_seats([2, 2], 32, _cfg(setupnet=0.25))
    with_seats = [w for w in plan["workers"] if w["seats"]]
    without = [w for w in plan["workers"] if not w["seats"]]
    assert len(with_seats) == 1 and len(without) == 1
    assert os_.worker_class(with_seats[0]) is os_.OpponentSeatsRolloutWorker
    assert os_.worker_class(without[0]) is worker.RolloutWorker and os_.worker_class(None) is worker.RolloutWorker
    assert os_.worker_kwargs(without[0]) == {} and os_.worker_kwargs(None) == {}
    entry = os_.worker_kwargs(with_seats[0])["seats"]
    assert pickle.loads(pickle.dumps(entry)) == entry, "Ray pickles it"


def test_the_seats_worker_and_wrapper_run_the_base_classes_own_code():
    """What makes a trajectory, a reward and an episode for the learner is not overridden anywhere."""
    for name in ("run", "_run_rl_loop", "_finalize_trajectory", "_cleanup_trajectory"):
        assert getattr(os_.OpponentSeatsRolloutWorker, name) is getattr(worker.RolloutWorker, name), name
    for name in ("step", "release_slot", "_run_bg_loop", "_maintenance_loop"):
        assert getattr(os_.SeatsVectorWrapper, name) is getattr(worker.PokeEnvVectorWrapper, name), name


# --------------------------------------------------------------------------- battles built from a request


def make_battle(moves=("swordsdance", "earthquake"), hp=1.0, status=None, boosts=None, opp="Blissey", opp_details="Blissey, L88, F",
                opp_status=None, opp_ability=None, opp_boosts=None, before=()):
    """Our Espeon (tests/fixtures/tera_request.json) with these moves, HP and status, against `opp`."""
    request = copy.deepcopy(REQUEST)
    mon = request["side"]["pokemon"][0]
    maxhp = int(mon["condition"].split("/")[1].split()[0])
    mon["condition"] = "%d/%d%s" % (max(1, round(hp * maxhp)), maxhp, " " + status if status else "")
    mon["moves"] = list(moves)
    request["active"][0]["moves"] = [{"move": m, "id": m, "pp": 8, "maxpp": 8, "target": "normal", "disabled": False} for m in moves]
    battle = Battle("battle-gen9randombattle-1", "tester", logging.getLogger("test_opponent_seats"), gen=9)
    battle.parse_request(request)
    for line in before:
        battle.parse_message(["", *line])
    battle.parse_message(["", "switch", "p2a: " + opp, opp_details, "100/100"])
    for stat, n in (boosts or {}).items():
        battle.parse_message(["", "-boost", "p1a: Espeon", stat, str(n)])
    for stat, n in (opp_boosts or {}).items():
        battle.parse_message(["", "-boost", "p2a: " + opp, stat, str(n)])
    if opp_status:
        battle.parse_message(["", "-status", "p2a: " + opp, opp_status])
    if opp_ability:
        battle.parse_message(["", "-ability", "p2a: " + opp, opp_ability])
    return battle


def pick(battle, **kw):
    rule = setup_rules.setup_rule(battle, **kw)
    return None if rule is None else (rule[0], rule[1].id)


def test_the_battles_are_what_they_say():
    b = make_battle(hp=0.4, status="slp", boosts={"spa": 2})
    assert b.active_pokemon.species == "espeon" and b.active_pokemon.status == Status.SLP
    assert abs(b.active_pokemon.current_hp_fraction - 0.4) < 0.01 and b.active_pokemon.boosts["spa"] == 2
    assert [m.id for m in b.available_moves] == ["swordsdance", "earthquake"] and b.opponent_active_pokemon.species == "blissey"


def test_set_up_when_safe_up_to_max_stage_the_move_that_raises_most():
    assert pick(make_battle()) == ("setup", "swordsdance")
    assert pick(make_battle(boosts={"atk": 4})) is None                                  # at max_stage: the network's move
    assert pick(make_battle(boosts={"atk": 4}), max_stage=6) == ("setup", "swordsdance")
    assert pick(make_battle(boosts={"atk": 2}), max_stage=2) is None
    assert pick(make_battle(("shellsmash", "calmmind"))) == ("setup", "shellsmash")
    assert pick(make_battle(("calmmind", "psychic"), boosts={"spa": 4, "spd": 3})) == ("setup", "calmmind")   # Sp. Def still has room
    assert pick(make_battle(("bellydrum", "earthquake"), hp=0.55)) is None
    assert pick(make_battle(("psychic", "shadowball"))) is None                          # nothing to set up with


def test_no_set_up_in_danger():
    assert pick(make_battle(hp=0.45)) is None                                           # half its HP or less
    dark = dict(opp="Kingambit", opp_details="Kingambit, L80, M")                      # Dark hits Espeon twice over
    assert pick(make_battle(hp=0.8, **dark)) is None
    assert pick(make_battle(hp=0.9, **dark)) == ("setup", "swordsdance")
    assert pick(make_battle(hp=0.8, opp_boosts={"spa": 2})) is None                     # a boosted attacker
    assert pick(make_battle(hp=0.8, opp_boosts={"spa": 1})) == ("setup", "swordsdance")


def test_sleep_when_the_target_can_fall_asleep():
    assert pick(make_battle(("spore", "swordsdance"))) == ("sleep", "spore")
    assert pick(make_battle(("spore", "swordsdance"), opp_status="par")) == ("setup", "swordsdance")
    asleep = (("switch", "p2a: Garchomp", "Garchomp, L80, M", "100/100"), ("-status", "p2a: Garchomp", "slp"))
    assert pick(make_battle(("spore", "swordsdance"), before=asleep)) == ("setup", "swordsdance")   # Sleep Clause
    grass = dict(opp="Venusaur", opp_details="Venusaur, L84, M")
    assert pick(make_battle(("spore", "swordsdance"), **grass)) == ("setup", "swordsdance")         # powder on a Grass type
    assert pick(make_battle(("hypnosis", "psychic"), **grass)) == ("sleep", "hypnosis")
    assert pick(make_battle(("spore", "psychic"), opp_ability="Insomnia")) is None
    assert pick(make_battle(("spore", "swordsdance"), hp=0.3)) == ("sleep", "spore"), "sleep comes before the danger check"
    sub = make_battle(("spore", "psychic"))
    sub.parse_message(["", "-start", "p2a: Blissey", "Substitute"])
    assert pick(sub) is None
    terrain = make_battle(("spore", "psychic"))
    terrain.parse_message(["", "-fieldstart", "move: Electric Terrain"])
    assert Field.ELECTRIC_TERRAIN in terrain.fields and pick(terrain) is None


def test_sleep_talk_when_asleep_and_recover_when_boosted_and_hurt():
    assert pick(make_battle(("sleeptalk", "curse", "rest"), status="slp")) == ("sleeptalk", "sleeptalk")
    assert pick(make_battle(("curse", "rest"), status="slp")) is None                  # asleep without Sleep Talk
    assert pick(make_battle(("calmmind", "recover", "psychic"), hp=0.4, boosts={"spa": 2})) == ("recover", "recover")
    assert pick(make_battle(("calmmind", "recover", "psychic"), hp=0.4, boosts={"spa": 2}), recover=False) is None
    assert pick(make_battle(("calmmind", "recover", "psychic"), hp=0.4)) is None        # not boosted: no heal rule, in danger
    assert setup_rules.kind_of("sleeppowder") == "sleep" and setup_rules.kind_of("roost") == "recover"


# --------------------------------------------------------------------------- setupnet's override


def offline_player(cls, username: str, **kwargs):
    """A real poke-env player that never connects: its message handler is called by hand."""
    from poke_env import AccountConfiguration, ServerConfiguration

    player = cls(account_configuration=AccountConfiguration(username, None),
                 server_configuration=ServerConfiguration("ws://127.0.0.1:9/showdown/websocket", "http://127.0.0.1:9/action.php?"),
                 battle_format="gen9randombattle", start_listening=False, log_level=40, **kwargs)
    player.sent = []

    async def send_message(message, room="", message_2=None):
        player.sent.append((room, message))

    player.ps_client.send_message = send_message
    return player


def run_cfg():
    base = config.RunConfig.default()
    return dataclasses.replace(base, learner=dataclasses.replace(base.learner, mode="ppo"))


def seat_kwargs(q=None, futures=None, slots: int = 64):
    return dict(cfg=run_cfg(), event_queue=q if q is not None else queue.Queue(), action_futures=futures if futures is not None else {},
                ep_sem=asyncio.Semaphore(slots))


def setupnet_seat(p=1.0, username="setupseat", seed="1", seat="b", q=None, futures=None, **settings):
    stats = os_.SeatStats("setupnet")
    player = offline_player(os_.SetupNetSeatPlayer, username, setupnet=SetupNetSettings(p=p, **settings), rng_seed=seed,
                            seat_stats=stats, seat=seat, **seat_kwargs(q, futures))
    return player, stats


def _move(battle, move_id):
    return next(m for m in battle.available_moves if m.id == move_id)


def test_with_p_1_the_rules_move_replaces_the_networks_and_with_p_0_never():
    player, stats = setupnet_seat(p=1.0)
    battle = make_battle()
    order, kind = player.setupnet_order(battle, _move(battle, "earthquake"), {})
    assert kind == "setup" and order.order.id == "swordsdance"
    assert stats.extra["override_setup"] == stats.bot_decisions == stats.extra["setup_played"] == 1 and stats.net_decisions == 0

    player, stats = setupnet_seat(p=0.0)
    order, kind = player.setupnet_order(battle, _move(battle, "earthquake"), {})
    assert kind is None and order.order.id == "earthquake" and stats.net_decisions == 1 and stats.bot_decisions == 0
    assert "drawn" not in stats.extra and "setup_played" not in stats.extra

    # the network's own pick of the rule's move stands, Tera and all
    player, stats = setupnet_seat(p=1.0)
    order, kind = player.setupnet_order(battle, _move(battle, "swordsdance"), {"terastallize": True})
    assert kind is None and order.order.id == "swordsdance" and order.terastallize
    assert stats.extra["agreed"] == 1 and stats.extra["net_setup"] == 1 and stats.extra["setup_played"] == 1
    # no rule applies: the network's move
    player, stats = setupnet_seat(p=1.0)
    low = make_battle(hp=0.3)
    order, kind = player.setupnet_order(low, _move(low, "earthquake"), {})
    assert kind is None and order.order.id == "earthquake" and stats.extra["eligible"] == 1 and "rule_ready" not in stats.extra


def test_a_switch_the_network_chose_is_kept_unless_told_otherwise():
    battle = make_battle()
    target = battle.available_switches[0]
    player, stats = setupnet_seat(p=1.0)
    order, kind = player.setupnet_order(battle, target, {})
    assert kind is None and order.order is target and "eligible" not in stats.extra
    player, stats = setupnet_seat(p=1.0, keep_switches=False)
    order, kind = player.setupnet_order(battle, target, {})
    assert kind == "setup" and order.order.id == "swordsdance"
    player, stats = setupnet_seat(p=1.0)
    order, kind = player.setupnet_order(battle, "DEFAULT", {})
    assert kind is None and order.message == "/choose default"


def test_p_is_the_share_of_the_draws_and_a_seed_gives_the_same_draws():
    battle = make_battle()
    eq = _move(battle, "earthquake")
    runs = []
    for seed in ("7", "7", "8"):
        player, stats = setupnet_seat(p=0.3, seed=seed)
        runs.append([player.setupnet_order(battle, eq, {})[1] for _ in range(4000)])
        assert stats.extra["rule_ready"] == 4000
    assert runs[0] == runs[1] and runs[0] != runs[2]
    share = sum(1 for k in runs[0] if k) / 4000.0
    assert abs(share - 0.3) < 0.03


# --------------------------------------------------------------------------- the seats on recorded battles


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


def _battles(revive=None):
    def has_revive(events):
        return any(e["t"] == "order" and (e.get("fp_reply") or {}).get("revive") for e in events)
    return [e for e in load_fixture() if revive is None or has_revive(e) == revive]


def split(room: str, data: str):
    return [[">" + room]] + [line.split("|") for line in data.split("\n")]


def _feed(player, events):
    async def feed_all():
        for event in events:
            if event["t"] == "msg":
                await player._handle_battle_message(split(event["room"], event["data"]))
    asyncio.run_coroutine_threadsafe(feed_all(), player.ps_client.loop).result(timeout=120)


def _sent(player):
    return [m for _room, m in player.sent if m.startswith("/choose")]


class Answerer:
    """Stands in for the rollout loop: keeps every event the players put on the queue and answers each STEP with the first legal action."""

    def __init__(self):
        self.queue = queue.Queue()
        self.futures = {}
        self.events = []
        from obs_assembler import ObservationAssembler
        self.m0, self.m1 = ObservationAssembler().offsets["action_mask"]
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop:
            try:
                event = self.queue.get(timeout=0.02)
            except queue.Empty:
                continue
            self.events.append(event)
            if event[0] == "STEP":
                legal = np.flatnonzero(np.asarray(event[2][self.m0:self.m1], dtype=np.float32) > 0.5)
                fut, loop = self.futures.pop(event[1])
                loop.call_soon_threadsafe(fut.set_result, int(legal[0]) if len(legal) else 0)

    def settle(self):
        assert tfb.wait_until(lambda: self.queue.empty(), timeout=10)
        time.sleep(0.05)
        return self

    def close(self):
        self._stop = True
        self._thread.join(timeout=5)


def _kinds(events):
    return [e[0] for e in events]


def test_the_recorded_battles_are_there():
    """The tests below loop over them: an empty fixture would pass them all."""
    battles = _battles()
    assert len(battles) == 7 and len(_battles(revive=False)) == 6
    assert all(len([e for e in b if e["t"] == "order"]) >= 10 and "won" in b[-1] and "username" in b[-1] for b in battles)


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_the_setupnet_seat_puts_a_step_per_decision_and_one_cleanup_never_a_done(p, monkeypatch):
    monkeypatch.chdir(REPO)
    answerer = Answerer()
    try:
        for seat, role in (("b", "p1"), ("a", "p2")):
            for events in _battles():
                username = events[-1]["username"]
                decisions = len([e for e in events if e["t"] == "order"])
                player, stats = setupnet_seat(p=p, username=username, seat=seat, q=answerer.queue, futures=answerer.futures)
                before = len(answerer.events)
                _feed(player, events)
                answerer.settle()
                mine = answerer.events[before:]
                tag = "%s_%s" % (username, events[0]["room"])
                assert _kinds(mine) == ["STEP"] * decisions + ["CLEANUP"], "no DONE: nothing of this seat reaches the learner"
                assert all(e[1] == tag for e in mine)
                assert len(_sent(player)) == decisions == stats.decisions
                assert stats.net_decisions + stats.bot_decisions == decisions
                if p == 0.0:
                    assert stats.bot_decisions == 0
                assert player.ep_sem._value == 63, "one slot taken at the first decision; the worker gives it back on CLEANUP"
                assert stats.games[role] == 1 and stats.mixed_games == 0
                assert stats.wins[role] == (0 if events[-1]["won"] else 1), "the seat lost exactly when the network won"
                assert not player._battles and not player._battle_starts and not player._episode_slot_state
                assert not player._seat_net_moves
    finally:
        answerer.close()


class FakeEngine:
    """PSBot's engine: takes the lines, answers the first legal move or switch of the request (or refuses)."""

    def __init__(self, answer=True):
        self.answer = answer
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)

    async def call(self, msg, timeout=20.0):
        from psbot_bridge import PSBotEngineError
        if not self.answer:
            raise PSBotEngineError("the engine is down")
        req = msg.get("request") or {}
        if req.get("active") and not req.get("forceSwitch"):
            for i, m in enumerate(req["active"][0]["moves"]):
                if not m.get("disabled"):
                    return {"choice": "move %d" % (i + 1)}
        for i, mon in enumerate((req.get("side") or {}).get("pokemon") or []):
            if not mon.get("active") and not str(mon.get("condition", "")).endswith(" fnt"):
                return {"choice": "switch %d" % (i + 1)}
        return {"choice": "default"}


def psbot_seat(answerer, username, engine, seat="b"):
    stats = os_.SeatStats("psbot")
    player = offline_player(os_.PSBotSeatPlayer, username, seat_stats=stats, seat=seat,
                            **seat_kwargs(answerer.queue, answerer.futures))
    player._psbot_engine = engine
    return player, stats


def test_psbots_seat_hands_nothing_to_the_learner(monkeypatch):
    monkeypatch.chdir(REPO)
    answerer = Answerer()
    try:
        for events in _battles(revive=False):
            username = events[-1]["username"]
            decisions = len([e for e in events if e["t"] == "order"])
            engine = FakeEngine(answer=True)
            player, stats = psbot_seat(answerer, username, engine)
            assert player.psbot_enabled
            before = len(answerer.events)
            _feed(player, events)
            answerer.settle()
            mine = _kinds(answerer.events[before:])
            assert "DONE" not in mine and "IMITATION" not in mine
            assert mine.count("STEP") == stats.net_decisions and mine.count("CLEANUP") == (1 if stats.net_decisions else 0)
            assert stats.bot_decisions + stats.net_decisions == decisions == len(_sent(player))
            assert stats.bot_decisions > decisions // 2, "PSBot played the battle"
            assert any(m["op"] == "feed" for m in engine.sent) and engine.sent[-1]["op"] == "end"
            assert stats.games["p1"] + stats.mixed_games == 1

            # the engine down: the network plays every decision, the steps are dropped at the end
            down = FakeEngine(answer=False)
            player, stats = psbot_seat(answerer, username, down)
            before = len(answerer.events)
            _feed(player, events)
            answerer.settle()
            assert _kinds(answerer.events[before:]) == ["STEP"] * decisions + ["CLEANUP"]
            assert stats.net_decisions == decisions and stats.mixed_games == 1 and stats.games == {"p1": 0, "p2": 0}
            assert player.ep_sem._value == 63 and not player._seat_net_moves
    finally:
        answerer.close()


def foulplay_seat(answerer, username, bridge, iterations=500, seat="b"):
    stats = os_.SeatStats("foulplay")
    player = offline_player(os_.FoulPlaySeatPlayer, username, fp_bridge=bridge,
                            fp_settings=FoulPlaySeatSettings(iterations=iterations, sizing_ms=50, timeout_s=8.0),
                            seat_stats=stats, seat=seat, **seat_kwargs(answerer.queue, answerer.futures))
    return player, stats


def test_foul_plays_seat_searches_the_fixed_iterations_and_hands_nothing_to_the_learner(fake_helper, monkeypatch):
    monkeypatch.chdir(REPO)
    answerer = Answerer()
    bridge = tfb.make_bridge(fake_helper)
    try:
        for events in _battles(revive=False):
            username = events[-1]["username"]
            decisions = len([e for e in events if e["t"] == "order"])
            player, stats = foulplay_seat(answerer, username, bridge)
            before = len(answerer.events)
            _feed(player, events)
            answerer.settle()
            assert answerer.events[before:] == [], "Foul Play answered every decision: no event at all"
            assert stats.bot_decisions == decisions == len(_sent(player)) and stats.net_decisions == 0
            assert "iterations_not_fixed" not in stats.extra, "every answer was searched at the fixed iterations"
            assert stats.games["p1"] == 1 and stats.mixed_games == 0 and player.ep_sem._value == 64
        assert bridge.stats()["battles"] == 0, "every finished battle was forgotten by the bridge"
    finally:
        answerer.close()
        bridge.close()

    refusing = tfb.make_bridge(fake_helper, env={"FAKE_REFUSE": "PanicException: Invalid PokemonMoveIndex: 4"})
    answerer = Answerer()
    try:
        events = _battles(revive=False)[0]
        username = events[-1]["username"]
        decisions = len([e for e in events if e["t"] == "order"])
        player, stats = foulplay_seat(answerer, username, refusing)
        _feed(player, events)
        answerer.settle()
        assert _kinds(answerer.events) == ["STEP"] * decisions + ["CLEANUP"]
        assert stats.extra.get("handed_error") == 1 and stats.mixed_games == 1 and stats.net_decisions == decisions
        assert not player._fp_given_up and not player._seat_net_moves
    finally:
        answerer.close()
        refusing.close()


# --------------------------------------------------------------------------- through the worker's own rollout loop


class _Call:
    def __init__(self, fn):
        self.remote = fn


class _LoopHarness:
    """worker.RolloutWorker's own loop with an inference actor (first legal action) and a learner (keeps what it is sent) standing in."""

    def __init__(self, monkeypatch, worker_class):
        from obs_assembler import ObservationAssembler
        monkeypatch.setattr(worker, "ray", SimpleNamespace(get=lambda value, **_k: value))
        asm = ObservationAssembler()
        m0, m1 = asm.offsets["action_mask"]
        self.sent, self.cleared, self.asked, self.released = [], [], [], []

        def infer(tags, obs):
            self.asked.extend(tags)
            obs32 = np.asarray(obs, dtype=np.float32)
            legal = obs32[:, m0:m1] > 0.5
            acts = np.array([int(np.flatnonzero(row)[0]) if row.any() else 0 for row in legal], dtype=np.int64)
            mark = obs32.sum(axis=1)
            return acts, (mark * 1e-3).astype(np.float32), (mark * 2e-3).astype(np.float32), (mark * 3e-3).astype(np.float32)

        vec = SimpleNamespace(event_queue=queue.Queue(), action_futures={}, release_slot=lambda: self.released.append(1))
        vec.step = lambda actions: worker.PokeEnvVectorWrapper.step(vec, actions)
        rw = object.__new__(worker_class)
        rw.cfg = run_cfg()
        rw.inference_actor = SimpleNamespace(infer_batch=_Call(infer), clear_cache=_Call(self.cleared.append))
        rw.learner_client = worker.SyncLearnerClient(SimpleNamespace(submit_packed_batch=_Call(lambda *packed: self.sent.append(packed))), rw.cfg)
        rw.vec_env, rw._traj, rw.assembler = vec, {}, rw.cfg.make_assembler()
        self.worker, self.vec = rw, vec
        threading.Thread(target=rw._run_rl_loop, daemon=True).start()      # it waits on the queue for the rest of the test run

    def settle(self):
        assert tfb.wait_until(lambda: self.vec.event_queue.empty() and self.worker.learner_client.q.empty(), timeout=20)
        time.sleep(0.3)
        return self


def test_through_the_workers_own_rollout_loop_only_the_trained_seat_reaches_the_learner(monkeypatch):
    """One recorded battle, twice: as the trained seat it is one episode for the learner; as the setupnet seat nothing, and everything is given back."""
    monkeypatch.chdir(REPO)
    events = _battles(revive=False)[0]
    username, room = events[-1]["username"], events[0]["room"]
    tag = "%s_%s" % (username, room)
    decisions = len([e for e in events if e["t"] == "order"])

    harness = _LoopHarness(monkeypatch, os_.OpponentSeatsRolloutWorker)
    player = offline_player(worker.SyncBridgePlayer, username, **seat_kwargs(harness.vec.event_queue, harness.vec.action_futures))
    _feed(player, events)
    harness.settle()
    assert len(harness.sent) == 1 and harness.asked == [tag] * decisions
    lengths = harness.sent[0][-1]
    assert lengths.tolist() == [decisions] and harness.cleared == [tag] and len(harness.released) == 1

    harness = _LoopHarness(monkeypatch, os_.OpponentSeatsRolloutWorker)
    stats = os_.SeatStats("setupnet")
    player = offline_player(os_.SetupNetSeatPlayer, username, setupnet=SetupNetSettings(p=1.0), rng_seed="3", seat_stats=stats, seat="b",
                            **seat_kwargs(harness.vec.event_queue, harness.vec.action_futures))
    _feed(player, events)
    harness.settle()
    assert harness.asked == [tag] * decisions, "the network was asked at every decision of the seat"
    assert harness.sent == [], "and not one of those steps was handed to the learner"
    assert harness.cleared == [tag] and len(harness.released) == 1 and harness.worker._traj == {}
    assert len(_sent(player)) == decisions and stats.games["p1"] == 1


# --------------------------------------------------------------------------- the wrapper's fallbacks and the log line


def test_a_kind_that_cannot_be_used_is_played_as_self_play(monkeypatch):
    made = []
    monkeypatch.setattr(worker.PokeEnvVectorWrapper, "__init__", lambda self, cfg, pairs, port: made.append((pairs, port)))
    monkeypatch.setattr(os_, "_psbot_usable", lambda: "no node here")
    monkeypatch.setattr(os_, "_foulplay_usable", lambda: "Foul Play is not installed")
    specs = [{"kind": "setupnet", "seat": "b"}, {"kind": "psbot", "seat": "a"}, {"kind": "foulplay", "seat": "b"}, {"kind": "nope", "seat": "a"}]
    settings = _cfg(setupnet=0.25, psbot=0.25, foulplay=0.25)
    wrapper = os_.SeatsVectorWrapper(run_cfg(), 4, 10399, specs, settings)
    assert made == [(4, 10399)]
    assert [s["kind"] for s in wrapper.seat_specs] == ["setupnet"]
    assert set(wrapper.seat_unavailable) == {"psbot", "foulplay"} and wrapper.fp_bridge is None
    beat = wrapper.seats_heartbeat()
    assert beat["pairs"] == {"setupnet": 1, "psbot": 0, "foulplay": 0} and beat["planned"] == {"setupnet": 1, "psbot": 1, "foulplay": 1}
    line = os_.report_line([beat])
    assert line.startswith("[Seats] setupnet: pairs 1, games 0") and "psbot: pairs 0 (of 1 planned: no node here)" in line

    imitation = dataclasses.replace(run_cfg(), learner=dataclasses.replace(run_cfg().learner, mode="imitation"))
    wrapper = os_.SeatsVectorWrapper(imitation, 4, 10399, specs[:1], settings)
    assert wrapper.seat_specs == [] and "teacher" in wrapper.seat_unavailable["setupnet"]


def test_the_log_line_sums_the_workers():
    def beat(port, games, wins, overrides):
        stats = {k: os_.SeatStats(k).as_dict() for k in os_.SEAT_KINDS}
        stats["setupnet"].update(games_p1=games, wins_p1=wins, decisions=100, bot_decisions=overrides, override_setup=overrides,
                                 setup_played=overrides + 2)
        return {"port": port, "pairs": {"setupnet": 1}, "planned": {"setupnet": 1}, "unavailable": {}, "failed_start": "", "stats": stats}

    line = os_.report_line([beat(1, 10, 6, 7), beat(2, 10, 8, 3), {"port": 3, "report_error": "x"}, None])
    assert line == ("[Seats] setupnet: pairs 2, games 20, network won 70.0% (p1 70.0% of 20), rule moves 10 of 200 decisions "
                    "(setup 10), set-up moves played 14")
    assert os_.report_line([]) == "[Seats] none"
