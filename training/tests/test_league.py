"""
The league run's pieces (bench/notes/LEAGUE.md): the frozen-network seats (frozen_pool.py, opponent seat
kind "frozen"), the worker's queue to the learner (--learn-queue-episodes) and the lag measurement
(--lag-stats). Every one off by default, and off nothing is built differently.

    python -m pytest tests/test_league.py -q
"""
from __future__ import annotations

import asyncio
import dataclasses
import os
import queue
import random
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
torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

import config  # noqa: E402
import frozen_pool as fp  # noqa: E402
import league  # noqa: E402
import opponent_seats as os_  # noqa: E402
import test_opponent_seats as tos  # noqa: E402  (its recorded battles, offline players and loop harness)
import worker  # noqa: E402
from opponent_seats import OpponentSeatsConfig  # noqa: E402


def _frozen(nets=(("m", 0.6), ("s", 0.4)), past=0, past_w=0.0, **kw):
    return fp.FrozenSeatSettings(nets=tuple(fp.FrozenNetSpec(n, f"/x/{n}.pt", w) for n, w in nets), past=past, past_weight=past_w, **kw)


# --------------------------------------------------------------------------- off by default


def test_without_the_flags_nothing_reaches_a_worker_or_the_learner():
    import train
    args = train.parse_args([])
    opts = league.from_args(args)
    assert opts.is_default() and league.worker_kwargs(opts) == {} and league.learner_kwargs(opts) == {}
    assert os_.from_args(args) is None
    for argv in (["--opponent-seats", "setupnet:0.15"], ["--opponent-seats", "setupnet:0.15,foulplay:0.1", "--seat-foulplay-iterations", "88000"]):
        st = os_.from_args(train.parse_args(argv))
        assert st.frozen is None and os_.kinds_of(st) == os_.SEAT_KINDS


def test_a_plan_without_frozen_seats_is_built_over_the_three_kinds_as_before():
    for shares in ({"setupnet": 0.15}, {"setupnet": 0.25, "psbot": 0.25, "foulplay": 0.25}, {"foulplay": 0.1}):
        plan = os_.plan_seats([4] * 36, 16, tos._cfg(**shares))
        for key in ("wanted", "placed", "net_p1", "net_p2"):
            assert tuple(plan[key]) == os_.SEAT_KINDS
        assert "frozen" not in os_.describe_plan(plan)


def test_the_workers_queue_is_160_episodes_unless_asked():
    learner = SimpleNamespace(submit_packed_batch=SimpleNamespace(remote=lambda *a, **k: None))
    cfg = config.RunConfig.default()
    assert worker.SyncLearnerClient(learner, cfg).q.maxsize == 160
    small = worker.SyncLearnerClient(learner, cfg, maxsize=32)
    assert small.q.maxsize == 32
    assert worker.LAG_STATS is False


def test_eviction_is_counted_and_the_newest_kept(monkeypatch):
    blocked = threading.Event()
    learner = SimpleNamespace(submit_packed_batch=SimpleNamespace(remote=lambda *a, **k: blocked.wait(5)))
    monkeypatch.setattr(worker, "ray", SimpleNamespace(get=lambda v, **k: v))
    cl = worker.SyncLearnerClient(learner, config.RunConfig.default(), maxsize=3)
    one = np.zeros((2, 4), np.float16)

    def submit(i):
        cl.submit_episode(one, np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), np.full(2, i, np.float32), np.ones(2))

    submit(0)
    deadline = time.time() + 5
    while cl.q.qsize() and time.time() < deadline:      # the thread takes it and waits on the learner
        time.sleep(0.005)
    time.sleep(0.05)
    for i in range(1, 10):
        submit(i)
    assert cl.submitted == 10
    # of the 9 after the one in flight, the queue holds the newest 3 and dropped the 6 oldest
    assert cl.q.qsize() == 3 and cl.evicted == 6
    newest = [cl.q.get_nowait()[5][0] for _ in range(3)]
    assert newest == [7.0, 8.0, 9.0]
    blocked.set()


def test_the_league_flags():
    import train
    a = train.parse_args(["--learn-queue-episodes", "32", "--lag-stats"])
    opts = league.from_args(a)
    assert league.worker_kwargs(opts) == {"learn_queue": 32, "lag_stats": True} and league.learner_kwargs(opts) == {"lag_stats": True}
    with pytest.raises(SystemExit):
        train.parse_args(["--learn-queue-episodes", "0"])


def test_the_lag_line():
    beats = [{"lag": {"submitted": 100, "evicted": 30, "q": 32, "maxsize": 32}}, {"lag": {"submitted": 50, "evicted": 0, "q": 5, "maxsize": 32}}, {}]
    assert league.lag_line(beats) == ("[Lag] worker queues: 30 of 150 episodes dropped as the oldest (20.0%), 37 held now "
                                      "(max 32 each, 2 workers)")
    assert league.lag_line([{}]) is None


def test_the_learners_lag_numbers():
    from learner import LearnerActor
    fake = SimpleNamespace(_push_times=[100.0, 106.0, 112.0, 118.0])
    now = time.time
    try:
        time.time = lambda: 120.0
        out = LearnerActor._lag_info(fake, torch.tensor([119.0, 110.0, 101.0, 0.0], dtype=torch.float64))
    finally:
        time.time = now
    assert out["lag_rows"] == 3.0
    # 119: no push after it (0 behind); 110: pushes at 112 and 118 (2); 101: 106, 112, 118 (3)
    assert out["lag_mean"] == pytest.approx(5.0 / 3.0) and out["age_s"] == pytest.approx((1 + 10 + 19) / 3.0)
    assert LearnerActor._lag_info(fake, None) == {"lag_rows": 0.0}


def test_the_player_records_the_decisions_time_only_with_lag_stats(monkeypatch):
    monkeypatch.chdir(REPO)
    p = tos.offline_player(worker.SyncBridgePlayer, "lagoff", **tos.seat_kwargs())
    assert p.lag_on is False and p.step_features is False
    monkeypatch.setattr(worker, "LAG_STATS", True)
    p = tos.offline_player(worker.SyncBridgePlayer, "lagon", **tos.seat_kwargs())
    assert p.lag_on is True and p.step_features is True
    battle = tos.make_battle()
    t0 = time.time()
    p._note_step("lagon_x", battle, [])
    assert p._step_cols["lagon_x"]["t_act"][0] >= t0


# --------------------------------------------------------------------------- the frozen seats' settings and plan


def test_frozen_settings_and_weights():
    s = _frozen(nets=(("m", 0.3), ("s", 0.2)), past=4, past_w=0.5)
    assert s.slots == ["m", "s", "past0", "past1", "past2", "past3"]
    w = s.slot_weights()
    assert w == pytest.approx([0.3, 0.2, 0.125, 0.125, 0.125, 0.125]) and sum(w) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        _frozen(nets=(("m", 1.0), ("m", 1.0)))
    with pytest.raises(ValueError):
        _frozen(nets=(("past1", 1.0),))
    with pytest.raises(ValueError):
        _frozen(nets=(), past=0, past_w=1.0)
    with pytest.raises(ValueError):
        _frozen(past_pick="newest")
    with pytest.raises(ValueError):
        _frozen(nets=(("m", 0.0),)).slot_weights()
    assert fp.parse_nets("a=/p/A.pt:0.3, b=C:/q/B.pt:0.2") == (fp.FrozenNetSpec("a", "/p/A.pt", 0.3), fp.FrozenNetSpec("b", "C:/q/B.pt", 0.2))
    assert fp.parse_past("4:0.5") == (4, 0.5) and fp.parse_past("") == (0, 0.0)
    with pytest.raises(ValueError):
        fp.parse_nets("a=/p/A.pt")


def test_the_frozen_share_needs_nets_and_is_planned_after_the_other_kinds():
    with pytest.raises(ValueError):
        OpponentSeatsConfig(shares=(("frozen", 0.25),))
    st = OpponentSeatsConfig(shares=(("setupnet", 0.15), ("frozen", 0.25)), frozen=_frozen())
    plan = os_.plan_seats([4] * 36, 16, st)
    assert plan["wanted"] == {"setupnet": 22, "psbot": 0, "foulplay": 0, "frozen": 36} == plan["placed"]
    assert plan["net_p1"]["frozen"] == plan["net_p2"]["frozen"] == 18
    # the setupnet pairs land exactly where they did without frozen seats
    alone = os_.plan_seats([4] * 36, 16, tos._cfg(setupnet=0.15))
    assert [[s for s in w["seats"] if s["kind"] == "setupnet"] for w in plan["workers"]] == [w["seats"] for w in alone["workers"]]
    assert "frozen 36 of 144 pairs" in os_.describe_plan(plan)


def test_obs_keys():
    run = config.RunConfig.default().obs
    assert fp.obs_key_of(dataclasses.asdict(run), run) == fp.RUN_OBS
    league_obs = dataclasses.replace(run, own_tera_type=False, main_line=True, matchup=True, movesets=True)
    sfin = {"own_tera_type": True, "matchup": True, "movesets": True}
    key = fp.obs_key_of(sfin, league_obs)
    assert key != fp.RUN_OBS and fp.obs_config_for(key, league_obs) == dataclasses.replace(league_obs, own_tera_type=True, main_line=False)
    widened = {"own_tera_type": False, "matchup": True, "movesets": True, "main_line": True}
    assert fp.obs_key_of(widened, league_obs) == fp.RUN_OBS


def test_pick_past():
    files = [(f"/c/learner_update_{i:06d}.pt", float(i)) for i in range(10)]
    a = fp.pick_past(files, 4, "uniform", random.Random("1"))
    b = fp.pick_past(files, 4, "uniform", random.Random("1"))
    assert a == b and len(set(a)) == 4
    r = fp.pick_past(files, 2, "recent:3", random.Random("2"))
    assert set(r) <= {files[i][0] for i in (7, 8, 9)}
    assert fp.pick_past(files[:1], 3, "uniform", random.Random("3")) == [files[0][0]] * 3
    assert fp.pick_past([], 3, "uniform", random.Random("3")) == []


# --------------------------------------------------------------------------- the pool, with a stand-in network


class FakeNet(torch.nn.Module):
    """forward(obs, kv_cache, kv_cache_len) as the model has it: the action is the legal one at index (bias + history) mod 14."""

    n_layers, kv_cache_len, d_model = 1, 4, 2

    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def recurrent_dtype(self):
        return torch.float32

    def forward(self, obs, kv_cache=None, kv_cache_len=None):
        b = obs.shape[0]
        hist = kv_cache_len.float()
        logits = torch.full((b, 14), -50.0)
        idx = ((self.bias.detach() + hist).long() % 14)
        logits[torch.arange(b), idx] = 50.0
        new_kv = [k.clone() for k in kv_cache]
        return logits, torch.zeros(b, 3), torch.zeros(b), new_kv, kv_cache_len + 1


def _pool(settings, nets):
    """A FrozenPool whose files are the given FakeNet biases."""
    def load(path):
        name = os.path.basename(path)
        return {"model": {"bias": torch.tensor([float(nets[name])])}}
    return fp.FrozenPool(settings, FakeNet, "cpu", load=load)


def test_the_pool_pins_a_battle_to_its_net_and_forgets_it():
    s = _frozen(nets=(("m", 0.5), ("s", 0.5)))
    pool = _pool(s, {"m.pt": 0, "s.pt": 5})
    obs = np.ones((3, 20), np.float16)
    acts = pool.infer(["t1", "t2", "t3"], obs, [0, 1, 0], (0, 14))
    assert acts.tolist() == [0, 5, 0]
    acts = pool.infer(["t1", "t2"], obs[:2], [1, 0], (0, 14))      # the slots asked now do not matter: the battle keeps its net
    assert acts.tolist() == [1, 6], "the same nets, one turn of history more"
    assert pool.stats()["sides"] == 3 and pool.stats()["battles_by_slot"] == {"m": 2, "s": 1}
    pool.forget(["t1", "t2", "t3"])
    assert pool.stats()["sides"] == 0 and len(pool._kv_free) >= 3
    assert pool.infer(["t1"], obs[:1], [1], (0, 14)).tolist() == [5], "a new battle under an old tag starts afresh"


def test_the_pool_answers_only_legal_actions():
    s = _frozen(nets=(("m", 1.0),))
    pool = _pool(s, {"m.pt": 3})
    obs = np.zeros((1, 20), np.float16)
    obs[0, 9] = 1.0                                     # only action 9 is legal
    assert pool.infer(["a"], obs, [0], (0, 14)).tolist() == [9]
    none = np.zeros((1, 20), np.float16)                # no legal action: action 0, as the inference actor does
    assert pool.infer(["b"], none, [0], (0, 14)).tolist() == [0]


def test_the_frozen_nets_never_change(tmp_path):
    s = _frozen(nets=(("m", 1.0),))
    pool = _pool(s, {"m.pt": 2})
    before = {k: v.clone() for k, v in pool._nets[0].state_dict().items()}
    obs = np.ones((4, 20), np.float16)
    for i in range(20):
        pool.infer([f"x{i}a", f"x{i}b"], obs[:2], [0, 0], (0, 14))
    after = pool._nets[0].state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert all(not p.requires_grad for p in pool._nets[0].parameters())
    assert pool.changed_nets() == [] and pool.stats()["changed"] == []
    with torch.no_grad():
        pool._nets[0].bias.add_(1.0)                    # what the check is for
    assert pool.stats()["changed"] == ["m.pt"]
    assert "FROZEN NETS CHANGED: ['m.pt']" in league.frozen_line([pool.stats()])


def test_past_slots_are_picked_again_and_running_battles_keep_their_net(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    old = time.time() - 3600
    for i, bias in ((100, 1), (200, 2), (300, 3)):
        p = d / f"learner_update_{i:06d}.pt"
        p.write_bytes(b"x")
        os.utime(p, (old + i, old + i))
    biases = {"learner_update_000100.pt": 1, "learner_update_000200.pt": 2, "learner_update_000300.pt": 3, "learner_update_000400.pt": 4}
    s = fp.FrozenSeatSettings(nets=(), past=1, past_weight=1.0, past_pick="recent:1", refresh_s=0.01, past_min_age_s=60.0,
                              ckpt_dir=str(d), seed=1)
    pool = _pool(s, biases)
    obs = np.ones((1, 20), np.float16)
    assert pool.infer(["old"], obs, [0], (0, 14)).tolist() == [3], "the newest save old enough"
    newer = d / "learner_update_000400.pt"
    newer.write_bytes(b"x")
    os.utime(newer, (old + 400, old + 400))
    time.sleep(0.05)
    pool.infer(["old"], obs, [0], (0, 14))              # starts the refresh on its thread
    pool.s = dataclasses.replace(pool.s, refresh_s=3600.0)   # and no other one in this test
    deadline = time.time() + 10
    while pool._pending is None and time.time() < deadline:
        time.sleep(0.01)
    assert pool.infer(["new"], obs, [0], (0, 14)).tolist() == [4], "a new battle gets the new pick"
    assert pool.infer(["old"], obs, [0], (0, 14)).tolist() == [(3 + 2) % 14], "the running battle keeps its net and history"
    assert pool.stats()["nets"] == 2
    pool.forget(["old"])
    assert pool.stats()["nets"] == 1 and pool.counts["nets_dropped"] == 1


# --------------------------------------------------------------------------- the seat on recorded battles


class FakeClient:
    """Stands in for frozen_pool.FrozenClient: answers the first legal action of the observation it is given (or fails)."""

    def __init__(self, m0, m1, fail=False):
        self.m0, self.m1, self.fail = m0, m1, fail
        self.asked, self.forgotten, self.obs_widths = [], [], []

    async def ask(self, tag, obs, slot):
        self.asked.append((tag, slot))
        self.obs_widths.append(int(np.asarray(obs).shape[0]))
        if self.fail:
            raise TimeoutError("the pool did not answer")
        legal = np.flatnonzero(np.asarray(obs[self.m0:self.m1], dtype=np.float32) > 0.5)
        return int(legal[0]) if len(legal) else 0

    def forget(self, tag):
        self.forgotten.append(tag)


def frozen_seat(answerer, username, client, settings, seat="b"):
    stats = os_.SeatStats("frozen")
    player = tos.offline_player(os_.FrozenSeatPlayer, username, frozen_client=client, frozen_settings=settings, rng_seed="7",
                                seat_stats=stats, seat=seat, **tos.seat_kwargs(answerer.queue, answerer.futures))
    return player, stats


def test_the_frozen_seat_hands_nothing_to_the_learner_and_plays_one_net_per_battle(monkeypatch):
    monkeypatch.chdir(REPO)
    from obs_assembler import ObservationAssembler
    m0, m1 = ObservationAssembler().offsets["action_mask"]
    settings = _frozen(nets=(("m", 0.5), ("s", 0.5)))
    answerer = tos.Answerer()
    try:
        slots_seen = set()
        for events in tos._battles(revive=False):
            username = events[-1]["username"]
            room = events[0]["room"]
            decisions = len([e for e in events if e["t"] == "order"])
            client = FakeClient(m0, m1)
            player, stats = frozen_seat(answerer, username, client, settings)
            before = len(answerer.events)
            tos._feed(player, events)
            answerer.settle()
            assert answerer.events[before:] == [], "the pool answered every decision: no event at all"
            tag = "%s_%s" % (username, room)
            assert len(client.asked) == decisions == stats.bot_decisions == len(tos._sent(player)) and stats.net_decisions == 0
            assert len({slot for _t, slot in client.asked}) == 1 and {t for t, _s in client.asked} == {tag}
            slots_seen.add(client.asked[0][1])
            assert client.forgotten == [tag], "its history in the pool is dropped at the end"
            assert stats.games["p1"] == 1 and stats.mixed_games == 0 and player.ep_sem._value == 64
            name = settings.slots[client.asked[0][1]]
            assert stats.extra["battles_" + name] == 1 and stats.extra["games_" + name] == 1
            assert stats.extra.get("wins_" + name, 0) == (0 if events[-1]["won"] else 1)
            assert not player._frozen_pick and not player._battle_starts and not player._episode_slot_state
        assert slots_seen <= {0, 1}
    finally:
        answerer.close()


def test_when_the_pool_does_not_answer_the_network_plays_the_battle_and_its_steps_are_dropped(monkeypatch):
    monkeypatch.chdir(REPO)
    from obs_assembler import ObservationAssembler
    m0, m1 = ObservationAssembler().offsets["action_mask"]
    answerer = tos.Answerer()
    try:
        events = tos._battles(revive=False)[0]
        username = events[-1]["username"]
        decisions = len([e for e in events if e["t"] == "order"])
        client = FakeClient(m0, m1, fail=True)
        player, stats = frozen_seat(answerer, username, client, _frozen(nets=(("m", 1.0),)))
        tos._feed(player, events)
        answerer.settle()
        assert tos._kinds(answerer.events) == ["STEP"] * decisions + ["CLEANUP"]
        assert len(client.asked) == 1, "asked once; then the network played the rest"
        assert stats.extra["no_answer"] == 1 and stats.mixed_games == 1 and stats.net_decisions == decisions
        assert "games_m" not in stats.extra
    finally:
        answerer.close()


def test_a_net_of_another_observation_gets_its_own_assembler(monkeypatch):
    monkeypatch.chdir(REPO)
    run_obs = config.RunConfig.default().obs
    key = fp.obs_key_of({"own_tera_type": True, "matchup": True, "movesets": True}, run_obs)
    settings = dataclasses.replace(_frozen(nets=(("m", 0.5), ("s", 0.5))),
                                   nets=(fp.FrozenNetSpec("m", "/x/m.pt", 0.5), fp.FrozenNetSpec("s", "/x/s.pt", 0.5, obs=key)))
    answerer = tos.Answerer()
    try:
        player, _stats = frozen_seat(answerer, "obsseat", FakeClient(0, 14), settings)
        assert player._frozen_assembler(0) is player.assembler
        other = player._frozen_assembler(1)
        assert other is not player.assembler and other.matchup and other.movesets and other.own_tera_type
    finally:
        answerer.close()


def test_the_frozen_client_batches_answers_in_order_and_forgets_after():
    class Actor:
        def __init__(self):
            self.calls = []
            self.infer_frozen = SimpleNamespace(remote=self._infer)
            self.frozen_forget = SimpleNamespace(remote=lambda tags: self.calls.append(("forget", list(tags))))

        def _infer(self, tags, obs, slots):
            self.calls.append(("infer", list(tags), list(slots)))
            return np.arange(len(tags)) + 10

    import ray as real_ray
    actor = Actor()
    orig = real_ray.get
    real_ray.get = lambda v, **k: v
    try:
        client = fp.FrozenClient(actor, wait_s=0.02)

        async def go():
            a = asyncio.ensure_future(client.ask("t1", np.zeros(5, np.float16), 0))
            b = asyncio.ensure_future(client.ask("t2", np.zeros(5, np.float16), 1))
            client.forget("t0")
            return await asyncio.gather(a, b)

        assert asyncio.run(go()) == [10, 11]
        time.sleep(0.05)
        assert actor.calls[0] == ("infer", ["t1", "t2"], [0, 1]) and actor.calls[1] == ("forget", ["t0"])
        assert client.counts["answered"] == 2 and client.counts["calls"] == 1
    finally:
        real_ray.get = orig


def test_the_frozen_client_hands_a_failure_to_every_question():
    class Actor:
        infer_frozen = SimpleNamespace(remote=lambda *a: (_ for _ in ()).throw(RuntimeError("down")))
        frozen_forget = SimpleNamespace(remote=lambda tags: None)

    client = fp.FrozenClient(Actor(), wait_s=0.0)

    async def go():
        with pytest.raises(RuntimeError):
            await client.ask("t1", np.zeros(5, np.float16), 0)

    asyncio.run(go())
    assert client.counts["failed_calls"] == 1


def test_the_seats_line_with_frozen_seats():
    stats = {k: os_.SeatStats(k).as_dict() for k in os_.ALL_KINDS}
    stats["frozen"].update(games_p1=10, wins_p1=6, decisions=400, bot_decisions=400, battles_m=6, games_m=6, wins_m=4,
                           battles_s=4, games_s=4, wins_s=2)
    beat = {"port": 1, "pairs": {"frozen": 1}, "planned": {"frozen": 1}, "unavailable": {}, "failed_start": "", "stats": stats}
    line = os_.report_line([beat])
    assert line.startswith("[Seats] frozen: pairs 1, games 10, network won 60.0%")
    assert "by net: m 6 battles, network won 66.7% of 6; s 4 battles, network won 50.0% of 4" in line


def test_the_wrapper_without_a_pool_plays_frozen_pairs_as_self_play(monkeypatch):
    made = []
    monkeypatch.setattr(worker.PokeEnvVectorWrapper, "__init__", lambda self, cfg, pairs, port: made.append((pairs, port)))
    st = OpponentSeatsConfig(shares=(("frozen", 0.25),), frozen=_frozen())
    wrapper = os_.SeatsVectorWrapper(tos.run_cfg(), 4, 10399, [{"kind": "frozen", "seat": "b"}], st)
    assert wrapper.seat_specs == [] and "no frozen pool" in wrapper.seat_unavailable["frozen"]
    beat = wrapper.seats_heartbeat()
    assert beat["planned"] == {"setupnet": 0, "psbot": 0, "foulplay": 0, "frozen": 1}
    # a worker without frozen pairs reports the three kinds, as before
    wrapper = os_.SeatsVectorWrapper(tos.run_cfg(), 4, 10399, [{"kind": "setupnet", "seat": "b"}], tos._cfg(setupnet=0.25))
    assert tuple(wrapper.seats_heartbeat()["planned"]) == os_.SEAT_KINDS


# --------------------------------------------------------------------------- after the review


def test_the_past_saves_are_this_phases_only(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    old = time.time() - 3600
    modes = {}
    for i, mode in ((0, "imitation"), (5, "warmup"), (8, "warmup"), (200, "ppo"), (400, "ppo")):
        p = d / f"learner_update_{i:06d}.pt"
        p.write_bytes(b"x")
        os.utime(p, (old + i, old + i))
        modes[str(p)] = mode
    fp._MODE_CACHE.clear()
    files = fp.list_saves(str(d), 60.0, mode="ppo", mode_of=lambda p: modes[p])
    assert sorted(os.path.basename(p) for p, _t in files) == ["learner_update_000200.pt", "learner_update_000400.pt"]
    biases = {os.path.basename(p): i for i, p in enumerate(sorted(modes))}
    s = fp.FrozenSeatSettings(nets=(), past=3, past_weight=1.0, past_pick="uniform", refresh_s=3600.0, past_min_age_s=60.0,
                              ckpt_dir=str(d), seed=2)

    def load(path):
        return {"model": {"bias": torch.tensor([float(biases[os.path.basename(path)])])}}

    pool = fp.FrozenPool(s, FakeNet, "cpu", load=load, mode="ppo", mode_of=lambda p: modes[p])
    assert set(pool.past_files) <= {str(d / "learner_update_000200.pt"), str(d / "learner_update_000400.pt")}
    # none of this phase yet: the newest file there is (the start, or warmup's last)
    d2 = tmp_path / "early"
    d2.mkdir()
    early = {}
    for i, mode in ((0, "imitation"), (5, "warmup")):
        p = d2 / f"learner_update_{i:06d}.pt"
        p.write_bytes(b"x")
        os.utime(p, (old + i, old + i))
        early[str(p)] = mode
    biases.update({"learner_update_000000.pt": 0, "learner_update_000005.pt": 5})
    s2 = dataclasses.replace(s, ckpt_dir=str(d2))
    pool = fp.FrozenPool(s2, FakeNet, "cpu", load=load, mode="ppo", mode_of=lambda p: early[p])
    assert pool.past_files == [str(d2 / "learner_update_000005.pt")] * 3
    assert pool.stats()["forwards"] == 0 and pool.stats()["busy_s"] == 0.0
    pool.infer(["a", "b"], np.ones((2, 20), np.float16), [0, 1], (0, 14))
    st = pool.stats()
    assert st["forwards"] >= 1 and st["busy_s"] > 0.0 and "forwards" in league.frozen_line([st])


def test_a_seat_keeps_its_draw_for_16_battles_and_the_weights_hold_over_many(monkeypatch):
    monkeypatch.chdir(REPO)
    answerer = tos.Answerer()
    try:
        player, _stats = frozen_seat(answerer, "sticky", FakeClient(0, 14), _frozen(nets=(("m", 0.7), ("s", 0.3))))
        draws = [player._draw_slot() for _ in range(16 * 400)]
        assert all(len(set(draws[i:i + 16])) == 1 for i in range(0, len(draws), 16))
        share = draws.count(0) / len(draws)
        assert abs(share - 0.7) < 0.06
        zero_last = _frozen(nets=(("m", 1.0), ("s", 0.0)))
        player, _stats = frozen_seat(answerer, "sticky2", FakeClient(0, 14), zero_last)
        player._frng.random = lambda: 1.0 - 1e-17          # past the sum by rounding
        assert player._draw_weighted() == 0, "never a slot of weight 0"
    finally:
        answerer.close()


def test_resolve_refuses_a_net_that_is_not_the_runs(tmp_path):
    run = config.RunConfig.default()
    template = {"a.weight": torch.zeros(2, 3), "set_head.x": torch.zeros(1)}
    good = tmp_path / "good.pt"
    torch.save({"model": {"a.weight": torch.ones(2, 3), "opp_action_head.y": torch.zeros(1)}, "run_cfg": {"obs": {}}}, good)
    bad = tmp_path / "bad.pt"
    torch.save({"model": {"a.weight": torch.ones(2, 4)}, "run_cfg": {"obs": {}}}, bad)
    s = fp.FrozenSeatSettings(nets=(fp.FrozenNetSpec("g", str(good), 1.0),))
    out = fp.resolve(s, run, template=template)
    assert out.ckpt_dir == run.learner.ckpt_dir and out.nets[0].obs != fp.RUN_OBS     # own_tera_type absent: another observation
    with pytest.raises(ValueError, match="not this run's network"):
        fp.resolve(fp.FrozenSeatSettings(nets=(fp.FrozenNetSpec("b", str(bad), 1.0),)), run, template=template)
    with pytest.raises(ValueError, match="not a file"):
        fp.resolve(fp.FrozenSeatSettings(nets=(fp.FrozenNetSpec("x", str(tmp_path / "none.pt"), 1.0),)), run, template=template)


def test_the_head_to_head_counts_a_tie_as_half(tmp_path):
    import json as _json
    sys.path.insert(0, os.path.join(REPO, "bench"))
    import league_h2h as lh
    cell = tmp_path / "c"
    cell.mkdir()
    rows = [{"arm": "net", "won": True}, {"arm": "net", "won": False, "tie": True}, {"arm": "net", "won": False},
            {"arm": "net", "won": True, "stalled": True}]
    (cell / "results.jsonl").write_text("\n".join(_json.dumps(r) for r in rows))
    assert lh.games(str(cell)) == (3, 1.5)
    lo, hi = lh.wilson(1.5, 3)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
