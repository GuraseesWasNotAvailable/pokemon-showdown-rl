"""
What a rollout worker keeps per battle, and what it hands to the learner (bench/MEMORY.md).

Real code on both sides: worker.py as it is in this tree, and release r1's worker.py taken from
git (commit 8227671, what the cloud box runs). Each drives real poke-env players that never
connect, fed with recorded battle streams (tests/fixtures/foulplay_bridge_trace.jsonl.gz: seven
battle streams of a real local server, with both sides of three rooms), through the client's own
message handler (so the room locks and worker.py's patch of that handler are in the path), with
the real RolloutWorker loop behind them. Only Ray is replaced: the inference actor is a function
of the observation, and the learner client records what it is handed.

Covered, for every way a battle ends in training:
  * a win, a tie, a forfeit: the episode handed over is byte for byte r1's, and nothing about the
    battle is kept afterwards;
  * a room the server closes ("|expire|", "|deinit", no result): r1 hands over nothing and keeps
    everything (the battle, its seat, its episode slot, its steps); now nothing is handed over
    either, and everything goes back;
  * a battle still open when the worker stops: nothing handed over, by either;
  * ppo and warmup (the policy decides), imitation (a teacher decides), with the PSBot engine's
    copy of the battle dropped too.
(The "/choose default" that answers an invalid choice is in tests/test_worker_choice_storm.py.)

The Foul Play opponent share is not in this tree's worker.py (nor in r1's): nothing to cover.
"""
from __future__ import annotations

import asyncio
import dataclasses
import gzip
import importlib.util
import json
import os
import queue
import subprocess
import sys
import threading
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)          # vocab.json is opened by a relative path

np = pytest.importorskip("numpy")
pytest.importorskip("poke_env")
pytest.importorskip("ray")

FIXTURE = os.path.join(REPO, "tests", "fixtures", "foulplay_bridge_trace.jsonl.gz")
R1_COMMIT = "8227671"

pytestmark = pytest.mark.skipif(not os.path.isfile(FIXTURE), reason="the recorded battles are missing")


# ---------------------------------------------------------------------------------------------
# the two worker modules
# ---------------------------------------------------------------------------------------------

_MODULES = {}


def _pristine_handler():
    """poke-env's own PSClient._handle_message, whatever has been patched over it by now."""
    import poke_env
    for name in ("worker", "worker_new_for_memory_test", "worker_r1_for_memory_test"):
        mod = sys.modules.get(name)
        if mod is not None and hasattr(mod, "_original_handle_message"):
            handler = mod._original_handle_message
            while getattr(handler, "__name__", "") == "loud_handle":      # patched before that module came
                break
            return handler
    return poke_env.ps_client.PSClient._handle_message


def worker_module(which: str, tmp_root: str):
    """worker.py of this tree ("new") or of release r1 ("r1"), each with its own patched handler."""
    if which in _MODULES:
        return _MODULES[which]
    import poke_env
    pristine = _MODULES.get("pristine")
    if pristine is None:
        pristine = _MODULES["pristine"] = _pristine_handler()
    if which == "new":
        path = os.path.join(REPO, "worker.py")
    else:
        source = subprocess.run(["git", "-C", REPO, "show", f"{R1_COMMIT}:worker.py"], capture_output=True, check=False)
        if source.returncode != 0:
            pytest.skip(f"commit {R1_COMMIT} (release r1) is not in this clone")
        path = os.path.join(tmp_root, "worker_r1_for_memory_test.py")
        with open(path, "wb") as fh:
            fh.write(source.stdout)
    before = poke_env.ps_client.PSClient._handle_message
    poke_env.ps_client.PSClient._handle_message = pristine           # each module wraps the original, not the other
    try:
        spec = importlib.util.spec_from_file_location(f"worker_{which}_for_memory_test", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        mod.handle_message = poke_env.ps_client.PSClient._handle_message      # this module's loud_handle
    finally:
        poke_env.ps_client.PSClient._handle_message = before
    _MODULES[which] = mod
    return mod


@pytest.fixture(scope="module")
def tmp_root(tmp_path_factory):
    return str(tmp_path_factory.mktemp("worker_memory"))


# ---------------------------------------------------------------------------------------------
# recorded battles
# ---------------------------------------------------------------------------------------------

def recorded_battles():
    """[(username, room, [message text, ...]), ...]: every battle stream of the recording."""
    out, current = [], None
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as fh:
        for line in fh:
            event = json.loads(line)
            if event.get("t") == "msg" and event["data"].startswith("|init|"):
                current = {"room": event["room"], "messages": [], "username": None}
                out.append(current)
            if current is None:
                continue
            if event.get("t") == "msg":
                current["messages"].append(event["data"])
            elif event.get("username"):
                current["username"] = event["username"]
    return [(b["username"], b["room"], b["messages"]) for b in out if b["username"]]


def renamed(messages, room, new_room):
    return [m.replace(room, new_room) for m in messages]


def cut_after_requests(messages, n):
    """The stream up to and including its n-th request that asks for a move."""
    seen = 0
    for i, m in enumerate(messages):
        for line in m.split("\n"):
            if line.startswith("|request|{") and '"wait":true' not in line:
                seen += 1
        if seen >= n:
            return messages[:i + 1]
    return list(messages)


def ending(kind, username, messages):
    """The recorded stream, or its first part with another ending put on it."""
    if kind == "win":
        return list(messages) + ["|deinit"]                 # the server's answer to poke-env's /leave
    part = cut_after_requests(messages, 4)
    if kind == "tie":
        return part + ["|\n|tie", "|deinit"]
    if kind == "forfeit":
        return part + [f"|-message|{username} forfeited.\n|\n|win|somebodyelse", "|deinit"]
    if kind == "expire":
        return part + ["|expire|", "|deinit"]
    if kind == "open":
        return part
    raise ValueError(kind)


# ---------------------------------------------------------------------------------------------
# a rollout worker without Ray
# ---------------------------------------------------------------------------------------------

class _Remote:
    def __init__(self, fn):
        self.remote = fn


class FakeInference:
    """Answers as a function of the observation alone, so two workers fed the same stream get the same numbers."""
    def __init__(self, mask):
        m0, m1 = mask
        self.cleared = []

        def infer(tags, obs):
            obs32 = obs.astype(np.float32)
            legal = obs32[:, m0:m1] > 0
            acts = np.where(legal.any(axis=1), legal.argmax(axis=1), 0).astype(np.int64)
            key = np.abs(obs32).sum(axis=1)
            return acts, (-(key % 7.0) / 7.0).astype(np.float32), (key % 5.0).astype(np.float32), (key % 3.0).astype(np.float32)

        self.infer_batch = _Remote(infer)
        self.clear_cache = _Remote(lambda tag: self.cleared.append(tag))


class RecordingLearner:
    def __init__(self):
        self.episodes = []
        self.q = queue.Queue()

    def submit_episode(self, obs, act, logp, val, var, rew, done):
        self.episodes.append(tuple(np.array(x, copy=True) for x in (obs, act, logp, val, var, rew, done)))
        return True


class Harness:
    """One RolloutWorker of the given module with offline players, its real loop on a thread."""
    SLOTS = 64

    def __init__(self, mod, mode="ppo", teacher="simple", seats=16):
        import config
        from poke_env import AccountConfiguration, ServerConfiguration
        from poke_env.concurrency import POKE_LOOP

        self.mod = mod
        base = config.RunConfig.default()
        self.cfg = dataclasses.replace(base, learner=dataclasses.replace(base.learner, mode=mode),
                                       rollout=dataclasses.replace(base.rollout, imitation_teacher=teacher))
        self.loop = POKE_LOOP
        self.event_queue = queue.Queue()
        self.action_futures = {}
        self.ep_sem = asyncio.run_coroutine_threadsafe(self._make_semaphore(), self.loop).result(10)
        self.players = {}
        self.seats = seats
        self._account, self._server = AccountConfiguration, ServerConfiguration

        worker = object.__new__(mod.RolloutWorker)              # no Ray actor: the fields its loop uses
        worker.cfg = self.cfg
        worker.learner_client = self.learner = RecordingLearner()
        worker._traj = {}
        worker.assembler = mod.ObservationAssembler()
        worker.inference_actor = self.inference = FakeInference(worker.assembler.offsets["action_mask"])
        vec = object.__new__(mod.PokeEnvVectorWrapper)
        vec.event_queue, vec.action_futures, vec.async_loop, vec.ep_sem = self.event_queue, self.action_futures, self.loop, self.ep_sem
        vec.active_pairs = []
        worker.vec_env = vec
        self.worker = worker
        self._real_get = mod.ray.get
        mod.ray.get = lambda x, *a, **k: x                      # the fake actors answer directly
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.error = None
        self.thread.start()

    async def _make_semaphore(self):
        return asyncio.Semaphore(self.SLOTS)

    def _loop(self):
        try:
            self.worker._run_rl_loop()
        except BaseException as e:      # noqa: BLE001
            self.error = e

    def player(self, username):
        if username not in self.players:
            p = self.mod.SyncBridgePlayer(
                cfg=self.cfg, event_queue=self.event_queue, action_futures=self.action_futures, ep_sem=self.ep_sem,
                account_configuration=self._account(username, None),
                server_configuration=self._server("ws://127.0.0.1:9/showdown/websocket", "http://127.0.0.1:9/action.php?"),
                battle_format="gen9randombattle", start_listening=False, log_level=40, max_concurrent_battles=self.seats)
            p.sent = []

            async def send_message(message, room="", message_2=None, _p=p):
                _p.sent.append((room, message))

            p.ps_client.send_message = send_message
            self.players[username] = p
        return self.players[username]

    def feed(self, username, room, messages, timeout=120.0):
        """The stream through the client's handler, as the websocket listener gives it: one task a message, in order."""
        player = self.player(username)
        given_up = []

        async def run():
            for data in messages:
                if given_up:
                    return
                await self.mod.handle_message(player.ps_client, ">" + room + "\n" + data)

        future = asyncio.run_coroutine_threadsafe(run(), self.loop)
        try:
            future.result(timeout=timeout)
        except BaseException:
            # A stream that cannot go on (r1 out of seats) must not stay parked on poke-env's loop.
            # poke-env's handler swallows a cancellation, so without the flag the loop above would
            # go on to the next message and park again, and at interpreter exit poke-env waits
            # (spinning) for every task of its loop to end: the test run never left.
            given_up.append(True)
            future.cancel()
            raise

    def settle(self, timeout=10.0):
        """Until the loop has taken everything the players queued."""
        end = time.time() + timeout
        while time.time() < end:
            if self.error is not None:
                raise self.error
            if self.event_queue.empty():
                time.sleep(0.05)
                if self.event_queue.empty():
                    asyncio.run_coroutine_threadsafe(asyncio.sleep(0), self.loop).result(5)    # release_slot calls land
                    return
            time.sleep(0.01)
        raise AssertionError("the worker loop did not take its events")

    def kept(self):
        """Everything that is held per battle, by name."""
        out = {"traj": len(self.worker._traj), "action_futures": len(self.action_futures),
               "slots_out": self.SLOTS - self.ep_sem._value}
        for name in ("_battles", "_battle_starts", "_last_act_time", "_episode_slot_state", "_battle_events"):
            out[name] = sum(len(getattr(p, name)) for p in self.players.values())
        out["seats"] = sum(p._battle_count_queue.qsize() for p in self.players.values())
        out["room_locks"] = sum(len(p.ps_client._battle_locks) for p in self.players.values())
        return out

    def close(self):
        self.mod.ray.get = self._real_get


NOTHING = {"traj": 0, "action_futures": 0, "slots_out": 0, "_battles": 0, "_battle_starts": 0, "_last_act_time": 0,
           "_episode_slot_state": 0, "_battle_events": 0, "seats": 0, "room_locks": 0}


def play(mod, kind, mode="ppo", battles=None, rooms=1):
    """Feeds every recorded stream with the given ending, `rooms` times under new room names. -> (harness, kept)"""
    h = Harness(mod, mode=mode)
    try:
        for copy in range(rooms):
            for username, room, messages in (battles or recorded_battles()):
                new_room = f"{room}-c{copy}" if copy else room
                h.feed(username, new_room, renamed(ending(kind, username, messages), room, new_room))
        h.settle()
        return h, h.kept()
    finally:
        h.close()


def same_episodes(a, b):
    assert len(a) == len(b)
    for ea, eb in zip(a, b):
        for xa, xb in zip(ea, eb):
            assert xa.dtype == xb.dtype and xa.shape == xb.shape and np.array_equal(xa, xb)


# ---------------------------------------------------------------------------------------------
# endings with a result: the same episode as r1, and nothing kept
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["ppo", "warmup", "imitation"])
@pytest.mark.parametrize("kind", ["win", "tie", "forfeit"])
def test_a_battle_with_a_result_hands_over_what_r1_hands_over_and_keeps_nothing(tmp_root, kind, mode):
    new, kept_new = play(worker_module("new", tmp_root), kind, mode)
    old, kept_old = play(worker_module("r1", tmp_root), kind, mode)
    n = len(recorded_battles())
    assert len(new.learner.episodes) == n                       # one episode a battle stream, its last step included
    same_episodes(new.learner.episodes, old.learner.episodes)
    for obs, act, logp, val, var, rew, done in new.learner.episodes:
        assert len(obs) == len(act) == len(logp) == len(val) == len(var) == len(rew) == len(done) > 0
        assert done[-1] == 1.0 and not done[:-1].any()          # one boundary, at the end
    assert sorted(new.inference.cleared) == sorted(old.inference.cleared)
    assert kept_new == NOTHING and kept_old == NOTHING          # r1 already let go of a battle that had a result
    assert all(p.rooms_closed == 0 for p in new.players.values())
    # What was sent to the server is the same too.
    assert {u: p.sent for u, p in new.players.items()} == {u: p.sent for u, p in old.players.items()}


def test_the_full_recorded_battles_end_where_the_recording_ends(tmp_root):
    """The win streams are the recording untouched: the episode's length is the number of moves asked for."""
    new, _kept = play(worker_module("new", tmp_root), "win")
    asked = []
    for _username, _room, messages in recorded_battles():
        asked.append(sum(1 for m in messages for line in m.split("\n")
                         if line.startswith("|request|{") and '"wait":true' not in line and '"teamPreview"' not in line))
    assert [len(e[1]) for e in new.learner.episodes] == asked
    rewards = [float(e[5][-1]) for e in new.learner.episodes]
    assert all(r != 0.0 for r in rewards)                       # every stream ends in a win or a loss for its player


# ---------------------------------------------------------------------------------------------
# a room the server closes, and a battle still open at the end
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["ppo", "warmup", "imitation"])
def test_a_room_the_server_closes_is_given_back_and_nothing_of_it_reaches_the_learner(tmp_root, mode):
    new, kept_new = play(worker_module("new", tmp_root), "expire", mode, rooms=2)
    old, kept_old = play(worker_module("r1", tmp_root), "expire", mode, rooms=2)
    n = 2 * len(recorded_battles())
    # r1 hands over nothing for such a battle, and neither does the new code: no half episode.
    assert old.learner.episodes == [] and new.learner.episodes == []
    # r1 keeps all of it, for the life of the worker ...
    assert kept_old["_battles"] == n and kept_old["seats"] == n and kept_old["slots_out"] == n and kept_old["traj"] == n
    assert kept_old["_battle_starts"] == n and kept_old["_episode_slot_state"] == n
    # ... and now everything is back where it was before the battles.
    assert kept_new == NOTHING
    assert sum(p.rooms_closed for p in new.players.values()) == n
    assert len(new.inference.cleared) == n and old.inference.cleared == []
    # Until the room closed both sent the server the same things.
    assert {u: p.sent for u, p in new.players.items()} == {u: p.sent for u, p in old.players.items()}


def test_r1_runs_out_of_seats_after_closed_rooms_and_the_new_code_does_not(tmp_root):
    """
    poke-env lets a player hold max_concurrent_battles battles (rooms_per_pair of them in training).
    A closed room kept its seat in r1, so once that many rooms had been closed the first message
    of the next battle waited for a seat that never came, holding its room's lock.
    """
    username, room, messages = recorded_battles()[0]
    for which, blocked in (("r1", True), ("new", False)):
        h = Harness(worker_module(which, tmp_root), seats=2)
        try:
            for i in range(2):
                r = f"{room}-x{i}"
                h.feed(username, r, renamed(ending("expire", username, messages), room, r))
            r = f"{room}-next"
            stream = renamed(ending("win", username, messages), room, r)
            if blocked:
                with pytest.raises(Exception):                   # concurrent.futures.TimeoutError
                    h.feed(username, r, stream, timeout=3.0)
                assert h.learner.episodes == []
            else:
                h.feed(username, r, stream, timeout=120.0)
                h.settle()
                assert len(h.learner.episodes) == 1 and h.kept() == NOTHING
        finally:
            h.close()


def test_a_battle_still_open_when_the_worker_stops_hands_over_nothing(tmp_root):
    new, kept_new = play(worker_module("new", tmp_root), "open")
    old, kept_old = play(worker_module("r1", tmp_root), "open")
    n = len(recorded_battles())
    assert new.learner.episodes == [] and old.learner.episodes == []
    assert kept_new == kept_old and kept_new["_battles"] == n and kept_new["traj"] == n      # still being played: kept, as in r1
    # The steps taken so far are the same steps.
    for tag, buf in new.worker._traj.items():
        other = old.worker._traj[tag]
        assert [int(a) for a in buf["act"]] == [int(a) for a in other["act"]]
        assert all(np.array_equal(x, y) for x, y in zip(buf["obs"], other["obs"]))


def test_a_closed_room_before_its_first_decision_takes_no_slot_back(tmp_root):
    """The episode slot is taken at the first decision. A room closed before that must not give one back."""
    username, room, messages = recorded_battles()[0]
    h = Harness(worker_module("new", tmp_root))
    try:
        h.feed(username, room, [messages[0], "|expire|", "|deinit"])       # "|init|" and nothing else
        h.settle()
        assert h.kept() == NOTHING and h.ep_sem._value == h.SLOTS and h.inference.cleared == []
        assert h.player(username).rooms_closed == 1
        # Lines that still arrive for the room are dropped: they do not wait for a battle that is gone.
        h.feed(username, room, ["|turn|99", "|deinit"], timeout=5.0)
        assert h.kept() == NOTHING
        # A "|deinit" for a room this player never saw does not wait either (poke-env would).
        h.feed(username, "battle-gen9randombattle-777777", ["|deinit"], timeout=5.0)
        assert h.kept() == NOTHING
    finally:
        h.close()


def test_the_psbot_engine_is_told_when_a_room_closes(tmp_root):
    mod = worker_module("new", tmp_root)
    username, room, messages = recorded_battles()[0]

    class Engine:
        def __init__(self):
            self.ops = []

        def send(self, msg):
            self.ops.append((msg["op"], msg.get("key")))

        async def call(self, msg, timeout=20.0):
            raise RuntimeError("no engine in this test")        # the teacher falls back to SimpleHeuristics

    h = Harness(mod, mode="imitation", teacher="psbot")
    try:
        player = h.player(username)
        assert player.psbot_enabled
        player._psbot_engine = engine = Engine()
        h.feed(username, room, ending("expire", username, messages))
        h.settle()
        assert ("end", f"{username}|{room}") in engine.ops and h.kept() == NOTHING
        assert h.learner.episodes == []
    finally:
        h.close()
