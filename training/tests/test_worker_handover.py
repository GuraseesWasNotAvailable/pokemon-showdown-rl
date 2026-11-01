"""
What the learner is handed, for every cut point and every ending (bench/MEMORY.md).

Written by the review of the memory fix and taken into the tests as it was, with the
repository's paths. It builds on the harness of tests/test_worker_memory.py: release r1's
worker.py from git against the tree's worker.py, real poke-env players that never connect,
recorded battle streams, the real rollout loop function, Ray replaced by a function of the
observation. No Ray, no learner, no inference actor, no server.

    python -m pytest tests/test_worker_handover.py -q -p no:cacheprovider -s
    PSPPO_HANDOVER_FULL=1 ...        every cut point of every stream in every mode (17 minutes)

What it adds to tests/test_worker_memory.py and tests/test_worker_choice_storm.py:
  * every cut point of every recorded stream (not only "after four requests"), with every ending:
    win, tie, forfeit, |expire|+|deinit, |deinit alone, left open (the worker stops);
  * the cut points whose last request is a forced switch are checked to be among them;
  * many rooms through ONE worker, both sides of a room in the same worker, streams interleaved;
  * the messages delivered the way the websocket listener does (one task a message, none awaited),
    so a |win| or an |expire| queues on the room lock behind a decision that is still out;
  * a refused choice after every request of every stream, against the same battle without it;
  * late lines for a room that was given back, a room tag that comes again;
  * episode slots and seats with few of them (a double release would show as one slot too many).

Without PSPPO_HANDOVER_FULL the two big comparisons run on a sixth to a ninth of the cut points
and two of the streams (two minutes); the review's twelve deliberate breaks of worker.py were
run against exactly that short set. PSPPO_REVIEW_WORKER is not used any more.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_worker_memory as twm  # noqa: E402  (chdirs into the repo: vocab.json is read by a relative path)

np = twm.np
BIG = 10 ** 6


@pytest.fixture(scope="module")
def tmp_root(tmp_path_factory):
    return str(tmp_path_factory.mktemp("worker_handover"))


class H(twm.Harness):
    """The first agent's harness, with the episode of every battle kept under its tag."""

    def __init__(self, mod, mode="ppo", teacher="simple", seats=BIG, slots=BIG):
        self.SLOTS = slots
        super().__init__(mod, mode=mode, teacher=teacher, seats=seats)
        self.by_tag, self.cleaned, self.finalized = {}, [], []
        w, learner = self.worker, self.learner
        fin, cln = w._finalize_trajectory, w._cleanup_trajectory

        def finalize(tag, battle):
            n = len(learner.episodes)
            fin(tag, battle)
            self.finalized.append(tag)
            if len(learner.episodes) != n:
                assert len(learner.episodes) == n + 1
                assert tag not in self.by_tag, f"two episodes for one battle: {tag}"
                self.by_tag[tag] = learner.episodes[-1]

        def cleanup(tag):
            self.cleaned.append(tag)
            cln(tag)

        w._finalize_trajectory, w._cleanup_trajectory = finalize, cleanup

    def player(self, username):
        p = super().player(username)
        p.DEFAULT_CHOICE_CHANCE = 0.0        # poke-env's one-in-a-thousand default: not in a comparison
        return p

    def feed_backlog(self, username, room, messages, timeout=120.0):
        """Every message is a task at once, as when the listener has read them all: they queue on the room lock."""
        player = self.player(username)

        async def run():
            tasks = [asyncio.ensure_future(self.mod.handle_message(player.ps_client, ">" + room + "\n" + d)) for d in messages]
            done = await asyncio.gather(*tasks, return_exceptions=True)
            return [repr(r) for r in done if isinstance(r, BaseException)]

        return asyncio.run_coroutine_threadsafe(run(), self.loop).result(timeout)

    def sent_in(self, username, room):
        return [m for r, m in self.player(username).sent if r == room]

    def traj(self, username, room):
        return self.worker._traj.get(f"{username}_{room}")


def n_requests(messages):
    return sum(1 for m in messages for line in m.split("\n") if line.startswith("|request|{") and '"wait":true' not in line)


def last_request_kind(messages):
    kind = None
    for m in messages:
        for line in m.split("\n"):
            if line.startswith("|request|{") and '"wait":true' not in line:
                kind = "force" if '"forceSwitch"' in line else "move"
    return kind


ENDINGS = ("win", "tie", "forfeit", "expire", "deinit", "open")


def with_ending(kind, username, messages, k):
    part = twm.cut_after_requests(messages, k)
    if kind == "win":
        return part + ["|\n|win|" + username, "|deinit"]
    if kind == "tie":
        return part + ["|\n|tie", "|deinit"]
    if kind == "forfeit":
        return part + [f"|-message|{username} forfeited.\n|\n|win|somebodyelse", "|deinit"]
    if kind == "expire":
        return part + ["|expire|", "|deinit"]
    if kind == "deinit":
        return part + ["|deinit"]
    if kind == "open":
        return part
    raise ValueError(kind)


def cases(stride=1):
    """(username, base room, messages, k) for every (stride-th) cut point of every recorded stream, its last one always."""
    out = []
    for username, room, messages in twm.recorded_battles():
        n = n_requests(messages)
        for k in sorted(set(range(1, n + 1, stride)) | {n}):
            out.append((username, room, messages, k))
    return out


def room_for(room, kind, k, extra=""):
    return f"{room}-{kind}{k}{extra}"


def same_arrays(ea, eb, what):
    assert len(ea) == len(eb) == 7, what
    for i, (xa, xb) in enumerate(zip(ea, eb)):
        assert xa.dtype == xb.dtype and xa.shape == xb.shape and np.array_equal(xa, xb), f"{what}: array {i} differs"


def same_traj(ta, tb, what):
    assert (ta is None) == (tb is None), what
    if ta is None:
        return
    assert set(ta) == set(tb), what
    for key in ta:
        assert len(ta[key]) == len(tb[key]), f"{what}: {key} has {len(ta[key])} and {len(tb[key])} steps"
        for xa, xb in zip(ta[key], tb[key]):
            assert np.array_equal(np.asarray(xa), np.asarray(xb)), f"{what}: {key} differs"


def run_all_endings(mod, mode, delivery, interleave=False, stride=1):
    h = H(mod, mode=mode)
    try:
        jobs = []
        for username, room, messages, k in cases(stride):
            for kind in ENDINGS:
                r = room_for(room, kind, k)
                jobs.append((username, r, twm.renamed(with_ending(kind, username, messages, k), room, r)))
        errors = []
        if delivery == "backlog":
            for username, r, stream in jobs:
                errors += h.feed_backlog(username, r, stream)
        elif interleave:
            # eight rooms at a time, one message of each in turn (both sides and several rooms in flight)
            for start in range(0, len(jobs), 8):
                group = [(u, r, list(s)) for u, r, s in jobs[start:start + 8]]
                while any(s for _u, _r, s in group):
                    for u, r, s in group:
                        if s:
                            h.feed(u, r, [s.pop(0)])
        else:
            for username, r, stream in jobs:
                h.feed(username, r, stream)
        h.settle(30.0)
        time.sleep(0.3)
        h.settle(30.0)
        return h, errors
    finally:
        h.close()


QUICK = os.environ.get("PSPPO_HANDOVER_FULL") != "1"         # the short set: a sixth of the cut points, two streams


@pytest.mark.parametrize("mode,delivery,interleave,stride", [("ppo", "backlog", False, 6), ("imitation", "sequential", False, 9)] if QUICK else [
    ("ppo", "sequential", False, 1), ("ppo", "backlog", False, 1), ("ppo", "sequential", True, 2),
    ("warmup", "sequential", False, 3), ("imitation", "sequential", False, 2), ("imitation", "backlog", False, 3),
])
def test_every_cut_and_every_ending_hands_over_what_r1_hands_over(tmp_root, mode, delivery, interleave, stride):
    new, new_err = run_all_endings(twm.worker_module("new", tmp_root), mode, delivery, interleave, stride)
    old, old_err = run_all_endings(twm.worker_module("r1", tmp_root), mode, delivery, interleave, stride)
    assert new_err == [] and old_err == [], (new_err[:3], old_err[:3])
    all_cases = cases(stride)
    forced_last = 0
    for username, room, messages, k in all_cases:
        part = twm.cut_after_requests(messages, k)
        if last_request_kind(part) == "force":
            forced_last += 1
        for kind in ENDINGS:
            r = room_for(room, kind, k)
            tag = f"{username}_{r}"
            what = f"{mode}/{delivery}/{kind} k={k} {tag}"
            # the same choices went to the server, in the same order
            assert new.sent_in(username, r) == old.sent_in(username, r), what
            if kind in ("win", "tie", "forfeit"):
                assert tag in new.by_tag and tag in old.by_tag, what
                same_arrays(new.by_tag[tag], old.by_tag[tag], what)
                obs, act, logp, val, var, rew, done = new.by_tag[tag]
                assert len(act) == k, f"{what}: {len(act)} steps for {k} requests answered"     # the last step is in it
                assert done[-1] == 1.0 and not done[:-1].any(), what
                rc = new.cfg.reward
                want = {"win": rc.terminal_win, "forfeit": rc.terminal_loss, "tie": getattr(rc, "terminal_tie", 0.0)}[kind]
                if getattr(rc, "shaping", "none") != "potential":
                    assert rew[-1] == want, what
                assert new.traj(username, r) is None and old.traj(username, r) is None, what
                assert tag not in new.cleaned and tag not in old.cleaned, what
            elif kind in ("expire", "deinit"):
                assert tag not in new.by_tag and tag not in old.by_tag, what       # nothing handed over, by either
                assert new.traj(username, r) is None, what                        # ... and the new code keeps no steps
                assert old.traj(username, r) is not None and len(old.traj(username, r)["act"]) == k, what
                assert new.cleaned.count(tag) == 1 and tag not in old.cleaned, what
            else:
                assert tag not in new.by_tag and tag not in old.by_tag, what
                same_traj(new.traj(username, r), old.traj(username, r), what)
                assert len(new.traj(username, r)["act"]) == k, what
    assert forced_last >= (10 if stride == 1 else 3), forced_last   # cut points whose last request is a forced switch are in the set
    n = len(all_cases)
    assert len(new.by_tag) == len(old.by_tag) == 3 * n
    # What is kept at the end: only the battles left open (as in r1); r1 also keeps the closed rooms.
    kept_new, kept_old = new.kept(), old.kept()
    assert kept_new["_battles"] == n and kept_new["traj"] == n and kept_new["slots_out"] == n and kept_new["seats"] == n, kept_new
    assert kept_new["_battle_starts"] == n and kept_new["_episode_slot_state"] == n and kept_new["action_futures"] == 0, kept_new
    assert kept_old["_battles"] == 3 * n and kept_old["traj"] == 3 * n and kept_old["slots_out"] == 3 * n, kept_old
    assert sum(p.rooms_closed for p in new.players.values()) == 2 * n
    print(f"\nHANDOVER {mode}/{delivery}{'/interleaved' if interleave else ''}: {n} cut points x {len(ENDINGS)} endings; "
          f"{forced_last} cut points end on a forced switch; episodes compared {3 * n}; shaping={getattr(new.cfg.reward, 'shaping', None)}; "
          f"kept new {kept_new}")


# ------------------------------------------------------------------------------------------------
# a refused choice after every request of every stream
# ------------------------------------------------------------------------------------------------

REPAIRABLE = "|error|[Invalid choice] Can't move: Your Garchomp doesn't have a move matching earthpower"
NOTHING = "|error|[Invalid choice] There's nothing to choose"
TOO_LATE = "|error|[Invalid choice] Sorry, too late to make a different move; the next turn has already started"
UNAVAILABLE = "|error|[Unavailable choice] Can't switch: The active Pokémon is trapped"


def STREAMS():
    return twm.recorded_battles()[:2] if QUICK else twm.recorded_battles()


def play_with_errors(mod, mode, error_sets):
    """Every stream, whole, with `lines` put in after its k-th request was answered, for every k. One worker."""
    h = H(mod, mode=mode)
    try:
        for username, room, messages in STREAMS():
            full = list(messages) + ["|deinit"]
            r = f"{room}-clean"
            h.feed(username, r, twm.renamed(full, room, r))
            for name, lines in error_sets.items():
                for k in range(1, n_requests(messages) + 1):
                    r = f"{room}-{name}{k}"
                    stream = twm.renamed(full, room, r)
                    head = len(twm.cut_after_requests(messages, k))
                    h.feed(username, r, stream[:head])
                    h.feed(username, r, lines)
                    h.feed(username, r, stream[head:])
        h.settle(30.0)
        time.sleep(0.3)
        h.settle(30.0)
        return h
    finally:
        h.close()


@pytest.mark.parametrize("mode", ["ppo"] if QUICK else ["ppo", "imitation"])
def test_a_refused_choice_after_any_request_changes_nothing_but_r1s_extra_steps(tmp_root, mode):
    sets = {"rep": [REPAIRABLE], "not": [NOTHING], "late": [TOO_LATE], "mix": [REPAIRABLE, NOTHING, TOO_LATE, NOTHING]}
    new = play_with_errors(twm.worker_module("new", tmp_root), mode, sets)
    old = play_with_errors(twm.worker_module("r1", tmp_root), mode, sets)
    compared = 0
    extra_steps_r1 = []
    for username, room, messages in STREAMS():
        clean_tag = f"{username}_{room}-clean"
        same_arrays(new.by_tag[clean_tag], old.by_tag[clean_tag], clean_tag)
        clean = new.by_tag[clean_tag]
        clean_sent = new.sent_in(username, f"{room}-clean")
        for name, lines in sets.items():
            for k in range(1, n_requests(messages) + 1):
                r = f"{room}-{name}{k}"
                tag = f"{username}_{r}"
                same_arrays(new.by_tag[tag], clean, f"{mode} {tag}")              # the episode of the battle without the error
                compared += 1
                sent = new.sent_in(username, r)
                defaults = 1 if name in ("rep", "mix") else 0
                assert sent.count("/choose default") == clean_sent.count("/choose default") + defaults, (tag, sent)
                assert [m for m in sent if m != "/choose default"] == [m for m in clean_sent if m != "/choose default"], tag
                extra_steps_r1.append(len(old.by_tag[tag][1]) - len(clean[1]))
    assert new.kept() == twm.NOTHING, new.kept()
    assert sum(len(p.ps_client.__dict__.get("_default_sent", ())) for p in new.players.values()) == 0
    assert min(extra_steps_r1) >= 1                                              # r1: at least one step more per battle with a refusal
    print(f"\nHANDOVER {mode}: {compared} battles with a refused choice equal the battle without it; "
          f"r1 adds these numbers of steps: {sorted(set(extra_steps_r1))}")


def test_an_unavailable_choice_is_asked_again_by_both(tmp_root):
    """[Unavailable choice] plus the server's updated request: a second step, in r1 and now. Not touched by the fix."""
    out = {}
    for which in ("new", "r1"):
        h = H(twm.worker_module(which, tmp_root))
        try:
            for username, room, messages in twm.recorded_battles():
                for k in range(1, n_requests(messages) + 1):
                    r = f"{room}-una{k}"
                    full = twm.renamed(list(messages) + ["|deinit"], room, r)
                    head = twm.cut_after_requests(full, k)
                    last_request = [line for line in head[-1].split("\n") if line.startswith("|request|{")][-1]
                    h.feed(username, r, head)
                    h.feed(username, r, [UNAVAILABLE, last_request])              # the error, then the request again
                    h.feed(username, r, full[len(head):])
            h.settle(30.0)
            out[which] = h
        finally:
            h.close()
    new, old = out["new"], out["r1"]
    assert set(new.by_tag) == set(old.by_tag) and len(new.by_tag) > 100
    longer = 0
    for tag in new.by_tag:
        same_arrays(new.by_tag[tag], old.by_tag[tag], tag)
    assert {u: p.sent for u, p in new.players.items()} == {u: p.sent for u, p in old.players.items()}
    assert all(p.choices_refused == 0 for p in new.players.values())
    print(f"\nHANDOVER unavailable choice: {len(new.by_tag)} battles, the same episodes and the same messages as r1")


# ------------------------------------------------------------------------------------------------
# after a battle was given back
# ------------------------------------------------------------------------------------------------

def test_late_lines_for_a_room_that_was_given_back_do_nothing(tmp_root):
    mod = twm.worker_module("new", tmp_root)
    h = H(mod)
    try:
        username, room, messages = twm.recorded_battles()[0]
        request = [line for m in messages for line in m.split("\n") if line.startswith("|request|{") and '"wait":true' not in line][2]
        late = [request, "|turn|9", NOTHING, TOO_LATE, "|\n|win|" + username, "|\n|tie", "|expire|", "|deinit", "|deinit"]
        n_eps = 0
        for kind in ("win", "expire", "deinit"):
            r = f"{room}-late-{kind}"
            h.feed(username, r, twm.renamed(with_ending(kind, username, messages, 5), room, r))
            h.settle()
            n_eps += kind == "win"
            before = (len(h.learner.episodes), list(h.cleaned), list(h.player(username).sent), h.kept())
            for line in late:
                h.feed(username, r, [line], timeout=10.0)
            errs = h.feed_backlog(username, r, late, timeout=10.0)
            h.settle()
            after = (len(h.learner.episodes), list(h.cleaned), list(h.player(username).sent), h.kept())
            assert before == after, kind                                   # no episode, no cleanup, nothing sent, nothing kept
            assert before[3] == twm.NOTHING, (kind, before[3])
            # two "|deinit" queued at once: poke-env pops the room's lock twice (KeyError). r1 does the same.
            assert all("KeyError" in e for e in errs), errs
        assert len(h.learner.episodes) == n_eps == 1
        # A repairable refusal for a room that is gone (the server does not send one to a player who left):
        # the patch answers before it knows the room, as r1's did. One default, and the room stays marked.
        r = f"{room}-late-win"
        n_sent = len(h.player(username).sent)
        for _ in range(3):
            h.feed(username, r, [REPAIRABLE], timeout=10.0)
        assert h.player(username).sent[n_sent:] == [(r, "/choose default")]
        # poke-env makes a lock for any battle-room message and drops it only on "|deinit": one lock stays (r1 too).
        assert h.kept() == dict(twm.NOTHING, room_locks=1) and len(h.learner.episodes) == 1
        print("\nHANDOVER rooms still marked as having had a default:", h.player(username).ps_client.__dict__.get("_default_sent"))
    finally:
        h.close()


def test_a_room_tag_that_comes_again(tmp_root):
    """
    The same room tag with a new |init| after the room was closed without a result. (A Showdown
    server numbers its battles upwards and a worker does not outlive its server, so this should
    not happen in training.) Reported, not judged: what each version does.
    """
    username, room, messages = twm.recorded_battles()[0]
    k = 4
    out = {}
    for which in ("new", "r1"):
        h = H(twm.worker_module(which, tmp_root))
        try:
            r = f"{room}-again"
            h.feed(username, r, twm.renamed(with_ending("expire", username, messages, k), room, r))
            h.settle()
            try:
                h.feed(username, r, twm.renamed(list(messages) + ["|deinit"], room, r), timeout=20.0)
            except BaseException as e:          # noqa: BLE001
                out[which + "_error"] = repr(e)
            h.settle()
            out[which] = (len(h.learner.episodes), [len(e[1]) for e in h.learner.episodes], h.kept())
        finally:
            h.close()
    print("\nHANDOVER a room tag that comes again:", out, "requests in the stream:", n_requests(messages), "k:", k)
    assert out["new"][0] == 0                                                    # new: nothing handed over


# ------------------------------------------------------------------------------------------------
# slots and seats
# ------------------------------------------------------------------------------------------------

def test_slots_and_seats_come_back_exactly_once(tmp_root):
    """
    Two episode slots and two seats a player, 60 rooms that the server closes (at every stage: before
    the first decision, after one, after several) mixed with battles that end. A slot or a seat that
    did not come back stops the next battle; one that came back twice shows as a count above 2.
    Then 28 rooms at once on the same two slots and two seats.
    """
    mod = twm.worker_module("new", tmp_root)
    h = H(mod, seats=2, slots=2)
    try:
        streams = twm.recorded_battles()
        highest = 0
        n_closed = n_won = 0
        reference = {}
        for i in range(60):
            username, room, messages = streams[i % len(streams)]
            r = f"{room}-s{i}"
            won = i % 3 == 2
            if won:
                stream = twm.renamed(list(messages) + ["|deinit"], room, r)
                n_won += 1
            else:
                k = (i * 7) % (n_requests(messages) + 1)                      # 0: closed before its first decision
                stream = [messages[0], "|expire|", "|deinit"] if k == 0 else with_ending("expire" if i % 2 else "deinit", username, messages, k)
                stream = twm.renamed(stream, room, r)
                n_closed += 1
            if i % 4 == 0:
                assert h.feed_backlog(username, r, stream, timeout=60.0) == []
            else:
                h.feed(username, r, stream, timeout=60.0)
            h.settle()
            if won:
                reference[(username, room)] = h.by_tag[f"{username}_{r}"]
            highest = max(highest, h.ep_sem._value)
            assert h.ep_sem._value <= 2, f"a slot came back twice after room {i}"
            assert all(p._battle_count_queue.qsize() == 0 for p in h.players.values()), i
        assert h.kept() == twm.NOTHING and h.ep_sem._value == 2 and highest == 2
        assert len(h.learner.episodes) == n_won
        assert sum(p.rooms_closed for p in h.players.values()) == n_closed
        assert len(reference) == len(streams)

        # Now many rooms at once for each player, with 2 slots in all and 2 seats a player: every room has to
        # wait for a seat and a slot that an earlier room gives back.
        jobs, wins = [], []
        for rep in range(4):
            for j, (username, room, messages) in enumerate(streams):
                r = f"{room}-c{rep}"
                if (rep + j) % 2 == 0:
                    jobs.append((username, r, twm.renamed(list(messages) + ["|deinit"], room, r)))
                    wins.append((username, room, r))
                else:
                    k = 1 + (rep * 5 + j * 3) % n_requests(messages)
                    jobs.append((username, r, twm.renamed(with_ending("expire", username, messages, k), room, r)))
        players = {u: h.player(u) for u, _r, _s in jobs}

        async def run():
            tasks = []
            pending = [(u, r, list(s)) for u, r, s in jobs]
            while any(s for _u, _r, s in pending):
                for u, r, s in pending:
                    if s:
                        tasks.append(asyncio.ensure_future(h.mod.handle_message(players[u].ps_client, ">" + r + "\n" + s.pop(0))))
            done = await asyncio.gather(*tasks, return_exceptions=True)
            return [repr(x) for x in done if isinstance(x, BaseException)]

        errs = asyncio.run_coroutine_threadsafe(run(), h.loop).result(300)
        assert errs == [], errs[:3]
        h.settle(30.0)
        assert h.kept() == twm.NOTHING and h.ep_sem._value == 2, (h.kept(), h.ep_sem._value)
        for username, room, r in wins:
            same_arrays(h.by_tag[f"{username}_{r}"], reference[(username, room)], f"concurrent {username} {r}")
        assert len(h.learner.episodes) == n_won + len(wins)
        print(f"\nHANDOVER slots and seats: 60 rooms one after the other, then {len(jobs)} at once ({len(wins)} with a result) "
              f"on 2 slots and 2 seats a player: all given back, episodes equal to the same battle played alone")
    finally:
        h.close()


def test_a_room_closed_while_its_decision_is_out_waits_for_it(tmp_root):
    """
    |expire| and |deinit arrive while the policy's answer to the last request is still out (the
    listener has made them tasks; they wait on the room's lock). The step is taken, the choice is
    sent, then the room is given back: one CLEANUP, after the step, and no episode.
    """
    mod = twm.worker_module("new", tmp_root)
    h = H(mod, slots=1)
    try:
        username, room, messages = twm.recorded_battles()[0]
        order = []
        w = h.worker
        cln = w._cleanup_trajectory

        def cleanup(tag):
            order.append(("cleanup", tag, len(w._traj.get(tag, {"act": []})["act"])))
            cln(tag)

        w._cleanup_trajectory = cleanup
        for k in (1, 3, 6):
            r = f"{room}-out{k}"
            stream = twm.renamed(with_ending("expire", username, messages, k), room, r)
            assert h.feed_backlog(username, r, stream) == []
            h.settle()
            tag = f"{username}_{r}"
            assert order[-1] == ("cleanup", tag, k), order[-1]                 # all k steps were in the buffer when it was dropped
            assert len([m for m in h.sent_in(username, r) if m.startswith("/choose")]) == k
        assert h.kept() == twm.NOTHING and h.learner.episodes == [] and h.ep_sem._value == 1
    finally:
        h.close()
