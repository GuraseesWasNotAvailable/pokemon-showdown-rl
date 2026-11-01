"""
The ladder A/B of net, mix and mix-deep (branch ladder-ab): the dropped-order fix (ladder.OrderGuard),
the network's own score in every decision note, the deep arm's budget and its clock fallback, the
A/B report (bench/ab_report.py), the observation of the real checkpoints, and the launcher's line.

    python -m pytest tests/test_ladder_ab.py -q -p no:cacheprovider

No Showdown server and no Foul Play are needed: the order guard is driven with a fake send, the
player replays the recorded battles of tests/fixtures (as tests/test_search_player.py does) with a
fake search. The checkpoint tests look for learner_update_024381.pt and learner_update_011850.pt
(PSPPO_CKPT_024381 / PSPPO_CKPT_011850, else the usual places on the box and the laptop) and are
skipped without them.
"""

from __future__ import annotations

import asyncio
import glob
import importlib.util
import json
import math
import os
import re
import sys
import types
from collections import OrderedDict

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import foulplay_bridge as fb  # noqa: E402
import ladder  # noqa: E402
import search_player as sp  # noqa: E402
from search_player import DEEP_ARM, DEEP_BUDGET, MixSettings, SearchBudget  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "bench"))
import ab_report  # noqa: E402

_spec = importlib.util.spec_from_file_location("tsp_for_ab", os.path.join(REPO, "tests", "test_search_player.py"))
tsp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tsp)

LAUNCHER = os.path.join(REPO, "launchers", "laptop", "Look Play AB.cmd")
BOX_RUN = os.path.join(REPO, "run_box_ab.sh")
BOX_HELPER = os.path.join(REPO, "cloud", "box_ladder.sh")
NOTICE = '|raw|<strong class="message-throttle-notice">Your message was not sent because you\'ve been typing too quickly.</strong>'


# =============================================================================================
# The dropped-order fix: OrderGuard with a fake send
# =============================================================================================

class Wire:
    """A fake websocket side: what was sent and when, and a guard in front of it as ladder.make_player puts it."""

    def __init__(self, **kw):
        self.sent = []                       # (time, room, message, message_2)
        self.rq = {}                         # room -> current rqid
        self.over = set()
        self.lines = []
        self.notes = []
        kw.setdefault("spacing_s", 0.0)
        kw.setdefault("resend_after_s", 0.05)
        kw.setdefault("jitter_s", 0.0)
        kw.setdefault("timer_after_s", 0.0)
        self.guard = ladder.OrderGuard(self.raw, me="Our Bot", current_rqid=lambda room: self.rq.get(room),
                                       finished=lambda room: room in self.over, say=self.lines.append,
                                       note=lambda room, data: self.notes.append((room, data)), **kw)

    async def send(self, message, room="", message_2=None):
        """The player's whole send_message: the gate, the bookkeeping, the websocket."""
        await self.guard.gate(message, room)
        if ladder.OrderGuard.is_order(message, room):
            self.guard.sent(room, self.rq.get(room), message)
        self.sent.append((asyncio.get_running_loop().time(), room, message, message_2))

    async def raw(self, message, room, rqid):
        """The websocket alone: how a copy goes out (ladder.make_player's send_copy), after the guard's own gate."""
        self.sent.append((asyncio.get_running_loop().time(), room, message, rqid))

    def orders(self, room=None):
        return [(m, m2) for _t, r, m, m2 in self.sent if m.startswith("/choose") and (room is None or r == room)]


def run(coro):
    return asyncio.run(coro)


def test_a_notice_in_the_battle_room_sends_its_order_again_once_with_the_request_id():
    async def go():
        w = Wire()
        w.rq["battle-x-1"] = 7
        await w.send("/choose switch Ariados", "battle-x-1")
        w.guard.on_frame(">battle-x-1\n" + NOTICE)
        await asyncio.sleep(0.15)
        return w

    w = run(go())
    assert w.orders("battle-x-1") == [("/choose switch Ariados", None), ("/choose switch Ariados", "7")]
    assert w.guard.stats["resends"] == 1 and w.guard.stats["notices"] == 1
    assert len(w.lines) == 1 and "sent again (1 of 3" in w.lines[0] and "/choose switch Ariados" in w.lines[0]
    assert [d for _r, d in w.notes if "throttle_resend" in d][0]["throttle_resend"]["rqid"] == 7
    assert w.guard.expects_error("battle-x-1")                   # its "too late" would be no news
    counts = w.guard.game_counts("battle-x-1")
    assert counts["throttle_notices"] == 1 and counts["throttle_resends"] == 1


def test_no_copy_when_a_new_request_came_the_server_reacted_or_the_game_ended():
    async def go(after):
        w = Wire()
        room = "battle-x-2"
        w.rq[room] = 3
        await w.send("/choose move 1", room)
        w.guard.on_frame(">%s\n%s" % (room, NOTICE))
        after(w, room)
        await asyncio.sleep(0.15)
        return w

    def new_request(w, room):
        w.rq[room] = 4

    def reaction(w, room):
        w.guard.on_frame(">%s\n|\n|t:|1\n|move|p1a: X|Tackle|p2a: Y\n|turn|5" % room)

    def over(w, room):
        w.over.add(room)

    def timer_only(w, room):
        w.guard.on_frame(">%s\n|inactive|Time left: 150 sec this turn | 150 sec total" % room)   # not a reaction

    for after, copies in ((new_request, 0), (reaction, 0), (over, 0), (timer_only, 1)):
        w = run(go(after))
        assert len(w.orders()) == 1 + copies, after.__name__
    w = run(go(new_request))
    assert w.guard.stats["not_resent"] == {"a new request": 1}


def test_at_most_three_copies_per_request_and_a_newer_order_is_the_one_sent():
    async def go():
        w = Wire()
        room = "battle-x-3"
        w.rq[room] = 9
        await w.send("/choose move 2", room)
        for _ in range(5):
            w.guard.on_frame(">%s\n%s" % (room, NOTICE))
            await asyncio.sleep(0.3)                         # the copies wait 0.05, 0.1 and 0.2 s (doubling)
        w.rq[room] = 10                                      # the next request: a fresh count
        await w.send("/choose move 3", room)
        w.guard.on_frame(">%s\n%s" % (room, NOTICE))
        await asyncio.sleep(0.1)
        return w

    w = run(go())
    copies = [o for o in w.orders() if o[1] is not None]
    assert copies == [("/choose move 2", "9")] * 3 + [("/choose move 3", "10")]
    assert w.guard.stats["not_resent"].get("sent again 3 times already") == 2


def test_the_notice_copies_wait_longer_each_time():
    async def go():
        w = Wire(resend_after_s=0.1)
        room = "battle-x-4"
        w.rq[room] = 1
        await w.send("/choose move 1", room)
        start = asyncio.get_running_loop().time()
        for _ in range(3):
            w.guard.on_frame(">%s\n%s" % (room, NOTICE))
            await asyncio.sleep(0.5)
        return w, start

    w, start = run(go())
    times = [t - start for t, _r, m, m2 in w.sent if m2 is not None]
    assert len(times) == 3
    assert 0.08 <= times[0] < 0.25 and 0.5 + 0.18 <= times[1] < 0.5 + 0.35 and 1.0 + 0.38 <= times[2] < 1.0 + 0.55, times


def test_timer_lines_send_copies_past_the_notice_cap():
    """Smoke 2's lost game (309126): an order and its 3 copies all dropped, then the timer at 120, 90, 60 s
    said the server still waited, and nothing was sent. Each timer line now sends one copy, up to 8."""
    clock = {"t": 100.0}

    async def go():
        w = Wire(clock=lambda: clock["t"], max_timer_resends=2)
        room = "battle-t-9"
        w.rq[room] = 4
        await w.send("/choose move 3", room)
        for _ in range(4):                                     # 3 copies, then the cap
            w.guard.on_frame(">%s\n%s" % (room, NOTICE))
            await asyncio.sleep(0.3)
        assert len(w.orders(room)) == 4
        for left in (120, 90, 60):
            clock["t"] += 30.0
            w.guard.on_frame(">%s\n|inactive|Our Bot has %d seconds left." % (room, left))
            await asyncio.sleep(0.05)
        return w

    w = run(go())
    copies = [o for o in w.orders("battle-t-9") if o[1] is not None]
    assert copies == [("/choose move 3", "4")] * 5                  # 3 for the notices, 2 for the timer (its cap here)
    s = w.guard.stats
    assert s["timer_lines"] == 3 and s["timer_resends"] == 2 and s["resends"] == 5
    assert s["not_resent"].get("sent again 3 times already") == 1
    assert s["not_resent"].get("sent again 2 times for the timer already") == 1
    timer_notes = [d["throttle_resend"] for _r, d in w.notes if "throttle_resend" in d and d["throttle_resend"]["timer"]]
    assert [n["attempt"] for n in timer_notes] == [4, 5]
    assert any("for the timer" in line for line in w.lines)


def test_a_copy_that_went_stale_while_it_waited_for_the_spacing_is_not_sent():
    """A new request comes while the copy waits for the gate: it would go with the old rqid, be refused
    "too late", and that error would make poke-env decide again."""
    async def go():
        w = Wire(spacing_s=0.3, max_wait_s=1.5, resend_after_s=0.0)
        room = "battle-s-1"
        w.rq[room] = 7
        await w.send("/choose move 1", room)
        w.guard.on_frame(">%s\n%s" % (room, NOTICE))             # the copy is due at once, then waits ~0.3 s
        await asyncio.sleep(0.1)
        w.rq[room] = 8                                           # the next request, while it waits
        await asyncio.sleep(0.4)
        return w

    w = run(go())
    assert w.orders("battle-s-1") == [("/choose move 1", None)]
    assert w.guard.stats["stale_after_spacing"] == 1 and w.guard.stats["not_resent"] == {"a new request": 1}
    assert w.guard.stats["resends"] == 0 and not w.lines


def test_a_copy_leaves_the_order_on_record_alone():
    """A copy is not the battle's order on record: the pending order keeps its own request id and text."""
    async def go():
        w = Wire()
        room = "battle-r-1"
        w.rq[room] = 3
        await w.send("/choose move 2", room)
        w.guard.on_frame(">%s\n%s" % (room, NOTICE))
        await asyncio.sleep(0.15)
        p = dict(w.guard.pending[room])
        w.rq[room] = 4
        await w.send("/choose switch Pikachu", room)
        return w, p

    w, p = run(go())
    assert p["rqid"] == 3 and p["message"] == "/choose move 2" and p["resends"] == 1 and p["timer_resends"] == 0
    q = w.guard.pending["battle-r-1"]
    assert q["rqid"] == 4 and q["message"] == "/choose switch Pikachu" and q["resends"] == 0


def test_one_late_error_per_copy_is_let_pass_whatever_the_request():
    clock = {"t": 50.0}

    async def go():
        w = Wire(clock=lambda: clock["t"])
        room = "battle-l-1"
        w.rq[room] = 5
        await w.send("/choose move 1", room)
        w.guard.on_frame(">%s\n%s" % (room, NOTICE))
        await asyncio.sleep(0.15)
        return w

    w = run(go())
    g = w.guard
    room = "battle-l-1"
    assert g.expects_error(room)
    w.rq[room] = 6                                              # a new request: the copy's "too late" is still no news
    assert g.late_errors_to_ignore(room, 2) == 1                # one per copy
    assert g.late_errors_to_ignore(room, 1) == 0 and not g.expects_error(room)
    assert g.late_errors_to_ignore("battle-other", 1) == 0
    g._copies[room] = [clock["t"]]
    clock["t"] += 11.0                                          # after late_window_s (10 s): news again
    assert g.late_errors_to_ignore(room, 1) == 0
    assert g.stats["late_ignored"] == 1


def test_orders_of_different_games_go_out_at_least_the_spacing_apart_and_the_wait_is_counted():
    async def go():
        w = Wire(spacing_s=0.3, max_wait_s=1.5)
        for n in range(3):
            w.rq["battle-x-%d" % n] = 1
        await asyncio.gather(*(w.send("/choose move 1", "battle-x-%d" % n) for n in range(3)))
        return w

    w = run(go())
    times = sorted(t for t, *_ in w.sent)
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert len(times) == 3 and all(g >= 0.29 for g in gaps), gaps
    s = w.guard.stats
    assert s["orders"] == 3 and s["waits"] == 2 and 550.0 <= s["wait_ms_max"] <= 1000.0
    assert sum(w.guard.games[r]["order_waits"] for r in w.guard.games) == 2


def test_the_spacing_never_holds_an_order_longer_than_max_wait():
    async def go():
        w = Wire(spacing_s=0.3, max_wait_s=0.4)
        for n in range(5):
            w.rq["battle-y-%d" % n] = 1
        await asyncio.gather(*(w.send("/choose move 1", "battle-y-%d" % n) for n in range(5)))
        return w

    w = run(go())
    assert w.guard.stats["wait_ms_max"] <= 400.0 + 60.0


def test_a_notice_without_a_room_sends_no_order_again():
    """Showdown sends the notice to the room of the dropped message (users.ts sendTo): an order always goes to its
    battle room, so a notice without a room was a /search, /utm or /leave. No order is sent again for it."""
    async def go():
        w = Wire()
        for room in ("battle-z-1", "battle-z-2"):
            w.rq[room] = 5
            await w.send("/choose move 4", room)
        w.guard.on_frame(NOTICE)                                                  # no ">room" line
        w.guard.on_frame(">lobby\n" + NOTICE)                                     # a chat room: the same
        await asyncio.sleep(0.15)
        return w

    w = run(go())
    assert w.orders("battle-z-1") == [("/choose move 4", None)] and w.orders("battle-z-2") == [("/choose move 4", None)]
    s = w.guard.stats
    assert s["notices"] == 2 and s["notices_without_room"] == 2 and s["resends"] == 0
    assert w.guard.no_room_notice_at is not None and w.guard.no_room_notice_wall is not None
    counts = w.guard.game_counts("battle-z-2")
    assert counts["throttle_notices"] == 0                     # not counted against every game
    assert len(w.lines) == 2 and "no order is sent again" in w.lines[0]


def test_chat_that_quotes_the_notice_is_not_a_notice():
    async def go():
        w = Wire()
        room = "battle-c-1"
        w.rq[room] = 2
        await w.send("/choose move 1", room)
        w.guard.on_frame(">%s\n|c|Opponent|<strong class=\"message-throttle-notice\">gotcha</strong>" % room)
        w.guard.on_frame(">%s\n|c:|1791|Opponent|message-throttle-notice" % room)
        w.guard.on_frame("|pm|Someone|Our Bot|" + NOTICE[5:])                     # a private message
        w.guard.on_frame("|pm|Someone|Our Bot|" + NOTICE)
        await asyncio.sleep(0.15)
        return w

    w = run(go())
    assert w.orders() == [("/choose move 1", None)]
    assert w.guard.stats["notices"] == 0 and w.guard.no_room_notice_at is None


def test_the_search_is_asked_again_only_when_its_notice_came_after_it_and_the_server_is_not_searching():
    f = ladder.search_was_dropped
    assert f(102.0, 101.5, 104.5, [], "gen9randombattle")                          # dropped: ask again
    assert not f(102.0, 101.5, 103.0, [], "gen9randombattle")                      # under 2 s: wait for |updatesearch|
    assert not f(102.0, 101.5, 104.5, ["gen9randombattle"], "gen9randombattle")    # the server holds it
    assert not f(101.0, 101.5, 104.5, [], "gen9randombattle")                      # the notice came before this search
    assert not f(None, 101.5, 104.5, [], "gen9randombattle") and not f(102.0, None, 104.5, None, "gen9randombattle")


def test_the_account_lock_refuses_a_second_bot_on_the_same_account(tmp_path, monkeypatch):
    """The same lock as the PSBot Lookahead app's ladder.py: main() refuses before any password or connection."""
    if os.name == "nt":
        pytest.skip("the flock path; Windows takes a named mutex (the same code as the app)")
    monkeypatch.setenv("PSBOT_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(ladder, "_ACCOUNT_LOCK", None)
    held = None
    try:
        assert ladder.take_account_lock("public", "abtestuser") is None
        held = ladder._ACCOUNT_LOCK
        again = ladder.take_account_lock("public", "abtestuser")
        assert again is not None and "another bot is already playing as abtestuser" in again
        assert ladder.take_account_lock("public", "someoneelse") is None              # another account: free
        ladder._ACCOUNT_LOCK.close()
        ckpt = tmp_path / "learner_update_000001.pt"
        ckpt.write_bytes(b"x")
        monkeypatch.setattr(ladder.getpass, "getpass", lambda *a, **k: pytest.fail("asked for a password"))
        monkeypatch.setattr(ladder, "run", lambda *a, **k: pytest.fail("started playing"))
        assert ladder.main(["--user", "ABTestUser", "--checkpoint", str(ckpt), "--games", "1"]) == 3
        assert sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".lock") == [
            "PSBotLadder.public.abtestuser.lock", "PSBotLadder.public.someoneelse.lock"]
    finally:
        if held is not None:
            held.close()


def test_the_timer_saying_we_have_not_chosen_sends_the_order_again(monkeypatch):
    clock = {"t": 100.0}
    async def go():
        w = Wire(clock=lambda: clock["t"])
        room = "battle-t-1"
        w.rq[room] = 2
        await w.send("/choose move 1", room)
        w.guard.on_frame(">%s\n|inactive|Our Bot has 120 seconds left." % room)      # at once: not yet evidence
        await asyncio.sleep(0.1)
        clock["t"] += 29.0
        w.guard.on_frame(">%s\n|inactive|Someone Else has 120 seconds left." % room)  # the opponent's: no
        await asyncio.sleep(0.1)
        w.guard.on_frame(">%s\n|inactive|Our Bot has 120 seconds left." % room)
        await asyncio.sleep(0.1)
        return w

    w = run(go())
    assert w.orders() == [("/choose move 1", None), ("/choose move 1", "2")]
    assert w.guard.stats["timer_lines"] == 1


def test_the_battle_file_of_the_lost_game_would_have_been_answered():
    """The frames of battle-gen9randombattle-2694716418 around the dropped forced switch."""
    async def go():
        w = Wire()
        room = "battle-gen9randombattle-2694716418-abc"
        w.guard.on_frame(">%s\n|request|{\"forceSwitch\":[true],\"rqid\":35}" % room)
        w.rq[room] = 35
        await w.send("/choose switch Ariados", room)
        w.guard.on_frame(">%s\n|inactive|Time left: 150 sec this turn | 150 sec total | 5 sec grace" % room)
        w.guard.on_frame(">%s\n%s" % (room, NOTICE))
        await asyncio.sleep(0.15)
        return w

    w = run(go())
    assert w.orders() == [("/choose switch Ariados", None), ("/choose switch Ariados", "35")]


# =============================================================================================
# The network's own score: value_fields, the value head kept from the same forward pass
# =============================================================================================

def test_value_fields_turn_v_exp_and_phi_into_a_win_estimate():
    f = sp.value_fields(0.2, 0.3, 0.4, 0.5)
    assert f == {"v_exp": 0.2, "v_std": 0.3, "phi": 0.4, "phi_scale": 0.5, "win_est": round(0.5 * (1 + 0.2 + 0.5 * 0.4), 4)}
    assert sp.value_fields(None, None, 0.4, 0.5) == {}
    assert sp.value_fields(1.4, None, 0.9, 0.5)["win_est"] == 1.0           # clipped
    assert sp.value_fields(0.2, None, None, 0.5)["win_est"] is None          # no Phi: no estimate
    assert sp.value_fields(0.2, None, None, 0.0)["win_est"] == 0.6           # trained without shaping: v_exp alone


def test_value_settings_come_from_the_checkpoints_run_config():
    import hybrid

    ck = {"run_cfg": {"reward": {"shaping": "potential", "shaping_scale": 0.3, "potential_file": "/x/w.json"}, "learner": {"gamma": 0.99}}}
    assert hybrid.value_settings_from_checkpoint(ck) == {"shaping": "potential", "scale": 0.3, "gamma": 0.99,
                                                         "potential_file": "/x/w.json", "from": "checkpoint"}
    d = hybrid.value_settings_from_checkpoint({})
    assert d["from"] == "default" and d["scale"] == 0.5 and d["shaping"] == "potential"


@tsp.needs_player
def test_the_value_head_is_kept_from_the_pass_that_made_the_logits():
    """A real (untrained) network: net_values holds exactly the v_exp of the forward pass, and v_std its spread."""
    import numpy as np
    import torch

    from players import batched_policy_player

    player = tsp.offline(batched_policy_player(), "valuecheck")
    obs = np.zeros((2, player.assembler.total_dim), dtype=np.float32)
    obs[1, :50] = 0.5
    logits = player._forward_batch(["b1", "b2"], obs)
    assert logits.shape == (2, 14) and set(player.net_values) == {"b1", "b2"}
    net = player.policy.net
    with torch.no_grad():
        # The pass _forward_batch makes for two battles with no history yet.
        cache = [torch.zeros(2, net.kv_cache_len, net.d_model) for _ in range(net.n_layers)]
        _out, v_logits, v_exp, _kv, _len = net(torch.from_numpy(obs), kv_cache=cache, kv_cache_len=torch.zeros(2, dtype=torch.long))
    for row, tag in enumerate(("b1", "b2")):
        got_exp, got_std = player.net_values[tag]
        assert math.isclose(got_exp, float(v_exp[row]), abs_tol=1e-5)
        probs = torch.softmax(v_logits[row].float(), dim=-1)
        std = float(((probs * (net.v_support - v_exp[row]) ** 2).sum()).sqrt())
        assert math.isclose(got_std, std, abs_tol=1e-5)


@pytest.fixture()
def classes():
    """
    The search player with the stub network of test_search_player, which also gives a value: v_exp
    0.25, v_std 0.1. For one test at a time: hybrid.PolicyNet is the stub only while it runs.
    """
    import numpy as np

    import hybrid

    real = hybrid.PolicyNet
    hybrid.PolicyNet = tsp.StubPolicy

    class Valued(sp.search_player()):
        def _forward_batch(self, tags, obs):
            for tag in tags:
                self.policy._cache[tag] = True
                self.net_values[tag] = (0.25, 0.1)
            self.batch_stats["forwards"] += 1
            self.batch_stats["rows"] += len(tags)
            return np.stack([tsp.stub_logits(row) for row in obs])

    yield {"search": Valued}
    hybrid.PolicyNet = real


@tsp.needs_player
@pytest.mark.parametrize("mode", ["net", "mix"])
def test_every_decision_note_of_net_and_mix_has_the_value(classes, mode):
    battles = tsp.load_fixture()[:2]
    bridge = tsp.FakeBridge(battles)
    notes = []
    try:
        for events in battles:
            player = tsp.searcher(classes, events[-1]["username"], bridge, mode=mode, on_decision=notes.append)
            tsp.play(player, events)
    finally:
        bridge.close()
    assert notes and all(n["mode"] == mode and n["arm"] == mode for n in notes)
    for n in notes:
        assert n["v_exp"] == 0.25 and n["v_std"] == 0.1 and n["phi_scale"] == 0.5
        assert isinstance(n["phi"], float) and -1.0 <= n["phi"] <= 1.0
        assert math.isclose(n["win_est"], max(0.0, min(1.0, 0.5 * (1.25 + 0.5 * n["phi"]))), abs_tol=1e-4)
    assert len({n["phi"] for n in notes}) > 1                         # it follows the position
    if mode == "mix":
        searched = [n for n in notes if n.get("iterations")]
        assert searched and all(n["asked_ms"] == 20 and n["worlds_x"] == 1 and n["worlds"] >= 1 for n in searched)
        assert all("iterations_per_world" in n and "full_width_turns" in n for n in searched)
        assert all(n["beside"] == {} for n in searched)              # one battle at a time: nothing beside


class _Searches:
    """Just what search_player's _search_done reads and writes."""

    def __init__(self):
        from collections import deque

        self._inflight = {}
        self._searches_done = deque(maxlen=64)


@tsp.needs_player
def test_search_done_says_which_searches_ran_beside_and_for_how_long():
    done = sp.search_player()._search_done
    me = _Searches()
    me._inflight = {"a": ("today", 0.0), "b": ("deep", 1.0), "c": ("today", 3.0), "old": ("deep", -40.0)}
    me._searches_done.append(("d", "deep", -5.0, 0.5))           # half a second of it overlapped a's search
    me._searches_done.append(("e", "today", -5.0, -1.0))         # over before a began
    assert done(me, "a", 0.0, 2.0) == {"deep": round((1.0 + 0.5) / 2.0, 3)}   # b from 1 to 2, d from 0 to 0.5; c not yet
    assert "a" not in me._inflight and "old" not in me._inflight              # a stale entry (a decision that raised) goes
    assert me._searches_done[-1] == ("a", "today", 0.0, 2.0)
    # b ends at 4: a beside it from 1 to 2 (today), c from 3 to 4 (today)
    assert done(me, "b", 1.0, 4.0) == {"today": round(2.0 / 3.0, 3)}


@tsp.needs_player
def test_the_game_summary_counts_the_paired_compute_groups(classes):
    battles = tsp.load_fixture()[:1]
    bridge = tsp.FakeBridge(battles)
    try:
        player = tsp.searcher(classes, battles[0][-1]["username"], bridge, mode="mix")
        tsp.play(player, battles[0])
    finally:
        bridge.close()
    summary = player.game_summary(battles[0][0]["room"])
    assert summary["searched"] > 0 and summary["n_no_deep"] == summary["searched"] and summary["n_deep_beside"] == 0
    assert summary["it_no_deep"] > 0 and summary["it_deep_beside"] == 0


# =============================================================================================
# The deep arm: its own budget, and its fall back to today's
# =============================================================================================

def test_the_deep_budget_is_five_times_todays_and_fits_its_deadline():
    today = SearchBudget(search_ms=500, worlds_x=1, deadline=2.5)
    assert DEEP_BUDGET.compute() / today.compute() == 5.0
    assert sp.minimum_deadline(1250, 2) < DEEP_BUDGET.deadline < 10.0
    assert sp.minimum_deadline(500) == pytest.approx(1.62)
    with pytest.raises(ValueError):
        SearchBudget(search_ms=1250, worlds_x=2, deadline=5.0)              # 6.62 s needed
    assert sp.split_mode(DEEP_ARM) == ("mix", None) and DEEP_ARM in sp.MODES and DEEP_ARM in ladder.PLAYER_MODES


def test_deep_workers_make_the_same_budget_fit_a_shorter_deadline():
    # The worlds W at a time: the same search (compute and the record's ms and worlds), less time.
    many = SearchBudget(search_ms=1250, worlds_x=2, deadline=3.0, workers=4)
    assert many.compute() == DEEP_BUDGET.compute() and many.as_dict()["workers"] == 4
    assert "workers" not in DEEP_BUDGET.as_dict()                          # one at a time, as before: the same record
    assert sp.minimum_deadline(1250, 2, 4) < 3.0 < sp.minimum_deadline(1250, 2)
    assert sp.minimum_deadline(1250, 2, 1) == sp.minimum_deadline(1250, 2) == pytest.approx(6.62)
    with pytest.raises(ValueError):
        SearchBudget(search_ms=1250, worlds_x=2, deadline=3.0)                # one at a time: 6.62 s needed
    with pytest.raises(ValueError):
        SearchBudget(search_ms=1250, worlds_x=2, deadline=7.5, workers=0)
    # 4 workers on 2x the worlds: 4 worlds at 1250 ms in one round, or 8 at 625 ms in two; 8 workers: one round either way.
    assert fb.search_seconds(1250, 2, 4) == pytest.approx(1.25 * 1.25 + 0.02 + 4 * fb.FORK_S_PER_WORKER)
    assert fb.search_seconds(1250, 2, 8) == pytest.approx(1.25 * 1.25 + 0.02 + 8 * fb.FORK_S_PER_WORKER)
    assert fb.search_seconds(1250, 2, 2) == pytest.approx(1.25 * 1.25 * 2 + 0.02 + 2 * fb.FORK_S_PER_WORKER)
    assert fb.search_seconds(500, 1, 1) == pytest.approx(2.5 * 0.5 + 0.02)


def test_the_helper_reckons_a_decision_as_the_bridge_does():
    path = os.path.join(REPO, "bench", "fp_bridge_server.py")
    spec = importlib.util.spec_from_file_location("fp_bridge_server_for_test", path)
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)                     # nothing of Foul Play is imported before a _Server is made
    assert server.FORK_S_PER_WORKER == fb.FORK_S_PER_WORKER
    for ms in (2, 20, 500, 1250):
        for p in (1, 2, 3):
            for w in (0, 1, 2, 3, 4, 8):
                assert server._Server.search_seconds(ms, p, w) == pytest.approx(fb.search_seconds(ms, p, w)), (ms, p, w)


def test_the_deep_workers_flag_and_more_helpers_for_six_games():
    args = ladder.parse_args(["--arms", "net,mix,mix-deep", "--parallel", "6", "--helpers", "7", "--deep-workers", "4",
                              "--deep-deadline", "4.0"])
    budget = ladder.deep_budget(args)
    assert (args.helpers, budget.workers, budget.deadline) == (7, 4, 4.0) and ladder.search_problems(args) == []
    assert ladder.parse_args(["--helpers", "40"]).helpers == ladder.MAX_HELPERS == 12
    assert ladder.parse_args(["--deep-workers", "0"]).deep_workers == 1
    assert ladder.deep_budget(ladder.parse_args([])).as_dict() == DEEP_BUDGET.as_dict()      # no flag: as before
    # One at a time the same deadline cannot hold the deep search: said before anything starts.
    args = ladder.parse_args(["--arms", "net,mix,mix-deep", "--parallel", "6", "--helpers", "7", "--deep-deadline", "4.0"])
    assert any("the deep arm" in p for p in ladder.search_problems(args))


def test_full_width_turns():
    assert sp.full_width_turns(450_000, 2, 9, 9) == round(math.log(225_000) / math.log(81), 3)
    assert sp.full_width_turns(2_250_000, 4, 9, 9) > sp.full_width_turns(450_000, 2, 9, 9)
    assert sp.full_width_turns(0, 2, 9, 9) is None and sp.full_width_turns(1000, 1, 1, 1) is None


def deep_player(classes, bridge, name, budget, **kw):
    return tsp.searcher(classes, name, bridge, mode=DEEP_ARM, budgets={DEEP_ARM: budget}, **kw)


@tsp.needs_player
def test_the_deep_arm_asks_its_own_time_worlds_and_deadline_and_mix_asks_todays(classes):
    battles = tsp.load_fixture()[:2]
    budget = SearchBudget(search_ms=40, worlds_x=2, deadline=1.0)
    bridge = tsp.FakeBridge(battles)
    try:
        deep = deep_player(classes, bridge, battles[0][-1]["username"], budget)
        tsp.play(deep, battles[0])
        asked_deep = list(bridge.asked)
        bridge.asked.clear()
        mix = tsp.searcher(classes, battles[1][-1]["username"], bridge, mode="mix")
        tsp.play(mix, battles[1])
        asked_mix = list(bridge.asked)
    finally:
        bridge.close()
    assert asked_deep and all(ms == 40 and more.get("parallelism") == 2 and 0.5 < ttl <= 1.0 for _k, _r, ttl, ms, more in asked_deep)
    assert asked_mix and all(ms == 20 and "parallelism" not in more and ttl <= 0.5 for _k, _r, ttl, ms, more in asked_mix)
    room = battles[0][0]["room"]
    recs = [d for d in deep.decisions_of(room) if d["asked"]]
    assert recs and all(d["asked_ms"] == 40 and d["worlds_x"] == 2 and d["deadline_s"] == 1.0 for d in recs)
    summary = deep.game_summary(room)
    assert summary["arm"] == DEEP_ARM and summary["budget"] == budget.as_dict() and summary["deep_fallback"] is None
    assert "budget" not in mix.game_summary(battles[1][0]["room"])
    line = ladder.arm_fields(summary, "p1", 500, "veto")
    assert line["search_ms"] == 40 and line["mix_rule"] == "veto"


@tsp.needs_player
def test_the_deep_arm_asks_for_its_workers_and_mix_does_not(classes):
    battles = tsp.load_fixture()[:2]
    budget = SearchBudget(search_ms=40, worlds_x=2, deadline=1.0, workers=4)
    bridge = tsp.FakeBridge(battles)
    try:
        deep = deep_player(classes, bridge, battles[0][-1]["username"], budget)
        tsp.play(deep, battles[0])
        asked_deep = list(bridge.asked)
        bridge.asked.clear()
        mix = tsp.searcher(classes, battles[1][-1]["username"], bridge, mode="mix")
        tsp.play(mix, battles[1])
        asked_mix = list(bridge.asked)
    finally:
        bridge.close()
    assert asked_deep and all(ms == 40 and more.get("parallelism") == 2 and more.get("workers") == 4 for _k, _r, _t, ms, more in asked_deep)
    assert asked_mix and all("workers" not in more and "parallelism" not in more for _k, _r, _t, _ms, more in asked_mix)
    recs = [d for d in deep.decisions_of(battles[0][0]["room"]) if d["asked"]]
    assert recs and all(d["workers"] == 4 and d["asked_ms"] == 40 and d["worlds_x"] == 2 for d in recs)
    assert deep.game_summary(battles[0][0]["room"])["budget"]["workers"] == 4


@tsp.needs_player
def test_the_deep_arm_falls_back_to_todays_budget_when_the_clock_is_low(classes):
    events = tsp.load_fixture()[0]
    room = events[0]["room"]
    budget = SearchBudget(search_ms=40, worlds_x=2, deadline=1.0, min_time_left=10_000)    # any clock reading is "low"
    bridge = tsp.FakeBridge([events])
    try:
        player = deep_player(classes, bridge, events[-1]["username"], budget)
        player._time_left[room] = 60.0
        tsp.play(player, events)
    finally:
        bridge.close()
    decisions = player.decisions_of(room)
    recs = [d for d in decisions if d["asked"]]
    assert decisions and "the battle timer said" in decisions[0]["deep_fallback"]["why"]     # at the first decision
    assert recs and all(d["asked_ms"] == 20 and d["worlds_x"] == 1 for d in recs)
    assert sum(1 for d in decisions if d.get("deep_fallback")) == 1                          # said once
    assert all("parallelism" not in more for *_x, more in bridge.asked)
    assert player.game_summary(room)["deep_fallback"]["turn"] >= 0


@tsp.needs_player
def test_the_deep_arm_falls_back_after_a_slow_decision(classes):
    events = tsp.load_fixture()[0]
    room = events[0]["room"]
    budget = SearchBudget(search_ms=40, worlds_x=2, deadline=1.0, max_decision_s=0.0)
    bridge = tsp.FakeBridge([events])
    try:
        player = deep_player(classes, bridge, events[-1]["username"], budget)
        tsp.play(player, events)
    finally:
        bridge.close()
    decisions = player.decisions_of(room)
    first = decisions[0]
    assert "a decision took" in first["deep_fallback"]["why"] and first.get("asked_ms", 40) == 40   # its own budget, then today's
    assert all(d["asked_ms"] == 20 for d in decisions[1:] if d["asked"])


# =============================================================================================
# The A/B report
# =============================================================================================

def rows_of(arm, n, wins, tag="ab3", rating=1500.0, iters=450_000, start=0, **extra):
    """n result lines of one arm with `wins` wins spread evenly over them."""
    out = []
    for i in range(n):
        won = ((i + 1) * wins) // n > (i * wins) // n
        out.append(dict({"experiment": tag, "arm": arm, "won": won, "opponent_rating": rating, "role": "p1" if i % 2 else "p2",
                         "searched": 0 if arm == "net" else 30, "decisions": 30,
                         "iterations_per_decision": 0 if arm == "net" else iters, "time": start + i}, **extra))
    return out


def shuffle_in(*lists):
    """Interleave the arms' games, as a run plays them."""
    out = []
    for group in zip(*lists):
        out.extend(group)
    return out


def test_report_before_the_interim_look_gives_no_verdict():
    out = ab_report.report(shuffle_in(rows_of("net", 249, 150), rows_of("mix", 249, 125), rows_of("mix-deep", 249, 125, iters=2_250_000)))
    assert out["verdict"] and out["verdict"][0].startswith("before the interim look") and "kept" not in out


def test_the_interim_look_stops_an_arm_only_at_z_2_8():
    for net_wins, stops in ((162, True), (155, False)):
        rows = shuffle_in(rows_of("net", 250, net_wins) + rows_of("net", 50, 25), rows_of("mix", 250, 125) + rows_of("mix", 50, 25),
                          rows_of("mix-deep", 250, 125, iters=2_250_000) + rows_of("mix-deep", 50, 25, iters=2_250_000))
        out = ab_report.report(rows)
        z = out["looks"]["interim"]["net"]["z"]
        assert (abs(z) >= 2.8) == stops
        assert ("net" in out.get("stopped", {})) == stops
        assert any(("STOP net" in v) == stops for v in out["verdict"] if v.startswith("interim look") and "net -" in v)
        assert out["verdict"][-1].startswith("before the final look")


def test_the_interim_look_reads_the_first_250_games_only():
    early = rows_of("net", 250, 125) + rows_of("net", 50, 50)      # a late streak does not move the interim look
    mix = rows_of("mix", 250, 125) + rows_of("mix", 50, 25)
    deep = rows_of("mix-deep", 250, 125, iters=2_250_000) + rows_of("mix-deep", 50, 25, iters=2_250_000)
    out = ab_report.report(shuffle_in(early, mix, deep))
    assert out["looks"]["interim"]["net"]["diff"] == 0.0
    assert out["now"]["net"]["diff"] > 0.0                          # while all games so far show it


@pytest.mark.parametrize("net, mix, deep, kept", [
    (246, 250, 254, "mix"),        # deep the highest; mix within 1 point of it, net 1.6 below: the cheaper mix
    (240, 250, 254, "mix"),        # net -2.0: mix stays; deep within 1 point: not deep
    (252, 250, 256, "net"),        # deep the highest; net within 1 point of it (mix 1.2 below): the cheapest, net
    (240, 250, 262, "mix-deep"),   # deep +2.4: the highest, kept (not significant)
    (270, 250, 262, "net"),        # net the highest
    (250, 250, 250, "net"),        # all level: the cheapest
])
def test_the_final_verdict_keeps_the_highest_and_the_cheaper_within_one_point(net, mix, deep, kept):
    # 500 games per arm, then 20 more losses each that the final look must not read.
    rows = shuffle_in(rows_of("net", 500, net) + rows_of("net", 20, 0), rows_of("mix", 500, mix) + rows_of("mix", 20, 0),
                      rows_of("mix-deep", 500, deep, iters=2_250_000) + rows_of("mix-deep", 20, 0, iters=2_250_000))
    out = ab_report.report(rows)
    assert out["kept"] == kept, out["verdict"]
    finals = [v for v in out["verdict"] if v.startswith("final")]
    assert len(finals) == 2 and all("not significant" in v for v in finals)
    assert out["stats"]["net"]["games"] == 520 and out["looks"]["final"]["net"]["diff"] == round((net - mix) / 500, 4)


def test_significance_at_z_2():
    rows = shuffle_in(rows_of("net", 500, 285), rows_of("mix", 500, 250), rows_of("mix-deep", 500, 250, iters=2_250_000))
    out = ab_report.report(rows, {"stop_z": 99})             # no early stop, to see the final
    assert out["looks"]["final"]["net"]["z"] >= 2.0 and any(v.startswith("final") and "net -" in v and ": significant" in v for v in out["verdict"])
    assert out["kept"] == "net"


def test_validity_rating_gap_and_compute():
    rows = shuffle_in(rows_of("net", 10, 5, rating=1520), rows_of("mix", 10, 5, iters=400_000), rows_of("mix-deep", 10, 5, iters=1_500_000))
    out = ab_report.report(rows)
    w = " | ".join(out["warnings"])
    assert "INVALID: net's opponents averaged 1520" in w and "COMPUTE: mix searched 400,000" in w and "mix-deep searched 3.8 times" in w
    # today's mix: 538,000 iterations per decision (the 90 games on 024381 at 3 at a time); within 10% is fine
    assert ab_report.DEFAULTS["mix_iterations_ref"] == 538_000
    ok = ab_report.report(shuffle_in(rows_of("net", 10, 5, rating=1514), rows_of("mix", 10, 5, iters=500_000),
                                     rows_of("mix-deep", 10, 5, iters=2_500_000)))
    assert ok["warnings"] == [] and ok["deep_ratio"] == 5.0


def spread_ratings(rows, mean, sd):
    """Give rows opponent ratings with this mean and (sample) standard deviation: alternately mean - sd' and mean + sd'."""
    n = len(rows)
    half = sd * math.sqrt((n - 1) / n)
    for i, d in enumerate(rows):
        d["opponent_rating"] = mean + (half if i % 2 else -half)
    return rows


def test_the_rating_gap_is_read_on_the_looks_games_with_its_standard_error():
    # 250 per arm: net's opponents 20 points higher than mix's, SD 100 each: SE about 8.9, z about 2.2.
    net = spread_ratings(rows_of("net", 250, 125), 1520.0, 100.0)
    mix = spread_ratings(rows_of("mix", 250, 125), 1500.0, 100.0)
    deep = spread_ratings(rows_of("mix-deep", 250, 125, iters=2_700_000), 1505.0, 100.0)
    # then 50 more per arm where net's opponents are level: the interim look must not read them
    late = [spread_ratings(rows_of(a, 50, 25, iters=(2_700_000 if a == "mix-deep" else 540_000)), 1400.0, 100.0)
            for a in ("net", "mix", "mix-deep")]
    out = ab_report.report(shuffle_in(net + late[0], mix + late[1], deep + late[2]))
    gaps = out["rating_gaps"]
    assert gaps["games"] == "the interim look's first 250 per arm"
    assert gaps["net"]["gap"] == 20.0 and gaps["net"]["se"] == pytest.approx(8.9, abs=0.1) and gaps["net"]["z"] == pytest.approx(2.24, abs=0.02)
    assert gaps["mix-deep"]["gap"] == 5.0
    inv = [w for w in out["warnings"] if w.startswith("INVALID")]
    assert len(inv) == 1 and "net's opponents averaged 1520, mix's 1500 (the interim look's first 250 per arm)" in inv[0]
    assert "SE 8.9, z +2.24" in inv[0] and "by chance" not in inv[0]
    text = ab_report.text(out, "ab3")
    assert "opponents' rating against mix's, on the interim look's first 250 per arm: net +20 (SE 8.9, z +2.24)" in text
    # a 16-point gap with a wide spread reads as chance
    small = ab_report.report(shuffle_in(spread_ratings(rows_of("net", 40, 20), 1516.0, 120.0),
                                        spread_ratings(rows_of("mix", 40, 20), 1500.0, 120.0),
                                        spread_ratings(rows_of("mix-deep", 40, 20, iters=2_700_000), 1500.0, 120.0)))
    inv = [w for w in small["warnings"] if w.startswith("INVALID")]
    assert len(inv) == 1 and "comes by chance" in inv[0] and "(all games so far)" in inv[0]


def test_the_paired_compute_check_flags_mix_losing_compute_beside_deep_searches():
    def with_pairs(rows, it_deep, n_deep, it_none, n_none):
        for d in rows:
            d.update(it_deep_beside=it_deep * n_deep, n_deep_beside=n_deep, it_no_deep=it_none * n_none, n_no_deep=n_none)
        return rows

    fine = ab_report.report(shuffle_in(rows_of("net", 20, 10), with_pairs(rows_of("mix", 20, 10, iters=540_000), 530_000, 12, 550_000, 18),
                                       rows_of("mix-deep", 20, 10, iters=2_700_000)))
    assert fine["paired_compute"]["ratio"] == round(530_000 / 550_000, 3) and fine["warnings"] == []
    starved = ab_report.report(shuffle_in(rows_of("net", 20, 10), with_pairs(rows_of("mix", 20, 10, iters=540_000), 450_000, 12, 560_000, 18),
                                          rows_of("mix-deep", 20, 10, iters=2_700_000)))
    w = [x for x in starved["warnings"] if "deep search beside it" in x]
    assert len(w) == 1 and "450,000" in w[0] and "560,000" in w[0] and "the deep arm takes compute from mix" in w[0]
    few = ab_report.report(shuffle_in(rows_of("net", 5, 2), with_pairs(rows_of("mix", 5, 2, iters=540_000), 450_000, 12, 560_000, 18),
                                      rows_of("mix-deep", 5, 2, iters=2_700_000)))
    assert not [x for x in few["warnings"] if "deep search beside it" in x]          # 60 decisions a side: too few to flag
    text = ab_report.text(starved, "ab3")
    assert "compute, paired: mix searched 450,000 iterations per decision with a deep search beside it" in text
    assert "turns (a diagnostic, unpaired)" in text


def test_report_counts_rejoined_games_per_arm():
    rows = rows_of("mix", 4, 2) + [dict(rows_of("mix", 1, 1)[0], rejoined=True)] + rows_of("net", 3, 1)
    out = ab_report.report(rows)
    assert out["stats"]["mix"]["rejoined"] == 1 and out["stats"]["net"]["rejoined"] == 0
    assert "picked up again after a lost connection" in ab_report.text(out, "ab3")


def test_report_counts_p1_p2_forfeits_timer_losses_and_resends(tmp_path):
    rows = rows_of("mix", 4, 2) + [dict(rows_of("net", 1, 1)[0], ending="forfeit", ending_by="them", throttle_resends=1),
                                   dict(rows_of("net", 1, 0)[0], ending="inactivity", ending_by="us", throttle_notices=2),
                                   dict(rows_of("net", 1, 0)[0], won=None, abandoned="x")]
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows + [{"experiment": "other", "arm": "net", "won": True}]) + "\nnot json\n")
    loaded = ab_report.load(str(path), "ab3")
    out = ab_report.report(loaded)
    net = out["stats"]["net"]
    assert (net["games"], net["wins"], net["no_result"]) == (2, 1, 1)
    assert (net["their_forfeits"], net["timer_losses"], net["throttle_resends"], net["throttle_notices"]) == (1, 1, 1, 2)
    assert out["stats"]["mix"]["roles"] == {"p2": {"games": 2, "wins": 0}, "p1": {"games": 2, "wins": 2}}
    text = ab_report.text(out, "ab3")
    assert "net lost 1 game(s) on time" in text and "VERDICT before the interim look" in text
    assert ab_report.main(["--experiment", "ab3", "--results", str(path)]) == 0


# =============================================================================================
# The observation of the real checkpoints
# =============================================================================================

def find_checkpoint(update):
    name = "learner_update_%06d.pt" % update
    env = os.environ.get("PSPPO_CKPT_%06d" % update)
    candidates = [env] if env else []
    home = os.path.expanduser("~")
    candidates += [os.path.join(home, "runs", "main", "checkpoints", name), os.path.join(home, "ps-ppo", "runs", "vm", "checkpoints", name),
                   r"C:\Users\you\ps-ppo\runs\vm\checkpoints\%s" % name]
    candidates += sorted(glob.glob(os.path.join(home, "dev", "*", "ckpt", name))) + sorted(glob.glob(os.path.join(home, "dev", "*", name)))
    return next((c for c in candidates if c and os.path.isfile(c)), None)


@pytest.mark.parametrize("update, r5", [(24381, True), (11850, False)])
def test_the_policy_builds_the_assembler_each_real_checkpoint_was_trained_with(update, r5):
    path = find_checkpoint(update)
    if path is None:
        pytest.skip("learner_update_%06d.pt is not here" % update)
    pytest.importorskip("torch")
    from hybrid import HybridConfig, PolicyNet
    from obs_abil import AbilityObservationAssembler
    from obs_assembler import ObservationAssembler

    pol = PolicyNet(HybridConfig(checkpoint=path, device="cpu"))
    if r5:
        assert type(pol.assembler) is AbilityObservationAssembler and pol.own_tera_type is True
        assert pol.obs_abilities and all(pol.obs_abilities.values())
    else:
        assert type(pol.assembler) is ObservationAssembler and pol.own_tera_type is False and not any((pol.obs_abilities or {}).values())
    vs = pol.value_settings
    assert vs["shaping"] in ("potential", "none", "faint") and 0.0 <= vs["scale"] <= 1.0
    if vs["from"] == "checkpoint":
        assert vs["gamma"] > 0.9


# =============================================================================================
# The launcher's line
# =============================================================================================

def launcher_argv(line):
    line = line.replace("%GAMES%", "30").replace("%PY%", "python")
    words = [w[1:-1] if w.startswith('"') and w.endswith('"') else w for w in re.findall(r'"[^"]*"|\S+', line)]
    return words[words.index("ladder.py") + 1:]


def test_the_launcher_line_parses_and_its_search_settings_can_be_played():
    with open(LAUNCHER, encoding="utf-8") as fh:
        text = fh.read()
    # The line that runs the A/B (not a remark, an echo or the scan for another ladder.py), and only one.
    runs = [l for l in text.splitlines() if l.startswith("%PY% ladder.py ")]
    assert len(runs) == 1, runs
    line = runs[0]
    args = ladder.parse_args(launcher_argv(line))
    assert args.arms == ["net", "mix", "mix-deep"] and ladder.experiment_tag(args) == "ab3-net-mix-deep-024381"
    assert (args.parallel, args.helpers, args.search_ms, args.deadline, args.predictions) == (3, 4, 500, 2.5, True)
    assert args.checkpoint.endswith("learner_update_024381.pt") and args.games == 30
    budget = ladder.deep_budget(args)
    assert budget.as_dict() == DEEP_BUDGET.as_dict()
    assert ladder.search_problems(args) == []
    assert "ab_report.py --experiment ab3-net-mix-deep-024381 --mix-iterations-ref 538000" in text
    assert "FreePhysicalMemory" in text and "GEQ 8" in text            # the memory check: warns, never refuses
    assert "cd /d C:\\Users\\you\\ps-ppo-predict" in text
    # another playing ladder.py (the app, the old Look Play) on this computer: it does not start
    scan = text.index("ladder\\.py' -and")
    assert (scan < text.index(":memcheck") < text.index("%PY% bench\\ab_deep_measure.py --quick")
            < text.index("%PY% ladder.py --games"))
    assert "-notmatch '--graph|--history'" in text and "exit /b 1" in text and "PSBot Lookahead app" in text
    assert "\r\n" in open(LAUNCHER, "rb").read().decode("utf-8") or os.name != "nt"


def box_run_argv():
    """The ladder.py command of run_box_ab.sh with its own settings and its defaults (the live run) put in."""
    with open(BOX_RUN, encoding="utf-8") as fh:
        text = fh.read()
    values = dict(re.findall(r"^([A-Z_]+)=([0-9.]+)$", text, re.M))                       # PARALLEL=6 ...
    values.update(dict(re.findall(r"^([A-Z_]+)=\$\{AB_[A-Z]+:-([^}$]*)\}$", text, re.M)))  # TAG=${AB_TAG:-...}
    values.update(DIR="/home/ubuntu/dev/ab-ladder/runs/ladder", STOP="/home/ubuntu/dev/ab-ladder/ladder.stop",
                  CKPT="/home/ubuntu/dev/ab-ladder-ckpt/learner_update_024381.pt",
                  RESULTS="/home/ubuntu/dev/ab-ladder/runs/ladder/results.jsonl",
                  EXITF="/home/ubuntu/dev/ab-ladder/runs/ladder/last_exit.txt")
    line = next(l for l in text.replace("\\\n", " ").splitlines() if l.lstrip().startswith('"$PY" ladder.py'))
    for name, value in sorted(values.items(), key=lambda kv: -len(kv[0])):
        line = line.replace('"$%s"' % name, value)
    words = line.split()
    argv = words[words.index("ladder.py") + 1:]
    assert not any("$" in w for w in argv), line
    return text, values, argv


def test_the_box_script_plays_the_preregistered_ab_with_the_calibrated_settings():
    text, values, argv = box_run_argv()
    args = ladder.parse_args(argv)
    assert args.arms == ["net", "mix", "mix-deep"] and ladder.experiment_tag(args) == "ab3-net-mix-deep-024381-box"
    assert (args.server, args.user, args.games, args.predictions, args.timer) == ("public", "YourShowdownName", 1500, True, True)
    assert (args.parallel, args.helpers, args.search_ms, args.deadline) == (6, 7, 500, 2.5)     # 6: Showdown's cap per account
    assert args.out == "/home/ubuntu/dev/ab-ladder/runs/ladder/results.jsonl" and args.stop_file == "/home/ubuntu/dev/ab-ladder/ladder.stop"
    assert args.checkpoint.endswith("learner_update_024381.pt") and args.device == "cuda"
    budget = ladder.deep_budget(args)
    assert budget.worlds_x == 2 and budget.workers >= 2 and 2.5 * args.search_ms <= budget.search_ms <= 3.0 * args.search_ms
    assert sp.minimum_deadline(budget.search_ms, budget.worlds_x, budget.workers) < budget.deadline <= 5.0 < budget.max_decision_s
    assert ladder.search_problems(args) == [] and not args.ask_password
    # The password: ladder.py's own prompt only.
    assert "unset PS_PASSWORD" in text and "PS_PASSWORD=" not in text and "--ask-password" not in text
    assert "REMINDER: the laptop must NOT play" in text
    # Keys typed before getpass's prompt are not echoed into tmux's scrollback; echo comes back after.
    assert text.index("stty -echo") < text.index('"$PY" ladder.py') < text.index("stty echo")
    assert '--experiment "$TAG" --mix-iterations-ref 538000' in text
    # Why it ended, in one line after it ends, and the window kept open to read it.
    assert args.exit_file == "/home/ubuntu/dev/ab-ladder/runs/ladder/last_exit.txt" and args.login_wait == 30.0
    assert text.index(': > "$EXITF"') < text.index('"$PY" ladder.py') < text.index('LINE=$(exit_line "$CODE")') < text.index('echo "$LINE"')
    assert "box_ladder.sh start again" in text and 'read -r -t 86400 -p "Press Enter to close this window. "' in text
    with open(BOX_HELPER, encoding="utf-8") as fh:
        helper = fh.read()
    assert 'ssh "${BOX_LADDER_TTY:--t}"' in helper and "tmux -u new -A -s $SESSION" in helper and "SESSION=ladder" in helper
    assert "run_box_ab.sh" in helper and "touch ~/$REMOTE/ladder.stop" in helper and "LOCAL=$HERE/runs/ladder-box" in helper
    assert "TAG=${AB_TAG:-ab3-net-mix-deep-024381-box}" in helper and "ab_report.py --results runs/ladder/results.jsonl" in helper
    assert 'bash "$VM" ip' in helper and "PS_PASSWORD" not in helper and "read -s" not in helper   # it never handles the password
    # The laptop's own bots (their lock cannot see the box): start refuses before it connects, status warns.
    start = helper[helper.index("  start)"):helper.index("  status)")]
    assert start.index("BOTS=$(laptop_bots)") < start.index("exit 1") < start.index("A=$(address)")
    status = helper[helper.index("  status)"):helper.index("  stop)")]
    assert status.index("BOTS=$(laptop_bots)") < status.index("A=$(address)")
    assert "'ladder\\.py'" in helper and "PSBot Lookahead*" in helper


def test_search_problems_name_a_deep_deadline_that_cannot_work():
    args = ladder.parse_args(["--arms", "net,mix,mix-deep", "--deep-deadline", "5.0"])
    assert any("the deep arm" in p for p in ladder.search_problems(args))
    args = ladder.parse_args(["--arms", "net,mix,mix-deep", "--deep-deadline", "12"])
    assert any("under 10 s" in p for p in ladder.search_problems(args))
    # a deadline at or past --deep-max-decision-s (9.0): a decision that runs to it sends the game back to today's budget
    args = ladder.parse_args(["--player", "mix-deep", "--deep-search-ms", "500", "--deep-worlds", "7", "--deep-deadline", "9.2"])
    assert any("is not under --deep-max-decision-s 9" in p for p in ladder.search_problems(args))
    args = ladder.parse_args(["--player", "mix-deep", "--deep-search-ms", "500", "--deep-worlds", "7", "--deep-deadline", "9.2",
                              "--deep-max-decision-s", "9.5"])
    assert not any("--deep-max-decision-s" in p for p in ladder.search_problems(args))
    args = ladder.parse_args(["--arms", "net,mix", "--helpers", "3", "--parallel", "3"])
    assert any("--helpers 3 with --parallel 3" in p for p in ladder.search_problems(args))
    assert ladder.experiment_tag(ladder.parse_args(["--arms", "net,mix"])) == "net,mix"
    with pytest.raises(SystemExit):
        ladder.parse_args(["--arms", "net,mix", "--experiment", " "])


def test_arm_counts_follow_the_experiment_tag():
    games = [{"arm": "net", "checkpoint": "c.pt", "experiment": "ab3"}, {"arm": "mix", "checkpoint": "c.pt", "experiment": "net,mix"},
             {"arm": "mix-deep", "checkpoint": "c.pt", "experiment": "ab3"}]
    assert ladder.arm_counts(games, "c.pt", experiment="ab3") == {"net": 1, "mix-deep": 1}
