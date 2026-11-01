"""
Tests for foulplay_bridge.py, foulplay_player.py and the Foul-Play-free parts of
bench/fp_bridge_server.py. None of them needs Foul Play or a Showdown server.

    python -m pytest tests/test_foulplay_bridge.py -q

What stands in for the real things:
  * a fake helper (FAKE_HELPER below): a small script that speaks the helper's JSON line protocol,
    remembers what it was fed, answers a decision from the last request it saw, and can be told to
    be slow, to hang, or to die. It is started by the real code path (FoulPlayBridge with
    command=...), so the queues, the threads, the restart and the rebuild are the real ones.
  * tests/fixtures/foulplay_bridge_trace.jsonl.gz: seven real battles as a poke-env player received
    them, with the answer the REAL helper (Foul Play 8f541f80, poke-engine 0.0.48) gave at every
    decision point. The action mapping is checked on those, with real poke-env battle objects.
    Among them: a Pokemon locked into Outrage, a Ditto that transforms, and Revival Blessing.

The real thing (Foul Play, poke-engine, a Showdown server, live games) was tested in WSL: the
commands and results are in bench/FOULPLAY_BRIDGE.md. When Foul Play is installed where the
bridge looks for it ($FOULPLAY_DIR), the last tests here also run the recorded battles through the
real helper.

Nothing here deletes anything. Temporary folders are made under $FOULPLAY_TEST_TMP (default: the
system temp folder) and left there. Every process is started by foulplay_bridge.py itself and
stopped by it, by PID.
"""

from __future__ import annotations

import asyncio
import gzip
import importlib.util
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import foulplay_bridge as fb  # noqa: E402
from foulplay_bridge import Advice, FoulPlayBridge, FoulPlayBridgeError, FoulPlayMixin, Option  # noqa: E402

FIXTURE = os.path.join(REPO, "tests", "fixtures", "foulplay_bridge_trace.jsonl.gz")

try:
    import numpy  # noqa: F401
    from obs_assembler import ObservationAssembler

    with open(os.path.join(REPO, "vocab.json"), "r", encoding="utf-8") as _fh:
        ASSEMBLER = ObservationAssembler(vocab=json.load(_fh))
except Exception:  # numpy is not installed where only the bridge is used
    ASSEMBLER = None

try:
    import poke_env  # noqa: F401

    HAVE_POKE_ENV = True
except Exception:
    HAVE_POKE_ENV = False


# ---------------------------------------------------------------------------------------------
# the fake helper
# ---------------------------------------------------------------------------------------------

FAKE_HELPER = r'''
import json, os, sys, time

if os.environ.get("FAKE_FATAL"):
    sys.stdout.write(json.dumps({"id": 0, "ok": False, "fatal": os.environ["FAKE_FATAL"]}) + "\n")
    sys.stdout.flush()
    sys.exit(2)
if os.environ.get("FAKE_EXIT_AT_START"):
    sys.exit(3)

battles = {}
gone = set()
counts = {"decisions": 0, "fed": 0, "expired": 0, "chose": 0, "beats": 0}


def clock():
    return time.monotonic() + float(os.environ.get("FAKE_CLOCK_OFFSET", "0"))


def say(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def options_for(request):
    if not request or request.get("wait"):
        return None
    out = []
    force = request.get("forceSwitch")
    if not force:
        active = request["active"][0]
        for i, move in enumerate(active["moves"]):
            if not move.get("disabled"):
                out.append("move %d" % (i + 1))
        if active.get("canTerastallize"):
            out += [c + " terastallize" for c in list(out)]
    if force or not request["active"][0].get("trapped"):
        for i, mon in enumerate(request["side"]["pokemon"]):
            if not mon.get("active") and not mon["condition"].endswith(" fnt"):
                out.append("switch %d" % (i + 1))
    return out


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    m = json.loads(line)
    op = m.get("op")
    key = m.get("key")
    if op == "ping":
        say({"id": m["id"], "ok": True, "version": 1, "pid": os.getpid(), "fake": True,
             "clock": clock(), "argv": sys.argv[1:]})
    elif op == "beat":
        counts["beats"] += 1
    elif op == "feed":
        if m["data"].startswith("|init|") and (key in gone or key in battles):
            battles.pop(key, None)       # the room was joined again: the battle starts over
            gone.discard(key)
        if key in gone:
            continue
        b = battles.setdefault(key, {"feeds": 0, "chose": [], "request": None, "user": m.get("user"), "room": m.get("room"), "seen": []})
        b["feeds"] += 1
        counts["fed"] += 1
        b["seen"].append(len(m["data"]))
        for ln in m["data"].split("\n"):
            if ln.startswith("|request|") and len(ln) > 9:
                b["request"] = json.loads(ln[9:])
    elif op == "chose":
        if key in battles:
            battles[key]["chose"].append(m.get("choice"))
            counts["chose"] += 1
    elif op == "end":
        battles.pop(key, None)
        gone.add(key)
    elif op == "decide":
        deadline = m.get("deadline")
        if deadline is not None and clock() + 2.5 * float(m.get("ms") or 100) / 1000.0 + 0.02 > float(deadline):
            counts["expired"] += 1       # as the real helper: nobody waits for it any more, so it is not searched
            say({"id": m["id"], "ok": False, "expired": True, "key": key, "error": "not searched: nobody waits for this decision any more"})
            continue
        if os.environ.get("FAKE_REFUSE"):
            say({"id": m["id"], "ok": False, "key": key, "error": "Foul Play's decision failed: " + os.environ["FAKE_REFUSE"]})
            continue
        began = time.monotonic()
        if "hang" in key:
            time.sleep(3600)
        if "poison" in key:
            os._exit(7)
        once = os.environ.get("FAKE_DIE_ONCE")
        if "die" in key and once and not os.path.exists(once):
            open(once, "w").close()
            os._exit(9)
        if "slow" in key:
            time.sleep(float(os.environ.get("FAKE_SLOW_S", "1.0")))
        b = battles.get(key)
        if b is None:
            say({"id": m["id"], "ok": False, "key": key, "error": "the battle is over" if key in gone else "unknown battle: nothing was fed for it"})
            continue
        request = b["request"] or {}
        if request.get("forceSwitch") and any(p.get("reviving") for p in request["side"]["pokemon"]):
            say({"id": m["id"], "ok": False, "key": key, "revive": True,
                 "error": "Foul Play has no answer for a Revival Blessing request (whom to revive)"})
            continue
        opts = options_for(b["request"])
        if not opts:
            say({"id": m["id"], "ok": False, "key": key, "error": "Foul Play has no move to make for request (a wait request)"})
            continue
        rqid = b["request"].get("rqid")
        if m.get("rqid") is not None and str(m["rqid"]) != str(rqid):
            say({"id": m["id"], "ok": False, "key": key, "error": "asked about request %s, Foul Play is at request %s" % (m["rqid"], rqid)})
            continue
        counts["decisions"] += 1
        total = sum(range(1, len(opts) + 1)) * 100
        rows = []
        for i, c in enumerate(opts):
            visits = 100 * (len(opts) - i)
            rows.append({"choice": c, "fp": "fake-" + c.replace(" ", "-"), "visits": visits, "total_score": visits * 0.5,
                         "score": 0.5, "policy": visits / total, "considered": i == 0})
        # A search limited by time: as many iterations as the time asked for allows on this "machine".
        iterations = int(float(os.environ["FAKE_ITERATIONS_PER_MS"]) * float(m.get("ms") or 100)) if os.environ.get("FAKE_ITERATIONS_PER_MS") else total
        say({"id": m["id"], "ok": True, "key": key, "choice": opts[0], "best": opts[0], "fp_choice": rows[0]["fp"],
             "fp_message": "/choose " + opts[0], "options": rows, "worlds": 2, "iterations": iterations, "iterations_per_world": [iterations // 2, iterations // 2],
             "search_ms": m.get("ms"), "ms": max(1.5, round((time.monotonic() - began) * 1000.0, 2)), "turn": 1, "rqid": rqid,
             "force_switch": bool(b["request"].get("forceSwitch")),
             "tera_dropped": False, "feeds": b["feeds"], "chose": list(b["chose"]), "pid": os.getpid(), "asked_ms": m.get("ms"),
             "clock": clock(), "asked": {k: m.get(k) for k in ("payoff_ms", "payoff", "iterations", "threads", "deadline")},
             "iterations_asked": m.get("iterations") or 0,
             # the opponent's side of the search: two worlds, one with a Pokemon nobody has seen
             "opponent": {"active": "garchomp", "revealed": ["rotomwash"], "options": [
                 {"name": "switch rotomwash", "fp": "switch rotomwash", "share": 0.2, "visits": 400, "score": 0.4, "worlds": 2, "seen": True},
                 {"name": "move earthquake", "fp": "earthquake", "share": 0.5, "visits": 1000, "score": 0.45, "worlds": 2},
                 {"name": "move earthquake terastallize", "fp": "earthquake-tera", "share": 0.2, "visits": 400, "score": 0.5, "worlds": 2},
                 {"name": "switch skarmory", "fp": "switch skarmory", "share": 0.1, "visits": 200, "score": 0.3, "worlds": 1, "seen": False}],
                 "worlds": [
                     {"chance": 0.5, "iterations": 1000, "active": {"species": "garchomp", "moves": ["earthquake", "outrage", "swordsdance", "stoneedge"],
                                                                    "item": "choicescarf", "ability": "roughskin", "tera_type": "steel", "hp": 300, "max_hp": 300},
                      "reserve": ["rotomwash", "skarmory"], "options": [{"name": "move earthquake", "fp": "earthquake", "visits": 600, "share": 0.6, "score": 0.45}]},
                     {"chance": 0.5, "iterations": 1000, "active": {"species": "garchomp", "moves": ["earthquake", "firefang", "spikes", "stealthrock"],
                                                                    "item": "rockyhelmet", "ability": "roughskin", "tera_type": "ground", "hp": 300, "max_hp": 300},
                      "reserve": ["rotomwash", "toxapex"], "options": [{"name": "move earthquake", "fp": "earthquake", "visits": 400, "share": 0.4, "score": 0.45}]}]},
             # only when asked: a table per world. World 2 has one column less and a pruned cell.
             **({"payoff": {"mode": m.get("payoff") or "engine", "budget_ms": m["payoff_ms"], "ms": 1.0, "unit": "engine evaluation, larger is better for us",
                            "worlds": [
                                {"chance": 0.5, "depth": 2, "choices": opts, "ours": ["fake-" + c for c in opts], "theirs": ["move earthquake", "switch rotomwash"],
                                 "matrix": [[10.0 * i, 10.0 * i + 1] for i in range(len(opts))], "cells": 2 * len(opts), "known": 2 * len(opts), "ms": 0.5},
                                {"chance": 0.5, "depth": 3, "choices": opts, "ours": ["fake-" + c for c in opts], "theirs": ["move earthquake"],
                                 "matrix": [[None if i == 1 else 10.0 * i + 4] for i in range(len(opts))], "cells": len(opts), "known": len(opts) - 1, "ms": 0.5}]}}
                if m.get("payoff_ms") else {})})
    elif op == "sync":
        say({"id": m["id"], "ok": True})
    elif op == "stats":
        say({"id": m["id"], "ok": True, "battles": len(battles), "decisions": counts["decisions"], "fed": counts["fed"],
             "expired": counts["expired"], "chose": counts["chose"], "beats": counts["beats"],
             "remembered_gone": len(gone), "pid": os.getpid(), "rss_mb": 1.0, "failed_battles": 0})
    elif op == "exit":
        break
'''


def new_dir(prefix: str) -> str:
    base = os.environ.get("FOULPLAY_TEST_TMP") or tempfile.gettempdir()
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix=prefix, dir=base)


def write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


@pytest.fixture(scope="module")
def fake_helper() -> str:
    path = os.path.join(new_dir("fpbridge_fake_"), "fake_helper.py")
    write(path, FAKE_HELPER)
    return path


def make_bridge(fake: str, helpers: int = 1, env=None, **kwargs) -> FoulPlayBridge:
    kwargs.setdefault("timeout", 20.0)
    return FoulPlayBridge(helpers=helpers, command=[sys.executable, "-u", fake], env=env, **kwargs)


def pid_alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid, "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # a zombie is not alive
        with open("/proc/%d/stat" % pid) as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


def kill_hard(pid: int) -> None:
    os.kill(pid, signal.SIGTERM if os.name == "nt" else signal.SIGKILL)   # TerminateProcess on Windows


def wait_until(check, timeout: float = 15.0, step: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(step)
    return bool(check())


def request_json(rqid: int = 1, moves=("tackle", "growl", "ember", "surf"), disabled=(), can_tera: bool = True,
                 force: bool = False, wait: bool = False, trapped: bool = False, fainted=(), team: int = 6) -> dict:
    side = {"name": "u", "id": "p1", "pokemon": [
        {"ident": "p1: Mon%d" % i, "details": "Mon%d, L80" % i, "condition": "0 fnt" if i in fainted else "100/100",
         "active": i == 0, "moves": list(moves)} for i in range(team)]}
    out = {"rqid": rqid, "side": side}
    if wait:
        out["wait"] = True
        return out
    if force:
        out["forceSwitch"] = [True]
        return out
    active = {"moves": [{"move": m.title(), "id": m, "pp": 10, "maxpp": 10, "disabled": m in disabled} for m in moves]}
    if can_tera:
        active["canTerastallize"] = "Fire"
    if trapped:
        active["trapped"] = True
    out["active"] = [active]
    return out


def feed_request(bridge: FoulPlayBridge, key: str, rqid: int = 1, **kwargs) -> None:
    user, room = key.split("|", 1)
    bridge.feed(key, user, room, "|request|" + json.dumps(request_json(rqid, **kwargs)))


# ---------------------------------------------------------------------------------------------
# finding the installation
# ---------------------------------------------------------------------------------------------

def fake_checkout(importable: bool = True, sets: bool = True, venv: bool = True) -> str:
    """A folder that looks like a Foul Play checkout to locate(). Its "virtualenv python" is this python."""
    root = new_dir("fpbridge_checkout_")
    write(os.path.join(root, "run.py"), "")
    write(os.path.join(root, "fp", "__init__.py"), "")
    if importable:
        write(os.path.join(root, "poke_engine.py"), "")
        write(os.path.join(root, "fp", "run_battle.py"), "")
        write(os.path.join(root, "fp", "modes", "__init__.py"), "")
        write(os.path.join(root, "fp", "modes", "base.py"), "")
        write(os.path.join(root, "fp", "search", "__init__.py"), "")
        write(os.path.join(root, "fp", "search", "main.py"), "")
    if sets:
        write(os.path.join(root, "fp", "data", "pkmn_sets_cache", fb.BATTLE_FORMAT + ".json"), "{" + " " * 20000 + "}")
    write(os.path.join(root, ".git", "HEAD"), fb.PINNED_COMMIT + "\n")
    return root


def test_available_reports_a_missing_installation(monkeypatch):
    missing = os.path.join(new_dir("fpbridge_none_"), "nothing-here")
    monkeypatch.setenv("FOULPLAY_DIR", missing)
    monkeypatch.delenv("FOULPLAY_PYTHON", raising=False)
    ok, why = fb.available()
    assert ok is False and "not installed" in why and missing in why


def test_available_reports_what_is_wrong_with_a_checkout(monkeypatch):
    monkeypatch.delenv("FOULPLAY_PYTHON", raising=False)
    not_a_checkout = new_dir("fpbridge_empty_")
    assert "not a Foul Play checkout" in fb.available(not_a_checkout)[1]
    no_venv = fake_checkout()
    ok, why = fb.available(no_venv)
    assert ok is False and "no virtualenv" in why
    monkeypatch.setenv("FOULPLAY_PYTHON", sys.executable)
    no_sets = fake_checkout(sets=False)
    ok, why = fb.available(no_sets)
    assert ok is False and "set data" in why
    broken = fake_checkout(importable=False)
    ok, why = fb.available(broken)
    assert ok is False and "cannot import" in why


def test_available_accepts_a_usable_checkout_and_reads_the_commit(monkeypatch):
    monkeypatch.setenv("FOULPLAY_PYTHON", sys.executable)
    good = fake_checkout()
    ok, text = fb.available(good)
    assert ok is True, text
    assert fb.PINNED_COMMIT in text and good in text and "not the pinned" not in text
    write(os.path.join(good, ".git", "HEAD"), "0123456789abcdef0123456789abcdef01234567\n")
    fb._LOCATE_CACHE.clear()
    assert "not the pinned commit" in fb.available(good)[1]


def test_available_never_raises(monkeypatch):
    monkeypatch.setattr(fb, "locate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    ok, why = fb.available()
    assert ok is False and "boom" in why


def test_a_bridge_without_foul_play_says_so_instead_of_starting(monkeypatch):
    monkeypatch.setenv("FOULPLAY_DIR", os.path.join(new_dir("fpbridge_none_"), "nothing-here"))
    monkeypatch.delenv("FOULPLAY_PYTHON", raising=False)
    bridge = FoulPlayBridge(helpers=1)
    try:
        with pytest.raises(FoulPlayBridgeError, match="not installed"):
            bridge.start()
        bridge.feed("u|battle-x-1", "u", "battle-x-1", "|init|battle")      # must not raise
        with pytest.raises(FoulPlayBridgeError, match="not installed"):
            bridge.advice_sync("u|battle-x-1", timeout=5)
    finally:
        bridge.close()


# ---------------------------------------------------------------------------------------------
# protocol and bookkeeping, against the fake helper
# ---------------------------------------------------------------------------------------------

def test_feed_decide_note_forget(fake_helper):
    bridge = make_bridge(fake_helper, search_ms=77)
    try:
        key = "alice|battle-gen9randombattle-1"
        bridge.feed(key, "alice", "battle-gen9randombattle-1", "|init|battle\n|title|alice vs. bob")
        feed_request(bridge, key, rqid=5)
        advice = bridge.advice_sync(key, rqid=5)
        assert isinstance(advice, Advice)
        assert advice.choice == "move 1" and advice.best == "move 1"
        assert advice.fp_message == "/choose move 1" and advice.worlds == 2 and advice.rqid == 5
        assert advice.raw["feeds"] == 2 and advice.raw["asked_ms"] == 77     # the bridge's own search time
        assert [o.choice for o in advice.options][:2] == ["move 1", "move 2"]
        assert abs(sum(o.policy for o in advice.options) - 1.0) < 1e-9
        assert advice.wall_ms > 0 and advice.helper == 0
        assert bridge.advice_sync(key, search_ms=20).raw["asked_ms"] == 20   # a search time for one call
        bridge.note(key, "switch 3")
        assert bridge.advice_sync(key).raw["chose"] == ["switch 3"]
        with pytest.raises(FoulPlayBridgeError, match="asked about request 4"):
            bridge.advice_sync(key, rqid=4)                                  # a stale question is refused, not answered
        assert bridge.stats()["battles"] == 1 and bridge.stats()["stored_bytes"] > 0
        bridge.forget(key)
        assert bridge.sync()
        assert bridge.helper_stats()[0]["battles"] == 0
        stats = bridge.stats()
        assert stats["battles"] == 0 and stats["stored_bytes"] == 0 and stats["remembered_gone"] == 1
        with pytest.raises(FoulPlayBridgeError, match="it has ended"):
            bridge.advice_sync(key)
    finally:
        bridge.close()


def test_wait_requests_and_unknown_battles_are_errors_not_moves(fake_helper):
    bridge = make_bridge(fake_helper)
    try:
        key = "alice|battle-gen9randombattle-2"
        feed_request(bridge, key, rqid=1, wait=True)
        with pytest.raises(FoulPlayBridgeError, match="no move to make"):
            bridge.advice_sync(key)
        with pytest.raises(FoulPlayBridgeError, match="no lines were fed"):
            bridge.advice_sync("alice|battle-gen9randombattle-999")
        assert bridge.counters["decision_errors"] == 2 and bridge.counters["decisions"] == 0
    finally:
        bridge.close()


def test_everything_fed_before_a_question_is_seen_before_it(fake_helper):
    bridge = make_bridge(fake_helper)
    try:
        key = "alice|battle-gen9randombattle-3"
        for i in range(400):
            bridge.feed(key, "alice", "battle-gen9randombattle-3", "|-damage|p2a: X|%d/100" % i + "x" * 500)
        feed_request(bridge, key, rqid=9)
        future = bridge.advice_future(key)            # queued behind the 401 messages
        bridge.feed(key, "alice", "battle-gen9randombattle-3", "|c|someone|hello")   # and this behind the question
        assert future.result(timeout=20).raw["feeds"] == 401
        assert bridge.advice_sync(key).raw["feeds"] == 402
    finally:
        bridge.close()


def test_battles_are_pinned_to_the_helper_with_the_fewest(fake_helper):
    bridge = make_bridge(fake_helper, helpers=3)
    try:
        keys = ["u|battle-gen9randombattle-%d" % i for i in range(7)]
        for key in keys:
            feed_request(bridge, key)
        assert [h["battles"] for h in bridge.stats()["helpers"]] == [3, 2, 2]
        owners = {key: bridge.advice_sync(key).helper for key in keys}
        assert [owners[k] for k in keys] == [0, 1, 2, 0, 1, 2, 0]
        pids = {bridge.advice_sync(key).raw["pid"] for key in keys}
        assert len(pids) == 3                                   # three processes really answered
        for key in keys:                                        # a battle stays where it is
            feed_request(bridge, key, rqid=2)
            assert bridge.advice_sync(key).helper == owners[key]
        for key in (keys[1], keys[4]):                          # both on helper 1
            bridge.forget(key)
        feed_request(bridge, "u|battle-gen9randombattle-100")
        feed_request(bridge, "u|battle-gen9randombattle-101")
        assert bridge.advice_sync("u|battle-gen9randombattle-100").helper == 1
        assert bridge.advice_sync("u|battle-gen9randombattle-101").helper == 1
        assert [h["battles"] for h in bridge.stats()["helpers"]] == [3, 2, 2]
        assert [s["battles"] for s in bridge.helper_stats()] == [3, 2, 2]
    finally:
        bridge.close()


def test_helpers_start_only_when_a_battle_needs_them(fake_helper):
    bridge = make_bridge(fake_helper, helpers=4)
    try:
        assert bridge.counters["helper_starts"] == 0
        feed_request(bridge, "u|battle-gen9randombattle-1")
        assert bridge.counters["helper_starts"] == 1
        bridge.start()
        assert bridge.counters["helper_starts"] == 4
        assert all(h["alive"] for h in bridge.stats()["helpers"])
    finally:
        bridge.close()


def test_late_lines_of_a_forgotten_battle_do_not_bring_it_back(fake_helper):
    bridge = make_bridge(fake_helper)
    try:
        key = "u|battle-gen9randombattle-7"
        feed_request(bridge, key)
        bridge.forget(key)
        bridge.feed(key, "u", "battle-gen9randombattle-7", "|deinit")
        bridge.note(key, "move 1")
        bridge.forget(key)
        assert bridge.sync()
        assert bridge.stats()["battles"] == 0 and bridge.counters["late_lines"] == 1
        assert bridge.helper_stats()[0]["battles"] == 0
    finally:
        bridge.close()


def test_a_room_that_is_joined_again_starts_over_instead_of_being_fed_twice(fake_helper):
    bridge = make_bridge(fake_helper, helpers=2)
    try:
        key, room = "u|battle-gen9randombattle-7", "battle-gen9randombattle-7"
        bridge.feed(key, "u", room, "|init|battle\n|title|u vs. v")
        feed_request(bridge, key, rqid=4)
        assert bridge.advice_sync(key).raw["feeds"] == 2
        # worker.py leaves and joins a stalled battle: the server sends the room from its start again
        bridge.feed(key, "u", room, "|init|battle\n|title|u vs. v\n|player|p1|u\n|turn|1\n|turn|2")
        assert bridge.counters["rejoined"] == 1 and bridge.stats()["battles"] == 1
        with pytest.raises(FoulPlayBridgeError, match="no move to make"):
            bridge.advice_sync(key)                      # the new battle has seen no request: an error, not a guess
        feed_request(bridge, key, rqid=5)
        assert bridge.advice_sync(key).raw["feeds"] == 2     # only what came after the second init
        bridge.forget(key)
        bridge.feed(key, "u", room, "|init|battle\n|title|u vs. v")   # and again after it was forgotten
        feed_request(bridge, key, rqid=6)
        assert bridge.advice_sync(key).rqid == 6 and bridge.counters["rejoined"] == 2
        assert bridge.sync() and sum(s["battles"] for s in bridge.helper_stats() if s) == 1
    finally:
        bridge.close()


def test_a_battle_that_ends_while_its_decision_is_waiting(fake_helper):
    bridge = make_bridge(fake_helper, env={"FAKE_SLOW_S": "0.8"})
    try:
        key, other = "u|battle-slow-1", "u|battle-gen9randombattle-2"
        feed_request(bridge, key, rqid=3)
        feed_request(bridge, other, rqid=9)
        waiting = bridge.advice_future(key)
        time.sleep(0.2)                                   # the helper is searching
        bridge.feed(key, "u", "battle-slow-1", "|win|v")  # the opponent forfeited
        bridge.forget(key)                                # and the player reported the battle finished
        assert not bridge.knows(key) and bridge.stats()["stored_bytes"] > 0      # only the other battle is stored
        assert waiting.result(timeout=10).rqid == 3       # the question was ahead of the end in the queue
        with pytest.raises(FoulPlayBridgeError, match="it has ended"):
            bridge.advice_sync(key)                       # asking again is an error at once, not a wait
        assert bridge.advice_sync(other).rqid == 9
        assert bridge.sync() and bridge.helper_stats()[0]["battles"] == 1
        assert bridge.stats()["helpers"][0]["waiting"] == 0
    finally:
        bridge.close()


def test_memory_follows_the_battles_in_progress_not_the_battles_played(fake_helper):
    bridge = make_bridge(fake_helper, helpers=2)
    try:
        for i in range(300):
            key = "u|battle-gen9randombattle-%d" % i
            bridge.feed(key, "u", key.split("|")[1], "|init|battle\n" + "x" * 2000)
            feed_request(bridge, key, rqid=i)
            assert bridge.advice_sync(key).rqid == i
            bridge.note(key, "move 1")
            bridge.forget(key)
        assert bridge.sync()
        stats = bridge.stats()
        assert stats["battles"] == 0 and stats["stored_bytes"] == 0
        assert stats["remembered_gone"] == 300 and stats["decisions"] == 300
        assert all(h["waiting"] == 0 and h["queued"] == 0 for h in stats["helpers"])
        # Each battle ended before the next began, so the second helper was never needed or started.
        first, second = bridge.helper_stats()
        assert second is None and bridge.counters["helper_starts"] == 1
        assert first["battles"] == 0 and first["decisions"] == 300
    finally:
        bridge.close()


def test_the_oldest_idle_battle_is_dropped_when_too_many_are_kept(fake_helper):
    bridge = make_bridge(fake_helper, max_battles=3)
    try:
        for i in range(5):
            feed_request(bridge, "u|battle-gen9randombattle-%d" % i)
            time.sleep(0.01)
        stats = bridge.stats()
        assert stats["battles"] == 3 and stats["evicted"] == 2
        assert not bridge.knows("u|battle-gen9randombattle-0") and bridge.knows("u|battle-gen9randombattle-4")
        assert bridge.sync() and bridge.helper_stats()[0]["battles"] == 3
    finally:
        bridge.close()


def test_a_battle_too_long_to_store_is_still_followed(fake_helper):
    bridge = make_bridge(fake_helper, max_battle_bytes=5000)
    try:
        key = "u|battle-gen9randombattle-1"
        for _ in range(10):
            bridge.feed(key, "u", "battle-gen9randombattle-1", "y" * 1000)
        feed_request(bridge, key, rqid=3)
        assert bridge.advice_sync(key).raw["feeds"] == 11       # the helper got every line
        assert bridge.counters["truncated"] == 1                # but the battle can no longer be rebuilt
    finally:
        bridge.close()


# ---------------------------------------------------------------------------------------------
# a helper that dies, hangs or never starts
# ---------------------------------------------------------------------------------------------

def test_a_killed_helper_is_replaced_and_its_battles_rebuilt(fake_helper):
    bridge = make_bridge(fake_helper, helpers=2)
    try:
        keys = ["u|battle-gen9randombattle-%d" % i for i in range(4)]
        for n, key in enumerate(keys):
            for i in range(5 + n):
                bridge.feed(key, "u", key.split("|")[1], "|turn|%d" % i)
            feed_request(bridge, key, rqid=1)
            bridge.note(key, "move %d" % (n + 1))
            feed_request(bridge, key, rqid=2)
        before = {key: bridge.advice_sync(key) for key in keys}
        victim = before[keys[0]].raw["pid"]
        assert before[keys[2]].raw["pid"] == victim and before[keys[1]].raw["pid"] != victim
        kill_hard(victim)
        assert wait_until(lambda: bridge.counters["helper_exits"] == 1)
        after = {key: bridge.advice_sync(key) for key in keys}
        for key in keys:
            assert after[key].raw["feeds"] == before[key].raw["feeds"]       # every line is there again
            assert after[key].raw["chose"] == before[key].raw["chose"]       # and what was played
            assert after[key].choice == before[key].choice
        assert after[keys[0]].raw["pid"] != victim and after[keys[2]].raw["pid"] == after[keys[0]].raw["pid"]
        assert after[keys[1]].raw["pid"] == before[keys[1]].raw["pid"]       # the other helper was not touched
        counters = bridge.counters
        assert counters["rebuilt_battles"] == 2 and counters["helper_starts"] == 3
        assert counters["refed_lines"] == (5 + 3) + (7 + 3)
        assert not pid_alive(victim)
        feed_request(bridge, keys[0], rqid=3)                                # and the battle goes on
        assert bridge.advice_sync(keys[0], rqid=3).raw["feeds"] == before[keys[0]].raw["feeds"] + 1
    finally:
        bridge.close()


def test_a_decision_that_was_waiting_when_the_helper_died_is_asked_again(fake_helper):
    bridge = make_bridge(fake_helper, env={"FAKE_SLOW_S": "1.5"})
    try:
        slow, other = "u|battle-slow-1", "u|battle-gen9randombattle-2"
        feed_request(bridge, slow, rqid=4)
        feed_request(bridge, other, rqid=8)
        assert bridge.advice_sync(other).rqid == 8
        pid = bridge.stats()["helpers"][0]["pid"]
        waiting_slow = bridge.advice_future(slow)
        waiting_other = bridge.advice_future(other)      # queued behind the slow one
        time.sleep(0.3)
        kill_hard(pid)
        a, b = waiting_slow.result(timeout=20), waiting_other.result(timeout=20)
        assert a.rqid == 4 and b.rqid == 8
        assert a.raw["pid"] != pid and b.raw["pid"] == a.raw["pid"]
        assert bridge.counters["asked_again"] == 2 and bridge.counters["helper_exits"] == 1
    finally:
        bridge.close()


def test_a_helper_that_dies_once_on_a_decision_costs_nothing(fake_helper):
    marker = os.path.join(new_dir("fpbridge_once_"), "died")
    bridge = make_bridge(fake_helper, env={"FAKE_DIE_ONCE": marker})
    try:
        key = "u|battle-die-1"
        feed_request(bridge, key, rqid=6)
        advice = bridge.advice_sync(key)                 # the helper dies on it, the next one answers
        assert advice.rqid == 6 and os.path.exists(marker)
        assert bridge.counters["helper_exits"] == 1 and bridge.counters["asked_again"] == 1
    finally:
        bridge.close()


def test_a_decision_that_kills_every_helper_fails_alone(fake_helper):
    bridge = make_bridge(fake_helper)
    try:
        bad, good = "u|battle-poison-1", "u|battle-gen9randombattle-2"
        feed_request(bridge, bad)
        feed_request(bridge, good, rqid=3)
        with pytest.raises(FoulPlayBridgeError, match="stopped while answering"):
            bridge.advice_sync(bad)                      # tried twice, on two helpers
        assert bridge.counters["helper_exits"] == 2
        assert bridge.advice_sync(good).rqid == 3        # the other battle was rebuilt twice and is fine
        assert bridge.advice_sync(good).raw["feeds"] == 1
    finally:
        bridge.close()


def test_a_helper_that_cannot_start_is_left_down_and_says_why(fake_helper):
    bridge = make_bridge(fake_helper, env={"FAKE_FATAL": "could not load Foul Play: no module named poke_engine"},
                         max_restarts=2, cooldown=1.0)
    try:
        key = "u|battle-gen9randombattle-1"
        feed_request(bridge, key)
        with pytest.raises(FoulPlayBridgeError, match="no module named poke_engine"):
            bridge.advice_sync(key)
        assert wait_until(lambda: bridge.stats()["helpers"][0]["down"])
        assert bridge.counters["helper_starts"] == 3     # the first and two replacements
        started = bridge.counters["helper_starts"]
        feed_request(bridge, key, rqid=2)                # lines are still taken while it is down
        with pytest.raises(FoulPlayBridgeError, match="is down for"):
            bridge.advice_sync(key)
        assert bridge.counters["helper_starts"] == started
        with pytest.raises(FoulPlayBridgeError, match="did not start|is down"):
            bridge.start()
    finally:
        bridge.close()


def test_a_helper_that_was_down_comes_back_with_its_battles(fake_helper):
    bridge = make_bridge(fake_helper, max_restarts=0, cooldown=0.5)
    try:
        key = "u|battle-gen9randombattle-1"
        feed_request(bridge, key, rqid=1)
        pid = bridge.advice_sync(key).raw["pid"]
        kill_hard(pid)                                   # with max_restarts=0 the first death takes the slot down
        assert wait_until(lambda: bridge.stats()["helpers"][0]["down"])
        feed_request(bridge, key, rqid=2)                # fed while down: only stored
        with pytest.raises(FoulPlayBridgeError, match="is down"):
            bridge.advice_sync(key)
        time.sleep(0.6)
        advice = bridge.advice_sync(key)                 # the cooldown is over: started again, rebuilt
        assert advice.rqid == 2 and advice.raw["feeds"] == 2 and advice.raw["pid"] != pid
    finally:
        bridge.close()


def test_with_fail_when_down_a_slot_left_down_is_told_at_once_and_comes_back_by_itself(fake_helper):
    """ladder.py: every helper gone (WSL restarting) costs no wait, and the watchdog brings the slot back unasked."""
    bridge = make_bridge(fake_helper, max_restarts=0, cooldown=0.5, background_spawn=True, fail_when_down=True, timeout=2.0)
    try:
        bridge.start(timeout=20)
        key = "u|battle-gen9randombattle-1"
        feed_request(bridge, key, rqid=1)
        pid = bridge.advice_sync(key).raw["pid"]
        assert bridge.helpers_up() == 1
        kill_hard(pid)                                   # with max_restarts=0 the first death takes the slot down
        assert wait_until(lambda: bridge.stats()["helpers"][0]["down"])
        assert bridge.helpers_up() == 0
        feed_request(bridge, key, rqid=2)                # stored while down
        t0 = time.monotonic()
        with pytest.raises(FoulPlayBridgeError) as info:
            bridge.advice_sync(key)
        assert info.value.down and time.monotonic() - t0 < 0.5
        # Nothing asks for the slot: the watchdog starts it again once the cooldown is over.
        assert wait_until(lambda: bridge.helpers_up() == 1, timeout=10)
        assert bridge.counters["slot_restarts"] >= 1
        advice = bridge.advice_sync(key)
        assert advice.rqid == 2 and advice.raw["feeds"] == 2 and advice.raw["pid"] != pid
    finally:
        bridge.close()


@pytest.mark.skipif(os.name == "nt", reason="SIGSTOP")
def test_with_fail_when_down_a_frozen_helper_is_told_at_once_after_timeout(fake_helper):
    """ladder.py: a frozen helper (WSL hung) costs at most `timeout` of waiting, then the questions fail at once."""
    bridge = make_bridge(fake_helper, background_spawn=True, fail_when_down=True, timeout=1.0, hang_timeout=4.0)
    try:
        bridge.start(timeout=20)
        key = "u|battle-gen9randombattle-1"
        feed_request(bridge, key, rqid=1)
        pid = bridge.advice_sync(key).raw["pid"]
        os.kill(pid, signal.SIGSTOP)
        feed_request(bridge, key, rqid=2)
        first = bridge.advice_future(key, ttl=1.0)       # waits in the frozen helper: it was up when asked
        time.sleep(0.3)
        assert not first.done() and bridge.helpers_up() == 1
        assert wait_until(lambda: bridge.helpers_up() == 0, timeout=3)   # nothing answered for `timeout` with work waiting
        t0 = time.monotonic()
        with pytest.raises(FoulPlayBridgeError) as info:
            bridge.advice_sync(key, rqid=2)
        assert info.value.down and time.monotonic() - t0 < 0.5
        # The watchdog stops it after hang_timeout and starts another, which has the battle.
        assert wait_until(lambda: bridge.helpers_up() == 1, timeout=15)
        assert bridge.counters["hangs"] >= 1
        advice = bridge.advice_sync(key, rqid=2)
        assert advice.rqid == 2 and advice.raw["pid"] != pid
    finally:
        try:
            os.kill(pid, signal.SIGCONT)
        except OSError:
            pass
        bridge.close()


def test_advice_times_out_without_stopping_the_helper(fake_helper):
    # search_ms=20: a search of 40 ms fits into the 0.2 s the caller waits, so the helper starts it
    # (a question whose search could not end in time is refused at once: see the tests further down).
    bridge = make_bridge(fake_helper, search_ms=20, env={"FAKE_SLOW_S": "1.0"})
    try:
        slow, fast = "u|battle-slow-1", "u|battle-gen9randombattle-2"
        feed_request(bridge, slow)
        feed_request(bridge, fast, rqid=2)

        async def scenario():
            t0 = time.monotonic()
            with pytest.raises(FoulPlayBridgeError, match="did not answer"):
                await bridge.advice(slow, timeout=0.2)
            waited = time.monotonic() - t0
            later = await bridge.advice(fast, timeout=10)      # behind the slow one in the same helper
            return waited, later

        waited, later = asyncio.run(scenario())
        assert waited < 0.9 and later.rqid == 2
        assert bridge.counters["timeouts"] == 1 and bridge.counters["helper_exits"] == 0
        assert bridge.stats()["helpers"][0]["waiting"] == 0
    finally:
        bridge.close()


def test_the_event_loop_keeps_running_while_the_helper_searches(fake_helper):
    bridge = make_bridge(fake_helper, env={"FAKE_SLOW_S": "0.6"})
    try:
        key = "u|battle-slow-1"
        feed_request(bridge, key)

        async def scenario():
            ticks = 0

            async def ticker():
                nonlocal ticks
                while True:
                    await asyncio.sleep(0.01)
                    ticks += 1

            task = asyncio.ensure_future(ticker())
            t0 = time.monotonic()
            advice = await bridge.advice(key)
            task.cancel()
            return ticks, time.monotonic() - t0, advice

        ticks, seconds, advice = asyncio.run(scenario())
        assert advice.choice == "move 1" and seconds >= 0.5
        assert ticks >= 20                               # the loop ran the whole time
    finally:
        bridge.close()


def test_a_hung_helper_is_stopped_and_replaced(fake_helper):
    bridge = make_bridge(fake_helper, hang_timeout=1.0)
    try:
        stuck, fine = "u|battle-hang-1", "u|battle-gen9randombattle-2"
        feed_request(bridge, stuck)
        feed_request(bridge, fine, rqid=5)
        first_pid = bridge.advice_sync(fine).raw["pid"]
        t0 = time.monotonic()
        with pytest.raises(FoulPlayBridgeError, match="stopped while answering|no answer"):
            bridge.advice_sync(stuck, timeout=30)        # hangs, is asked again in a new helper, hangs again
        assert 1.5 < time.monotonic() - t0 < 15
        assert bridge.counters["hangs"] == 2 and bridge.counters["helper_exits"] == 2
        again = bridge.advice_sync(fine)
        assert again.rqid == 5 and again.raw["pid"] != first_pid
        assert not pid_alive(first_pid)
    finally:
        bridge.close()


def test_a_long_queue_is_not_a_hang(fake_helper):
    """Six slow decisions wait far longer than hang_timeout in one helper that keeps answering: it must be left alone."""
    bridge = make_bridge(fake_helper, hang_timeout=1.0, env={"FAKE_SLOW_S": "0.5"})
    try:
        keys = ["u|battle-slow-%d" % i for i in range(6)]
        for i, key in enumerate(keys):
            feed_request(bridge, key, rqid=i)
        futures = [bridge.advice_future(key) for key in keys]
        t0 = time.monotonic()
        assert [f.result(timeout=30).rqid for f in futures] == list(range(6))
        assert time.monotonic() - t0 > 2.5                       # the last one waited about 3 s
        assert bridge.counters["hangs"] == 0 and bridge.counters["helper_exits"] == 0
        assert len({f.result().raw["pid"] for f in futures}) == 1
    finally:
        bridge.close()


# ---------------------------------------------------------------------------------------------
# decisions nobody waits for any more, and helpers that are behind
# ---------------------------------------------------------------------------------------------

def outcomes_of(futures, timeout: float = 30.0):
    """The rqid of every answer, or "expired" for a question the helper did not search."""
    out = []
    for future in futures:
        try:
            out.append(future.result(timeout=timeout).rqid)
        except FoulPlayBridgeError as exc:
            assert exc.expired and "nobody waits" in str(exc), exc
            out.append("expired")
    return out


def test_a_decision_nobody_waits_for_any_more_is_not_searched(fake_helper):
    """
    Sixteen players ask a helper that needs 0.2 s a decision, and each waits 1.5 s. A caller that
    gives up tells nobody, so all sixteen used to be searched (3.2 s), half of them for nobody,
    and with players that keep asking nothing was ever answered in time again. The helper now
    knows until when each answer is of use and does not start a search that would end after it.
    """
    bridge = make_bridge(fake_helper, search_ms=20, env={"FAKE_SLOW_S": "0.2"})
    try:
        keys = ["u|battle-slow-%d" % i for i in range(16)]
        for i, key in enumerate(keys):
            feed_request(bridge, key, rqid=i)
        bridge.start()                                      # the helper has shown its clock
        assert bridge.stats()["helpers"][0]["deadlines"] is True
        t0 = time.monotonic()
        outcomes = outcomes_of([bridge.advice_future(key, ttl=1.5) for key in keys])
        took = time.monotonic() - t0
        assert outcomes[:4] == [0, 1, 2, 3]                 # answered while their callers waited
        assert outcomes[9:] == ["expired"] * 7              # their turn came after 1.6 s at the earliest
        expired = outcomes.count("expired")
        assert bridge.helper_stats()[0]["decisions"] == 16 - expired     # and no search was run for them
        assert bridge.helper_stats()[0]["expired"] == expired == bridge.counters["expired"]
        assert took < 2.6                                   # 3.2 s when all sixteen are searched
        assert bridge.stats()["helpers"][0]["waiting"] == 0
    finally:
        bridge.close()


def test_a_caller_that_cannot_wait_for_a_search_is_told_at_once(fake_helper):
    """Waiting 150 ms for a decision that takes 270 ms: the answer is "no" now, not a search for nobody later."""
    bridge = make_bridge(fake_helper, search_ms=100, env={"FAKE_SLOW_S": "0.4"})
    try:
        key = "u|battle-slow-1"
        feed_request(bridge, key, rqid=3)
        bridge.start()
        t0 = time.monotonic()
        with pytest.raises(FoulPlayBridgeError, match="nobody waits") as caught:
            bridge.advice_sync(key, timeout=0.15)
        assert caught.value.expired is True
        assert bridge.helper_stats()[0]["decisions"] == 0 and time.monotonic() - t0 < 0.4   # nothing was searched
        assert bridge.counters["expired"] == 1 and bridge.counters["timeouts"] == 0
        assert bridge.advice_sync(key, timeout=10).rqid == 3      # with time to wait it is answered as before
        assert bridge.helper_stats()[0]["decisions"] == 1
    finally:
        bridge.close()


def test_a_helper_asked_more_than_it_can_answer_keeps_answering_in_time(fake_helper):
    """
    The overload of the review, small: six players on a helper that needs 0.2 s a decision. Each
    asks, waits 0.6 s at most, moves on and asks again, for 3 s. Six questions take the helper
    1.2 s, so without deadlines every answer after the first three came too late, each still cost
    a search, and the questions piled up without end. With them the helper answers as many as it
    can, about five a second, and every one of those in time.
    """
    import threading

    bridge = make_bridge(fake_helper, search_ms=80, env={"FAKE_SLOW_S": "0.2"})   # 80 ms: the fake's search is as long as a real one
    try:
        keys = ["u|battle-slow-%d" % i for i in range(6)]
        for i, key in enumerate(keys):
            feed_request(bridge, key, rqid=i)
        bridge.start()
        counts = {"answered": 0, "expired": 0, "gave_up": 0}
        lock = threading.Lock()
        stop_at = time.monotonic() + 3.0

        def player(key):
            while time.monotonic() < stop_at:
                try:
                    bridge.advice_sync(key, timeout=0.6)
                    what = "answered"
                except FoulPlayBridgeError as exc:
                    what = "expired" if exc.expired else "gave_up"
                with lock:
                    counts[what] += 1

        threads = [threading.Thread(target=player, args=(key,), daemon=True) for key in keys]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert bridge.sync(timeout=30)
        stats = bridge.stats()
        searched = bridge.helper_stats()[0]["decisions"]
        assert counts["answered"] >= 9, counts                      # 3 without the fix, 15 at the helper's full pace
        assert searched <= counts["answered"] + 3, (searched, counts)   # next to no search for nobody
        assert counts["expired"] >= 6 and stats["expired"] >= counts["expired"]
        assert stats["helpers"][0]["waiting"] == 0 and stats["helpers"][0]["queued"] == 0    # nothing piles up
        assert 150 < stats["helpers"][0]["decide_ms"] < 500         # what a decision takes there, for whoever watches
    finally:
        bridge.close()


def test_a_helper_that_reads_another_clock_is_sent_no_deadlines(fake_helper):
    """Deadlines are times on a clock both processes must read alike. If they do not, none is sent."""
    bridge = make_bridge(fake_helper, search_ms=20, env={"FAKE_SLOW_S": "0.2", "FAKE_CLOCK_OFFSET": "100000"})
    try:
        keys = ["u|battle-slow-%d" % i for i in range(4)]
        for i, key in enumerate(keys):
            feed_request(bridge, key, rqid=i)
        bridge.start()
        assert bridge.stats()["helpers"][0]["deadlines"] is False
        # No deadline, so every question is searched, as before: late, but not all refused as "expired".
        assert outcomes_of([bridge.advice_future(key, ttl=0.3) for key in keys]) == [0, 1, 2, 3]
        assert bridge.counters["expired"] == 0 and bridge.helper_stats()[0]["expired"] == 0
    finally:
        bridge.close()


def test_a_decision_asked_again_after_a_helper_died_keeps_its_deadline(fake_helper):
    bridge = make_bridge(fake_helper, search_ms=20, env={"FAKE_SLOW_S": "0.5"})
    try:
        key = "u|battle-slow-1"
        feed_request(bridge, key, rqid=4)
        bridge.start()
        pid = bridge.stats()["helpers"][0]["pid"]
        patient = bridge.advice_future(key, ttl=30.0)
        hurried = bridge.advice_future(key, ttl=0.3)         # behind the patient one, which takes 0.5 s
        time.sleep(0.25)
        kill_hard(pid)
        assert patient.result(timeout=20).rqid == 4          # asked again in the new helper, and answered
        with pytest.raises(FoulPlayBridgeError) as caught:
            hurried.result(timeout=20)                       # asked again too, with the deadline it had: over by then
        assert caught.value.expired and bridge.counters["asked_again"] == 2
        assert bridge.helper_stats()[0]["decisions"] == 1 and bridge.counters["expired"] == 1
    finally:
        bridge.close()


# ---------------------------------------------------------------------------------------------
# a battle whose room is gone
# ---------------------------------------------------------------------------------------------

def test_a_battle_whose_room_is_gone_is_forgotten(fake_helper):
    """
    The training server closes every battle room after five minutes, and worker.py gives up
    battles of its own. The player then gets "|deinit" and never a result, so nobody reports the
    battle as finished. Its lines here and its state in the helper used to stay for good.
    """
    bridge = make_bridge(fake_helper, helpers=2)
    try:
        key, room = "u|battle-gen9randombattle-7", "battle-gen9randombattle-7"
        other = "u|battle-gen9randombattle-8"
        bridge.feed(key, "u", room, "|init|battle\n|title|u vs. v")
        feed_request(bridge, key, rqid=4)
        feed_request(bridge, other, rqid=9)
        assert bridge.advice_sync(key).rqid == 4 and bridge.stats()["battles"] == 2
        bridge.feed(key, "u", room, "|expire|")              # what the server sends when it closes a room,
        bridge.feed(key, "u", room, "|deinit")               # and to every player it takes out of it
        assert not bridge.knows(key) and bridge.knows(other)
        stats = bridge.stats()
        assert stats["left"] == 1 and stats["battles"] == 1 and [h["battles"] for h in stats["helpers"]] == [0, 1]
        assert bridge.sync() and [s["battles"] for s in bridge.helper_stats()] == [0, 1]     # the helper dropped its state
        with pytest.raises(FoulPlayBridgeError, match="it has ended"):
            bridge.advice_sync(key)
        bridge.feed(key, "u", room, "|deinit")               # once more: a late line, nothing else
        assert bridge.counters["late_lines"] == 1 and bridge.counters["left"] == 1
        assert bridge.advice_sync(other).rqid == 9           # the other battle is untouched
        # The next battle goes to the helper that is free again, not to the one a dead battle was counted on.
        feed_request(bridge, "u|battle-gen9randombattle-9", rqid=1)
        assert bridge.advice_sync("u|battle-gen9randombattle-9").helper == 0
    finally:
        bridge.close()


def test_leaving_a_room_that_was_never_followed_starts_nothing(fake_helper):
    bridge = make_bridge(fake_helper)
    try:
        bridge.feed("u|battle-gen9randombattle-1", "u", "battle-gen9randombattle-1", "|deinit")
        assert bridge.stats()["battles"] == 0 and bridge.counters["helper_starts"] == 0 and bridge.counters["left"] == 0
    finally:
        bridge.close()


def test_a_room_that_is_left_and_joined_again_gets_errors_not_a_stale_state(fake_helper):
    """
    Seen on the live server: /leave gives "|deinit", /join then sends the whole battle again from
    "|init|". Whatever happened in between was missed, and Foul Play cannot pick a battle up
    midway: the battle starts over and has no first request, so it answers errors.
    """
    bridge = make_bridge(fake_helper)
    try:
        key, room = "u|battle-gen9randombattle-7", "battle-gen9randombattle-7"
        bridge.feed(key, "u", room, "|init|battle\n|title|u vs. v")
        feed_request(bridge, key, rqid=4)
        assert bridge.advice_sync(key).rqid == 4
        bridge.feed(key, "u", room, "|deinit")
        bridge.feed(key, "u", room, "|init|battle\n|title|u vs. v\n|player|p1|u\n|turn|1\n|turn|2")
        assert bridge.counters["left"] == 1 and bridge.counters["rejoined"] == 1 and bridge.knows(key)
        with pytest.raises(FoulPlayBridgeError, match="no move to make"):
            bridge.advice_sync(key)
    finally:
        bridge.close()


# ---------------------------------------------------------------------------------------------
# how hard Foul Play searched, and when a run should stop
# ---------------------------------------------------------------------------------------------

def test_stats_say_how_hard_foul_play_searched(fake_helper):
    bridge = make_bridge(fake_helper, search_ms=50, env={"FAKE_ITERATIONS_PER_MS": "1200"})
    try:
        key = "u|battle-gen9randombattle-1"
        feed_request(bridge, key, rqid=1)
        for _ in range(3):
            assert bridge.advice_sync(key).iterations == 60000
        assert bridge.advice_sync(key, search_ms=20).iterations == 24000      # a starved or shorter search
        stats = bridge.stats()
        assert stats["decisions"] == 4 and stats["iterations"] == 204000
        assert stats["iterations_per_decision"] == 51000
        assert stats["recent_decisions"] == 4 and stats["recent_iterations_per_decision"] == 51000
    finally:
        bridge.close()


def test_the_recent_search_effort_follows_the_last_decisions(fake_helper, monkeypatch):
    monkeypatch.setattr(fb, "_RECENT_DECISIONS", 4)
    bridge = make_bridge(fake_helper, search_ms=50, env={"FAKE_ITERATIONS_PER_MS": "1200"})
    try:
        key = "u|battle-gen9randombattle-1"
        feed_request(bridge, key, rqid=1)
        for _ in range(6):
            bridge.advice_sync(key)                          # 60,000 each
        for _ in range(4):
            bridge.advice_sync(key, search_ms=10)            # then the helpers lose their cores: 12,000 each
        stats = bridge.stats()
        assert stats["iterations_per_decision"] == 40800     # the mean over the whole run hides it
        assert stats["recent_decisions"] == 4 and stats["recent_iterations_per_decision"] == 12000
    finally:
        bridge.close()


def test_teacher_problems_say_when_a_run_is_no_longer_foul_plays():
    assert fb.teacher_problems(asked=0, unanswered=0) == []
    assert fb.teacher_problems(asked=43485, unanswered=6) == []                 # what the live runs looked like
    assert fb.teacher_problems(asked=1000, unanswered=50) == []                 # the floor: 50 are tolerated in any case
    assert len(fb.teacher_problems(asked=1000, unanswered=51)) == 1
    assert fb.teacher_problems(asked=100000, unanswered=2000) == []             # 2% of what was asked
    late = fb.teacher_problems(asked=100000, unanswered=2001)
    assert len(late) == 1 and "2001 of 100000" in late[0] and "another bot moved" in late[0]
    assert len(fb.teacher_problems(asked=349, unanswered=347)) == 1             # the overload run of the review
    # iterations: judged only after enough decisions, and against the floor
    assert fb.teacher_problems(1000, 0, decisions=fb.MIN_ITERATIONS_SAMPLE - 1, iterations_per_decision=20000) == []
    low = fb.teacher_problems(1000, 0, decisions=fb.MIN_ITERATIONS_SAMPLE, iterations_per_decision=22867)   # the 20 ms that lost 56 of 80
    assert len(low) == 1 and "22867 iterations" in low[0] and str(fb.MIN_ITERATIONS) in low[0]
    assert fb.teacher_problems(1000, 0, decisions=5000, iterations_per_decision=58818) == []                # the 50 ms that held level
    assert fb.teacher_problems(1000, 0, decisions=5000, iterations_per_decision=20000, min_iterations=0) == []
    assert len(fb.teacher_problems(1000, 51, decisions=5000, iterations_per_decision=20000)) == 2
    assert fb.MIN_ITERATIONS < fb.TARGET_ITERATIONS


def test_calibrate_chooses_the_search_time_by_iterations(fake_helper):
    """On a machine where a millisecond is 1,000 iterations, 60,000 iterations are 60 ms, not 50."""
    if not os.path.isfile(FIXTURE):
        pytest.skip("tests/fixtures/foulplay_bridge_trace.jsonl.gz is missing")
    bridge = make_bridge(fake_helper, helpers=2, env={"FAKE_ITERATIONS_PER_MS": "1000"})
    try:
        result = fb.calibrate((20, 50, 75), target=60000, bridge=bridge)
        assert result["helpers"] == 2 and result["problems"] == []
        assert [row["ms"] for row in result["rows"]] == [20, 50, 75]
        assert [row["iterations_per_decision"] for row in result["rows"]] == [20000, 50000, 75000]
        orders = sum(1 for events in fb.load_trace() for e in events if e["t"] == "order")
        revives = sum(1 for events in fb.load_trace() for e in events if e["t"] == "order" and e["fp_reply"].get("revive"))
        assert all(row["decisions"] == 2 * (orders - revives) for row in result["rows"])     # every decision point, on both helpers
        assert result["search_ms"] == 60
        assert fb.calibrate((20, 50), target=60000, bridge=bridge, battles=1)["search_ms"] is None    # not reached: say so
        assert fb.calibrate((75, 100), target=60000, bridge=bridge, battles=1)["search_ms"] == 75     # the shortest measured
        assert bridge.sync() and bridge.stats()["battles"] == 0
        assert [h["battles"] for h in bridge.stats()["helpers"]] == [0, 0]
    finally:
        bridge.close()


def test_close_stops_every_helper_and_fails_what_was_waiting(fake_helper):
    bridge = make_bridge(fake_helper, helpers=3, env={"FAKE_SLOW_S": "30"})
    feed_request(bridge, "u|battle-slow-1")
    bridge.start()
    pids = [h["pid"] for h in bridge.stats()["helpers"]]
    assert len(pids) == 3 and all(pid_alive(p) for p in pids)
    waiting = bridge.advice_future("u|battle-slow-1")
    time.sleep(0.2)
    t0 = time.monotonic()
    bridge.close()
    assert time.monotonic() - t0 < 10
    assert wait_until(lambda: not any(pid_alive(p) for p in pids), timeout=10)
    with pytest.raises(FoulPlayBridgeError):
        waiting.result(timeout=5)
    bridge.feed("u|battle-gen9randombattle-9", "u", "battle-gen9randombattle-9", "|init|battle")   # ignored, no error
    with pytest.raises(FoulPlayBridgeError, match="no lines were fed|closed"):
        bridge.advice_sync("u|battle-gen9randombattle-9", timeout=2)
    bridge.close()                                       # twice is fine
    assert not any(p.pid in pids for p in fb._LIVE)


def test_nothing_is_started_once_the_interpreter_is_exiting(fake_helper, monkeypatch):
    """A battle line that arrives during shutdown (a room's deinit, say) must not start a helper."""
    fb.close_shared()
    monkeypatch.setattr(fb, "_exiting", True)
    bridge = make_bridge(fake_helper)
    try:
        feed_request(bridge, "u|battle-gen9randombattle-1")          # taken, but nothing is started for it
        assert bridge.counters["helper_starts"] == 0
        with pytest.raises(FoulPlayBridgeError, match="exiting"):
            bridge.start()
        with pytest.raises(FoulPlayBridgeError, match="exiting"):
            fb.shared_bridge()
        player = MixedPlayer()                                       # relies on the process-wide bridge
        asyncio.run(player._handle_battle_message(split("battle-gen9randombattle-2", "|deinit")))
        assert len(player.handled) == 1 and fb._shared is None
    finally:
        bridge.close()


def test_a_player_keeps_the_bridge_it_was_given_also_when_it_is_closed(fake_helper):
    fb.close_shared()
    bridge = make_bridge(fake_helper)
    player = MixedPlayer()
    player.foulplay = bridge
    battle = make_battle()
    bridge.close()
    assert player.foulplay is bridge

    async def scenario():
        await player._handle_battle_message(split(battle.battle_tag, "|request|" + json.dumps(battle.last_request)))
        return await player.foulplay_decision(battle)

    order, advice = asyncio.run(scenario())
    assert order is None and advice is None and "closed" in player.foulplay_problems[0]
    assert fb._shared is None                                        # no other pool of helpers was made behind its back
    assert bridge.counters["helper_starts"] == 0


def test_helpers_stop_when_the_process_that_started_them_is_killed(fake_helper):
    """A client that is killed outright closes the helper's stdin by dying: the helper must leave."""
    script = os.path.join(new_dir("fpbridge_parent_"), "parent.py")
    write(script, "\n".join([
        "import json, sys, time",
        "sys.path.insert(0, %r)" % REPO,
        "from foulplay_bridge import FoulPlayBridge",
        "b = FoulPlayBridge(helpers=2, command=[sys.executable, '-u', %r]).start()" % fake_helper,
        "print(json.dumps([h['pid'] for h in b.stats()['helpers']]), flush=True)",
        "time.sleep(600)",
    ]))
    parent = subprocess.Popen([sys.executable, "-u", script], stdout=subprocess.PIPE, text=True)
    try:
        pids = json.loads(parent.stdout.readline())
        assert len(pids) == 2 and all(pid_alive(p) for p in pids)
        kill_hard(parent.pid)
        parent.wait(timeout=10)
        assert wait_until(lambda: not any(pid_alive(p) for p in pids), timeout=15)
    finally:
        if parent.poll() is None:
            parent.kill()
        for pid in locals().get("pids", []):
            if pid_alive(pid):
                kill_hard(pid)


# ---------------------------------------------------------------------------------------------
# our 14 actions, on made-up battles
# ---------------------------------------------------------------------------------------------

class FMove:
    def __init__(self, id):
        self.id = id


class FMon:
    def __init__(self, name, moves=(), hp=100):
        self.name = self.species = name
        self.current_hp = hp
        self.moves = {m: FMove(m) for m in moves}

    def __bool__(self):
        return True


class FOrder:
    def __init__(self, order, terastallize=False, **_flags):
        self.order = order
        self.terastallize = terastallize
        self.mega = self.dynamax = self.z_move = False

    @property
    def message(self):
        if hasattr(self.order, "current_hp"):
            return "/choose switch " + self.order.name
        return "/choose move " + self.order.id + (" terastallize" if self.terastallize else "")


class FPlayer:
    username = "u"

    def create_order(self, order, **flags):
        return FOrder(order, **flags)


class FBattle:
    battle_tag = "battle-gen9randombattle-1"
    turn = 3


def make_battle(known_moves=("tackle", "growl", "ember", "surf"), request_moves=None, disabled=(), can_tera=True,
                force=False, trapped=False, fainted=(), switchable=None, active_slot=0) -> FBattle:
    """A battle as poke-env would hold it, and the request it came from. Team order is fixed; the request lists the active one first."""
    battle = FBattle()
    names = ["Mon%d" % i for i in range(6)]
    mons = [FMon(n, known_moves if i == active_slot else ("a", "b")) for i, n in enumerate(names)]
    battle.team = {"p1: " + m.name: m for m in mons}
    battle.active_pokemon = mons[active_slot]
    order = [active_slot] + [i for i in range(6) if i != active_slot]
    request_moves = list(request_moves if request_moves is not None else known_moves)
    side = {"name": "u", "id": "p1", "pokemon": [
        {"ident": "p1: " + names[i], "details": names[i] + ", L80", "condition": "0 fnt" if i in fainted else "100/100",
         "active": i == active_slot} for i in order]}
    request = {"rqid": 7, "side": side}
    if force:
        request["forceSwitch"] = [True]
        battle.available_moves = []
    else:
        active = {"moves": [{"move": m, "id": m, "disabled": m in disabled} for m in request_moves]}
        if can_tera:
            active["canTerastallize"] = "Fire"
        if trapped:
            active["trapped"] = True
        request["active"] = [active]
        battle.available_moves = [mons[active_slot].moves.get(m) or FMove(m) for m in request_moves if m not in disabled]
    if switchable is None:
        switchable = [] if trapped else [i for i in range(6) if i != active_slot and i not in fainted]
    battle.available_switches = [mons[i] for i in switchable]
    battle.can_tera = "fire" if (can_tera and not force) else False
    battle.force_switch = force
    battle.last_request = request
    return battle


def make_revive_battle(fainted=(2, 5)) -> FBattle:
    """The request after Revival Blessing: a forced "switch" whose targets are the fainted Pokemon."""
    battle = make_battle(force=True, fainted=fainted, switchable=list(fainted))
    for mon in battle.last_request["side"]["pokemon"]:
        mon["reviving"] = mon["condition"].endswith(" fnt")
    return battle


def against_assembler(battle) -> None:
    if ASSEMBLER is None:
        return
    assert [int(x) for x in ASSEMBLER.create_action_mask(battle, list(battle.team.values()))] == fb.legal_mask(battle)
    for move in battle.available_moves:
        for tera in (False, True):
            order = FOrder(move, tera)
            assert int(ASSEMBLER.map_order_to_index(order, battle)) == fb.order_to_index(order, battle)
    for mon in battle.team.values():
        order = FOrder(mon)
        assert int(ASSEMBLER.map_order_to_index(order, battle)) == fb.order_to_index(order, battle)
    for index in range(fb.ACTION_DIM):
        target, kwargs = ASSEMBLER.map_index_to_order(index, battle)
        mine, tera = fb.index_to_target(index, battle)
        assert (None if target == "DEFAULT" else target) is mine
        assert bool(kwargs.get("terastallize", False)) == tera


def test_an_ordinary_turn_maps_onto_the_14_actions():
    battle, player = make_battle(), FPlayer()
    against_assembler(battle)
    assert fb.legal_mask(battle) == [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert fb.choice_to_index(player, battle, "move 1") == 0
    assert fb.choice_to_index(player, battle, "move 4") == 3
    assert fb.choice_to_index(player, battle, "move 2 terastallize") == 11
    # "switch 2" is the second Pokemon of the REQUEST; our index is its place in battle.team
    assert fb.choice_to_index(player, battle, "switch 2") == 5
    assert fb.choice_to_index(player, battle, "switch 6") == 9
    assert fb.choice_to_index(player, battle, "switch 1") is None       # the active one
    assert fb.choice_to_index(player, battle, "move 5") is None
    assert fb.choice_to_index(player, battle, "") is None and fb.choice_to_index(player, battle, None) is None
    assert fb.choice_to_index(player, battle, "default") is None


def test_the_request_order_and_the_team_order_are_kept_apart():
    battle, player = make_battle(active_slot=3), FPlayer()
    against_assembler(battle)
    # request order: Mon3 (active), Mon0, Mon1, Mon2, Mon4, Mon5. Team order: Mon0 .. Mon5.
    assert fb.legal_mask(battle) == [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    assert fb.choice_to_index(player, battle, "switch 2") == 4          # Mon0
    assert fb.choice_to_index(player, battle, "switch 4") == 6          # Mon2
    assert fb.choice_to_index(player, battle, "switch 5") == 8          # Mon4
    assert fb.order_to_choice(battle, FOrder(battle.team["p1: Mon4"])) == "switch 5"
    assert fb.order_to_choice(battle, FOrder(battle.active_pokemon.moves["ember"], True)) == "move 3 terastallize"
    assert fb.order_to_choice(battle, FOrder("/choose default")) == "default"
    assert fb.order_to_choice(battle, None) == "default"


def test_disabled_moves_fainted_pokemon_and_no_tera_are_not_actions():
    battle, player = make_battle(disabled=("growl",), can_tera=False, fainted=(2, 5)), FPlayer()
    against_assembler(battle)
    assert fb.legal_mask(battle) == [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0]
    assert fb.choice_to_index(player, battle, "move 2") is None
    assert fb.choice_to_index(player, battle, "move 1 terastallize") is None    # no Tera on offer: not an action of ours
    assert fb.choice_to_index(player, battle, "switch 3") is None
    assert fb.choice_to_index(player, battle, "switch 4") == 7


def test_a_forced_switch_has_only_switches():
    battle, player = make_battle(force=True, fainted=(0,)), FPlayer()
    against_assembler(battle)
    assert fb.legal_mask(battle) == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    assert fb.choice_to_index(player, battle, "switch 3") == 6
    assert fb.choice_to_index(player, battle, "move 1") is None


def test_struggle_and_recharge_are_action_0():
    player = FPlayer()
    struggle = make_battle(request_moves=("struggle",), can_tera=False, trapped=False)
    against_assembler(struggle)
    assert fb.legal_mask(struggle)[:4] == [1, 0, 0, 0]
    assert fb.choice_to_index(player, struggle, "move 1") == 0
    recharge = make_battle(request_moves=("recharge",), can_tera=False, trapped=True)
    against_assembler(recharge)
    assert fb.legal_mask(recharge) == [1] + [0] * 13
    assert fb.choice_to_index(player, recharge, "move 1") == 0


def test_a_move_the_14_actions_cannot_say_is_not_given_an_index():
    # The request offers a move poke-env does not list among the active Pokemon's moves, next to
    # ones it does. The assembler would call it action 0, and action 0 plays another move.
    battle, player = make_battle(request_moves=("tackle", "copycat"), can_tera=False), FPlayer()
    against_assembler(battle)
    order = fb.choice_to_order(player, battle, "move 2")
    assert order is not None and order.order.id == "copycat"
    assert fb.order_to_index(order, battle) == 0
    assert fb.choice_to_index(player, battle, "move 2") is None
    assert fb.choice_to_index(player, battle, "move 1") == 0


def advice_of(rows, choice=None, best=None, tera_dropped=False) -> Advice:
    options = [Option(choice=c, fp="fp-%s" % c, visits=v, score=0.5, total_score=v * 0.5, policy=p) for c, v, p in rows]
    return Advice(choice=choice or rows[0][0], best=best or rows[0][0], options=options, tera_dropped=tera_dropped)


def test_fill_actions_puts_the_search_on_our_indices():
    battle, player = make_battle(), FPlayer()
    advice = advice_of([("move 3", 600, 0.6), ("switch 4", 300, 0.3), ("move 1 terastallize", 100, 0.1)], choice="switch 4")
    fb.fill_actions(player, battle, advice)
    assert advice.mask == fb.legal_mask(battle)
    assert advice.visits == [0, 0, 600, 0, 0, 0, 0, 300, 0, 0, 100, 0, 0, 0]
    assert advice.policy[2] == 0.6 and advice.policy[7] == 0.3 and advice.policy[10] == 0.1
    assert abs(sum(advice.policy) - 1.0) < 1e-9
    assert advice.score[2] == 0.5 and advice.score[0] is None
    assert advice.action == 7 and advice.best_action == 2 and advice.unmapped == []


def test_fill_actions_adds_up_options_that_are_the_same_choice_and_keeps_what_does_not_fit():
    # Locked into a move: the helper reports Foul Play's "outrage-tera" as the plain move.
    battle, player = make_battle(request_moves=("tackle",), can_tera=False, trapped=True), FPlayer()
    advice = advice_of([("move 1", 700, 0.7), ("move 1", 200, 0.2), (None, 100, 0.1)], tera_dropped=True)
    fb.fill_actions(player, battle, advice)
    assert advice.visits[0] == 900 and abs(advice.policy[0] - 0.9) < 1e-9
    assert advice.action == 0 and advice.best_action == 0
    assert [o.choice for o in advice.unmapped] == [None] and abs(sum(advice.policy) - 0.9) < 1e-9
    assert advice.tera_dropped is True


def test_fill_actions_uses_the_callers_own_index_function_and_mask():
    battle, player = make_battle(), FPlayer()
    seen = []

    def index_of(order, b):
        seen.append(order.message)
        return fb.order_to_index(order, b)

    advice = advice_of([("move 2", 10, 1.0)])
    fb.fill_actions(player, battle, advice, index_of=index_of, mask=[0, 1] + [0] * 12)
    assert seen == ["/choose move growl"] and advice.action == 1 and advice.mask == [0, 1] + [0] * 12
    refused = advice_of([("move 3", 10, 1.0)])
    fb.fill_actions(player, battle, refused, index_of=index_of, mask=[0, 1] + [0] * 12)
    assert refused.action is None and len(refused.unmapped) == 1      # masked by the caller: not an action


# ---------------------------------------------------------------------------------------------
# the mixin, on a player that is not a poke-env player
# ---------------------------------------------------------------------------------------------

class BasePlayer(FPlayer):
    def __init__(self):
        self.handled = []
        self.finished = []

    async def _handle_battle_message(self, split_messages):
        self.handled.append(split_messages)

    def _battle_finished_callback(self, battle):
        self.finished.append(battle.battle_tag)


class MixedPlayer(FoulPlayMixin, BasePlayer):
    foulplay_enabled = True


def split(room: str, data: str):
    return [[">" + room]] + [line.split("|") for line in data.split("\n")]


def test_the_mixin_is_off_until_switched_on(fake_helper):
    class Quiet(FoulPlayMixin, BasePlayer):
        pass

    bridge = make_bridge(fake_helper)
    try:
        player = Quiet()
        player.foulplay = bridge
        asyncio.run(player._handle_battle_message(split("battle-gen9randombattle-1", "|init|battle")))
        player._battle_finished_callback(make_battle())
        assert len(player.handled) == 1 and player.finished == ["battle-gen9randombattle-1"]
        assert bridge.counters["fed"] == 0 and bridge.counters["helper_starts"] == 0
    finally:
        bridge.close()


def test_the_mixin_forwards_battle_rooms_decides_notes_and_forgets(fake_helper):
    bridge = make_bridge(fake_helper)
    try:
        player = MixedPlayer()
        player.foulplay = bridge
        battle = make_battle(active_slot=1)
        room = battle.battle_tag
        key = "u|" + room

        async def scenario():
            await player._handle_battle_message(split("lobby", "|c|someone|hi"))                  # not a battle room
            await player._handle_battle_message(split(room, "|init|battle\n|title|u vs. v"))
            await player._handle_battle_message(split(room, "|request|" + json.dumps(battle.last_request)))
            return await player.foulplay_decision(battle)

        order, advice = asyncio.run(scenario())
        assert len(player.handled) == 3 and bridge.counters["fed"] == 2
        assert order.message == "/choose move tackle" and advice.choice == "move 1" and advice.action == 0
        assert advice.raw["feeds"] == 2 and advice.rqid == 7
        assert advice.mask == fb.legal_mask(battle) and abs(sum(advice.policy) - 1.0) < 1e-9
        assert sum(1 for v in advice.visits if v) == 13 and advice.unmapped == []
        assert player.foulplay_counts == {"asked": 1, "answered": 1, "errors": 0, "illegal": 0, "tera_dropped": 0, "revive": 0}
        assert bridge.advice_sync(key).raw["chose"] == ["move 1"]       # the order was noted as played
        player.foulplay_note(battle, FOrder(battle.team["p1: Mon4"]))
        assert bridge.advice_sync(key).raw["chose"] == ["move 1", "switch 5"]
        player._battle_finished_callback(battle)
        assert player.finished == [room] and not bridge.knows(key)
    finally:
        bridge.close()


def test_the_mixin_counts_what_it_could_not_use(fake_helper):
    bridge = make_bridge(fake_helper)
    try:
        player = MixedPlayer()
        player.foulplay = bridge
        battle = make_battle()

        async def scenario():
            first = await player.foulplay_decision(battle)                  # nothing fed: the bridge refuses
            await player._handle_battle_message(split(battle.battle_tag, "|request|" + json.dumps(battle.last_request)))
            battle.available_moves = battle.available_moves[1:]             # poke-env disagrees about move 1
            second = await player.foulplay_decision(battle)
            return first, second

        (order1, advice1), (order2, advice2) = asyncio.run(scenario())
        assert order1 is None and advice1 is None
        assert order2 is None and advice2 is not None and advice2.choice == "move 1" and advice2.action is None
        assert player.foulplay_counts["errors"] == 1 and player.foulplay_counts["illegal"] == 1
        assert len(player.foulplay_problems) == 2 and "not a legal order" in player.foulplay_problems[1]
        assert bridge.counters["notes"] == 0                                # nothing was played, nothing noted
    finally:
        bridge.close()


def test_a_revival_blessing_request_is_no_answer_and_not_a_bridge_error(fake_helper):
    """Whom to revive: Foul Play cannot say. The player decides, and it is counted apart from failures."""
    bridge = make_bridge(fake_helper)
    try:
        player = MixedPlayer()
        player.foulplay = bridge
        battle = make_revive_battle()
        assert fb.legal_mask(battle) == [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]     # the two fainted ones
        against_assembler(battle)

        async def scenario():
            await player._handle_battle_message(split(battle.battle_tag, "|request|" + json.dumps(battle.last_request)))
            return await player.foulplay_decision(battle)

        order, advice = asyncio.run(scenario())
        assert order is None and advice is None
        assert player.foulplay_counts["revive"] == 1 and player.foulplay_counts["errors"] == 0
        assert player.foulplay_problems == []
        with pytest.raises(FoulPlayBridgeError) as caught:
            bridge.advice_sync("u|" + battle.battle_tag)
        assert caught.value.revive is True and "Revival Blessing" in str(caught.value)
        player.foulplay_note(battle, FOrder(battle.team["p1: Mon5"]))                  # whom the player revived
        assert bridge.counters["notes"] == 1
    finally:
        bridge.close()


def test_the_mixin_can_take_the_largest_share_instead_of_the_drawn_move(fake_helper):
    bridge = make_bridge(fake_helper)
    try:
        player = MixedPlayer()
        player.foulplay = bridge
        battle = make_battle()
        asyncio.run(player._handle_battle_message(split(battle.battle_tag, "|request|" + json.dumps(battle.last_request))))
        original = fb._advice_from

        def drawn_second(reply, wall_ms, helper):
            advice = original(reply, wall_ms, helper)
            advice.choice = "move 2"                                        # Foul Play drew its second option
            return advice

        fb._advice_from = drawn_second
        try:
            drawn, _ = asyncio.run(player.foulplay_decision(battle))
            best, _ = asyncio.run(player.foulplay_decision(battle, pick="best"))
        finally:
            fb._advice_from = original
        assert drawn.message == "/choose move growl" and best.message == "/choose move tackle"
    finally:
        bridge.close()


def test_a_teacher_step_is_a_label_only_when_it_is_foul_plays_own_move(fake_helper):
    """
    foulplay_teacher() is what a player that LEARNS from Foul Play asks. Its action index is the
    label; None means "no label on this request", whatever the reason, and the caller's own move
    for that request must not be recorded as Foul Play's.
    """
    refusing = make_bridge(fake_helper, env={"FAKE_REFUSE": "PanicException: Invalid PokemonMoveIndex: 4"})
    bridge = make_bridge(fake_helper)
    try:
        player = MixedPlayer()
        player.foulplay = bridge
        battle = make_battle(active_slot=1)

        async def ask(p, b, **kwargs):
            await p._handle_battle_message(split(b.battle_tag, "|request|" + json.dumps(b.last_request)))
            return await p.foulplay_teacher(b, **kwargs)

        # 1. Foul Play answers: its order, and the index that plays exactly that order.
        order, action, advice = asyncio.run(ask(player, battle))
        assert order.message == "/choose move tackle" and action == 0 == advice.action
        target, tera = fb.index_to_target(action, battle)
        assert target is order.order and tera is False
        order, action, advice = asyncio.run(ask(player, battle, pick="best"))
        assert order.message == "/choose move tackle" and action == 0

        # 2. A battle Foul Play's own code cannot follow (its engine panicked): no label.
        lost = MixedPlayer()
        lost.foulplay = refusing
        assert asyncio.run(ask(lost, battle)) == (None, None, None)
        assert lost.foulplay_counts["errors"] == 1 and "PanicException" in lost.foulplay_problems[0]

        # 3. Whom Revival Blessing revives: no label, and not counted as a failure.
        revive = make_revive_battle()
        revive.battle_tag = "battle-gen9randombattle-2"
        assert asyncio.run(ask(player, revive)) == (None, None, None)
        assert player.foulplay_counts["revive"] == 1 and player.foulplay_counts["errors"] == 0

        # 4. An answer poke-env does not hold legal: no label (the advice is there to look at).
        illegal = make_battle()
        illegal.battle_tag = "battle-gen9randombattle-3"

        async def disagree():
            await player._handle_battle_message(split(illegal.battle_tag, "|request|" + json.dumps(illegal.last_request)))
            illegal.available_moves = illegal.available_moves[1:]
            return await player.foulplay_teacher(illegal)

        order, action, advice = asyncio.run(disagree())
        assert order is None and action is None and advice is not None and player.foulplay_counts["illegal"] == 1

        # 5. A legal move our 14 actions cannot say (the assembler would call it action 0, which
        #    plays another move): no label either, and it is written down.
        unsayable = make_battle(request_moves=("copycat", "tackle"), can_tera=False)
        unsayable.battle_tag = "battle-gen9randombattle-4"
        assert fb.choice_to_order(player, unsayable, "move 1").order.id == "copycat"
        order, action, advice = asyncio.run(ask(player, unsayable))
        assert order is None and action is None and advice.choice == "move 1"
        assert "cannot say" in player.foulplay_problems[-1]

        # 6. Nobody answers in time (the helper is far behind): no label, at once.
        slow = make_bridge(fake_helper, search_ms=100, env={"FAKE_SLOW_S": "0.4"})
        try:
            hurried = MixedPlayer()
            hurried.foulplay = slow
            hurried.foulplay_timeout = 0.15
            late = make_battle()
            late.battle_tag = "battle-slow-5"
            slow.start()
            assert asyncio.run(ask(hurried, late)) == (None, None, None)
            assert hurried.foulplay_counts["errors"] == 1 and slow.helper_stats()[0]["decisions"] == 0
        finally:
            slow.close()
    finally:
        bridge.close()
        refusing.close()


def test_the_mixin_forgets_a_battle_whose_room_is_gone(fake_helper):
    bridge = make_bridge(fake_helper)
    try:
        player = MixedPlayer()
        player.foulplay = bridge
        battle = make_battle()
        room = battle.battle_tag

        async def scenario():
            await player._handle_battle_message(split(room, "|init|battle\n|title|u vs. v"))
            await player._handle_battle_message(split(room, "|request|" + json.dumps(battle.last_request)))
            await player.foulplay_decision(battle)
            await player._handle_battle_message(split(room, "|expire|"))
            await player._handle_battle_message(split(room, "|deinit"))     # no "|win|": the finished callback never comes

        asyncio.run(scenario())
        assert player.finished == [] and not bridge.knows("u|" + room)
        assert bridge.sync() and bridge.helper_stats()[0]["battles"] == 0 and bridge.stats()["stored_bytes"] == 0
    finally:
        bridge.close()


def test_a_missing_bridge_does_not_stop_the_player(monkeypatch, caplog):
    monkeypatch.setenv("FOULPLAY_DIR", os.path.join(new_dir("fpbridge_none_"), "nothing-here"))
    monkeypatch.delenv("FOULPLAY_PYTHON", raising=False)
    fb.close_shared()
    player = MixedPlayer()                                                  # no bridge given: the shared one
    battle = make_battle()
    try:
        with caplog.at_level(logging.WARNING, logger="foulplay_bridge"):
            async def scenario():
                for _ in range(3):
                    await player._handle_battle_message(split(battle.battle_tag, "|turn|1"))
                return await player.foulplay_decision(battle)

            order, advice = asyncio.run(scenario())
        assert order is None and advice is None and len(player.handled) == 3
        assert player.foulplay_counts["errors"] == 1
        assert "not installed" in player.foulplay_problems[0]
    finally:
        fb.close_shared()


# ---------------------------------------------------------------------------------------------
# the Foul-Play-free parts of the helper itself
# ---------------------------------------------------------------------------------------------

def load_server_module():
    spec = importlib.util.spec_from_file_location("fp_bridge_server_under_test", os.path.join(REPO, "bench", "fp_bridge_server.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_helper_file_imports_nothing_of_foul_play_until_it_is_started():
    server = load_server_module()
    assert server.PROTOCOL_VERSION == 2 and "fp" not in sys.modules and "poke_engine" not in sys.modules


def test_the_fed_socket_lets_a_battle_loop_be_advanced_by_hand():
    """The technique the helper rests on: a coroutine that awaits the socket, driven with send()."""
    server = load_server_module()
    socket = server._FedSocket()
    seen = []

    async def battle_loop(client):
        first = await client.receive_message()
        seen.append(("start", first))
        while True:
            message = await client.receive_message()
            if "|win|" in message:
                await client.send_message("room", ["gg"])
                await client.leave_battle("room")
                return message.split("|win|")[1]
            seen.append(("update", message))
            if "|request|" in message:
                await client.send_message("room", ["/choose move 1", "3"])

    coro = battle_loop(socket)
    assert isinstance(coro.send(None), server._NextMessage)        # runs up to the first receive
    coro.send(">room\n|init|battle")
    coro.send(">room\n|request|{}")
    assert socket.sent == 1 and socket.last_sent == ("room", ["/choose move 1", "3"])
    assert seen == [("start", ">room\n|init|battle"), ("update", ">room\n|request|{}")]
    with pytest.raises(StopIteration) as stop:
        coro.send(">room\n|win|alice")
    assert stop.value.value == "alice" and socket.sent == 2


def test_the_inline_executor_runs_the_search_where_it_is_submitted():
    server = load_server_module()
    order = []
    with server._InlineExecutor(max_workers=1) as pool:
        a = pool.submit(lambda x: order.append(x) or x * 2, 21)
        order.append("after first")
        b = pool.submit(lambda: 1 / 0)
    assert a.result() == 42 and order == [21, "after first"]
    with pytest.raises(ZeroDivisionError):
        b.result()


# ---------------------------------------------------------------------------------------------
# recorded battles: real poke-env battle objects, the real helper's recorded answers
# ---------------------------------------------------------------------------------------------

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


needs_recording = pytest.mark.skipif(not (HAVE_POKE_ENV and os.path.isfile(FIXTURE)),
                                     reason="needs poke-env and tests/fixtures/foulplay_bridge_trace.jsonl.gz")


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


def run_on_player_loop(player, coro, timeout: float = 120.0):
    return asyncio.run_coroutine_threadsafe(coro, player.ps_client.loop).result(timeout=timeout)


@needs_recording
def test_recorded_answers_of_the_real_helper_map_onto_legal_actions():
    """
    Every decision of the recorded battles: the real helper's answer for that very request, put
    on our action indices with the real poke-env battle object, and checked against the
    repository's ObservationAssembler the way worker.py uses it.
    """
    from poke_env.player import Player

    checked = {"decisions": 0, "tera_dropped": 0, "tera": 0, "switch": 0, "revive": 0, "problems": []}

    class Replayer(Player):
        recorded = None

        def choose_move(self, battle):
            event = self.recorded.pop(0)
            assert event["rqid"] == battle.last_request["rqid"]
            if not event["fp_reply"].get("ok"):
                # The one kind of request Foul Play has no answer to: whom Revival Blessing revives.
                assert event["fp_reply"].get("revive") is True, event["fp_reply"]
                assert battle.force_switch and any(m.get("reviving") for m in battle.last_request["side"]["pokemon"])
                checked["revive"] += 1
                played = fb.choice_to_order(self, battle, event["choice"])
                assert played is not None and played.order.fainted       # poke-env offers the fainted ones
                assert fb.legal_mask(battle)[fb.order_to_index(played, battle)] == 1
                return played
            advice = fb.fill_actions(self, battle, fb._advice_from(dict(event["fp_reply"]), 1.0, 0),
                                     index_of=ASSEMBLER.map_order_to_index if ASSEMBLER else None)
            where = "%s turn %s" % (battle.battle_tag, battle.turn)
            checked["decisions"] += 1
            checked["tera_dropped"] += advice.tera_dropped
            if advice.action is None or not advice.mask[advice.action]:
                checked["problems"].append("%s: %r is not a legal action" % (where, advice.choice))
            elif advice.action >= 10:
                checked["tera"] += 1
            elif advice.action >= 4:
                checked["switch"] += 1
            if advice.unmapped:
                checked["problems"].append("%s: unmapped %s" % (where, [(o.choice, o.fp) for o in advice.unmapped]))
            if abs(sum(advice.policy) - 1.0) > 0.01:
                checked["problems"].append("%s: shares sum to %.4f" % (where, sum(advice.policy)))
            if any(v and not m for v, m in zip(advice.visits, advice.mask)):
                checked["problems"].append("%s: visits on a masked action" % where)
            order = fb.choice_to_order(self, battle, advice.choice)
            if order is None:
                checked["problems"].append("%s: %r is no legal poke-env order" % (where, advice.choice))
            elif ASSEMBLER is not None:
                index = int(ASSEMBLER.map_order_to_index(order, battle))
                mask = [int(x) for x in ASSEMBLER.create_action_mask(battle, list(battle.team.values()))]
                target, kwargs = ASSEMBLER.map_index_to_order(index, battle)
                if mask != advice.mask or index != advice.action or not mask[index]:
                    checked["problems"].append("%s: the assembler says index %s, mask %s" % (where, index, mask))
                elif target == "DEFAULT" or self.create_order(target, **kwargs).message != order.message:
                    checked["problems"].append("%s: action %s does not play %s" % (where, index, order.message))
            # Play what the recorded player played, so that the battle follows the recording.
            played = fb.choice_to_order(self, battle, event["choice"])
            assert played is not None, "%s: the recorded choice %r is not legal for poke-env" % (where, event["choice"])
            assert fb.order_to_choice(battle, played) == event["choice"]
            return played

    for number, events in enumerate(load_fixture()):
        username = events[-1]["username"]
        player = offline_player(Replayer, username)
        player.recorded = [e for e in events if e["t"] == "order"]
        expected = len(player.recorded)

        async def feed_all():
            for event in events:
                if event["t"] == "msg":
                    await player._handle_battle_message(split(event["room"], event["data"]))

        run_on_player_loop(player, feed_all())
        assert player.recorded == [], "battle %d: %d recorded decisions were never asked for" % (number, len(player.recorded))
        assert len([m for _room, m in player.sent if m.startswith("/choose")]) == expected
        assert player.n_finished_battles == 1

    assert checked["problems"] == []
    assert checked["decisions"] >= 100 and checked["switch"] > 0 and checked["tera"] > 0
    assert checked["tera_dropped"] >= 1      # the recording holds a Pokemon locked into Outrage
    assert checked["revive"] >= 1            # and a battle in which the recorded player used Revival Blessing


@needs_recording
def test_the_bridge_player_plays_recorded_battles_through_the_fake_helper(fake_helper):
    """
    FoulPlayBridgePlayer end to end with real poke-env objects: every request gets a legal order,
    and the only fallbacks are the requests that ask whom Revival Blessing revives.
    """
    from foulplay_player import FoulPlayBridgePlayer

    bridge = make_bridge(fake_helper, helpers=2)
    try:
        asked = revives = 0
        for events in load_fixture():
            username = events[-1]["username"]
            player = offline_player(FoulPlayBridgePlayer, username, bridge=bridge, search_ms=33, keep_advice=True)
            orders = [e for e in events if e["t"] == "order"]
            expected = len(orders)
            revive = sum(1 for e in orders if e["fp_reply"].get("revive"))

            async def feed_all():
                for event in events:
                    if event["t"] == "msg":
                        await player._handle_battle_message(split(event["room"], event["data"]))

            run_on_player_loop(player, feed_all())
            assert player.foulplay_decisions == expected, player.foulplay_problems
            assert player.foulplay_fallbacks == revive and player.foulplay_counts["revive"] == revive
            assert player.foulplay_counts["errors"] == 0 and player.foulplay_counts["illegal"] == 0
            assert len([m for _room, m in player.sent if m.startswith("/choose")]) == expected
            assert player.last_advice == {} and player.n_finished_battles == 1
            asked += expected
            revives += revive
        assert revives >= 1
        assert bridge.sync()
        stats = bridge.stats()
        assert stats["decisions"] == asked - revives and stats["notes"] == asked
        assert stats["battles"] == 0 and stats["stored_bytes"] == 0 and stats["forgotten"] == len(load_fixture())
        assert sum(s["battles"] for s in bridge.helper_stats() if s) == 0
    finally:
        bridge.close()


@needs_recording
def test_a_finished_battle_is_not_asked_about(fake_helper):
    from foulplay_player import FoulPlayBridgePlayer

    bridge = make_bridge(fake_helper)
    try:
        events = load_fixture()[0]
        player = offline_player(FoulPlayBridgePlayer, events[-1]["username"], bridge=bridge)

        async def scenario():
            for event in events:
                if event["t"] == "msg":
                    await player._handle_battle_message(split(event["room"], event["data"]))
            battle = next(iter(player.battles.values()))
            assert battle.finished
            return await player.choose_move(battle)          # poke-env asks again after the server's late error

        before = bridge.counters["decision_errors"]
        order = run_on_player_loop(player, scenario())
        assert order.message == "/choose default"
        assert player.foulplay_after_end == 1 and player.foulplay_fallbacks == 0
        assert bridge.counters["decision_errors"] == before
    finally:
        bridge.close()


# ---------------------------------------------------------------------------------------------
# worker.py, once Foul Play is wired into it as a teacher
# ---------------------------------------------------------------------------------------------

_WORKER = {}


def wired_worker():
    """
    worker.py with Foul Play as a teacher: the repository's own once the requested change is in
    it, or the file $FOULPLAY_TEST_WORKER names (the change applied to a copy, which is how it
    was checked before anybody edited worker.py). Skips while there is neither, and where what
    worker.py imports (numpy, ray, torch, poke-env) is not installed.
    """
    if "module" not in _WORKER:
        path = os.environ.get("FOULPLAY_TEST_WORKER") or os.path.join(REPO, "worker.py")
        with open(path, "r", encoding="utf-8") as fh:
            if "FoulPlayMixin" not in fh.read():
                pytest.skip("worker.py is not wired to the Foul Play bridge yet (the requested change is not applied)")
        for needed in ("numpy", "poke_env", "ray", "torch"):
            pytest.importorskip(needed)
        spec = importlib.util.spec_from_file_location("worker_with_foulplay", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["worker_with_foulplay"] = module
        spec.loader.exec_module(module)
        _WORKER["module"] = module
    return _WORKER["module"]


def teaching_player(worker, username: str, mode: str, bridge: FoulPlayBridge):
    """A worker.SyncBridgePlayer that learns from Foul Play, offline: its messages are fed by hand."""
    import dataclasses
    import queue as queue_module

    import config

    base = config.RunConfig.default()
    cfg = dataclasses.replace(base, learner=dataclasses.replace(base.learner, mode=mode),
                              rollout=dataclasses.replace(base.rollout, imitation_teacher="foulplay"),
                              reward=dataclasses.replace(base.reward, shaping="none"))
    player = offline_player(worker.SyncBridgePlayer, username, cfg=cfg, event_queue=queue_module.Queue(),
                            action_futures={}, ep_sem=asyncio.Semaphore(64))
    assert player.foulplay_enabled is True
    player.foulplay = bridge
    return player


def play_recorded_battle(player, events) -> dict:
    """Feeds one recorded battle to the player and says what it recorded and what it sent."""
    async def feed_all():
        for event in events:
            if event["t"] == "msg":
                await player._handle_battle_message(split(event["room"], event["data"]))

    run_on_player_loop(player, feed_all())
    recorded = list(player.event_queue.queue)
    return {"labels": [e for e in recorded if e[0] == "IMITATION"],
            "sent": [m for _room, m in player.sent if m.startswith("/choose")]}


@needs_recording
def test_the_worker_never_records_another_bots_move_as_foul_plays_label(fake_helper, monkeypatch):
    """
    The review's first finding. When Foul Play has no answer (a helper down or behind, a battle
    its code could not follow, Revival Blessing) SimpleHeuristics moves so that the game goes
    on. That move was recorded as the teacher's label, exactly like a Foul Play label, and
    SimpleHeuristics is the weakest teacher there is. It must be played and NOT recorded.
    """
    worker = wired_worker()
    monkeypatch.chdir(REPO)                                   # ObservationAssembler reads vocab.json from here
    events = next(e for e in load_fixture() if not any(x["t"] == "order" and x["fp_reply"].get("revive") for x in e))
    username = events[-1]["username"]
    decisions = len([e for e in events if e["t"] == "order"])
    answering = make_bridge(fake_helper)
    refusing = make_bridge(fake_helper, env={"FAKE_REFUSE": "PanicException: Invalid PokemonMoveIndex: 4"})
    try:
        # Foul Play answers: every decision is a label, and it is Foul Play's.
        taught = teaching_player(worker, username, "imitation", answering)
        got = play_recorded_battle(taught, events)
        assert len(got["sent"]) == decisions and taught.teacher_fallbacks == 0
        assert len(got["labels"]) == decisions
        assert taught.foulplay_counts["answered"] == decisions

        # Foul Play has no answer: the game goes on, every move is told to Foul Play, nothing is recorded.
        untaught = teaching_player(worker, username, "imitation", refusing)
        got = play_recorded_battle(untaught, events)
        assert len(got["sent"]) == decisions                  # another bot moved every time
        assert untaught.teacher_fallbacks == decisions        # and every one of them was counted
        assert got["labels"] == []                            # none of them as a teacher's step
        assert refusing.sync() and refusing.counters["notes"] == decisions

        # jepa_pretraining trains no policy: the action is what the world model is conditioned on,
        # whoever chose it, so there the step stays.
        world_model = teaching_player(worker, username, "jepa_pretraining", refusing)
        got = play_recorded_battle(world_model, events)
        assert len(got["labels"]) == len(got["sent"]) == decisions
    finally:
        answering.close()
        refusing.close()


@needs_recording
def test_the_worker_gives_back_a_battle_whose_room_the_server_closed(fake_helper, monkeypatch):
    """
    rlspawn.ts closes every battle room after five minutes: "|expire|", "|deinit", no "|win|".
    poke-env reports no finished battle then, and worker.py's repair loop never runs under
    poke-env 0.15 (it lives in _handle_query, which that version never calls), so the battle's
    episode slot was never given back. The player has to give it back itself.
    """
    worker = wired_worker()
    with open(worker.__file__, "r", encoding="utf-8") as fh:
        if '== ["deinit"]' not in fh.read():
            pytest.skip("worker.py does not have the requested handling of a room the server closes")
    monkeypatch.chdir(REPO)
    events = next(e for e in load_fixture() if not any(x["t"] == "order" and x["fp_reply"].get("revive") for x in e))
    username, room = events[-1]["username"], events[0]["room"]
    bridge = make_bridge(fake_helper)
    try:
        player = teaching_player(worker, username, "imitation", bridge)
        tag = "%s_%s" % (username, room)
        cut = [i for i, e in enumerate(events) if e["t"] == "order"][2]      # up to the third decision

        async def feed(messages):
            for data in messages:
                await player._handle_battle_message(split(room, data))

        run_on_player_loop(player, feed([e["data"] for e in events[:cut] if e["t"] == "msg"]))
        recorded = list(player.event_queue.queue)
        assert len([e for e in recorded if e[0] == "IMITATION"]) == 3
        assert tag in player._episode_slot_state and room in player._battles and bridge.knows(username + "|" + room)
        assert player._battle_count_queue.qsize() == 1                       # poke-env counts the battle

        run_on_player_loop(player, feed(["|expire|", "|deinit"]))            # the server closes the room
        recorded = list(player.event_queue.queue)
        assert [e[:2] for e in recorded if e[0] in ("CLEANUP", "DONE")] == [("CLEANUP", tag)]    # the slot goes back, once
        assert tag not in player._episode_slot_state and room not in player._battles
        assert player._battle_count_queue.qsize() == 0                       # the seat too: the next battle can be taken
        assert not bridge.knows(username + "|" + room)                       # and Foul Play's copy is gone
        # A line that still arrives for the room is dropped: it must not wait for a battle that is gone.
        run_on_player_loop(player, feed(["|turn|99"]), timeout=10)
        assert len(list(player.event_queue.queue)) == len(recorded)
    finally:
        bridge.close()


# ---------------------------------------------------------------------------------------------
# the real helper, when Foul Play is installed where the bridge looks for it
# ---------------------------------------------------------------------------------------------

def real_foul_play() -> bool:
    ok, _why = fb.available()
    return ok


needs_foul_play = pytest.mark.skipif(not (os.path.isfile(FIXTURE) and real_foul_play()),
                                     reason="Foul Play is not installed here (bench/install_foulplay.sh), or the recording is missing")


@needs_foul_play
def test_the_real_helper_follows_recorded_battles_and_survives_being_killed():
    bridge = FoulPlayBridge(helpers=1, search_ms=20).start()
    try:
        info = bridge.stats()["helpers"][0]
        events = load_fixture()[0]
        user = events[-1]["username"]
        room = events[0]["room"]
        key = user + "|" + room
        orders = [e for e in events if e["t"] == "order"]
        kill_at = len(orders) // 2
        seen = 0
        for event in events:
            if event["t"] == "msg":
                bridge.feed(key, user, room, event["data"])
            elif event["t"] == "order":
                if seen == kill_at:
                    kill_hard(info["pid"])                   # in the middle of the battle
                    assert wait_until(lambda: bridge.counters["helper_exits"] == 1)
                advice = bridge.advice_sync(key, rqid=event["rqid"])
                recorded = {o["choice"] for o in event["fp_reply"]["options"]}
                assert {o.choice for o in advice.options} == recorded          # the same root actions as when recorded
                assert advice.choice in recorded and advice.worlds in (2, 4) and advice.iterations > 0
                assert abs(sum(o.policy for o in advice.options) - 1.0) < 0.01
                bridge.note(key, event["choice"])
                seen += 1
        assert seen == len(orders) and bridge.counters["rebuilt_battles"] == 1
        with pytest.raises(FoulPlayBridgeError):
            bridge.advice_sync(key)                          # the battle is over: Foul Play's loop has returned
        stats = bridge.helper_stats()[0]
        assert stats["battles"] == 0 and stats["finished_battles"] == 1 and stats["failed_battles"] == 0
        assert stats["pid"] != info["pid"]
    finally:
        bridge.close()


@needs_foul_play
@pytest.mark.skipif(os.name == "nt" or bool(fb.locate(check_imports=False)[2].get("wsl")),
                    reason="a decision's own pool forks the helper: a helper on Linux (also inside WSL, but not from here)")
def test_the_real_helper_searches_a_decisions_worlds_on_its_workers_at_the_same_time():
    """
    "workers": the worlds of one decision in processes of Foul Play's own pool, forked from the
    helper: every world searched as asked (exactly N iterations each when asked for N), the same
    number of worlds as one after the other, in less time.
    """
    bridge = FoulPlayBridge(helpers=1, search_ms=20).start()
    try:
        events = load_fixture()[0]
        user, room = events[-1]["username"], events[0]["room"]
        key = user + "|" + room
        checked = 0
        for event in events:
            if event["t"] == "msg":
                bridge.feed(key, user, room, event["data"])
            elif event["t"] == "order":
                if checked < 2 and len(event["fp_reply"]["options"]) > 1:
                    fixed = bridge.advice_sync(key, rqid=event["rqid"], parallelism=2, workers=4, iterations=3000, timeout=60)
                    assert fixed.raw["workers"] == 4 and fixed.raw["parallelism"] == 2
                    assert fixed.worlds in (4, 8) and fixed.raw["iterations_per_world"] == [3000] * fixed.worlds
                    one = bridge.advice_sync(key, rqid=event["rqid"], search_ms=150, parallelism=2, timeout=60)
                    many = bridge.advice_sync(key, rqid=event["rqid"], search_ms=150, parallelism=2, workers=8, timeout=60)
                    assert "workers" not in one.raw and many.raw["workers"] == 8 and many.worlds == one.worlds
                    assert many.iterations > 0 and {o.choice for o in many.options} == {o.choice for o in one.options}
                    assert many.wall_ms < 0.6 * one.wall_ms, (many.wall_ms, one.wall_ms)     # 8 worlds at once, not one by one
                    checked += 1
                bridge.note(key, event["choice"])
        assert checked == 2 and bridge.helper_stats()[0]["failed_battles"] == 0
    finally:
        bridge.close()


@needs_foul_play
def test_the_real_helper_says_so_when_foul_play_mistakes_the_sides():
    """
    Foul Play finds the opponent's name by removing its own from the room title. When the
    opponent's name is part of the player's, it takes the player's own line for the opponent's
    and follows the battle from the wrong side. The helper must notice, not advise.
    """
    events = load_fixture()[0]
    user, room = events[-1]["username"], events[0]["room"]
    title = next(line for line in events[0]["data"].split("\n") if line.startswith("|title|"))
    opponent = title[len("|title|"):].replace(user, "").replace("vs.", "").strip()
    assert opponent and opponent != user
    short = user[:-3]                                   # the opponent is now called like the start of our name
    bridge = FoulPlayBridge(helpers=1, search_ms=20).start()
    try:
        key = user + "|" + room
        first_order = next(e for e in events if e["t"] == "order")
        for event in events:
            if event is first_order:
                break
            if event["t"] == "msg":
                bridge.feed(key, user, room, event["data"].replace(opponent, short))
        with pytest.raises(FoulPlayBridgeError, match="for both sides"):
            bridge.advice_sync(key, rqid=first_order["rqid"])
        assert bridge.helper_stats()[0]["failed_battles"] == 1
    finally:
        bridge.close()


@needs_foul_play
def test_the_real_helper_gets_through_a_battle_with_revival_blessing():
    """
    Foul Play cannot say whom to revive, and a revive taken for a switch stops its own code a turn
    later. The helper answers that one request with "no answer" and keeps the battle alive.
    """
    battles = [events for events in load_fixture() if any(e["t"] == "order" and e["fp_reply"].get("revive") for e in events)]
    assert battles, "the recording holds no battle with Revival Blessing"
    bridge = FoulPlayBridge(helpers=1, search_ms=20).start()
    try:
        for events in battles:
            user, room = events[-1]["username"], events[0]["room"]
            key = user + "|" + room
            answered = revive = 0
            for event in events:
                if event["t"] == "msg":
                    bridge.feed(key, user, room, event["data"])
                elif event["t"] == "order":
                    if event["fp_reply"].get("revive"):
                        with pytest.raises(FoulPlayBridgeError) as caught:
                            bridge.advice_sync(key, rqid=event["rqid"])
                        assert caught.value.revive is True
                        revive += 1
                    else:
                        advice = bridge.advice_sync(key, rqid=event["rqid"])      # also every decision AFTER the revive
                        assert advice.choice in {o["choice"] for o in event["fp_reply"]["options"]}
                        answered += 1
                    bridge.note(key, event["choice"])
            assert revive >= 1 and answered == len([e for e in events if e["t"] == "order"]) - revive
        stats = bridge.helper_stats()[0]
        assert stats["failed_battles"] == 0 and stats["finished_battles"] == len(battles)
        assert stats["revive_requests"] >= len(battles) and stats["notes_revive"] >= len(battles)
    finally:
        bridge.close()


@needs_foul_play
def test_the_real_helper_does_not_search_for_a_caller_that_gave_up():
    """The review's finding, on the real helper: a caller that waits 50 ms for a 200 ms search used to get it searched for nobody."""
    events = load_fixture()[0]
    user, room = events[-1]["username"], events[0]["room"]
    key = user + "|" + room
    first_order = next(e for e in events if e["t"] == "order")
    bridge = FoulPlayBridge(helpers=1, search_ms=200).start()
    try:
        assert bridge.stats()["helpers"][0]["deadlines"] is True     # the real helper reads the clock we read
        for event in events:
            if event is first_order:
                break
            if event["t"] == "msg":
                bridge.feed(key, user, room, event["data"])
        assert bridge.sync()                                         # the helper has read the battle's first lines
        with pytest.raises(FoulPlayBridgeError, match="nobody waits") as caught:
            bridge.advice_sync(key, rqid=first_order["rqid"], timeout=0.05)
        assert caught.value.expired is True
        time.sleep(0.6)                                              # where the search used to run
        stats = bridge.helper_stats()[0]
        assert stats["decisions"] == 0 and stats["expired"] == 1 and stats["decide_ms"] == 0
        advice = bridge.advice_sync(key, rqid=first_order["rqid"], timeout=30)       # a caller that waits is answered
        assert advice.iterations > 0 and bridge.helper_stats()[0]["decisions"] == 1
        assert bridge.stats()["helpers"][0]["decide_ms"] > 100
    finally:
        bridge.close()


@needs_foul_play
def test_calibrate_measures_the_real_search():
    """More milliseconds are more iterations, on this machine, now: the number a search time has to be chosen by."""
    result = fb.calibrate((10, 40), helpers=1, battles=1, target=1)
    low, high = result["rows"]
    assert result["problems"] == [] and low["decisions"] == high["decisions"] > 10
    assert high["iterations_per_decision"] > 1.5 * low["iterations_per_decision"] > 0
    assert high["decision_ms"] > low["decision_ms"] > 0 and result["search_ms"] == 10


# ---------------------------------------------------------------------------------------------
# the exact check: the helper against Foul Play's own unmodified battle loop
# ---------------------------------------------------------------------------------------------

REPLAY_FIXTURE = os.path.join(REPO, "tests", "fixtures", "foulplay_bridge_replay.jsonl.gz")
REPLAY_CHECK = os.path.join(REPO, "bench", "fp_bridge_replay.py")

needs_replay = pytest.mark.skipif(not (os.path.isfile(REPLAY_FIXTURE) and real_foul_play()) or bool(fb.locate(check_imports=False)[2].get("wsl")),
                                  reason="Foul Play is not installed here (bench/install_foulplay.sh), or the replay fixture is missing, "
                                         "or Foul Play is inside WSL: bench/fp_bridge_replay.py starts Foul Play's python itself, so run it there")


def replay_check(*arguments: str) -> dict:
    """bench/fp_bridge_replay.py check, as a child process (it starts Foul Play's python itself)."""
    _ok, _text, info = fb.locate()
    done = subprocess.run([sys.executable, REPLAY_CHECK, "check", REPLAY_FIXTURE, "--foulplay-dir", info["dir"], "--json", *arguments],
                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=900)
    assert done.returncode in (0, 1), done.stdout.decode("utf-8", "replace")[-600:] + done.stderr.decode("utf-8", "replace")[-600:]
    result = json.loads(done.stdout.decode("utf-8").strip().splitlines()[-1])
    assert result["ok"] == (done.returncode == 0)
    return result


@needs_replay
def test_the_helper_keeps_every_battle_exactly_as_foul_plays_own_loop_does():
    """
    No search, no server: recorded battles are replayed through Foul Play's unmodified battle
    loop and through the helper, and at every decision the engine state of every sampled world,
    the whole Battle object, the message Foul Play would send and last_selected_move must be
    equal. Win rates and "same move" percentages cannot see a helper that is wrong in one
    position out of fifty; this fails on one line that was not fed (the two tests below).
    """
    result = replay_check()
    assert result["first_differences"] == [] and result["problems"] == []
    assert result["ok"] is True and result["differences"] == 0
    assert result["battles"] >= 9 and result["compared"] == result["decisions"] >= 300
    assert result["worlds"] >= 2 * result["decisions"]                    # two or four sampled worlds at each one
    assert result["mixed"] == result["decisions"]                         # again with all battles interleaved in one helper
    assert result["rebuilt"] >= result["decisions"] // 6                  # and again in battles rebuilt from their stored lines
    assert result["counters"]["failed_battles"] == 0 and result["counters"]["notes_unmapped"] == 0


@needs_replay
def test_the_exact_check_notices_one_line_that_was_not_fed():
    result = replay_check("--mutate", "drop-line", "--battles", "2")
    assert len(result["mutated"]) == 1 and "not fed" in result["mutated"][0]
    assert result["ok"] is False and result["by_field"]["states"] > 0 and result["by_field"]["picture"] > 0


@needs_replay
def test_the_exact_check_notices_one_choice_that_was_noted_wrongly():
    result = replay_check("--mutate", "wrong-note", "--battles", "2")
    assert len(result["mutated"]) == 1 and "noted" in result["mutated"][0]
    assert result["ok"] is False and result["by_field"]["lsm"] > 0


# ---------------------------------------------------------------------------------------------
# the opponent's side of the search, the payoff table, exact iterations (the search plumbing)
# ---------------------------------------------------------------------------------------------

def test_the_opponents_side_of_the_search_comes_with_the_advice(fake_helper):
    bridge = make_bridge(fake_helper)
    try:
        key = "u|battle-x-opp"
        feed_request(bridge, key)
        advice = bridge.advice_sync(key)
        assert advice.opponent_active == "garchomp" and advice.opponent_revealed == ["rotomwash"]
        assert [o.name for o in advice.opponent] == ["switch rotomwash", "move earthquake", "move earthquake terastallize", "switch skarmory"]
        by_name = {o.name: o for o in advice.opponent}
        assert by_name["move earthquake"].share == 0.5 and by_name["move earthquake"].fp == "earthquake"
        assert by_name["move earthquake terastallize"].fp == "earthquake-tera" and by_name["move earthquake"].seen is None
        assert by_name["switch rotomwash"].seen is True and by_name["switch skarmory"].seen is False    # a guess of the sampling
        assert by_name["switch skarmory"].worlds == 1 and abs(sum(o.share for o in advice.opponent) - 1.0) < 1e-9
        sets = [w["active"] for w in advice.opponent_worlds]
        assert [s["item"] for s in sets] == ["choicescarf", "rockyhelmet"] and sets[0]["moves"][1] == "outrage"
        assert advice.opponent_worlds[1]["reserve"] == ["rotomwash", "toxapex"]
        assert advice.payoff is None and advice.payoff_table == []                    # nobody asked for a table
    finally:
        bridge.close()


def test_an_old_helper_without_the_opponents_side_still_gives_advice():
    reply = {"ok": True, "choice": "move 1", "best": "move 1", "options": [], "worlds": 2, "iterations": 10}
    advice = fb._advice_from(reply, 1.0, 0)
    assert advice.opponent == [] and advice.opponent_worlds == [] and advice.payoff is None and advice.iterations_asked == 0


def test_a_payoff_table_is_asked_for_and_lands_on_our_14_actions(fake_helper):
    bridge = make_bridge(fake_helper)
    try:
        key = "u|battle-x-payoff"
        feed_request(bridge, key)
        advice = bridge.advice_sync(key, payoff_ms=300, payoff="full", iterations=20000, threads=2)
        asked = advice.raw["asked"]
        assert asked["payoff_ms"] == 300 and asked["payoff"] == "full" and asked["iterations"] == 20000 and asked["threads"] == 2
        assert advice.iterations_asked == 20000
        battle = make_battle()
        fb.fill_actions(FPlayer(), battle, advice)
        # The fake's rows are its options in the order "move 1..4", the same with Tera, "switch 2..6":
        # row i is [10 i, 10 i + 1] in the first world and [10 i + 4] (move 2: pruned) in the second.
        assert advice.payoff_theirs == ["move earthquake", "switch rotomwash"] and advice.payoff_depth == [2, 3]
        table, known = advice.payoff_table, advice.payoff_known
        assert len(table) == 14 and all(len(row) == 2 for row in table)
        assert table[0] == [2.0, 1.0] and known[0] == [1.0, 0.5]          # move 1: (0 + 4) / 2 over both worlds, 1 in the one that has the column
        assert table[1] == [10.0, 11.0] and known[1] == [0.5, 0.5]        # move 2: pruned in the second world, so the first world alone
        assert table[10] == [42.0, 41.0]                                  # move 1 with Tera is the fake's row 4: (40 + 44) / 2, and 41
        assert table[5] == [82.0, 81.0]                                   # "switch 2" is the second Pokemon of the team: action 5, row 8
        assert table[4] == [None, None] and known[4] == [0.0, 0.0]        # the active Pokemon is no switch
        assert all(table[a] == [None, None] for a in range(14) if not advice.mask[a])
        world = advice.payoff_worlds[1]
        assert world["theirs"] == ["move earthquake"] and world["table"][1] == [None] and world["table"][0] == [4.0] and world["chance"] == 0.5
        plain = bridge.advice_sync(key)
        assert plain.raw["asked"]["payoff_ms"] is None and plain.payoff is None    # only when asked
    finally:
        bridge.close()


def test_a_decision_with_a_payoff_table_is_not_started_when_it_cannot_end_in_time(fake_helper):
    """The helper adds the payoff budget to what a decision takes. Here the real helper's rule, without Foul Play."""
    server = load_server_module()
    made = object.__new__(server._Server)
    made.counters = {"expired": 0, "decide_errors": 0}
    made.decide = lambda *a, **k: {"ok": True}
    now = time.monotonic()
    ok = made.handle({"op": "decide", "key": "k", "ms": 100, "deadline": now + 1.0})
    late = made.handle({"op": "decide", "key": "k", "ms": 100, "payoff_ms": 900, "deadline": now + 1.0})
    assert ok["ok"] is True and "clock" in ok
    assert late["ok"] is False and late["expired"] is True and made.counters["expired"] == 1


class _Opt:
    def __init__(self, move_choice, visits, total_score):
        self.move_choice, self.visits, self.total_score = move_choice, visits, total_score


class _Result:
    def __init__(self, side_one, side_two):
        self.side_one, self.side_two = side_one, side_two
        self.total_visits = sum(o.visits for o in side_one)


class _Mon:
    def __init__(self, name, **more):
        self.name = name
        self.moves = [type("M", (), {"name": m})() for m in more.pop("moves", [])]
        for key, value in more.items():
            setattr(self, key, value)


def _world(active, reserve):
    side = type("Side", (), {})()
    side.active, side.reserve = active, reserve
    battle = type("Battle", (), {})()
    battle.opponent = side
    return battle


def test_the_helper_names_the_opponents_options_and_averages_them_over_the_worlds():
    server = load_server_module()
    made = object.__new__(server._Server)
    name = server._Server._opponent_name
    assert [name(x) for x in ("earthquake", "earthquake-tera", "switch garchomp", "No Move", "uturn")] == [
        "move earthquake", "move earthquake terastallize", "switch garchomp", "none", "move uturn"]
    assert [server._Server._engine_argument(x) for x in ("earthquake-tera", "switch garchomp", "No Move")] == ["earthquake-tera", "garchomp", "none"]
    prepared = _world(_Mon("garchomp"), [_Mon("rotomwash")])
    results = [
        (_Result([_Opt("tackle", 1000, 600.0)], [_Opt("earthquake", 700, 280.0), _Opt("earthquake-tera", 100, 50.0), _Opt("switch rotomwash", 200, 70.0)]), 0.5, 0),
        (_Result([_Opt("tackle", 2000, 1100.0)], [_Opt("earthquake", 400, 200.0), _Opt("switch skarmory", 1600, 700.0)]), 0.5, 1),
    ]
    worlds = [(_world(_Mon("garchomp", moves=["earthquake", "outrage"], item="choicescarf", ability="roughskin", tera_type="steel",
                           hp=300, max_hp=300, level=80, status=None, terastallized=False), [_Mon("rotomwash"), _Mon("skarmory")]), None),
              (_world(_Mon("garchomp", moves=["earthquake"], item="rockyhelmet", ability="roughskin", tera_type="ground",
                           hp=300, max_hp=300, level=80, status=None, terastallized=False), [_Mon("rotomwash"), _Mon("skarmory")]), None)]
    out = made._opponent(prepared, results, worlds)
    assert out["active"] == "garchomp" and out["revealed"] == ["rotomwash"]
    by_name = {o["name"]: o for o in out["options"]}
    assert out["options"][0]["name"] == "move earthquake"                               # largest share first
    assert abs(by_name["move earthquake"]["share"] - (0.5 * 0.7 + 0.5 * 0.2)) < 1e-6    # the mean over the worlds of visits / iterations
    assert abs(by_name["switch skarmory"]["share"] - 0.4) < 1e-6 and by_name["switch skarmory"]["worlds"] == 1
    assert by_name["switch skarmory"]["seen"] is False and by_name["switch rotomwash"]["seen"] is True
    assert abs(by_name["move earthquake"]["score"] - 480.0 / 1100) < 1e-4               # from the opponent's side
    assert abs(sum(o["share"] for o in out["options"]) - 1.0) < 1e-6
    assert out["worlds"][0]["active"]["item"] == "choicescarf" and out["worlds"][0]["active"]["moves"] == ["earthquake", "outrage"]
    assert out["worlds"][1]["reserve"] == ["rotomwash", "skarmory"] and out["worlds"][1]["options"][1]["share"] == 0.8
    json.dumps(out)                                                                     # it has to travel as JSON


class _FakeEngine:
    """poke-engine as far as the payoff table uses it. States are strings; a child is "<state>|<ours>|<theirs>|<outcome>"."""

    nan = float("nan")

    class State:
        def __init__(self, text):
            self.text = text

        @classmethod
        def from_string(cls, text):
            return cls(text)

        def to_string(self):
            return self.text

        def apply_instructions(self, outcome):
            return type(self)(self.text + "|" + outcome.label)

    def __init__(self):
        self.searched = []

    def iterative_deepening_expectiminimax(self, state, ms):
        self.searched.append((state.text, ms))
        result = type("R", (), {})()
        if "|" not in state.text:                 # the root: 2 x 3, one cell pruned
            result.side_one, result.side_two, result.depth_searched = ["tackle", "switch b"], ["earthquake", "earthquake-tera", "switch x"], 3
            result.matrix = [1.0, 2.0, 3.0, 4.0, self.nan, 6.0]
            return result
        _root, ours, theirs, outcome = state.text.split("|")
        if ours == "b" and theirs == "x":
            raise BaseException("a panic in the engine")
        value = {"tackle": 10.0, "b": 20.0}[ours] + {"earthquake": 1.0, "earthquake-tera": 2.0, "x": 3.0}[theirs] + (100.0 if outcome == "crit" else 0.0)
        result.side_one, result.side_two, result.depth_searched = ["a", "b"], ["c", "d"], 1
        result.matrix = [value, value + 5.0, value - 50.0, self.nan]      # the best row by its worst known cell: value
        return result

    def generate_instructions(self, state, ours, theirs):
        def outcome(label, percentage):
            made = type("O", (), {})()
            made.label, made.percentage = "%s|%s|%s" % (ours, theirs, label), percentage
            return made
        return [outcome("hit", 75.0), outcome("crit", 25.0)]


def _payoff_server(server):
    made = object.__new__(server._Server)
    made.counters = {"payoffs": 0, "payoff_ms": 0.0, "payoff_errors": 0}
    made.poke_engine = _FakeEngine()
    made._choice_of = lambda prepared, names, name: {"tackle": "move 1", "switch b": "switch 2"}.get(name)
    return made


def test_the_helper_returns_the_engines_own_matrix_with_its_pruned_cells_as_unknown():
    server = load_server_module()
    made = _payoff_server(server)
    state = made.poke_engine.State("root")
    results = [(_Result([_Opt("tackle", 10, 5.0)], [_Opt("earthquake", 10, 5.0)]), 0.5, 0), (_Result([_Opt("tackle", 10, 5.0)], [_Opt("earthquake", 10, 5.0)]), 0.5, 1)]
    out = made._payoff(None, None, results, [(None, state), (None, state)], 200.0, "engine", lambda choice: choice)
    assert out["mode"] == "engine" and out["budget_ms"] == 200.0 and len(out["worlds"]) == 2 and made.counters["payoffs"] == 1
    world = out["worlds"][0]
    assert world["depth"] == 3 and world["choices"] == ["move 1", "switch 2"] and world["ours"] == ["tackle", "switch b"]
    assert world["theirs"] == ["move earthquake", "move earthquake terastallize", "switch x"]
    assert world["matrix"] == [[1.0, 2.0, 3.0], [4.0, None, 6.0]] and world["cells"] == 6 and world["known"] == 5
    assert [ms for _text, ms in made.poke_engine.searched][0] in (99, 100) and "theirs_fp" not in world   # the budget is shared among the worlds
    json.dumps(out)                                                                                    # no NaN on the wire


def test_the_helper_can_fill_every_cell_by_playing_each_pair_out():
    server = load_server_module()
    made = _payoff_server(server)
    state = made.poke_engine.State("root")
    result = _Result([_Opt("tackle", 900, 500.0), _Opt("switch b", 100, 40.0)],
                     [_Opt("earthquake", 600, 300.0), _Opt("earthquake-tera", 100, 60.0), _Opt("switch x", 300, 100.0)])
    out = made._payoff(None, None, [(result, 1.0, 0)], [(None, state)], 5000.0, "full", lambda choice: choice)
    world = out["worlds"][0]
    assert out["mode"] == "full" and world["depth"] == 2 and world["choices"] == ["move 1", "switch 2"]
    assert world["theirs"] == ["move earthquake", "move earthquake terastallize", "switch x"]
    # every outcome of a pair weighted by its chance: 75% the plain value, 25% the value with a crit (+100)
    assert world["matrix"] == [[36.0, 37.0, 38.0], [46.0, 47.0, None]]       # the pair whose search panicked stays unknown
    assert world["known"] == 5 and world["failed"] == 1 and world["searches"] == 10
    # out of time after the first pair: the cells that are there are the ones the search visited most, the rest is unknown
    short = _payoff_server(server)
    search = short.poke_engine.iterative_deepening_expectiminimax
    short.poke_engine.iterative_deepening_expectiminimax = lambda state, ms: (time.sleep(0.01), search(state, ms))[1]
    out = short._payoff(None, None, [(result, 1.0, 0)], [(None, state)], 1.0, "full", lambda choice: choice)
    assert out["worlds"][0]["matrix"] == [[36.0, None, None], [None, None, None]] and out["worlds"][0]["known"] == 1 and out["worlds"][0]["cells"] == 6


def test_the_value_of_a_searched_state_is_its_best_row_by_its_worst_known_cell():
    server = load_server_module()
    nan = float("nan")

    def result(n1, n2, matrix):
        made = type("R", (), {})()
        made.side_one, made.side_two, made.matrix = ["a"] * n1, ["b"] * n2, matrix
        return made

    value = server._Server._maximin
    assert value(result(2, 2, [1.0, 5.0, 3.0, 4.0])) == 3.0
    assert value(result(2, 2, [1.0, 5.0, 0.5, nan])) == 1.0          # a pruned row was already known to be worse
    assert value(result(1, 1, [nan])) is None and value(result(0, 0, [])) is None


@pytest.mark.skipif(os.name == "nt", reason="the helper's own watch on its parent is for helpers on Linux and inside WSL")
def test_a_helper_that_hears_nothing_from_its_bridge_leaves():
    """
    What ends a helper inside WSL whose wsl.exe was killed together with the Windows process: its
    stdin stays open (measured), so the bridge's line every 2 seconds is what it goes by.
    """
    script = os.path.join(new_dir("fpbridge_beat_"), "beat.py")
    write(script, "\n".join([
        "import importlib.util, sys, time",
        "spec = importlib.util.spec_from_file_location('srv', %r)" % os.path.join(REPO, "bench", "fp_bridge_server.py"),
        "srv = importlib.util.module_from_spec(spec); spec.loader.exec_module(srv)",
        "srv.BEAT_TIMEOUT_S = 1.0",
        "srv._ALIVE['last'], srv._ALIVE['busy'] = time.monotonic(), sys.argv[1] == 'busy'",
        "srv._watch_parent('beat')",
        "end = time.monotonic() + 5.0",
        "while time.monotonic() < end:",
        "    if sys.argv[1] == 'heard':",
        "        srv._ALIVE['last'] = time.monotonic()",
        "    time.sleep(0.1)",
        "sys.exit(0)",
    ]))
    began = time.monotonic()
    silent = subprocess.run([sys.executable, script, "silent"], timeout=60)
    assert silent.returncode == 113 and time.monotonic() - began < 4.5       # left by itself, long before its 5 seconds were over
    assert subprocess.run([sys.executable, script, "heard"], timeout=60).returncode == 0     # the bridge is there: it stays
    assert subprocess.run([sys.executable, script, "busy"], timeout=60).returncode == 0      # a search may take as long as it takes


# ---------------------------------------------------------------------------------------------
# Windows: the helpers run inside WSL
# ---------------------------------------------------------------------------------------------

def test_windows_paths_are_given_to_wsl_as_it_sees_them():
    assert fb.wsl_path("/home/u/foul-play") == "/home/u/foul-play" and fb.wsl_path("~/foul-play") == "~/foul-play"
    assert fb.wsl_path("\\\\wsl.localhost\\Ubuntu\\home\\u\\x y") == "/home/u/x y"
    assert fb.wsl_path("\\\\wsl$\\Ubuntu\\home\\u") == "/home/u"
    if os.name == "nt":
        assert fb.wsl_path("C:\\Users\\you\\ps ppo\\bench\\fp_bridge_server.py") == "/mnt/c/Users/you/ps ppo/bench/fp_bridge_server.py"
        assert fb.wsl_path("d:/data/x.py") == "/mnt/d/data/x.py"
        assert fb.wsl_path(str(fb.SERVER_PY)).endswith("/bench/fp_bridge_server.py") and fb.wsl_path(str(fb.SERVER_PY)).startswith("/mnt/")


def test_wsl_is_only_for_windows_and_only_for_an_installation_inside_it(monkeypatch):
    native = fake_checkout()
    monkeypatch.delenv("FOULPLAY_WSL", raising=False)
    monkeypatch.delenv("FOULPLAY_DIR", raising=False)
    if os.name != "nt":
        assert fb._wsl_wanted() is False and fb._wsl_wanted("/home/u/foul-play") is False      # Linux: unchanged, whatever is asked
        monkeypatch.setenv("FOULPLAY_WSL", "1")
        assert fb._wsl_wanted() is False
        return
    assert fb._wsl_wanted("/home/u/foul-play") is True             # a path inside WSL
    assert fb._wsl_wanted("~/nothing-of-that-name-here") is True   # no such folder on the Windows side
    assert fb._wsl_wanted(native) is False                         # a Windows folder: as before
    assert fb._wsl_wanted(os.path.join(native, "missing")) is False
    monkeypatch.setenv("FOULPLAY_DIR", native)
    assert fb._wsl_wanted() is False
    monkeypatch.setenv("FOULPLAY_WSL", "1")
    assert fb._wsl_wanted() is True
    monkeypatch.setenv("FOULPLAY_WSL", "0")
    assert fb._wsl_wanted("/home/u/foul-play") is False            # switched off: never


class _Done:
    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def test_what_is_found_inside_wsl_is_reported_like_a_native_installation(monkeypatch):
    """locate() for a Windows process, with WSL stood in for by canned answers of the probe script."""
    asked = []
    answer = {}

    def run(command, **kwargs):
        asked.append((command, kwargs))
        if isinstance(answer["out"], Exception):
            raise answer["out"]
        return _Done(answer.get("code", 0), answer["out"].encode("utf-8") if isinstance(answer["out"], str) else answer["out"], answer.get("err", b""))

    monkeypatch.setattr(fb.subprocess, "run", run)
    monkeypatch.delenv("FOULPLAY_PYTHON", raising=False)
    monkeypatch.delenv("FOULPLAY_DIR", raising=False)
    monkeypatch.setenv("FOULPLAY_WSL_DISTRO", "Ubuntu")
    good = ("first=/home/u/psppo-bench/foul-play\ndir=/home/u/foul-play\ncheckout=1\npython=/home/u/foul-play/.venv/bin/python\n"
            "sets=/home/u/foul-play/fp/data/pkmn_sets_cache/gen9randombattle.json\nsets_bytes=4000000\ncommit=%s\nimport=0\nimport_error=\n" % fb.PINNED_COMMIT)

    def located(text, check=True, **more):
        fb._LOCATE_CACHE.clear()
        answer.clear()
        answer.update(out=text, **more)
        return fb._locate_wsl(None, None, check)

    ok, message, info = located(good)
    assert ok is True and info == {"dir": "/home/u/foul-play", "python": "/home/u/foul-play/.venv/bin/python", "commit": fb.PINNED_COMMIT, "wsl": "1"}
    assert "in WSL Ubuntu" in message and fb.PINNED_COMMIT in message and "not the pinned" not in message
    command, kwargs = asked[-1]
    assert command[1:6] == ["-d", "Ubuntu", "--exec", "sh", "-s"] and command[-2:] == list(fb.WSL_DIRS)      # both places are looked in
    assert b"\r" not in kwargs["input"] and b"pkmn_sets_cache" in kwargs["input"]                            # the script goes in on stdin
    assert located(good)[0] is True and fb._locate_wsl(None, None, True)[0] is True and len(asked) == 2      # asked once per installation
    assert "not the pinned commit" in located(good.replace(fb.PINNED_COMMIT, "0123456789abcdef0123456789abcdef01234567"))[1]
    ok, message, _ = located("first=/home/u/psppo-bench/foul-play\ndir=\n")
    assert ok is False and "not installed in WSL" in message and "~/psppo-bench/foul-play or ~/foul-play" in message
    assert "not a Foul Play checkout" in located(good.replace("checkout=1\n", ""))[1]
    assert "no virtualenv" in located(good.replace("python=/home/u/foul-play/.venv/bin/python", "python="))[1]
    assert "set data" in located(good.replace("sets_bytes=4000000", "sets_bytes=12"))[1]
    broken = located(good.replace("import=0\nimport_error=", "import=1\nimport_error=ModuleNotFoundError: No module named 'poke_engine'"))
    assert broken[0] is False and "cannot import" in broken[1] and "poke_engine" in broken[1]
    assert located(good.replace("import=0", "import=1"), check=False)[0] is True                             # not asked to try the imports
    # WSL itself is not there, or says something in its own UTF-16
    assert "wsl.exe could not be run" in located(FileNotFoundError("no wsl.exe"))[1]
    complaint = "Windows Subsystem for Linux has no installed distributions.".encode("utf-16-le")
    ok, message, _ = located(b"", code=1, err=complaint)
    assert ok is False and "has no installed distributions" in message
    assert "did not answer" in located(subprocess.TimeoutExpired("wsl.exe", 180))[1]


@pytest.mark.skipif(os.name == "nt" or not os.path.exists("/bin/sh"), reason="the probe script is an sh script: it is run where there is one")
def test_the_script_that_looks_inside_wsl_finds_a_checkout_and_says_what_is_missing():
    def probe(*arguments):
        done = subprocess.run(["sh", "-s", "--", *arguments], input=fb._WSL_PROBE.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        assert done.returncode == 0, done.stderr
        return dict(line.split("=", 1) for line in done.stdout.decode("utf-8").splitlines() if "=" in line)

    good = fake_checkout()
    missing = os.path.join(new_dir("fpbridge_none_"), "nothing-here")
    seen = probe(sys.executable, "1", fb.BATTLE_FORMAT, missing, good)
    assert seen["first"] == missing and seen["dir"] == good and seen["checkout"] == "1" and seen["python"] == sys.executable
    assert int(seen["sets_bytes"]) > 10_000 and seen["commit"] == fb.PINNED_COMMIT and seen["import"] == "0"
    assert probe("", "0", fb.BATTLE_FORMAT, missing) == {"first": missing, "dir": ""}
    no_venv = probe("", "1", fb.BATTLE_FORMAT, good)
    assert no_venv["python"] == "" and "import" not in no_venv
    broken = probe(sys.executable, "1", fb.BATTLE_FORMAT, fake_checkout(importable=False, sets=False))
    assert broken["import"] != "0" and "poke_engine" in broken["import_error"] and "sets_bytes" not in broken
    write(os.path.join(good, ".git", "HEAD"), "ref: refs/heads/main\n")
    write(os.path.join(good, ".git", "packed-refs"), "# pack-refs\n0123456789abcdef0123456789abcdef01234567 refs/heads/main\n")
    assert probe(sys.executable, "0", fb.BATTLE_FORMAT, good)["commit"] == "0123456789abcdef0123456789abcdef01234567"


def test_a_helper_inside_wsl_is_started_by_wsl_exe_with_its_settings_on_the_command_line(monkeypatch):
    monkeypatch.setattr(fb, "locate", lambda *a, **k: (True, "foul-play x at /home/u/foul-play in WSL",
                                                       {"dir": "/home/u/foul-play", "python": "/home/u/foul-play/.venv/bin/python", "commit": "x", "wsl": "1"}))
    monkeypatch.setenv("FOULPLAY_WSL_DISTRO", "Ubuntu")
    bridge = FoulPlayBridge(helpers=2, pool="fork", env={"FP_BRIDGE_LOG_LEVEL": "INFO"})
    try:
        bridge._resolve_command()
        command = bridge._command
        assert bridge._wsl is True and bridge._cwd is None and bridge.install.endswith("in WSL")
        assert command[1:5] == ["-d", "Ubuntu", "--exec", "/usr/bin/env"] and "wsl" in os.path.basename(command[0]).lower()
        assert "FP_BRIDGE_LOG_LEVEL=INFO" in command and "PYTHONUNBUFFERED=1" in command        # the Windows environment does not reach it
        python = command.index("/home/u/foul-play/.venv/bin/python")
        assert command[python + 1] == "-u" and command[python + 2] == fb.wsl_path(str(fb.SERVER_PY))
        tail = command[python + 3:]
        assert tail == ["--foulplay-dir", "/home/u/foul-play", "--format", fb.BATTLE_FORMAT, "--pool", "fork", "--parent-watch", "beat", "--own-group"]
        assert bridge.counters["helper_starts"] == 0                                              # nothing was started for this
    finally:
        bridge.close()


def test_a_helper_inside_wsl_gets_deadlines_on_its_own_clock_and_a_line_that_says_the_bridge_is_there(fake_helper, monkeypatch):
    """
    A helper inside WSL reads another kernel's clock and cannot see the Windows process. So the
    bridge keeps the difference between the two clocks and gives deadlines on the helper's, and
    sends it a line every two seconds, without which the helper leaves.
    """
    monkeypatch.setattr(fb, "WSL_BEAT_S", 0.2)
    bridge = make_bridge(fake_helper, search_ms=100, env={"FAKE_SLOW_S": "0.4", "FAKE_CLOCK_OFFSET": "100000"})
    bridge._wsl = True              # as _resolve_command() sets it for an installation inside WSL
    try:
        bridge.start()
        helper = bridge._helpers[0]
        assert bridge.stats()["helpers"][0]["deadlines"] is True and bridge.stats()["helpers"][0]["wsl"] is True
        assert abs(helper.clock_offset - 100000) < 5.0
        if os.name == "nt":
            assert helper.tag.startswith(fb._WSL_TOKEN) and helper.info["argv"][-2:] == ["--tag", helper.tag]     # what it is found by inside WSL
        key = "u|battle-x-slow"
        feed_request(bridge, key)
        with pytest.raises(FoulPlayBridgeError, match="nobody waits") as caught:
            bridge.advice_sync(key, timeout=0.05)                    # no time for a search: told at once, on its clock
        assert caught.value.expired is True
        advice = bridge.advice_sync(key, timeout=20)                 # and a deadline that is far away is far away on its clock too
        assert advice.choice == "move 1" and 100000 < advice.raw["asked"]["deadline"] - time.monotonic() < 100000 + 25
        assert wait_until(lambda: (bridge.helper_stats()[0] or {}).get("beats", 0) >= 3, timeout=10)
        stats = bridge.stats()["helpers"][0]
        assert stats["helper_pid"] == helper.info["pid"] and stats["pid"] == helper.proc.pid
    finally:
        bridge.close()
    assert not pid_alive(helper.proc.pid)


def test_an_ordinary_helper_is_not_sent_the_line_meant_for_wsl(fake_helper, monkeypatch):
    monkeypatch.setattr(fb, "WSL_BEAT_S", 0.1)
    bridge = make_bridge(fake_helper)
    try:
        bridge.start()
        time.sleep(1.2)
        assert bridge.helper_stats()[0]["beats"] == 0 and bridge._helpers[0].tag == "" and bridge.stats()["helpers"][0]["wsl"] is False
        assert "--tag" not in bridge._helpers[0].info["argv"]
    finally:
        bridge.close()


def in_wsl() -> bool:
    """True on Windows when Foul Play was found inside WSL: the tests below start real helpers there."""
    if os.name != "nt":
        return False
    ok, _text, info = fb.locate()
    return bool(ok and info.get("wsl"))


needs_wsl = pytest.mark.skipif(not (os.path.isfile(FIXTURE) and in_wsl()),
                               reason="only on Windows with Foul Play installed inside WSL")


@needs_foul_play
def test_the_real_helper_gives_the_opponents_side_a_payoff_table_and_exact_iterations():
    events = load_fixture()[0]
    user, room = events[-1]["username"], events[0]["room"]
    key = user + "|" + room
    bridge = FoulPlayBridge(helpers=1, search_ms=50, timeout=120.0).start()
    try:
        seen = 0
        for event in events:
            if event["t"] == "msg":
                bridge.feed(key, user, room, event["data"])
            elif event["t"] == "order":
                seen += 1
                more = {1: {"payoff_ms": 150}, 2: {"iterations": 3000}, 3: {"payoff_ms": 600, "payoff": "full"}, 4: {"threads": 2}}.get(seen, {})
                advice = bridge.advice_sync(key, rqid=event["rqid"], **more)
                # the opponent's side: the same worlds, named options, shares that are a distribution
                assert len(advice.opponent_worlds) == advice.worlds and advice.opponent_active
                assert abs(sum(o.share for o in advice.opponent) - 1.0) < 0.01
                for option in advice.opponent:
                    assert option.name == "none" or option.name.split(" ")[0] in ("move", "switch")
                    assert (option.seen is not None) == option.name.startswith("switch ")
                for world, iterations in zip(advice.opponent_worlds, advice.iterations_per_world):
                    assert world["iterations"] == iterations and world["active"]["species"] == advice.opponent_active
                    assert 1 <= len(world["active"]["moves"]) <= 4 and world["active"]["max_hp"] > 0
                    ours = sum(o.total_score for o in advice.options)
                    assert abs(sum(o["share"] for o in world["options"]) - 1.0) < 0.01
                if seen == 2:
                    assert advice.iterations_per_world == [3000] * advice.worlds and advice.iterations_asked == 3000
                if advice.payoff:
                    assert advice.payoff["mode"] == ("full" if seen == 3 else "engine") and len(advice.payoff["worlds"]) == advice.worlds
                    for world in advice.payoff["worlds"]:
                        assert "error" not in world and world["depth"] >= 1 and world["known"] >= 1
                        assert len(world["matrix"]) == len(world["choices"]) and all(len(row) == len(world["theirs"]) for row in world["matrix"])
                        assert set(c for c in world["choices"] if c) <= {o.choice for o in advice.options}     # our side: the options of the search
                    if seen == 3:
                        assert all(w["depth"] == 2 for w in advice.payoff["worlds"])
                else:
                    assert seen not in (1, 3)
                bridge.note(key, event["choice"])
                if seen == 5:
                    break
        assert seen == 5 and bridge.counters["decision_errors"] == 0
    finally:
        bridge.close()


@needs_wsl
def test_a_windows_process_gets_real_advice_from_helpers_inside_wsl_and_leaves_none_behind():
    events = load_fixture()[0]
    user, room = events[-1]["username"], events[0]["room"]
    key = user + "|" + room
    bridge = FoulPlayBridge(helpers=2, search_ms=50, timeout=120.0).start()
    try:
        stats = bridge.stats()["helpers"]
        inside = [h["helper_pid"] for h in stats]
        assert all(h["wsl"] and h["deadlines"] and h["alive"] for h in stats) and len(set(inside)) == 2
        assert sorted(fb.wsl_leftovers()) == sorted(inside)                    # by their PIDs inside WSL
        assert bridge._helpers[0].info["pgid"] == inside[0]                    # each leads its own process group there
        asked = 0
        for event in events:
            if event["t"] == "msg":
                bridge.feed(key, user, room, event["data"])
            elif event["t"] == "order":
                advice = bridge.advice_sync(key, rqid=event["rqid"])
                assert advice.choice in {o["choice"] for o in event["fp_reply"]["options"]} and advice.iterations > 0
                assert advice.wall_ms < advice.ms + 500                         # the pipe through wsl.exe adds milliseconds, not more
                bridge.note(key, event["choice"])
                asked += 1
                if asked == 6:
                    break
        assert asked == 6 and bridge.counters["decision_errors"] == 0
    finally:
        bridge.close()
    assert fb.wsl_leftovers() == []                                            # nothing of this process is left inside WSL


@needs_wsl
@pytest.mark.parametrize("with_wsl_exe", [False, True])
def test_helpers_inside_wsl_leave_when_the_windows_process_is_killed(with_wsl_exe):
    """
    Killed outright, the Windows process closes the helpers' stdin by dying and they leave. Killed
    together with its wsl.exe processes (the whole tree), nothing closes it: they leave because
    the bridge's line no longer comes (within the helper's 20 seconds).
    """
    script = os.path.join(new_dir("fpbridge_wslparent_"), "parent.py")
    write(script, "\n".join([
        "import json, sys, time",
        "sys.path.insert(0, %r)" % REPO,
        "import foulplay_bridge as fb",
        "b = fb.FoulPlayBridge(helpers=2).start()",
        "print(json.dumps({'tag': fb._WSL_TOKEN, 'pids': [h['helper_pid'] for h in b.stats()['helpers']]}), flush=True)",
        "time.sleep(600)",
    ]))
    parent = subprocess.Popen([sys.executable, "-u", script], stdout=subprocess.PIPE, text=True)
    try:
        told = json.loads(parent.stdout.readline())
        assert len(told["pids"]) == 2 and sorted(fb.wsl_leftovers(told["tag"])) == sorted(told["pids"])
        if with_wsl_exe:
            subprocess.run(["taskkill", "/PID", str(parent.pid), "/T", "/F"], capture_output=True, timeout=60)
        else:
            kill_hard(parent.pid)
        parent.wait(timeout=30)
        assert wait_until(lambda: fb.wsl_leftovers(told["tag"]) == [], timeout=90, step=1.0)
    finally:
        if parent.poll() is None:
            parent.kill()
        if "told" in locals():
            fb._wsl_kill(told["tag"])


# ---------------------------------------------------------------------------------------------
# the options for live play (ladder.py): early_expire, idle_ping, background_spawn
# ---------------------------------------------------------------------------------------------

def test_early_expire_fails_a_question_that_cannot_be_answered_in_time_at_once(fake_helper):
    for early in (True, False):
        bridge = make_bridge(fake_helper, search_ms=100, early_expire=early, env={"FAKE_SLOW_S": "1.0"})
        try:
            bridge.start(timeout=30)                          # its clock is known: questions carry their deadline
            first, second = "u|battle-slow-1", "u|battle-slow-2"
            feed_request(bridge, first)
            feed_request(bridge, second, rqid=2)
            busy = bridge.advice_future(first, ttl=5.0)       # about 1 s in the helper
            t0 = time.monotonic()
            late = bridge.advice_future(second, ttl=0.4)      # behind it: cannot end within 0.4 s
            if early:
                assert late.done() and time.monotonic() - t0 < 0.1
                with pytest.raises(FoulPlayBridgeError) as info:
                    late.result()
                assert info.value.expired and bridge.counters["expired_early"] == 1
            else:
                assert not late.done()
                with pytest.raises(FoulPlayBridgeError) as info:
                    late.result(timeout=10)
                assert info.value.expired and bridge.counters["expired_early"] == 0   # said by the helper, at its turn
            assert busy.result(timeout=10).rqid == 1
            # A question with nothing ahead of it is asked as before.
            assert bridge.advice_future(second, ttl=5.0).result(timeout=10).rqid == 2
        finally:
            bridge.close()


@pytest.mark.skipif(os.name == "nt", reason="SIGSTOP")
def test_idle_ping_finds_a_frozen_helper_with_no_battle(fake_helper):
    bridge = make_bridge(fake_helper, hang_timeout=1.0, idle_ping=0.3, background_spawn=True)
    try:
        bridge.start(timeout=30)
        pid = bridge.stats()["helpers"][0]["pid"]
        os.kill(pid, signal.SIGSTOP)                     # frozen, with nothing asked of it
        assert wait_until(lambda: bridge.counters["hangs"] >= 1, timeout=10)
        assert wait_until(lambda: bridge.stats()["helpers"][0]["alive"] and bridge.stats()["helpers"][0]["pid"] != pid, timeout=10)
        assert not pid_alive(pid)
        feed_request(bridge, "u|battle-gen9randombattle-1", rqid=3)
        assert bridge.advice_sync("u|battle-gen9randombattle-1", timeout=10).rqid == 3
        assert bridge.counters["idle_pings"] >= 1
    finally:
        bridge.close()


@pytest.mark.skipif(os.name == "nt", reason="process groups")
def test_background_spawn_replaces_a_dead_helper_and_keeps_its_battles(fake_helper):
    bridge = make_bridge(fake_helper, background_spawn=True)
    try:
        key = "u|battle-gen9randombattle-7"
        feed_request(bridge, key, rqid=1)
        first = bridge.advice_sync(key, timeout=10).raw["pid"]
        kill_hard(first)
        assert wait_until(lambda: bridge.counters["helper_exits"] == 1, timeout=10)
        t0 = time.monotonic()
        feed_request(bridge, key, rqid=2)                # never waits for a process to be created
        assert time.monotonic() - t0 < 0.5
        again = bridge.advice_sync(key, timeout=20)
        assert again.rqid == 2 and again.raw["pid"] != first
        assert bridge.counters["rebuilt_battles"] >= 1
    finally:
        bridge.close()
        assert wait_until(lambda: all(not h["alive"] for h in bridge.stats()["helpers"]), timeout=10)


def test_restart_helpers_replaces_every_helper_and_rebuilds_its_battles(fake_helper):
    # ladder.py's health rule restarts helpers that are up but fail every question (the network
    # gone and back): each is stopped, started again, and its battles rebuilt from their lines.
    bridge = make_bridge(fake_helper, helpers=2)
    try:
        keys = ["u|battle-gen9randombattle-%d" % i for i in range(3)]
        for n, key in enumerate(keys):
            for i in range(4 + n):
                bridge.feed(key, "u", key.split("|")[1], "|turn|%d" % i)
            feed_request(bridge, key, rqid=1)
        before = {key: bridge.advice_sync(key) for key in keys}
        pids = {a.raw["pid"] for a in before.values()}
        assert len(pids) == 2
        assert bridge.restart_helpers("a test") == 2
        assert wait_until(lambda: bridge.counters["helper_exits"] == 2)
        after = {key: bridge.advice_sync(key) for key in keys}
        for key in keys:
            assert after[key].raw["feeds"] == before[key].raw["feeds"]      # every line is there again
            assert after[key].raw["pid"] not in pids
        assert bridge.counters["restarts_asked"] == 2
        for pid in pids:
            assert wait_until(lambda: not pid_alive(pid))
    finally:
        bridge.close()
    assert bridge.restart_helpers("closed") == 0


def test_a_battle_joined_again_is_resumed_from_the_turns_it_missed(fake_helper):
    # After a lost connection the server sends the whole battle again from |init|. The battle is
    # kept and fed only what came after the last turn it had: the search goes on (2026-10-15).
    bridge = make_bridge(fake_helper)
    room = "battle-gen9randombattle-7"
    key = "u|" + room
    try:
        bridge.feed(key, "u", room, "|init|battle\n|title|a vs. b")
        bridge.feed(key, "u", room, "|\n|move|p1a: X|Tackle\n|turn|1")
        bridge.feed(key, "u", room, "|\n|move|p1a: X|Tackle\n|turn|2")
        feed_request(bridge, key, rqid=3)
        before = bridge.advice_sync(key)
        replay = ("|init|battle\n|title|a vs. b\n|\n|move|p1a: X|Tackle\n|turn|1\n|\n|move|p1a: X|Tackle\n|turn|2"
                  "\n|\n|move|p1a: X|Growl\n|turn|3")
        bridge.feed(key, "u", room, replay)
        feed_request(bridge, key, rqid=5)
        after = bridge.advice_sync(key)
        assert bridge.counters["resumed"] == 1 and bridge.counters["rejoined"] == 0
        assert after.raw["feeds"] == before.raw["feeds"] + 2      # the missed turn and the request, nothing twice
        assert after.rqid == 5
        bridge.feed(key, "u", room, replay)                        # joined again with nothing missed: nothing fed
        assert bridge.counters["resumed"] == 2
    finally:
        bridge.close()


def test_a_battle_joined_again_whose_last_turn_is_not_in_the_replay_starts_over(fake_helper):
    bridge = make_bridge(fake_helper)
    room = "battle-gen9randombattle-8"
    key = "u|" + room
    try:
        bridge.feed(key, "u", room, "|init|battle\n|title|a vs. b")
        bridge.feed(key, "u", room, "|\n|turn|4")
        bridge.feed(key, "u", room, "|init|battle\n|title|a vs. b\n|\n|turn|1")
        assert bridge.counters.get("resumed", 0) == 0 and bridge.counters["rejoined"] == 1
        assert bridge.knows(key)
    finally:
        bridge.close()
