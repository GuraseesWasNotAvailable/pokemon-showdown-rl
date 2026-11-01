"""
Claude alone (claude_player.py): the request as Anthropic's Messages API takes it, its answer read
back, the rails, the launcher's wiring, and whole battles through the player with a fake API
(tests/fake_anthropic.py, which refuses what the real API refuses for these models).

    python -m pytest -q tests/test_claude.py

No real request is made and no key is needed.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import time
from dataclasses import replace

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("anthropic")

import claude_player as cp  # noqa: E402
import ladder  # noqa: E402
import luna_player as lp  # noqa: E402
import search_player as sp  # noqa: E402
from fake_anthropic import FALLBACK_BETA, FakeAnthropic, check_request  # noqa: E402

_spec = importlib.util.spec_from_file_location("tl_for_claude", os.path.join(REPO, "tests", "test_luna.py"))
tl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tl)
tsp = tl.tsp
classes = tl.classes                         # the fixture: the search player with the stub network
KEY = "sk-ant-fake-0000"
NAMES = ("fable", "opus", "sonnet", "haiku")


def game_body(legal=(0, 10, 1, 5, 7)):
    choices = [{"value": "a%d-%s" % (a, "switch-mon%d" % a if 4 <= a <= 9 else "move%d" % (a % 10) + ("-tera" if a >= 10 else "")),
                "description": "Choice %d of the position." % a} for a in legal]
    return {"model": "ignored", "input": "THE BRIEF OF THE POSITION",
            "questions": [{"type": "choice", "name": lp.QUESTION_NAME, "instructions": lp.SOLO_INSTRUCTIONS, "choices": choices}]}


def client_for(fake, name="fable", timeout=5.0, **kw):
    said = []
    kw.setdefault("say", said.append)
    spec = replace(cp.MODELS[name], url=fake.url, timeout=timeout)
    client = cp.ClaudeClient(KEY, spec, **kw)
    client.said = said
    return client


def decider_for(fake, name="fable", timeout=5.0, **kw):
    client = client_for(fake, name, timeout, **kw)
    return lp.Luna(client, lp.LunaSettings(timeout=timeout, guard=False), spec=client.spec)


# =============================================================================================
# The models and the arms
# =============================================================================================

def test_the_four_models_their_ids_prices_and_arms():
    assert tuple(cp.MODELS) == NAMES and cp.SOLO_ARMS == tuple(n + "-solo" for n in NAMES) == sp.CLAUDE_SOLO_ARMS
    ids = {n: s.model for n, s in cp.MODELS.items()}
    assert ids == {"fable": "claude-fable-5-1", "opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5", "haiku": "claude-haiku-4-5"}
    assert [(s.usd_in, s.usd_cached, s.usd_out) for s in cp.MODELS.values()] == [(10.0, 0.25, 50.0), (4.0, 0.20, 20.0), (2.0, 0.20, 10.0), (1.0, 0.10, 5.0)]
    assert all(s.api == "anthropic" and s.url == "https://api.anthropic.com" and s.timeout == cp.DECISION_SECONDS == 25.0 for s in cp.MODELS.values())
    assert [s.effort for s in cp.MODELS.values()] == ["low", "low", "low", ""]            # Haiku 4.5 is not sent one
    for arm, name in zip(cp.SOLO_ARMS, NAMES):
        assert arm in sp.MODES and arm in sp.SOLO_ARMS and arm in ladder.PLAYER_MODES and sp.DECIDERS[arm] == name
        assert sp.split_mode(arm) == ("net", None)                                        # no search helpers are started
        args = ladder.parse_args(["--player", arm])
        assert ladder.decider_models(args) == [name] and ladder.wants_luna(args) and ladder.search_problems(args) == []
        assert ladder.model_label(name) == ladder.model_label(arm) == ladder.model_label(cp.MODELS[name].model) == cp.MODELS[name].label
    assert ladder.CLAUDE_LABELS == {n: s.label for n, s in cp.MODELS.items()} and ladder.CLAUDE_MODELS == NAMES
    # the older names still mean what they meant ("sonnet-solo" and "opus-solo" have "sol" in them)
    assert [ladder.model_label(x) for x in ("sol", "sol-solo", "gpt-6-sol", "luna", "luna-solo", None)] == ["Sol", "Sol", "Sol", "Luna", "Luna", "Luna"]
    assert sp.SOLO_ARMS[:2] == ("luna-solo", "sol-solo") and sp.DECIDERS["sol-solo"] == "sol"


def test_the_key_goes_to_anthropic_or_to_this_machine_only():
    assert cp.endpoint_problem("https://api.anthropic.com") == "" and cp.endpoint_problem("http://127.0.0.1:8123") == ""
    for bad in ("http://api.anthropic.com", "https://api.openai.com/v1", "https://anthropic.com.evil.example", "https://example.com"):
        assert "not to" in cp.endpoint_problem(bad)
    with pytest.raises(lp.LunaError) as e:
        cp.ClaudeClient(KEY, replace(cp.MODELS["opus"], url="https://example.com"))
    assert e.value.kind == "endpoint"
    with pytest.raises(lp.LunaError):
        cp.ClaudeClient("", cp.MODELS["opus"])
    assert cp.read_key(environ={cp.KEY_ENV: "  sk-ant-abc123456  "}) == "sk-ant-abc123456"
    with pytest.raises(lp.LunaError):
        cp.read_key(environ={}, ask=lambda prompt: "")
    with pytest.raises(lp.LunaError):
        cp.read_key(environ={cp.KEY_ENV: "sk-ant with space"})
    assert "sk-ant-abc123456" not in lp.redact("the key sk-ant-abc123456 was refused") and KEY not in repr(cp.ClaudeClient(KEY, cp.MODELS["haiku"]))


# =============================================================================================
# The request
# =============================================================================================

@pytest.mark.parametrize("name", NAMES)
def test_the_request_is_one_the_api_takes_for_that_model(name):
    spec = cp.MODELS[name]
    request, maps = cp.messages_request(game_body(), spec, spec.effort)
    assert check_request(request, "/v1/messages", "") == []
    assert request["model"] == spec.model and request["max_tokens"] == cp.MAX_TOKENS
    assert "thinking" not in request and "tool_choice" not in request and "tools" not in request and "temperature" not in request
    (system,) = request["system"]
    assert system["text"] == lp.SOLO_INSTRUCTIONS + "\n\n" + cp.FORMAT_NOTE and system["cache_control"] == {"type": "ephemeral"}
    (message,) = request["messages"]
    assert message["role"] == "user" and message["content"].startswith("THE BRIEF OF THE POSITION\n\nQUESTION \"action\"\nCHOICES:\n- a0: Choice 0")
    assert [line.split(":")[0] for line in message["content"].split("CHOICES:\n")[1].splitlines()] == ["- a0", "- a10", "- a1", "- a5", "- a7"]
    assert ("effort" in request["output_config"]) == bool(spec.effort) and request["output_config"].get("effort", "low") == "low"
    schema = request["output_config"]["format"]["schema"]
    assert request["output_config"]["format"]["type"] == "json_schema" and schema["required"] == ["action"]
    assert schema["properties"]["action"]["properties"]["choice"]["enum"] == ["a%d" % i for i in range(14)]
    assert maps == {"action": {"a0": "a0-move0", "a10": "a10-move0-tera", "a1": "a1-move1", "a5": "a5-switch-mon5", "a7": "a7-switch-mon7"}}


def test_a_games_schema_is_the_same_at_every_move_and_a_checks_is_its_own():
    spec = cp.MODELS["opus"]
    one, _ = cp.messages_request(game_body((0, 1, 2, 3)), spec, "low")
    two, _ = cp.messages_request(game_body((4, 5, 6)), spec, "low")
    assert one["output_config"] == two["output_config"] and one["system"] == two["system"]       # nothing a new move recompiles or re-caches
    check, maps = cp.messages_request(lp.check_body(spec.model), spec, "low")
    assert check_request(check, "/v1/messages", "") == []
    props = check["output_config"]["format"]["schema"]["properties"]
    assert props["sum"]["properties"]["choice"]["enum"] == ["three", "four", "five"] and props["two_plus_two"]["required"] == ["probability"]
    assert maps == {"sum": {"three": "three", "four": "four", "five": "five"}} and check["system"][0]["text"] == cp.FORMAT_NOTE
    sample, maps = cp.messages_request(lp.sample_body(spec.model), spec, "low")
    assert sample["output_config"]["format"]["schema"]["properties"]["sample"]["properties"]["choice"]["enum"] == cp.ACTION_IDS
    assert maps["sample"]["a10"] == "a10-sludgebomb-tera"
    # values that are not a game's (two with one action id, an id past the last action) are offered as they are
    odd = game_body((0, 1))
    odd["questions"][0]["choices"][1]["value"] = "a0-other"
    assert cp.messages_request(odd, spec, "low")[0]["output_config"]["format"]["schema"]["properties"]["action"]["properties"]["choice"]["enum"] == ["a0-move0", "a0-other"]


def test_the_fake_refuses_what_the_api_refuses():
    spec = cp.MODELS["fable"]
    good, _ = cp.messages_request(game_body(), spec, "low")
    for change, word in (({"thinking": {"type": "disabled"}}, "thinking"), ({"thinking": {"type": "enabled", "budget_tokens": 2000}}, "thinking"),
                         ({"tool_choice": {"type": "tool", "name": "x"}}, "tool_choice"), ({"temperature": 0.2}, "temperature"),
                         ({"fallbacks": [{"model": "claude-opus-5"}]}, "fallbacks")):
        assert any(word in p for p in check_request(dict(good, **change), "/v1/messages", "")), change
    assert check_request(dict(good, fallbacks=[{"model": "claude-opus-5"}]), "/v1/messages?beta=true", FALLBACK_BETA) == []
    assert check_request(dict(good, thinking={"type": "adaptive"}), "/v1/messages", "") == []
    haiku, _ = cp.messages_request(game_body(), cp.MODELS["haiku"], "low")
    assert any("effort" in p for p in check_request(haiku, "/v1/messages", ""))
    loose = json.loads(json.dumps(good))
    del loose["output_config"]["format"]["schema"]["properties"]["action"]["additionalProperties"]
    assert any("additionalProperties" in p for p in check_request(loose, "/v1/messages", ""))


# =============================================================================================
# The client against the fake
# =============================================================================================

@pytest.mark.parametrize("name", NAMES)
def test_an_answer_comes_back_as_the_value_we_offered_with_what_it_cost(name):
    with FakeAnthropic(key=KEY, mode="last", cache_read=200, cache_write=100) as fake:
        client = client_for(fake, name)
        reply = client.ask_sync(game_body(), 5.0)
        client.close()
    (record,) = fake.requests
    assert record["status"] == 200 and record["problems"] == [] and record["key"] == "ok" and record["version"]
    spec = cp.MODELS[name]
    body = record["body"]
    assert body["model"] == spec.model and "thinking" not in body
    assert ("fallbacks" in body) == (name == "fable") and (FALLBACK_BETA in record["beta"]) == (name == "fable")
    assert ("beta=true" in record["path"]) == (name == "fable")
    assert reply.answers == [{"type": "choice", "name": "action", "choice": "a7-switch-mon7", "confidence": 0.8, "probabilities": [],
                              "reason": "The fake's reason for a7."}]
    parsed = lp.parse_answer(reply.answers, lp.QUESTION_NAME, {"a0-move0": 0, "a10-move0-tera": 10, "a1-move1": 1, "a5-switch-mon5": 5, "a7-switch-mon7": 7})
    assert parsed["kind"] == "choice" and parsed["action"] == 7 and parsed["reason"] == "The fake's reason for a7."
    assert reply.cached_tokens == 200 and reply.input_tokens >= 300 and reply.output_tokens > 0 and reply.request_id and reply.served_by is None
    fresh = reply.input_tokens - 300
    want = (fresh * spec.usd_in + 200 * spec.usd_cached + 100 * spec.usd_in * 1.25 + reply.output_tokens * spec.usd_out) / 1e6
    assert reply.cost_usd == pytest.approx(want) and reply.cost_usd > 0
    assert KEY not in json.dumps(fake.requests)


def test_what_the_decider_records_is_the_models_own_cost_and_name():
    import asyncio

    with FakeAnthropic(key=KEY, mode="first") as fake:
        fable = decider_for(fake, "fable")
        built = tl.toy_built(3, 1)
        facts = tl.toy_facts(3)
        facts.solo = True
        action, info = asyncio.run(fable.choose(built, facts, 5.0))
        fable.stamp(info)
        fable.client.close()
    assert info["asked"] and not info["fell_back"] and info["who"] == "fable" and info["model"] == "claude-fable-5-1"
    assert action == info["choice"] == next(iter(built.values.values())) and info["reason"].startswith("The fake's reason")
    usd, _ = cp.request_cost_usd(cp.MODELS["fable"], type("U", (), fake.requests[0] and {
        "input_tokens": info["input_tokens"], "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "output_tokens": info["output_tokens"]})())
    assert info["cost_usd"] == pytest.approx(usd, abs=1e-7) and info["cost_usd"] > 0.001      # Fable's prices, not Luna's
    assert fable.label == "Claude Fable 5.1"


def test_a_refused_fallback_form_is_dropped_once_and_the_move_is_still_answered():
    with FakeAnthropic(key=KEY, mode="first", refuse_fallbacks=True) as fake:
        client = client_for(fake, "fable")
        first = client.ask_sync(game_body(), 5.0)
        second = client.ask_sync(game_body(), 5.0)
        client.close()
    assert [r["status"] for r in fake.requests] == [400, 200, 200] and first.answers[0]["choice"] == second.answers[0]["choice"] == "a0-move0"
    assert "fallbacks" in fake.requests[0]["body"] and all("fallbacks" not in r["body"] and "beta=true" not in r["path"] for r in fake.requests[1:])
    assert client.fallback_model is None and client.fallback_refused and len(client.said) == 1 and "refusal fallback" in client.said[0]
    with FakeAnthropic(key=KEY, mode="served_by_fallback") as fake:               # the first model declined, the fallback answered
        client = client_for(fake, "fable")
        reply = client.ask_sync(game_body(), 5.0)
        client.close()
    assert reply.served_by == "claude-opus-5" and reply.answers[0]["choice"] == "a0-move0"
    with FakeAnthropic(key=KEY) as fake:                                           # and it can be left out from the start
        client = client_for(fake, "fable", refusal_fallback=False)
        client.ask_sync(game_body(), 5.0)
        client.close()
    assert "fallbacks" not in fake.requests[0]["body"]


def test_every_failure_is_an_error_the_rails_know():
    values = {"a0-move0": 0, "a10-move0-tera": 10, "a1-move1": 1, "a5-switch-mon5": 5, "a7-switch-mon7": 7}
    with FakeAnthropic(key=KEY, hang_s=1.5) as fake:
        client = client_for(fake, "opus")
        fake.script.extend(["refusal", "illegal", "cut", "not_json", "http_429", "http_500", "first", "http_529", "http_529", "hang"])
        refused = client.ask_sync(game_body(), 5.0)
        assert lp.parse_answer(refused.answers, "action", values)["kind"] == "refusal"
        illegal = client.ask_sync(game_body(), 5.0)
        assert illegal.answers[0]["choice"] == "a2" and lp.parse_answer(illegal.answers, "action", values)["kind"] == "illegal"
        for kind, word in (("malformed", "cut off at"), ("malformed", "not JSON")):
            with pytest.raises(lp.LunaError) as e:
                client.ask_sync(game_body(), 5.0)
            assert e.value.kind == kind and word in e.value.text
        with pytest.raises(lp.LunaError) as e:                                 # a 429 is not asked again
            client.ask_sync(game_body(), 5.0)
        assert e.value.reason == "http_429" and e.value.attempts == 1
        again = client.ask_sync(game_body(), 5.0)                              # a 500 is, once
        assert again.attempts == 2 and again.answers[0]["choice"] == "a0-move0"
        with pytest.raises(lp.LunaError) as e:                                 # twice overloaded: the error
            client.ask_sync(game_body(), 5.0)
        assert e.value.reason == "http_529" and e.value.attempts == 2
        began = time.monotonic()
        with pytest.raises(lp.LunaError) as e:
            client.ask_sync(game_body(), 0.4)
        assert e.value.kind == "timeout" and time.monotonic() - began < 1.2
        client.close()
    with FakeAnthropic(key="sk-ant-another-key") as fake:                      # the key is refused: off for the run, said once
        client = client_for(fake, "sonnet")
        for _ in range(2):
            with pytest.raises(lp.LunaError) as e:
                client.ask_sync(game_body(), 5.0)
            client.failed(e.value)
        assert e.value.reason == "http_401" and client.available() == (False, "key_refused") and len(client.said) == 1
        assert KEY not in client.said[0] and KEY not in e.value.text
        client.close()
    with FakeAnthropic(key=KEY, hang_s=0.6, mode="hang") as fake:              # slow answers do not pause the model
        client = client_for(fake, "haiku")
        for _ in range(6):
            with pytest.raises(lp.LunaError) as e:
                client.ask_sync(game_body(), 0.1)
            client.failed(e.value)
        assert client.available() == (True, "") and client.pauses == 0
        client.close()


# =============================================================================================
# The launcher
# =============================================================================================

def test_the_launchers_flags_and_keys(monkeypatch):
    args = ladder.parse_args(["--player", "opus-solo"])
    assert (args.claude_effort, args.claude_timeout, args.claude_url) == ("low", 25.0, "https://api.anthropic.com")
    assert ladder.model_timeout(args, "opus") == 25.0 and ladder.model_timeout(ladder.parse_args(["--claude-timeout", "12"]), "fable") == 12.0
    with pytest.raises(SystemExit):
        ladder.parse_args(["--claude-timeout", "0.2"])
    with pytest.raises(SystemExit):
        ladder.parse_args(["--claude-effort", "max"])
    # the key: the environment variable, taken out of the environment at once
    monkeypatch.setenv(cp.KEY_ENV, KEY)
    monkeypatch.delenv(lp.KEY_ENV, raising=False)
    assert ladder.model_keys(args, ["opus"]) == {"anthropic": KEY} and cp.KEY_ENV not in os.environ
    # under --key-stdin the next line is the key of the one company the run needs
    args = ladder.parse_args(["--player", "fable-solo", "--key-stdin"])
    monkeypatch.setattr(sys, "stdin", type("In", (), {"buffer": io.BytesIO((KEY + "\n").encode())})())
    assert ladder.model_keys(args, ["fable"]) == {"anthropic": KEY}
    monkeypatch.setattr(sys, "stdin", type("In", (), {"buffer": io.BytesIO(b"\n")})())
    with pytest.raises(lp.LunaError):                                          # an empty line and no variable: no key, no prompt
        ladder.model_keys(args, ["fable"])
    with FakeAnthropic(key=KEY) as fake:
        args = ladder.parse_args(["--player", "haiku-solo", "--claude-url", fake.url, "--claude-effort", "medium", "--claude-timeout", "7"])
        haiku = ladder.model_client(args, "haiku", KEY)
        assert isinstance(haiku, cp.ClaudeClient) and haiku.effort == "" and haiku.spec.timeout == 7.0 and haiku.url == fake.url
        opus = ladder.model_client(args, "opus", KEY)
        assert opus.effort == "medium" and opus.model == "claude-opus-5-5"
        text = ladder.claude_settings_text(args, "opus", opus)
        assert "Claude Opus 5.5" in text and "effort medium" in text and "up to 7 s" in text and KEY not in text
        assert "no key was given" in ladder.claude_settings_text(args, "fable", None)
        haiku.close()
        opus.close()


@pytest.mark.parametrize("name", ["fable", "haiku"])
def test_the_check_before_a_run_asks_the_real_questions(name, monkeypatch, capsys):
    with FakeAnthropic(key=KEY) as fake:
        monkeypatch.setenv(cp.KEY_ENV, KEY)
        args = ladder.parse_args(["--luna-check", "--model", name, "--claude-url", fake.url])
        assert ladder.luna_check_only(args) == 0
        out = capsys.readouterr().out
        assert cp.MODELS[name].label + " check OK" in out and KEY not in out and "about $" in out
        assert [r["status"] for r in fake.requests] == [200, 200, 200] and all(r["problems"] == [] for r in fake.requests)
        monkeypatch.setenv(cp.KEY_ENV, "sk-ant-not-the-key")
        assert ladder.luna_check_only(args) == ladder.LUNA_FAILED
        out = capsys.readouterr().out
        assert "CHECK FAILED" in out and "HTTP 401" in out and "sk-ant-not-the-key" not in out


# =============================================================================================
# Whole battles
# =============================================================================================

def play_bridge(classes, battles, mode, deciders, **kw):
    bridge = tsp.FakeBridge(battles)
    notes, players = [], []
    try:
        for events in battles:
            mine = []
            player = tsp.searcher(classes, events[-1]["username"], bridge, mode=mode, deadline=1.5, on_decision=mine.append, deciders=deciders, **kw)
            tsp.play(player, events)
            player.notes = mine
            players.append(player)
            notes += mine
    finally:
        bridge.close()
    return notes, players, bridge


@tsp.needs_player
@pytest.mark.parametrize("name", ["fable", "haiku"])
def test_claude_alone_plays_its_own_moves_and_no_search_is_asked(classes, name):
    battles = tsp.load_fixture()[:2]
    with FakeAnthropic(key=KEY, mode="last") as fake:
        who = decider_for(fake, name, timeout=4.0)
        notes, players, bridge = play_bridge(classes, battles, name + "-solo", {name: who})
        who.client.close()
    assert bridge.asked == []                                                  # the search was never asked
    choice = [n for n in notes if n["legal"] > 1]
    assert choice and len(fake.requests) == len(choice) and all(r["status"] == 200 and r["problems"] == [] for r in fake.requests)
    for record, n in zip(fake.requests, choice):
        lu = n["luna"]
        body = record["body"]
        assert body["system"][0]["text"].startswith(lp.SOLO_INSTRUCTIONS[:80]) and body["model"] == cp.MODELS[name].model
        offered = [int(x[1:]) for x in __import__("re").findall(r"^- (a\d+):", body["messages"][0]["content"], __import__("re").M)]
        assert sorted(offered) == [a for a in range(14) if n["mask"][a]]      # every legal action is offered, and nothing else
        text = body["messages"][0]["content"]
        for word in ("MIX'S PICK", "ADVISORS", "The network alone", "win estimate"):
            assert word not in text
        assert lu["solo"] and lu["asked"] and not lu["fell_back"] and lu["who"] == name and n["source"] == name
        assert n["played"] == lu["choice"] == offered[-1] and lu["cost_usd"] > 0
    assert any(n["played"] != n["net"] for n in choice)                        # the network's pick is written down and not played
    for player, events in zip(players, battles):
        s = player.game_summary(events[0]["room"])
        mine = [n for n in player.notes if n["legal"] > 1]
        assert s["arm"] == name + "-solo" and s["luna_solo"] is True and s["luna_played"] == len(mine) and s["searched"] == 0
        assert cp.MODELS[name].label + " alone: asked on %d of %d" % (len(mine), len(mine)) in player.search_summary()
    assert KEY not in json.dumps(notes)


@tsp.needs_player
@pytest.mark.parametrize("behaviour, why", [("http_500", "http_500"), ("hang", "timeout"), ("illegal", "illegal"), ("refusal", "refusal")])
def test_when_claude_gives_no_move_the_rule_plays_not_the_network(classes, behaviour, why):
    battles = tsp.load_fixture()[:1]
    with FakeAnthropic(key=KEY, mode=behaviour, hang_s=1.0) as fake:
        who = decider_for(fake, "opus", timeout=0.4)
        who.client.breaker_failures = 10 ** 6
        who.client.retries = 0
        notes, players, bridge = play_bridge(classes, battles, "opus-solo", {"opus": who})
        who.client.close()
    assert bridge.asked == []
    choice = [n for n in notes if n["legal"] > 1]
    assert choice and all(n["luna"]["fell_back"] and n["luna"]["why"] == why and n["luna"]["solo"] for n in choice)
    assert all(n["source"] == "rule" and n["played"] == n["luna"]["mix"] and n["mask"][n["played"]] == 1 and n["played"] < 10 for n in choice)
    assert all(n["ms"] < 2500 for n in choice)
    assert "the fallback rule played instead on %d" % len(choice) in players[0].search_summary()


def test_the_window_offers_the_models_and_only_what_is_useful():
    source = open(os.path.join(REPO, "app", "PSBotLookahead.cs"), encoding="utf-8").read()
    for text in ('static readonly string[] ModeKeys = { "lookahead", "net", "fable", "opus", "sonnet", "haiku", "solalone", "lunaalone" };',
                 '"Claude Fable 5.1 alone"', '"Claude Opus 5.5 alone"', '"Claude Sonnet 5 alone"', '"Claude Haiku 4.5 alone"',
                 'if (IsClaude(mode)) return mode + "-solo";', 'if (mode == "solalone") return "sol-solo";', 'if (mode == "lunaalone") return "luna-solo";',
                 '"ANTHROPIC_API_KEY"', '"OPENAI_API_KEY"', 'args.Add("--claude-effort");', 'allNetsBox'):
        assert text in source, text
    for gone in ('"Luna decides', '"Sol decides', 'Read ahead 5 turns before each move'):
        assert gone not in source, gone
