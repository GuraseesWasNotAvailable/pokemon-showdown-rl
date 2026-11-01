"""
The Luna arm (luna_player.py, the arm "luna" of search_player.py and ladder.py, the launcher).

    python -m pytest tests/test_luna.py -q -p no:cacheprovider

No key of OpenAI's and no call to OpenAI: every request goes to tests/fake_decisions.py, a fake
of the Decisions API on 127.0.0.1 that holds the request to the documented shape and answers as
it is told (a normal answer, a slow one, HTTP 401/429/500, a refusal, an answer of the wrong
shape, a choice that was not offered). The key in these tests is the fake's own.

The player tests replay the recorded battles of tests/fixtures/foulplay_bridge_trace.jsonl.gz
through the real player class with the fake search of tests/test_search_player.py (the answers
the real helper gave) and its stub network. The brief tests replay ten real ladder games of
tests/fixtures/luna_ladder_games.jsonl.gz (bench/luna_fixture.py made it from the battle files:
what the room sent our player, and each decision's note) and write the brief of every decision.
"""

from __future__ import annotations

import asyncio
import gzip
import importlib.util
import json
import os
import re
import sys
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (REPO, os.path.join(REPO, "bench"), os.path.join(REPO, "tests")):
    if path not in sys.path:
        sys.path.insert(0, path)

import ladder  # noqa: E402
import luna_player as lp  # noqa: E402
import search_player as sp  # noqa: E402
import ab_report  # noqa: E402
import luna_audit  # noqa: E402
import luna_report  # noqa: E402
from fake_decisions import MIX_MARK, FakeDecisions, check_request  # noqa: E402

_spec = importlib.util.spec_from_file_location("tsp_for_luna", os.path.join(REPO, "tests", "test_search_player.py"))
tsp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tsp)

FAKE_KEY = "sk-fake-luna-test-9f3b2c7d1e55"         # the fake server's own key: not a key of OpenAI's
LADDER_GAMES = os.path.join(REPO, "tests", "fixtures", "luna_ladder_games.jsonl.gz")
LAUNCHER = os.path.join(REPO, "launchers", "laptop", "Luna Play.cmd")
STOPPER = os.path.join(REPO, "launchers", "laptop", "Luna Stop.cmd")


def client_for(fake, key=FAKE_KEY, **kw):
    said = []
    kw.setdefault("say", said.append)
    client = lp.LunaClient(key, url=fake.url, **kw)
    client.said = said
    return client


def toy_built(n=3, mix=0):
    """A request with n choices a0..a(n-1), the mix one marked as the brief marks it."""
    choices = [{"value": "a%d-toy" % a, "description": "Choice %d. net 10%%.%s" % (a, " " + MIX_MARK + "." if a == mix else "")}
               for a in range(n)]
    question = {"type": "choice", "name": lp.QUESTION_NAME, "instructions": "Pick one.", "choices": choices}
    body = {"model": lp.DEFAULT_MODEL, "input": "A toy position.", "questions": [question]}
    return lp.Built(body=body, values={c["value"]: a for a, c in enumerate(choices)}, brief=body["input"], question=question,
                    tokens=lp.request_tokens(body))


def toy_facts(n=3, mix=0, probs=None, shares=None):
    mask = [1.0 if a < n else 0.0 for a in range(14)]
    probs = list(probs) if probs is not None else [1.0 / n if a < n else 0.0 for a in range(14)]
    return lp.Facts(mask=mask, actions={}, probs=probs + [0.0] * (14 - len(probs)), net_action=mix, mix_action=mix,
                    shares=None if shares is None else list(shares) + [0.0] * (14 - len(shares)))


# =============================================================================================
# The key never shows
# =============================================================================================

def test_redact_takes_out_the_key_a_bearer_token_and_an_authorization_header():
    lp.remember_secret(FAKE_KEY)
    assert FAKE_KEY not in lp.redact("the key %s was refused" % FAKE_KEY)
    assert lp.redact("Authorization: Bearer abc.DEF-123_xyz") == "Authorization: ***"
    assert "abcDEF123456" not in lp.redact("sent with bearer abcDEF123456 today")
    assert lp.redact("Incorrect API key provided: sk-ab12*********************wxyz. You can") == "Incorrect API key provided: sk-***. You can"
    assert lp.redact({"headers": {"Authorization": "Bearer sk-proj-AAAA1111bbbb2222"}}).count("sk-proj") == 0
    assert lp.redact("nothing secret here") == "nothing secret here"


def test_the_key_comes_from_the_environment_else_from_a_hidden_prompt():
    asked = []
    assert lp.read_key({"OPENAI_API_KEY": "  " + FAKE_KEY + "\n"}, ask=lambda prompt: asked.append(prompt) or "x") == FAKE_KEY
    assert asked == []
    assert lp.read_key({}, ask=lambda prompt: asked.append(prompt) or FAKE_KEY + " ") == FAKE_KEY
    assert len(asked) == 1 and "not shown" in asked[0] and FAKE_KEY not in asked[0]
    with pytest.raises(lp.LunaError) as none:
        lp.read_key({}, ask=lambda prompt: "")
    assert none.value.kind == "key" and "OPENAI_API_KEY" in none.value.text
    with pytest.raises(lp.LunaError) as broken:
        lp.read_key({"OPENAI_API_KEY": "sk-two words-0123456789"})
    assert broken.value.kind == "key" and "two words" not in broken.value.text


def test_the_key_goes_to_openai_or_to_this_machine_only():
    assert lp.endpoint_problem(lp.DEFAULT_URL) == ""
    assert lp.endpoint_problem("http://127.0.0.1:11450/v1/decisions") == "" and lp.endpoint_problem("http://localhost:1/x") == ""
    for url in ("https://api.openai.com.evil.example/v1/decisions", "http://api.openai.com/v1/decisions", "https://example.com/v1/decisions",
                "http://10.0.0.5:8000/v1/decisions", "ftp://127.0.0.1/x", "nonsense"):
        assert lp.endpoint_problem(url), url
        with pytest.raises(lp.LunaError) as refused:
            lp.LunaClient(FAKE_KEY, url=url)
        assert refused.value.kind == "endpoint" and FAKE_KEY not in refused.value.text
    client = lp.LunaClient(FAKE_KEY)
    assert client.url == "https://api.openai.com/v1/decisions" and client.model == "gpt-6-luna"
    assert FAKE_KEY not in repr(client) and FAKE_KEY not in str(vars(client).keys())
    client.close()


def test_console_log_hides_the_api_key_in_the_file_and_in_the_window(tmp_path):
    import io

    log = ladder.ConsoleLog(str(tmp_path), secret="hunter2pass", more=[FAKE_KEY])
    window = io.StringIO()
    tee = log.wrap(window, "out")
    tee.write("a line with %s in it\n" % FAKE_KEY)
    tee.write("the password hunter2pass\n")
    tee.write("an ordinary line\n")
    written = open(log.path(), encoding="utf-8").read()
    assert FAKE_KEY not in written and FAKE_KEY not in window.getvalue() and "***" in window.getvalue()
    assert "hunter2pass" not in written and "an ordinary line" in written and "an ordinary line" in window.getvalue()
    plain = ladder.ConsoleLog(str(tmp_path / "plain"))          # as before, without further secrets
    window2 = io.StringIO()
    plain.wrap(window2, "out").write("as it was %s\n" % FAKE_KEY)
    assert FAKE_KEY in window2.getvalue()


# =============================================================================================
# Tokens
# =============================================================================================

def test_the_token_estimate_counts_long_names_numbers_and_signs():
    assert lp.estimate_tokens("") == 0
    assert lp.estimate_tokens("the cat sat") <= 5
    assert lp.estimate_tokens("Corviknight") >= 3                       # a name is several tokens
    assert lp.estimate_tokens("12345678") >= 3
    assert lp.estimate_tokens("a, b; c: d.") >= 8
    text = "Earthquake: Ground physical, power 100, accuracy 100%. Est. 45-53% of the foe's HP (super effective)."
    assert 20 <= lp.estimate_tokens(text) <= 60
    body = toy_built(4).body
    assert lp.request_tokens(body) > lp.estimate_tokens(body["input"]) + 4 * 4


# =============================================================================================
# The request on the wire, and what comes back
# =============================================================================================

def test_the_request_is_the_documented_one():
    with FakeDecisions(key=FAKE_KEY, mode="mix") as fake:
        client = client_for(fake)
        built = toy_built(4, mix=2)
        reply = client.ask_sync(built.body, 5.0)
        client.close()
    assert len(fake.requests) == 1
    r = fake.requests[0]
    assert r["method"] == "POST" and r["path"] == "/v1/decisions" and r["problems"] == [] and r["status"] == 200
    assert r["authorization"] == "ok" and r["content_type"] == "application/json"
    assert set(r["body"]) == {"model", "input", "questions"} and r["body"]["model"] == "gpt-6-luna"
    assert isinstance(r["body"]["input"], str)
    (question,) = r["body"]["questions"]
    assert set(question) == {"type", "name", "instructions", "choices"} and question["type"] == "choice"
    assert all(set(c) == {"value", "description"} for c in question["choices"])
    assert r["body"] == built.body                                     # nothing added, nothing lost on the way
    assert reply.status == 200 and reply.attempts == 1 and reply.latency_ms >= 0 and reply.input_tokens is None
    parsed = lp.parse_answer(reply.answers, lp.QUESTION_NAME, built.values)
    assert parsed["kind"] == "choice" and parsed["action"] == 2 and parsed["value"] == "a2-toy"
    assert parsed["confidence"] == 0.85 and parsed["probabilities"][2] == 0.9 and set(parsed["probabilities"]) == {0, 1, 2, 3}


def test_the_fake_holds_requests_to_the_documented_shape():
    good = toy_built(3).body
    assert check_request(good) == []
    assert check_request(dict(good, temperature=0)) == ["unknown fields: temperature"]
    assert check_request({"model": "gpt-6-luna", "input": "x"}) == ["missing questions"]
    two = dict(good, questions=good["questions"] * 2)
    assert any("two questions named" in p for p in check_request(two))
    bad_choice = json.loads(json.dumps(good))
    bad_choice["questions"][0]["choices"][0]["label"] = "x"
    assert any("not exactly {value, description}" in p for p in check_request(bad_choice))
    assert check_request(lp.check_body()) == []                        # the check's own request: a predicate and a 3-choice
    image = {"model": "gpt-6-luna", "input": [{"role": "user", "content": [{"type": "input_text", "text": "x"}]}],
             "questions": [{"type": "predicate", "name": "p", "instructions": "ok?"}]}
    assert check_request(image) == []
    with FakeDecisions(key=FAKE_KEY) as fake:
        client = client_for(fake)
        with pytest.raises(lp.LunaError) as bad:
            client.ask_sync(dict(good, temperature=0), 5.0)
        assert bad.value.kind == "http" and bad.value.status == 400 and "unknown fields" in bad.value.text
        with pytest.raises(lp.LunaError) as model:
            client.ask_sync(dict(good, model="gpt-5"), 5.0)
        assert model.value.status == 404
        client.close()


def test_usage_is_read_when_the_api_sends_one():
    with FakeDecisions(key=FAKE_KEY, usage=True) as fake:
        client = client_for(fake)
        reply = client.ask_sync(toy_built().body, 5.0)
        client.close()
    assert isinstance(reply.input_tokens, int) and reply.input_tokens > 0


@pytest.mark.parametrize("behaviour, kind, status, requests_seen", [
    ("http_401", "http", 401, 1), ("http_403", "http", 403, 1), ("http_404", "http", 404, 1), ("http_429", "http", 429, 1),
    ("http_400", "http", 400, 1), ("http_500", "http", 500, 2), ("http_503", "http", 503, 2), ("html", "http", 502, 2),
    ("malformed_json", "malformed", 200, 1), ("malformed_shape", "malformed", 200, 1), ("reset", "network", None, 2),
])
def test_every_failure_of_the_api_is_a_luna_error_with_the_status_and_without_the_key(behaviour, kind, status, requests_seen):
    with FakeDecisions(key=FAKE_KEY, mode=behaviour) as fake:
        client = client_for(fake)
        with pytest.raises(lp.LunaError) as failed:
            client.ask_sync(toy_built().body, 5.0)
        client.close()
    error = failed.value
    assert error.kind == kind and error.status == status
    # a 5xx and a dropped connection are tried once more, a 4xx and a 429 are not
    assert len(fake.requests) == requests_seen and error.attempts == requests_seen
    assert FAKE_KEY not in error.text and FAKE_KEY not in str(error) and FAKE_KEY not in repr(error)
    if status and status != 200:
        assert ("HTTP %d" % status) in error.text
    if behaviour == "http_401":
        assert "Incorrect API key provided: ***." in error.text                             # the API's own message, the key as ***
    if behaviour == "http_429":
        assert "Rate limit" in error.text and error.reason == "http_429"
    if behaviour == "html":
        assert "a web page" in error.text


def test_a_wrong_key_is_refused_and_never_echoed():
    wrong = "sk-wrong-key-000011112222"
    with FakeDecisions(key=FAKE_KEY) as fake:
        client = client_for(fake, key=wrong)
        with pytest.raises(lp.LunaError) as failed:
            client.ask_sync(toy_built().body, 5.0)
        client.close()
    assert failed.value.status == 401 and wrong not in failed.value.text and fake.requests[0]["authorization"] == "wrong"


def test_a_slow_answer_is_given_up_at_the_timeout():
    with FakeDecisions(key=FAKE_KEY, mode="hang", hang_s=3.0) as fake:
        client = client_for(fake)
        began = time.monotonic()
        with pytest.raises(lp.LunaError) as failed:
            client.ask_sync(toy_built().body, 0.3)
        took = time.monotonic() - began
        assert failed.value.kind == "timeout" and failed.value.reason == "timeout" and 0.25 <= took < 1.0

        async def from_the_loop():
            t0 = time.monotonic()
            with pytest.raises(lp.LunaError) as late:
                await client.ask(toy_built().body, 0.3)
            return late.value, time.monotonic() - t0

        error, took = asyncio.run(from_the_loop())
        assert error.kind == "timeout" and took < 0.6                     # the loop's own limit, whatever the socket does
        client.close()


def test_one_retry_of_a_server_error_when_there_is_time():
    with FakeDecisions(key=FAKE_KEY, mode="mix") as fake:
        client = client_for(fake)
        fake.script.extend(["http_500"])                                   # then the mode: a normal answer
        reply = client.ask_sync(toy_built(3, mix=1).body, 5.0)
        assert reply.attempts == 2 and len(fake.requests) == 2
        fake.script.extend(["http_500", "http_500"])
        with pytest.raises(lp.LunaError) as failed:
            client.ask_sync(toy_built().body, 5.0)                         # one retry only
        assert failed.value.status == 500 and len(fake.requests) == 4
        slow = lp.LunaClient(FAKE_KEY, url=fake.url, retry_min_left=0.5)
        fake.script.extend(["http_500"])
        fake.delay = (0.25, 0.25001)
        with pytest.raises(lp.LunaError):
            slow.ask_sync(toy_built().body, 0.6)                           # under retry_min_left of the time is left: no retry
        assert len(fake.requests) == 5
        slow.close()
        client.close()


def test_parse_answer_says_what_is_wrong_with_an_answer():
    values = {"a0-x": 0, "a4-y": 4}
    ok = [{"type": "choice", "name": "action", "choice": "a4-y", "confidence": 0.7,
           "probabilities": [{"value": "a0-x", "probability": 0.3}, {"value": "a4-y", "probability": 0.7}, {"value": "zz", "probability": 0.1}]}]
    assert lp.parse_answer(ok, "action", values) == {"kind": "choice", "action": 4, "value": "a4-y", "confidence": 0.7,
                                                      "probabilities": {0: 0.3, 4: 0.7}}
    assert lp.parse_answer([{"type": "refusal", "name": "action"}], "action", values)["kind"] == "refusal"
    assert lp.parse_answer([{"type": "choice", "name": "action", "choice": "a9-z"}], "action", values)["kind"] == "illegal"
    assert lp.parse_answer([{"type": "choice", "name": "action", "choice": 4}], "action", values)["kind"] == "malformed"
    assert lp.parse_answer([{"type": "choice", "name": "action"}], "action", values)["kind"] == "malformed"
    assert lp.parse_answer([{"type": "predicate", "name": "action", "probability": 0.5}], "action", values)["kind"] == "malformed"
    assert lp.parse_answer([{"type": "choice", "name": "other", "choice": "a0-x"}], "action", values)["kind"] == "malformed"
    for nothing in ([], None, {"answers": []}, "text", [None]):
        assert lp.parse_answer(nothing, "action", values)["kind"] == "malformed"
    bare = lp.parse_answer([{"type": "choice", "name": "action", "choice": "a0-x"}], "action", values)
    assert bare["kind"] == "choice" and bare["confidence"] is None and bare["probabilities"] == {}


# =============================================================================================
# The blunder guard
# =============================================================================================

def spread(**kw):
    out = [0.0] * 14
    for key, value in kw.items():
        out[int(key[1:])] = value
    return out


def test_the_blunder_guard_fires_only_under_one_percent_of_both_against_over_sixty_of_either():
    net = spread(a0=0.70, a1=0.295, a2=0.005)
    shares = spread(a0=0.50, a1=0.495, a2=0.005)
    assert lp.blunder_guard(2, 0, net, shares)                               # under 1% of both, mix's 70% of the network
    assert "0.5% of the network" in lp.blunder_guard(2, 0, net, shares)
    assert lp.blunder_guard(1, 0, net, shares) == ""                         # Luna's pick has support
    assert lp.blunder_guard(0, 0, net, shares) == ""                         # mix's own pick
    assert lp.blunder_guard(2, 0, spread(a0=0.59, a1=0.405, a2=0.005), spread(a0=0.59, a1=0.405, a2=0.005)) == ""   # mix's is not over 60%
    assert lp.blunder_guard(2, 0, spread(a0=0.50, a1=0.495, a2=0.005), spread(a0=0.61, a1=0.385, a2=0.005))         # over 60% of the search
    assert lp.blunder_guard(2, 0, net, spread(a0=0.50, a1=0.48, a2=0.02)) == ""                                     # 2% of the search: not a blunder
    assert lp.blunder_guard(2, 0, spread(a0=0.70, a1=0.28, a2=0.02), shares) == ""                                  # 2% of the network
    # No search answer: the search speaks neither for nor against anything.
    assert lp.blunder_guard(2, 0, net, None) and lp.blunder_guard(1, 0, net, None) == ""
    assert lp.blunder_guard(2, 0, spread(a0=0.55, a1=0.445, a2=0.005), None) == ""
    # A move and its Tera form are one move to the network (Facts.merged): the Tera form of a 70% move is not a blunder.
    facts = toy_facts(14, mix=0, probs=spread(a0=0.70, a1=0.295, a10=0.005), shares=spread(a0=0.9, a1=0.095, a10=0.005))
    facts.mask = [1.0 if a in (0, 1, 10) else 0.0 for a in range(14)]
    assert lp.blunder_guard(10, 0, facts.merged(), facts.shares) == ""
    assert lp.blunder_guard(10, 0, facts.probs, facts.shares)                # (the raw split would have called it one)


# =============================================================================================
# One decision: Luna.choose and the rails
# =============================================================================================

def choose(fake, mix=0, n=3, time_left=5.0, facts=None, client=None, **settings):
    settings.setdefault("timeout", 0.4)
    client = client or client_for(fake)
    luna = lp.Luna(client, lp.LunaSettings(**settings))
    built = toy_built(n, mix)
    facts = facts or toy_facts(n, mix)
    action, info = asyncio.run(luna.choose(built, facts, time_left))
    return action, info, luna


def test_a_normal_answer_is_played_and_recorded():
    with FakeDecisions(key=FAKE_KEY, mode="other") as fake:
        action, info, luna = choose(fake, mix=1)
        luna.client.close()
    assert action == 0 and info["choice"] == 0 and info["mix"] == 1 and not info["fell_back"] and not info["overruled"]
    assert info["asked"] and info["confidence"] == 0.85 and info["probabilities"] == {"0": 0.9, "1": 0.05, "2": 0.05}
    assert info["latency_ms"] >= 0 and info["tokens_from"] == "estimate" and info["input_tokens"] == toy_built(3, 1).tokens
    assert info["why"] is None and info["choice_value"] == "a0-toy" and info["timeout_ms"] == 400.0
    json.dumps(info)
    with FakeDecisions(key=FAKE_KEY, mode="mix", usage=True) as fake:
        action, info, luna = choose(fake, mix=2)
        luna.client.close()
    assert action == 2 and info["choice"] == 2 and info["tokens_from"] == "usage"


@pytest.mark.parametrize("behaviour, why", [
    ("hang", "timeout"), ("http_401", "http_401"), ("http_429", "http_429"), ("http_500", "http_500"), ("refusal", "refusal"),
    ("malformed_json", "malformed"), ("malformed_shape", "malformed"), ("no_choice", "malformed"), ("wrong_name", "malformed"),
    ("illegal", "illegal"), ("value:a9-not-legal", "illegal"), ("reset", "network"), ("html", "http_502"),
])
def test_every_rail_plays_mixs_pick_and_says_why(behaviour, why):
    with FakeDecisions(key=FAKE_KEY, mode=behaviour, hang_s=2.0) as fake:
        action, info, luna = choose(fake, mix=1, timeout=0.3)
        luna.client.close()
    assert action == 1 and info["fell_back"] and info["why"] == why and info["choice"] is None and info["asked"]
    assert not info["overruled"] and FAKE_KEY not in json.dumps(info)
    if why.startswith("http_"):
        assert info["http_status"] == int(why.split("_")[1]) and ("HTTP %s" % why.split("_")[1]) in info["error"]
    if why == "illegal":
        assert info["choice_value"] in ("zz-not-offered", "a9-not-legal")


def test_no_time_left_means_luna_is_not_asked():
    with FakeDecisions(key=FAKE_KEY, mode="other") as fake:
        action, info, luna = choose(fake, mix=1, time_left=0.1)
        assert action == 1 and info["why"] == "no_time" and not info["asked"] and fake.requests == []
        # The wait is the smaller of the timeout and what the deadline leaves.
        fake.mode = "hang"
        began = time.monotonic()
        action, info, luna2 = choose(fake, mix=1, time_left=0.3, timeout=5.0)
        assert action == 1 and info["why"] == "timeout" and info["timeout_ms"] == 300.0 and time.monotonic() - began < 0.8
        luna.client.close()
        luna2.client.close()


def test_five_failures_in_a_row_pause_luna_for_a_minute():
    now = [1000.0]
    with FakeDecisions(key=FAKE_KEY, mode="http_500") as fake:
        client = client_for(fake, clock=lambda: now[0], retries=0)
        for i in range(4):
            action, info, _ = choose(fake, mix=1, client=client)
            assert info["why"] == "http_500" and client.available() == (True, "")
        fake.mode = "other"                                                # a good answer in between starts the count again
        action, info, _ = choose(fake, mix=1, client=client)
        assert action == 0 and client.available() == (True, "")
        fake.mode = "http_500"
        for i in range(5):
            choose(fake, mix=1, client=client)
        assert client.available() == (False, "paused") and client.pauses == 1
        assert len(client.said) == 1 and "5 API failures in a row" in client.said[0] and "60 s" in client.said[0]
        asked = len(fake.requests)
        fake.mode = "other"
        action, info, _ = choose(fake, mix=1, client=client)               # not asked while it is paused
        assert action == 1 and info["why"] == "paused" and not info["asked"] and len(fake.requests) == asked
        now[0] += 59.0
        assert client.available() == (False, "paused")
        now[0] += 2.0
        assert client.available() == (True, "")
        action, info, _ = choose(fake, mix=1, client=client)
        assert action == 0 and len(fake.requests) == asked + 1
        client.close()


def test_a_refusal_or_an_illegal_choice_is_not_an_api_failure_but_a_wrong_shape_is():
    with FakeDecisions(key=FAKE_KEY, mode="refusal") as fake:
        client = client_for(fake, retries=0)
        for _ in range(7):
            choose(fake, client=client)
        fake.mode = "illegal"
        for _ in range(7):
            choose(fake, client=client)
        assert client.available() == (True, "") and client.pauses == 0
        fake.mode = "malformed_shape"
        for _ in range(5):
            choose(fake, client=client)
        assert client.available() == (False, "paused")
        client.close()


@pytest.mark.parametrize("status", [401, 403, 404])
def test_a_refused_key_turns_luna_off_for_the_run(status):
    with FakeDecisions(key=FAKE_KEY, mode="http_%d" % status) as fake:
        client = client_for(fake)
        action, info, _ = choose(fake, mix=1, client=client)
        assert action == 1 and info["why"] == "http_%d" % status
        assert client.disabled and ("HTTP %d" % status) in client.disabled and FAKE_KEY not in client.disabled
        assert client.available() == (False, "key_refused")
        assert len(client.said) == 1 and "OFF for the rest of this run" in client.said[0] and FAKE_KEY not in client.said[0]
        fake.mode = "other"
        action, info, _ = choose(fake, mix=1, client=client)
        assert action == 1 and info["why"] == "key_refused" and not info["asked"] and len(fake.requests) == 1
        client.close()


def test_the_blunder_guard_overrules_and_can_be_switched_off():
    probs, shares = spread(a0=0.005, a1=0.70, a2=0.295), spread(a0=0.004, a1=0.60, a2=0.396)
    with FakeDecisions(key=FAKE_KEY, mode="first") as fake:
        action, info, luna = choose(fake, mix=1, facts=toy_facts(3, 1, probs, shares))
        assert action == 1 and info["overruled"] and info["choice"] == 0 and not info["fell_back"]
        assert "0.5% of the network" in info["overruled_why"]
        luna.client.close()
        action, info, luna = choose(fake, mix=1, facts=toy_facts(3, 1, probs, shares), guard=False)
        assert action == 0 and not info["overruled"] and info["choice"] == 0
        luna.client.close()
        fake.mode = "last"                                               # a pick with support stands
        action, info, luna = choose(fake, mix=1, facts=toy_facts(3, 1, probs, shares))
        assert action == 2 and not info["overruled"]
        luna.client.close()


def test_the_counts_of_a_game_and_their_line():
    counts = lp.new_counts()
    base = {"mix": 1, "input_tokens": 2000, "asked": True, "latency_ms": 200.0, "fell_back": False, "overruled": False}
    lp.add_counts(counts, dict(base, choice=1))                           # agreed with mix
    lp.add_counts(counts, dict(base, choice=0, latency_ms=400.0))         # another action, played
    lp.add_counts(counts, dict(base, choice=2, overruled=True))           # overruled
    lp.add_counts(counts, dict(base, choice=None, fell_back=True, why="timeout", latency_ms=None))
    lp.add_counts(counts, dict(base, choice=None, fell_back=True, why="http_429"))
    lp.add_counts(counts, {"mix": 1, "asked": False, "choice": None, "fell_back": True, "why": "paused", "input_tokens": 1900})
    fresh = {}
    lp.add_counts(fresh, dict(base, choice=1))                            # a dict without the keys gets them
    assert fresh["luna_decisions"] == 1
    r = lp.result_fields(counts)
    assert r == {"luna_decisions": 6, "luna_asked": 5, "luna_answered": 3, "luna_played": 2, "luna_overruled": 1,
                 "luna_agreed_with_mix": 1, "luna_changed": 1, "luna_fallbacks": {"timeout": 1, "http_429": 1, "paused": 1},
                 "luna_latency_ms_mean": 250.0, "luna_latency_ms_max": 400.0, "luna_tokens": 10000, "luna_cost_usd": 0.001,
                 "luna_output_tokens": 0, "luna_model": None}       # (the two fields of the models: Luna charges input only)
    text = lp.counts_text(counts)
    assert "asked on 5 of 6 decisions" in text and "answered 3 (played 2, overruled by the blunder guard 1)" in text
    assert "mix played instead on 3 (http_429 1, paused 1, timeout 1)" in text and "10,000 input tokens, about $0.0010" in text
    json.dumps(r)


# =============================================================================================
# luna-check
# =============================================================================================

def test_the_check_asks_one_tiny_question_and_prints_what_came_back():
    lines = []
    with FakeDecisions(key=FAKE_KEY) as fake:
        client = client_for(fake)
        assert lp.luna_check(client, out=lines.append) == 0
        client.close()
    text = "\n".join(lines)
    assert "Luna check OK" in text and "predicate two_plus_two (is 2+2=4): probability 0.99" in text
    assert "choice sum (what is 2+2): four, confidence 0.85, probabilities three 0.05, four 0.90, five 0.05" in text
    assert "HTTP 200 in" in text and "key ***" in text and FAKE_KEY not in text
    # one tiny question, then a question of a game's size twice on the open connection
    assert len(fake.requests) == 3 and all(r["problems"] == [] for r in fake.requests)
    names = [[q["name"] for q in r["body"]["questions"]] for r in fake.requests]
    assert names == [["two_plus_two", "sum"], ["sample"], ["sample"]]
    assert [q["type"] for q in fake.requests[0]["body"]["questions"]] == ["predicate", "choice"]
    assert len(fake.requests[0]["body"]["questions"][1]["choices"]) == 3
    sized = fake.requests[1]["body"]
    assert len(sized["questions"][0]["choices"]) == 13 and sized["questions"][0]["instructions"] == lp.INSTRUCTIONS
    assert sized["input"].startswith("Pokemon Showdown, gen 9 random battle") and 1500 < lp.request_tokens(sized) < lp.TOKEN_BUDGET
    assert "a question of a game's size (13 choices, about %d tokens): HTTP 200, choice a2-dracometeor, confidence 0.85" % lp.request_tokens(sized) in text
    assert "its answers on the open connection took:" in text


def test_the_checks_sample_is_a_real_decision_without_names():
    body = lp.sample_body()
    assert body is not None and check_request(body) == [] and body["model"] == "gpt-6-luna"
    text = json.dumps(body)
    assert "Player" not in text and "Opponent" not in text and "MIX'S PICK" in text and "FOE'S ACTIVE:" in body["input"]
    assert sum(MIX_MARK in c["description"] for c in body["questions"][0]["choices"]) == 1
    assert lp.sample_body(path=os.path.join(REPO, "no-such-file.json")) is None
    lines = []
    with FakeDecisions(key=FAKE_KEY) as fake:                              # without the sample: the tiny question again
        client = client_for(fake)
        assert lp.luna_check(client, out=lines.append, sample=None) == 0
        client.close()
    assert [[q["name"] for q in r["body"]["questions"]] for r in fake.requests] == [["two_plus_two", "sum"]] * 3


def test_a_check_whose_game_sized_question_fails_refuses_to_play():
    lines = []
    with FakeDecisions(key=FAKE_KEY, fail_checks=True, mode="first") as fake:
        fake.script.extend(["first", "http_400"])                          # the tiny question is answered, the large one is not taken
        client = client_for(fake)
        assert lp.luna_check(client, out=lines.append) == 1
        fake.script.extend(["first", "refusal"])
        assert lp.luna_check(client, out=lines.append) == 1
        client.close()
    failed = [l for l in lines if l.startswith("LUNA CHECK FAILED")]
    assert len(failed) == 2 and "a question of a game's size (13 choices" in failed[0] and "HTTP 400" in failed[0]
    assert "got no usable answer: refusal" in failed[1] and "Luna check OK" not in "\n".join(lines) and FAKE_KEY not in "\n".join(lines)


@pytest.mark.parametrize("behaviour, said", [
    ("http_401", "HTTP 401"), ("http_429", "HTTP 429"), ("http_500", "HTTP 500"), ("malformed_shape", "\"answers\""),
    ("refusal", "refused"), ("illegal", "illegal"), ("hang", "no answer within"),
])
def test_a_failed_check_says_why_in_one_line_without_the_key(behaviour, said):
    lines = []
    with FakeDecisions(key=FAKE_KEY, mode=behaviour, fail_checks=True, hang_s=2.0) as fake:
        client = client_for(fake)
        assert lp.luna_check(client, timeout=0.5, out=lines.append) == 1
        client.close()
    text = "\n".join(lines)
    failed = [l for l in lines if l.startswith("LUNA CHECK FAILED")]
    assert len(failed) == 1 and said in text and FAKE_KEY not in text and "Luna check OK" not in text


def test_ladder_luna_check_reads_the_key_from_the_environment_and_never_prints_it(monkeypatch, capsys):
    with FakeDecisions(key=FAKE_KEY) as fake:
        monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
        assert ladder.main(["--luna-check", "--luna-url", fake.url]) == 0
        assert "OPENAI_API_KEY" not in os.environ                           # taken out of this process's environment
        out = capsys.readouterr()
        assert "Luna check OK" in out.out and FAKE_KEY not in out.out and FAKE_KEY not in out.err
        wrong = "sk-wrong-key-9988776655"
        monkeypatch.setenv("OPENAI_API_KEY", wrong)
        assert ladder.main(["--luna-check", "--luna-url", fake.url]) == ladder.LUNA_FAILED == 5
        out = capsys.readouterr()
        assert "LUNA CHECK FAILED: HTTP 401" in out.out and wrong not in out.out and wrong not in out.err
        # No key in the environment: asked once at the hidden prompt.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        prompts = []
        monkeypatch.setattr(lp.getpass, "getpass", lambda prompt="": prompts.append(prompt) or FAKE_KEY)
        assert ladder.main(["--luna-check", "--luna-url", fake.url]) == 0
        assert len(prompts) == 1 and FAKE_KEY not in capsys.readouterr().out
        monkeypatch.setattr(lp.getpass, "getpass", lambda prompt="": "")
        assert ladder.main(["--luna-check", "--luna-url", fake.url]) == 5
        assert "no OpenAI API key" in capsys.readouterr().out
        # An address that is not OpenAI's and not this machine: the key is not sent, and nothing is asked.
        monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
        asked = len(fake.requests)
        assert ladder.main(["--luna-check", "--luna-url", "https://example.com/v1/decisions"]) == 5
        assert "to OpenAI" in capsys.readouterr().out and len(fake.requests) == asked


def test_a_run_with_the_luna_arm_checks_first_and_plays_nothing_when_it_fails(monkeypatch, capsys, tmp_path):
    with FakeDecisions(key=FAKE_KEY, mode="http_500", fail_checks=True) as fake:
        monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
        args = ladder.parse_args(["--arms", "mix,luna", "--luna-url", fake.url, "--exit-file", str(tmp_path / "exit.txt")])
        assert ladder.wants_luna(args) and ladder.luna_ready(args) == 5 and getattr(args, "_luna", None) is None
        out = capsys.readouterr().out
        assert "Nothing was played" in out and "HTTP 500" in out and FAKE_KEY not in out
        note = open(tmp_path / "exit.txt", encoding="utf-8").read()
        assert note.startswith("luna:") and FAKE_KEY not in note
        fake.fail_checks = False
        monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
        args = ladder.parse_args(["--player", "luna", "--luna-url", fake.url, "--luna-guard", "off", "--luna-timeout", "0.9",
                                  "--no-luna-save-briefs"])
        assert ladder.luna_ready(args) == 0
        luna = args._luna
        assert isinstance(luna, lp.Luna) and luna.settings.guard is False and luna.settings.timeout == 0.9
        assert luna.settings.save_briefs is False and luna.client.url == fake.url
        assert FAKE_KEY not in capsys.readouterr().out and FAKE_KEY in ladder.luna_secrets()
        luna.client.close()
        # --luna-skip-check: nothing is asked before the first game
        monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
        asked = len(fake.requests)
        args = ladder.parse_args(["--player", "luna", "--luna-url", fake.url, "--luna-skip-check"])
        assert ladder.luna_ready(args) == 0 and len(fake.requests) == asked
        args._luna.client.close()


# =============================================================================================
# ladder.py: the arm, the flags, the lines
# =============================================================================================

def test_luna_is_a_mode_of_the_player_and_of_ladder():
    assert sp.LUNA_ARM == lp.LUNA_ARM == "luna" and "luna" in sp.MODES and ladder.PLAYER_MODES == sp.MODES
    assert sp.split_mode("luna") == ("mix", None)                         # mix, with Luna deciding after it
    assert not sp.guarded("luna", sp.MixSettings())
    args = ladder.parse_args(["--player", "luna"])
    assert ladder.wants_luna(args) and (args.luna_timeout, args.luna_guard, args.luna_model) == (1.2, "on", "gpt-6-luna")
    assert args.luna_url == lp.DEFAULT_URL and args.luna_save_briefs is True and not args.luna_check and not args.luna_skip_check
    args = ladder.parse_args(["--arms", "mix,luna", "--experiment", "luna-vs-mix-024381"])
    assert args.arms == ["mix", "luna"] and ladder.wants_luna(args) and ladder.experiment_tag(args) == "luna-vs-mix-024381"
    assert not ladder.wants_luna(ladder.parse_args(["--arms", "net,mix"])) and not ladder.wants_luna(ladder.parse_args([]))
    source = open(os.path.join(REPO, "ladder.py"), encoding="utf-8").read()
    assert "--luna-key" not in source and "--openai-key" not in source and "--api-key" not in source      # the key is never a flag
    with pytest.raises(SystemExit):
        ladder.parse_args(["--player", "luna", "--luna-guard", "maybe"])
    with pytest.raises(SystemExit):
        ladder.parse_args(["--player", "luna", "--luna-timeout", "0"])


def test_the_deadline_has_to_hold_the_search_and_luna():
    ok = ladder.parse_args(["--arms", "mix,luna", "--parallel", "3", "--helpers", "4", "--search-ms", "500", "--deadline", "2.5"])
    assert ladder.search_problems(ok) == []
    tight = ladder.parse_args(["--player", "luna", "--parallel", "3", "--helpers", "4", "--search-ms", "500", "--deadline", "1.7"])
    problems = ladder.search_problems(tight)
    assert len(problems) == 1 and "leaves Luna no time" in problems[0]
    mix = ladder.parse_args(["--player", "mix", "--parallel", "3", "--helpers", "4", "--search-ms", "500", "--deadline", "1.7"])
    assert ladder.search_problems(mix) == []                               # mix alone: as before


def test_the_results_line_and_the_arms_text_carry_lunas_counts():
    counts = lp.new_counts()
    for i in range(10):
        lp.add_counts(counts, {"mix": 1, "choice": 1 if i < 7 else 0, "asked": True, "input_tokens": 1800, "latency_ms": 300.0,
                               "fell_back": False, "overruled": False})
    lp.add_counts(counts, {"mix": 1, "choice": None, "asked": True, "input_tokens": 1800, "fell_back": True, "why": "timeout"})
    summary = dict({"arm": "luna", "decisions": 12, "asked": 11, "searched": 11, "fallbacks": 0}, **lp.result_fields(counts))
    line = dict(ladder.arm_fields(summary, "p1", 500, "veto"), won=True)
    assert line["arm"] == "luna" and line["mix_rule"] == "veto" and line["luna_asked"] == 11 and line["luna_answered"] == 10
    assert line["luna_fallbacks"] == {"timeout": 1} and line["luna_changed"] == 3
    text = ladder.arms_text([line, dict(line, won=False), {"arm": "mix", "won": True, "decisions": 12}])
    luna_row = next(l for l in text.splitlines() if l.strip().startswith("luna"))
    assert "Luna: asked on 22 of 22 decisions with a choice, answered 20" in luna_row and "timeout 2" in luna_row
    assert "Luna" not in next(l for l in text.splitlines() if l.strip().startswith("mix"))
    assert ladder.luna_text([{"arm": "mix", "won": True}]) == ""


def test_ab_report_reads_the_luna_arm_against_mix():
    rows = []
    for i in range(40):
        rows.append({"experiment": "luna-vs-mix-024381", "arm": "mix", "won": i % 2 == 0, "opponent_rating": 2300, "role": "p1",
                     "decisions": 30, "searched": 30, "iterations_per_decision": 500000})
        rows.append({"experiment": "luna-vs-mix-024381", "arm": "luna", "won": i % 3 != 0, "opponent_rating": 2305, "role": "p2",
                     "decisions": 30, "searched": 30, "iterations_per_decision": 500000, "luna_decisions": 28, "luna_asked": 27,
                     "luna_answered": 25, "luna_played": 24, "luna_overruled": 1, "luna_agreed_with_mix": 20, "luna_changed": 4,
                     "luna_fallbacks": {"timeout": 2, "no_time": 1}, "luna_latency_ms_mean": 320.0, "luna_tokens": 50000})
    out = ab_report.report(rows, {"mix_iterations_ref": 500000}, ["mix", "luna"])
    assert out["arms"] == ["mix", "luna"] and "luna" in out["now"] and out["stats"]["mix"]["luna"] is None
    lu = out["stats"]["luna"]["luna"]
    assert lu["asked"] == 27 * 40 and lu["fallbacks"] == {"timeout": 80, "no_time": 40} and lu["played_share"] == round(24 / 28, 4)
    text = ab_report.text(out, "luna-vs-mix-024381")
    assert "Luna in luna's 40 games: asked on 1080 of 1120 decisions" in text and "before the interim look: 40 of 250" in text
    assert ab_report.COST["luna"] > ab_report.COST["mix-deep"] > ab_report.COST["mix"]      # the tie rule keeps mix within a point
    # A Luna arm that mostly did not answer is flagged: it was mix under another name.
    silent = [dict(r, luna_played=5, luna_answered=5) if r["arm"] == "luna" else r for r in rows]
    warned = ab_report.report(silent, {"mix_iterations_ref": 500000}, ["mix", "luna"])
    assert any("Luna's choice was the order on only" in w for w in warned["warnings"])
    assert not any("Luna's choice" in w for w in out["warnings"])


def test_luna_report_adds_up_the_luna_games(tmp_path):
    path = tmp_path / "results.jsonl"
    rows = [{"arm": "luna", "won": True, "experiment": "e1", "luna_decisions": 20, "luna_asked": 20, "luna_answered": 18, "luna_played": 18,
             "luna_overruled": 0, "luna_agreed_with_mix": 15, "luna_changed": 3, "luna_fallbacks": {"timeout": 2},
             "luna_latency_ms_mean": 250.0, "luna_latency_ms_max": 900.0, "luna_tokens": 40000},
            {"arm": "luna", "won": False, "luna_decisions": 10, "luna_asked": 0, "luna_answered": 0, "luna_played": 0, "luna_overruled": 0,
             "luna_agreed_with_mix": 0, "luna_changed": 0, "luna_fallbacks": {"paused": 10}, "luna_tokens": 0},
            {"arm": "mix", "won": True}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8")
    s = luna_report.summary(luna_report.load(str(path)))
    assert (s["games"], s["wins"], s["luna_asked"], s["luna_answered"], s["silent_games"]) == (2, 1, 20, 18, 1)
    assert s["luna_fallbacks"] == {"timeout": 2, "paused": 10} and s["luna_cost_usd"] == 0.004
    text = luna_report.text(s, None)
    assert "Luna: 2 games, 1 wins and 1 losses (50%)" in text and "In 1 games Luna never answered" in text
    assert luna_report.summary(luna_report.load(str(path), "e1"))["games"] == 1
    assert "no games of the arm luna" in luna_report.text(luna_report.summary([]), None)


# =============================================================================================
# The launcher
# =============================================================================================

def launcher_argv(line, arm="luna"):
    """A ladder.py line of the launcher as its arguments, with the model's arm the first two questions set (luna unless told)."""
    line = line.replace("%GAMES%", "30").replace("%PY%", "python").replace("%CKPT%", '"C:\\ckpt\\learner_update_024381.pt"')
    line = line.replace("%ARM%", arm).replace("%CHECKMODEL%", arm.split("-")[0])
    words = [w[1:-1] if w.startswith('"') and w.endswith('"') else w for w in re.findall(r'"[^"]*"|\S+', line)]
    return words[words.index("ladder.py") + 1:]


def test_the_launcher_checks_first_warns_and_plays_the_two_choices():
    raw = open(LAUNCHER, "rb").read().decode("utf-8")
    text = raw.replace("\r\n", "\n")
    assert raw.count("\r\n") == raw.count("\n") > 40                       # a .cmd keeps CRLF
    runs = [l for l in text.splitlines() if l.startswith("%PY% ladder.py ")]
    assert len(runs) == 3 and runs[0] == "%PY% ladder.py --luna-check --model %CHECKMODEL%"
    check = ladder.parse_args(launcher_argv(runs[0]))
    assert check.luna_check and check.model == "luna" and ladder.parse_args(launcher_argv(runs[0], "sol-lines")).model == "sol"
    assert text.index("%PY% ladder.py --luna-check") < text.index("if errorlevel 1 goto checkfailed") < text.index("set /p PS_USERNAME")
    assert "nothing was played" in text[text.index(":checkfailed"):]
    ab = ladder.parse_args(launcher_argv(runs[1]))
    assert ab.arms == ["mix", "luna"] and ladder.experiment_tag(ab) == "luna-vs-mix-024381" and ab.server == "public"
    assert (ab.games, ab.parallel, ab.helpers, ab.search_ms, ab.deadline, ab.predictions) == (30, 3, 4, 500, 2.5, True)
    assert ab.checkpoint.endswith("learner_update_024381.pt") and ab.search_remote is None and ladder.search_problems(ab) == []
    assert (ab.luna_timeout, ab.luna_guard, ab.luna_url) == (1.2, "on", lp.DEFAULT_URL) and not ab.luna_skip_check
    only = ladder.parse_args(launcher_argv(runs[2]))
    assert only.player == "luna" and not only.arms and ladder.wants_luna(only) and ladder.search_problems(only) == []
    assert (only.parallel, only.helpers, only.search_ms, only.deadline) == (3, 4, 500, 2.5)
    assert "learner_update_024381.pt" in text and "cd /d C:\\Users\\you\\ps-ppo-luna" in text
    # the question, with the A/B as the default, and 30 games
    assert 'if "%MODE%"=="" set MODE=2' in text and 'if "%GAMES%"=="" set GAMES=30' in text and 'if "%MODE%"=="1" goto lunaonly' in text
    assert "2 = A/B: %NAME% and today's bot alternate (shows whether %NAME% is better)" in text and "1 = %NAME% every game" in text
    # the model and reading ahead: Luna and no lines unless asked, and the same two runs for every answer
    assert text.index("set ARM=luna") < text.index('if "%MODEL%"=="2" set ARM=sol') < text.index('if /i "%AHEAD%"=="y" set ARM=%ARM%-lines')
    assert text.index('if /i "%AHEAD%"=="y" set ARM=%ARM%-lines') < text.index("%PY% ladder.py --luna-check")
    for arm in ("sol", "luna-lines", "sol-lines"):
        other = ladder.parse_args(launcher_argv(runs[1], arm))
        assert other.arms == ["mix", arm] and ladder.experiment_tag(other) == arm + "-vs-mix-024381" and ladder.search_problems(other) == []
        assert ladder.parse_args(launcher_argv(runs[2], arm)).player == arm
    # the warning, before anything is asked
    assert "RATED games" in text and text.index("RATED games on the Showdown account") < text.index("set /p PS_USERNAME")
    # one bot per account, as Look Play AB: another playing ladder.py and it does not start
    scan = text.index("ladder\\.py' -and")
    assert scan < text.index("%PY% ladder.py --luna-check") and "exit /b 1" in text and "-notmatch '--graph|--history|--luna-check'" in text
    # the key: asked only when the variable is not set, hidden, never on a command line, cleared at the end
    assert text.index('if not "%OPENAI_API_KEY%"=="" goto keyok') < text.index("Read-Host -AsSecureString")
    assert all("OPENAI_API_KEY" not in l and "sk-" not in l for l in runs)
    assert 'if "%ASKEDKEY%"=="1" set OPENAI_API_KEY=' in text and "echo %OPENAI_API_KEY%" not in text
    # the reports after the run
    assert "%PY% bench\\ab_report.py --experiment %ARM%-vs-mix-024381 --control mix --arms mix,%ARM%" in text
    assert "%PY% bench\\luna_report.py" in text
    assert text.index("%PY% ladder.py --games") < text.index("%PY% bench\\ab_report.py") < text.index("%PY% bench\\luna_report.py")
    stop = open(STOPPER, "rb").read().decode("utf-8")
    assert 'type nul > "C:\\Users\\you\\ps-ppo-luna\\ladder.stop"' in stop and "\r\n" in stop
    assert ladder.parse_args([]).stop_file == os.path.join(REPO, "ladder.stop")       # the file Luna Stop makes, from this folder


# =============================================================================================
# The audit of a run's files (bench/luna_audit.py): what counts as hidden
# =============================================================================================

def test_the_audit_knows_what_the_battle_had_shown():
    seen = luna_audit.Seen()
    seen.feed('|player|p1|Alice|1|\n|player|p2|Bob|2|')
    seen.feed('|request|{"side":{"id":"p2","pokemon":[{"details":"Garchomp, L77, M"},{"details":"Rotom-Wash, L83"}]}}')
    seen.feed("|switch|p1a: Corviknight|Corviknight, L79, F|100/100\n|switch|p2a: Garchomp|Garchomp, L77, M|331/331\n|turn|1")
    seen.feed("|move|p1a: Corviknight|Brave Bird|p2a: Garchomp\n|-damage|p2a: Garchomp|212/331\n|c|Alice|i have a Gengar")
    whole = seen.copy()
    whole.feed("|switch|p1a: Gengar|Gengar, L80|100/100\n|move|p1a: Corviknight|Roost|p1a: Corviknight")
    assert seen.role == "p2" and seen.foe_species == {"Corviknight"} and whole.foe_species == {"Corviknight", "Gengar"}
    assert seen.our_species == {"Garchomp", "Rotom-Wash"} and seen.foe_moves == {"Corviknight": {"bravebird"}}
    fine = "FOE'S ACTIVE: Corviknight L79 (Flying/Steel) HP 100%; item not shown (likely Leftovers); ability not shown; moves seen: Brave Bird."
    assert luna_audit.hidden_problems(fine, [{"value": "a0-earthquake", "description": "Earthquake."}], seen, whole) == []
    leaks = luna_audit.hidden_problems(fine + "\nFOE'S BENCH: Gengar L80.", [], seen, whole)
    assert leaks and "Gengar" in leaks[0]
    moved = luna_audit.hidden_problems(fine.replace("Brave Bird.", "Brave Bird, Roost."), [], seen, whole)
    assert moved and "Roost" in moved[0]
    assert luna_audit.hidden_problems(fine.replace("Brave Bird.", "Brave Bird, Roost."), [], whole, whole) == []    # once it was used
    item = luna_audit.hidden_problems(fine.replace("item not shown (likely Leftovers)", "item Rocky Helmet"), [], seen, whole)
    assert item and "Rocky Helmet" in item[0]
    named = luna_audit.hidden_problems(fine + " Alice plays it.", [], seen, whole)
    assert named and "user's name" in named[0]
    # the opponent's hidden Pokemon of a species we have ourselves is not a leak
    whole.feed("|switch|p1a: Garchomp|Garchomp, L77, F|100/100")
    assert luna_audit.hidden_problems(fine + "\nSwitch to Garchomp L77.", [], seen, whole) == []


# =============================================================================================
# The player: the arm on recorded battles, with the fake search and the fake API
# =============================================================================================

@pytest.fixture()
def classes():
    """The search player with the stub network of test_search_player, which also gives a value (as tests/test_ladder_ab.py)."""
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


def luna_for(fake, **settings):
    settings.setdefault("timeout", 0.5)
    return lp.Luna(client_for(fake), lp.LunaSettings(**settings))


def play_arm(classes, battles, mode, luna=None, deadline=1.5, **kw):
    """The battles played by one player per battle in `mode`: (their records, their orders, the players)."""
    bridge = tsp.FakeBridge(battles)
    notes, orders, players = [], [], []
    try:
        for events in battles:
            mine = []
            player = tsp.searcher(classes, events[-1]["username"], bridge, mode=mode, deadline=deadline, on_decision=mine.append,
                                  luna=luna, **kw)
            tsp.play(player, events)
            orders.append(tsp.chosen(player))
            players.append(player)
            player.notes = mine                  # this battle's records (two recordings are the two sides of one room)
            notes += mine
    finally:
        bridge.close()
    return notes, orders, players


@tsp.needs_player
def test_luna_answering_mixs_pick_plays_exactly_what_mix_plays(classes):
    battles = tsp.load_fixture()
    mix_notes, mix_orders, _ = play_arm(classes, battles, "mix")
    with FakeDecisions(key=FAKE_KEY, mode="mix") as fake:
        luna = luna_for(fake)
        notes, orders, players = play_arm(classes, battles, "luna", luna)
        luna.client.close()
    assert orders == mix_orders and sum(len(o) for o in orders) > 60          # identity: order for order
    assert [n["played"] for n in notes] == [n["played"] for n in mix_notes]
    assert all(n["mode"] == "luna" and n["arm"] == "luna" for n in notes)
    choice = [n for n in notes if n["legal"] > 1]
    assert choice and all("luna" in n for n in choice) and not any("luna" in n for n in notes if n["legal"] <= 1)
    assert not any("luna" in n for n in mix_notes)                             # mix's own records: as before
    for n in choice:
        lu = n["luna"]
        assert lu["asked"] and lu["choice"] == lu["mix"] == n["played"] and not lu["fell_back"] and not lu["overruled"]
        assert n["source"] == "luna" and n["mask"][n["played"]] == 1
        assert lu["confidence"] == 0.85 and lu["latency_ms"] >= 0 and lu["input_tokens"] > 100 and lu["tokens_from"] == "estimate"
        assert set(lu["probabilities"]) == {str(a) for a in range(14) if n["mask"][a]}
        assert lu["brief"].startswith("Pokemon Showdown, gen 9 random battle") and len(lu["choices"]) == n["legal"]
        assert lu["mix_text"] and lu["choice_text"] == lu["mix_text"] and lu["brief_ms"] < 2000
        assert "brief_problems" not in lu
    # the question on the wire: the documented request, the choices exactly the legal actions, mix's pick marked once
    games = fake.game_requests()
    assert len(games) == len(choice) and all(r["problems"] == [] and r["authorization"] == "ok" for r in fake.requests)
    for r, n in zip(games, choice):
        (q,) = r["body"]["questions"]
        assert q["name"] == "action" and q["type"] == "choice" and q["instructions"] == lp.INSTRUCTIONS
        ids = sorted(int(c["value"].split("-")[0][1:]) for c in q["choices"])
        assert ids == [a for a in range(14) if n["mask"][a]]
        assert sum(MIX_MARK in c["description"] for c in q["choices"]) == 1
        assert all(re.fullmatch(r"a\d+-[a-z0-9\-]+", c["value"]) for c in q["choices"])
        assert r["body"]["input"] == n["luna"]["brief"] and q["choices"] == n["luna"]["choices"]
    # the game's line and the run's line
    for player, events in zip(players, battles):
        s = player.game_summary(events[0]["room"])
        mine = [n for n in player.notes if n["legal"] > 1]
        assert s["arm"] == "luna" and s["luna_decisions"] == s["luna_asked"] == s["luna_answered"] == s["luna_played"] == len(mine)
        assert s["luna_agreed_with_mix"] == len(mine) and s["luna_changed"] == 0 and s["luna_overruled"] == 0 and s["luna_fallbacks"] == {}
        assert s["luna_tokens"] == sum(n["luna"]["input_tokens"] for n in mine) and s["luna_cost_usd"] > 0
        assert s["late"] == 0 and "Luna: asked on %d of %d decisions" % (len(mine), len(mine)) in player.search_summary()
    text = json.dumps(notes) + json.dumps([p.game_summary(e[0]["room"]) for p, e in zip(players, battles)])
    assert FAKE_KEY not in text and "Bearer" not in text and "uthorization" not in text


@tsp.needs_player
@pytest.mark.parametrize("behaviour, why", [
    ("hang", "timeout"), ("http_401", "http_401"), ("http_429", "http_429"), ("http_500", "http_500"), ("refusal", "refusal"),
    ("malformed_json", "malformed"), ("malformed_shape", "malformed"), ("illegal", "illegal"), ("reset", "network"),
])
def test_every_failure_of_luna_plays_mixs_move_and_is_counted(classes, behaviour, why):
    battles = tsp.load_fixture()[:1]
    _mix_notes, mix_orders, _ = play_arm(classes, battles, "mix")
    with FakeDecisions(key=FAKE_KEY, mode=behaviour, hang_s=1.0) as fake:
        luna = luna_for(fake, timeout=0.12)
        luna.client.breaker_failures = 10 ** 6                            # the breaker has a test of its own below
        notes, orders, players = play_arm(classes, battles, "luna", luna)
        said = list(luna.client.said)
        luna.client.close()
    assert orders == mix_orders
    choice = [n for n in notes if n["legal"] > 1]
    summary = players[0].game_summary(battles[0][0]["room"])
    assert all(n["luna"]["fell_back"] and n["luna"]["choice"] is None and n["source"] != "luna" for n in choice)
    assert all(n["played"] == n["luna"]["mix"] for n in choice)
    if why == "http_401":
        # the key was refused at the first question: Luna is off for the run, and says so once
        assert [n["luna"]["why"] for n in choice] == ["http_401"] + ["key_refused"] * (len(choice) - 1)
        assert summary["luna_asked"] == 1 and summary["luna_fallbacks"] == {"http_401": 1, "key_refused": len(choice) - 1}
        assert len(said) == 1 and "OFF for the rest of this run" in said[0] and luna.client.disabled
    else:
        assert {n["luna"]["why"] for n in choice} == {why} and summary["luna_fallbacks"] == {why: len(choice)}
        assert summary["luna_asked"] == len(choice)
    assert summary["luna_answered"] == 0 and summary["luna_played"] == 0 and summary["late"] == 0
    assert "mix played instead on %d" % len(choice) in players[0].search_summary()
    assert FAKE_KEY not in json.dumps(notes) + json.dumps(summary) + " ".join(said)


@tsp.needs_player
def test_lunas_own_pick_is_the_order_and_the_guard_overrules_only_a_blunder(classes):
    battles = tsp.load_fixture()
    mix_notes, mix_orders, _ = play_arm(classes, battles, "mix")
    with FakeDecisions(key=FAKE_KEY, mode="random", seed=11) as fake:
        luna = luna_for(fake)
        notes, orders, players = play_arm(classes, battles, "luna", luna)
        luna.client.close()
    choice = [n for n in notes if n["legal"] > 1]
    by_request = {(n["battle"], n["rqid"]): n for n in mix_notes}                  # (a request id belongs to one side of a room)
    played = overruled = changed = 0
    for n in choice:
        lu = n["luna"]
        assert lu["choice"] is not None and n["mask"][lu["choice"]] == 1 and n["mask"][n["played"]] == 1
        assert lu["mix"] == by_request[(n["battle"], n["rqid"])]["played"]          # what mix plays at this very request
        facts_net = sp.merged_prior(n["probs"], n["mask"])
        shares = n.get("shares")
        blunder = bool(lp.blunder_guard(lu["choice"], lu["mix"], facts_net, shares))
        # (the record's numbers are rounded to 4 places: a value on a threshold is not judged here)
        edge = [facts_net[lu["choice"]] - 0.01, facts_net[lu["mix"]] - 0.60] + (
            [shares[lu["choice"]] - 0.01, shares[lu["mix"]] - 0.60] if shares else [])
        if min(abs(x) for x in edge) > 2e-4:
            assert lu["overruled"] == blunder
        if lu["overruled"]:
            overruled += 1
            assert n["played"] == lu["mix"] and n["source"] != "luna" and lu["overruled_why"]
        else:
            played += 1
            assert n["played"] == lu["choice"] and n["source"] == "luna"
            changed += int(lu["choice"] != lu["mix"])
    assert played > 20 and changed > 10                                            # random picks: mostly not mix's
    totals = [p.game_summary(e[0]["room"]) for p, e in zip(players, battles)]
    assert sum(s["luna_overruled"] for s in totals) == overruled and sum(s["luna_changed"] for s in totals) == changed
    assert sum(s["luna_played"] for s in totals) == played
    # Luna picking the action with the least support of all, with a network as sure of itself as the real one (the
    # stub's logits times 4): where that pick is a blunder by the rule, mix's pick is the order.
    class Sure(classes["search"]):
        def _forward_batch(self, tags, obs):
            return 4.0 * super()._forward_batch(tags, obs)

    with FakeDecisions(key=FAKE_KEY, mode="worst") as fake:
        luna = luna_for(fake)
        worst, worst_orders, worst_players = play_arm({"search": Sure}, battles, "luna", luna)
        luna.client.close()
    fired = [n for n in worst if n["legal"] > 1 and n["luna"]["overruled"]]
    stood = [n for n in worst if n["legal"] > 1 and n["luna"]["choice"] is not None and not n["luna"]["overruled"]]
    assert len(fired) >= 5 and stood, (len(fired), len(stood))
    for n in fired:
        lu = n["luna"]
        net = sp.merged_prior(n["probs"], n["mask"])
        assert n["played"] == lu["mix"] != lu["choice"] and n["source"] != "luna" and not lu["fell_back"]
        assert net[lu["choice"]] < 0.0101 and (n.get("shares") is None or n["shares"][lu["choice"]] < 0.0101)
        assert net[lu["mix"]] > 0.5999 or (n.get("shares") is not None and n["shares"][lu["mix"]] > 0.5999)
        assert "of the network" in lu["overruled_why"]
    assert all(n["played"] == n["luna"]["choice"] for n in stood)
    assert sum(p.game_summary(e[0]["room"])["luna_overruled"] for p, e in zip(worst_players, battles)) == len(fired)
    # an order went out for every decision on record
    for player, sent in zip(players, orders):
        assert len(sent) == len(player.notes)
    # the guard switched off: every answer is the order
    with FakeDecisions(key=FAKE_KEY, mode="random", seed=11) as fake:
        luna = luna_for(fake, guard=False)
        notes_off, _orders, _players = play_arm(classes, battles, "luna", luna)
        luna.client.close()
    off = [n for n in notes_off if n["legal"] > 1]
    assert not any(n["luna"]["overruled"] for n in off) and all(n["played"] == n["luna"]["choice"] for n in off)


@tsp.needs_player
def test_a_luna_battle_without_a_key_is_mix_and_says_off(classes):
    battles = tsp.load_fixture()[:1]
    _n, mix_orders, _ = play_arm(classes, battles, "mix")
    notes, orders, players = play_arm(classes, battles, "luna", None)
    assert orders == mix_orders
    choice = [n for n in notes if n["legal"] > 1]
    assert choice and all(n["luna"]["why"] == "off" and not n["luna"]["asked"] for n in choice)
    assert players[0].game_summary(battles[0][0]["room"])["luna_fallbacks"] == {"off": len(choice)}


@tsp.needs_player
def test_luna_never_makes_a_decision_late(classes):
    battles = tsp.load_fixture()[:1]
    with FakeDecisions(key=FAKE_KEY, mode="other", delay=(0.05, 0.9), seed=3) as fake:
        luna = luna_for(fake, timeout=1.2)
        luna.client.breaker_failures = 10 ** 6                            # (five slow answers in a row would pause it)
        # a search of 0.25 s inside a 0.9 s deadline: Luna gets what is left of it, never more
        bridge = tsp.FakeBridge(battles, search_s=0.25)
        notes = []
        try:
            player = tsp.searcher(classes, battles[0][-1]["username"], bridge, mode="luna", deadline=0.9, on_decision=notes.append, luna=luna)
            tsp.play(player, battles[0])
        finally:
            bridge.close()
        luna.client.close()
    choice = [n for n in notes if n["legal"] > 1]
    assert max(n["ms"] for n in notes) <= 900.0 + 60.0 and player.game_summary(battles[0][0]["room"])["late"] == 0
    whys = {n["luna"]["why"] for n in choice}
    assert "timeout" in whys and None in whys                                 # some answers came in time, some did not
    assert all(n["luna"]["timeout_ms"] <= 650.0 for n in choice if n["luna"]["asked"])   # 0.9 s less the search and the reserve
    assert all(n["played"] == (n["luna"]["choice"] if n["luna"]["choice"] is not None and not n["luna"]["overruled"]
                               else n["luna"]["mix"]) for n in choice)


@tsp.needs_player
def test_other_modes_do_not_touch_luna(classes):
    battles = tsp.load_fixture()[:1]
    with FakeDecisions(key=FAKE_KEY, mode="other") as fake:
        luna = luna_for(fake)
        for mode in ("net", "mix", "search"):
            notes, _orders, players = play_arm(classes, battles, mode, luna)
            assert not any("luna" in n for n in notes)
            assert "luna_asked" not in players[0].game_summary(battles[0][0]["room"]) and "Luna" not in players[0].search_summary()
        luna.client.close()
        assert fake.requests == []


# =============================================================================================
# The brief on real ladder decisions: the budget, and nothing hidden
# =============================================================================================

def load_ladder_games():
    if not os.path.exists(LADDER_GAMES):
        pytest.skip("the logged public ladder games are not published with the code")
    games, current = [], None
    with gzip.open(LADDER_GAMES, "rt", encoding="utf-8") as fh:
        for line in fh:
            e = json.loads(line)
            if e["t"] == "game":
                current = {"head": e, "rows": []}
                games.append(current)
            else:
                current["rows"].append(e)
    return games


def briefs_of_the_ladder_games(classes):
    """
    Every decision with a choice of the ten logged ladder games, as (the note, the request built
    for it, what the battle had shown our side by then, what it showed by its end). The battle is
    rebuilt by poke-env from what the room sent our player; the numbers are the decision's note's.
    """
    import numpy as np  # noqa: F401

    out = []
    base = classes["search"]

    for game in load_ladder_games():
        room = game["head"]["battle"]
        decisions = {}
        for row in game["rows"]:
            if row["t"] == "decision":
                decisions[row.get("rqid")] = row
        whole = luna_audit.Seen()
        for row in game["rows"]:
            if row["t"] == "in":
                whole.feed(row["data"])
        seen = luna_audit.Seen()
        log = lp.TurnLog()
        built_here = []

        class Replay(base):
            async def choose_move(self, battle):
                rqid = (getattr(battle, "last_request", None) or {}).get("rqid")
                d = decisions.get(rqid)
                if d is not None and int(d.get("legal") or 0) > 1 and not getattr(battle, "reviving", False):
                    actions = {}
                    for a in range(14):
                        if d["mask"][a]:
                            target, kw = self.assembler.map_index_to_order(a, battle)
                            if target != "DEFAULT":
                                actions[a] = (target, dict(kw or {}))
                    shares = d.get("shares")
                    scores = None
                    if shares is not None:
                        # the note keeps three of the search's results; the others are filled in so that the brief is as long as a live one
                        scores = [0.5 if d["mask"][a] and shares[a] > 0 else None for a in range(14)]
                        for key, a in (("score_net", d["net"]), ("score_search", d.get("search")), ("score_played", d["played"])):
                            if a is not None and d.get(key) is not None:
                                scores[a] = d[key]
                    pred = d.get("predict") or {}
                    facts = lp.Facts(mask=[float(m) for m in d["mask"]], actions=actions, probs=d["probs"], net_action=d["net"],
                                     mix_action=d["played"], shares=shares, scores=scores, search_reason=d.get("reason") or "",
                                     iterations=int(d.get("iterations") or 0),
                                     prediction=[(n, p) for n, p in pred["top"]] if pred.get("top") else None,
                                     p_switch=pred.get("p_switch"), p_tera=pred.get("p_tera"),
                                     opponent_options=[tuple(o) for o in d["opponent"]] if d.get("opponent") else None,
                                     worlds=None, win_est=d.get("win_est"), last_turn=list(log.last), this_turn=list(log.turn))
                    if d["played"] in actions and len(actions) > 1:
                        built_here.append((d, lp.build_request(battle, facts), seen.copy(), whole))
                return self.choose_default_move()

        player = tsp.offline(Replay, "Player", mode="net", warm_up=False)

        async def feed():
            for row in game["rows"]:
                if row["t"] != "in":
                    continue
                seen.feed(row["data"])
                parts = tsp.split(room, row["data"])
                log.feed(parts[1:])
                await player._handle_battle_message(parts)

        tsp.on_loop(player, feed())
        out += built_here
    return out


@tsp.needs_player
def test_the_brief_of_real_ladder_decisions_fits_the_budget_and_has_nothing_hidden(classes):
    built = briefs_of_the_ladder_games(classes)
    assert len(built) >= 200, len(built)
    tokens = sorted(b.tokens for _d, b, _s, _w in built)
    over = [b.tokens for _d, b, _s, _w in built if b.tokens > lp.TOKEN_BUDGET]
    assert not over, (len(over), max(over))
    assert tokens[-1] <= lp.TOKEN_BUDGET and tokens[len(tokens) // 2] > 600                # a real brief, not a stub of one
    assert sum(1 for _d, b, _s, _w in built if b.detail == 0) >= 0.9 * len(built)          # nearly all at full detail
    took = sorted(b.ms for _d, b, _s, _w in built)
    assert took[int(0.95 * len(took))] < 100.0 and took[len(took) // 2] < 40.0             # written in the event loop: it must be quick
    hidden = []
    for d, b, seen, whole in built:
        assert b.problems == [], (d["turn"], b.problems)                                   # every part of the brief could be written
        choices = b.question["choices"]
        assert len(choices) == len(b.values) == sum(1 for a in range(14) if d["mask"][a] and a in b.values.values())
        assert sorted(b.values.values()) == sorted(a for a in range(14) if d["mask"][a])   # exactly the legal actions
        assert sum(MIX_MARK in c["description"] for c in choices) == 1
        assert next(c for c in choices if MIX_MARK in c["description"])["value"].startswith("a%d-" % d["played"])
        assert check_request(b.body) == [] and b.body["model"] == "gpt-6-luna"
        text = b.brief + " " + json.dumps(choices)
        assert "Player" not in text and "Opponent" not in text                              # no user names (the fixture's two)
        assert "FOE'S ACTIVE:" in b.brief and "ADVISORS:" in b.brief and "POKEMON LEFT:" in b.brief
        if not d.get("forced"):
            assert "REQUEST: choose a move or a switch." in b.brief and "OUR ACTIVE:" in b.brief
        else:
            assert "REQUEST: forced switch." in b.brief
        hidden += ["%s turn %s: %s" % (d.get("turn"), d.get("rqid"), p) for p in luna_audit.hidden_problems(b.brief, choices, seen, whole)]
    assert hidden == [], hidden[:10]
    # the opponent's Pokemon that came out later exist in these games: the check had something to find
    assert any(whole.foe_species - seen.foe_species for _d, _b, seen, whole in built)
    print("\nbriefs of %d real decisions: tokens mean %.0f, median %d, largest %d (budget %d); written in %.1f ms on average"
          % (len(built), sum(tokens) / len(tokens), tokens[len(tokens) // 2], tokens[-1], lp.TOKEN_BUDGET,
             sum(b.ms for _d, b, _s, _w in built) / len(built)))


@tsp.needs_player
def test_the_token_estimate_is_not_under_a_real_tokenizer(classes):
    tiktoken = pytest.importorskip("tiktoken")
    try:
        enc = tiktoken.get_encoding("o200k_base")
    except Exception as exc:                 # no network for the encoding's file
        pytest.skip("the o200k_base encoding could not be loaded: %r" % (exc,))
    built = briefs_of_the_ladder_games(classes)[::5]
    ratios = []
    for _d, b, _s, _w in built:
        parts = [b.brief, b.question["instructions"]] + [c["value"] + "\n" + c["description"] for c in b.question["choices"]]
        real = sum(len(enc.encode(p)) for p in parts)
        ratios.append(b.tokens / real)
    assert min(ratios) >= 1.0 and max(ratios) <= 1.20, (min(ratios), max(ratios))
    print("\nestimate over o200k_base on %d requests: %.2f to %.2f times" % (len(ratios), min(ratios), max(ratios)))


@tsp.needs_player
def test_a_sample_brief_reads_as_plain_words(classes):
    built = briefs_of_the_ladder_games(classes)
    d, b, _seen, _whole = next(x for x in built if x[0].get("shares") and x[0]["legal"] >= 8 and not x[0].get("forced") and x[0]["turn"] >= 4)
    lines = b.brief.split("\n")
    assert lines[0].startswith("Pokemon Showdown, gen 9 random battle, singles.") and "Turn %d." % d["turn"] in lines[0]
    heads = [l.split(":")[0] for l in lines[1:]]
    for head in ("POKEMON LEFT", "OUR ACTIVE", "FOE'S ACTIVE", "FIELD", "ADVISORS"):
        assert head in heads, heads
    moves = [c for c in b.question["choices"] if "-switch-" not in c["value"]]
    switches = [c for c in b.question["choices"] if "-switch-" in c["value"]]
    assert moves and switches
    assert all(re.search(r"net (<1|\d+)%", c["description"]) and "search " in c["description"] for c in b.question["choices"])
    assert any("Est. " in c["description"] and "of the foe's HP" in c["description"] for c in moves)
    assert all(c["description"].startswith("Switch to ") and " HP " in c["description"] for c in switches)
    dash = chr(0x2014)
    assert dash not in b.brief and dash not in json.dumps(b.question, ensure_ascii=False)       # plain words: no long dash


# =============================================================================================
# Review of 2026-10-12: what the guide leaves open, and two facts of the brief
# =============================================================================================

def test_the_only_answer_without_a_name_is_the_answer_to_our_one_question():
    """The guide's SDK examples treat a refusal's name as optional: missing or null, it is still a refusal, not a wrong shape."""
    values = {"a0-x": 0, "a4-y": 4}
    assert lp.parse_answer([{"type": "refusal"}], "action", values)["kind"] == "refusal"
    assert lp.parse_answer([{"type": "refusal", "name": None}], "action", values)["kind"] == "refusal"
    assert lp.parse_answer([{"type": "choice", "name": None, "choice": "a4-y"}], "action", values)["action"] == 4
    assert lp.parse_answer([{"type": "choice", "choice": "a0-x"}, {"type": "choice", "choice": "a4-y"}], "action", values)["kind"] == "malformed"
    assert lp.parse_answer([{"type": "refusal", "name": "another"}], "action", values)["kind"] == "malformed"


def test_fields_the_guide_does_not_name_are_ignored_and_a_usage_of_another_shape_is_not_trusted():
    values = {"a0-x": 0, "a4-y": 4}
    answer = {"type": "choice", "name": "action", "choice": "a0-x", "confidence": "high", "logprobs": None, "extra": {"k": 1},
              "probabilities": [{"value": "a0-x", "probability": 0.8, "rank": 1}, {"value": "a4-y"}, "text", {"value": "a4-y", "probability": "0.2"}]}
    parsed = lp.parse_answer([answer], "action", values)
    assert parsed == {"kind": "choice", "action": 0, "value": "a0-x", "confidence": None, "probabilities": {0: 0.8}}


def test_a_redirect_is_not_followed_so_the_key_goes_to_the_checked_address_only():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    seen = {"second": 0, "auth": None}

    class Second(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            seen["second"] += 1
            seen["auth"] = self.headers.get("Authorization")
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            data = b'{"answers": []}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    second = ThreadingHTTPServer(("127.0.0.1", 0), Second)

    class First(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(307)
            self.send_header("Location", "http://localhost:%d/v1/decisions" % second.server_port)
            self.send_header("Content-Length", "0")
            self.end_headers()

    first = ThreadingHTTPServer(("127.0.0.1", 0), First)
    for server in (first, second):
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
    client = lp.LunaClient(FAKE_KEY, url="http://127.0.0.1:%d/v1/decisions" % first.server_port, say=lambda text: None)
    try:
        with pytest.raises(lp.LunaError) as caught:
            client.ask_sync(toy_built().body, 2.0)
        assert caught.value.kind == "http" and caught.value.status == 307 and caught.value.reason == "http_307"
        assert seen == {"second": 0, "auth": None} and FAKE_KEY not in caught.value.text
    finally:
        client.close()
        first.shutdown(), second.shutdown()
        first.server_close(), second.server_close()


def test_knock_off_counts_its_half_more_on_a_target_that_holds_an_item():
    import setup_counter as sc
    import setup_moves as sm

    stats = {"hp": 300.0, "atk": 200.0, "def": 200.0, "spa": 200.0, "spd": 200.0, "spe": 200.0}
    ours = sc.Fighter(level=85, stats=dict(stats), types=("DRAGON",), base_types=("DRAGON",), item="leftovers")
    bare = sc.Fighter(level=85, stats=dict(stats), types=("DRAGON",), base_types=("DRAGON",), item="")
    foe = sc.Fighter(level=85, stats=dict(stats), types=("POISON",), base_types=("POISON",))
    board = lp._Board.__new__(lp._Board)
    board.sc, board.sm, board.weather, board.our_side, board.foe_side = sc, sm, "", {}, {}
    board.our_f, board.foe_f = ours, foe
    plain = sc.damage_share("knockoff", ours, foe, roll=1.0)
    board.foe_item_shown = ""                                    # not shown: a random battle set holds one
    assert board.our_damage("knockoff", False)[1] == pytest.approx(1.5 * plain)
    board.foe_item_shown = "leftovers"
    assert board.our_damage("knockoff", False)[1] == pytest.approx(1.5 * plain)
    board.foe_item_shown = "none"                                # knocked off or used up already
    assert board.our_damage("knockoff", False)[1] == pytest.approx(plain)
    assert board.our_damage("crunch", False)[1] == pytest.approx(sc.damage_share("crunch", ours, foe, roll=1.0))
    hit, name = board.foe_hit(ours, ["knockoff"])                # the foe's Knock Off on ours
    assert name == "Knock Off" and hit == pytest.approx(1.5 * sc.damage_share("knockoff", foe, ours, roll=1.0))
    assert board.foe_hit(bare, ["knockoff"])[0] == pytest.approx(sc.damage_share("knockoff", foe, bare, roll=1.0))


def test_the_launcher_says_the_account_again_and_plays_nothing_without_one():
    text = open(LAUNCHER, "rb").read().decode("utf-8").replace("\r\n", "\n")
    asked = text.index("set /p PS_USERNAME")
    assert asked < text.index('if "%PS_USERNAME%"=="" goto nouser') < text.index("RATED games will be played on the Showdown account %PS_USERNAME%")
    assert text.index("RATED games will be played on the Showdown account %PS_USERNAME%") < text.index("%PY% ladder.py --games")
    assert text.index(":nouser") < text.index("nothing was played") < text.index(":checkfailed") < text.index(":done")
    assert "goto done" in text[text.index(":nouser"):text.index(":checkfailed")]


def test_a_knockout_is_not_promised_through_a_substitute():
    from types import SimpleNamespace

    import setup_counter as sc
    import setup_moves as sm

    board = lp._Board.__new__(lp._Board)
    board.our_f = sc.Fighter()
    board.theirs = SimpleNamespace(effects={"SUBSTITUTE": 1})
    assert lp._behind_substitute(board, sm.move_entry("earthquake"))
    assert not lp._behind_substitute(board, sm.move_entry("hypervoice"))        # a sound move goes through
    assert not lp._behind_substitute(board, sm.move_entry("bulletseed"))        # several hits: the later ones land
    board.our_f = sc.Fighter(ability="infiltrator")
    assert not lp._behind_substitute(board, sm.move_entry("earthquake"))
    board.our_f, board.theirs = sc.Fighter(), SimpleNamespace(effects={})
    assert not lp._behind_substitute(board, sm.move_entry("earthquake"))


def test_no_terminal_to_type_the_key_at_is_no_key_said_in_one_line():
    def closed(prompt):
        raise EOFError

    with pytest.raises(lp.LunaError) as none:
        lp.read_key({}, ask=closed)
    assert none.value.kind == "key" and "OPENAI_API_KEY" in none.value.text
